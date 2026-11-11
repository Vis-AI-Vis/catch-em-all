import os
import io
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ==============================================================================
# 0. Setup & Safe File Saving (Windows / VS Code safe)
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_CSV_PATH = os.path.join(SCRIPT_DIR, "phase1_fg_with_animals.csv")
FALLBACK_CSV_PATH = os.path.join(SCRIPT_DIR, "phase1_fg_analysis.csv")
OUT_DEEPDIVE_CSV = os.path.join(SCRIPT_DIR, "phase1_deepdive_features.csv")
FORCE_RECOMPUTE = False  # Set True if you want to re-extract image/CLIP features


def safe_savefig(fig, filename):
    """Safely saves matplotlib figures on Windows without Errno 22 file-lock crashes."""
    out_path = os.path.join(SCRIPT_DIR, filename)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    buf.close()
    plt.close(fig)
    try:
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"--> Saved {out_path}")
    except OSError:
        alt_path = os.path.join(SCRIPT_DIR, filename.replace(".png", "_new.png"))
        with open(alt_path, "wb") as f:
            f.write(data)
        print(f"--> [File locked] Saved instead to {alt_path}")


def resolve_img_path(raw_path):
    """Finds the actual image file across relative/Windows paths."""
    clean_p = str(raw_path).replace("\\", "/")
    candidates = [
        clean_p,
        os.path.join(SCRIPT_DIR, clean_p),
        os.path.join(SCRIPT_DIR, "fg", os.path.basename(clean_p)),
        os.path.join("./fg", os.path.basename(clean_p)),
        os.path.join(os.path.dirname(SCRIPT_DIR), "fg", os.path.basename(clean_p)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return clean_p


def extract_clip_text_embeds(model, inputs):
    out = model.get_text_features(**inputs)
    if not isinstance(out, torch.Tensor):
        out = getattr(out, "text_embeds", getattr(out, "pooler_output", out[0]))
    if hasattr(model, "text_projection") and out.shape[-1] != model.text_projection.out_features:
        out = model.text_projection(out)
    return F.normalize(out, p=2, dim=-1)


def extract_clip_image_embeds(model, inputs):
    out = model.get_image_features(**inputs)
    if not isinstance(out, torch.Tensor):
        out = getattr(out, "image_embeds", getattr(out, "pooler_output", out[0]))
    if hasattr(model, "visual_projection") and out.shape[-1] != model.visual_projection.out_features:
        out = model.visual_projection(out)
    return F.normalize(out, p=2, dim=-1)


# ==============================================================================
# 1. Load Data & Automatically Extract Visual + Semantic Pose Features
# ==============================================================================
if os.path.exists(OUT_DEEPDIVE_CSV) and not FORCE_RECOMPUTE:
    print(f"--> Loading cached deep-dive features from {OUT_DEEPDIVE_CSV}...")
    df = pd.read_csv(OUT_DEEPDIVE_CSV)
else:
    source_csv = IN_CSV_PATH if os.path.exists(IN_CSV_PATH) else FALLBACK_CSV_PATH
    print(f"--> Loading base CSV from {source_csv}...")
    df = pd.read_csv(source_csv)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--> Extracting low-level visual stats + CLIP semantic pose/pattern scores on {device}...")

    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    clip_model.eval()

    # Semantic attribute prompts to test Pose (Full Body vs Close-Up) and Pattern (Stripes/Horns vs Plain)
    pose_prompts = [
        "a full body side profile photo of an animal standing on a black background",
        "a close-up zoomed-in portrait of an animal head or face on a black background"
    ]
    pattern_prompts = [
        "an animal with prominent horns, antlers, stripes, or spots on a black background",
        "an animal with plain smooth uniform fur or skin on a black background"
    ]

    with torch.no_grad():
        pose_inp = clip_processor(text=pose_prompts, return_tensors="pt", padding=True).to(device)
        pose_feats = extract_clip_text_embeds(clip_model, pose_inp)

        pat_inp = clip_processor(text=pattern_prompts, return_tensors="pt", padding=True).to(device)
        pat_feats = extract_clip_text_embeds(clip_model, pat_inp)

    cov_list, sat_list, bri_list = [], [], []
    contrast_list, edge_list, aspect_list = [], [], []
    fullbody_scores, pattern_scores = [], []

    BATCH_SIZE = 32
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Analyzing Image Properties"):
        batch_paths = df["filepath"].iloc[i : i + BATCH_SIZE].tolist()
        batch_imgs = []

        for p in batch_paths:
            img_path = resolve_img_path(p)
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(img)

            arr = np.array(img)
            fg_mask = arr.sum(axis=-1) > 15  # Non-black foreground pixels

            # 1. Foreground Coverage (Size of animal in frame)
            cov_list.append(round(float(np.mean(fg_mask)), 4))

            # 2. HSV Saturation & Brightness (Color vividness & lighting)
            hsv = np.array(img.convert("HSV"))
            if np.any(fg_mask):
                sat_list.append(round(float(hsv[:, :, 1][fg_mask].mean() / 255.0), 4))
                bri_list.append(round(float(hsv[:, :, 2][fg_mask].mean() / 255.0), 4))
                # 3. Internal Color/Luminance Contrast (Std Dev of brightness inside animal)
                contrast_list.append(round(float(hsv[:, :, 2][fg_mask].std() / 255.0), 4))
            else:
                sat_list.append(0.0)
                bri_list.append(0.0)
                contrast_list.append(0.0)

            # 4. Edge/Silhouette Complexity (Gradient magnitude inside foreground)
            gray = np.mean(arr, axis=-1) / 255.0
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gx**2 + gy**2)
            edge_score = float(grad_mag[fg_mask].mean()) if np.any(fg_mask) else 0.0
            edge_list.append(round(edge_score, 4))

            # 5. Bounding Box Aspect Ratio (Width / Height of the animal)
            ys, xs = np.where(fg_mask)
            if len(xs) > 10 and len(ys) > 10:
                w = float(xs.max() - xs.min() + 1)
                h = float(ys.max() - ys.min() + 1)
                aspect_list.append(round(w / max(h, 1.0), 4))
            else:
                aspect_list.append(1.0)

        # 6. CLIP Zero-Shot Pose & Pattern Probing
        img_inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feats = extract_clip_image_embeds(clip_model, img_inputs)
            pose_probs = ((img_feats @ pose_feats.T) * 100.0).softmax(dim=-1).cpu().numpy()
            pat_probs = ((img_feats @ pat_feats.T) * 100.0).softmax(dim=-1).cpu().numpy()

        for pp, ptp in zip(pose_probs, pat_probs):
            fullbody_scores.append(round(float(pp[0]), 4))   # Prob of Full Body vs Close-Up Face
            pattern_scores.append(round(float(ptp[0]), 4))   # Prob of Horns/Stripes/Spots vs Plain

    df["fg_coverage"] = cov_list
    df["fg_saturation_only"] = sat_list
    df["fg_brightness_only"] = bri_list
    df["fg_contrast"] = contrast_list
    df["edge_complexity"] = edge_list
    df["bbox_aspect_ratio"] = aspect_list
    df["clip_full_body_score"] = fullbody_scores
    df["clip_pattern_score"] = pattern_scores
    df["model_certainty"] = 1.0 - df["normalized_entropy"]

    df.to_csv(OUT_DEEPDIVE_CSV, index=False)
    print(f"--> Saved enriched deep-dive CSV to {OUT_DEEPDIVE_CSV}")


# ==============================================================================
# 2. Automated Statistical Discovery: Print Ranked Feature Correlations
# ==============================================================================
FEATURES_META = [
    ("fg_brightness_only", "Foreground Brightness (Lighting)"),
    ("fg_saturation_only", "Color Saturation (Vividness)"),
    ("fg_coverage", "Animal Frame Coverage (Area %)"),
    ("fg_contrast", "Internal Luminance Contrast"),
    ("edge_complexity", "Edge / Texture Complexity"),
    ("clip_full_body_score", "CLIP Pose: Full-Body vs. Close-Up"),
    ("clip_pattern_score", "CLIP Features: Horns/Stripes/Spots"),
    ("bbox_aspect_ratio", "Bounding Box Aspect Ratio (W/H)")
]

print("\n" + "=" * 75)
print("AUTOMATED PATTERN DISCOVERY: WHAT DRIVES HIGH POKÉMON CONFIDENCE?")
print("=" * 75)
stat_rows = []
high_mask = df["top1_conf"] >= 0.85
low_mask = df["top1_conf"] < 0.40

for col, label in FEATURES_META:
    valid = df[[col, "top1_conf", "model_certainty"]].dropna()
    r_spear, p_val = stats.spearmanr(valid[col], valid["top1_conf"])
    mean_high = valid.loc[high_mask, col].mean()
    mean_low = valid.loc[low_mask, col].mean()
    std_all = max(valid[col].std(), 1e-6)
    cohens_d = (mean_high - mean_low) / std_all  # Standardized effect size

    stat_rows.append({
        "feature": col,
        "label": label,
        "spearman_r": r_spear,
        "p_value": p_val,
        "mean_high_conf": mean_high,
        "mean_low_conf": mean_low,
        "effect_size_d": cohens_d
    })
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    print(f"{label:38s} | r = {r_spear:+.3f} ({sig:3s}) | Effect d (>85% vs <40%) = {cohens_d:+.2f}")
print("=" * 75 + "\n")

stat_df = pd.DataFrame(stat_rows).sort_values("effect_size_d", key=abs, ascending=False)


# ==============================================================================
# 3. Plot A: 6-Panel Continuous Feature vs. Confidence Scatter + Trend Dashboard
# ==============================================================================
top6_features = stat_df.head(6).to_dict("records")
fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
axes = axes.flatten()

# Categorize confidence tiers for color coding
df["conf_tier"] = pd.cut(
    df["top1_conf"],
    bins=[-0.01, 0.40, 0.85, 1.01],
    labels=["Low (<40% Sink)", "Medium (40-85%)", "High Trap (>85%)"]
)
palette_tiers = {"Low (<40% Sink)": "#90A4AE", "Medium (40-85%)": "#64B5F6", "High Trap (>85%)": "#D32F2F"}

for idx, feat_info in enumerate(top6_features):
    ax = axes[idx]
    col = feat_info["feature"]
    label = feat_info["label"]
    r_val = feat_info["spearman_r"]
    p_val = feat_info["p_value"]

    # Scatter plot of all 1,767 images
    sns.scatterplot(
        data=df, x=col, y="top1_conf", hue="conf_tier",
        palette=palette_tiers, alpha=0.45, s=28, edgecolor="none", ax=ax, legend=(idx == 0)
    )

    # Compute binned mean confidence trendline (10 quantile bins)
    df["bin"] = pd.qcut(df[col], q=10, duplicates="drop")
    bin_means = df.groupby("bin", observed=False)[[col, "top1_conf"]].mean().dropna()
    ax.plot(
        bin_means[col], bin_means["top1_conf"],
        color="#111111", linewidth=2.8, marker="o", markersize=6, label="Decile Mean Trend"
    )
    ax.axhline(0.85, color="#D32F2F", linestyle="--", linewidth=1.2, alpha=0.8)

    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.set_title(f"{label}\n(Spearman r = {r_val:+.2f}, {p_str})", fontsize=11, fontweight="bold")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Top-1 Pokémon Confidence", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)

    if idx == 0:
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

fig.suptitle(
    "Automated Feature Discovery: Continuous Image Characteristics vs. Model Confidence\n"
    "(Red Dashed Line = >85% High-Confidence Trap Threshold | Black Curve = Binned Decile Mean)",
    fontsize=14, fontweight="bold", y=1.02
)
fig.tight_layout()
safe_savefig(fig, "phase1_feature_correlations_dashboard.png")


# ==============================================================================
# 4. Plot B: Effect Size & Distribution Shift (High Trap >85% vs. Low Sink <40%)
# ==============================================================================
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150, gridspec_kw={"width_ratios": [1.1, 1.3]})

# Left Panel: Standardized Effect Size (Cohen's d) Bar Chart
stat_sorted = stat_df.sort_values("effect_size_d", ascending=True)
bar_colors = ["#D32F2F" if d > 0 else "#1976D2" for d in stat_sorted["effect_size_d"]]
ax_left.barh(stat_sorted["label"], stat_sorted["effect_size_d"], color=bar_colors, edgecolor="black", alpha=0.85)
ax_left.axvline(0, color="black", linewidth=1.2)
ax_left.set_xlabel("Standardized Shift (Cohen's d: >85% Traps minus <40% Sinks)", fontsize=10, fontweight="bold")
ax_left.set_title(
    "Which Image Features Trigger >85% Confidence?\n(Red = Higher in >85% Traps | Blue = Lower in >85% Traps)",
    fontsize=11.5, fontweight="bold"
)
ax_left.grid(True, axis="x", linestyle="--", alpha=0.4)

# Right Panel: Normalized Violin Comparison for Top 4 Differentiating Features
extremes_df = df[df["conf_tier"].isin(["Low (<40% Sink)", "High Trap (>85%)"])].copy()
extremes_df["conf_tier"] = extremes_df["conf_tier"].astype(str)
top4_cols = stat_df.head(4)["feature"].tolist()
top4_labels = stat_df.head(4)["label"].tolist()

melted_rows = []
for c, lbl in zip(top4_cols, top4_labels):
    z_vals = (extremes_df[c] - df[c].mean()) / max(df[c].std(), 1e-6)
    for z, tier in zip(z_vals, extremes_df["conf_tier"]):
        melted_rows.append({"Feature": lbl.split(" (")[0], "Z-Score": z, "Confidence Regime": tier})
melt_df = pd.DataFrame(melted_rows)

sns.violinplot(
    data=melt_df, x="Feature", y="Z-Score", hue="Confidence Regime",
    split=True, inner="quart", palette={"Low (<40% Sink)": "#90A4AE", "High Trap (>85%)": "#EF5350"},
    ax=ax_right
)
ax_right.axhline(0, color="black", linestyle="--", alpha=0.5)
ax_right.set_title("Feature Distribution Shift: High Traps (>85%) vs. Low Sinks (<40%)", fontsize=11.5, fontweight="bold")
ax_right.set_ylabel("Standardized Feature Value (Z-Score)", fontsize=10)
ax_right.tick_params(axis="x", rotation=15)
ax_right.grid(True, axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
safe_savefig(fig, "phase1_high_vs_low_conf_distributions.png")


# ==============================================================================
# 5. Plot C: Slide 2 Item-Level Drill-Down (Top-4 Highest vs. Bottom-4 Lowest Images)
#    Automatically selects the 4 animals with the most >85% traps and displays
#    actual image thumbnails side-by-side (High-Conf Trap vs. Low-Conf Sink)
# ==============================================================================
trap_counts = (
    df[df["top1_conf"] >= 0.85]
    .groupby("animal_category")
    .size()
    .sort_values(ascending=False)
)
focus_animals = trap_counts.head(4).index.tolist()
if len(focus_animals) < 4:
    focus_animals = df["animal_category"].value_counts().head(4).index.tolist()

n_rows = len(focus_animals)
n_cols = 8  # Left 4 = Top-4 Highest Confidence (>85% Traps) | Right 4 = Bottom-4 Lowest Confidence (<40% Sinks)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.2 * n_rows), dpi=150)
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)

for r_idx, animal in enumerate(focus_animals):
    sub = df[df["animal_category"] == animal].sort_values("top1_conf", ascending=False)
    high_samples = sub.head(4)
    low_samples = sub.tail(4).sort_values("top1_conf", ascending=True)
    combined = pd.concat([high_samples, low_samples], axis=0)

    for c_idx in range(n_cols):
        ax = axes[r_idx, c_idx]
        if c_idx >= len(combined):
            ax.axis("off")
            continue

        row = combined.iloc[c_idx]
        img_path = resolve_img_path(row["filepath"])
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize & pad to uniform square thumbnail
            img = ImageOps.pad(img, (224, 224), color=(0, 0, 0))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center")

        is_high = c_idx < 4
        border_col = "#2E7D32" if is_high else "#C62828"
        tag = "HIGH TRAP" if is_high else "LOW SINK"

        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3.0)

        ax.set_xticks([])
        ax.set_yticks([])

        # Show predicted Pokémon, confidence, brightness, and full-body pose score
        title_str = (
            f"[{tag}] {row['top1_pokemon']} ({row['top1_conf']*100:.0f}%)\n"
            f"Bri:{row['fg_brightness_only']:.2f} | Sat:{row['fg_saturation_only']:.2f} | Pose:{row['clip_full_body_score']:.2f}"
        )
        ax.set_title(title_str, fontsize=8, fontweight="bold", color=border_col, pad=4)

        if c_idx == 0:
            ax.set_ylabel(
                f"{animal.upper()}\n(n={len(sub)})",
                fontsize=11, fontweight="black", rotation=0, labelpad=45, va="center"
            )

fig.suptitle(
    "Slide 2 Item-Level Drill-Down: Top-4 Highest Confidence Traps (Green) vs. Bottom-4 Lowest Confidence Sinks (Red)\n"
    "(Comparing Pose, Lighting Brightness [Bri], Color Saturation [Sat], and Full-Body Score [Pose] Inside the Same Animal Class)",
    fontsize=13.5, fontweight="bold", y=1.02
)
fig.tight_layout()
safe_savefig(fig, "phase1_item_drilldown_thumbnails.png")