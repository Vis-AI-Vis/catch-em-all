import io
import os
import matplotlib
matplotlib.use("Agg")  # Safe non-interactive backend for Windows/VS Code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ORIG_CSV = os.path.join(SCRIPT_DIR, "phase1_original_with_animals.csv")
FG_CSV = os.path.join(SCRIPT_DIR, "phase1_fg_with_animals.csv")
MASK_CSV = os.path.join(SCRIPT_DIR, "phase1_mask_with_animals.csv")


def safe_savefig(fig, filename):
    """Safely saves a matplotlib figure on Windows, avoiding OneDrive/VS Code file-lock Errno 22."""
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
        alt_path = os.path.join(SCRIPT_DIR, filename.replace(".png", "_unlocked.png"))
        with open(alt_path, "wb") as f:
            f.write(data)
        print(f"--> [File was locked] Saved instead to: {alt_path}")


# ==============================================================================
# 1. Load & Align All Three Modalities by File Stem (Matches .jpg with .png)
# ==============================================================================
df_orig = pd.read_csv(ORIG_CSV)
df_fg = pd.read_csv(FG_CSV)
df_mask = pd.read_csv(MASK_CSV)

for d in (df_orig, df_fg, df_mask):
    if "filename" not in d.columns:
        d["filename"] = d["filepath"].apply(lambda p: os.path.basename(str(p).replace("\\", "/")))
    # Strip extension (.jpg / .png) so triplets match across folders
    d["file_stem"] = d["filename"].apply(lambda f: os.path.splitext(str(f))[0].strip())

# Deduplicate on file_stem just in case
df_orig = df_orig.drop_duplicates(subset=["file_stem"]).copy()
df_fg = df_fg.drop_duplicates(subset=["file_stem"]).copy()
df_mask = df_mask.drop_duplicates(subset=["file_stem"]).copy()

base_cols = [
    "file_stem", "filename", "filepath", "top1_pokemon", "top1_conf",
    "top2_pokemon", "top2_conf", "normalized_entropy",
    "mean_saturation", "mean_brightness", "edge_density", "fg_coverage"
]

# Keep only columns that actually exist in each DataFrame
cols_fg = [c for c in base_cols if c in df_fg.columns]
cols_orig = [c for c in base_cols if c in df_orig.columns]
cols_mask = [c for c in base_cols if c in df_mask.columns]

m = (
    df_fg[cols_fg + ["animal_category"]]
    .rename(columns={c: f"{c}_fg" for c in cols_fg if c != "file_stem"})
    .merge(
        df_orig[cols_orig].rename(columns={c: f"{c}_orig" for c in cols_orig if c != "file_stem"}),
        on="file_stem", how="inner"
    )
    .merge(
        df_mask[cols_mask].rename(columns={c: f"{c}_mask" for c in cols_mask if c != "file_stem"}),
        on="file_stem", how="inner"
    )
)

if len(m) == 0:
    raise RuntimeError(
        "0 matched images found across the three CSVs! Check that 'filepath' or 'filename' "
        "in original, fg, and mask share the same base image IDs."
    )

m["filename"] = m["filename_fg"]

# Compute Counterfactual Deltas & Stability Flags
m["same_orig_fg"] = m["top1_pokemon_orig"] == m["top1_pokemon_fg"]
m["same_fg_mask"] = m["top1_pokemon_fg"] == m["top1_pokemon_mask"]
m["same_all_three"] = m["same_orig_fg"] & m["same_fg_mask"]

m["delta_conf_bg"] = m["top1_conf_fg"] - m["top1_conf_orig"]      # Effect of removing background
m["delta_conf_tex"] = m["top1_conf_mask"] - m["top1_conf_fg"]     # Effect of removing texture/color
m["delta_ent_bg"] = m["normalized_entropy_fg"] - m["normalized_entropy_orig"]
m["delta_ent_tex"] = m["normalized_entropy_mask"] - m["normalized_entropy_fg"]

out_merged_csv = os.path.join(SCRIPT_DIR, "phase1_tri_modal_merged.csv")
m.to_csv(out_merged_csv, index=False)
print(f"--> Merged {len(m)} matched image triplets into {out_merged_csv}")
print(f"    Prediction Agreement (Orig == FG):   {m['same_orig_fg'].mean()*100:.1f}%")
print(f"    Prediction Agreement (FG == Mask):   {m['same_fg_mask'].mean()*100:.1f}%")
print(f"    Prediction Agreement (All 3 Modal):  {m['same_all_three'].mean()*100:.1f}%")

# ==============================================================================
# 2. VIS 1: 3-Stage Alluvial / Redistribution Flow (Original -> FG -> Mask)
# ==============================================================================
fig, ax = plt.subplots(figsize=(15, 9), dpi=150)

top_pokes = (
    pd.concat([m["top1_pokemon_orig"], m["top1_pokemon_fg"], m["top1_pokemon_mask"]])
    .value_counts()
    .head(9)
    .index.tolist()
)

def group_poke(series):
    return series.apply(lambda x: x if x in top_pokes else "Other Pokémon")

s_orig = group_poke(m["top1_pokemon_orig"])
s_fg = group_poke(m["top1_pokemon_fg"])
s_mask = group_poke(m["top1_pokemon_mask"])

categories = top_pokes + ["Other Pokémon"]
palette = sns.color_palette("tab10", len(categories))
color_map = dict(zip(categories, palette))

stage_x = [0.0, 1.0, 2.0]
stage_series = [s_orig, s_fg, s_mask]
stage_titles = [
    "1. ORIGINAL\n(Scene + Texture + Shape)",
    "2. FOREGROUND (FG)\n(Black BG + Texture + Shape)",
    "3. BINARY MASK\n(Silhouette Shape Only)"
]

N = len(m)
node_bounds = []

bar_width = 0.14
for s_idx, ser in enumerate(stage_series):
    counts = ser.value_counts().reindex(categories, fill_value=0)
    y_cursor = 0.0
    bounds = {}
    for cat in categories:
        h = counts[cat] / N
        if h > 0:
            ax.bar(
                stage_x[s_idx], h, bottom=y_cursor, width=bar_width,
                color=color_map[cat], edgecolor="black", linewidth=1.0, zorder=5
            )
            if h >= 0.035:
                ax.text(
                    stage_x[s_idx], y_cursor + h / 2,
                    f"{cat}\n({h*100:.1f}%)",
                    ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white" if cat != "Other Pokémon" else "#111111",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, ec="none") if cat != "Other Pokémon" else None,
                    zorder=10
                )
        bounds[cat] = (y_cursor, y_cursor + h)
        y_cursor += h
    node_bounds.append(bounds)


def draw_alluvial_ribbons(s_left, s_right, x_left, x_right, b_left, b_right):
    trans = pd.crosstab(s_left, s_right).reindex(index=categories, columns=categories, fill_value=0) / N
    left_offsets = {cat: b_left[cat][0] for cat in categories}
    right_offsets = {cat: b_right[cat][0] for cat in categories}

    xs = np.linspace(x_left + bar_width / 2, x_right - bar_width / 2, 60)
    t = (xs - xs[0]) / (xs[-1] - xs[0])
    sigmoid = 3 * t**2 - 2 * t**3

    for c_from in categories:
        for c_to in categories:
            flow = trans.loc[c_from, c_to]
            if flow <= 0.003:
                continue
            y0_bot = left_offsets[c_from]
            y0_top = y0_bot + flow
            left_offsets[c_from] = y0_top

            y1_bot = right_offsets[c_to]
            y1_top = y1_bot + flow
            right_offsets[c_to] = y1_top

            curve_bot = y0_bot + (y1_bot - y0_bot) * sigmoid
            curve_top = y0_top + (y1_top - y0_top) * sigmoid
            ax.fill_between(xs, curve_bot, curve_top, color=color_map[c_from], alpha=0.38, edgecolor="none", zorder=2)


draw_alluvial_ribbons(s_orig, s_fg, stage_x[0], stage_x[1], node_bounds[0], node_bounds[1])
draw_alluvial_ribbons(s_fg, s_mask, stage_x[1], stage_x[2], node_bounds[1], node_bounds[2])

ax.set_xticks(stage_x)
ax.set_xticklabels(stage_titles, fontsize=11.5, fontweight="bold")
ax.set_ylabel(f"Proportion of {N:,} AM-2k Images", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.02)
ax.set_xlim(-0.25, 2.25)
ax.set_title(
    "Counterfactual Prediction Redistribution Across Input Modalities (Original -> FG -> Mask)\n"
    f"Agreement: Orig->FG = {m['same_orig_fg'].mean()*100:.1f}% | FG->Mask = {m['same_fg_mask'].mean()*100:.1f}% | All 3 Stages = {m['same_all_three'].mean()*100:.1f}%",
    fontsize=13, fontweight="bold"
)
sns.despine(ax=ax, top=True, right=True, left=False)
fig.tight_layout()
safe_savefig(fig, "vis_comp1_alluvial_redistribution.png")

# ==============================================================================
# 3. VIS 2: Archetype Survival & Feature Attribution Trajectory (FIXED)
# ==============================================================================
pair_counts = (
    m.groupby(["animal_category", "top1_pokemon_fg"], as_index=False)
    .size()
    .rename(columns={"size": "fg_count"})
)
anim_totals = m["animal_category"].value_counts()
pair_counts["fg_pct"] = (pair_counts["fg_count"] / pair_counts["animal_category"].map(anim_totals)) * 100.0
top_archetypes = pair_counts.sort_values("fg_pct", ascending=False).head(8)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)

# Panel A: Routing % Trajectory across Original -> FG -> Mask
ax1 = axes[0]
palette_arch = sns.color_palette("Dark2", max(len(top_archetypes), 1))

for idx, (_, row) in enumerate(top_archetypes.iterrows()):
    anim = row["animal_category"]
    poke = row["top1_pokemon_fg"]
    sub = m[m["animal_category"] == anim]

    pct_orig = (sub["top1_pokemon_orig"] == poke).mean() * 100.0
    pct_fg = (sub["top1_pokemon_fg"] == poke).mean() * 100.0
    pct_mask = (sub["top1_pokemon_mask"] == poke).mean() * 100.0

    label = f"{anim} -> {poke}"
    ax1.plot(
        ["Original", "FG (No BG)", "Mask (Shape Only)"],
        [pct_orig, pct_fg, pct_mask],
        marker="o", markersize=8, linewidth=2.5, color=palette_arch[idx], label=label
    )

ax1.set_title("A. Specialist Attractor Survival Rate (% of Animal Assigned)", fontsize=12, fontweight="bold")
ax1.set_ylabel("% of Animal Class Routed to Attractor Pokémon", fontsize=10.5, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(loc="best", fontsize=8.5, frameon=True)

# Panel B: Mean Top-1 Confidence Trajectory for Those Same Pairs
ax2 = axes[1]
for idx, (_, row) in enumerate(top_archetypes.iterrows()):
    anim = row["animal_category"]
    poke = row["top1_pokemon_fg"]
    sub_trapped = m[(m["animal_category"] == anim) & (m["top1_pokemon_fg"] == poke)]
    if len(sub_trapped) < 3:
        continue

    c_orig = sub_trapped["top1_conf_orig"].mean() * 100.0
    c_fg = sub_trapped["top1_conf_fg"].mean() * 100.0
    c_mask = sub_trapped["top1_conf_mask"].mean() * 100.0

    label = f"{anim} -> {poke} (n={len(sub_trapped)})"
    ax2.plot(
        ["Original", "FG (No BG)", "Mask (Shape Only)"],
        [c_orig, c_fg, c_mask],
        marker="s", markersize=8, linewidth=2.5, linestyle="-", color=palette_arch[idx], label=label
    )

ax2.set_title("B. Confidence Trajectory of FG-Trapped Images Across Stages", fontsize=12, fontweight="bold")
ax2.set_ylabel("Mean Top-1 Confidence (%)", fontsize=10.5, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="best", fontsize=8.5, frameon=True)

fig.suptitle(
    "Decomposing OOD Traps: Background Amplification (Orig->FG) vs. Texture Collapse (FG->Mask)",
    fontsize=13.5, fontweight="bold", y=1.02
)
fig.tight_layout()
safe_savefig(fig, "vis_comp2_archetype_survival_trajectories.png")

# ==============================================================================
# 4. VIS 3: Confidence & Entropy Shift Distributions (Violin + Regime Scatter)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)

stage_df = pd.DataFrame({
    "Modality": ["1. Original"] * N + ["2. FG (Black BG)"] * N + ["3. Mask (Silhouette)"] * N,
    "Top-1 Confidence": np.concatenate([m["top1_conf_orig"], m["top1_conf_fg"], m["top1_conf_mask"]]),
    "Normalized Entropy": np.concatenate([m["normalized_entropy_orig"], m["normalized_entropy_fg"], m["normalized_entropy_mask"]])
})

sns.violinplot(
    data=stage_df, x="Modality", y="Top-1 Confidence",
    hue="Modality", palette=["#4FC3F7", "#7E57C2", "#FFB74D"],
    legend=False, inner="quartile", cut=0, ax=axes[0]
)
axes[0].axhline(0.80, color="#D32F2F", linestyle="--", lw=1.2, label="High-Conf Trap Threshold (0.80)")
axes[0].set_title("A. Top-1 Confidence Distribution by Modality", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Top-1 Softmax Confidence", fontsize=10.5, fontweight="bold")
axes[0].legend(loc="upper right", fontsize=9)
axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

anim_deltas = (
    m.groupby("animal_category", as_index=False)
    .agg(
        d_bg=("delta_conf_bg", "mean"),
        d_tex=("delta_conf_tex", "mean"),
        fg_conf=("top1_conf_fg", "mean")
    )
)

sc = axes[1].scatter(
    anim_deltas["d_bg"] * 100.0, anim_deltas["d_tex"] * 100.0,
    s=anim_deltas["fg_conf"] * 900, c=anim_deltas["fg_conf"],
    cmap="viridis", edgecolors="black", linewidth=1.2, alpha=0.9, zorder=5
)
axes[1].axhline(0, color="black", linestyle="--", lw=1.0)
axes[1].axvline(0, color="black", linestyle="--", lw=1.0)

for _, r in anim_deltas.iterrows():
    axes[1].annotate(
        r["animal_category"],
        (r["d_bg"] * 100.0, r["d_tex"] * 100.0),
        xytext=(5, 4), textcoords="offset points", fontsize=8.5, fontweight="bold"
    )

cbar = plt.colorbar(sc, ax=axes[1])
cbar.set_label("Mean FG Confidence", fontsize=9.5, fontweight="bold")
axes[1].set_xlabel("Background Removal Shift: Conf(FG) - Conf(Orig) [% pts]", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Texture Removal Shift: Conf(Mask) - Conf(FG) [% pts]", fontsize=10, fontweight="bold")
axes[1].set_title("B. Animal Class Sensitivity: Background vs. Texture Ablation", fontsize=12, fontweight="bold")
axes[1].grid(True, linestyle=":", alpha=0.5)

fig.tight_layout()
safe_savefig(fig, "vis_comp3_confidence_shift_regimes.png")

# ==============================================================================
# 5. VIS 4: Counterfactual Image Triplet Drill-Down Grid [Original | FG | Mask]
# ==============================================================================
case1 = m[m["same_fg_mask"] & (m["top1_conf_fg"] > 0.60)].sort_values("top1_conf_mask", ascending=False).head(1)
case2 = m[(m["top1_conf_fg"] > 0.75) & (~m["same_fg_mask"])].sort_values("delta_conf_tex", ascending=True).head(1)
case3 = m[(m["top1_conf_fg"] > 0.75) & (m["delta_conf_bg"] > 0.25)].sort_values("delta_conf_bg", ascending=False).head(1)
case4 = m[(m["top1_conf_orig"] > 0.60) & (~m["same_orig_fg"])].sort_values("delta_conf_bg", ascending=True).head(1)

cases = pd.concat([case1, case2, case3, case4]).drop_duplicates(subset=["file_stem"])
case_labels = [
    "1. Shape-Resilient Trap\n(Survives Silhouette Ablation)",
    "2. Texture-Dependent Trap\n(Collapses Without Fur/Pattern)",
    "3. Background-Suppressed Trap\n(Removing BG Unlocks Trap)",
    "4. Habitat/Background Illusion\n(Triggered by Natural Scene)"
]

if len(cases) > 0:
    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.6 * n_rows), dpi=150)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    def resolve_img(path_str, folder_name, fname, fstem):
        p = str(path_str).replace("\\", "/")
        if os.path.exists(p):
            return p
        cand = os.path.join(SCRIPT_DIR, folder_name, str(fname))
        if os.path.exists(cand):
            return cand
        for ext in (".jpg", ".png", ".jpeg", ".webp"):
            cand_ext = os.path.join(SCRIPT_DIR, folder_name, f"{fstem}{ext}")
            if os.path.exists(cand_ext):
                return cand_ext
        return None

    for r_idx, (_, row) in enumerate(cases.iterrows()):
        fstem = row["file_stem"]
        anim = row["animal_category"]
        row_title = case_labels[r_idx] if r_idx < len(case_labels) else f"Case {r_idx+1}"

        specs = [
            (
                "ORIGINAL",
                resolve_img(row["filepath_orig"], "original", row["filename_orig"], fstem),
                row["top1_pokemon_orig"], row["top1_conf_orig"], row["normalized_entropy_orig"]
            ),
            (
                "FOREGROUND (FG)",
                resolve_img(row["filepath_fg"], "fg", row["filename_fg"], fstem),
                row["top1_pokemon_fg"], row["top1_conf_fg"], row["normalized_entropy_fg"]
            ),
            (
                "SILHOUETTE (MASK)",
                resolve_img(row["filepath_mask"], "mask", row["filename_mask"], fstem),
                row["top1_pokemon_mask"], row["top1_conf_mask"], row["normalized_entropy_mask"]
            )
        ]

        for c_idx, (st_name, img_p, poke, conf, ent) in enumerate(specs):
            ax = axes[r_idx, c_idx]
            if img_p and os.path.exists(img_p):
                im = Image.open(img_p).convert("RGB")
                ax.imshow(im)
            else:
                ax.text(0.5, 0.5, f"[{st_name}\n{fstem}]", ha="center", va="center", fontsize=9)
                ax.set_facecolor("#EEEEEE")

            border_col = "#D32F2F" if conf >= 0.75 else ("#FBC02D" if conf >= 0.40 else "#78909C")
            for spine in ax.spines.values():
                spine.set_edgecolor(border_col)
                spine.set_linewidth(3.0)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{st_name}: {poke}\nConf: {conf*100:.1f}% | Ent: {ent:.2f}",
                fontsize=9.5, fontweight="bold",
                color="#B71C1C" if conf >= 0.75 else "#222222"
            )
            if c_idx == 0:
                ax.set_ylabel(f"{row_title}\n[True: {anim.upper()}]", fontsize=9.5, fontweight="bold", labelpad=10)

    fig.suptitle(
        "Counterfactual Image Triplet Drill-Down: Tracking Predictions Across [Original | FG | Mask]",
        fontsize=13.5, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    safe_savefig(fig, "vis_comp4_counterfactual_triplet_grid.png")