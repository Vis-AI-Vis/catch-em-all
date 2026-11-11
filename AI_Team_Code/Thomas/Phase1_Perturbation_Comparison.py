import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input CSV paths (matching your existing Phase 1 CSV format)
FG_CSV = os.path.join(SCRIPT_DIR, "phase1_fg_with_animals.csv")
ORIG_CSV = os.path.join(SCRIPT_DIR, "phase1_original_analysis.csv")  # or phase1_bg_analysis.csv
MASK_CSV = os.path.join(SCRIPT_DIR, "phase1_mask_analysis.csv")


def safe_savefig(fig, filename):
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
        print(f"--> Saved {alt_path}")


def normalize_id(fname):
    """Strips extension and path so .jpg in original and .png in fg/mask match 1-to-1."""
    base = os.path.basename(str(fname).replace("\\", "/"))
    return os.path.splitext(base)[0]


# 1. Load & Merge Original, FG, and Mask Predictions by Image ID
df_fg = pd.read_csv(FG_CSV)
df_fg["img_id"] = df_fg["filename"].apply(normalize_id)

df_orig = pd.read_csv(ORIG_CSV)
df_orig["img_id"] = df_orig["filename"].apply(normalize_id)

df_mask = pd.read_csv(MASK_CSV)
df_mask["img_id"] = df_mask["filename"].apply(normalize_id)

merged = (
    df_fg[["img_id", "filename", "animal_category", "top1_pokemon", "top1_conf", "normalized_entropy"]]
    .rename(columns={
        "top1_pokemon": "poke_fg",
        "top1_conf": "conf_fg",
        "normalized_entropy": "ent_fg"
    })
    .merge(
        df_orig[["img_id", "top1_pokemon", "top1_conf", "normalized_entropy"]].rename(columns={
            "top1_pokemon": "poke_orig",
            "top1_conf": "conf_orig",
            "normalized_entropy": "ent_orig"
        }),
        on="img_id", how="inner"
    )
    .merge(
        df_mask[["img_id", "top1_pokemon", "top1_conf", "normalized_entropy"]].rename(columns={
            "top1_pokemon": "poke_mask",
            "top1_conf": "conf_mask",
            "normalized_entropy": "ent_mask"
        }),
        on="img_id", how="inner"
    )
)

# Compute prediction stability & confidence deltas
merged["same_orig_fg"] = merged["poke_orig"] == merged["poke_fg"]
merged["same_fg_mask"] = merged["poke_fg"] == merged["poke_mask"]
merged["delta_conf_bg_removal"] = merged["conf_fg"] - merged["conf_orig"]
merged["delta_conf_tex_removal"] = merged["conf_mask"] - merged["conf_fg"]

merged.to_csv(os.path.join(SCRIPT_DIR, "phase1_perturbation_merged.csv"), index=False)
print(f"--> Merged {len(merged)} images across Original, FG, and Mask!")
print(f"    Prediction Agreement (Original vs FG): {merged['same_orig_fg'].mean()*100:.1f}%")
print(f"    Prediction Agreement (FG vs Mask):     {merged['same_fg_mask'].mean()*100:.1f}%")

# ==============================================================================
# Plot 1: Global Confidence & Entropy Shift + Top Attractor Redistribution
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.8), dpi=150)

# Panel A: Confidence & Certainty Distribution across the 3 Input Stages
stage_df = pd.DataFrame({
    "Stage": ["1. Original\n(With Background)"] * len(merged) +
             ["2. Foreground (FG)\n(Black Background)"] * len(merged) +
             ["3. Binary Mask\n(2D Silhouette Only)"] * len(merged),
    "Top-1 Confidence": np.concatenate([merged["conf_orig"], merged["conf_fg"], merged["conf_mask"]]),
    "Model Certainty (1 - Entropy)": np.concatenate([
        1.0 - merged["ent_orig"], 1.0 - merged["ent_fg"], 1.0 - merged["ent_mask"]
    ])
})

sns.violinplot(
    data=stage_df, x="Stage", y="Top-1 Confidence",
    palette=["#66BB6A", "#42A5F5", "#AB47BC"], inner="quart", cut=0, ax=ax1
)
ax1.axhline(0.85, color="#D32F2F", linestyle="--", label="High-Conf Trap (≥85%)")
ax1.set_title("A. Confidence Shift Across Input Perturbations\n(Original Habitat vs. Black FG vs. Silhouette Mask)", fontsize=11.5, fontweight="bold")
ax1.set_ylabel("Top-1 Pokémon Confidence", fontsize=10.5, fontweight="bold")
ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
ax1.legend(loc="upper right")

# Panel B: Archetype Survival Trajectory (Original -> FG -> Mask)
archetype_pairs = [
    ("cattle", "Bouffalant", "Cattle -> Bouffalant (Horns + Brown Coat)", "#D32F2F", "-o"),
    ("antelope", "Sawsbuck", "Antelope -> Sawsbuck (Branching Antlers)", "#1976D2", "-s"),
    ("cat", "Mightyena", "Cat -> Mightyena (Smooth Dark Coat)", "#37474F", "-^"),
    ("zebra", "Zebstrika", "Zebra -> Zebstrika (Stripe Texture)", "#F57C00", "-D"),
    ("deer", "Stantler", "Deer -> Stantler (Slender Antler Silhouette)", "#388E3C", "-P")
]

stages_x = ["1. Original\n(Habitat BG)", "2. Foreground\n(Black BG)", "3. Binary Mask\n(Silhouette Only)"]
for anim, poke, lbl, col, marker in archetype_pairs:
    sub = merged[merged["animal_category"] == anim]
    if len(sub) == 0:
        continue
    pct_orig = (sub["poke_orig"] == poke).mean() * 100
    pct_fg = (sub["poke_fg"] == poke).mean() * 100
    pct_mask = (sub["poke_mask"] == poke).mean() * 100
    ax2.plot(stages_x, [pct_orig, pct_fg, pct_mask], marker, color=col, linewidth=2.8, markersize=8, label=lbl)

ax2.set_title("B. Texture vs. Silhouette Survival Across Perturbations\n(% of Animal Class Assigned to its Specialist Pokémon Attractor)", fontsize=11.5, fontweight="bold")
ax2.set_ylabel("% of Animal Images Assigned to Attractor", fontsize=10.5, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.35)
ax2.legend(loc="best", fontsize=8.5, framealpha=0.95)

fig.suptitle("Phase 1 Input Perturbation Analysis: Isolating Background Bias vs. Texture & Shape Bias", fontsize=13.5, fontweight="bold", y=1.02)
fig.tight_layout()
safe_savefig(fig, "phase1_perturbation_shift_analysis.png")