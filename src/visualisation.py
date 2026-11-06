import matplotlib.pyplot as plt

# Categorical colours in fixed order: one per dataset
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_MARKERS = ["o", "^", "s", "D"]  # second encoding, so colour is not the only cue
INK = "#3d3d3a"
GRID = "#e1e0d9"


def visualise_confidence_vs_entropy(dfs, save_path=None):
    """
    Scatter plot of the highest confidence vs. the normalized entropy of each prediction.

    dfs: dict mapping dataset name -> predictions DataFrame
         (needs the columns "confidence" and "entropy_norm")
    """
    if len(dfs) > len(SERIES_COLOURS):
        raise ValueError(f"At most {len(SERIES_COLOURS)} datasets can be plotted at once")

    fig, ax = plt.subplots(figsize=(7, 6))

    for (name, df), colour, marker in zip(dfs.items(), SERIES_COLOURS, SERIES_MARKERS):
        ax.scatter(
            df["confidence"],
            df["entropy_norm"],
            s=12,
            c=colour,
            marker=marker,
            alpha=0.4,
            linewidths=0,
            label=f"{name} (n={len(df)})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Highest confidence (max. softmax probability)", color=INK)
    ax.set_ylabel("Normalized entropy (0 = certain, 1 = uniform)", color=INK)
    ax.set_title("Confidence vs. normalized entropy of the predictions", color=INK)

    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)

    legend = ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
