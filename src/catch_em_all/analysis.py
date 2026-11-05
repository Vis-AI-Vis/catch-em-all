"""Tables, figures, and a compact report for Phase I."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _comparison_montage(category_predictions: pd.DataFrame, category: str, path: Path) -> None:
    """Compare the six most and least confident images inside one category."""

    ordered = category_predictions.sort_values("top1_confidence")
    selected = pd.concat([ordered.tail(6), ordered.head(6)], ignore_index=True)
    figure, axes = plt.subplots(2, 6, figsize=(15, 6))
    for index, (axis, row) in enumerate(zip(axes.flat, selected.itertuples())):
        with Image.open(row.path) as image:
            axis.imshow(image.convert("RGB"))
        group = "High" if index < 6 else "Low"
        axis.set_title(f"{group}: {row.top1_label}\n{row.top1_confidence:.1%}", fontsize=9)
        axis.axis("off")
    figure.suptitle(f"{category.title()}: highest vs. lowest confidence", fontsize=15)
    _save_figure(path)


def create_analysis(predictions: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    """Write reproducible summary artifacts from an inference table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4.8))
    sns.histplot(predictions["top1_confidence"], bins=40)
    plt.axvline(threshold, color="crimson", linestyle="--", label=f"threshold = {threshold:.0%}")
    plt.legend()
    plt.xlabel("Top-1 confidence")
    _save_figure(figures / "confidence_histogram.png")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=predictions,
        x="top1_confidence",
        y="normalized_entropy",
        alpha=0.55,
        s=24,
        hue=predictions["top1_confidence"].ge(threshold),
        palette={False: "#4c78a8", True: "#e45756"},
        legend=False,
    )
    plt.xlabel("Top-1 confidence")
    plt.ylabel("Normalized entropy")
    _save_figure(figures / "confidence_vs_entropy.png")

    high = predictions[predictions["top1_confidence"] >= threshold].copy()
    high.to_csv(output_dir / "high_confidence_predictions.csv", index=False)
    category_available = predictions["category"].notna().all()

    if category_available:
        category_summary = (
            predictions.assign(high_confidence=predictions["top1_confidence"] >= threshold)
            .groupby("category", as_index=False)
            .agg(
                image_count=("filename", "size"),
                median_confidence=("top1_confidence", "median"),
                high_confidence_count=("high_confidence", "sum"),
            )
        )
        category_summary["high_confidence_rate"] = (
            category_summary["high_confidence_count"] / category_summary["image_count"]
        )
        category_summary.to_csv(output_dir / "category_summary.csv", index=False)

        matrix_all = pd.crosstab(predictions["category"], predictions["top1_label"])
        matrix_high = pd.crosstab(high["category"], high["top1_label"])
        matrix_all.to_csv(output_dir / "prediction_matrix_all.csv")
        matrix_high.to_csv(output_dir / "prediction_matrix_high_confidence.csv")

        top_labels = predictions["top1_label"].value_counts().head(20).index
        heatmap_data = matrix_all.reindex(columns=top_labels, fill_value=0)
        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data, cmap="Blues", linewidths=0.2)
        plt.xlabel("Predicted Pokémon (20 most frequent overall)")
        plt.ylabel("Real animal category")
        _save_figure(figures / "prediction_matrix_all_top20.png")

        ordered = category_summary.sort_values("high_confidence_rate", ascending=False)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=ordered, x="category", y="high_confidence_rate", color="#4c78a8")
        plt.xticks(rotation=55, ha="right")
        plt.xlabel("Animal category")
        plt.ylabel(f"Share with confidence ≥ {threshold:.0%}")
        _save_figure(figures / "high_confidence_rate_by_category.png")

        focus = predictions[predictions["category"].isin(["cattle", "antelope", "cat"])]
        focus.to_csv(output_dir / "focus_categories_predictions.csv", index=False)
        for category in ["cattle", "antelope", "cat"]:
            category_predictions = focus[focus["category"] == category]
            if len(category_predictions) >= 12:
                _comparison_montage(
                    category_predictions,
                    category,
                    figures / f"{category}_high_vs_low.png",
                )

    lines = [
        "# Phase I OOD analysis summary",
        "",
        f"- Images analyzed: **{len(predictions):,}**",
        f"- Classifier output labels: **{int(predictions['num_model_labels'].iloc[0]):,}**",
        f"- Median top-1 confidence: **{predictions['top1_confidence'].median():.2%}**",
        f"- Median normalized entropy: **{predictions['normalized_entropy'].median():.3f}**",
        f"- Images at or above {threshold:.0%}: **{len(high):,} ({len(high) / len(predictions):.2%})**",
        "",
        "The confidence cutoff is an exploratory screening threshold, not a validated OOD detector.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
