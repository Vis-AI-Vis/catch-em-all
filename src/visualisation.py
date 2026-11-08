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
        raise ValueError(
            f"At most {len(SERIES_COLOURS)} datasets can be plotted at once"
        )

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


# Plotly marker symbols matching SERIES_MARKERS
PLOTLY_SYMBOLS = ["circle", "triangle-up", "square", "diamond"]


# Same shapes as text glyphs, used next to the dropdown labels
SYMBOL_GLYPHS = ["●", "▲", "■", "◆"]


def _trace_data(df):
    """x, y and tooltip data (image, true label, predicted label) of a predictions DataFrame."""
    from pathlib import Path

    n = len(df)
    # "<folder>/<file>", e.g. "Abra/0.jpg" - enough to find the image again
    image = (
        df["image_path"].map(lambda p: "/".join(Path(p).parts[-2:]))
        if "image_path" in df
        else [""] * n
    )
    true_label = df["true_label"] if "true_label" in df else [""] * n
    predicted_label = df["predicted_label"] if "predicted_label" in df else [""] * n
    return (
        df["confidence"].tolist(),
        df["entropy_norm"].tolist(),
        [list(row) for row in zip(image, true_label, predicted_label)],
    )


def visualise_confidence_vs_entropy_interactive(dfs, filterable=(), save_path=None):
    """
    Interactive (Plotly) version of visualise_confidence_vs_entropy.

    Hover a point to see the image, true and predicted label; click a legend entry
    to hide/show a dataset (double-click to isolate it); drag to zoom.

    dfs: dict mapping dataset name -> predictions DataFrame
         (needs "confidence" and "entropy_norm"; "image_path", "true_label" and
         "predicted_label" are shown in the tooltip if present)
    filterable: names of datasets that get a dropdown above the chart to show
         only one class ("true_label"), e.g. a single animal
    save_path: optional .html file to save a standalone interactive version
    """
    import plotly.graph_objects as go

    if len(dfs) > len(SERIES_COLOURS):
        raise ValueError(
            f"At most {len(SERIES_COLOURS)} datasets can be plotted at once"
        )
    unknown = set(filterable) - set(dfs)
    if unknown:
        raise ValueError(f"Unknown datasets in filterable: {sorted(unknown)}")

    fig = go.Figure()
    dropdowns = []

    for index, ((name, df), colour, symbol, glyph) in enumerate(
        zip(dfs.items(), SERIES_COLOURS, PLOTLY_SYMBOLS, SYMBOL_GLYPHS)
    ):
        n = len(df)
        x, y, customdata = _trace_data(df)

        if name in filterable:
            # One button per class; each only restyles this dataset's trace
            buttons = [
                dict(
                    label=f"All ({n})",
                    method="restyle",
                    args=[
                        {"x": [x], "y": [y], "customdata": [customdata],
                         "name": f"{name} (n={n})"},
                        [index],
                    ],
                )
            ]
            for label, group in sorted(df.groupby("true_label"), key=lambda g: str(g[0])):
                gx, gy, gcustom = _trace_data(group)
                buttons.append(
                    dict(
                        label=f"{label} ({len(group)})",
                        method="restyle",
                        args=[
                            {"x": [gx], "y": [gy], "customdata": [gcustom],
                             "name": f"{name}: {label} (n={len(group)})"},
                            [index],
                        ],
                    )
                )
            dropdowns.append((name, colour, glyph, buttons))

        fig.add_trace(
            go.Scattergl(  # WebGL: stays smooth with thousands of points
                x=x,
                y=y,
                mode="markers",
                name=f"{name} (n={n})",
                marker=dict(
                    color=colour, symbol=symbol, size=6, opacity=0.45, line_width=0
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>confidence %{x:.3f}</b> · entropy %{y:.3f}<br>"
                    "predicted: %{customdata[2]}<br>"
                    "true: %{customdata[1]}<br>"
                    "<span style='color:#73726c'>%{customdata[0]}</span>"
                    f"<extra>{name}</extra>"
                ),
            )
        )

    axis_style = dict(
        range=[0, 1],
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        color=INK,
    )
    fig.update_layout(
        title=dict(
            text="Confidence vs. normalized entropy of the predictions",
            font_color=INK,
            y=0.98,  # pinned to the top, above the dropdown row
            yref="container",
            yanchor="top",
        ),
        xaxis=dict(
            title="Highest confidence (max. softmax probability)",
            constrain="domain",  # keep 0-1 instead of widening to fill the plot
            **axis_style,
        ),
        yaxis=dict(
            title="Normalized entropy (0 = certain, 1 = uniform)",
            scaleanchor="x",  # equal aspect, like the static version
            **axis_style,
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            bgcolor="rgba(255,255,255,0.8)",
            font_color=INK,
            itemsizing="constant",
        ),  # full-size legend markers despite small points
        hovermode="closest",  # nearest point, not only exact hits
        hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=GRID),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=800,
        height=760 if dropdowns else 700,
        # Room for the row of dropdowns between title and plot
        margin=dict(l=70, r=30, t=140 if dropdowns else 60, b=60),
    )

    # Dropdowns in one row above the plot, each with its dataset name as label
    slot = 1 / max(len(dropdowns), 1)
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                active=0,
                showactive=True,
                x=i * slot,
                xanchor="left",
                y=1.14,
                yanchor="top",
                pad=dict(t=4),
                bgcolor="white",
                bordercolor=GRID,
                font=dict(color=INK, size=12),
            )
            for i, (_, _, _, buttons) in enumerate(dropdowns)
        ],
        annotations=[
            dict(
                text=f"<span style='color:{colour}'>{glyph}</span> {name}",
                x=i * slot,
                xanchor="left",
                y=1.14,
                yanchor="bottom",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=INK, size=12),
            )
            for i, (name, colour, glyph, _) in enumerate(dropdowns)
        ],
    )

    if save_path is not None:
        fig.write_html(save_path, include_plotlyjs="cdn")
    fig.show()
    return fig
