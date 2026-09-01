#!/usr/bin/env python
"""Plot matched MERFISH graph-training controls across sagittal sections."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


DEFAULT_INPUT = Path(
    "workflows/graph_training_controls/summary/merfish_graph_training_controls.csv"
)
DEFAULT_OUTPUT = Path("workflows/graph_training_controls/summary")

SECTIONS = (
    "C57BL6J-3.008",
    "C57BL6J-3.009",
    "C57BL6J-3.010",
    "C57BL6J-3.011",
    "C57BL6J-3.012",
)
ROW_SPECS = (
    ("GraphST", "ari", "GraphST\nARI"),
    ("STAGATE", "ari", "STAGATE\nARI"),
    ("GraphST", "nmi", "GraphST\nNMI"),
    ("STAGATE", "nmi", "STAGATE\nNMI"),
)
REGION_SPECS = (
    ("major", "Major brain regions"),
    ("ccf", "CCF regions"),
)
MODE_STYLES = {
    "Whole graph": {
        "color": "#D25F3D",
        "marker": "o",
    },
    "Complete-neighborhood": {
        "color": "#3478A8",
        "marker": "s",
    },
    "Sampled": {
        "color": "#27866D",
        "marker": "^",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.facecolor": "#FCFBF8",
            "figure.facecolor": "#F5F1E8",
            "savefig.facecolor": "#F5F1E8",
            "axes.edgecolor": "#8D887F",
            "axes.labelcolor": "#252525",
            "xtick.color": "#3F3F3F",
            "ytick.color": "#3F3F3F",
        }
    )


def plot(frame, output_dir):
    apply_style()
    fig, axes = plt.subplots(
        len(ROW_SPECS),
        len(REGION_SPECS),
        figsize=(12.2, 11.0),
        sharex=True,
    )
    x = list(range(len(SECTIONS)))

    for row, (method, metric, row_label) in enumerate(ROW_SPECS):
        for column, (region, region_label) in enumerate(REGION_SPECS):
            ax = axes[row, column]
            panel = frame.loc[frame["method"].eq(method)]
            for mode, style in MODE_STYLES.items():
                values = (
                    panel.loc[panel["training_mode"].eq(mode)]
                    .set_index("section")
                    .reindex(SECTIONS)[f"{region}_{metric}"]
                )
                if values.notna().any():
                    ax.plot(
                        x,
                        values,
                        color=style["color"],
                        marker=style["marker"],
                        linewidth=1.8,
                        markersize=5.8,
                        markeredgecolor="#FCFBF8",
                        markeredgewidth=0.6,
                    )
            ax.grid(axis="y", color="#DCD6CC", linewidth=0.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-0.25, len(SECTIONS) - 0.75)
            ax.set_ylim(0.08, 0.68)
            if row == 0:
                ax.set_title(region_label, fontsize=11.0, fontweight="bold", pad=9)
            if column == 0:
                ax.set_ylabel(row_label, fontsize=10.5, fontweight="bold")
            if row == len(ROW_SPECS) - 1:
                ax.set_xticks(x, SECTIONS, rotation=35, ha="right", fontsize=8.2)

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=6.0,
            markeredgewidth=0,
            label=mode,
        )
        for mode, style in MODE_STYLES.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.2,
    )
    fig.suptitle(
        "MERFISH graph-training controls across sections",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=16.0,
        fontweight="bold",
    )
    fig.text(0.5, 0.014, "MERFISH sagittal section", ha="center", fontsize=10.5)
    fig.tight_layout(rect=(0.035, 0.045, 0.995, 0.925), h_pad=1.4, w_pad=1.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "merfish_graph_training_controls.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "merfish_graph_training_controls.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(frame, args.output_dir)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
