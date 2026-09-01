#!/usr/bin/env python
"""Plot DLPFC GraphST and STAGATE graph-training controls."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "workflows/graph_training_controls/summary/dlpfc_graph_mode_summary.csv"
)
DEFAULT_OUTPUT = Path("workflows/graph_training_controls/summary")
TRAINING_STYLES = {
    "whole": {"label": "Whole graph", "color": "#D25F3D", "marker": "o"},
    "fullneighbors": {
        "label": "Complete-neighborhood",
        "color": "#3478A8",
        "marker": "s",
    },
    "sampled": {"label": "Sampled", "color": "#27866D", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.facecolor": "#FCFBF8",
            "figure.facecolor": "#F5F1E8",
            "savefig.facecolor": "#F5F1E8",
            "axes.edgecolor": "#8D887F",
        }
    )


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv, dtype={"section": str})
    frame = frame.loc[frame["method"].isin(["graphst", "stagate"])].copy()
    graph_settings = (
        ("spatial", "spatial", "S -> S"),
        ("annotation", "spatial", "A -> S"),
        ("spatial", "annotation", "S -> A"),
        ("annotation", "annotation", "A -> A"),
    )
    panel_specs = (
        ("graphst", "ari", "GraphST ARI"),
        ("graphst", "nmi", "GraphST NMI"),
        ("stagate", "ari", "STAGATE ARI"),
        ("stagate", "nmi", "STAGATE NMI"),
    )
    offsets = {"whole": -0.20, "fullneighbors": 0.0, "sampled": 0.20}
    x = np.arange(len(graph_settings), dtype=float)
    rng = np.random.default_rng(0)
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), sharex=True)

    for axis, (method, metric_name, title) in zip(axes.flat, panel_specs):
        method_frame = frame.loc[frame["method"].eq(method)]
        for training_mode, style in TRAINING_STYLES.items():
            means = []
            positions = []
            for index, (train_graph, infer_graph, _) in enumerate(graph_settings):
                values = method_frame.loc[
                    method_frame["training_mode"].eq(training_mode)
                    & method_frame["train_graph"].eq(train_graph)
                    & method_frame["infer_graph"].eq(infer_graph),
                    metric_name,
                ].to_numpy(dtype=float)
                if len(values) != 12:
                    raise ValueError(
                        f"Expected 12 values for {method}/{training_mode}/"
                        f"{train_graph}/{infer_graph}; found {len(values)}"
                    )
                position = x[index] + offsets[training_mode]
                axis.scatter(
                    np.full(len(values), position) + rng.normal(0, 0.018, len(values)),
                    values,
                    color=style["color"],
                    s=17,
                    alpha=0.24,
                    linewidth=0,
                )
                means.append(values.mean())
                positions.append(position)
            axis.plot(
                positions,
                means,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=6.5,
            )
        axis.set_title(title, loc="left", fontsize=11.0, fontweight="bold")
        axis.set_ylim(0.1, 0.82)
        axis.grid(axis="y", color="#DDD7CC", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(x, [setting[2] for setting in graph_settings])

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=6.5,
            label=style["label"],
        )
        for style in TRAINING_STYLES.values()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.93), h_pad=1.8, w_pad=1.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "dlpfc_graph_training_controls.png", dpi=300)
    fig.savefig(args.output_dir / "dlpfc_graph_training_controls.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
