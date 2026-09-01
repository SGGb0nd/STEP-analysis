#!/usr/bin/env python
"""Summarize matched Cavity recovery across all five MOSTA sections."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "workflows/mosta_cavity_controls/method_comparison"
E2S7_ROOT = OUTPUT_ROOT / "E16.5_E2S7"
SECTION_ORDER = (
    "E16.5_E2S1",
    "E16.5_E2S4",
    "E16.5_E2S7",
    "E16.5_E2S10",
    "E16.5_E2S13",
)
METHOD_ORDER = (
    "STEP sampled",
    "NicheCompass",
    "BANKSY (Harmony)",
    "GraphST full graph",
    "GraphST sampled",
    "STAGATE full graph",
    "STAGATE sampled",
)
SPATIAL_METHOD_ORDER = (
    "STEP sampled",
    "GraphST full graph",
    "GraphST sampled",
    "STAGATE full graph",
    "STAGATE sampled",
)
CAVITY = "Cavity"
CAVITY_COLOR = "#D95F02"
OTHER_COLOR = "#D9D9D9"


def section_root(section: str) -> Path:
    return E2S7_ROOT if section == "E16.5_E2S7" else OUTPUT_ROOT / section


def cluster_label(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def load_results() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics = []
    assignments = {}
    for section in SECTION_ORDER:
        root = section_root(section)
        frame = pd.read_csv(root / "common_spot_metrics_all_training_modes.csv")
        frame.insert(0, "section", section)
        metrics.append(frame)
        assignments[section] = pd.read_csv(
            root / "common_spot_assignments_all_training_modes.csv.gz",
            low_memory=False,
        )
    return pd.concat(metrics, ignore_index=True), assignments


def plot_spatial(axis: plt.Axes, frame: pd.DataFrame, mask: np.ndarray) -> None:
    x = frame["x"].to_numpy(dtype=float)
    y = -frame["y"].to_numpy(dtype=float)
    axis.scatter(
        x[~mask],
        y[~mask],
        c=OTHER_COLOR,
        s=0.12,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        x[mask],
        y[mask],
        c=CAVITY_COLOR,
        s=0.3,
        linewidths=0,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.axis("off")


def main() -> None:
    metrics, assignments = load_results()
    metrics.to_csv(OUTPUT_ROOT / "all_section_cavity_metrics.csv", index=False)

    figure = plt.figure(figsize=(15.5, 17.5), constrained_layout=True)
    grid = figure.add_gridspec(
        7,
        5,
        height_ratios=(1, 1, 1, 1, 1, 1, 1.35),
    )
    spatial_axes = np.asarray(
        [[figure.add_subplot(grid[row, col]) for col in range(5)] for row in range(6)]
    )
    heatmap_axis = figure.add_subplot(grid[6, :])

    for column, section in enumerate(SECTION_ORDER):
        frame = assignments[section]
        truth = frame["annotation"].astype(str).eq(CAVITY).to_numpy()
        plot_spatial(spatial_axes[0, column], frame, truth)
        for row, method in enumerate(SPATIAL_METHOD_ORDER, start=1):
            method_metric = metrics.loc[
                metrics["section"].eq(section) & metrics["method"].eq(method)
            ].iloc[0]
            target = cluster_label(method_metric["cavity_cluster"])
            predicted = (
                frame[f"cluster_{method}"].map(cluster_label).eq(target).to_numpy()
            )
            plot_spatial(spatial_axes[row, column], frame, predicted)
        spatial_axes[0, column].set_title(section.replace("E16.5_", ""), fontsize=11)
    spatial_axes[0, 0].text(
        -0.08,
        0.5,
        "Manual Cavity",
        transform=spatial_axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=11,
        fontweight="bold",
    )
    for row, method in enumerate(SPATIAL_METHOD_ORDER, start=1):
        spatial_axes[row, 0].text(
            -0.08,
            0.5,
            method,
            transform=spatial_axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

    matrix = (
        metrics.pivot(index="method", columns="section", values="cavity_f1")
        .reindex(index=METHOD_ORDER, columns=SECTION_ORDER)
        .rename(columns=lambda value: value.replace("E16.5_", ""))
    )
    sns.heatmap(
        matrix,
        ax=heatmap_axis,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Cavity F1"},
    )
    heatmap_axis.set_xlabel("MOSTA section")
    heatmap_axis.set_ylabel("")
    heatmap_axis.tick_params(axis="x", rotation=0)
    heatmap_axis.tick_params(axis="y", rotation=0)

    figure.legend(
        handles=(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CAVITY_COLOR,
                markeredgecolor="none",
                label="Cavity or matched cluster",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=OTHER_COLOR,
                markeredgecolor="none",
                label="Other tissue",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
    )

    figure.savefig(
        OUTPUT_ROOT / "mosta_cavity_all_sections.png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT_ROOT / "mosta_cavity_all_sections.pdf",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

    means = (
        metrics.groupby("method", observed=True)["cavity_f1"]
        .mean()
        .reindex(METHOD_ORDER)
    )
    summary = {
        "sections": list(SECTION_ORDER),
        "section_reference": {
            section: {
                "n_spots": int(assignments[section].shape[0]),
                "n_cavity": int(assignments[section]["annotation"].astype(str).eq(CAVITY).sum()),
                "n_reference_annotations": int(
                    assignments[section]["annotation"].astype(str).nunique()
                ),
            }
            for section in SECTION_ORDER
        },
        "mean_cavity_f1": {
            method: float(value) for method, value in means.items()
        },
    }
    (OUTPUT_ROOT / "all_section_cavity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
