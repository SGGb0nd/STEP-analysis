#!/usr/bin/env python
"""Plot original and matched-rerun STEP assignments for prostate sections."""


import argparse
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SOURCE = Path("results/slide-seq/prostate_slideseq_step.h5ad")
DEFAULT_ASSIGNMENTS = Path(
    "workflows/step_complete_neighborhood_controls/prostate/assignments.csv.gz"
)
DEFAULT_OUTPUT = Path(
    "workflows/step_complete_neighborhood_controls/prostate/"
    "prostate_original_matched_rerun_spatial"
)
SAMPLE_ORDER = ("Benign04", "HP1", "Tumor08", "Tumor02")
SAMPLE_TITLES = {
    "Benign04": "Healthy",
    "HP1": "Adjacent normal (LG)",
    "Tumor08": "Tumor (LG)",
    "Tumor02": "Tumor (HG)",
}
DOMAIN_TO_ZONE = {
    "1": "Epi",
    "2": "Boundary",
    "4": "Epi2",
    "5": "Fibro",
    "3": "Tumor",
}
ZONE_COLORS = {
    "Epi": "#1f77b4",
    "Boundary": "#ff7f0e",
    "Tumor": "#279e68",
    "Epi2": "#d62728",
    "Fibro": "#aa40fc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def maximum_overlap_map(reference: np.ndarray, candidate: np.ndarray) -> dict[str, str]:
    reference_levels = np.unique(reference)
    candidate_levels = np.unique(candidate)
    contingency = np.zeros(
        (len(reference_levels), len(candidate_levels)), dtype=np.int64
    )
    for i, reference_level in enumerate(reference_levels):
        for j, candidate_level in enumerate(candidate_levels):
            contingency[i, j] = np.sum(
                (reference == reference_level) & (candidate == candidate_level)
            )
    rows, columns = linear_sum_assignment(-contingency)
    return {
        str(candidate_levels[column]): str(reference_levels[row])
        for row, column in zip(rows, columns)
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.source, backed="r")
    assignments = pd.read_csv(args.assignments, index_col=0)
    batches = adata.obs["sample"].astype(str).to_numpy()
    if not np.array_equal(batches, assignments["batch"].astype(str).to_numpy()):
        raise AssertionError("source and assignment batch order differs")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    adata.file.close()

    sampled = assignments["sampled_domain"].astype(str).to_numpy()
    full = assignments["full_domain"].astype(str).to_numpy()
    full_to_sampled = maximum_overlap_map(sampled, full)
    full_aligned = np.asarray([full_to_sampled[value] for value in full])

    figure = plt.figure(figsize=(14.2, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        5,
        width_ratios=(0.11, 1, 1, 1, 1),
        wspace=0.04,
        hspace=0.05,
    )
    row_names = ("Original STEP", "Matched full-section rerun")
    label_sets = (sampled, full_aligned)
    for row, (row_name, labels) in enumerate(zip(row_names, label_sets)):
        label_axis = figure.add_subplot(grid[row, 0])
        label_axis.text(
            0.5,
            0.5,
            row_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        label_axis.axis("off")
        for column, sample in enumerate(SAMPLE_ORDER, start=1):
            axis = figure.add_subplot(grid[row, column])
            mask = batches == sample
            colors = [ZONE_COLORS[DOMAIN_TO_ZONE[value]] for value in labels[mask]]
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors,
                s=0.45,
                linewidths=0,
                rasterized=True,
            )
            axis.set_aspect("equal")
            axis.axis("off")
            if row == 0:
                axis.set_title(SAMPLE_TITLES[sample], fontsize=13)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ZONE_COLORS[zone],
            markeredgecolor="none",
            markersize=7,
            label=zone,
        )
        for zone in ("Epi", "Boundary", "Tumor", "Epi2", "Fibro")
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=11,
    )
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    args.output.with_name("prostate_sampled_fullgraph_spatial_legend.md").write_text(
        "**Prostate spatial assignments.** Rows indicate the original STEP result "
        "and a matched full-section rerun. Colors indicate the five matched spatial domains: "
        "Epi, Boundary, Epi2, Fibro, and Tumor.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
