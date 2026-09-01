#!/usr/bin/env python
"""Plot original sampled and matched full-neighbor CRC assignments."""


import argparse
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from scipy.optimize import linear_sum_assignment


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SOURCE = Path("results/visium-hd/crc_8um_5slices_domain.h5ad")
DEFAULT_RUN_DIR = Path("workflows/step_complete_neighborhood_controls/crc")
SECTION_ORDER = (
    "cancer_p1",
    "cancer_p2",
    "cancer_p5",
    "normal_p3",
    "normal_p5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_RUN_DIR / "assignments.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN_DIR / "crc_sampled_fullneighbor_spatial",
    )
    return parser.parse_args()


def maximum_overlap_map(reference: np.ndarray, candidate: np.ndarray) -> dict[str, str]:
    """Map candidate labels one-to-one onto reference labels by global overlap."""
    reference_levels = np.unique(reference)
    candidate_levels = np.unique(candidate)
    contingency = np.zeros(
        (len(reference_levels), len(candidate_levels)), dtype=np.int64
    )
    for row, reference_level in enumerate(reference_levels):
        reference_mask = reference == reference_level
        for column, candidate_level in enumerate(candidate_levels):
            contingency[row, column] = np.count_nonzero(
                reference_mask & (candidate == candidate_level)
            )
    rows, columns = linear_sum_assignment(-contingency)
    return {
        str(candidate_levels[column]): str(reference_levels[row])
        for row, column in zip(rows, columns)
    }


def numeric_label_key(label: str) -> tuple[int, str]:
    try:
        return int(label), label
    except ValueError:
        return 10**9, label


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.source, backed="r")
    assignments = pd.read_csv(args.assignments, index_col=0)
    sections = adata.obs["batch"].astype(str).to_numpy()
    assigned_sections = assignments["batch"].astype(str).to_numpy()
    if not np.array_equal(sections, assigned_sections):
        raise AssertionError("source and assignment section order differs")

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    domain_categories = [str(value) for value in adata.obs["domain"].cat.categories]
    domain_colors = [str(value) for value in adata.uns["domain_colors"]]
    adata.file.close()
    color_map = dict(zip(domain_categories, domain_colors, strict=True))

    sampled = assignments["sampled_domain"].astype(str).to_numpy()
    full = assignments["full_domain"].astype(str).to_numpy()
    full_to_sampled = maximum_overlap_map(sampled, full)
    full_aligned = np.asarray([full_to_sampled[value] for value in full])

    figure = plt.figure(figsize=(18.0, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        6,
        width_ratios=(0.09, 1, 1, 1, 1, 1),
        wspace=0.015,
        hspace=0.015,
    )
    row_names = ("Original sampled STEP", "Matched full-neighbor STEP")
    label_sets = (sampled, full_aligned)

    for row, (row_name, labels) in enumerate(zip(row_names, label_sets, strict=True)):
        row_axis = figure.add_subplot(grid[row, 0])
        row_axis.text(
            0.5,
            0.5,
            row_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        row_axis.axis("off")

        point_colors = np.asarray([color_map[value] for value in labels])
        for column, section in enumerate(SECTION_ORDER, start=1):
            axis = figure.add_subplot(grid[row, column])
            mask = sections == section
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=point_colors[mask],
                s=0.055,
                linewidths=0,
                rasterized=True,
            )
            axis.set_aspect("equal")
            axis.invert_yaxis()
            axis.axis("off")
            if row == 0:
                axis.set_title(section, fontsize=12)

    labels = sorted(np.unique(sampled), key=numeric_label_key)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[label],
            markeredgecolor="none",
            markersize=6,
            label=f"Domain {label}",
        )
        for label in labels
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=6,
        frameon=False,
        fontsize=9,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, resolution=300.0)


if __name__ == "__main__":
    main()
