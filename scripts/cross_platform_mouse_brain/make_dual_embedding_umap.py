"""Plot matched no-graph and graph-aggregated cross-technology embeddings."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D


TECHNOLOGY_COLORS = {
    "MERFISH": "#1B6CA8",
    "STARmap PLUS": "#D97732",
}

BROAD_CELL_TYPE_COLORS = {
    "excitatory neuron": "#D8A03D",
    "inhibitory neuron": "#C95B5B",
    "cholinergic or monoaminergic neuron": "#A862A8",
    "neuroblast": "#8F7AC2",
    "astrocyte": "#4C9A73",
    "oligodendrocyte": "#377EB8",
    "oligodendrocyte precursor cell": "#75AADB",
    "microglia": "#8B6B4A",
    "macrophage": "#B07C5A",
    "pericyte": "#4CA6A8",
    "endothelial cell": "#66B3A6",
    "vascular and leptomeningeal cell": "#7A9A50",
    "vascular smooth muscle cell": "#6C8B7B",
    "olfactory ensheathing cell": "#C17F9E",
    "ependymal cell": "#9B8B63",
}

DOMAIN_COLORS = {
    "Fiber tracts": "#4C78A8",
    "Hippocampal formation": "#F58518",
    "Isocortex": "#54A24B",
    "Olfactory areas": "#E45756",
    "Striatum": "#B279A2",
}


def compute_umap(values: np.ndarray, seed: int) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="cosine",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(values)


def scatter_categories(
    ax,
    coordinates: np.ndarray,
    labels: np.ndarray,
    colors: dict[str, str],
) -> None:
    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=1.6,
            c=colors.get(label, "#777777"),
            alpha=0.52,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def legend_handles(labels: list[str], colors: dict[str, str], size: float = 5):
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=colors.get(label, "#777777"),
            label=label,
            markersize=size,
        )
        for label in labels
    ]


def plot_dual_embedding(
    cell_sample: pd.DataFrame,
    cell_coordinates: dict[str, np.ndarray],
    domain_sample: pd.DataFrame,
    step_domain_coordinates: np.ndarray,
    output_dir: Path,
) -> None:
    sns.set_theme(style="white", context="paper")
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.4))

    cell_technology = cell_sample["technology"].astype(str).to_numpy()
    cell_types = cell_sample["broad_cell_type"].astype(str).to_numpy()
    domain_technology = domain_sample["technology"].astype(str).to_numpy()
    domains = domain_sample["shared_domain_label"].astype(str).to_numpy()

    columns = [
        (
            "STEP no-graph",
            cell_coordinates["STEP"],
            cell_technology,
            cell_types,
            BROAD_CELL_TYPE_COLORS,
        ),
        (
            "STEP graph-aggregated",
            step_domain_coordinates,
            domain_technology,
            domains,
            DOMAIN_COLORS,
        ),
        (
            "NicheCompass latent",
            cell_coordinates["NicheCompass"],
            cell_technology,
            cell_types,
            BROAD_CELL_TYPE_COLORS,
        ),
    ]
    for column, (title, coordinates, technology, annotation, colors) in enumerate(
        columns
    ):
        scatter_categories(
            axes[0, column], coordinates, technology, TECHNOLOGY_COLORS
        )
        scatter_categories(axes[1, column], coordinates, annotation, colors)
        axes[0, column].set_title(title, fontsize=12, fontweight="bold")

    axes[0, 0].set_ylabel("Technology", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Reference annotation", fontsize=11, fontweight="bold")

    technology_labels = list(TECHNOLOGY_COLORS)
    cell_type_labels = sorted(np.unique(cell_types).tolist())
    domain_labels = sorted(np.unique(domains).tolist())
    figure.legend(
        handles=legend_handles(technology_labels, TECHNOLOGY_COLORS, size=6),
        loc="lower left",
        bbox_to_anchor=(0.035, 0.03),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    figure.legend(
        handles=legend_handles(cell_type_labels, BROAD_CELL_TYPE_COLORS),
        loc="lower center",
        bbox_to_anchor=(0.47, 0.005),
        ncol=3,
        frameon=False,
        fontsize=7.1,
    )
    figure.legend(
        handles=legend_handles(domain_labels, DOMAIN_COLORS, size=6),
        loc="lower right",
        bbox_to_anchor=(0.985, 0.03),
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.995,
        top=0.94,
        bottom=0.23,
        wspace=0.06,
        hspace=0.07,
    )
    for suffix in ["png", "pdf"]:
        figure.savefig(
            output_dir / f"step_nichecompass_dual_embedding_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot both STEP representation levels in the matched atlas."
    )
    parser.add_argument("--celltype-comparison-dir", type=Path, required=True)
    parser.add_argument("--domain-comparison-dir", type=Path, required=True)
    parser.add_argument("--step-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cell_sample = pd.read_parquet(
        args.celltype_comparison_dir / "evaluation_cells.parquet"
    )
    cell_coordinates = {
        "STEP": np.load(
            args.celltype_comparison_dir / "step_evaluation_umap.npy"
        ),
        "NicheCompass": np.load(
            args.celltype_comparison_dir / "nichecompass_evaluation_umap.npy"
        ),
    }

    domain_sample = pd.read_parquet(
        args.domain_comparison_dir / "evaluation_cells.parquet"
    )
    domain_sample = domain_sample.loc[
        domain_sample["analysis"].astype(str)
        == "Shared anatomical-region integration"
    ].reset_index(drop=True)
    step_embedding = np.load(
        args.step_output_dir / "step_spatial_embedding.npy", mmap_mode="r"
    )
    step_domain_embedding = np.asarray(
        step_embedding[domain_sample["step_index"].to_numpy(dtype=np.int64)],
        dtype=np.float32,
    )
    step_domain_coordinates = compute_umap(step_domain_embedding, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_dir / "step_shared_domain_umap.npy",
        step_domain_coordinates,
    )
    plot_dual_embedding(
        cell_sample,
        cell_coordinates,
        domain_sample,
        step_domain_coordinates,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
