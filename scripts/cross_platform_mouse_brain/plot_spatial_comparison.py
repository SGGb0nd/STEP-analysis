"""Plot matched spatial clustering from STEP and NicheCompass embeddings."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans

from compare_step_nichecompass import (
    BROAD_CELL_TYPE_COLORS,
    load_matched_embeddings,
    read_h5ad_obsm_rows,
)


DEFAULT_SECTIONS = {
    "MERFISH": "C57BL6J-1.014",
    "STARmap PLUS": "sagittal3",
}
OTHER_LABEL = "other or unmatched"
OTHER_COLOR = "#D0D0D0"


def fit_cluster_label_map(
    embedding: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[MiniBatchKMeans, dict[int, str]]:
    unique_labels = np.sort(np.unique(labels))
    model = MiniBatchKMeans(
        n_clusters=len(unique_labels),
        random_state=seed,
        n_init=20,
        batch_size=4096,
    ).fit(embedding)
    clusters = model.labels_
    contingency = np.zeros((len(unique_labels), model.n_clusters), dtype=np.int64)
    for label_idx, label in enumerate(unique_labels):
        contingency[label_idx] = np.bincount(
            clusters[labels == label], minlength=model.n_clusters
        )
    label_rows, cluster_cols = linear_sum_assignment(-contingency)
    mapping = {
        int(cluster): str(unique_labels[label])
        for label, cluster in zip(label_rows, cluster_cols)
    }
    return model, mapping


def load_selected_sections(
    observations_path: Path,
    annotations_path: Path,
    sections: dict[str, str],
) -> dict[str, pd.DataFrame]:
    observations = pd.read_parquet(
        observations_path,
        columns=["source_obs_name", "technology", "section", "x", "y"],
    )
    selected = {}
    for technology, section in sections.items():
        mask = (
            observations["technology"].astype(str).eq(technology)
            & observations["section"].astype(str).eq(section)
        ).to_numpy()
        rows = np.flatnonzero(mask)
        if len(rows) == 0:
            raise ValueError(f"Section {technology}::{section} was not found")
        frame = observations.iloc[rows].copy()
        frame["embedding_index"] = rows
        annotations = pd.read_parquet(
            annotations_path,
            filters=[("technology", "==", technology), ("section", "==", section)],
            columns=["source_obs_name", "broad_cell_type"],
        ).set_index("source_obs_name")
        frame["broad_cell_type"] = annotations["broad_cell_type"].reindex(
            frame["source_obs_name"]
        ).to_numpy()
        if frame["broad_cell_type"].isna().any():
            raise ValueError(f"Missing public annotations in {technology}::{section}")
        selected[technology] = frame
    return selected


def predict_selected_sections(
    selected: dict[str, pd.DataFrame],
    step_output_dir: Path,
    nichecompass_h5ad: Path,
    models: dict[str, MiniBatchKMeans],
    mappings: dict[str, dict[int, str]],
) -> None:
    step_embedding = np.load(
        step_output_dir / "step_spatial_embedding.npy", mmap_mode="r"
    )
    for frame in selected.values():
        rows = frame["embedding_index"].to_numpy(dtype=np.int64)
        method_embeddings = {
            "STEP": np.asarray(step_embedding[rows], dtype=np.float32),
            "NicheCompass": read_h5ad_obsm_rows(
                nichecompass_h5ad, "nichecompass_latent", rows
            ),
        }
        for method, embedding in method_embeddings.items():
            clusters = models[method].predict(embedding)
            frame[method] = [mappings[method][int(cluster)] for cluster in clusters]


def plot_spatial_comparison(
    selected: dict[str, pd.DataFrame],
    shared_labels: list[str],
    output_dir: Path,
) -> None:
    sns.set_theme(style="white", context="paper")
    technologies = list(DEFAULT_SECTIONS)
    rows = [
        ("broad_cell_type", "Public annotation"),
        ("STEP", "STEP"),
        ("NicheCompass", "NicheCompass"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(10.8, 12.0))
    shared = set(shared_labels)

    for column, technology in enumerate(technologies):
        frame = selected[technology]
        point_size = max(0.12, min(2.0, 35000.0 / len(frame)))
        for row, (label_key, row_label) in enumerate(rows):
            ax = axes[row, column]
            labels = frame[label_key].astype(str).where(
                frame[label_key].astype(str).isin(shared), OTHER_LABEL
            )
            order = [OTHER_LABEL] + shared_labels
            for label in order:
                mask = labels.eq(label).to_numpy()
                if not np.any(mask):
                    continue
                ax.scatter(
                    frame.loc[mask, "x"],
                    frame.loc[mask, "y"],
                    s=point_size,
                    c=OTHER_COLOR if label == OTHER_LABEL else BROAD_CELL_TYPE_COLORS[label],
                    alpha=0.28 if label == OTHER_LABEL else 0.8,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if row == 0:
                ax.set_title(
                    f"{technology}\n{DEFAULT_SECTIONS[technology]}",
                    fontsize=12,
                    fontweight="bold",
                )
            if column == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=BROAD_CELL_TYPE_COLORS[label],
            label=label,
            markersize=5,
        )
        for label in shared_labels
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=OTHER_COLOR,
            label=OTHER_LABEL,
            markersize=5,
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(bottom=0.16, wspace=0.04, hspace=0.08)
    for suffix in ["png", "pdf"]:
        figure.savefig(
            output_dir / f"step_nichecompass_spatial_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot matched spatial results for STEP and NicheCompass."
    )
    parser.add_argument("--step-output-dir", type=Path, required=True)
    parser.add_argument("--nichecompass-h5ad", type=Path, required=True)
    parser.add_argument("--annotation-parquet", type=Path, required=True)
    parser.add_argument("--evaluation-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluation = pd.read_parquet(args.evaluation_cells)
    if not np.array_equal(
        evaluation["step_index"].to_numpy(),
        evaluation["nichecompass_index"].to_numpy(),
    ):
        raise ValueError("STEP and NicheCompass row indices are not aligned")
    evaluation_embeddings = load_matched_embeddings(
        evaluation, args.step_output_dir, args.nichecompass_h5ad
    )
    labels = evaluation["broad_cell_type"].astype(str).to_numpy()
    models = {}
    mappings = {}
    for method, embedding in evaluation_embeddings.items():
        models[method], mappings[method] = fit_cluster_label_map(
            embedding, labels, args.seed
        )

    selected = load_selected_sections(
        args.step_output_dir / "observations.parquet",
        args.annotation_parquet,
        DEFAULT_SECTIONS,
    )
    predict_selected_sections(
        selected,
        args.step_output_dir,
        args.nichecompass_h5ad,
        models,
        mappings,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_labels = sorted(np.unique(labels))
    plot_spatial_comparison(selected, shared_labels, args.output_dir)


if __name__ == "__main__":
    main()
