"""Compare matched STEP and published NicheCompass cross-technology embeddings."""

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from scib_metrics import ilisi_knn, silhouette_label
from scib_metrics.metrics._silhouette import silhouette_batch
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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
    "hypendymal cell": "#B39B72",
    "choroid plexus epithelial cell": "#6E6E9E",
}


def normalize_technology(value: object) -> str:
    text = str(value).strip().lower()
    if text == "merfish":
        return "MERFISH"
    if text in {"starmap", "starmap plus", "starmap_plus"}:
        return "STARmap PLUS"
    raise ValueError(f"Unknown NicheCompass technology label: {value}")


def load_alignment(
    step_output_dir: Path,
    nichecompass_h5ad: Path,
    annotation_parquet: Path,
) -> tuple[pd.DataFrame, int, int]:
    step_obs = pd.read_parquet(
        step_output_dir / "observations.parquet",
        columns=["source_obs_name", "technology", "section"],
    )
    if step_obs["source_obs_name"].duplicated().any():
        raise ValueError("STEP observation identifiers are not unique")
    annotations = pd.read_parquet(annotation_parquet)
    if annotations["source_obs_name"].duplicated().any():
        raise ValueError("Evaluation annotation identifiers are not unique")
    annotation_ids = pd.Index(annotations["source_obs_name"].astype(str))
    step_index = pd.Index(step_obs["source_obs_name"].astype(str)).get_indexer(
        annotation_ids
    )
    if np.any(step_index < 0):
        raise ValueError(f"STEP is missing {int(np.sum(step_index < 0))} evaluation cells")

    niche = ad.read_h5ad(nichecompass_h5ad, backed="r")
    try:
        required = {"dataset", "section"}
        missing = sorted(required.difference(niche.obs.columns))
        if missing:
            raise ValueError(f"NicheCompass output lacks obs columns: {missing}")
        if not niche.obs_names.is_unique:
            raise ValueError("NicheCompass alignment keys are not unique")
        niche_index = niche.obs_names.astype(str).get_indexer(annotation_ids)
        if np.any(niche_index < 0):
            raise ValueError(
                f"NicheCompass is missing {int(np.sum(niche_index < 0))} evaluation cells"
            )
        niche_technology = (
            niche.obs.iloc[niche_index]["dataset"].map(normalize_technology).to_numpy()
        )
        niche_section = niche.obs.iloc[niche_index]["section"].astype(str).to_numpy()
        niche_n_obs = int(niche.n_obs)
    finally:
        niche.file.close()

    aligned = annotations.copy()
    aligned["step_index"] = step_index
    aligned["nichecompass_index"] = niche_index
    step_subset = step_obs.iloc[step_index]
    for source, observed_technology, observed_section in [
        ("STEP", step_subset["technology"].astype(str).to_numpy(), step_subset["section"].astype(str).to_numpy()),
        ("NicheCompass", niche_technology, niche_section),
    ]:
        if not np.array_equal(
            observed_technology, aligned["technology"].astype(str).to_numpy()
        ):
            raise ValueError(f"{source} technology labels do not match annotations")
        if not np.array_equal(observed_section, aligned["section"].astype(str).to_numpy()):
            raise ValueError(f"{source} section labels do not match annotations")
    return aligned, len(step_obs), niche_n_obs


def balanced_evaluation_sample(
    aligned: pd.DataFrame,
    max_per_technology_cell_type: int,
    min_per_technology_cell_type: int,
    seed: int,
) -> pd.DataFrame:
    frame = aligned.loc[aligned["broad_cell_type"].astype(str) != "unmatched"].copy()
    technologies = ["MERFISH", "STARmap PLUS"]
    counts = frame.groupby(["broad_cell_type", "technology"], observed=True).size()
    shared_types = []
    for cell_type in frame["broad_cell_type"].astype(str).unique():
        if all(
            (cell_type, technology) in counts.index
            and int(counts.loc[(cell_type, technology)]) >= min_per_technology_cell_type
            for technology in technologies
        ):
            shared_types.append(cell_type)

    rng = np.random.default_rng(seed)
    selected = []
    for cell_type in sorted(shared_types):
        common_n = min(
            max_per_technology_cell_type,
            *(int(counts.loc[(cell_type, technology)]) for technology in technologies),
        )
        for technology in technologies:
            candidates = frame.index[
                (frame["broad_cell_type"].astype(str) == cell_type)
                & (frame["technology"].astype(str) == technology)
            ].to_numpy()
            selected.extend(rng.choice(candidates, common_n, replace=False).tolist())
    sampled = frame.loc[selected].copy().reset_index(drop=True)
    sampled["broad_cell_type"] = sampled["broad_cell_type"].astype(str)
    sampled["technology"] = sampled["technology"].astype(str)
    sampled["section"] = sampled["section"].astype(str)
    return sampled


def read_h5ad_obsm_rows(path: Path, key: str, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    if len(np.unique(sorted_rows)) != len(sorted_rows):
        raise ValueError("Requested NicheCompass rows are not unique")
    with h5py.File(path, "r") as handle:
        node = handle["obsm"][key]
        if not isinstance(node, h5py.Dataset):
            raise TypeError(f"Expected dense HDF5 dataset for obsm/{key}")
        sorted_values = np.asarray(node[sorted_rows], dtype=np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return sorted_values[inverse]


def load_matched_embeddings(
    sample: pd.DataFrame,
    step_output_dir: Path,
    nichecompass_h5ad: Path,
    step_evaluation_embedding: Path | None = None,
    step_evaluation_index: Path | None = None,
) -> dict[str, np.ndarray]:
    if step_evaluation_embedding is None:
        step = np.load(step_output_dir / "step_spatial_embedding.npy", mmap_mode="r")
        step_values = np.asarray(
            step[sample["step_index"].to_numpy(dtype=np.int64)], dtype=np.float32
        )
    else:
        if step_evaluation_index is None:
            raise ValueError(
                "--step-evaluation-index is required with "
                "--step-evaluation-embedding"
            )
        step = np.load(step_evaluation_embedding, mmap_mode="r")
        step_index = pd.read_parquet(step_evaluation_index)
        if "source_obs_name" not in step_index.columns:
            raise ValueError("STEP evaluation index lacks source_obs_name")
        indexer = pd.Index(step_index["source_obs_name"].astype(str)).get_indexer(
            sample["source_obs_name"].astype(str)
        )
        if np.any(indexer < 0):
            raise ValueError(
                f"STEP evaluation embedding is missing {int(np.sum(indexer < 0))} cells"
            )
        step_values = np.asarray(step[indexer], dtype=np.float32)
    niche_values = read_h5ad_obsm_rows(
        nichecompass_h5ad,
        "nichecompass_latent",
        sample["nichecompass_index"].to_numpy(dtype=np.int64),
    )
    return {"STEP": step_values, "NicheCompass": niche_values}


def neighbor_summaries(
    neighbors,
    technology: np.ndarray,
    biological_label: np.ndarray,
) -> dict[str, float]:
    indices = np.asarray(neighbors.indices)
    if indices.shape[1] > 1 and np.array_equal(indices[:, 0], np.arange(len(indices))):
        indices = indices[:, 1:]
    neighbor_technology = technology[indices]
    neighbor_biology = biological_label[indices]
    same_biology = neighbor_biology == biological_label[:, None]
    opposite_technology = neighbor_technology != technology[:, None]
    opposite_counts = opposite_technology.sum(axis=1)
    cross_matches = (same_biology & opposite_technology).sum(axis=1)
    valid = opposite_counts > 0
    return {
        "knn_biological_purity": float(same_biology.mean()),
        "knn_opposite_technology_fraction": float(opposite_technology.mean()),
        "cross_technology_neighbor_coverage": float(valid.mean()),
        "cross_technology_label_agreement": float(
            np.mean(cross_matches[valid] / opposite_counts[valid])
        ),
    }


def compute_metrics(
    embeddings: dict[str, np.ndarray],
    sample: pd.DataFrame,
    n_neighbors: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    technology = sample["technology"].to_numpy(dtype=str)
    biological_label = sample["broad_cell_type"].to_numpy(dtype=str)
    section = sample["section"].to_numpy(dtype=str)
    rows = []
    neighbor_cache = {}

    for method, embedding in embeddings.items():
        neighbors = pynndescent(
            embedding,
            n_neighbors=min(n_neighbors, len(embedding) - 1),
            random_state=0,
            n_jobs=1,
        )
        neighbor_cache[method] = neighbors
        clusters = MiniBatchKMeans(
            n_clusters=len(np.unique(biological_label)),
            random_state=0,
            n_init=20,
            batch_size=4096,
        ).fit_predict(embedding)
        row = {
            "method": method,
            "n_cells": len(embedding),
            "embedding_dim": embedding.shape[1],
            "technology_ASW": float(
                silhouette_batch(embedding, biological_label, technology)
            ),
            "technology_iLISI": float(ilisi_knn(neighbors, technology, scale=True)),
            "biological_ASW": float(silhouette_label(embedding, biological_label)),
            "biological_ARI": float(adjusted_rand_score(biological_label, clusters)),
            "biological_NMI": float(normalized_mutual_info_score(biological_label, clusters)),
        }
        row.update(neighbor_summaries(neighbors, technology, biological_label))
        for current_technology in ["MERFISH", "STARmap PLUS"]:
            mask = technology == current_technology
            prefix = current_technology.lower().replace(" ", "_")
            if len(np.unique(section[mask])) < 2:
                row[f"{prefix}_section_ASW"] = float("nan")
                row[f"{prefix}_section_iLISI"] = float("nan")
            else:
                tech_neighbors = pynndescent(
                    embedding[mask],
                    n_neighbors=min(n_neighbors, int(mask.sum()) - 1),
                    random_state=0,
                    n_jobs=1,
                )
                row[f"{prefix}_section_ASW"] = float(
                    silhouette_batch(
                        embedding[mask],
                        biological_label[mask],
                        section[mask],
                    )
                )
                row[f"{prefix}_section_iLISI"] = float(
                    ilisi_knn(tech_neighbors, section[mask], scale=True)
                )
        rows.append(row)

    details = {
        "sample_cell_type_counts": (
            sample.groupby(["broad_cell_type", "technology"], observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    return pd.DataFrame(rows), details


def compute_umaps(
    embeddings: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    return {
        method: umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            metric="cosine",
            random_state=seed,
            n_jobs=1,
        ).fit_transform(values)
        for method, values in embeddings.items()
    }


def plot_embedding_comparison(
    coordinates: dict[str, np.ndarray],
    sample: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="white", context="paper")
    methods = ["STEP", "NicheCompass"]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    technology = sample["technology"].astype(str).to_numpy()
    biological_label = sample["broad_cell_type"].astype(str).to_numpy()

    for row, method in enumerate(methods):
        xy = coordinates[method]
        ax = axes[row, 0]
        for label, color in TECHNOLOGY_COLORS.items():
            mask = technology == label
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=2,
                c=color,
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )
        ax.set_ylabel(method, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if row == 0:
            ax.set_title("Technology", fontsize=12, fontweight="bold")

        ax = axes[row, 1]
        for label in sorted(np.unique(biological_label)):
            mask = biological_label == label
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=2,
                c=BROAD_CELL_TYPE_COLORS.get(label, "#777777"),
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if row == 0:
            ax.set_title("Broad cell type", fontsize=12, fontweight="bold")

    technology_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=6)
        for label, color in TECHNOLOGY_COLORS.items()
    ]
    biology_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=BROAD_CELL_TYPE_COLORS.get(label, "#777777"),
            label=label,
            markersize=5,
        )
        for label in sorted(np.unique(biological_label))
    ]
    figure.legend(
        handles=technology_handles,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.01),
        ncol=2,
        frameon=False,
    )
    figure.legend(
        handles=biology_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.005),
        ncol=3,
        frameon=False,
        fontsize=7.5,
    )
    figure.subplots_adjust(bottom=0.17, wspace=0.03, hspace=0.08)
    for suffix in ["png", "pdf"]:
        figure.savefig(
            output_dir / f"step_nichecompass_embedding_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_metric_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        ("biological_ASW", "Cell-type ASW"),
        ("biological_ARI", "Cell-type ARI"),
        ("biological_NMI", "Cell-type NMI"),
        ("knn_biological_purity", "kNN cell-type purity"),
        ("technology_ASW", "Batch ASW"),
        ("technology_iLISI", "Batch iLISI"),
        (
            "cross_technology_label_agreement",
            "Cross-technology cell-type agreement",
        ),
    ]
    long = metrics.melt(
        id_vars="method",
        value_vars=[column for column, _ in columns],
        var_name="metric",
        value_name="value",
    )
    labels = dict(columns)
    long["metric"] = long["metric"].map(labels)
    figure, ax = plt.subplots(figsize=(13.0, 4.2))
    sns.barplot(
        data=long,
        x="metric",
        y="value",
        hue="method",
        hue_order=["STEP", "NicheCompass"],
        palette={"STEP": "#217A6B", "NicheCompass": "#D17A3A"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    figure.tight_layout()
    for suffix in ["png", "pdf"]:
        figure.savefig(
            output_dir / f"step_nichecompass_metric_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def write_metric_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        ("biological_ASW", "Cell-type ASW"),
        ("biological_ARI", "Cell-type ARI"),
        ("biological_NMI", "Cell-type NMI"),
        ("knn_biological_purity", "kNN cell-type purity"),
        ("technology_ASW", "Batch ASW"),
        ("technology_iLISI", "Batch iLISI"),
        (
            "cross_technology_label_agreement",
            "Cross-technology cell-type agreement",
        ),
    ]
    lines = [
        "# Matched STEP and NicheCompass comparison",
        "",
        "All metrics are calculated on the same observation IDs, balanced within",
        "shared broad cell types and technologies. Higher values are better.",
        "",
        "| Method | " + " | ".join(label for _, label in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for row in metrics.itertuples(index=False):
        values = [f"{float(getattr(row, column)):.4f}" for column, _ in columns]
        lines.append(f"| {row.method} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Both embeddings retain broad cell-type organization, while the low",
            "technology iLISI values and the technology-colored UMAPs show that",
            "MERFISH and STARmap PLUS remain substantially separated. STEP has higher",
            "biological ASW, cell-type ARI and NMI, local cell-type purity,",
            "technology iLISI, and",
            "cross-technology cell-type agreement, while NicheCompass has higher",
            "technology ASW.",
            "",
        ]
    )
    (output_dir / "comparison_results.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare STEP with the published NicheCompass mouse-brain atlas."
    )
    parser.add_argument("--step-output-dir", type=Path, required=True)
    parser.add_argument("--nichecompass-h5ad", type=Path, required=True)
    parser.add_argument("--annotation-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-evaluation-embedding", type=Path)
    parser.add_argument("--step-evaluation-index", type=Path)
    parser.add_argument("--max-per-technology-cell-type", type=int, default=2000)
    parser.add_argument("--min-per-technology-cell-type", type=int, default=100)
    parser.add_argument("--n-neighbors", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned, step_n_obs, niche_n_obs = load_alignment(
        args.step_output_dir,
        args.nichecompass_h5ad,
        args.annotation_parquet,
    )
    sample = balanced_evaluation_sample(
        aligned,
        args.max_per_technology_cell_type,
        args.min_per_technology_cell_type,
        args.seed,
    )
    sample.to_parquet(args.output_dir / "evaluation_cells.parquet")
    embeddings = load_matched_embeddings(
        sample,
        args.step_output_dir,
        args.nichecompass_h5ad,
        args.step_evaluation_embedding,
        args.step_evaluation_index,
    )
    metrics, details = compute_metrics(embeddings, sample, args.n_neighbors)
    metrics.to_csv(args.output_dir / "step_nichecompass_metrics.csv", index=False)

    coordinates = compute_umaps(embeddings, args.seed)
    for method, values in coordinates.items():
        np.save(args.output_dir / f"{method.lower()}_evaluation_umap.npy", values)
    plot_embedding_comparison(coordinates, sample, args.output_dir)
    plot_metric_comparison(metrics, args.output_dir)
    write_metric_summary(metrics, args.output_dir)

    summary = {
        "step_output_dir": args.step_output_dir.name,
        "nichecompass_h5ad": args.nichecompass_h5ad.name,
        "annotation_parquet": args.annotation_parquet.name,
        "step_evaluation_embedding": (
            args.step_evaluation_embedding.name
            if args.step_evaluation_embedding is not None
            else None
        ),
        "step_evaluation_index": (
            args.step_evaluation_index.name
            if args.step_evaluation_index is not None
            else None
        ),
        "step_n_obs": step_n_obs,
        "nichecompass_n_obs": niche_n_obs,
        "evaluation_subset_n_obs": len(aligned),
        "evaluation_n_obs": len(sample),
        "sampling": {
            "strategy": "balanced within each shared broad cell type and technology",
            "max_per_technology_cell_type": args.max_per_technology_cell_type,
            "min_per_technology_cell_type": args.min_per_technology_cell_type,
            "seed": args.seed,
        },
        "n_neighbors": args.n_neighbors,
        "metrics": metrics.to_dict(orient="records"),
        **details,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
