"""Compare STEP and NicheCompass using external anatomical-domain labels."""

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scib_metrics import ilisi_knn, silhouette_label
from scib_metrics.metrics._silhouette import silhouette_batch
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def normalize_technology(value: object) -> str:
    text = str(value).strip().lower()
    if text == "merfish":
        return "MERFISH"
    if text in {"starmap", "starmap plus", "starmap_plus"}:
        return "STARmap PLUS"
    raise ValueError(f"Unknown technology label: {value}")


def load_alignment(
    step_output_dir: Path,
    nichecompass_h5ad: Path,
    annotation_parquet: Path,
) -> pd.DataFrame:
    step_obs = pd.read_parquet(
        step_output_dir / "observations.parquet",
        columns=["source_obs_name", "technology", "section"],
    )
    annotations = pd.read_parquet(annotation_parquet)
    if annotations["source_obs_name"].duplicated().any():
        raise ValueError("Domain annotation identifiers are not unique")
    annotation_ids = pd.Index(annotations["source_obs_name"].astype(str))
    step_index = pd.Index(step_obs["source_obs_name"].astype(str)).get_indexer(
        annotation_ids
    )
    if np.any(step_index < 0):
        raise ValueError(f"STEP is missing {int(np.sum(step_index < 0))} cells")

    niche = ad.read_h5ad(nichecompass_h5ad, backed="r")
    try:
        niche_index = niche.obs_names.astype(str).get_indexer(annotation_ids)
        if np.any(niche_index < 0):
            raise ValueError(
                f"NicheCompass is missing {int(np.sum(niche_index < 0))} cells"
            )
        niche_technology = (
            niche.obs.iloc[niche_index]["dataset"].map(normalize_technology).to_numpy()
        )
        niche_section = niche.obs.iloc[niche_index]["section"].astype(str).to_numpy()
    finally:
        niche.file.close()

    aligned = annotations.copy()
    aligned["step_index"] = step_index
    aligned["nichecompass_index"] = niche_index
    step_subset = step_obs.iloc[step_index]
    expected_technology = aligned["technology"].astype(str).to_numpy()
    expected_section = aligned["section"].astype(str).to_numpy()
    for source, technology, section in [
        (
            "STEP",
            step_subset["technology"].astype(str).to_numpy(),
            step_subset["section"].astype(str).to_numpy(),
        ),
        ("NicheCompass", niche_technology, niche_section),
    ]:
        if not np.array_equal(technology, expected_technology):
            raise ValueError(f"{source} technology labels do not align")
        if not np.array_equal(section, expected_section):
            raise ValueError(f"{source} section labels do not align")
    return aligned


def balanced_sample(
    frame: pd.DataFrame,
    group_columns: list[str],
    label_column: str,
    max_per_group: int,
    min_per_group: int,
    seed: int,
) -> pd.DataFrame:
    frame = frame.loc[frame[label_column].astype(str) != "unassigned"].copy()
    counts = frame.groupby(group_columns, observed=True).size()
    valid_groups = counts.loc[counts >= min_per_group].index
    rng = np.random.default_rng(seed)
    selected = []
    for group in valid_groups:
        group = group if isinstance(group, tuple) else (group,)
        mask = np.ones(len(frame), dtype=bool)
        for column, value in zip(group_columns, group):
            mask &= frame[column].astype(str).to_numpy() == str(value)
        candidates = frame.index[mask].to_numpy()
        n_cells = min(max_per_group, len(candidates))
        selected.extend(rng.choice(candidates, n_cells, replace=False).tolist())
    sample = frame.loc[selected].copy().reset_index(drop=True)
    for column in set(group_columns + [label_column, "technology", "section"]):
        sample[column] = sample[column].astype(str)
    return sample


def balanced_shared_sample(
    aligned: pd.DataFrame,
    max_per_group: int,
    min_per_group: int,
    seed: int,
) -> pd.DataFrame:
    frame = aligned.loc[
        aligned["shared_domain_label"].astype(str) != "unassigned"
    ].copy()
    technologies = ["MERFISH", "STARmap PLUS"]
    counts = frame.groupby(
        ["shared_domain_label", "technology"], observed=True
    ).size()
    shared = [
        label
        for label in frame["shared_domain_label"].astype(str).unique()
        if all(
            (label, technology) in counts.index
            and counts.loc[(label, technology)] >= min_per_group
            for technology in technologies
        )
    ]
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(shared):
        common_n = min(
            max_per_group,
            *(int(counts.loc[(label, technology)]) for technology in technologies),
        )
        for technology in technologies:
            candidates = frame.index[
                (frame["shared_domain_label"].astype(str) == label)
                & (frame["technology"].astype(str) == technology)
            ].to_numpy()
            selected.extend(rng.choice(candidates, common_n, replace=False).tolist())
    sample = frame.loc[selected].copy().reset_index(drop=True)
    for column in ["shared_domain_label", "technology", "section"]:
        sample[column] = sample[column].astype(str)
    return sample


def read_h5ad_obsm_rows(path: Path, key: str, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["obsm"][key][sorted_rows], dtype=np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return values[inverse]


def load_embeddings(
    sample: pd.DataFrame,
    step_output_dir: Path,
    nichecompass_h5ad: Path,
) -> dict[str, np.ndarray]:
    step = np.load(step_output_dir / "step_spatial_embedding.npy", mmap_mode="r")
    step_values = np.asarray(
        step[sample["step_index"].to_numpy(dtype=np.int64)], dtype=np.float32
    )
    niche_values = read_h5ad_obsm_rows(
        nichecompass_h5ad,
        "nichecompass_latent",
        sample["nichecompass_index"].to_numpy(dtype=np.int64),
    )
    return {"STEP": step_values, "NicheCompass": niche_values}


def knn_purity(neighbor_indices: np.ndarray, labels: np.ndarray) -> float:
    indices = np.asarray(neighbor_indices)
    if indices.shape[1] > 1 and np.array_equal(indices[:, 0], np.arange(len(indices))):
        indices = indices[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))


def domain_metrics(
    embeddings: dict[str, np.ndarray],
    sample: pd.DataFrame,
    label_column: str,
    analysis: str,
    n_neighbors: int,
    report_batch_mixing: bool,
) -> pd.DataFrame:
    labels = sample[label_column].to_numpy(dtype=str)
    batches = sample["technology"].to_numpy(dtype=str)
    if len(np.unique(batches)) == 1:
        batches = sample["section"].to_numpy(dtype=str)
    rows = []
    for method, embedding in embeddings.items():
        neighbors = pynndescent(
            embedding,
            n_neighbors=min(n_neighbors, len(embedding) - 1),
            random_state=0,
            n_jobs=1,
        )
        clusters = MiniBatchKMeans(
            n_clusters=len(np.unique(labels)),
            random_state=0,
            n_init=20,
            batch_size=4096,
        ).fit_predict(embedding)
        row = {
            "analysis": analysis,
            "method": method,
            "n_cells": len(embedding),
            "n_domains": len(np.unique(labels)),
            "domain_ASW": float(silhouette_label(embedding, labels)),
            "domain_ARI": float(adjusted_rand_score(labels, clusters)),
            "domain_NMI": float(normalized_mutual_info_score(labels, clusters)),
            "knn_domain_purity": knn_purity(neighbors.indices, labels),
        }
        if report_batch_mixing and len(np.unique(batches)) > 1:
            row["batch_ASW"] = float(silhouette_batch(embedding, labels, batches))
            row["batch_iLISI"] = float(ilisi_knn(neighbors, batches, scale=True))
        else:
            row["batch_ASW"] = float("nan")
            row["batch_iLISI"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        ("domain_ASW", "Domain ASW"),
        ("domain_ARI", "Domain ARI"),
        ("domain_NMI", "Domain NMI"),
        ("knn_domain_purity", "kNN domain purity"),
        ("batch_ASW", "Batch ASW"),
        ("batch_iLISI", "Batch iLISI"),
    ]
    long = metrics.melt(
        id_vars=["analysis", "method"],
        value_vars=[column for column, _ in columns],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(dict(columns))
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 10.5), sharex=True)
    for ax, analysis in zip(axes, metrics["analysis"].unique()):
        subset = long.loc[long["analysis"] == analysis]
        sns.barplot(
            data=subset,
            x="metric",
            y="value",
            hue="method",
            hue_order=["STEP", "NicheCompass"],
            palette={"STEP": "#217A6B", "NicheCompass": "#D17A3A"},
            ax=ax,
        )
        ax.set_title(analysis, loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.legend(frameon=False, title="")
        elif ax.legend_ is not None:
            ax.legend_.remove()
    axes[-1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    for suffix in ["png", "pdf"]:
        figure.savefig(
            output_dir / f"step_nichecompass_domain_metrics.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare spatial embeddings against external anatomical domains."
    )
    parser.add_argument("--step-output-dir", type=Path, required=True)
    parser.add_argument("--nichecompass-h5ad", type=Path, required=True)
    parser.add_argument("--domain-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-domain", type=int, default=2000)
    parser.add_argument("--min-per-domain", type=int, default=100)
    parser.add_argument("--n-neighbors", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned = load_alignment(
        args.step_output_dir,
        args.nichecompass_h5ad,
        args.domain_annotations,
    )
    analyses = []
    sample_summaries = {}
    all_samples = []
    for technology in ["MERFISH", "STARmap PLUS"]:
        sample = balanced_sample(
            aligned.loc[aligned["technology"].astype(str) == technology],
            ["domain_label"],
            "domain_label",
            args.max_per_domain,
            args.min_per_domain,
            args.seed,
        )
        analysis = f"{technology} anatomical-domain recovery"
        analyses.append(
            domain_metrics(
                load_embeddings(sample, args.step_output_dir, args.nichecompass_h5ad),
                sample,
                "domain_label",
                analysis,
                args.n_neighbors,
                report_batch_mixing=True,
            )
        )
        sample["analysis"] = analysis
        all_samples.append(sample)
        sample_summaries[analysis] = (
            sample["domain_label"].value_counts().sort_index().to_dict()
        )

    shared = balanced_shared_sample(
        aligned,
        args.max_per_domain,
        args.min_per_domain,
        args.seed,
    )
    shared_analysis = "Shared anatomical-region integration"
    analyses.append(
        domain_metrics(
            load_embeddings(shared, args.step_output_dir, args.nichecompass_h5ad),
            shared,
            "shared_domain_label",
            shared_analysis,
            args.n_neighbors,
            report_batch_mixing=True,
        )
    )
    shared["analysis"] = shared_analysis
    all_samples.append(shared)
    sample_summaries[shared_analysis] = (
        shared.groupby(["shared_domain_label", "technology"])
        .size()
        .rename("n_cells")
        .reset_index()
        .to_dict(orient="records")
    )

    metrics = pd.concat(analyses, ignore_index=True)
    metrics.to_csv(args.output_dir / "step_nichecompass_domain_metrics.csv", index=False)
    pd.concat(all_samples, ignore_index=True).to_parquet(
        args.output_dir / "evaluation_cells.parquet", index=False
    )
    plot_metrics(metrics, args.output_dir)
    summary = {
        "step_embedding": "X_smoothed",
        "nichecompass_embedding": "nichecompass_latent",
        "domain_annotations": args.domain_annotations.name,
        "sampling": {
            "max_per_domain": args.max_per_domain,
            "min_per_domain": args.min_per_domain,
            "seed": args.seed,
        },
        "sample_counts": sample_summaries,
        "metrics": metrics.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
