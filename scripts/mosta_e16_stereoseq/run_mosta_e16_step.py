"""Train STEP and evaluate spatial domains on five MOSTA E16.5 sections."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scib_metrics import ilisi_knn
from scib_metrics.metrics._silhouette import silhouette_batch
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from step import stModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_H5AD = REPO_ROOT / "data" / "mosta_e16_test.h5ad"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "mosta_e16_step"

MODEL_CONFIG = {
    "module_dim": 30,
    "hidden_dim": 64,
    "n_modules": 32,
    "n_dec_hid_layers": 2,
    "batch_injection_mode": "scale",
    "n_glayers": 2,
    "edge_clip": 1.5,
}

TRAINING_CONFIG = {
    "graph_batch_size": 2,
    "sample_rate": 2048,
    "batch_inference": True,
    "inference_batch_size": 64,
}

N_CLUSTERS = 27


def cal_chaos(cluster_labels: np.ndarray, locations: np.ndarray) -> float:
    """Calculate the CHAOS spatial continuity score."""
    labels = np.asarray(cluster_labels)
    coordinates = StandardScaler().fit_transform(np.asarray(locations))
    cluster_distances: list[float] = []
    for cluster_id in np.unique(labels):
        cluster_points = coordinates[labels == cluster_id]
        if len(cluster_points) <= 2:
            continue
        tree = KDTree(cluster_points)
        distances, _ = tree.query(cluster_points, k=2)
        cluster_distances.append(float(np.sum(distances[:, 1])))
    return float(np.sum(cluster_distances) / len(labels))


def cal_pas(
    cluster_labels: np.ndarray,
    locations: np.ndarray,
    k: int = 10,
) -> float:
    """Calculate the percentage of adjacent spots with matching labels."""
    labels = np.asarray(cluster_labels)
    tree = KDTree(np.asarray(locations))
    _, indices = tree.query(locations, k=k + 1)
    neighbor_labels = labels[indices[:, 1:]]
    return float(np.mean(neighbor_labels == labels[:, np.newaxis]))


def load_mosta_input(path: Path) -> ad.AnnData:
    """Load counts and preserve the section order stored in the input file."""
    adata = ad.read_h5ad(path)
    required_obs = {"annotation", "batch", "x", "y"}
    missing_obs = sorted(required_obs.difference(adata.obs.columns))
    if missing_obs:
        raise ValueError(f"Missing required obs columns: {missing_obs}")
    if "count" not in adata.layers:
        raise ValueError("The input AnnData must contain raw counts in layers['count']")

    batch_values = adata.obs["batch"].astype(str)
    adata.obs["batch"] = pd.Categorical(
        batch_values,
        categories=list(dict.fromkeys(batch_values.tolist())),
        ordered=True,
    )
    return adata


def compute_batch_metrics(
    adata: ad.AnnData,
    embedding_key: str = "X_smoothed",
) -> dict[str, float]:
    """Calculate dataset-level batch ASW and iLISI."""
    embedding = np.asarray(adata.obsm[embedding_key], dtype=float)
    labels = adata.obs["annotation"].astype(str).to_numpy()
    batches = adata.obs["batch"].astype(str).to_numpy()
    neighbors = pynndescent(
        embedding,
        n_neighbors=min(90, len(embedding) - 1),
        random_state=0,
        n_jobs=1,
    )
    return {
        "ASW": float(silhouette_batch(embedding, labels, batches)),
        "iLISI": float(ilisi_knn(neighbors, batches, scale=True)),
        "n_spots": int(adata.n_obs),
    }


def compute_section_metrics(
    adata: ad.AnnData,
    batch_metrics: dict[str, float],
) -> pd.DataFrame:
    """Calculate domain recovery and spatial continuity for each section."""
    rows: list[dict[str, object]] = []
    for section in adata.obs["batch"].cat.categories:
        section_data = adata[adata.obs["batch"] == section]
        domains = section_data.obs["domain"].astype(str).to_numpy()
        annotations = section_data.obs["annotation"].astype(str).to_numpy()
        coordinates = np.asarray(section_data.obsm["spatial"])
        rows.append(
            {
                "dataset": "MOSTA E16.5",
                "method": "STEP",
                "batch": str(section),
                "NMI": float(normalized_mutual_info_score(domains, annotations)),
                "ARI": float(adjusted_rand_score(domains, annotations)),
                "CHAOS": cal_chaos(domains, coordinates),
                "PAS": cal_pas(domains, coordinates),
                "ASW": batch_metrics["ASW"],
                "iLISI": batch_metrics["iLISI"],
            }
        )
    return pd.DataFrame(rows)


def build_result_adata(adata: ad.AnnData) -> ad.AnnData:
    """Create the compact result object used by metric and spatial plots."""
    obs_columns = [
        column
        for column in ["annotation", "batch", "x", "y", "domain"]
        if column in adata.obs.columns
    ]
    result = ad.AnnData(obs=adata.obs[obs_columns].copy())
    result.obsm["spatial"] = np.asarray(adata.obsm["spatial"]).copy()
    result.obsm["X_rep"] = np.asarray(adata.obsm["X_rep"]).copy()
    result.obsm["X_smoothed"] = np.asarray(adata.obsm["X_smoothed"]).copy()
    return result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train STEP on the five-section MOSTA E16.5 dataset."
    )
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT_H5AD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = load_mosta_input(args.input_h5ad)

    model = stModel(
        adata=adata,
        batch_key="batch",
        coord_keys=("x", "y"),
        layer_key="count",
        log_transformed=False,
        n_top_genes=None,
        geneset_to_use=adata.var_names.to_list(),
        **MODEL_CONFIG,
    )
    model.run(**TRAINING_CONFIG)
    model.cluster(n_clusters=N_CLUSTERS)

    result = build_result_adata(model.adata)
    result_path = args.output_dir / "mosta_e16_step.h5ad"
    result.write_h5ad(result_path)

    batch_metrics = compute_batch_metrics(model.adata)
    section_metrics = compute_section_metrics(model.adata, batch_metrics)
    section_metrics.to_csv(args.output_dir / "step_mosta_metrics.csv", index=False)
    write_json(args.output_dir / "step_batch_metrics.json", batch_metrics)
    write_json(
        args.output_dir / "settings.json",
        {
            "generated_at": datetime.now().isoformat(),
            "dataset": "MOSTA E16.5",
            "sections": model.adata.obs["batch"].cat.categories.tolist(),
            "input_h5ad": str(args.input_h5ad.resolve()),
            "input_shape": [int(adata.n_obs), int(adata.n_vars)],
            "expression_layer": "count",
            "model": MODEL_CONFIG,
            "training": TRAINING_CONFIG,
            "n_clusters": N_CLUSTERS,
            "outputs": {
                "result_h5ad": result_path.name,
                "section_metrics": "step_mosta_metrics.csv",
                "batch_metrics": "step_batch_metrics.json",
            },
        },
    )


if __name__ == "__main__":
    main()
