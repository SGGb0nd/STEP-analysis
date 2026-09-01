#!/usr/bin/env python
"""Run tutorial-matched spatial-domain methods on one DLPFC section."""


import argparse
import gc
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from banksy_default_kernel import (
    banksy_default_matrix,
    generate_banksy_default_weights,
)


SECTIONS = (
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
)
N_CLUSTERS = {
    **{section: 7 for section in SECTIONS[:4]},
    **{section: 5 for section in SECTIONS[4:8]},
    **{section: 7 for section in SECTIONS[8:]},
}
SUBJECT_GROUP = {
    **{section: "P1" for section in SECTIONS[:4]},
    **{section: "P2" for section in SECTIONS[4:8]},
    **{section: "P3" for section in SECTIONS[8:]},
}
METHODS = ("graphst", "stagate", "banksy")
R_LIBRARY = Path.home() / "R" / "x86_64-pc-linux-gnu-library" / "4.6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--section", choices=SECTIONS)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_section(input_root: Path, section: str) -> ad.AnnData:
    path = input_root / section / f"{section}_annotated.h5ad"
    adata = ad.read_h5ad(path)
    adata.var_names_make_unique()
    if "gd" not in adata.obs or "spatial" not in adata.obsm:
        raise ValueError(f"{path} lacks gd labels or spatial coordinates")
    return adata


def graphst_embedding(
    adata: ad.AnnData,
    *,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    from GraphST import GraphST as graphst_module

    model = graphst_module.GraphST(
        adata,
        device=device,
        epochs=600,
        random_seed=seed,
        datatype="10X",
    )
    result = model.train()
    reconstructed = np.asarray(result.obsm["emb"], dtype=np.float32)
    embedding = PCA(n_components=20, random_state=42).fit_transform(reconstructed)
    metadata = {
        "package": "GraphST",
        "package_version": importlib.metadata.version("GraphST"),
        "feature_preprocessing": (
            "3000 Seurat-v3 HVGs; normalize_total target_sum=10000; log1p; "
            "non-centered scaling capped at 10"
        ),
        "graph": "official 3-nearest-neighbor 10X graph",
        "training": "official defaults: 600 epochs, 64 latent dimensions",
        "clustering_input": "PCA20 of reconstructed expression",
    }
    del model, result, reconstructed
    return embedding.astype(np.float32), metadata


def stagate_embedding(
    adata: ad.AnnData,
    *,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    # Upstream imports torch_sparse even on its Tensor edge-index path.
    if "torch_sparse" not in sys.modules:
        torch_sparse = types.ModuleType("torch_sparse")

        class SparseTensor:
            pass

        def set_diag(_: object) -> object:
            raise TypeError("SparseTensor input is not used by this benchmark")

        torch_sparse.SparseTensor = SparseTensor
        torch_sparse.set_diag = set_diag
        sys.modules["torch_sparse"] = torch_sparse
    try:
        from STAGATE_pyG import Cal_Spatial_Net, train_STAGATE
    except ImportError as exc:
        raise RuntimeError("STAGATE_pyG and torch-geometric are required") from exc

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=3000,
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    Cal_Spatial_Net(adata, rad_cutoff=150, verbose=False)
    result = train_STAGATE(
        adata,
        hidden_dims=[512, 30],
        n_epochs=500,
        lr=0.001,
        gradient_clipping=5.0,
        weight_decay=0.0001,
        verbose=False,
        random_seed=seed,
        device=device,
    )
    embedding = np.asarray(result.obsm["STAGATE"], dtype=np.float32)
    metadata = {
        "package": "STAGATE_pyG",
        "package_version": importlib.metadata.version("STAGATE_pyG"),
        "feature_preprocessing": (
            "3000 Seurat-v3 HVGs; normalize_total target_sum=10000; log1p"
        ),
        "graph": "radius graph with rad_cutoff=150",
        "training": (
            "DLPFC tutorial settings: hidden dimensions 512/30, 500 epochs, "
            "alpha=0 equivalent reconstruction objective"
        ),
        "clustering_input": "30-dimensional STAGATE latent representation",
    }
    del result
    return embedding, metadata


def banksy_embedding(
    adata: ad.AnnData,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=2000,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    # The BANKSY DLPFC vignette uses Seurat's RC normalization without log1p.
    sc.pp.normalize_total(adata, target_sum=5000)
    expression = adata.X
    if not sp.issparse(expression):
        expression = sp.csr_matrix(np.asarray(expression, dtype=np.float32))
    else:
        expression = expression.tocsr().astype(np.float32)

    weights = generate_banksy_default_weights(
        np.asarray(adata.obsm["spatial"], dtype=float),
        num_neighbors=6,
        decay_type="scaled_gaussian",
    )
    lambda_value = 0.2
    banksy_matrix = banksy_default_matrix(
        expression,
        weights,
        lambda_value=lambda_value,
    )
    embedding = PCA(
        n_components=20,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(banksy_matrix)
    metadata = {
        "package": "pybanksy",
        "package_version": importlib.metadata.version("pybanksy"),
        "feature_preprocessing": (
            "2000 Seurat-v3 HVGs; RC normalize_total target_sum=5000; "
            "separate gene-wise z-scores for self, m=0, and m=1 AGF features"
        ),
        "graph": "scaled-Gaussian BANKSY max_m=1 kernels (6 and 12 neighbors)",
        "training": "lambda=0.2 with default first-order AGF",
        "clustering_input": "20-dimensional BANKSY PCA representation",
    }
    del banksy_matrix
    return embedding.astype(np.float32), metadata


def mclust_eee(
    embedding: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    script = r'''suppressPackageStartupMessages(library(mclust))
args <- commandArgs(trailingOnly=TRUE)
x <- as.matrix(read.csv(args[[1]], header=FALSE, check.names=FALSE))
set.seed(as.integer(args[[4]]))
fit <- Mclust(x, G=as.integer(args[[3]]), modelNames="EEE", verbose=FALSE)
write.table(fit$classification, args[[2]], row.names=FALSE, col.names=FALSE, sep=",")
'''
    with tempfile.TemporaryDirectory(prefix="dlpfc_mclust_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "embedding.csv"
        output_path = tmp_path / "labels.csv"
        script_path = tmp_path / "mclust.R"
        np.savetxt(input_path, embedding, delimiter=",", fmt="%.8g")
        script_path.write_text(script)
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(R_LIBRARY)
        subprocess.run(
            [
                "Rscript",
                str(script_path),
                str(input_path),
                str(output_path),
                str(n_clusters),
                str(seed),
            ],
            check=True,
            env=env,
        )
        labels = np.loadtxt(output_path, delimiter=",", dtype=int)
    if len(labels) != len(embedding):
        raise RuntimeError("mclust returned the wrong number of labels")
    return labels


def graphst_refine(
    labels: np.ndarray,
    coordinates: np.ndarray,
    *,
    n_neighbors: int = 50,
) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    indices = neighbors.kneighbors(coordinates, return_distance=False)[:, 1:]
    refined = []
    for row in indices:
        neighbor_labels = labels[row].tolist()
        # Match GraphST's first-occurrence tie behavior.
        refined.append(max(neighbor_labels, key=neighbor_labels.count))
    return np.asarray(refined, dtype=labels.dtype)


def cal_pas(labels: np.ndarray, coordinates: np.ndarray, k: int = 10) -> float:
    n_neighbors = min(k + 1, len(labels))
    _, indices = KDTree(coordinates).query(coordinates, k=n_neighbors)
    neighbor_labels = labels[indices[:, 1:]]
    return float(np.mean(neighbor_labels == labels[:, None]))


def cal_chaos(labels: np.ndarray, coordinates: np.ndarray) -> float:
    scaled = StandardScaler().fit_transform(coordinates)
    total = 0.0
    for cluster in np.unique(labels):
        points = scaled[labels == cluster]
        if len(points) <= 2:
            continue
        distances, _ = KDTree(points).query(points, k=2)
        total += float(distances[:, 1].sum())
    return total / len(labels)


def evaluate(
    truth: np.ndarray,
    prediction: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, float | int]:
    valid = ~pd.isna(truth)
    truth_valid = truth[valid].astype(str)
    prediction_valid = prediction[valid].astype(str)
    coordinates_valid = coordinates[valid]
    return {
        "n_eval_spots": int(valid.sum()),
        "n_reference_domains": int(np.unique(truth_valid).size),
        "n_predicted_domains": int(np.unique(prediction_valid).size),
        "ari": float(adjusted_rand_score(truth_valid, prediction_valid)),
        "nmi": float(normalized_mutual_info_score(truth_valid, prediction_valid)),
        "pas": cal_pas(prediction_valid, coordinates_valid),
        "chaos": cal_chaos(prediction_valid, coordinates_valid),
    }


def run_one(args: argparse.Namespace) -> None:
    if args.method is None or args.section is None:
        raise ValueError("--method and --section are required")
    run_dir = args.output_dir / args.section / args.method
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"Skipping completed run: {run_dir}")
        return

    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.method == "banksy":
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    adata = read_section(args.input_root, args.section)
    truth = adata.obs["gd"].to_numpy()
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)

    method_start = time.perf_counter()
    if args.method == "graphst":
        embedding, method_metadata = graphst_embedding(
            adata,
            device=device,
            seed=args.seed,
        )
    elif args.method == "stagate":
        embedding, method_metadata = stagate_embedding(
            adata,
            device=device,
            seed=args.seed,
        )
    else:
        embedding, method_metadata = banksy_embedding(adata, seed=args.seed)
    representation_seconds = time.perf_counter() - method_start

    cluster_start = time.perf_counter()
    raw_labels = mclust_eee(
        embedding,
        n_clusters=N_CLUSTERS[args.section],
        seed=2020,
    )
    predictions = {"mclust": raw_labels}
    if args.method == "graphst":
        predictions["mclust_refined"] = graphst_refine(raw_labels, coordinates)
    clustering_seconds = time.perf_counter() - cluster_start

    rows = []
    for variant, labels in predictions.items():
        row = {
            "section": args.section,
            "subject_group": SUBJECT_GROUP[args.section],
            "method": args.method,
            "variant": variant,
        }
        row.update(evaluate(truth, labels, coordinates))
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "section_metrics.csv", index=False)

    assignments = pd.DataFrame(
        {
            "ground_truth": truth,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            **{variant: labels for variant, labels in predictions.items()},
        },
        index=adata.obs_names,
    )
    assignments.to_csv(run_dir / "assignments.csv.gz", compression="gzip")
    np.savez_compressed(
        run_dir / "embedding.npz",
        obs_names=adata.obs_names.to_numpy(dtype=str),
        embedding=embedding,
    )

    result = {
        "section": args.section,
        "subject_group": SUBJECT_GROUP[args.section],
        "method": args.method,
        "source": str(
            (args.input_root / args.section / f"{args.section}_annotated.h5ad").resolve()
        ),
        "input_semantics": "raw integer counts in X",
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_clusters": N_CLUSTERS[args.section],
        "seed": args.seed,
        "device": str(device),
        "method_settings": method_metadata,
        "clustering": {
            "method": "mclust EEE",
            "seed": 2020,
            "graphst_refinement": (
                "majority label among 50 nearest spatial spots"
                if args.method == "graphst"
                else None
            ),
        },
        "metrics": rows,
        "timing_seconds": {
            "representation": representation_seconds,
            "clustering": clustering_seconds,
            "total": time.perf_counter() - total_start,
        },
        "resource": {
            "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
            "peak_cuda_reserved_gib": (
                torch.cuda.max_memory_reserved() / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        },
    }
    metrics_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

    del adata, embedding, predictions
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def summarize(output_dir: Path) -> None:
    frames = [
        pd.read_csv(path)
        for path in sorted(output_dir.glob("*/*/section_metrics.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No completed DLPFC runs under {output_dir}")
    section_metrics = pd.concat(frames, ignore_index=True)
    section_metrics.to_csv(output_dir / "dlpfc_section_metrics.csv", index=False)

    preferred = section_metrics.loc[
        (section_metrics["method"] != "graphst")
        | (section_metrics["variant"] == "mclust_refined")
    ].copy()
    summary = (
        preferred.groupby(["method", "variant"], as_index=False)
        .agg(
            n_sections=("section", "nunique"),
            ari_mean=("ari", "mean"),
            ari_sd=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_sd=("nmi", "std"),
            pas_mean=("pas", "mean"),
            chaos_mean=("chaos", "mean"),
        )
        .sort_values("ari_mean", ascending=False)
    )
    summary.to_csv(output_dir / "dlpfc_method_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        summarize(args.output_dir)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
