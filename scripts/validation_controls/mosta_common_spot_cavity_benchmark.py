#!/usr/bin/env python
"""Compare MOSTA Cavity recovery on an identical set of spots."""

import argparse
import gc
import json
import os
import random
import resource
import time
import types
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from advanced_graph_training_modes import (
    SectionGraph,
    SparseGraphST,
    graphst_components,
    stagate_components,
    train_sampled,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "workflows/mosta_cavity_controls/method_comparison"
STEP_RESULT = REPO_ROOT / "results/mosta_e16_step/mosta_e16_step.h5ad"
NICHE_RESULT = REPO_ROOT / "external_results/mosta/nichecompass.h5ad"
BANKSY_EMBEDDING = REPO_ROOT / "external_results/mosta/banksy_harmony.csv"
RAW_DATA = REPO_ROOT / "data/mosta_e16_test.h5ad"
CAVITY = "Cavity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step-result", type=Path, default=STEP_RESULT)
    parser.add_argument("--nichecompass-result", type=Path, default=NICHE_RESULT)
    parser.add_argument("--banksy-embedding", type=Path, default=BANKSY_EMBEDDING)
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA)
    parser.add_argument(
        "--action",
        choices=("prepare", "train", "train-sampled", "evaluate"),
        required=True,
    )
    parser.add_argument("--method", choices=("graphst", "stagate"))
    parser.add_argument(
        "--section",
        help="MOSTA section name, with or without the .MOSTA suffix",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--sampled-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def canonical_key(sample: object, barcode: object) -> str:
    sample_value = str(sample).replace(".MOSTA", "")
    barcode_value = str(barcode)
    if "__" in barcode_value:
        embedded_sample, barcode_value = barcode_value.split("__", 1)
        if embedded_sample != sample_value:
            raise ValueError(
                f"Sample mismatch in spot name: {embedded_sample} != {sample_value}"
            )
    prefix = f"{sample_value}.MOSTA-"
    if barcode_value.startswith(prefix):
        barcode_value = barcode_value[len(prefix) :]
    return f"{sample_value}__{barcode_value}"


def indexed_keys(samples: pd.Series, barcodes: pd.Index) -> pd.Index:
    return pd.Index(
        [canonical_key(sample, barcode) for sample, barcode in zip(samples, barcodes)]
    )


def ensure_unique(keys: pd.Index, source: str) -> None:
    if not keys.is_unique:
        duplicated = keys[keys.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Duplicate canonical keys in {source}: {duplicated}")


def load_banksy_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["key"] = [
        canonical_key(sample, barcode)
        for sample, barcode in zip(frame["sample"], frame["barcode"])
    ]
    if frame["key"].duplicated().any():
        raise ValueError("BANKSY embedding contains duplicate canonical spot keys")
    return frame.set_index("key", verify_integrity=True)


def common_spot_sources(
    step_result: Path,
    niche_result: Path,
    raw_data: Path,
    banksy_embedding: Path,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, pd.DataFrame, pd.Index]:
    step = ad.read_h5ad(step_result, backed="r")
    niche = ad.read_h5ad(niche_result, backed="r")
    raw = ad.read_h5ad(raw_data, backed="r")

    step_keys = indexed_keys(step.obs["batch"], step.obs_names)
    niche_keys = indexed_keys(niche.obs["batch"], niche.obs_names)
    raw_keys = indexed_keys(raw.obs["batch"], raw.obs_names)
    ensure_unique(step_keys, "STEP")
    ensure_unique(niche_keys, "NicheCompass")
    ensure_unique(raw_keys, "raw MOSTA")

    banksy = load_banksy_frame(banksy_embedding)
    common = (
        step_keys.intersection(niche_keys, sort=False)
        .intersection(banksy.index, sort=False)
        .intersection(raw_keys, sort=False)
    )
    if len(common) == 0:
        raise RuntimeError("No common MOSTA spots were found")

    step.obs["_canonical_key"] = step_keys.to_numpy()
    niche.obs["_canonical_key"] = niche_keys.to_numpy()
    raw.obs["_canonical_key"] = raw_keys.to_numpy()
    return step, niche, raw, banksy, common


def annotation_by_key(adata: ad.AnnData) -> pd.Series:
    return pd.Series(
        adata.obs["annotation"].astype(str).to_numpy(),
        index=adata.obs["_canonical_key"].astype(str),
    )


def prepare(
    output_dir: Path,
    step_result: Path,
    niche_result: Path,
    raw_data: Path,
    banksy_embedding: Path,
    requested_section: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    step, niche, raw, banksy, common = common_spot_sources(
        step_result,
        niche_result,
        raw_data,
        banksy_embedding,
    )
    try:
        step_annotation = annotation_by_key(step).loc[common]
        niche_annotation = annotation_by_key(niche).loc[common]
        raw_annotation = annotation_by_key(raw).loc[common]
        if not step_annotation.equals(niche_annotation):
            raise AssertionError("STEP and NicheCompass annotations differ on common spots")
        if not step_annotation.equals(raw_annotation):
            raise AssertionError("STEP and raw annotations differ on common spots")

        samples = pd.Index(common.str.split("__", n=1).str[0], name="sample")
        overview = pd.DataFrame(
            {
                "key": common,
                "sample": samples,
                "annotation": step_annotation.to_numpy(),
            }
        )
        section_counts = (
            overview.groupby("sample", observed=True)
            .agg(
                n_spots=("key", "size"),
                n_cavity=("annotation", lambda values: int((values == CAVITY).sum())),
            )
            .sort_values("n_cavity", ascending=False)
        )
        section = (
            requested_section.replace(".MOSTA", "")
            if requested_section is not None
            else str(section_counts.index[0])
        )
        if section not in section_counts.index:
            raise ValueError(
                f"Section {section!r} is unavailable; choose from "
                f"{section_counts.index.tolist()}"
            )
        selected_keys = common[samples == section]

        step_index = pd.Series(
            np.arange(step.n_obs), index=step.obs["_canonical_key"].astype(str)
        ).loc[selected_keys].to_numpy()
        niche_index = pd.Series(
            np.arange(niche.n_obs), index=niche.obs["_canonical_key"].astype(str)
        ).loc[selected_keys].to_numpy()
        raw_index = pd.Series(
            np.arange(raw.n_obs), index=raw.obs["_canonical_key"].astype(str)
        ).loc[selected_keys].to_numpy()

        step_embedding = np.asarray(step.obsm["X_smoothed"][step_index], dtype=np.float32)
        niche_embedding = np.asarray(
            niche.obsm["nichecompass_latent"][niche_index], dtype=np.float32
        )
        banksy_columns = [column for column in banksy.columns if column.startswith("PC")]
        banksy_embedding = banksy.loc[selected_keys, banksy_columns].to_numpy(
            dtype=np.float32
        )
        np.save(output_dir / "step_sampled_embedding.npy", step_embedding)
        np.save(output_dir / "nichecompass_embedding.npy", niche_embedding)
        np.save(output_dir / "banksy_harmony_embedding.npy", banksy_embedding)

        counts = raw.layers["count"][raw_index]
        if not sp.issparse(counts):
            counts = sp.csr_matrix(np.asarray(counts))
        else:
            counts = counts.tocsr()
        if not np.allclose(counts.data, np.rint(counts.data)):
            raise AssertionError("raw count layer contains non-integer values")

        coordinates = raw.obs.iloc[raw_index][["x", "y"]].to_numpy(dtype=np.float32)
        selected_annotation = raw.obs.iloc[raw_index]["annotation"].astype(str).to_numpy()
        prepared = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(
                {
                    "key": selected_keys.to_numpy(),
                    "sample": section,
                    "annotation": selected_annotation,
                    "x": coordinates[:, 0],
                    "y": coordinates[:, 1],
                },
                index=pd.Index(selected_keys, name="spot"),
            ),
            var=raw.var.copy(),
        )
        prepared.obsm["spatial"] = coordinates
        prepared.write_h5ad(output_dir / "common_spots_raw_counts.h5ad", compression="gzip")
        prepared.obs.reset_index().to_csv(
            output_dir / "common_spot_manifest.csv.gz", index=False, compression="gzip"
        )
        section_counts.to_csv(output_dir / "section_common_spot_counts.csv")

        summary = {
            "n_common_spots_all_sections": int(len(common)),
            "selected_section": section,
            "n_selected_spots": int(len(selected_keys)),
            "n_selected_cavity": int((selected_annotation == CAVITY).sum()),
            "n_reference_annotations": int(pd.Series(selected_annotation).nunique()),
            "spot_key": "sample + original spot name",
            "count_source": f"{raw_data}:layer=count",
            "section_counts": section_counts.reset_index().to_dict(orient="records"),
        }
        (output_dir / "prepare_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        step.file.close()
        niche.file.close()
        raw.file.close()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_normalize(counts: sp.csr_matrix) -> sp.csr_matrix:
    matrix = counts.astype(np.float32, copy=True)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    factors = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = sp.diags(factors) @ matrix
    matrix.data = np.log1p(matrix.data)
    return matrix.tocsr()


def graphst_scale(matrix: sp.csr_matrix) -> sp.csr_matrix:
    n_obs = matrix.shape[0]
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean**2, 0)
    if n_obs > 1:
        variance *= n_obs / (n_obs - 1)
    std = np.sqrt(variance)
    inverse = np.divide(1.0, std, out=np.zeros_like(std), where=std > 0)
    scaled = matrix.multiply(inverse).tocsr()
    np.minimum(scaled.data, 10.0, out=scaled.data)
    return scaled


def spatial_operators(
    coordinates: np.ndarray,
    n_neighbors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neighbor_index = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(
        coordinates
    ).kneighbors(return_distance=False)[:, 1:]
    n_obs = len(coordinates)
    row = np.repeat(np.arange(n_obs), n_neighbors)
    col = neighbor_index.reshape(-1)

    directed = coo_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)),
        shape=(n_obs, n_obs),
    ).tocsr()
    context = directed + sp.eye(n_obs, dtype=np.float32, format="csr")
    context = sp.diags(1.0 / np.asarray(context.sum(axis=1)).ravel()) @ context

    adjacency = directed.maximum(directed.T)
    adjacency = adjacency + sp.eye(n_obs, dtype=np.float32, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    adjacency = sp.diags(np.power(degree, -0.5)) @ adjacency @ sp.diags(
        np.power(degree, -0.5)
    )

    def scipy_to_torch(matrix: sp.spmatrix) -> torch.Tensor:
        value = matrix.tocoo()
        indices = torch.from_numpy(np.vstack((value.row, value.col))).long()
        data = torch.from_numpy(value.data.astype(np.float32, copy=False))
        return torch.sparse_coo_tensor(
            indices,
            data,
            size=value.shape,
            device=device,
        ).coalesce()

    edge_row = np.concatenate((row, np.arange(n_obs)))
    edge_col = np.concatenate((col, np.arange(n_obs)))
    edge_index = torch.from_numpy(np.vstack((edge_row, edge_col))).long().to(device)
    return scipy_to_torch(adjacency), scipy_to_torch(context), edge_index


def sampled_section_graph(
    coordinates: np.ndarray,
    n_neighbors: int,
    method: str,
) -> SectionGraph:
    """Build the same full spatial graph used to draw GraphSAINT subgraphs."""
    import dgl

    neighbor_index = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(
        coordinates
    ).kneighbors(return_distance=False)[:, 1:]
    n_obs = len(coordinates)
    row = np.repeat(np.arange(n_obs), n_neighbors)
    col = neighbor_index.reshape(-1)
    directed = coo_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)),
        shape=(n_obs, n_obs),
    ).tocsr()
    if method == "graphst":
        matrix = directed.maximum(directed.T)
        matrix = matrix + sp.eye(n_obs, dtype=np.float32, format="csr")
        coo = matrix.tocoo()
        src, dst = coo.col, coo.row
    else:
        matrix = directed + sp.eye(n_obs, dtype=np.float32, format="csr")
        coo = matrix.tocoo()
        src, dst = coo.row, coo.col
    graph = dgl.graph((src, dst), num_nodes=n_obs)
    graph.ndata["node_ids"] = torch.arange(n_obs)
    return SectionGraph(
        name="MOSTA section",
        global_indices=np.arange(n_obs),
        graph=graph,
    )


def install_stagate_module() -> type[torch.nn.Module]:
    if "torch_sparse" not in __import__("sys").modules:
        import sys

        torch_sparse = types.ModuleType("torch_sparse")

        class SparseTensor:
            pass

        def set_diag(_: object) -> object:
            raise TypeError("SparseTensor input is not used in this benchmark")

        torch_sparse.SparseTensor = SparseTensor
        torch_sparse.set_diag = set_diag
        sys.modules["torch_sparse"] = torch_sparse
    from STAGATE_pyG.STAGATE import STAGATE

    return STAGATE


def restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not path.exists():
        return 0
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(payload["cuda_rng"].cpu(), device=device)
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return int(payload["completed_epochs"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    elapsed_seconds: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epochs": completed_epochs,
            "elapsed_seconds": elapsed_seconds,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        path,
    )


def train_sampled_method(args: argparse.Namespace) -> None:
    if args.method is None:
        raise ValueError("--method is required for --action train-sampled")
    method_dir = args.output_dir / f"{args.method}_sampled"
    method_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = method_dir / "embedding.npy"
    summary_path = method_dir / "training_summary.json"
    if embedding_path.exists() and summary_path.exists():
        print(f"Skipping completed sampled {args.method} run", flush=True)
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    adata = ad.read_h5ad(args.output_dir / "common_spots_raw_counts.h5ad")
    counts = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    normalized = log_normalize(counts)
    matrix = graphst_scale(normalized) if args.method == "graphst" else normalized
    features = torch.from_numpy(matrix.toarray()).to(device=device, dtype=torch.float32)
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    section_graph = sampled_section_graph(
        coordinates,
        args.n_neighbors,
        args.method,
    )

    if args.method == "graphst":
        model, loss_fn = graphst_components(features.shape[1], device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        gradient_clip = None
    else:
        model, loss_fn = stagate_components(features.shape[1], device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=0.0001,
        )
        gradient_clip = 5.0

    train_start = time.perf_counter()
    actual_node_inclusions = train_sampled(
        model,
        loss_fn,
        features=features,
        section_graphs=[section_graph],
        sample_size=args.sample_size,
        iterations=args.sampled_iterations,
        optimizer=optimizer,
        gradient_clip=gradient_clip,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_start

    adjacency, neighbor_mean, edge_index = spatial_operators(
        coordinates,
        args.n_neighbors,
        device,
    )
    model.eval()
    with torch.no_grad():
        if args.method == "graphst":
            _, reconstruction, _, _ = model(
                features,
                features,
                adjacency,
                neighbor_mean,
            )
            reconstructed = reconstruction.detach().cpu().numpy().astype(np.float32)
            embedding = PCA(
                n_components=20,
                svd_solver="randomized",
                random_state=args.seed,
            ).fit_transform(reconstructed).astype(np.float32)
            del reconstructed
        else:
            embedding, _ = model(features, edge_index)
            embedding = embedding.detach().cpu().numpy().astype(np.float32)
    np.save(embedding_path, embedding)

    summary = {
        "method": args.method,
        "training_mode": "GraphSAINT node-sampled induced subgraphs",
        "inference_mode": "full section spatial graph",
        "sample_size": args.sample_size,
        "optimizer_updates": args.sampled_iterations,
        "actual_node_inclusions": actual_node_inclusions,
        "mean_node_inclusions": actual_node_inclusions / adata.n_obs,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_neighbors": args.n_neighbors,
        "input": "raw count layer; all 2000 supplied genes",
        "preprocessing": (
            "library-size normalization to 10000, log1p, non-centered gene scaling capped at 10"
            if args.method == "graphst"
            else "library-size normalization to 10000 and log1p"
        ),
        "embedding": (
            "PCA20 of GraphST reconstructed expression"
            if args.method == "graphst"
            else "30-dimensional STAGATE latent representation"
        ),
        "training_seconds": training_seconds,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        ),
        "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**20,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    del model, features, matrix, normalized, counts, adata
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train(args: argparse.Namespace) -> None:
    if args.method is None:
        raise ValueError("--method is required for --action train")
    default_epochs = 600 if args.method == "graphst" else 500
    target_epochs = args.epochs or default_epochs
    method_dir = args.output_dir / args.method
    method_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = method_dir / "embedding.npy"
    summary_path = method_dir / "training_summary.json"
    if embedding_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if int(summary["completed_epochs"]) >= target_epochs:
            print(f"Skipping completed {args.method} run", flush=True)
            return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    adata = ad.read_h5ad(args.output_dir / "common_spots_raw_counts.h5ad")
    counts = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    normalized = log_normalize(counts)
    matrix = graphst_scale(normalized) if args.method == "graphst" else normalized
    features = torch.from_numpy(matrix.toarray()).to(device=device, dtype=torch.float32)
    adjacency, neighbor_mean, edge_index = spatial_operators(
        np.asarray(adata.obsm["spatial"], dtype=np.float32),
        args.n_neighbors,
        device,
    )

    if args.method == "graphst":
        model = SparseGraphST(input_dim=features.shape[1], output_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    else:
        STAGATE = install_stagate_module()
        model = STAGATE(hidden_dims=[features.shape[1], 512, 30]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    checkpoint = method_dir / "training_checkpoint.pt"
    completed = restore_checkpoint(checkpoint, model, optimizer, device)
    start = time.perf_counter()
    model.train()
    for epoch in range(completed, target_epochs):
        optimizer.zero_grad(set_to_none=True)
        if args.method == "graphst":
            corrupted = features[torch.randperm(features.shape[0], device=device)]
            _, reconstruction, logits, corrupted_logits = model(
                features,
                corrupted,
                adjacency,
                neighbor_mean,
            )
            labels = torch.stack(
                (
                    torch.ones(features.shape[0], device=device),
                    torch.zeros(features.shape[0], device=device),
                ),
                dim=1,
            )
            loss = 10.0 * F.mse_loss(reconstruction, features)
            loss = loss + F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + F.binary_cross_entropy_with_logits(corrupted_logits, labels)
        else:
            _, reconstruction = model(features, edge_index)
            loss = F.mse_loss(reconstruction, features)
        loss.backward()
        if args.method == "stagate":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        completed = epoch + 1
        if completed == 1 or completed % 10 == 0:
            print(
                json.dumps(
                    {
                        "method": args.method,
                        "epoch": completed,
                        "target": target_epochs,
                        "loss": float(loss.detach().cpu()),
                    }
                ),
                flush=True,
            )
        if completed % args.checkpoint_every == 0 or completed == target_epochs:
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                completed,
                time.perf_counter() - start,
            )

    model.eval()
    with torch.no_grad():
        if args.method == "graphst":
            _, reconstruction, _, _ = model(features, features, adjacency, neighbor_mean)
            reconstructed = reconstruction.detach().cpu().numpy().astype(np.float32)
            embedding = PCA(
                n_components=20,
                svd_solver="randomized",
                random_state=args.seed,
            ).fit_transform(reconstructed).astype(np.float32)
            del reconstructed
        else:
            embedding, _ = model(features, edge_index)
            embedding = embedding.detach().cpu().numpy().astype(np.float32)
    np.save(embedding_path, embedding)

    summary = {
        "method": args.method,
        "completed_epochs": completed,
        "default_epochs": default_epochs,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_neighbors": args.n_neighbors,
        "graph": "full section-wise 3-nearest-neighbor graph",
        "input": "raw count layer; all 2000 supplied genes",
        "preprocessing": (
            "library-size normalization to 10000, log1p, non-centered gene scaling capped at 10"
            if args.method == "graphst"
            else "library-size normalization to 10000 and log1p"
        ),
        "embedding": (
            "PCA20 of GraphST reconstructed expression"
            if args.method == "graphst"
            else "30-dimensional STAGATE latent representation"
        ),
        "training_seconds_this_process": time.perf_counter() - start,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        ),
        "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**20,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    del model, features, matrix, normalized, counts, adata
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def largest_component_fraction(adjacency: sp.csr_matrix, mask: np.ndarray) -> float:
    index = np.flatnonzero(mask)
    if len(index) == 0:
        return 0.0
    _, labels = connected_components(adjacency[index][:, index], directed=False)
    return float(np.bincount(labels).max() / len(index))


def best_cavity_cluster(
    clusters: np.ndarray,
    truth: np.ndarray,
    adjacency: sp.csr_matrix,
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame]:
    cavity = truth == CAVITY
    rows = []
    best: tuple[float, float, float, str, np.ndarray] | None = None
    for cluster in np.unique(clusters):
        predicted = clusters == cluster
        true_positive = int(np.logical_and(predicted, cavity).sum())
        false_positive = int(np.logical_and(predicted, ~cavity).sum())
        false_negative = int(np.logical_and(~predicted, cavity).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
        iou = true_positive / max(np.logical_or(predicted, cavity).sum(), 1)
        rows.append(
            {
                "cluster": str(cluster),
                "n_cluster": int(predicted.sum()),
                "n_cavity": true_positive,
                "cavity_fraction": precision,
                "cavity_recall": recall,
                "cavity_f1": f1,
                "cavity_iou": iou,
            }
        )
        candidate = (f1, iou, precision, str(cluster), predicted)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    f1, iou, precision, cluster, predicted = best
    recall = int(np.logical_and(predicted, cavity).sum()) / max(int(cavity.sum()), 1)
    return (
        {
            "cavity_cluster": cluster,
            "cavity_precision": precision,
            "cavity_recall": recall,
            "cavity_f1": f1,
            "cavity_iou": iou,
            "cavity_lcc_fraction": largest_component_fraction(adjacency, predicted),
            "n_predicted_cavity": int(predicted.sum()),
        },
        predicted,
        pd.DataFrame(rows),
    )


def evaluate(args: argparse.Namespace) -> None:
    prepared = ad.read_h5ad(args.output_dir / "common_spots_raw_counts.h5ad", backed="r")
    truth = prepared.obs["annotation"].astype(str).to_numpy()
    coordinates = np.asarray(prepared.obsm["spatial"], dtype=float)
    n_clusters = int(pd.Series(truth).nunique())
    radius_neighbors = NearestNeighbors(radius=1.5).fit(coordinates)
    graph = radius_neighbors.radius_neighbors_graph(coordinates, mode="connectivity")
    graph.setdiag(0)
    graph.eliminate_zeros()

    embedding_paths = {
        "STEP sampled": args.output_dir / "step_sampled_embedding.npy",
        "NicheCompass": args.output_dir / "nichecompass_embedding.npy",
        "BANKSY (Harmony)": args.output_dir / "banksy_harmony_embedding.npy",
        "GraphST full graph": args.output_dir / "graphst/embedding.npy",
        "STAGATE full graph": args.output_dir / "stagate/embedding.npy",
    }
    sampled_paths = {
        "GraphST sampled": args.output_dir / "graphst_sampled/embedding.npy",
        "STAGATE sampled": args.output_dir / "stagate_sampled/embedding.npy",
    }
    sampled_present = [path.exists() for path in sampled_paths.values()]
    if any(sampled_present) and not all(sampled_present):
        raise FileNotFoundError("Only one sampled-method embedding is available")
    if all(sampled_present):
        embedding_paths.update(sampled_paths)
    methods = tuple(embedding_paths)
    missing = [str(path) for path in embedding_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing method embeddings: {missing}")

    metric_rows = []
    contingency_rows = []
    predictions = {}
    assignments = prepared.obs[["key", "sample", "annotation", "x", "y"]].copy()
    for method in methods:
        embedding = np.load(embedding_paths[method], mmap_mode="r")
        if len(embedding) != len(truth):
            raise ValueError(f"{method} embedding has {len(embedding)} rows")
        clusters = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=args.seed,
            n_init=10,
            batch_size=4096,
        ).fit_predict(embedding)
        cavity_values, predicted, contingency = best_cavity_cluster(
            clusters.astype(str), truth, graph.tocsr()
        )
        metric_rows.append(
            {
                "method": method,
                "n_spots": len(truth),
                "n_clusters": n_clusters,
                "ARI": adjusted_rand_score(truth, clusters),
                "NMI": normalized_mutual_info_score(truth, clusters),
                **cavity_values,
            }
        )
        contingency.insert(0, "method", method)
        contingency_rows.append(contingency)
        predictions[method] = predicted
        assignments[f"cluster_{method}"] = clusters.astype(str)

    output_suffix = "_all_training_modes" if all(sampled_present) else ""
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        args.output_dir / f"common_spot_metrics{output_suffix}.csv",
        index=False,
    )
    pd.concat(contingency_rows, ignore_index=True).to_csv(
        args.output_dir / f"cavity_cluster_contingency{output_suffix}.csv",
        index=False,
    )
    assignments.to_csv(
        args.output_dir / f"common_spot_assignments{output_suffix}.csv.gz",
        index=False,
        compression="gzip",
    )

    panel_names = ("Ground truth", *methods)
    n_columns = 4 if len(panel_names) > 6 else 3
    n_rows = int(np.ceil(len(panel_names) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4.7 * n_columns, 4.5 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    cavity_color = "#D95F02"
    for axis, name in zip(axes, panel_names):
        mask = truth == CAVITY if name == "Ground truth" else predictions[name]
        axis.scatter(
            coordinates[~mask, 0],
            -coordinates[~mask, 1],
            c="#D9D9D9",
            s=0.18,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            coordinates[mask, 0],
            -coordinates[mask, 1],
            c=cavity_color,
            s=0.35,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.axis("off")
    for axis in axes[len(panel_names) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(
        args.output_dir / f"common_spot_cavity_comparison{output_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        args.output_dir / f"common_spot_cavity_comparison{output_suffix}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    prepared.file.close()

    section = str(prepared.obs["sample"].iloc[0])
    legend = (
        "**Common-spot MOSTA Cavity comparison.** Ground-truth Cavity annotation "
        "and the single cluster with the highest Cavity F1 score for each method "
        f"in section {section}. All methods were evaluated on the same spots and "
        "reclustered into the same number of domains. Orange indicates Cavity or "
        "the matched cluster; gray indicates all remaining spots.\n"
    )
    (args.output_dir / f"figure_legend{output_suffix}.md").write_text(
        legend,
        encoding="utf-8",
    )
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        prepare(
            args.output_dir,
            args.step_result,
            args.nichecompass_result,
            args.raw_data,
            args.banksy_embedding,
            args.section,
        )
    elif args.action == "train":
        train(args)
    elif args.action == "train-sampled":
        train_sampled_method(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
