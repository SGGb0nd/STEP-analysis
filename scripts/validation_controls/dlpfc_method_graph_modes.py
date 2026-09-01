#!/usr/bin/env python
"""Compare method graph-training and graph-inference controls on DLPFC."""


import argparse
import gc
import json
import math
import resource
import time
from pathlib import Path

import anndata as ad
import dgl
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from step.utils.misc import generate_adj
from advanced_graph_training_modes import (
    SectionGraph,
    graphst_components,
    infer_embeddings,
    make_sample_loader,
    stagate_components,
    train_full,
    train_graphst_fullneighbors,
)
from banksy_default_kernel import (
    banksy_default_matrix,
    generate_banksy_default_weights,
    restrict_banksy_weights_to_labels,
)
from dlpfc_tutorial_methods import (
    N_CLUSTERS,
    SECTIONS,
    evaluate,
    graphst_refine,
    mclust_eee,
    set_seed,
)


METHODS = ("graphst", "stagate", "banksy")
TRAINING_MODES = ("whole", "fullneighbors", "sampled", "fixed")
GRAPH_MODES = ("spatial", "annotation")
DEFAULT_EPOCHS = {"graphst": 600, "stagate": 500}
DEFAULT_NEIGHBORS = {"graphst": 3, "banksy": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--training-mode", choices=TRAINING_MODES)
    parser.add_argument("--train-graph", choices=GRAPH_MODES, default="spatial")
    parser.add_argument("--infer-graph", choices=GRAPH_MODES, default="spatial")
    parser.add_argument("--section", choices=SECTIONS, default="151673")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--fullneighbor-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-model-cache",
        action="store_true",
        help="Reuse a training cache while forcing regeneration of a result.",
    )
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def read_section(input_root: Path, section: str) -> ad.AnnData:
    path = input_root / section / f"{section}_annotated.h5ad"
    adata = ad.read_h5ad(path)
    adata.var_names_make_unique()
    if "gd" not in adata.obs or "spatial" not in adata.obsm:
        raise ValueError(f"{path} lacks gd labels or spatial coordinates")
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    adata.obs["array_row"] = coordinates[:, 0]
    adata.obs["array_col"] = coordinates[:, 1]
    return adata


def preprocess(adata: ad.AnnData, method: str) -> ad.AnnData:
    result = adata.copy()
    if method == "banksy":
        sc.pp.highly_variable_genes(result, flavor="seurat_v3", n_top_genes=2000)
        result = result[:, result.var["highly_variable"]].copy()
        sc.pp.normalize_total(result, target_sum=5000)
        return result

    sc.pp.highly_variable_genes(result, flavor="seurat_v3", n_top_genes=3000)
    result = result[:, result.var["highly_variable"]].copy()
    sc.pp.normalize_total(result, target_sum=1e4)
    sc.pp.log1p(result)
    return result


def graphst_features(adata: ad.AnnData) -> sp.csr_matrix:
    matrix = adata.X
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    else:
        matrix = matrix.tocsr().astype(np.float32)
    n_obs = matrix.shape[0]
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second_moment = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second_moment - mean**2, 0)
    if n_obs > 1:
        variance *= n_obs / (n_obs - 1)
    std = np.sqrt(variance)
    inv_std = np.zeros_like(std)
    inv_std[std > 0] = 1.0 / std[std > 0]
    matrix = matrix.multiply(inv_std).tocsr()
    np.minimum(matrix.data, 10.0, out=matrix.data)
    return matrix


def filter_graph_by_annotation(
    graph: dgl.DGLGraph,
    labels: np.ndarray,
) -> dgl.DGLGraph:
    src, dst = graph.edges()
    source = src.numpy()
    target = dst.numpy()
    keep = labels[source] == labels[target]
    filtered = dgl.graph(
        (src[keep], dst[keep]),
        num_nodes=graph.num_nodes(),
        idtype=graph.idtype,
    )
    filtered.ndata["xy"] = graph.ndata["xy"].clone()
    return dgl.add_self_loop(dgl.remove_self_loop(filtered))


def build_base_graph(adata: ad.AnnData, method: str) -> dgl.DGLGraph:
    if method == "graphst":
        return generate_adj(adata, edge_clip=None, max_neighbors=3)
    if method == "stagate":
        return generate_adj(adata, edge_clip=150, max_neighbors=6)
    raise ValueError(f"No DGL graph is defined for {method}")


def build_graph(
    adata: ad.AnnData,
    method: str,
    graph_mode: str,
) -> dgl.DGLGraph:
    graph = build_base_graph(adata, method)
    if graph_mode == "annotation":
        graph = filter_graph_by_annotation(
            graph,
            adata.obs["gd"].astype(str).to_numpy(),
        )
    graph.ndata["node_ids"] = torch.arange(adata.n_obs)
    return graph


def k_hop_input_nodes(
    graph: dgl.DGLGraph,
    targets: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    nodes = targets.unique()
    frontier = nodes
    for _ in range(depth):
        sources, _ = graph.in_edges(frontier)
        expanded = torch.unique(torch.cat((nodes, sources)))
        if len(expanded) == len(nodes):
            break
        frontier_mask = ~torch.isin(expanded, nodes)
        frontier = expanded[frontier_mask]
        nodes = expanded
    return nodes.sort().values


def prepare_stagate_fullneighbor_batches(
    graph: dgl.DGLGraph,
    batch_size: int,
    seed: int,
) -> list[tuple[dgl.DGLGraph, torch.Tensor, int]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(graph.num_nodes(), generator=generator)
    batches = []
    for start in range(0, graph.num_nodes(), batch_size):
        targets = order[start : start + batch_size]
        input_nodes = k_hop_input_nodes(graph, targets, depth=4)
        subgraph = dgl.node_subgraph(graph, input_nodes)
        global_nodes = subgraph.ndata[dgl.NID]
        inverse = torch.empty(graph.num_nodes(), dtype=torch.long)
        inverse[global_nodes] = torch.arange(len(global_nodes))
        target_local = inverse[targets]
        batches.append((subgraph, target_local, len(targets)))
    return batches


def train_stagate_fullneighbors(
    model: torch.nn.Module,
    *,
    features: torch.Tensor,
    graph: dgl.DGLGraph,
    batch_size: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
) -> int:
    """Accumulate exact full-neighbor gradients before each full-graph update."""
    batches = prepare_stagate_fullneighbor_batches(graph, batch_size, seed)
    node_exposures = 0
    total_nodes = graph.num_nodes()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for subgraph, target_local, target_count in batches:
            global_nodes = subgraph.ndata[dgl.NID]
            src, dst = subgraph.edges()
            edge_index = torch.stack((src, dst)).to(device)
            batch_features = features[global_nodes.to(device)]
            _, reconstruction = model(batch_features, edge_index)
            target_local = target_local.to(device)
            loss = F.mse_loss(
                reconstruction[target_local],
                batch_features[target_local],
            )
            (target_count / total_nodes * loss).backward()
            node_exposures += target_count
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return node_exposures


def train_sampled_to_exposure(
    model: torch.nn.Module,
    loss_fn: object,
    *,
    features: torch.Tensor,
    section_graphs: list[SectionGraph],
    sample_size: int,
    target_exposures: int,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float | None,
    device: torch.device,
) -> tuple[int, int]:
    # SAINT's budget is not its exact returned node count, so stop on observed
    # inclusions instead of converting exposure to updates nominally.
    upper_updates = math.ceil(target_exposures / sample_size * 1.5) + 10
    loader = make_sample_loader(
        section_graphs,
        sample_size=sample_size,
        iterations=upper_updates,
    )
    node_exposures = 0
    updates = 0
    model.train()
    for subgraph in loader:
        node_ids = subgraph.ndata["node_ids"].to(device)
        subgraph = subgraph.to("cpu")
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, features[node_ids], subgraph)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        updates += 1
        node_exposures += int(node_ids.numel())
        if node_exposures >= target_exposures:
            return node_exposures, updates
    raise RuntimeError(
        f"SAINT loader ended at {node_exposures} exposures before "
        f"the target {target_exposures}"
    )


def banksy_embedding(
    adata: ad.AnnData,
    *,
    graph_mode: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    expression = adata.X
    if not sp.issparse(expression):
        expression = sp.csr_matrix(np.asarray(expression, dtype=np.float32))
    else:
        expression = expression.tocsr().astype(np.float32)
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
    weights = generate_banksy_default_weights(
        coordinates,
        num_neighbors=DEFAULT_NEIGHBORS["banksy"],
        decay_type="scaled_gaussian",
    )
    if graph_mode == "annotation":
        weights = restrict_banksy_weights_to_labels(
            weights,
            adata.obs["gd"].astype(str).to_numpy(),
        )
    matrix = banksy_default_matrix(
        expression,
        weights,
        lambda_value=0.2,
    )
    embedding = PCA(
        n_components=20,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(matrix)
    return embedding.astype(np.float32), sum(weights.edge_counts.values())


def train_graph_method(
    args: argparse.Namespace,
    adata: ad.AnnData,
    device: torch.device,
    cache_path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    matrix = graphst_features(adata) if args.method == "graphst" else adata.X
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    else:
        matrix = matrix.tocsr().astype(np.float32)
    features = torch.from_numpy(matrix.toarray()).to(device)

    train_graph = build_graph(adata, args.method, args.train_graph)
    infer_graph = build_graph(adata, args.method, args.infer_graph)
    train_section = [
        SectionGraph(
            name=args.section,
            global_indices=np.arange(adata.n_obs),
            graph=train_graph,
        )
    ]
    infer_section = [
        SectionGraph(
            name=args.section,
            global_indices=np.arange(adata.n_obs),
            graph=infer_graph,
        )
    ]

    if args.method == "graphst":
        model, loss_fn = graphst_components(features.shape[1], device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        gradient_clip = None
    else:
        model, loss_fn = stagate_components(features.shape[1], device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=0.0001,
        )
        gradient_clip = 5.0

    epochs = DEFAULT_EPOCHS[args.method]
    target_exposures = epochs * adata.n_obs
    cache_reused = cache_path.exists() and (
        not args.force or args.reuse_model_cache
    )
    if cache_reused:
        cached = torch.load(cache_path, map_location=device, weights_only=False)
        model.load_state_dict(cached["state_dict"])
        cached_training = cached["training"]
        updates = int(cached_training["optimizer_updates"])
        actual_exposures = int(cached_training["actual_node_exposures"])
        training_seconds = float(cached_training["training_seconds"])
        training_seconds_this_run = 0.0
    else:
        training_start = time.perf_counter()
        if args.training_mode == "whole":
            updates = epochs
            actual_exposures = train_full(
                model,
                loss_fn,
                features=features,
                section_graphs=train_section,
                epochs=epochs,
                optimizer=optimizer,
                gradient_clip=gradient_clip,
                device=device,
            )
        elif args.training_mode == "sampled":
            actual_exposures, updates = train_sampled_to_exposure(
                model,
                loss_fn,
                features=features,
                section_graphs=train_section,
                sample_size=args.sample_size,
                target_exposures=target_exposures,
                optimizer=optimizer,
                gradient_clip=gradient_clip,
                device=device,
            )
        elif args.training_mode == "fullneighbors":
            updates = epochs
            if args.method == "graphst":
                actual_exposures = train_graphst_fullneighbors(
                    model,
                    features=features,
                    section_graphs=train_section,
                    batch_size=args.fullneighbor_batch_size,
                    epochs=epochs,
                    optimizer=optimizer,
                    device=device,
                )
            else:
                actual_exposures = train_stagate_fullneighbors(
                    model,
                    features=features,
                    graph=train_graph,
                    batch_size=args.fullneighbor_batch_size,
                    epochs=epochs,
                    optimizer=optimizer,
                    device=device,
                    seed=args.seed,
                )
        else:
            raise ValueError("fixed training mode is only valid for BANKSY")
        torch.cuda.synchronize() if device.type == "cuda" else None
        training_seconds = time.perf_counter() - training_start
        training_seconds_this_run = training_seconds
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "training": {
                    "optimizer_updates": updates,
                    "actual_node_exposures": actual_exposures,
                    "training_seconds": training_seconds,
                },
            },
            cache_path,
        )

    inference_start = time.perf_counter()
    raw_embedding = infer_embeddings(
        args.method,
        model,
        features=features,
        section_graphs=infer_section,
        device=device,
    )
    if args.method == "graphst":
        embedding = PCA(n_components=20, random_state=42).fit_transform(raw_embedding)
    else:
        embedding = raw_embedding
    torch.cuda.synchronize() if device.type == "cuda" else None
    inference_seconds = time.perf_counter() - inference_start

    details = {
        "training_seconds": training_seconds,
        "training_seconds_this_run": training_seconds_this_run,
        "model_cache_reused": cache_reused,
        "inference_seconds": inference_seconds,
        "optimizer_updates": updates,
        "target_node_exposures": target_exposures,
        "actual_node_exposures": actual_exposures,
        "gradient_accumulation": args.training_mode == "fullneighbors",
        "optimizer_step_scope": (
            "one complete node pass"
            if args.training_mode in {"whole", "fullneighbors"}
            else "sampled subgraph"
        ),
        "train_edges": int(train_graph.num_edges()),
        "infer_edges": int(infer_graph.num_edges()),
    }
    del model, features, matrix, raw_embedding
    return np.asarray(embedding, dtype=np.float32), details


def run_one(args: argparse.Namespace) -> None:
    if args.method is None or args.training_mode is None:
        raise ValueError("--method and --training-mode are required")
    if args.method == "banksy" and args.training_mode != "fixed":
        raise ValueError("BANKSY supports only --training-mode fixed")
    if args.method != "banksy" and args.training_mode == "fixed":
        raise ValueError("fixed training mode is only valid for BANKSY")

    name = (
        f"{args.section}_{args.method}_{args.training_mode}_"
        f"train-{args.train_graph}_infer-{args.infer_graph}"
    )
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "metrics.json"
    if result_path.exists() and not args.force:
        print(f"Skipping completed run: {run_dir}", flush=True)
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    source = args.input_root / args.section / f"{args.section}_annotated.h5ad"
    source_adata = read_section(args.input_root, args.section)
    adata = preprocess(source_adata, args.method)
    truth = source_adata.obs["gd"].to_numpy()
    coordinates = np.asarray(source_adata.obsm["spatial"], dtype=float)

    if args.method == "banksy":
        start = time.perf_counter()
        embedding, infer_edges = banksy_embedding(
            adata,
            graph_mode=args.infer_graph,
            seed=args.seed,
        )
        training_details = {
            "training_seconds": 0.0,
            "inference_seconds": time.perf_counter() - start,
            "optimizer_updates": None,
            "target_node_exposures": None,
            "actual_node_exposures": None,
            "train_edges": None,
            "infer_edges": infer_edges,
        }
    else:
        algorithm_tag = (
            "_accumulated-full-v1" if args.training_mode == "fullneighbors" else ""
        )
        cache_name = (
            f"{args.section}_{args.method}_{args.training_mode}_"
            f"train-{args.train_graph}_sample-{args.sample_size}_"
            f"fulln-{args.fullneighbor_batch_size}_seed-{args.seed}"
            f"{algorithm_tag}.pt"
        )
        embedding, training_details = train_graph_method(
            args,
            adata,
            device,
            args.output_dir / "_model_cache" / cache_name,
        )

    raw_labels = mclust_eee(
        embedding,
        n_clusters=N_CLUSTERS[args.section],
        seed=2020,
    )
    predictions = {"mclust": raw_labels}
    if args.method == "graphst":
        predictions["mclust_refined"] = graphst_refine(raw_labels, coordinates)

    rows = []
    for variant, labels in predictions.items():
        row = {
            "section": args.section,
            "method": args.method,
            "training_mode": args.training_mode,
            "train_graph": None if args.method == "banksy" else args.train_graph,
            "infer_graph": args.infer_graph,
            "variant": variant,
        }
        row.update(evaluate(truth, labels, coordinates))
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "section_metrics.csv", index=False)
    pd.DataFrame(
        {
            "ground_truth": truth,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            **predictions,
        },
        index=source_adata.obs_names,
    ).to_csv(run_dir / "assignments.csv.gz", compression="gzip")
    np.savez_compressed(
        run_dir / "embedding.npz",
        obs_names=source_adata.obs_names.to_numpy(dtype=str),
        embedding=embedding,
    )

    result = {
        "section": args.section,
        "method": args.method,
        "training_mode": args.training_mode,
        "train_graph": None if args.method == "banksy" else args.train_graph,
        "infer_graph": args.infer_graph,
        "annotation_key": "gd",
        "source": str(source),
        "input_semantics": "raw integer counts in X",
        "n_spots": int(adata.n_obs),
        "n_genes_after_selection": int(adata.n_vars),
        "default_whole_graph_epochs": DEFAULT_EPOCHS.get(args.method),
        "banksy_configuration": (
            {
                "max_m": 1,
                "lambda": 0.2,
                "branches": ["self", "m0", "m1_agf"],
                "m0_neighbors": DEFAULT_NEIGHBORS["banksy"],
                "m1_neighbors": 2 * DEFAULT_NEIGHBORS["banksy"],
            }
            if args.method == "banksy"
            else None
        ),
        "training": training_details,
        "metrics": rows,
        "total_seconds": time.perf_counter() - total_start,
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
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    del source_adata, adata, embedding, predictions
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def summarize(output_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(output_dir.glob("*/*section_metrics.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No completed runs under {output_dir}")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(output_dir / "dlpfc_graph_mode_metrics.csv", index=False)
    preferred = frame.loc[
        (frame["method"] != "graphst")
        | (frame["variant"] == "mclust_refined")
    ].copy()
    preferred = preferred.sort_values(
        ["method", "training_mode", "train_graph", "infer_graph"],
        na_position="first",
    )
    preferred.to_csv(output_dir / "dlpfc_graph_mode_summary.csv", index=False)
    return preferred


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(summarize(args.output_dir).to_string(index=False), flush=True)
    else:
        run_one(args)
        print(summarize(args.output_dir).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
