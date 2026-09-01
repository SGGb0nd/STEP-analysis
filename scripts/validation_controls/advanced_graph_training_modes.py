#!/usr/bin/env python
"""Run matched graph-input controls for GraphST and STAGATE on MERFISH."""


import argparse
import gc
import json
import math
import resource
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anndata as ad
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F

from step.utils.gbolt import MultiGraphsAllNodesSampler
from step.utils.misc import generate_adj
from merfish_graph_training_modes import (
    DEFAULT_INPUT,
    DEFAULT_SECTIONS,
    IGNORE_LABEL_VALUES,
    LABEL_KEYS,
    annotation_induced_graph,
    cluster_embedding,
    configure_torch_cuda_workarounds,
    evaluate,
    graph_summary,
    load_subset,
    set_seed,
)


METHODS = ("graphst", "stagate")
MODES = (
    "sampled",
    "full",
    "oracle_major",
    "oracle_ccf",
    "sampled_oracle_major",
    "sampled_oracle_ccf",
    "fullneighbors",
)
GRAPH_LABEL_KEYS = {
    "oracle_major": "major_brain_region",
    "oracle_ccf": "ccf_region_name",
    "sampled_oracle_major": "major_brain_region",
    "sampled_oracle_ccf": "ccf_region_name",
}


@dataclass
class SectionGraph:
    name: str
    global_indices: np.ndarray
    graph: dgl.DGLGraph


class BilinearDiscriminator(nn.Module):
    """GraphST's bilinear discriminator."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(
        self,
        context: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (self.score(positive, context), self.score(negative, context)),
            dim=1,
        )


class SparseGraphST(nn.Module):
    """Sparse implementation of the official GraphST 1.1.1 encoder."""

    def __init__(self, input_dim: int, output_dim: int = 64) -> None:
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(input_dim, output_dim))
        self.weight2 = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        self.discriminator = BilinearDiscriminator(output_dim)

    def forward(
        self,
        features: torch.Tensor,
        corrupted: torch.Tensor,
        adjacency: torch.Tensor,
        neighbor_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_linear = torch.sparse.mm(adjacency, features @ self.weight1)
        reconstruction = torch.sparse.mm(adjacency, latent_linear @ self.weight2)
        latent = F.relu(latent_linear)

        corrupted_linear = torch.sparse.mm(adjacency, corrupted @ self.weight1)
        corrupted_latent = F.relu(corrupted_linear)

        context = torch.sigmoid(torch.sparse.mm(neighbor_mean, latent))
        corrupted_context = torch.sigmoid(
            torch.sparse.mm(neighbor_mean, corrupted_latent)
        )
        logits = self.discriminator(context, latent, corrupted_latent)
        corrupted_logits = self.discriminator(
            corrupted_context,
            corrupted_latent,
            latent,
        )
        return latent_linear, reconstruction, logits, corrupted_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--n-sections", type=int, choices=[1])
    parser.add_argument("--sections", nargs=5, default=list(DEFAULT_SECTIONS))
    parser.add_argument("--batch-key", default="brain_section_label")
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--fullneighbor-batch-size", type=int, default=2029)
    parser.add_argument("--fullneighbor-updates", type=int, default=3000)
    parser.add_argument("--sampled-iterations", type=int, default=3000)
    parser.add_argument("--sampled-node-inclusions", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def build_section_graphs(
    adata: ad.AnnData,
    *,
    sections: list[str],
    batch_key: str,
    max_neighbors: int,
    graph_label_key: str | None,
) -> list[SectionGraph]:
    section_values = adata.obs[batch_key].astype(str).to_numpy()
    result = []
    for section in sections:
        global_indices = np.flatnonzero(section_values == section)
        section_adata = adata[global_indices]
        if graph_label_key is None:
            graph = generate_adj(
                section_adata,
                edge_clip=None,
                max_neighbors=max_neighbors,
            )
        else:
            graph = annotation_induced_graph(
                section_adata,
                label_key=graph_label_key,
                max_neighbors=max_neighbors,
            )
        graph.ndata["node_ids"] = torch.from_numpy(global_indices)
        result.append(
            SectionGraph(
                name=section,
                global_indices=global_indices,
                graph=graph,
            )
        )
    return result


def scale_graphst_features(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Apply GraphST's non-centered per-gene scaling without re-normalizing X."""
    matrix = matrix.astype(np.float32, copy=True)
    n_obs = matrix.shape[0]
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second_moment = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second_moment - mean**2, 0)
    if n_obs > 1:
        variance *= n_obs / (n_obs - 1)
    std = np.sqrt(variance)
    inv_std = np.zeros_like(std)
    nonzero = std > 0
    inv_std[nonzero] = 1.0 / std[nonzero]
    matrix = matrix.multiply(inv_std).tocsr()
    np.minimum(matrix.data, 10.0, out=matrix.data)
    return matrix


def graph_tensors(
    graph: dgl.DGLGraph,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return PyG edges plus GraphST normalized and mean adjacency matrices."""
    src, dst = graph.edges()
    src = src.to(device=device, dtype=torch.long)
    dst = dst.to(device=device, dtype=torch.long)
    n_nodes = graph.num_nodes()
    edge_index = torch.stack((src, dst), dim=0)

    degree = torch.bincount(dst, minlength=n_nodes).float().clamp_min_(1)
    inv_sqrt = degree.rsqrt()
    normalized_values = inv_sqrt[dst] * inv_sqrt[src]
    adjacency = torch.sparse_coo_tensor(
        torch.stack((dst, src)),
        normalized_values,
        size=(n_nodes, n_nodes),
        device=device,
    ).coalesce()
    neighbor_mean = torch.sparse_coo_tensor(
        torch.stack((dst, src)),
        degree[dst].reciprocal(),
        size=(n_nodes, n_nodes),
        device=device,
    ).coalesce()
    return edge_index, adjacency, neighbor_mean


def graphst_loss(
    model: SparseGraphST,
    features: torch.Tensor,
    graph: dgl.DGLGraph,
    device: torch.device,
) -> torch.Tensor:
    _, adjacency, neighbor_mean = graph_tensors(graph, device)
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
    contrastive = F.binary_cross_entropy_with_logits(logits, labels)
    contrastive += F.binary_cross_entropy_with_logits(corrupted_logits, labels)
    return 10.0 * F.mse_loss(reconstruction, features) + contrastive


def stagate_components(
    input_dim: int,
    device: torch.device,
) -> tuple[nn.Module, Callable[[nn.Module, torch.Tensor, dgl.DGLGraph], torch.Tensor]]:
    # Upstream imports torch_sparse even when its Tensor-only edge path is used.
    if "torch_sparse" not in sys.modules:
        torch_sparse = types.ModuleType("torch_sparse")

        class SparseTensor:  # pragma: no cover - compatibility type only
            pass

        def set_diag(_: object) -> object:  # pragma: no cover - never used here
            raise TypeError("SparseTensor input is not supported by this benchmark")

        torch_sparse.SparseTensor = SparseTensor
        torch_sparse.set_diag = set_diag
        sys.modules["torch_sparse"] = torch_sparse
    try:
        from STAGATE_pyG.STAGATE import STAGATE
    except ImportError as exc:
        raise RuntimeError(
            "STAGATE_pyG is required. Run with uv --with torch-geometric and "
            "the pinned STAGATE_pyG git dependency."
        ) from exc

    model = STAGATE(hidden_dims=[input_dim, 512, 30]).to(device)

    def loss_fn(
        current_model: nn.Module,
        features: torch.Tensor,
        graph: dgl.DGLGraph,
    ) -> torch.Tensor:
        edge_index, _, _ = graph_tensors(graph, device)
        _, reconstruction = current_model(features, edge_index)
        return F.mse_loss(reconstruction, features)

    return model, loss_fn


def graphst_components(
    input_dim: int,
    device: torch.device,
) -> tuple[SparseGraphST, Callable[[nn.Module, torch.Tensor, dgl.DGLGraph], torch.Tensor]]:
    model = SparseGraphST(input_dim=input_dim, output_dim=64).to(device)

    def loss_fn(
        current_model: nn.Module,
        features: torch.Tensor,
        graph: dgl.DGLGraph,
    ) -> torch.Tensor:
        return graphst_loss(current_model, features, graph, device)

    return model, loss_fn


def make_sample_loader(
    section_graphs: list[SectionGraph],
    *,
    sample_size: int,
    iterations: int,
) -> dgl.dataloading.DataLoader:
    graph = dgl.batch([item.graph for item in section_graphs])
    if len(section_graphs) == 1:
        sampler = dgl.dataloading.SAINTSampler(mode="node", budget=sample_size)
    else:
        sampler = MultiGraphsAllNodesSampler(
            mode="node",
            budget=1,
            n_graphs=1,
            ratio=sample_size,
        )
    return dgl.dataloading.DataLoader(
        graph,
        torch.arange(iterations),
        sampler,
        shuffle=True,
    )


def train_sampled(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, torch.Tensor, dgl.DGLGraph], torch.Tensor],
    *,
    features: torch.Tensor,
    section_graphs: list[SectionGraph],
    sample_size: int,
    iterations: int,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float | None,
    device: torch.device,
) -> int:
    loader = make_sample_loader(
        section_graphs,
        sample_size=sample_size,
        iterations=iterations,
    )
    node_inclusions = 0
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
        node_inclusions += int(node_ids.numel())
    return node_inclusions


def graphst_block_matrices(
    block: dgl.DGLGraph,
    full_degree: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build rectangular GraphST operators using full-graph normalization."""
    src, dst = block.edges(order="eid")
    src_global = block.srcdata[dgl.NID][src]
    dst_global = block.dstdata[dgl.NID][dst]
    src_degree = full_degree[src_global]
    dst_degree = full_degree[dst_global]
    indices = torch.stack((dst, src)).to(device)
    normalized_values = (dst_degree.rsqrt() * src_degree.rsqrt()).to(device)
    mean_values = dst_degree.reciprocal().to(device)
    shape = (block.num_dst_nodes(), block.num_src_nodes())
    adjacency = torch.sparse_coo_tensor(
        indices,
        normalized_values,
        size=shape,
        device=device,
    ).coalesce()
    neighbor_mean = torch.sparse_coo_tensor(
        indices,
        mean_values,
        size=shape,
        device=device,
    ).coalesce()
    return adjacency, neighbor_mean


def graphst_fullneighbor_loss(
    model: SparseGraphST,
    input_features: torch.Tensor,
    corrupted_features: torch.Tensor,
    target_features: torch.Tensor,
    blocks: list[dgl.DGLGraph],
    full_degree: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute GraphST loss for targets with their exact two-hop neighbors."""
    if len(blocks) != 2:
        raise ValueError("GraphST full-neighbor training requires two blocks")
    if not torch.equal(
        blocks[0].dstdata[dgl.NID],
        blocks[1].srcdata[dgl.NID],
    ):
        raise RuntimeError("DGL full-neighbor blocks are not aligned")

    adjacency_0, _ = graphst_block_matrices(blocks[0], full_degree, device)
    adjacency_1, neighbor_mean_1 = graphst_block_matrices(
        blocks[1],
        full_degree,
        device,
    )

    latent_linear = torch.sparse.mm(adjacency_0, input_features @ model.weight1)
    reconstruction = torch.sparse.mm(
        adjacency_1,
        latent_linear @ model.weight2,
    )
    latent = F.relu(latent_linear)

    corrupted_linear = torch.sparse.mm(
        adjacency_0,
        corrupted_features @ model.weight1,
    )
    corrupted_latent = F.relu(corrupted_linear)
    target_count = blocks[1].num_dst_nodes()
    positive = latent[:target_count]
    negative = corrupted_latent[:target_count]
    context = torch.sigmoid(torch.sparse.mm(neighbor_mean_1, latent))
    corrupted_context = torch.sigmoid(
        torch.sparse.mm(neighbor_mean_1, corrupted_latent)
    )
    logits = model.discriminator(context, positive, negative)
    corrupted_logits = model.discriminator(
        corrupted_context,
        negative,
        positive,
    )
    labels = torch.stack(
        (
            torch.ones(target_count, device=device),
            torch.zeros(target_count, device=device),
        ),
        dim=1,
    )
    contrastive = F.binary_cross_entropy_with_logits(logits, labels)
    contrastive += F.binary_cross_entropy_with_logits(corrupted_logits, labels)
    return 10.0 * F.mse_loss(reconstruction, target_features) + contrastive


def train_graphst_fullneighbors(
    model: SparseGraphST,
    *,
    features: torch.Tensor,
    section_graphs: list[SectionGraph],
    batch_size: int,
    updates: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Train GraphST on target batches with their exact two-hop inputs."""
    if len(section_graphs) != 1:
        raise ValueError("GraphST full-neighbor control supports one section")
    graph = section_graphs[0].graph
    loader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(graph.num_nodes()),
        dgl.dataloading.MultiLayerFullNeighborSampler(2),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    full_degree = graph.in_degrees().float().clamp_min_(1)
    target_node_exposures = 0
    total_nodes = graph.num_nodes()
    model.train()
    loader_iterator = iter(loader)
    for _ in range(updates):
        try:
            input_nodes, output_nodes, blocks = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            input_nodes, output_nodes, blocks = next(loader_iterator)
        permutation = torch.randperm(features.shape[0], device=device)
        optimizer.zero_grad(set_to_none=True)
        input_nodes_device = input_nodes.to(device)
        output_nodes_device = output_nodes.to(device)
        loss = graphst_fullneighbor_loss(
            model,
            features[input_nodes_device],
            features[permutation[input_nodes_device]],
            features[output_nodes_device],
            blocks,
            full_degree,
            device,
        )
        loss.backward()
        optimizer.step()
        target_node_exposures += int(output_nodes.numel())
    return target_node_exposures


def train_stagate_fullneighbors(
    model: nn.Module,
    *,
    features: torch.Tensor,
    section_graphs: list[SectionGraph],
    batch_size: int,
    updates: int,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float | None,
    device: torch.device,
) -> int:
    """Train STAGATE on targets with the complete graph-dependent receptive field."""
    if len(section_graphs) != 1:
        raise ValueError("STAGATE full-neighbor control supports one section")
    graph = section_graphs[0].graph
    loader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(graph.num_nodes()),
        dgl.dataloading.MultiLayerFullNeighborSampler(2),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    loader_iterator = iter(loader)
    target_node_exposures = 0
    model.train()
    for _ in range(updates):
        try:
            input_nodes, output_nodes, _ = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            input_nodes, output_nodes, _ = next(loader_iterator)

        subgraph = dgl.node_subgraph(graph, input_nodes)
        global_nodes = subgraph.ndata[dgl.NID]
        global_to_local = torch.empty(graph.num_nodes(), dtype=torch.long)
        global_to_local[global_nodes] = torch.arange(global_nodes.numel())
        target_local = global_to_local[output_nodes].to(device)
        src, dst = subgraph.edges(order="eid")
        edge_index = torch.stack((src, dst)).to(device)
        batch_features = features[global_nodes.to(device)]

        optimizer.zero_grad(set_to_none=True)
        _, reconstruction = model(batch_features, edge_index)
        loss = F.mse_loss(
            reconstruction[target_local],
            batch_features[target_local],
        )
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        target_node_exposures += int(output_nodes.numel())
    return target_node_exposures


def train_full(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, torch.Tensor, dgl.DGLGraph], torch.Tensor],
    *,
    features: torch.Tensor,
    section_graphs: list[SectionGraph],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float | None,
    device: torch.device,
) -> int:
    total_nodes = sum(item.graph.num_nodes() for item in section_graphs)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for item in section_graphs:
            node_ids = torch.from_numpy(item.global_indices).to(device)
            weight = item.graph.num_nodes() / total_nodes
            loss = loss_fn(model, features[node_ids], item.graph)
            (weight * loss).backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    return epochs * total_nodes


@torch.no_grad()
def infer_embeddings(
    method: str,
    model: nn.Module,
    *,
    features: torch.Tensor,
    section_graphs: list[SectionGraph],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output_dim = features.shape[1] if method == "graphst" else 30
    result = np.empty((features.shape[0], output_dim), dtype=np.float32)
    for item in section_graphs:
        node_ids = torch.from_numpy(item.global_indices).to(device)
        section_features = features[node_ids]
        edge_index, adjacency, neighbor_mean = graph_tensors(item.graph, device)
        if method == "graphst":
            _, reconstruction, _, _ = model(
                section_features,
                section_features,
                adjacency,
                neighbor_mean,
            )
            embedding = reconstruction
        else:
            embedding, _ = model(section_features, edge_index)
        result[item.global_indices] = embedding.detach().cpu().numpy()
    return result


def prepare_features(adata: ad.AnnData, method: str) -> sp.csr_matrix:
    matrix = adata.X
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    else:
        matrix = matrix.tocsr().astype(np.float32)
    if method == "graphst":
        matrix = scale_graphst_features(matrix)
    return matrix


def run_one(args: argparse.Namespace) -> None:
    if args.method is None or args.mode is None or args.n_sections is None:
        raise ValueError("--method, --mode, and --n-sections are required")
    if args.sampled_node_inclusions is None:
        raise ValueError("--sampled-node-inclusions is required for matched runs")
    run_dir = args.output_dir / f"n{args.n_sections}_{args.method}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "metrics.json"
    if result_path.exists() and not args.force:
        print(f"Skipping completed run: {run_dir}", flush=True)
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        configure_torch_cuda_workarounds()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    sections = list(args.sections[: args.n_sections])
    adata = load_subset(args.input, sections, args.batch_key, LABEL_KEYS)
    matrix = prepare_features(adata, args.method)
    features = torch.from_numpy(matrix.toarray()).to(device)

    graph_label_key = GRAPH_LABEL_KEYS.get(args.mode)
    section_graphs = build_section_graphs(
        adata,
        sections=sections,
        batch_key=args.batch_key,
        max_neighbors=args.max_neighbors,
        graph_label_key=graph_label_key,
    )
    graph_details = {}
    for item in section_graphs:
        section_adata = adata[item.global_indices]
        details = graph_summary(
            section_adata,
            label_keys=LABEL_KEYS,
            max_neighbors=args.max_neighbors,
        )
        details["training_graph_edges_including_self"] = int(item.graph.num_edges())
        graph_details[item.name] = details

    if args.method == "graphst":
        model, loss_fn = graphst_components(features.shape[1], device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        gradient_clip = None
        embedding_definition = "GraphST reconstructed expression"
    else:
        model, loss_fn = stagate_components(features.shape[1], device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=0.0001,
        )
        gradient_clip = 5.0
        embedding_definition = "STAGATE 30-dimensional latent representation"

    train_start = time.perf_counter()
    sampled_training = args.mode.startswith("sampled")
    if sampled_training:
        optimizer_updates = args.sampled_iterations
        actual_node_exposures = train_sampled(
            model,
            loss_fn,
            features=features,
            section_graphs=section_graphs,
            sample_size=args.sample_size,
            iterations=args.sampled_iterations,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
            device=device,
        )
        full_epochs = None
    elif args.mode == "fullneighbors":
        optimizer_updates = args.fullneighbor_updates
        if args.method == "graphst":
            actual_node_exposures = train_graphst_fullneighbors(
                model,
                features=features,
                section_graphs=section_graphs,
                batch_size=args.fullneighbor_batch_size,
                updates=args.fullneighbor_updates,
                optimizer=optimizer,
                device=device,
            )
        else:
            actual_node_exposures = train_stagate_fullneighbors(
                model,
                features=features,
                section_graphs=section_graphs,
                batch_size=args.fullneighbor_batch_size,
                updates=args.fullneighbor_updates,
                optimizer=optimizer,
                gradient_clip=gradient_clip,
                device=device,
            )
        full_epochs = None
    else:
        full_epochs = max(1, round(args.sampled_node_inclusions / adata.n_obs))
        optimizer_updates = full_epochs
        actual_node_exposures = train_full(
            model,
            loss_fn,
            features=features,
            section_graphs=section_graphs,
            epochs=full_epochs,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
            device=device,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_start

    inference_start = time.perf_counter()
    embedding = infer_embeddings(
        args.method,
        model,
        features=features,
        section_graphs=section_graphs,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start

    evaluation_start = time.perf_counter()
    pooled = {}
    clustering = {}
    section_frames = []
    for label_key in LABEL_KEYS:
        suffix = "major" if label_key == "major_brain_region" else "ccf"
        prediction_key = f"domain_{suffix}"
        valid = ~adata.obs[label_key].astype(str).isin(IGNORE_LABEL_VALUES)
        n_clusters = adata.obs.loc[valid, label_key].nunique()
        prediction, clusterer_name = cluster_embedding(
            embedding,
            n_clusters=n_clusters,
            seed=args.seed,
        )
        adata.obs[prediction_key] = pd.Categorical(prediction.astype(str))
        metrics, section_metrics = evaluate(
            adata,
            batch_key=args.batch_key,
            label_key=label_key,
            prediction_key=prediction_key,
        )
        pooled[label_key] = metrics
        clustering[label_key] = clusterer_name
        section_frames.append(section_metrics)
    pd.concat(section_frames, ignore_index=True).to_csv(
        run_dir / "section_metrics.csv",
        index=False,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    assignments = adata.obs[
        [args.batch_key, *LABEL_KEYS, "domain_major", "domain_ccf", "x", "y"]
    ].copy()
    assignments.to_csv(run_dir / "assignments.csv.gz", compression="gzip")

    result = {
        "method": args.method,
        "mode": args.mode,
        "scenario": "single" if args.n_sections == 1 else "multiple",
        "n_sections": args.n_sections,
        "sections": sections,
        "source": str(args.input),
        "input_semantics": (
            "source X; already log-normalized; no additional normalize_total/log1p"
        ),
        "feature_preprocessing": (
            "GraphST non-centered per-gene scaling with max 10"
            if args.method == "graphst"
            else "source log-normalized expression"
        ),
        "embedding_definition": embedding_definition,
        "graph_label_key": graph_label_key,
        "graph_definition": (
            f"section-wise spatial {args.max_neighbors}-NN graph"
            if graph_label_key is None
            else (
                f"section-wise spatial {args.max_neighbors}-NN graph restricted "
                f"to same-{graph_label_key} edges"
            )
        ),
        "training": {
            "sampling": (
                "GraphSAINT node"
                if sampled_training
                else "full-neighbor"
                if args.mode == "fullneighbors"
                else "full graph"
            ),
            "sample_size": (
                args.sample_size
                if sampled_training
                else args.fullneighbor_batch_size
                if args.mode == "fullneighbors"
                else None
            ),
            "sampled_iterations": (
                args.sampled_iterations if sampled_training else None
            ),
            "fullneighbor_updates": (
                args.fullneighbor_updates if args.mode == "fullneighbors" else None
            ),
            "full_graph_epochs": full_epochs,
            "optimizer_updates": optimizer_updates,
            "target_sampled_node_inclusions": args.sampled_node_inclusions,
            "actual_node_exposures": actual_node_exposures,
            "node_exposure_ratio": (
                actual_node_exposures / args.sampled_node_inclusions
            ),
            "seed": args.seed,
        },
        "clustering": clustering,
        "graph_summary": graph_details,
        "metrics": pooled,
        "timing_seconds": {
            "training": training_seconds,
            "inference": inference_seconds,
            "evaluation": evaluation_seconds,
            "total": time.perf_counter() - total_start,
        },
        "resource": {
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else 0.0
            ),
            "peak_cuda_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2**30
                if device.type == "cuda"
                else 0.0
            ),
            "peak_process_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    del model, features, embedding, matrix, adata
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def summarize(output_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(output_dir.glob("n*_*_*/metrics.json")):
        record = json.loads(metrics_path.read_text())
        rows.append(
            {
                "n_sections": record["n_sections"],
                "method": record["method"],
                "mode": record["mode"],
                **{
                    f"{label}_{metric}": value
                    for label, metrics in record["metrics"].items()
                    for metric, value in metrics.items()
                },
                **record["timing_seconds"],
                **record["resource"],
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["method", "n_sections", "mode"])
    frame.to_csv(output_dir / "benchmark_summary.csv", index=False)
    return frame


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(summarize(args.output_dir).to_string(index=False), flush=True)
    else:
        run_one(args)
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
