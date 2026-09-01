#!/usr/bin/env python
"""Compare sampled, full, and annotation-induced graph training on MERFISH."""


import argparse
import gc
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import dgl
import dgl.function as dgl_fn
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from step import stModel
from step.models.transcriptformer import Readout
from step.utils.misc import generate_adj
from banksy_default_kernel import (
    BanksyWeights,
    generate_banksy_default_weights,
)


DEFAULT_INPUT = Path("data/WB_MERFISH_animal3_sagittal.h5ad")
DEFAULT_SECTIONS = tuple(f"C57BL6J-3.{i:03d}" for i in (10, 11, 9, 8, 12))
LABEL_KEYS = ("major_brain_region", "ccf_region_name")
IGNORE_LABEL_VALUES = frozenset({"n/a"})
WHOLE_GRAPH_MODES = ("sampled", "full", "oracle_major", "oracle_ccf")
FULLNEIGHBOR_MODES = (
    "sampled",
    "fullneighbors",
    "oracle_major_fullneighbors",
    "oracle_ccf_fullneighbors",
)
FIXED_KERNEL_FULLNEIGHBOR_MODE = "fixedkernel_fullneighbors"
FIXED_KERNEL_SAMPLED_MODE = "fixedkernel_sampled"
GAUSSIAN_KERNEL_SAMPLED_MODE = "gaussiankernel_sampled"
M1_KERNEL_SAMPLED_MODE = "m1kernel_sampled"
FIXED_KERNEL_MODES = (
    FIXED_KERNEL_FULLNEIGHBOR_MODE,
    FIXED_KERNEL_SAMPLED_MODE,
)
MODES = tuple(
    dict.fromkeys(
        (
            *WHOLE_GRAPH_MODES,
            *FULLNEIGHBOR_MODES,
            *FIXED_KERNEL_MODES,
            GAUSSIAN_KERNEL_SAMPLED_MODE,
            M1_KERNEL_SAMPLED_MODE,
        )
    )
)


def configure_torch_cuda_workarounds() -> None:
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--n-sections", type=int, choices=range(1, 6))
    parser.add_argument("--sections", nargs=5, default=list(DEFAULT_SECTIONS))
    parser.add_argument("--batch-key", default="brain_section_label")
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--fullneighbor-batch-size", type=int, default=256)
    parser.add_argument("--full-graph-encoder-batch-size", type=int, default=None)
    parser.add_argument("--equivalent-epochs", type=int, default=20)
    parser.add_argument(
        "--sampled-iterations",
        type=int,
        default=None,
        help="Fixed sampled iteration count; overrides the equivalent-epoch schedule.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument(
        "--contrast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--match-node-exposure",
        action="store_true",
        help="Match the number of nodes contributing to the training loss.",
    )
    parser.add_argument(
        "--sampled-node-inclusions",
        type=int,
        default=None,
        help="Measured total node inclusions across sampled training batches.",
    )
    parser.add_argument("--run-matrix", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--matrix-profile",
        choices=("whole", "fullneighbors"),
        default="whole",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subset(
    path: Path,
    sections: list[str],
    batch_key: str,
    label_keys: tuple[str, ...],
) -> ad.AnnData:
    source = ad.read_h5ad(path, backed="r")
    batch = source.obs[batch_key].astype(str)
    keep = batch.isin(sections)
    for label_key in label_keys:
        keep &= source.obs[label_key].notna()
    subset = source[keep.to_numpy()].to_memory()
    source.file.close()

    present = set(subset.obs[batch_key].astype(str))
    missing = [section for section in sections if section not in present]
    if missing:
        raise ValueError(f"Sections contain no annotated cells: {missing}")

    subset.obs[batch_key] = pd.Categorical(
        subset.obs[batch_key].astype(str), categories=sections, ordered=True
    )
    for label_key in label_keys:
        subset.obs[label_key] = subset.obs[label_key].astype(str).astype("category")
    coords = np.asarray(subset.obsm["X_spatial_coords"], dtype=np.float32)
    subset.obs["x"] = coords[:, 0]
    subset.obs["y"] = coords[:, 1]
    subset.obs["array_row"] = coords[:, 0]
    subset.obs["array_col"] = coords[:, 1]
    return subset


def annotation_induced_graph(
    adata: ad.AnnData,
    *,
    label_key: str,
    max_neighbors: int,
) -> dgl.DGLGraph:
    """Keep only same-label edges from the ordinary spatial kNN graph."""
    graph = generate_adj(adata, edge_clip=None, max_neighbors=max_neighbors)
    src, dst = graph.edges()
    labels = adata.obs[label_key].astype(str).to_numpy()
    keep = labels[src.numpy()] == labels[dst.numpy()]
    filtered = dgl.graph(
        (src[keep], dst[keep]),
        num_nodes=graph.num_nodes(),
        idtype=graph.idtype,
    )
    filtered.ndata["xy"] = graph.ndata["xy"].clone()
    return dgl.add_self_loop(dgl.remove_self_loop(filtered))


def banksy_default_weights(
    adata: ad.AnnData,
    *,
    max_neighbors: int,
) -> BanksyWeights:
    """Return BANKSY's default m=0 and m=1 spatial kernels."""
    coordinates = np.asarray(adata.obsm["X_spatial_coords"], dtype=np.float32)
    return generate_banksy_default_weights(
        coordinates,
        num_neighbors=max_neighbors,
        decay_type="scaled_gaussian",
    )


def banksy_default_graph(
    adata: ad.AnnData,
    *,
    max_neighbors: int,
) -> dgl.DGLGraph:
    """Build a DGL graph carrying both BANKSY default kernels."""
    coordinates = np.asarray(adata.obsm["X_spatial_coords"], dtype=np.float32)
    weights = banksy_default_weights(
        adata,
        max_neighbors=max_neighbors,
    )
    m0 = weights.m0.tocoo()
    m1 = weights.m1.tocoo()

    rows = np.concatenate((m0.row, m1.row)).astype(np.int64, copy=False)
    cols = np.concatenate((m0.col, m1.col)).astype(np.int64, copy=False)
    n0, n1 = m0.nnz, m1.nnz

    # BANKSY row i contains weights for neighbors j. DGL messages therefore
    # travel from column j to row i.
    graph = dgl.graph(
        (
            torch.from_numpy(cols),
            torch.from_numpy(rows),
        ),
        num_nodes=adata.n_obs,
    )
    graph.edata["banksy_m0_weight"] = torch.from_numpy(
        np.concatenate((m0.data.real, np.zeros(n1))).astype(np.float32)
    )
    graph.edata["banksy_m1_real"] = torch.from_numpy(
        np.concatenate((np.zeros(n0), m1.data.real)).astype(np.float32)
    )
    graph.edata["banksy_m1_imag"] = torch.from_numpy(
        np.concatenate((np.zeros(n0), m1.data.imag)).astype(np.float32)
    )
    graph.edata["banksy_m1_support"] = torch.from_numpy(
        np.concatenate((np.zeros(n0), np.ones(n1))).astype(np.float32)
    )
    graph.ndata["xy"] = torch.from_numpy(coordinates)
    return graph


def banksy_gaussian_graph(
    adata: ad.AnnData,
    *,
    max_neighbors: int,
) -> dgl.DGLGraph:
    """Build the row-normalized BANKSY m=0 Gaussian kernel."""
    coordinates = np.asarray(adata.obsm["X_spatial_coords"], dtype=np.float32)
    m0 = banksy_default_weights(adata, max_neighbors=max_neighbors).m0.tocoo()
    graph = dgl.graph(
        (
            torch.from_numpy(m0.col.astype(np.int64, copy=False)),
            torch.from_numpy(m0.row.astype(np.int64, copy=False)),
        ),
        num_nodes=adata.n_obs,
    )
    graph.edata["banksy_gaussian_weight"] = torch.from_numpy(
        m0.data.real.astype(np.float32, copy=False)
    )
    graph.ndata["xy"] = torch.from_numpy(coordinates)
    return graph


class BanksyGaussianKernel(torch.nn.Module):
    """Apply the single-branch row-normalized BANKSY Gaussian kernel."""

    @staticmethod
    def _aggregate(graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            graph.srcdata["_banksy_input"] = features
            src, dst = graph.edges(order="eid")
            if graph.is_block:
                src_ids = graph.srcdata[dgl.NID][src]
                dst_ids = graph.dstdata[dgl.NID][dst]
            else:
                src_ids, dst_ids = src, dst
            weights = graph.edata["banksy_gaussian_weight"].to(features.dtype)
            weights = weights.masked_fill(src_ids == dst_ids, 0).unsqueeze(-1)
            graph.edata["_banksy_weight"] = weights
            graph.update_all(
                dgl_fn.u_mul_e(
                    "_banksy_input",
                    "_banksy_weight",
                    "_banksy_message",
                ),
                dgl_fn.sum("_banksy_message", "_banksy_output"),
            )
            return graph.dstdata["_banksy_output"]

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        return self._aggregate(graph, features)

    def batch_forward(
        self,
        blocks: list[dgl.DGLGraph],
        features: torch.Tensor,
    ) -> torch.Tensor:
        if len(blocks) != 1:
            raise ValueError("BANKSY Gaussian kernel requires one graph layer")
        return self._aggregate(blocks[0], features)


class BanksyDefaultKernel(torch.nn.Module):
    """Apply BANKSY's self, m=0, and m=1 branches to hidden features."""

    scale_squared = (0.8, 2.0 / 15.0, 1.0 / 15.0)

    @staticmethod
    def _zscore(features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=0, keepdim=True)
        variance = features.var(dim=0, correction=0, keepdim=True)
        return torch.nan_to_num((features - mean) / variance.sqrt())

    def _branches(
        self,
        graph: dgl.DGLGraph,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = {
            "banksy_m0_weight",
            "banksy_m1_real",
            "banksy_m1_imag",
            "banksy_m1_support",
        }
        missing = required.difference(graph.edata.keys())
        if missing:
            raise KeyError(f"missing BANKSY edge features: {sorted(missing)}")

        with graph.local_scope():
            graph.srcdata["_banksy_input"] = features

            src, dst = graph.edges(order="eid")
            if graph.is_block:
                src_ids = graph.srcdata[dgl.NID][src]
                dst_ids = graph.dstdata[dgl.NID][dst]
            else:
                src_ids, dst_ids = src, dst
            synthetic_self = src_ids == dst_ids

            def weighted_sum(edge_key: str, output_key: str) -> torch.Tensor:
                edge_values = graph.edata[edge_key].to(features.dtype)
                edge_values = edge_values.masked_fill(synthetic_self, 0).unsqueeze(-1)
                graph.edata["_banksy_edge_value"] = edge_values
                graph.update_all(
                    dgl_fn.u_mul_e(
                        "_banksy_input",
                        "_banksy_edge_value",
                        "_banksy_message",
                    ),
                    dgl_fn.sum("_banksy_message", output_key),
                )
                return graph.dstdata[output_key]

            m0 = weighted_sum("banksy_m0_weight", "_banksy_m0")
            m1_real = weighted_sum("banksy_m1_real", "_banksy_m1_real")
            m1_imag = weighted_sum("banksy_m1_imag", "_banksy_m1_imag")
            support_sum = weighted_sum("banksy_m1_support", "_banksy_support_sum")

            support = graph.edata["banksy_m1_support"].to(features.dtype)
            support = support.masked_fill(synthetic_self, 0).unsqueeze(-1)
            graph.edata["_banksy_support"] = support
            graph.update_all(
                dgl_fn.copy_e("_banksy_support", "_banksy_support_message"),
                dgl_fn.sum("_banksy_support_message", "_banksy_support_count"),
            )
            count = graph.dstdata["_banksy_support_count"].clamp_min(1)
            neighbor_mean = support_sum / count

            def edge_sum(edge_key: str, output_key: str) -> torch.Tensor:
                edge_values = graph.edata[edge_key].to(features.dtype)
                edge_values = edge_values.masked_fill(synthetic_self, 0).unsqueeze(-1)
                graph.edata["_banksy_edge_sum_input"] = edge_values
                graph.update_all(
                    dgl_fn.copy_e(
                        "_banksy_edge_sum_input",
                        "_banksy_edge_sum_message",
                    ),
                    dgl_fn.sum("_banksy_edge_sum_message", output_key),
                )
                return graph.dstdata[output_key]

            sum_real = edge_sum("banksy_m1_real", "_banksy_sum_real")
            sum_imag = edge_sum("banksy_m1_imag", "_banksy_sum_imag")
            centered_real = m1_real - sum_real * neighbor_mean
            centered_imag = m1_imag - sum_imag * neighbor_mean
            m1 = torch.sqrt(centered_real.square() + centered_imag.square())

            n_dst = graph.num_dst_nodes() if graph.is_block else graph.num_nodes()
            own = features[:n_dst]
            return own, m0, m1

    def _aggregate(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        branches = self._branches(graph, features)
        return torch.cat(
            [
                math.sqrt(scale) * self._zscore(branch)
                for scale, branch in zip(self.scale_squared, branches)
            ],
            dim=-1,
        )

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        return self._aggregate(graph, features)

    def batch_forward(
        self,
        blocks: list[dgl.DGLGraph],
        features: torch.Tensor,
    ) -> torch.Tensor:
        if len(blocks) != 1:
            raise ValueError("BANKSY default kernel requires exactly one graph layer")
        return self._aggregate(blocks[0], features)


class BanksyM1Kernel(BanksyDefaultKernel):
    """Apply only BANKSY's centered m=1 angular-moment magnitude."""

    def _aggregate(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        _, _, m1 = self._branches(graph, features)
        return self._zscore(m1)


class BanksyDualReadout(torch.nn.Module):
    """Dispatch intrinsic and BANKSY-concatenated inputs to compatible readouts."""

    def __init__(
        self,
        intrinsic: Readout,
        *,
        hidden_dim: int,
        module_dim: int,
    ) -> None:
        super().__init__()
        self.intrinsic = intrinsic
        self.spatial = Readout(
            3 * hidden_dim,
            module_dim,
            variational=intrinsic.variational,
        )
        self.variational = intrinsic.variational
        self.hidden_dim = hidden_dim
        self._active = intrinsic

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] == self.hidden_dim:
            self._active = self.intrinsic
        elif features.shape[-1] == 3 * self.hidden_dim:
            self._active = self.spatial
        else:
            raise ValueError(f"unexpected readout input width: {features.shape[-1]}")
        return self._active(features)

    def kl_loss(self) -> torch.Tensor:
        return self._active.kl_loss()

    def clear(self) -> None:
        self.intrinsic.clear()
        self.spatial.clear()


def graph_summary(
    adata: ad.AnnData,
    *,
    label_keys: tuple[str, ...],
    max_neighbors: int,
) -> dict[str, object]:
    graph = generate_adj(adata, edge_clip=None, max_neighbors=max_neighbors)
    src, dst = graph.edges()
    nonself = src != dst
    src = src[nonself].numpy()
    dst = dst[nonself].numpy()
    summary = {
        "nodes": int(graph.num_nodes()),
        "spatial_edges_excluding_self": int(len(src)),
    }
    homophily = {}
    same_label_edges = {}
    for label_key in label_keys:
        labels = adata.obs[label_key].astype(str).to_numpy()
        same = labels[src] == labels[dst]
        same_label_edges[label_key] = int(same.sum())
        homophily[label_key] = float(same.mean()) if len(same) else 1.0
    summary["same_label_spatial_edges"] = same_label_edges
    summary["spatial_edge_label_homophily"] = homophily
    return summary


def cal_pas(labels: np.ndarray, locations: np.ndarray, k: int = 10) -> float:
    if len(labels) < 2:
        return float("nan")
    n_neighbors = min(k + 1, len(labels))
    tree = KDTree(locations)
    _, indices = tree.query(locations, k=n_neighbors)
    neighbors = labels[indices[:, 1:]]
    return float(np.mean(neighbors == labels[:, None]))


def cal_chaos(labels: np.ndarray, locations: np.ndarray) -> float:
    scaled = StandardScaler().fit_transform(locations)
    total = 0.0
    for cluster in np.unique(labels):
        points = scaled[labels == cluster]
        if len(points) <= 2:
            continue
        distances, _ = KDTree(points).query(points, k=2)
        total += float(distances[:, 1].sum())
    return total / len(labels)


def evaluate(
    adata: ad.AnnData,
    *,
    batch_key: str,
    label_key: str,
    prediction_key: str,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    reference = adata.obs[label_key].astype(str).to_numpy()
    prediction = adata.obs[prediction_key].astype(str).to_numpy()
    coords = adata.obs[["x", "y"]].to_numpy(dtype=float)
    valid = ~np.isin(reference, tuple(IGNORE_LABEL_VALUES))

    rows = []
    for section in adata.obs[batch_key].cat.categories:
        section_mask = adata.obs[batch_key].astype(str).eq(str(section)).to_numpy()
        mask = section_mask & valid
        rows.append(
            {
                "section": str(section),
                "label_key": label_key,
                "n_total": int(section_mask.sum()),
                "n_cells": int(mask.sum()),
                "n_reference_domains": int(np.unique(reference[mask]).size),
                "ari": float(adjusted_rand_score(reference[mask], prediction[mask])),
                "nmi": float(
                    normalized_mutual_info_score(reference[mask], prediction[mask])
                ),
                "pas": cal_pas(prediction[mask], coords[mask]),
                "chaos": cal_chaos(prediction[mask], coords[mask]),
            }
        )
    section_metrics = pd.DataFrame(rows)
    weights = section_metrics["n_cells"].to_numpy(dtype=float)
    pooled = {
        "n_total": int(adata.n_obs),
        "n_cells": int(valid.sum()),
        "n_reference_domains": int(np.unique(reference[valid]).size),
        "ari": float(adjusted_rand_score(reference[valid], prediction[valid])),
        "nmi": float(normalized_mutual_info_score(reference[valid], prediction[valid])),
        "pas_section_weighted": float(
            np.average(section_metrics["pas"], weights=weights)
        ),
        "chaos_section_weighted": float(
            np.average(section_metrics["chaos"], weights=weights)
        ),
        "ari_section_weighted": float(
            np.average(section_metrics["ari"], weights=weights)
        ),
        "nmi_section_weighted": float(
            np.average(section_metrics["nmi"], weights=weights)
        ),
    }
    return pooled, section_metrics


def cluster_embedding(
    embedding: np.ndarray,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    if n_clusters <= 50:
        clusterer = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        name = "KMeans"
    else:
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
            batch_size=4096,
        )
        name = "MiniBatchKMeans"
    return clusterer.fit_predict(embedding), name


def run_one(args: argparse.Namespace) -> None:
    if args.mode is None or args.n_sections is None:
        raise ValueError("--mode and --n-sections are required for a single run")
    if (
        args.mode
        in {
            *FIXED_KERNEL_MODES,
            GAUSSIAN_KERNEL_SAMPLED_MODE,
            M1_KERNEL_SAMPLED_MODE,
        }
        and args.n_sections != 1
    ):
        raise ValueError("BANKSY fixed-kernel control supports one section")

    sections = list(args.sections[: args.n_sections])
    run_dir = args.output_dir / f"n{args.n_sections}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "metrics.json"
    if result_path.exists() and not args.force:
        print(f"Skipping completed run: {run_dir}", flush=True)
        return

    set_seed(args.seed)
    if torch.cuda.is_available():
        configure_torch_cuda_workarounds()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    adata = load_subset(args.input, sections, args.batch_key, LABEL_KEYS)
    load_seconds = time.perf_counter() - load_start

    section_graphs = {}
    for section in sections:
        section_adata = adata[adata.obs[args.batch_key].astype(str).eq(section)]
        section_graphs[section] = graph_summary(
            section_adata,
            label_keys=LABEL_KEYS,
            max_neighbors=args.max_neighbors,
        )

    init_start = time.perf_counter()
    model = stModel(
        adata=adata,
        n_top_genes=None,
        geneset_to_use=adata.var_names.to_list(),
        batch_key=args.batch_key,
        edge_clip=None,
        max_neighbors=args.max_neighbors,
        filtered=True,
        log_transformed=True,
        coord_keys=("x", "y"),
    )
    graph_label_key = {
        "oracle_major": "major_brain_region",
        "oracle_ccf": "ccf_region_name",
        "oracle_major_fullneighbors": "major_brain_region",
        "oracle_ccf_fullneighbors": "ccf_region_name",
    }.get(args.mode)
    if args.mode == M1_KERNEL_SAMPLED_MODE:
        model._functional.graph_ops._gener_graph = lambda x: banksy_default_graph(
            x,
            max_neighbors=args.max_neighbors,
        )
        model.model.smoother = BanksyM1Kernel()
        model.model.smoother_type = "BANKSYM1Kernel"
        model.model.gargs["n_layers"] = 1
    elif args.mode == GAUSSIAN_KERNEL_SAMPLED_MODE:
        model._functional.graph_ops._gener_graph = lambda x: banksy_gaussian_graph(
            x,
            max_neighbors=args.max_neighbors,
        )
        model.model.smoother = BanksyGaussianKernel()
        model.model.smoother_type = "BANKSYGaussianKernel"
        model.model.gargs["n_layers"] = 1
    elif args.mode in FIXED_KERNEL_MODES:
        model._functional.graph_ops._gener_graph = lambda x: banksy_default_graph(
            x,
            max_neighbors=args.max_neighbors,
        )
        model.model.smoother = BanksyDefaultKernel()
        model.model.readout = BanksyDualReadout(
            model.model.readout,
            hidden_dim=model.model.hidden_dim,
            module_dim=model.model.module_dim,
        )
        model.model.smoother_type = "BANKSYDefaultKernel"
        model.model.gargs["n_layers"] = 1
    elif graph_label_key is not None:
        model._functional.graph_ops._gener_graph = lambda x: annotation_induced_graph(
            x,
            label_key=graph_label_key,
            max_neighbors=args.max_neighbors,
        )
    init_seconds = time.perf_counter() - init_start

    section_sizes = (
        adata.obs.groupby(args.batch_key, observed=True)
        .size()
        .reindex(sections)
        .to_numpy()
    )
    sampled_iterations = args.sampled_iterations
    if sampled_iterations is None:
        sampled_iterations = args.equivalent_epochs * int(
            sum(math.ceil(int(size) / args.sample_size) for size in section_sizes)
        )
    if args.mode in {
        "sampled",
        FIXED_KERNEL_SAMPLED_MODE,
        GAUSSIAN_KERNEL_SAMPLED_MODE,
        M1_KERNEL_SAMPLED_MODE,
    }:
        sampling = "saint"
    elif args.mode in ("full", "oracle_major", "oracle_ccf"):
        sampling = "full"
    else:
        sampling = "fullneighbors"

    sampled_node_inclusions = (
        args.sampled_node_inclusions
        if args.sampled_node_inclusions is not None
        else sampled_iterations * args.sample_size
    )
    full_graph_epochs = args.equivalent_epochs
    if sampling in {"full", "fullneighbors"} and args.match_node_exposure:
        full_graph_epochs = max(
            1,
            round(sampled_node_inclusions / int(section_sizes.sum())),
        )
    fullneighbor_steps_per_epoch = math.ceil(
        int(section_sizes.sum()) / args.fullneighbor_batch_size
    )
    fullneighbor_epochs = full_graph_epochs
    fullneighbor_block_steps = fullneighbor_epochs * fullneighbor_steps_per_epoch
    training_iterations = sampled_iterations
    training_batch_size = (
        args.fullneighbor_batch_size
        if sampling == "fullneighbors"
        else args.sample_size
    )
    optimizer_updates = {
        "saint": sampled_iterations,
        "full": full_graph_epochs,
        "fullneighbors": fullneighbor_epochs,
    }[sampling]
    fullneighbor_node_exposures = fullneighbor_epochs * int(section_sizes.sum())
    estimated_node_exposures = {
        "saint": sampled_node_inclusions,
        "full": full_graph_epochs * int(section_sizes.sum()),
        "fullneighbors": fullneighbor_node_exposures,
    }[sampling]

    run_start = time.perf_counter()
    model.run(
        sampling=sampling,
        n_iterations=training_iterations,
        n_samples=training_batch_size,
        sample_rate=args.sample_size,
        graph_batch_size=1,
        smooth_epochs=full_graph_epochs,
        full_graph_encoder_batch_size=args.full_graph_encoder_batch_size,
        contrast=args.contrast,
        batch_inference=True,
        inference_batch_size=args.inference_batch_size,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_inference_seconds = time.perf_counter() - run_start

    eval_start = time.perf_counter()
    embedding = np.asarray(model.adata.obsm["X_smoothed"])
    np.save(run_dir / "embedding.npy", embedding.astype(np.float32, copy=False))
    pooled = {}
    clustering = {}
    section_frames = []
    for label_key in LABEL_KEYS:
        suffix = "major" if label_key == "major_brain_region" else "ccf"
        prediction_key = f"domain_{suffix}"
        prediction, clusterer_name = cluster_embedding(
            embedding,
            n_clusters=model.adata.obs.loc[
                ~model.adata.obs[label_key].astype(str).isin(IGNORE_LABEL_VALUES),
                label_key,
            ].nunique(),
            seed=args.seed,
        )
        model.adata.obs[prediction_key] = pd.Categorical(prediction.astype(str))
        label_metrics, label_sections = evaluate(
            model.adata,
            batch_key=args.batch_key,
            label_key=label_key,
            prediction_key=prediction_key,
        )
        pooled[label_key] = label_metrics
        clustering[label_key] = clusterer_name
        section_frames.append(label_sections)
    section_metrics = pd.concat(section_frames, ignore_index=True)
    evaluation_seconds = time.perf_counter() - eval_start

    assignments = model.adata.obs[
        [args.batch_key, *LABEL_KEYS, "domain_major", "domain_ccf", "x", "y"]
    ].copy()
    assignments.to_csv(run_dir / "assignments.csv.gz", compression="gzip")
    section_metrics.to_csv(run_dir / "section_metrics.csv", index=False)

    result = {
        "method": (
            "step_banksy_m1_kernel"
            if args.mode == M1_KERNEL_SAMPLED_MODE
            else "step_banksy_gaussian_kernel"
            if args.mode == GAUSSIAN_KERNEL_SAMPLED_MODE
            else
            "step_banksy_fixed_kernel"
            if args.mode in FIXED_KERNEL_MODES
            else "step"
        ),
        "mode": args.mode,
        "scenario": "single" if args.n_sections == 1 else "multiple",
        "n_sections": args.n_sections,
        "sections": sections,
        "source": str(args.input),
        "input_semantics": "source X; log-normalized floating-point expression",
        "evaluation_label_keys": list(LABEL_KEYS),
        "ignored_evaluation_label_values": sorted(IGNORE_LABEL_VALUES),
        "graph_label_key": graph_label_key,
        "graph_definition": (
            "BANKSY centered m=1 AGF magnitude over 40 neighbors"
            if args.mode == M1_KERNEL_SAMPLED_MODE
            else "BANKSY row-normalized scaled-Gaussian 20-NN fixed kernel"
            if args.mode == GAUSSIAN_KERNEL_SAMPLED_MODE
            else
            (
                "BANKSY default self, m=0, and m=1 AGF branches with "
                f"scaled-Gaussian {args.max_neighbors}/{2 * args.max_neighbors}-NN "
                "fixed kernels and lambda=0.2"
            )
            if args.mode in FIXED_KERNEL_MODES
            else f"section-wise spatial {args.max_neighbors}-NN graph"
            if graph_label_key is None
            else (
                f"section-wise spatial {args.max_neighbors}-NN graph restricted "
                f"to same-{graph_label_key} edges"
            )
        ),
        "training": {
            "sampling": sampling,
            "execution": (
                "full graph (chunked full neighborhoods)"
                if sampling == "fullneighbors"
                else "whole graph"
                if sampling == "full"
                else "GraphSAINT sampled subgraphs"
            ),
            "contrast": args.contrast,
            "sample_size": args.sample_size,
            "fullneighbor_batch_size": args.fullneighbor_batch_size,
            "full_graph_encoder_batch_size": args.full_graph_encoder_batch_size,
            "sampled_iteration_setting": args.sampled_iterations,
            "sampled_iterations": sampled_iterations if sampling == "saint" else None,
            "full_graph_epochs": full_graph_epochs if sampling == "full" else None,
            "fullneighbor_steps_per_epoch": (
                fullneighbor_steps_per_epoch if sampling == "fullneighbors" else None
            ),
            "fullneighbor_epochs": (
                fullneighbor_epochs if sampling == "fullneighbors" else None
            ),
            "fullneighbor_block_steps": (
                fullneighbor_block_steps if sampling == "fullneighbors" else None
            ),
            "optimizer_step_scope": (
                "all target nodes in one full epoch"
                if sampling == "fullneighbors"
                else None
            ),
            "match_node_exposure": args.match_node_exposure,
            "sampled_node_inclusions": sampled_node_inclusions,
            "optimizer_updates": optimizer_updates,
            "gradient_accumulation": sampling == "fullneighbors",
            "estimated_node_exposures": estimated_node_exposures,
            "spatial_smoother": model.model.smoother_type,
            "trainable_smoother_parameters": sum(
                parameter.numel()
                for parameter in model.model.smoother.parameters()
                if parameter.requires_grad
            ),
            "seed": args.seed,
        },
        "clustering": clustering,
        "graph_summary": section_graphs,
        "metrics": pooled,
        "timing_seconds": {
            "load": load_seconds,
            "model_init": init_seconds,
            "training_and_inference": training_inference_seconds,
            "evaluation": evaluation_seconds,
            "total": time.perf_counter() - total_start,
        },
        "resource": {
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
            "peak_cuda_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
            "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 2**20,
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    del model, adata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.glob("n*_*")):
        metrics_path = path / "metrics.json"
        if not metrics_path.exists():
            continue
        record = json.loads(metrics_path.read_text())
        rows.append(
            {
                "n_sections": record["n_sections"],
                "scenario": record["scenario"],
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
    frame = pd.DataFrame(rows).sort_values(["n_sections", "mode"])
    frame.to_csv(output_dir / "benchmark_summary.csv", index=False)
    return frame


def run_matrix(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_modes = (
        WHOLE_GRAPH_MODES if args.matrix_profile == "whole" else FULLNEIGHBOR_MODES
    )
    for n_sections in range(1, 6):
        for mode in matrix_modes:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--input",
                str(args.input),
                "--output-dir",
                str(args.output_dir),
                "--mode",
                mode,
                "--n-sections",
                str(n_sections),
                "--batch-key",
                args.batch_key,
                "--max-neighbors",
                str(args.max_neighbors),
                "--sample-size",
                str(args.sample_size),
                "--fullneighbor-batch-size",
                str(args.fullneighbor_batch_size),
                "--equivalent-epochs",
                str(args.equivalent_epochs),
                "--seed",
                str(args.seed),
                "--inference-batch-size",
                str(args.inference_batch_size),
                "--sections",
                *args.sections,
            ]
            command.append("--contrast" if args.contrast else "--no-contrast")
            if args.sampled_iterations is not None:
                command.extend(["--sampled-iterations", str(args.sampled_iterations)])
            if args.full_graph_encoder_batch_size is not None:
                command.extend(
                    [
                        "--full-graph-encoder-batch-size",
                        str(args.full_graph_encoder_batch_size),
                    ]
                )
            if args.match_node_exposure:
                command.append("--match-node-exposure")
            if args.sampled_node_inclusions is not None:
                command.extend(
                    [
                        "--sampled-node-inclusions",
                        str(args.sampled_node_inclusions),
                    ]
                )
            if args.force:
                command.append("--force")
            print(f"Running n={n_sections}, mode={mode}", flush=True)
            subprocess.run(command, check=True)
    print(summarize(args.output_dir).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(summarize(args.output_dir).to_string(index=False), flush=True)
    elif args.run_matrix:
        run_matrix(args)
    else:
        run_one(args)
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
