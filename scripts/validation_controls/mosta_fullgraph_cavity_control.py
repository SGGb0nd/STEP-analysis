#!/usr/bin/env python
"""Compare an existing sampled MOSTA result with full-graph STEP."""


import argparse
import gc
import json
import math
import random
import resource
import time
from pathlib import Path
from types import MethodType

import anndata as ad
import dgl
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    jaccard_score,
)
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from torch.utils.checkpoint import checkpoint

from step import stModel


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/mosta_e16_test.h5ad"
DEFAULT_SAMPLED = REPO_ROOT / "results/mosta_e16_step/mosta_e16_step.h5ad"
DEFAULT_OUTPUT = REPO_ROOT / "workflows/mosta_cavity_controls/full_graph"
FOCUS_SECTION = "E16.5_E2S4.MOSTA"
FOCUS_LABEL = "Cavity"
SECTION_ORDER = (
    "E16.5_E2S1.MOSTA",
    "E16.5_E2S4.MOSTA",
    "E16.5_E2S7.MOSTA",
    "E16.5_E2S10.MOSTA",
    "E16.5_E2S13.MOSTA",
)
SAMPLED_ITERATIONS = 2000
SAMPLED_GRAPHS_PER_UPDATE = 2
SAMPLED_NODES_PER_GRAPH = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sampled-result", type=Path, default=DEFAULT_SAMPLED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-result", type=Path)
    parser.add_argument(
        "--section",
        choices=SECTION_ORDER,
        help="Restrict full-graph training and evaluation to one section.",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contrast-chunk-size", type=int, default=256)
    parser.add_argument("--decoder-chunk-size", type=int, default=2048)
    parser.add_argument("--encoder-batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--optimizer-updates", type=int, default=SAMPLED_ITERATIONS)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--n-glayers", type=int, default=2)
    parser.add_argument("--n-dec-hid-layers", type=int, default=2)
    parser.add_argument("--edge-clip", type=float, default=1.5)
    parser.add_argument("--pair-samples", type=int, default=500_000)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_training_checkpoint(
    path: Path,
    model: stModel,
    optimizer: torch.optim.Optimizer,
    completed_updates: int,
    elapsed_seconds: float,
) -> None:
    state = {
        "model": model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "completed_updates": completed_updates,
        "elapsed_seconds": elapsed_seconds,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_checkpoint(
    path: Path,
    model: stModel,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[int, float]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    return int(state["completed_updates"]), float(state["elapsed_seconds"])


def run_optimizer_matched_full_graph_training(
    model: stModel,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> tuple[int, float]:
    """Run one exact five-section full-graph objective per optimizer update."""
    if args.optimizer_updates <= 0:
        raise ValueError("optimizer updates must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint interval must be positive")

    functional = model._functional
    functional.graph_loss.configure(
        e2e=True,
        contrast=True,
        kl_contrast=False,
        beta=1e-3,
        kl_cutoff=None,
    )
    functional.unsupervised_loss.configure(beta=1e-3, kl_cutoff=None)
    functional.model.to(functional.device)
    functional.optimizer = torch.optim.Adam(
        functional.model.parameters(),
        lr=args.learning_rate,
    )
    functional.model._smooth = True

    batched_graph = functional.graph_ops.construct_batched_graph(
        model.adata,
        model.dataset,
    )
    gloader = functional.graph_ops.make_full_graph_loader(
        batched_graph,
        graph_batch_size=1,
    )
    loss_fn = functional.graph_ops.make_full_graph_loss_fn(
        model.dataset,
        functional.graph_loss,
        encoder_batch_size=args.encoder_batch_size,
    )

    completed_updates = 0
    elapsed_before = 0.0
    if checkpoint_path.exists() and not args.force:
        completed_updates, elapsed_before = restore_training_checkpoint(
            checkpoint_path,
            model,
            functional.optimizer,
            functional.device,
        )
    started = time.perf_counter()
    while completed_updates < args.optimizer_updates:
        current = min(
            args.checkpoint_every,
            args.optimizer_updates - completed_updates,
        )
        functional.training.train_graph_loop(
            gloader=gloader,
            loss_fn=loss_fn,
            epochs=current,
            show_progress=True,
            accumulate_gradients=True,
            accumulation_total_weight=batched_graph.num_nodes(),
        )
        completed_updates += current
        elapsed = elapsed_before + time.perf_counter() - started
        save_training_checkpoint(
            checkpoint_path,
            model,
            functional.optimizer,
            completed_updates,
            elapsed,
        )
    return completed_updates, elapsed_before + time.perf_counter() - started


def install_exact_chunked_contrastive_loss(model: stModel, chunk_size: int) -> None:
    """Use the unchanged all-node contrastive objective with bounded logits."""
    if chunk_size <= 0:
        raise ValueError("contrast chunk size must be positive")

    def chunked_loss(loss_self, rep_q: torch.Tensor, rep_k: torch.Tensor):
        q_norm = F.normalize(rep_q)
        k_norm = F.normalize(rep_k).detach()
        n_nodes = q_norm.shape[0]
        total = q_norm.new_zeros(())

        for start in range(0, n_nodes, chunk_size):
            stop = min(start + chunk_size, n_nodes)
            labels = torch.arange(start, stop, device=q_norm.device)

            def block_loss(
                query: torch.Tensor,
                targets: torch.Tensor,
            ) -> torch.Tensor:
                logits = torch.matmul(query, k_norm.T)
                return F.cross_entropy(
                    logits / loss_self.config.contrast_temp,
                    targets,
                    reduction="sum",
                )

            total = total + checkpoint(
                block_loss,
                q_norm[start:stop],
                labels,
                use_reentrant=False,
            )

        return loss_self.config.contrast_weight * total / n_nodes

    graph_loss = model._functional.graph_loss
    graph_loss._contrastive_loss = MethodType(chunked_loss, graph_loss)


def _reconstruction_from_decode(loss_self, decoded, x: torch.Tensor) -> torch.Tensor:
    return loss_self.reconstruction_loss(
        x=x,
        px_rate=decoded["px_rate"],
        px_r=decoded["px_r"],
        decoder_type=decoded["decoder_type"],
        px_dropout=decoded.get("px_dropout"),
        px_scale=decoded.get("px_scale"),
    )


def _full_hidden_chunked_reconstruction(
    loss_self,
    model,
    cls_rep: torch.Tensor,
    x: torch.Tensor,
    batch_rep: torch.Tensor | None,
    chunk_size: int,
    rep_ts: torch.Tensor | None = None,
) -> torch.Tensor:
    decoder = model.decoder
    batch_label = None
    decoder_batch_rep = None
    if batch_rep is not None:
        batch_label = batch_rep.argmax(dim=1)
        decoder_batch_rep = batch_rep @ model.batch_embedding
        if model.batch_readout is not None:
            cls_rep = model.batch_readout(cls_rep, decoder_batch_rep)
    library = x.sum(-1, keepdim=True)
    hidden = decoder.ffn_(
        cls_rep,
        z_=rep_ts,
        batch_rep=decoder_batch_rep,
    )

    total = hidden.new_zeros(())
    n_nodes = len(x)
    for start in range(0, n_nodes, chunk_size):
        stop = min(start + chunk_size, n_nodes)
        if batch_label is None:
            def block_loss(
                hidden_chunk: torch.Tensor,
                library_chunk: torch.Tensor,
                x_chunk: torch.Tensor,
            ):
                px_scale = decoder.px_scale_decoder(hidden_chunk)
                decoded = {
                    "px_rate": library_chunk * px_scale * decoder.get_l_scale(None),
                    "px_dropout": decoder.px_dropout_decoder(hidden_chunk),
                    "px_scale": px_scale,
                    "px_r": model.get_px_r(None),
                    "decoder_type": model.decoder_type,
                }
                return _reconstruction_from_decode(loss_self, decoded, x_chunk)

            block = checkpoint(
                block_loss,
                hidden[start:stop],
                library[start:stop],
                x[start:stop],
                use_reentrant=False,
            )
        else:
            def block_loss(
                hidden_chunk: torch.Tensor,
                library_chunk: torch.Tensor,
                x_chunk: torch.Tensor,
                batch_label_chunk: torch.Tensor,
            ):
                px_scale = decoder.px_scale_decoder(hidden_chunk)
                decoded = {
                    "px_rate": (
                        library_chunk
                        * px_scale
                        * decoder.get_l_scale(batch_label_chunk)
                    ),
                    "px_dropout": decoder.px_dropout_decoder(hidden_chunk),
                    "px_scale": px_scale,
                    "px_r": model.get_px_r(batch_label_chunk),
                    "decoder_type": model.decoder_type,
                }
                return _reconstruction_from_decode(loss_self, decoded, x_chunk)

            block = checkpoint(
                block_loss,
                hidden[start:stop],
                library[start:stop],
                x[start:stop],
                batch_label[start:stop],
                use_reentrant=False,
            )
        total = total + block * ((stop - start) / n_nodes)
    return total


def install_memory_bounded_full_graph_loss(
    model: stModel,
    decoder_chunk_size: int,
) -> None:
    """Checkpoint decoder likelihood blocks without changing the mean loss."""
    if decoder_chunk_size <= 0:
        raise ValueError("decoder chunk size must be positive")

    def compute(loss_self, module, x, batch_rep=None, **kwargs):
        module._smooth = True
        encoder_batch_size = kwargs.get("encoder_batch_size")
        if encoder_batch_size is None:
            rep_ts = module.encode_ts(x, batch_rep)
        elif encoder_batch_size == 0:
            if batch_rep is None:
                rep_ts = checkpoint(
                    module.encode_ts,
                    x,
                    use_reentrant=False,
                )
            else:
                rep_ts = checkpoint(
                    module.encode_ts,
                    x,
                    batch_rep,
                    use_reentrant=False,
                )
        else:
            rep_ts = loss_self._checkpointed_encode_ts(
                module,
                x,
                batch_rep,
                batch_size=encoder_batch_size,
            )
        rep_k = rep_ts.clone().detach()
        rep_q = module.local_smooth(rep_k)
        cls_rep = module.readout(rep_q)
        recon = _full_hidden_chunked_reconstruction(
            loss_self,
            module,
            cls_rep,
            x,
            batch_rep,
            decoder_chunk_size,
        )
        loss_dict = {"recon_loss": recon}

        kl = loss_self.kl_loss(module.readout)
        if kl is not None:
            loss_dict["kl_loss"] = kl
        if loss_self.config.contrast:
            loss_dict["contrast_loss"] = loss_self._contrastive_loss(rep_q, rep_k)

        if loss_self.config.e2e:
            module._smooth = False
            intrinsic_cls = module.readout_(rep_ts)
            loss_dict["recon_loss"] = loss_dict["recon_loss"] + (
                _full_hidden_chunked_reconstruction(
                    loss_self,
                    module,
                    intrinsic_cls,
                    x,
                    batch_rep,
                    decoder_chunk_size,
                    rep_ts=rep_ts,
                )
            )
            kl2 = loss_self.kl_loss(module.readout)
            if kl2 is not None and "kl_loss" in loss_dict:
                loss_dict["kl_loss"] = loss_dict["kl_loss"] + kl2
        return loss_dict

    graph_loss = model._functional.graph_loss
    graph_loss.compute = MethodType(compute, graph_loss)


def validate_chunked_contrastive_loss(model: stModel, chunk_size: int) -> None:
    graph_loss = model._functional.graph_loss
    generator = torch.Generator().manual_seed(17)
    n_nodes = chunk_size + 3
    query_expected = torch.randn(
        n_nodes, 8, generator=generator, requires_grad=True
    )
    query_observed = query_expected.detach().clone().requires_grad_(True)
    key = torch.randn(n_nodes, 8, generator=generator)
    logits = torch.matmul(F.normalize(query_expected), F.normalize(key).T)
    expected = graph_loss.config.contrast_weight * F.cross_entropy(
        logits / graph_loss.config.contrast_temp,
        torch.arange(n_nodes),
    )
    observed = graph_loss._contrastive_loss(query_observed, key)
    if not torch.allclose(observed, expected, atol=1e-6, rtol=1e-6):
        raise AssertionError(
            f"chunked contrast mismatch: {observed.item()} versus {expected.item()}"
        )
    expected.backward()
    observed.backward()
    if not torch.allclose(
        query_observed.grad,
        query_expected.grad,
        atol=1e-6,
        rtol=1e-5,
    ):
        raise AssertionError("chunked contrast gradient does not match direct loss")


def validate_chunked_reconstruction(model: stModel) -> None:
    loss = model._functional.graph_loss
    module = model.model
    n_nodes = 7
    generator = torch.Generator().manual_seed(23)
    cls_expected = torch.randn(
        n_nodes,
        module.module_dim,
        generator=generator,
        requires_grad=True,
    )
    cls_observed = cls_expected.detach().clone().requires_grad_(True)
    x = torch.poisson(
        torch.full((n_nodes, module.input_dim), 1.5),
        generator=generator,
    )
    num_batches = getattr(module, "num_batches", 1)
    batch_rep = None
    if num_batches > 1:
        batch_label = torch.arange(n_nodes) % num_batches
        batch_rep = F.one_hot(
            batch_label,
            num_classes=num_batches,
        ).float()

    direct = module.decode(cls_expected, x, batch_rep=batch_rep)
    expected = _reconstruction_from_decode(loss, direct, x)
    observed = _full_hidden_chunked_reconstruction(
        loss,
        module,
        cls_observed,
        x,
        batch_rep,
        chunk_size=3,
    )
    if not torch.allclose(observed, expected, atol=1e-5, rtol=1e-5):
        raise AssertionError(
            f"chunked reconstruction mismatch: {observed.item()} versus "
            f"{expected.item()}"
        )
    expected_grad = torch.autograd.grad(expected, cls_expected)[0]
    observed_grad = torch.autograd.grad(observed, cls_observed)[0]
    if not torch.allclose(observed_grad, expected_grad, atol=1e-5, rtol=1e-4):
        raise AssertionError("chunked reconstruction gradient does not match direct loss")


def train_full_graph(args: argparse.Namespace) -> Path:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    full_result = args.full_result or output / "mosta_fullgraph_slim.h5ad"
    if full_result.exists() and not args.force:
        return full_result

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    adata = ad.read_h5ad(args.input)
    if args.section is not None:
        section_mask = adata.obs["batch"].astype(str).eq(args.section).to_numpy()
        adata = adata[section_mask].copy()
    genes = adata.var_names.to_list()
    model = stModel(
        adata=adata,
        batch_key="batch",
        coord_keys=("x", "y"),
        layer_key="count",
        log_transformed=False,
        edge_clip=args.edge_clip,
        n_glayers=args.n_glayers,
        n_dec_hid_layers=args.n_dec_hid_layers,
        batch_injection_mode="scale",
        use_batch_readout=True,
        n_top_genes=None,
        geneset_to_use=genes,
    )
    install_exact_chunked_contrastive_loss(model, args.contrast_chunk_size)
    install_memory_bounded_full_graph_loss(model, args.decoder_chunk_size)
    validate_chunked_contrastive_loss(model, args.contrast_chunk_size)
    validate_chunked_reconstruction(model)

    n_cells = model.adata.n_obs
    completed_updates, training_seconds = run_optimizer_matched_full_graph_training(
        model,
        args,
        output / "full_graph_training_checkpoint.pt",
    )
    functional = model._functional
    functional.optimizer.zero_grad(set_to_none=True)
    functional.model.g = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        model.adata.obsm["X_rep"] = functional.graph_ops._embed_without_graph(
            model.dataset
        )
    model.adata.obsm["X_smoothed"] = functional.graph_ops.embed_with_graph(
        model.dataset,
        batch_size=args.inference_batch_size,
    )
    n_clusters = args.n_clusters or int(model.adata.obs["annotation"].nunique())
    model.cluster(n_clusters=n_clusters, seed=args.seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    keep_obs = ["annotation", "batch", "x", "y", "domain"]
    slim = ad.AnnData(obs=model.adata.obs[keep_obs].copy())
    slim.obsm["spatial"] = np.asarray(model.adata.obsm["spatial"]).copy()
    slim.obsm["X_rep"] = np.asarray(model.adata.obsm["X_rep"]).copy()
    slim.obsm["X_smoothed"] = np.asarray(model.adata.obsm["X_smoothed"]).copy()
    slim.write_h5ad(full_result)

    sampled_exposure = (
        SAMPLED_ITERATIONS
        * SAMPLED_GRAPHS_PER_UPDATE
        * SAMPLED_NODES_PER_GRAPH
    )
    full_exposure = completed_updates * n_cells
    run = {
        "input": str(args.input.resolve()),
        "sampled_result_reused": str(args.sampled_result.resolve()),
        "full_result": str(full_result.resolve()),
        "n_cells": n_cells,
        "n_genes": model.adata.n_vars,
        "sections": (
            [args.section] if args.section is not None else list(SECTION_ORDER)
        ),
        "n_clusters": n_clusters,
        "sampled_iterations": SAMPLED_ITERATIONS,
        "sampled_graphs_per_update": SAMPLED_GRAPHS_PER_UPDATE,
        "sampled_nodes_per_graph": SAMPLED_NODES_PER_GRAPH,
        "estimated_sampled_node_exposure": sampled_exposure,
        "matching_axis": "optimizer updates",
        "full_graph_optimizer_updates": completed_updates,
        "full_graph_epochs": completed_updates,
        "full_graph_node_exposure": full_exposure,
        "exposure_ratio_full_over_sampled": full_exposure / sampled_exposure,
        "graph_batch_size": 1,
        "n_glayers": args.n_glayers,
        "n_dec_hid_layers": args.n_dec_hid_layers,
        "edge_clip": args.edge_clip,
        "contrast": True,
        "contrast_evaluation": "exact all-node InfoNCE, memory-bounded logits",
        "contrast_chunk_size": args.contrast_chunk_size,
        "decoder_chunk_size": args.decoder_chunk_size,
        "encoder_batch_size": args.encoder_batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint_every_updates": args.checkpoint_every,
        "training_seconds": training_seconds,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else None
        ),
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024**2,
    }
    write_json(output / "training_summary.json", run)
    return full_result


def assert_same_observations(sampled: ad.AnnData, full: ad.AnnData) -> None:
    if sampled.n_obs != full.n_obs:
        raise AssertionError(
            f"sampled/full observation counts differ: {sampled.n_obs} vs {full.n_obs}"
        )
    for key in ("batch", "annotation", "x", "y"):
        left = sampled.obs[key].astype(str).to_numpy()
        right = full.obs[key].astype(str).to_numpy()
        if not np.array_equal(left, right):
            mismatch = int(np.flatnonzero(left != right)[0])
            raise AssertionError(f"{key} differs first at row {mismatch}")


def cal_chaos(labels: np.ndarray, locations: np.ndarray) -> float:
    scaled = StandardScaler().fit_transform(locations)
    total = 0.0
    for label in np.unique(labels):
        points = scaled[labels == label]
        if len(points) <= 2:
            continue
        distances, _ = KDTree(points).query(points, k=2)
        total += float(distances[:, 1].sum())
    return total / len(labels)


def cal_pas(labels: np.ndarray, locations: np.ndarray, k: int = 10) -> float:
    tree = KDTree(locations)
    _, neighbors = tree.query(locations, k=min(k + 1, len(labels)))
    return float(np.mean(labels[neighbors[:, 1:]] == labels[:, None]))


def section_metrics(
    adata: ad.AnnData,
    method: str,
    sections: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    batches = adata.obs["batch"].astype(str).to_numpy()
    annotations = adata.obs["annotation"].astype(str).to_numpy()
    domains = adata.obs["domain"].astype(str).to_numpy()
    coords = adata.obs[["x", "y"]].to_numpy(dtype=float)
    for section in sections:
        mask = batches == section
        rows.append(
            {
                "method": method,
                "section": section.replace(".MOSTA", ""),
                "n_cells": int(mask.sum()),
                "ARI": adjusted_rand_score(annotations[mask], domains[mask]),
                "NMI": normalized_mutual_info_score(
                    annotations[mask], domains[mask]
                ),
                "PAS": cal_pas(domains[mask], coords[mask]),
                "CHAOS": cal_chaos(domains[mask], coords[mask]),
            }
        )
    return pd.DataFrame(rows)


def hungarian_annotation_map(
    annotations: np.ndarray, domains: np.ndarray
) -> dict[str, str]:
    reference = np.unique(annotations)
    prediction = np.unique(domains)
    matrix = np.zeros((len(prediction), len(reference)), dtype=np.int64)
    reference_index = {value: idx for idx, value in enumerate(reference)}
    prediction_index = {value: idx for idx, value in enumerate(prediction)}
    for pred, ref in zip(domains, annotations):
        matrix[prediction_index[pred], reference_index[ref]] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    return {
        str(prediction[row]): str(reference[col])
        for row, col in zip(row_ind, col_ind)
    }


def radius_adjacency(coords: np.ndarray, radius: float = 1.5):
    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    data = np.ones(len(rows), dtype=np.uint8)
    return coo_matrix((data, (rows, cols)), shape=(len(coords), len(coords))).tocsr()


def component_summary(adjacency, mask: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return {
            "n_cells": 0,
            "n_components": 0,
            "components_per_1000_cells": math.nan,
            "largest_component_cells": 0,
            "largest_component_fraction": math.nan,
        }, np.array([], dtype=int)
    n_components, labels = connected_components(
        adjacency[indices][:, indices], directed=False
    )
    sizes = np.bincount(labels)
    largest_label = int(np.argmax(sizes))
    largest_indices = indices[labels == largest_label]
    return {
        "n_cells": int(len(indices)),
        "n_components": int(n_components),
        "components_per_1000_cells": float(n_components * 1000 / len(indices)),
        "largest_component_cells": int(sizes[largest_label]),
        "largest_component_fraction": float(sizes[largest_label] / len(indices)),
    }, largest_indices


def cavity_metrics(
    adata: ad.AnnData,
    method: str,
    annotation_map: dict[str, str],
    adjacency,
) -> tuple[dict[str, float | str], np.ndarray]:
    annotations = adata.obs["annotation"].astype(str).to_numpy()
    domains = adata.obs["domain"].astype(str).to_numpy()
    truth = annotations == FOCUS_LABEL
    cavity_domains = [key for key, value in annotation_map.items() if value == FOCUS_LABEL]
    if len(cavity_domains) != 1:
        raise AssertionError(f"Expected one mapped Cavity domain, got {cavity_domains}")
    cavity_domain = cavity_domains[0]
    prediction = domains == cavity_domain
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        prediction,
        average="binary",
        zero_division=0,
    )
    component, _ = component_summary(adjacency, prediction)
    return {
        "method": method,
        "mapped_cavity_domain": cavity_domain,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(jaccard_score(truth, prediction, zero_division=0)),
        **component,
    }, prediction


def distance_agreement(
    coords: np.ndarray,
    cavity_lcc: np.ndarray,
    sampled_domains: np.ndarray,
    full_domains: np.ndarray,
    pair_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    left = rng.choice(cavity_lcc, size=pair_samples, replace=True)
    right = rng.choice(cavity_lcc, size=pair_samples, replace=True)
    keep = left != right
    left, right = left[keep], right[keep]
    distances = np.linalg.norm(coords[left] - coords[right], axis=1)
    quantiles = np.unique(np.quantile(distances, np.linspace(0, 1, 6)))
    if len(quantiles) < 3:
        raise AssertionError("Cavity pair distances do not span enough bins")
    bins = np.clip(np.digitize(distances, quantiles[1:-1], right=True), 0, 4)
    rows = []
    for method, domains in (
        ("Sampled", sampled_domains),
        ("Full graph", full_domains),
    ):
        same = domains[left] == domains[right]
        for bin_index in range(len(quantiles) - 1):
            mask = bins == bin_index
            rows.append(
                {
                    "method": method,
                    "distance_bin": bin_index + 1,
                    "distance_min": float(quantiles[bin_index]),
                    "distance_max": float(quantiles[bin_index + 1]),
                    "distance_midpoint": float(np.median(distances[mask])),
                    "n_pairs": int(mask.sum()),
                    "same_domain_fraction": float(same[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_cavity_spatial(
    coords: np.ndarray,
    truth: np.ndarray,
    sampled: np.ndarray,
    full: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)
    color = "#D95F02"
    for ax, title, mask in zip(
        axes,
        ("Ground truth", "Sampled", "Full graph"),
        (truth, sampled, full),
    ):
        ax.scatter(
            coords[:, 0],
            -coords[:, 1],
            c="#D9D9D9",
            s=0.25,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            coords[mask, 0],
            -coords[mask, 1],
            c=color,
            s=0.9,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor="none", markersize=7, label="Cavity"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#D9D9D9",
                   markeredgecolor="none", markersize=7, label="Other tissue"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cavity_quantification(
    cavity: pd.DataFrame,
    distance: pd.DataFrame,
    output: Path,
) -> None:
    colors = {"Sampled": "#4C78A8", "Full graph": "#E45756"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    metrics = ["precision", "recall", "f1", "iou", "largest_component_fraction"]
    labels = ["Precision", "Recall", "F1", "IoU", "LCC fraction"]
    x = np.arange(len(metrics))
    width = 0.36
    for offset, method in zip((-width / 2, width / 2), ("Sampled", "Full graph")):
        row = cavity.loc[cavity["method"] == method].iloc[0]
        axes[0].bar(
            x + offset,
            [row[metric] for metric in metrics],
            width,
            color=colors[method],
            label=method,
        )
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Score")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)

    for method in ("Sampled", "Full graph"):
        frame = distance.loc[distance["method"] == method]
        axes[1].plot(
            frame["distance_bin"],
            frame["same_domain_fraction"],
            marker="o",
            linewidth=2,
            color=colors[method],
            label=method,
        )
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xlabel("Within-Cavity distance quintile")
    axes[1].set_ylabel("Same-domain fraction")
    axes[1].set_ylim(0, 1.02)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_section_metrics(metrics: pd.DataFrame, output: Path) -> None:
    colors = {"Sampled": "#4C78A8", "Full graph": "#E45756"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    section_labels = [section.replace("E16.5_", "") for section in metrics["section"].unique()]
    for ax, metric in zip(axes.flat, ("ARI", "NMI", "PAS", "CHAOS")):
        for method in ("Sampled", "Full graph"):
            frame = metrics.loc[metrics["method"] == method]
            ax.plot(
                np.arange(len(frame)),
                frame[metric],
                marker="o",
                linewidth=2,
                color=colors[method],
                label=method,
            )
        ax.set_title(metric)
        ax.set_xticks(np.arange(len(section_labels)), section_labels)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_figure_legends(output: Path, focus_section: str) -> None:
    section_label = focus_section.replace(".MOSTA", "")
    text = f"""# Figure legends

**MOSTA Cavity spatial comparison.** Ground-truth Cavity annotation and the domains mapped to Cavity by maximum-overlap assignment for the existing GraphSAINT-sampled STEP result and full-graph STEP in section {section_label}. Orange indicates Cavity; gray indicates all other tissue positions.

**MOSTA Cavity quantification.** Left, precision, recall, F1 score, intersection over union (IoU), and largest-connected-component (LCC) fraction for the predicted domain mapped to the Cavity annotation. Right, fraction of cell pairs assigned to the same spatial domain across quintiles of Euclidean separation within the largest connected component of the ground-truth Cavity region. The same cell pairs were used for sampled and full-graph STEP.

**MOSTA spatial-domain performance.** Per-section ARI, NMI, PAS, and CHAOS for the existing GraphSAINT-sampled STEP result and optimizer-update-matched full-graph STEP. Higher values indicate better performance for ARI, NMI, and PAS; lower values indicate better spatial compactness for CHAOS.
"""
    output.write_text(text)


def evaluate(args: argparse.Namespace, full_result: Path) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sampled = ad.read_h5ad(args.sampled_result)
    full = ad.read_h5ad(full_result)
    sections = (args.section,) if args.section is not None else SECTION_ORDER
    if args.section is not None:
        sampled = sampled[
            sampled.obs["batch"].astype(str).eq(args.section).to_numpy()
        ].copy()
    assert_same_observations(sampled, full)

    metrics = pd.concat(
        [
            section_metrics(sampled, "Sampled", sections),
            section_metrics(full, "Full graph", sections),
        ],
        ignore_index=True,
    )
    metrics.to_csv(output / "section_metrics.csv", index=False)

    annotations = sampled.obs["annotation"].astype(str).to_numpy()
    sampled_domains = sampled.obs["domain"].astype(str).to_numpy()
    full_domains = full.obs["domain"].astype(str).to_numpy()
    sampled_map = hungarian_annotation_map(annotations, sampled_domains)
    full_map = hungarian_annotation_map(annotations, full_domains)

    focus_section = args.section or FOCUS_SECTION
    section_mask = sampled.obs["batch"].astype(str).eq(focus_section).to_numpy()
    sampled_section = sampled[section_mask].copy()
    full_section = full[section_mask].copy()
    coords = sampled_section.obs[["x", "y"]].to_numpy(dtype=float)
    adjacency = radius_adjacency(coords, radius=args.edge_clip)
    truth = sampled_section.obs["annotation"].astype(str).eq(FOCUS_LABEL).to_numpy()
    truth_components, truth_lcc = component_summary(adjacency, truth)
    sampled_cavity, sampled_prediction = cavity_metrics(
        sampled_section,
        "Sampled",
        sampled_map,
        adjacency,
    )
    full_cavity, full_prediction = cavity_metrics(
        full_section,
        "Full graph",
        full_map,
        adjacency,
    )
    cavity = pd.DataFrame([sampled_cavity, full_cavity])
    cavity.to_csv(output / "cavity_metrics.csv", index=False)

    distance = distance_agreement(
        coords,
        truth_lcc,
        sampled_section.obs["domain"].astype(str).to_numpy(),
        full_section.obs["domain"].astype(str).to_numpy(),
        pair_samples=args.pair_samples,
        seed=args.seed,
    )
    distance.to_csv(output / "cavity_distance_agreement.csv", index=False)

    plot_cavity_spatial(
        coords,
        truth,
        sampled_prediction,
        full_prediction,
        output / "mosta_cavity_spatial_comparison",
    )
    plot_cavity_quantification(
        cavity,
        distance,
        output / "mosta_cavity_quantification",
    )
    plot_section_metrics(metrics, output / "mosta_fullgraph_section_metrics")
    write_figure_legends(output / "figure_legends.md", focus_section)

    summary = {
        "sampled_result": str(args.sampled_result.resolve()),
        "full_result": str(full_result.resolve()),
        "focus_section": focus_section,
        "focus_label": FOCUS_LABEL,
        "truth_connectivity": truth_components,
        "sampled_cavity": sampled_cavity,
        "full_graph_cavity": full_cavity,
        "sampled_annotation_map": sampled_map,
        "full_graph_annotation_map": full_map,
        "distance_agreement_farthest_quintile": {
            method: float(
                distance.loc[
                    (distance["method"] == method)
                    & (distance["distance_bin"] == distance["distance_bin"].max()),
                    "same_domain_fraction",
                ].iloc[0]
            )
            for method in ("Sampled", "Full graph")
        },
    }
    write_json(output / "evaluation_summary.json", summary)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        if args.full_result is None:
            raise ValueError("--evaluate-only requires --full-result")
        full_result = args.full_result
    else:
        full_result = train_full_graph(args)
    evaluate(args, full_result)


if __name__ == "__main__":
    main()
