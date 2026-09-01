#!/usr/bin/env python
"""Run exact full-section STEP controls on prostate or CRC data."""


import argparse
import gc
import json
import random
import resource
import time
from pathlib import Path

import anndata as ad
import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from step import stModel
from mosta_fullgraph_cavity_control import (
    install_exact_chunked_contrastive_loss,
    install_memory_bounded_full_graph_loss,
    restore_training_checkpoint,
    save_training_checkpoint,
    validate_chunked_contrastive_loss,
    validate_chunked_reconstruction,
)


PROSTATE_INPUT = Path("results/slide-seq/prostate_slideseq_step.h5ad")
CRC_INPUT = Path("results/visium-hd/crc_8um_5slices_domain.h5ad")
CRC_SECTIONS = (
    "cancer_p1",
    "cancer_p2",
    "cancer_p5",
    "normal_p3",
    "normal_p5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("prostate", "crc"), required=True)
    parser.add_argument("--section", choices=CRC_SECTIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer-updates", type=int, default=200)
    parser.add_argument("--target-batch-size", type=int, default=2048)
    parser.add_argument(
        "--all-section-targets",
        action="store_true",
        help="Use every node in the randomly selected section as the loss batch.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("matched-batch", "all-node"),
        default="matched-batch",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-batch-size", type=int, default=1024)
    parser.add_argument("--decoder-chunk-size", type=int, default=2048)
    parser.add_argument("--contrast-chunk-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args: argparse.Namespace) -> tuple[ad.AnnData, dict[str, object]]:
    path = PROSTATE_INPUT if args.dataset == "prostate" else CRC_INPUT
    source = ad.read_h5ad(path, backed="r")
    if args.dataset == "crc" and args.section is not None:
        mask = source.obs["batch"].astype(str).eq(args.section).to_numpy()
        adata = source[mask].to_memory()
    else:
        if args.dataset == "prostate" and args.section is not None:
            raise ValueError("--section applies only to CRC")
        adata = source.to_memory()
    source.file.close()
    adata.var_names_make_unique()
    return adata, {
        "source": str(path),
        "dataset": args.dataset,
        "section": args.section,
    }


def build_model(args: argparse.Namespace, adata: ad.AnnData) -> tuple[stModel, float]:
    common = dict(
        adata=adata,
        n_top_genes=None,
        geneset_to_use=adata.var_names.to_list(),
        layer_key="counts",
        filtered=True,
        log_transformed=True,
    )
    if args.dataset == "prostate":
        model = stModel(
            **common,
            batch_key="sample",
            coord_keys=("x", "y"),
            decoder_type="zinb",
            dispersion="gene",
            edge_clip=None,
            max_neighbors=20,
            n_glayers=4,
            variational=True,
        )
        beta = 1e-2
    else:
        model = stModel(
            **common,
            batch_key="batch",
            coord_keys=("array_row", "array_col"),
            edge_clip=1,
            n_glayers=3,
        )
        beta = 1e-3
    return model, beta


def train_all_node_full_graph(
    model: stModel,
    args: argparse.Namespace,
    beta: float,
) -> tuple[int, float]:
    functional = model._functional
    functional.graph_loss.configure(
        e2e=True,
        contrast=True,
        kl_contrast=False,
        beta=beta,
        kl_cutoff=None,
    )
    functional.unsupervised_loss.configure(beta=beta, kl_cutoff=None)
    functional.model.to(functional.device)
    functional.optimizer = torch.optim.Adam(
        functional.model.parameters(), lr=args.learning_rate
    )
    functional.model._smooth = True

    graph = functional.graph_ops.construct_batched_graph(model.adata, model.dataset)
    loader = functional.graph_ops.make_full_graph_loader(graph, graph_batch_size=1)
    loss_fn = functional.graph_ops.make_full_graph_loss_fn(
        model.dataset,
        functional.graph_loss,
        encoder_batch_size=args.encoder_batch_size,
    )
    checkpoint = args.output_dir / "full_graph_training_checkpoint.pt"
    completed = 0
    elapsed_before = 0.0
    if checkpoint.exists() and not args.force:
        completed, elapsed_before = restore_training_checkpoint(
            checkpoint,
            model,
            functional.optimizer,
            functional.device,
        )
    started = time.perf_counter()
    while completed < args.optimizer_updates:
        count = min(args.checkpoint_every, args.optimizer_updates - completed)
        for _ in range(count):
            functional.model.train()
            functional.optimizer.zero_grad(set_to_none=True)
            for graph_batch in loader:
                loss_dict = loss_fn(graph_batch)
                loss_dict.pop("graph_ids", None)
                loss = sum(
                    value
                    for value in loss_dict.values()
                    if isinstance(value, torch.Tensor)
                )
                weight = graph_batch.num_nodes() / graph.num_nodes()
                (loss * weight).backward()
                functional.model.g = None
                del loss, loss_dict, graph_batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            functional.optimizer.step()
            completed += 1
        elapsed = elapsed_before + time.perf_counter() - started
        save_training_checkpoint(
            checkpoint,
            model,
            functional.optimizer,
            completed,
            elapsed,
        )
        print(
            f"completed full-graph updates: {completed}/{args.optimizer_updates}",
            flush=True,
        )
    return completed, elapsed_before + time.perf_counter() - started


def _matched_fullneighbor_loss(
    functional,
    dataset,
    batched_graph: dgl.DGLGraph,
    graph_to_dataset: torch.Tensor,
    sampler,
    target_nodes: torch.Tensor,
    encoder_batch_size: int,
) -> tuple[dict[str, torch.Tensor], int]:
    input_nodes, output_nodes, blocks = sampler.sample_blocks(
        batched_graph,
        target_nodes,
    )
    n_output = int(output_nodes.numel())
    if not torch.equal(input_nodes[:n_output], output_nodes):
        raise RuntimeError("DGL full-neighbor input/output nodes are not aligned")

    dataset_nodes = graph_to_dataset[input_nodes]
    x_inp = dataset.gene_expr[dataset_nodes].clone().to(functional.device)
    x_out = x_inp[:n_output]
    output_mask = torch.zeros(len(input_nodes), dtype=torch.bool)
    output_mask[:n_output] = True

    batch_rep = None
    batch_rep_out = None
    if functional._num_batches > 1:
        batch_label = dataset.batch_label[dataset_nodes].clone()
        batch_rep = F.one_hot(
            batch_label.long(), num_classes=functional._num_batches
        ).float().to(functional.device)
        batch_rep_out = batch_rep[:n_output]

    device_blocks = [block.to(functional.device) for block in blocks]
    functional.model._smooth = True
    loss_dict = functional.graph_loss.compute_blocks(
        functional.model,
        device_blocks,
        x_inp,
        x_out,
        output_mask,
        batch_rep=batch_rep,
        batch_rep_out=batch_rep_out,
        encoder_batch_size=encoder_batch_size,
    )
    return loss_dict, int(input_nodes.numel())


def train_matched_batch_full_graph(
    model: stModel,
    args: argparse.Namespace,
    beta: float,
) -> tuple[int, float]:
    """Train exact full-neighbor targets with the sampled run's loss batch."""
    if not args.all_section_targets and args.target_batch_size <= 0:
        raise ValueError("target batch size must be positive")

    functional = model._functional
    functional.graph_loss.configure(
        e2e=True,
        contrast=True,
        kl_contrast=False,
        beta=beta,
        kl_cutoff=None,
    )
    functional.unsupervised_loss.configure(beta=beta, kl_cutoff=None)
    functional.model.to(functional.device)
    functional.optimizer = torch.optim.Adam(
        functional.model.parameters(), lr=args.learning_rate
    )

    batched_graph = functional.graph_ops.construct_batched_graph(
        model.adata,
        model.dataset,
    )
    graph_to_dataset = batched_graph.ndata["node_ids"].cpu()
    components = dgl.unbatch(batched_graph)
    component_offsets = []
    component_degrees = []
    offset = 0
    for component in components:
        component_offsets.append(offset)
        component_degrees.append(component.out_degrees().float().clamp(min=1))
        offset += component.num_nodes()

    n_layers = int(functional.model.gargs["n_layers"])
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    batch_key = "sample" if args.dataset == "prostate" else "batch"
    section_names = list(model.adata.obs[batch_key].cat.categories)
    checkpoint = args.output_dir / "matched_batch_training_checkpoint.pt"
    history_path = args.output_dir / "matched_batch_training.jsonl"

    completed = 0
    elapsed_before = 0.0
    if checkpoint.exists() and not args.force:
        completed, elapsed_before = restore_training_checkpoint(
            checkpoint,
            model,
            functional.optimizer,
            functional.device,
        )
    elif args.force or not history_path.exists():
        history_path.write_text("", encoding="utf-8")

    started = time.perf_counter()
    while completed < args.optimizer_updates:
        update_started = time.perf_counter()
        component_index = random.randrange(len(components))
        if args.all_section_targets:
            local_targets = torch.arange(components[component_index].num_nodes())
        else:
            local_targets = torch.multinomial(
                component_degrees[component_index],
                num_samples=args.target_batch_size,
                replacement=True,
            ).unique()
        target_nodes = local_targets + component_offsets[component_index]

        functional.model.train()
        functional.optimizer.zero_grad(set_to_none=True)
        loss_dict, n_input = _matched_fullneighbor_loss(
            functional,
            model.dataset,
            batched_graph,
            graph_to_dataset,
            sampler,
            target_nodes,
            args.encoder_batch_size,
        )
        loss = sum(
            value for value in loss_dict.values() if isinstance(value, torch.Tensor)
        )
        loss.backward()
        functional.optimizer.step()
        completed += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record = {
            "update": completed,
            "section": section_names[component_index],
            "n_targets": int(target_nodes.numel()),
            "n_fullneighbor_inputs": n_input,
            "seconds": time.perf_counter() - update_started,
            "losses": {
                key: float(value.detach().cpu())
                for key, value in loss_dict.items()
                if isinstance(value, torch.Tensor)
            },
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

        if completed % args.checkpoint_every == 0 or completed == args.optimizer_updates:
            elapsed = elapsed_before + time.perf_counter() - started
            save_training_checkpoint(
                checkpoint,
                model,
                functional.optimizer,
                completed,
                elapsed,
            )
        print(
            "completed matched full-neighbor update "
            f"{completed}/{args.optimizer_updates}: "
            f"section={record['section']} targets={record['n_targets']} "
            f"inputs={record['n_fullneighbor_inputs']} "
            f"seconds={record['seconds']:.2f}",
            flush=True,
        )
        functional.model.g = None
        del loss, loss_dict

    return completed, elapsed_before + time.perf_counter() - started


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    cross = left.T @ right
    numerator = np.square(cross).sum()
    denominator = np.linalg.norm(left.T @ left) * np.linalg.norm(right.T @ right)
    return float(numerator / max(denominator, np.finfo(float).eps))


def cal_pas(labels: np.ndarray, coords: np.ndarray, k: int = 10) -> float:
    _, neighbors = KDTree(coords).query(coords, k=min(k + 1, len(coords)))
    return float(np.mean(labels[neighbors[:, 1:]] == labels[:, None]))


def cal_chaos(labels: np.ndarray, coords: np.ndarray) -> float:
    coords = StandardScaler().fit_transform(coords)
    total = 0.0
    for label in np.unique(labels):
        points = coords[labels == label]
        if len(points) <= 2:
            continue
        distances, _ = KDTree(points).query(points, k=2)
        total += float(distances[:, 1].sum())
    return total / len(labels)


def evaluate(
    model: stModel,
    sampled_embedding: np.ndarray,
    sampled_domains: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    functional = model._functional
    full_embedding = functional.graph_ops.embed_with_graph(
        model.dataset,
        batch_size=args.inference_batch_size,
    )
    n_clusters = int(np.unique(sampled_domains).size)
    full_domains = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=args.seed,
        n_init=10,
        batch_size=4096,
    ).fit_predict(full_embedding)
    np.save(args.output_dir / "full_embedding.npy", full_embedding.astype(np.float32))

    batch_key = "sample" if args.dataset == "prostate" else "batch"
    coords = model.adata.obs[
        ["x", "y"] if args.dataset == "prostate" else ["array_row", "array_col"]
    ].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "batch": model.adata.obs[batch_key].astype(str).to_numpy(),
            "sampled_domain": sampled_domains.astype(str),
            "full_domain": full_domains.astype(str),
        },
        index=model.adata.obs_names,
    )
    frame.to_csv(args.output_dir / "assignments.csv.gz", compression="gzip")

    rows = []
    for batch in pd.unique(frame["batch"]):
        mask = frame["batch"].eq(batch).to_numpy()
        rows.append(
            {
                "batch": batch,
                "n_spots": int(mask.sum()),
                "sampled_full_ari": adjusted_rand_score(
                    sampled_domains[mask], full_domains[mask]
                ),
                "sampled_full_nmi": normalized_mutual_info_score(
                    sampled_domains[mask], full_domains[mask]
                ),
                "embedding_linear_cka": linear_cka(
                    sampled_embedding[mask], full_embedding[mask]
                ),
                "sampled_pas": cal_pas(sampled_domains[mask], coords[mask]),
                "full_pas": cal_pas(full_domains[mask], coords[mask]),
                "sampled_chaos": cal_chaos(sampled_domains[mask], coords[mask]),
                "full_chaos": cal_chaos(full_domains[mask], coords[mask]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    adata, source_info = load_data(args)
    sampled_embedding = np.asarray(adata.obsm["X_smoothed"], dtype=np.float32).copy()
    sampled_domains = adata.obs["domain"].astype(str).to_numpy().copy()
    model, beta = build_model(args, adata)
    if args.objective_mode == "all-node":
        install_exact_chunked_contrastive_loss(model, args.contrast_chunk_size)
        install_memory_bounded_full_graph_loss(model, args.decoder_chunk_size)
        validate_chunked_contrastive_loss(model, args.contrast_chunk_size)
        validate_chunked_reconstruction(model)
        completed, seconds = train_all_node_full_graph(model, args, beta)
    else:
        if args.all_section_targets:
            install_exact_chunked_contrastive_loss(model, args.contrast_chunk_size)
            validate_chunked_contrastive_loss(model, args.contrast_chunk_size)
        completed, seconds = train_matched_batch_full_graph(model, args, beta)
    metrics = evaluate(model, sampled_embedding, sampled_domains, args)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    summary = {
        **source_info,
        "n_spots": int(model.adata.n_obs),
        "n_genes": int(model.adata.n_vars),
        "optimizer_updates": completed,
        "training_seconds": seconds,
        "objective_mode": args.objective_mode,
        "training_graph": (
            "complete section graph"
            if args.objective_mode == "all-node"
            else "complete multilayer neighborhoods from the full section graph"
        ),
        "target_batch_size": (
            None if args.all_section_targets else args.target_batch_size
            if args.objective_mode == "matched-batch"
            else None
        ),
        "target_batch_policy": (
            "all nodes in the randomly selected section"
            if args.all_section_targets
            else "degree-weighted sampled nodes"
        ),
        "contrastive_negative_pool": (
            "all section nodes"
            if args.objective_mode == "all-node"
            else "sampled target batch"
        ),
        "inference_graph": "complete section graph via exact full-neighbor blocks",
        "contrast": (
            "all-node InfoNCE with memory-bounded logits"
            if args.objective_mode == "all-node"
            else "target-batch InfoNCE matched to GraphSAINT"
        ),
        "encoder_batch_size": args.encoder_batch_size,
        "decoder_chunk_size": args.decoder_chunk_size,
        "contrast_chunk_size": args.contrast_chunk_size,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else None
        ),
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**20,
        "metrics": metrics.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
