#!/usr/bin/env python3
"""Evaluate repeated GraphSAINT node sampling on a regular spatial grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def square_grid_edges(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    src: list[int] = []
    dst: list[int] = []
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            if r + 1 < n_rows:
                src.append(i)
                dst.append((r + 1) * n_cols + c)
            if c + 1 < n_cols:
                src.append(i)
                dst.append(r * n_cols + c + 1)
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def component_stats(sampled_nodes: np.ndarray, src: np.ndarray, dst: np.ndarray, n_nodes: int) -> dict[str, float]:
    node_to_local = {int(node): i for i, node in enumerate(sampled_nodes)}
    parent = np.arange(len(sampled_nodes), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    sampled_edge_count = 0
    for a, b in zip(src, dst, strict=True):
        la = node_to_local.get(int(a))
        lb = node_to_local.get(int(b))
        if la is None or lb is None:
            continue
        sampled_edge_count += 1
        union(la, lb)

    roots = np.asarray([find(i) for i in range(len(sampled_nodes))], dtype=np.int64)
    sizes = np.bincount(roots, minlength=len(sampled_nodes))
    sizes = sizes[sizes > 0]
    n_sampled = max(len(sampled_nodes), 1)
    return {
        "sampled_edges": float(sampled_edge_count),
        "sampled_node_fraction": float(len(sampled_nodes) / n_nodes),
        "sampled_edge_fraction": float(sampled_edge_count / max(len(src), 1)),
        "sampled_mean_degree_self_inclusive": float((2 * sampled_edge_count + len(sampled_nodes)) / n_sampled),
        "n_components": float(len(sizes)),
        "components_per_1k_nodes": float(len(sizes) / n_sampled * 1000.0),
        "largest_component_fraction": float(sizes.max() / n_sampled) if len(sizes) else 0.0,
        "isolated_node_fraction": float(np.mean(sizes == 1)) if len(sizes) else 0.0,
    }


def run_budget(
    *,
    label: str,
    n_rows: int,
    n_cols: int,
    sample_fraction: float,
    n_iterations: int,
    seed: int,
) -> dict[str, float | int | str]:
    src, dst = square_grid_edges(n_rows, n_cols)
    n_nodes = n_rows * n_cols
    degree = np.bincount(np.concatenate([src, dst]), minlength=n_nodes).astype(float)
    prob = np.clip(degree, 1.0, None)
    prob = prob / prob.sum()

    n_sampled = max(1, min(int(round(sample_fraction * n_nodes)), n_nodes))
    rng = np.random.default_rng(seed)
    node_hits = np.zeros(n_nodes, dtype=np.int32)
    edge_hits = np.zeros(len(src), dtype=np.int32)
    rows: list[dict[str, float]] = []

    for _ in range(n_iterations):
        sampled = np.sort(rng.choice(n_nodes, size=n_sampled, replace=False, p=prob))
        mask = np.zeros(n_nodes, dtype=bool)
        mask[sampled] = True
        sampled_edges = mask[src] & mask[dst]
        node_hits += mask
        edge_hits += sampled_edges
        row = {"unique_sampled_nodes": float(n_sampled)}
        row.update(component_stats(sampled, src, dst, n_nodes))
        rows.append(row)

    mean_row = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    degree_with_self = degree + 1.0
    neighbor_hits = node_hits * degree_with_self
    mean_row.update(
        {
            "label": label,
            "target_sample_fraction": float(sample_fraction),
            "target_sample_percent": float(100 * sample_fraction),
            "n_iterations": int(n_iterations),
            "cumulative_node_coverage_fraction": float(np.mean(node_hits > 0)),
            "cumulative_edge_coverage_fraction": float(np.mean(edge_hits > 0)),
            "mean_node_frequency": float(node_hits.mean()),
            "p10_node_frequency": float(np.quantile(node_hits, 0.1)),
            "min_node_frequency": float(node_hits.min()),
            "mean_edge_frequency": float(edge_hits.mean()),
            "p10_edge_frequency": float(np.quantile(edge_hits, 0.1)),
            "min_edge_frequency": float(edge_hits.min()),
            "mean_neighbor_frequency_self_inclusive": float(neighbor_hits.mean()),
            "p10_neighbor_frequency_self_inclusive": float(np.quantile(neighbor_hits, 0.1)),
            "min_neighbor_frequency_self_inclusive": float(neighbor_hits.min()),
        }
    )
    return mean_row


def plot_ladder(summary: dict[str, object], out_prefix: Path) -> None:
    sample_df = pd.DataFrame(summary["sample_budget_ladder"])
    iter_df = pd.DataFrame(summary["iteration_budget_ladder"])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    axes[0].plot(sample_df["target_sample_percent"], sample_df["components_per_1k_nodes"], marker="o")
    axes[0].set_xlabel("Sampled nodes per update (%)")
    axes[0].set_ylabel("Components per 1k sampled nodes")
    axes[0].set_title("Single-update fragmentation")

    axes[1].plot(iter_df["n_iterations"], iter_df["mean_node_frequency"], marker="o", label="Node")
    axes[1].plot(iter_df["n_iterations"], iter_df["mean_edge_frequency"], marker="o", label="Edge")
    axes[1].plot(iter_df["n_iterations"], iter_df["mean_neighbor_frequency_self_inclusive"], marker="o", label="Self-inclusive neighbor")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Mean exposure frequency")
    axes[1].set_yscale("log")
    axes[1].set_title("Repeated sampling exposure")
    axes[1].legend(frameon=False)

    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate GraphSAINT node sampling on a square grid.")
    parser.add_argument("--n-rows", type=int, default=64)
    parser.add_argument("--n-cols", type=int, default=64)
    parser.add_argument("--sample-fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5])
    parser.add_argument("--iterations", type=int, nargs="+", default=[1000, 2000, 3000])
    parser.add_argument("--accepted-sample-fraction", type=float, default=0.25)
    parser.add_argument("--accepted-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--out-dir", type=Path, default=Path("workflows/graphsaint_sampling"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_src, _ = square_grid_edges(args.n_rows, args.n_cols)
    summary: dict[str, object] = {
        "graph": {
            "layout": "square_grid",
            "n_rows": args.n_rows,
            "n_cols": args.n_cols,
            "n_nodes": args.n_rows * args.n_cols,
            "n_edges": len(full_src),
            "mean_degree": float(2 * len(full_src) / (args.n_rows * args.n_cols)),
        },
        "sampling_note": "GraphSAINT node sampling is random. On a near-k-regular square-grid graph, degree-based node-loss weights are approximately constant.",
    }
    summary["sample_budget_ladder"] = [
        run_budget(
            label=f"sample_fraction_{frac:g}",
            n_rows=args.n_rows,
            n_cols=args.n_cols,
            sample_fraction=frac,
            n_iterations=args.accepted_iterations,
            seed=args.seed + i * 1000,
        )
        for i, frac in enumerate(args.sample_fractions)
    ]
    summary["iteration_budget_ladder"] = [
        run_budget(
            label=f"iteration_budget_{n_iter}",
            n_rows=args.n_rows,
            n_cols=args.n_cols,
            sample_fraction=args.accepted_sample_fraction,
            n_iterations=n_iter,
            seed=args.seed + 10000 + i * 1000,
        )
        for i, n_iter in enumerate(args.iterations)
    ]

    rows = pd.DataFrame(summary["sample_budget_ladder"] + summary["iteration_budget_ladder"])
    rows.to_csv(args.out_dir / "graphsaint_simulation_ladder.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_ladder(summary, args.out_dir / "graphsaint_ladder")


if __name__ == "__main__":
    main()
