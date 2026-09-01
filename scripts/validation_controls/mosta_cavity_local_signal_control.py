#!/usr/bin/env python
"""Test whether local expression consistency can recover the MOSTA Cavity."""

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull, cKDTree, distance
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/mosta_e16_test.h5ad"
DEFAULT_OUTPUT = REPO_ROOT / "workflows/mosta_cavity_controls/local_signal"
SECTION = "E16.5_E2S7.MOSTA"
CAVITY = "Cavity"
METHOD_ORDER = (
    "No graph",
    "1-hop Gaussian",
    "2-hop Gaussian",
    "4-hop Gaussian",
    "8-hop Gaussian",
)
METHOD_COLORS = {
    "No graph": "#4C78A8",
    "1-hop Gaussian": "#F2A541",
    "2-hop Gaussian": "#2A9D8F",
    "4-hop Gaussian": "#D66BA0",
    "8-hop Gaussian": "#6F4E7C",
}
SPATIAL_CONDITIONS = (
    (0.5, "No graph"),
    (0.5, "2-hop Gaussian"),
    (4.0, "2-hop Gaussian"),
    (4.0, "4-hop Gaussian"),
    (4.0, "8-hop Gaussian"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--section", default=SECTION)
    parser.add_argument("--edge-radius", type=float, default=1.5)
    parser.add_argument("--kernel-sigma", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--mean-separation", type=float, default=3.0)
    parser.add_argument("--other-sigma", type=float, default=0.8)
    parser.add_argument(
        "--cavity-sigmas",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 1.0, 2.0, 4.0),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def radius_kernel(
    coords: np.ndarray,
    radius: float,
    sigma: float,
) -> tuple[csr_matrix, csr_matrix, np.ndarray]:
    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    distances = np.linalg.norm(coords[rows] - coords[cols], axis=1)
    weights = np.exp(-(distances**2) / (2.0 * sigma**2))
    n_cells = len(coords)
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(n_cells, n_cells),
    ).tocsr()
    kernel = coo_matrix(
        (weights.astype(np.float32), (rows, cols)),
        shape=(n_cells, n_cells),
    ).tocsr()
    kernel = kernel + diags(np.ones(n_cells, dtype=np.float32))
    kernel = diags(1.0 / np.asarray(kernel.sum(axis=1)).ravel()) @ kernel
    return adjacency, kernel.tocsr(), pairs


def annotation_mapping(truth: np.ndarray, clusters: np.ndarray) -> dict[int, str]:
    truth_values = np.unique(truth)
    cluster_values = np.unique(clusters)
    overlap = np.zeros((len(cluster_values), len(truth_values)), dtype=np.int64)
    truth_index = {value: idx for idx, value in enumerate(truth_values)}
    cluster_index = {value: idx for idx, value in enumerate(cluster_values)}
    for cluster, label in zip(clusters, truth):
        overlap[cluster_index[cluster], truth_index[label]] += 1
    rows, cols = linear_sum_assignment(-overlap)
    return {
        int(cluster_values[row]): str(truth_values[col])
        for row, col in zip(rows, cols)
    }


def largest_component_fraction(adjacency: csr_matrix, mask: np.ndarray) -> float:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return 0.0
    n_components, labels = connected_components(
        adjacency[indices][:, indices],
        directed=False,
    )
    if n_components == 0:
        return 0.0
    return float(np.bincount(labels).max() / len(indices))


def evaluate_embedding(
    embedding: np.ndarray,
    truth: np.ndarray,
    cavity_truth: np.ndarray,
    adjacency: csr_matrix,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    n_clusters = len(np.unique(truth))
    clusters = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=8192,
        n_init=3,
        max_iter=150,
        random_state=seed,
    ).fit_predict(embedding)
    mapping = annotation_mapping(truth, clusters)
    mapped = np.asarray([mapping.get(int(value), "Unmapped") for value in clusters])
    cavity_prediction = mapped == CAVITY
    true_positive = int(np.logical_and(cavity_truth, cavity_prediction).sum())
    false_positive = int(np.logical_and(~cavity_truth, cavity_prediction).sum())
    false_negative = int(np.logical_and(cavity_truth, ~cavity_prediction).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    metrics = {
        "ARI": float(adjusted_rand_score(truth, clusters)),
        "NMI": float(normalized_mutual_info_score(truth, clusters)),
        "cavity_precision": float(precision),
        "cavity_recall": float(recall),
        "cavity_f1": float(f1),
        "cavity_iou": float(iou),
        "cavity_lcc_fraction": largest_component_fraction(
            adjacency,
            cavity_prediction,
        ),
    }
    return metrics, cavity_prediction


def generate_embeddings(
    labels: np.ndarray,
    cavity_sigma: float,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    categories = np.unique(labels)
    means = rng.normal(size=(len(categories), args.embedding_dim))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    means *= args.mean_separation
    category_index = {value: idx for idx, value in enumerate(categories)}
    indices = np.asarray([category_index[value] for value in labels])
    scales = np.full(len(labels), args.other_sigma, dtype=np.float32)
    scales[labels == CAVITY] = cavity_sigma
    noise = rng.normal(size=(len(labels), args.embedding_dim)).astype(np.float32)
    return means[indices].astype(np.float32) + noise * scales[:, None]


def hop_reachability(
    coords: np.ndarray,
    cavity_truth: np.ndarray,
    adjacency: csr_matrix,
    pairs: np.ndarray,
) -> dict[str, float]:
    edge_vectors = coords[pairs[:, 1]] - coords[pairs[:, 0]]
    unique_vectors = np.unique(
        np.round(np.concatenate((edge_vectors, -edge_vectors)), decimals=8),
        axis=0,
    )
    two_hop_vectors = (
        unique_vectors[:, None, :] + unique_vectors[None, :, :]
    ).reshape(-1, 2)
    one_hop_reach = float(np.linalg.norm(unique_vectors, axis=1).max())
    two_hop_reach = float(np.linalg.norm(two_hop_vectors, axis=1).max())

    cavity_indices = np.flatnonzero(cavity_truth)
    cavity_graph = adjacency[cavity_indices][:, cavity_indices]
    _, component_labels = connected_components(cavity_graph, directed=False)
    largest_label = np.bincount(component_labels).argmax()
    largest_local = np.flatnonzero(component_labels == largest_label)
    largest_graph = cavity_graph[largest_local][:, largest_local]
    first = dijkstra(largest_graph, directed=False, unweighted=True, indices=0)
    farthest = int(np.nanargmax(np.where(np.isfinite(first), first, -1)))
    second = dijkstra(
        largest_graph,
        directed=False,
        unweighted=True,
        indices=farthest,
    )
    hop_diameter_lower_bound = float(np.nanmax(second[np.isfinite(second)]))

    cavity_coords = coords[cavity_indices[largest_local]]
    hull = ConvexHull(cavity_coords)
    euclidean_diameter = float(
        distance.pdist(cavity_coords[hull.vertices]).max()
    )
    return {
        "one_hop_max_coordinate_distance": one_hop_reach,
        "two_hop_max_coordinate_distance": two_hop_reach,
        "cavity_lcc_euclidean_diameter": euclidean_diameter,
        "cavity_lcc_hop_diameter_lower_bound": hop_diameter_lower_bound,
        "two_hop_fraction_of_cavity_diameter": two_hop_reach
        / euclidean_diameter,
    }


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), constrained_layout=True)
    panels = (
        ("cavity_f1", "Cavity F1"),
        ("cavity_recall", "Cavity recall"),
        ("ARI", "ARI"),
    )
    grouped = metrics.groupby(["cavity_sigma", "method"])
    summary = grouped.mean(numeric_only=True).reset_index()
    spread = grouped.std(numeric_only=True).reset_index()
    sigma_ticks = np.sort(metrics["cavity_sigma"].unique())
    for ax, (metric, title) in zip(axes, panels):
        for method in METHOD_ORDER:
            frame = summary.loc[summary["method"] == method]
            ax.plot(
                frame["cavity_sigma"],
                frame[metric],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2,
                label=method,
            )
            error = spread.loc[spread["method"] == method, metric].to_numpy()
            mean = frame[metric].to_numpy()
            ax.fill_between(
                frame["cavity_sigma"],
                np.clip(mean - error, 0, 1),
                np.clip(mean + error, 0, 1),
                color=METHOD_COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sigma_ticks)
        ax.set_xticklabels([f"{value:g}" for value in sigma_ticks])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Cavity embedding SD")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spatial(
    coords: np.ndarray,
    truth: np.ndarray,
    predictions: dict[tuple[float, str], np.ndarray],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(21.5, 4.1), constrained_layout=True)
    color = "#D95F02"
    for ax, title, mask in zip(
        axes,
        (
            "Ground truth",
            *(f"{method}, SD = {sigma:g}" for sigma, method in SPATIAL_CONDITIONS),
        ),
        (truth, *(predictions[condition] for condition in SPATIAL_CONDITIONS)),
    ):
        ax.scatter(
            coords[:, 0],
            -coords[:, 1],
            c="#D9D9D9",
            s=0.2,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            coords[mask, 0],
            -coords[mask, 1],
            c=color,
            s=0.7,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="none",
                label="Cavity",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="none",
                label="Other tissue",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    metrics: pd.DataFrame,
    coords: np.ndarray,
    truth: np.ndarray,
    predictions: dict[tuple[float, str], np.ndarray],
    output: Path,
) -> None:
    fig = plt.figure(figsize=(21.5, 8.2), constrained_layout=True)
    subfigures = fig.subfigures(2, 1, height_ratios=(1.0, 1.05))
    metric_axes = subfigures[0].subplots(1, 3)
    spatial_axes = subfigures[1].subplots(1, 6)

    grouped = metrics.groupby(["cavity_sigma", "method"])
    summary = grouped.mean(numeric_only=True).reset_index()
    spread = grouped.std(numeric_only=True).reset_index()
    sigma_ticks = np.sort(metrics["cavity_sigma"].unique())
    panels = (
        ("cavity_f1", "Cavity F1"),
        ("cavity_recall", "Cavity recall"),
        ("ARI", "ARI"),
    )
    for ax, (metric, title) in zip(metric_axes, panels):
        for method in METHOD_ORDER:
            frame = summary.loc[summary["method"] == method]
            error = spread.loc[spread["method"] == method, metric].to_numpy()
            mean = frame[metric].to_numpy()
            ax.plot(
                frame["cavity_sigma"],
                mean,
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2,
                label=method,
            )
            ax.fill_between(
                frame["cavity_sigma"],
                np.clip(mean - error, 0, 1),
                np.clip(mean + error, 0, 1),
                color=METHOD_COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sigma_ticks)
        ax.set_xticklabels([f"{value:g}" for value in sigma_ticks])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Cavity embedding SD")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    metric_axes[0].legend(frameon=False)

    spatial_masks = (
        truth,
        *(predictions[condition] for condition in SPATIAL_CONDITIONS),
    )
    spatial_titles = (
        "Ground truth",
        *(f"{method}, SD = {sigma:g}" for sigma, method in SPATIAL_CONDITIONS),
    )
    cavity_color = "#D95F02"
    for ax, title, mask in zip(spatial_axes, spatial_titles, spatial_masks):
        ax.scatter(
            coords[:, 0],
            -coords[:, 1],
            c="#D9D9D9",
            s=0.2,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            coords[mask, 0],
            -coords[mask, 1],
            c=cavity_color,
            s=0.7,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")
    subfigures[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=cavity_color,
                markeredgecolor="none",
                label="Cavity",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="none",
                label="Other tissue",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_legends(output: Path, section: str) -> None:
    section_label = section.replace(".MOSTA", "")
    output.write_text(
        "# Figure legends\n\n"
        "**Local-signal Cavity control.** Semi-synthetic embeddings were generated "
        f"independently at the observed {section_label} positions from "
        "annotation-specific distributions. Only the within-Cavity embedding "
        "standard deviation was varied. Curves show domain recovery from the "
        "no-graph embeddings or one, two, four, or eight successive applications "
        "of the same local Gaussian kernel. Points show means over "
        "independent simulations. No long-range edges or label-guided propagation "
        "were used.\n\n"
        "**Local-signal Cavity spatial recovery.** Ground-truth Cavity positions "
        "and the Cavity-mapped clusters obtained from the no-graph embedding and "
        "after two local Gaussian smoothing steps at within-Cavity embedding SD = 0.5, "
        "followed by two, four, and eight smoothing steps at SD = 4. "
        "Orange indicates Cavity; gray indicates all other tissue positions. The "
        "combined summary places the metric curves above these spatial maps.\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.input)
    adata = adata[
        adata.obs["batch"].astype(str).eq(args.section).to_numpy()
    ].copy()
    coords = adata.obs[["x", "y"]].to_numpy(dtype=float)
    labels = adata.obs["annotation"].astype(str).to_numpy()
    cavity_truth = labels == CAVITY
    adjacency, kernel, pairs = radius_kernel(
        coords,
        radius=args.edge_radius,
        sigma=args.kernel_sigma,
    )
    reach = hop_reachability(coords, cavity_truth, adjacency, pairs)

    rows = []
    spatial_predictions: dict[tuple[float, str], np.ndarray] = {}
    for sigma in args.cavity_sigmas:
        for repeat in range(args.repeats):
            simulation_seed = args.seed + repeat
            clustering_seed = args.seed + 10_000 + repeat
            no_graph = generate_embeddings(labels, sigma, args, simulation_seed)
            embeddings = {"No graph": no_graph}
            smoothed = no_graph
            for hop in range(1, 9):
                smoothed = kernel @ smoothed
                if hop in (1, 2, 4, 8):
                    embeddings[f"{hop}-hop Gaussian"] = smoothed
            for method in METHOD_ORDER:
                embedding = embeddings[method]
                values, prediction = evaluate_embedding(
                    np.asarray(embedding, dtype=np.float32),
                    labels,
                    cavity_truth,
                    adjacency,
                    clustering_seed,
                )
                rows.append(
                    {
                        "cavity_sigma": sigma,
                        "repeat": repeat,
                        "method": method,
                        **values,
                    }
                )
                if repeat == 0:
                    spatial_predictions[(float(sigma), method)] = prediction

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    write_json(
        args.output_dir / "hop_reachability.json",
        {
            "section": args.section,
            "n_cells": int(len(adata)),
            "n_cavity_cells": int(cavity_truth.sum()),
            "edge_radius": args.edge_radius,
            "kernel_sigma": args.kernel_sigma,
            "graph_layers": 2,
            **reach,
        },
    )
    plot_metrics(metrics, args.output_dir / "cavity_local_signal_metrics")
    missing_conditions = [
        condition
        for condition in SPATIAL_CONDITIONS
        if condition not in spatial_predictions
    ]
    if missing_conditions:
        raise ValueError(f"Missing spatial display conditions: {missing_conditions}")
    plot_spatial(
        coords,
        cavity_truth,
        spatial_predictions,
        args.output_dir / "cavity_local_signal_spatial",
    )
    plot_summary(
        metrics,
        coords,
        cavity_truth,
        spatial_predictions,
        args.output_dir / "cavity_local_signal_summary",
    )
    write_legends(args.output_dir / "figure_legends.md", args.section)


if __name__ == "__main__":
    main()
