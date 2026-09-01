"""Render the image-based kNN sensitivity summary from saved sweep results."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "workflows" / "image_based_k_sweep"
DEFAULT_METRICS_JSON = DEFAULT_OUTDIR / "image_based_k_sweep_metrics.json"


def _load_module(filename: str, name: str):
    if filename == "k_sweep_image_based.py":
        path = Path(__file__).with_name(filename)
    elif filename == "step_module_ablation_sim.py":
        path = REPO_ROOT / "scripts" / "module_ablation" / filename
    else:
        path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _align_to_truth(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    truth_labels = sorted(np.unique(truth).tolist())
    pred_labels = sorted(np.unique(pred).tolist())
    cm = confusion_matrix(truth, pred, labels=truth_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {pred_labels[col]: truth_labels[row] for row, col in zip(row_ind, col_ind)}
    return np.asarray([mapping.get(label, truth_labels[0]) for label in pred], dtype=object)


def _scatter(ax, coords: np.ndarray, labels: np.ndarray, palette: dict[str, tuple[float, float, float]], title: str) -> None:
    colors = [palette[str(label)] for label in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=1.2, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_frame_on(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the manuscript-facing image-based k-sweep figure.")
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    metrics_json = args.metrics_json
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_mod = _load_module("k_sweep_image_based.py", "k_sweep_image_based")
    ablation_mod = _load_module("step_module_ablation_sim.py", "step_module_ablation_sim")

    sim = ablation_mod.SpatialCoherenceSimulator(
        grid_size=(128, 128),
        n_genes=2200,
        n_domains=6,
        n_batches=1,
        batch_effect_strength=0.0,
        random_seed=42,
    )
    adata = sweep_mod.irregularize_coords(
        sim.simulate(),
        seed=42 + 500,
        scale=0.45,
        rotation_max=0.45,
        warp_x_amp=1.1,
        warp_x_period=4.2,
        warp_y_amp=0.8,
        warp_y_period=4.8,
    )
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    truth = adata.obs["spatial_domain"].astype(str).to_numpy()

    ks_to_show = [6, 12, 20]
    pred_by_k: dict[int, np.ndarray] = {}
    for k in ks_to_show:
        rec, pred, _ = sweep_mod.run_one_trial(
            mod=ablation_mod,
            adata=adata,
            k_neighbors=k,
            train_seed=42 + 13 * k,
            cluster_seed=42 + 17,
            n_iterations=40,
            n_top_genes=2000,
        )
        pred_by_k[k] = _align_to_truth(pred.astype(str), truth.astype(str))

    with open(metrics_json) as f:
        metrics = json.load(f)
    summary = pd.DataFrame(metrics["summary_by_k"]).sort_values("k")

    domain_labels = sorted(np.unique(truth).tolist())
    palette_list = sns.color_palette("Set2", len(domain_labels))
    palette = {label: palette_list[i] for i, label in enumerate(domain_labels)}

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9])
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_metrics = fig.add_subplot(gs[1, 0:2])
    ax_agree = fig.add_subplot(gs[1, 2:4])

    _scatter(axes_top[0], coords, truth, palette, "Simulated irregular spatial domains")
    _scatter(axes_top[1], coords, pred_by_k[6], palette, "STEP spatial result, k = 6")
    _scatter(axes_top[2], coords, pred_by_k[12], palette, "STEP spatial result, k = 12")
    _scatter(axes_top[3], coords, pred_by_k[20], palette, "STEP spatial result, k = 20")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=palette[label], label=label, markersize=6)
        for label in domain_labels
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.98))

    x = summary["k"].to_numpy()
    ari_mean = summary["domain_ari_mean"].to_numpy()
    ari_std = summary["domain_ari_std"].to_numpy()
    nmi_mean = summary["domain_nmi_mean"].to_numpy()
    nmi_std = summary["domain_nmi_std"].to_numpy()
    ax_metrics.plot(x, ari_mean, marker="o", label="Domain ARI", color="#1f77b4")
    ax_metrics.fill_between(x, ari_mean - ari_std, ari_mean + ari_std, alpha=0.2, color="#1f77b4")
    ax_metrics.plot(x, nmi_mean, marker="s", label="Domain NMI", color="#d62728")
    ax_metrics.fill_between(x, nmi_mean - nmi_std, nmi_mean + nmi_std, alpha=0.2, color="#d62728")
    ax_metrics.set_xlabel("k")
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_ylim(0.7, 1.0)
    ax_metrics.set_title("Spatial-domain identification across k")
    ax_metrics.grid(alpha=0.2)
    ax_metrics.legend(frameon=False)

    agree_mean = summary["domain_neighbor_agreement_mean"].to_numpy()
    agree_std = summary["domain_neighbor_agreement_std"].to_numpy()
    ref_mean = summary["ari_vs_ref_k_mean"].to_numpy()
    ref_std = summary["ari_vs_ref_k_std"].to_numpy()
    ax_agree.plot(x, agree_mean, marker="o", label="Local neighbor-label agreement", color="#2ca02c")
    ax_agree.fill_between(x, agree_mean - agree_std, agree_mean + agree_std, alpha=0.2, color="#2ca02c")
    ax_agree.plot(x, ref_mean, marker="s", label="ARI vs reference k = 12", color="#9467bd")
    ax_agree.fill_between(x, ref_mean - ref_std, ref_mean + ref_std, alpha=0.2, color="#9467bd")
    ax_agree.set_xlabel("k")
    ax_agree.set_ylabel("Value")
    ax_agree.set_ylim(0.65, 1.0)
    ax_agree.set_title("Stability of local organization across k")
    ax_agree.grid(alpha=0.2)
    ax_agree.legend(frameon=False)

    fig.suptitle("Simulation-based k-sweep for irregular/image-based spatial graphs", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "image_based_k_sweep_simulation_summary.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / "image_based_k_sweep_simulation_summary.pdf", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
