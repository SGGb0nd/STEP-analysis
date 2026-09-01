#!/usr/bin/env python3
"""Run matched STEP module ablations and render representative spatial cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from scripts.module_ablation.step_module_ablation_sim import (
    SpatialCoherenceSimulator,
    permute_expression_within_batch,
    run_step_trial,
)


CASE_ORDER = [
    "Ground truth",
    "STEP full",
    "No SpM",
    "No BEM",
    "No BBM",
]


def split_batch_panels(adata: ad.AnnData, labels: np.ndarray | pd.Series) -> dict[str, dict[str, np.ndarray]]:
    label_arr = np.asarray(labels).astype(str)
    batches = adata.obs["batch"].astype(str).to_numpy()
    coords = adata.obs[["array_row", "array_col"]].to_numpy(dtype=float)
    out: dict[str, dict[str, np.ndarray]] = {}
    for batch in pd.unique(batches):
        mask = batches == batch
        out[str(batch)] = {
            "coords": coords[mask],
            "labels": label_arr[mask],
        }
    return out


def remap_labels_to_reference(ref: np.ndarray | pd.Series, pred: np.ndarray | pd.Series) -> np.ndarray:
    ref_arr = np.asarray(ref).astype(str)
    pred_arr = np.asarray(pred).astype(str)
    ref_labels = pd.Index(pd.unique(ref_arr))
    pred_labels = pd.Index(pd.unique(pred_arr))
    contingency = np.zeros((len(pred_labels), len(ref_labels)), dtype=int)
    for i, pl in enumerate(pred_labels):
        for j, rl in enumerate(ref_labels):
            contingency[i, j] = int(np.sum((pred_arr == pl) & (ref_arr == rl)))
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {pred_labels[i]: ref_labels[j] for i, j in zip(row_ind, col_ind)}
    fallback = {
        label: f"pred_{k + 1}"
        for k, label in enumerate(pred_labels)
        if label not in mapping
    }
    mapping.update(fallback)
    return np.array([mapping[label] for label in pred_arr], dtype=object)


def infer_case_labels(model) -> tuple[np.ndarray, np.ndarray]:
    emb_spatial = np.asarray(model.adata.obsm["X_smoothed"])
    if "X_rep" in model.adata.obsm_keys():
        emb_individual = np.asarray(model.adata.obsm["X_rep"])
    else:
        emb_individual = np.asarray(model.embed())
    domain_true = model.adata.obs["spatial_domain"].astype(str).to_numpy()
    ct_true = model.adata.obs["cell_type"].astype(str).to_numpy()
    n_domains = int(pd.Index(domain_true).nunique())
    n_celltypes = int(pd.Index(ct_true).nunique())
    pred_domain = KMeans(n_clusters=n_domains, random_state=42, n_init=10).fit_predict(emb_spatial)
    pred_celltype = KMeans(n_clusters=n_celltypes, random_state=53, n_init=10).fit_predict(emb_individual)
    return (
        remap_labels_to_reference(domain_true, pred_domain),
        remap_labels_to_reference(ct_true, pred_celltype),
    )


def categorical_palette(labels: list[str]) -> ListedColormap:
    base = plt.get_cmap("tab20")
    colors = [base(i % 20) for i in range(len(labels))]
    return ListedColormap(colors)


def plot_panel(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    cats = pd.Categorical(labels)
    cmap = categorical_palette(list(cats.categories))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=cats.codes,
        cmap=cmap,
        s=5,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_frame_on(False)


def make_case_figure(
    regime_name: str,
    adata: ad.AnnData,
    case_panels: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    gt_domain = split_batch_panels(adata, adata.obs["spatial_domain"].astype(str).to_numpy())
    gt_celltype = split_batch_panels(adata, adata.obs["cell_type"].astype(str).to_numpy())
    batches = list(gt_domain)
    if len(batches) != 2:
        raise ValueError(f"Expected exactly 2 batches for case figure layout, found {len(batches)}")

    fig, axes = plt.subplots(4, len(CASE_ORDER), figsize=(4.2 * len(CASE_ORDER), 13.2), constrained_layout=True)
    row_titles = [
        f"{batches[0]} spatial domain",
        f"{batches[0]} cell type",
        f"{batches[1]} spatial domain",
        f"{batches[1]} cell type",
    ]

    for col, case in enumerate(CASE_ORDER):
        if case == "Ground truth":
            panels_domain = gt_domain
            panels_celltype = gt_celltype
        else:
            panels_domain = split_batch_panels(adata, case_panels[case]["domain"])
            panels_celltype = split_batch_panels(adata, case_panels[case]["cell_type"])

        plot_panel(axes[0, col], panels_domain[batches[0]]["coords"], panels_domain[batches[0]]["labels"], case)
        plot_panel(axes[1, col], panels_celltype[batches[0]]["coords"], panels_celltype[batches[0]]["labels"], "")
        plot_panel(axes[2, col], panels_domain[batches[1]]["coords"], panels_domain[batches[1]]["labels"], "")
        plot_panel(axes[3, col], panels_celltype[batches[1]]["coords"], panels_celltype[batches[1]]["labels"], "")

    for row, text in enumerate(row_titles):
        axes[row, 0].annotate(
            text,
            xy=(-0.25, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=10,
            rotation=90,
        )

    fig.suptitle(regime_name, fontsize=14)
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def run_regime(
    batch_effect: float,
    seed: int,
    out_dir: Path,
    grid_rows: int,
    grid_cols: int,
    n_genes: int,
    n_domains: int,
    n_cell_types: int,
    n_batches: int,
    pattern_mode: str,
    n_iterations: int,
    n_top_genes: int,
    edge_clip: float,
) -> dict[str, str]:
    sim = SpatialCoherenceSimulator(
        grid_size=(grid_rows, grid_cols),
        n_genes=n_genes,
        n_domains=n_domains,
        n_cell_types=n_cell_types,
        n_batches=n_batches,
        batch_effect_strength=batch_effect,
        pattern_mode=pattern_mode,
        random_seed=seed,
    )
    adata = sim.simulate()

    common = {
        "n_top_genes": n_top_genes,
        "module_dim": 30,
        "hidden_dim": 96,
        "n_modules": 8,
        "n_glayers": 6,
        "n_enc_layer": 3,
        "edge_clip": edge_clip,
        "graph_loss_mode": "legacy",
        "n_iterations": n_iterations,
        "sample_rate": 1.0,
        "lr": 1.8e-3,
        "beta": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip_norm": 2.0,
        "train_seed": seed + 11,
        "cluster_seed": 42,
        "use_bem": True,
    }

    case_panels: dict[str, dict[str, np.ndarray]] = {}
    export: dict[str, str] = {}

    full_rec, full_model = run_step_trial(adata, {"tag": "full_bbm_bem_spm", **common}, return_model=True)
    d, c = infer_case_labels(full_model)
    case_panels["STEP full"] = {"domain": d, "cell_type": c}
    export["STEP full"] = json.dumps({"celltype_ari": full_rec["celltype_ari"], "domain_ari": full_rec["domain_ari"]})
    full_model.model.cpu()

    rec, model = run_step_trial(adata, {"tag": "ablate_spm", **common, "n_glayers": 0}, return_model=True)
    d, c = infer_case_labels(model)
    case_panels["No SpM"] = {"domain": d, "cell_type": c}
    export["No SpM"] = json.dumps({"celltype_ari": rec["celltype_ari"], "domain_ari": rec["domain_ari"]})
    model.model.cpu()

    rec, model = run_step_trial(adata, {"tag": "ablate_bem", **common, "use_bem": False}, return_model=True)
    d, c = infer_case_labels(model)
    case_panels["No BEM"] = {"domain": d, "cell_type": c}
    export["No BEM"] = json.dumps({"celltype_ari": rec["celltype_ari"], "domain_ari": rec["domain_ari"]})
    model.model.cpu()

    adata_bbm_permuted = permute_expression_within_batch(adata, batch_key="batch", seed=seed + 7919)
    rec, model = run_step_trial(adata_bbm_permuted, {"tag": "ablate_bbm_signal", **common}, return_model=True)
    d, c = infer_case_labels(model)
    case_panels["No BBM"] = {"domain": d, "cell_type": c}
    export["No BBM"] = json.dumps({"celltype_ari": rec["celltype_ari"], "domain_ari": rec["domain_ari"]})
    model.model.cpu()

    regime_name = "No-batch regime" if batch_effect <= 0 else "Batch-effect regime"
    stem = "ablation_cases_no_batch" if batch_effect <= 0 else "ablation_cases_batch_effect"
    make_case_figure(regime_name, adata, case_panels, out_dir / stem)
    (out_dir / f"{stem}.json").write_text(json.dumps(export, indent=2))
    return {
        "png": str((out_dir / stem).with_suffix(".png")),
        "pdf": str((out_dir / stem).with_suffix(".pdf")),
        "json": str(out_dir / f"{stem}.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("workflows/module_ablation"))
    parser.add_argument("--grid-rows", type=int, default=24)
    parser.add_argument("--grid-cols", type=int, default=24)
    parser.add_argument("--n-genes", type=int, default=2400)
    parser.add_argument("--n-domains", type=int, default=3)
    parser.add_argument("--n-cell-types", type=int, default=3)
    parser.add_argument("--n-batches", type=int, default=2)
    parser.add_argument("--pattern-mode", type=str, choices=["simple", "complex"], default="complex")
    parser.add_argument("--n-iterations", type=int, default=120)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--edge-clip", type=float, default=1.5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "no_batch": run_regime(
            batch_effect=0.0,
            seed=42,
            out_dir=args.out_dir,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            n_genes=args.n_genes,
            n_domains=args.n_domains,
            n_cell_types=args.n_cell_types,
            n_batches=args.n_batches,
            pattern_mode=args.pattern_mode,
            n_iterations=args.n_iterations,
            n_top_genes=args.n_top_genes,
            edge_clip=args.edge_clip,
        ),
        "batch_effect": run_regime(
            batch_effect=0.8,
            seed=42,
            out_dir=args.out_dir,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            n_genes=args.n_genes,
            n_domains=args.n_domains,
            n_cell_types=args.n_cell_types,
            n_batches=args.n_batches,
            pattern_mode=args.pattern_mode,
            n_iterations=args.n_iterations,
            n_top_genes=args.n_top_genes,
            edge_clip=args.edge_clip,
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
