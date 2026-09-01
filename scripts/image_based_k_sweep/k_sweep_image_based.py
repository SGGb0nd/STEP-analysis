#!/usr/bin/env python3
"""
k-sweep robustness for image-based (irregular) spatial graphs.

Focus:
- kNN graph sensitivity in irregular or image-based spatial data
- Quantify how spatial-domain identification changes when k varies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_ablation_module():
    path = Path(__file__).resolve().parents[1] / "module_ablation" / "step_module_ablation_sim.py"
    spec = importlib.util.spec_from_file_location("step_module_ablation_sim", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def irregularize_coords(
    adata,
    seed: int,
    scale: float = 0.45,
    rotation_max: float = 0.45,
    warp_x_amp: float = 0.9,
    warp_x_period: float = 4.5,
    warp_y_amp: float = 0.7,
    warp_y_period: float = 5.0,
):
    rng = np.random.RandomState(seed)
    out = adata.copy()
    xy = out.obsm["spatial"].copy().astype(np.float64)
    theta = rng.uniform(-rotation_max, rotation_max)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    xy = xy @ rot.T
    xy[:, 0] += warp_x_amp * np.sin(xy[:, 1] / warp_x_period)
    xy[:, 1] += warp_y_amp * np.cos(xy[:, 0] / warp_y_period)
    xy += rng.normal(scale=scale, size=xy.shape)
    out.obsm["spatial"] = xy
    out.obs["array_row"] = xy[:, 0]
    out.obs["array_col"] = xy[:, 1]
    return out


def valid_gt_mask(labels: np.ndarray) -> np.ndarray:
    x = np.asarray(labels, dtype=str)
    bad = {"", "NA", "na", "Na", "n/a", "None", "none", "nan", "NaN"}
    return np.array([v not in bad for v in x], dtype=bool)


def evaluate_domain_labels(
    emb_spatial: np.ndarray,
    domain_true: np.ndarray,
    cluster_seed: int,
) -> tuple[np.ndarray, float, float]:
    n_domains = int(len(np.unique(domain_true)))
    pred = KMeans(
        n_clusters=n_domains,
        random_state=int(cluster_seed),
        n_init=10,
    ).fit_predict(emb_spatial)
    mask = valid_gt_mask(domain_true)
    ari = float(adjusted_rand_score(domain_true[mask], pred[mask]))
    nmi = float(normalized_mutual_info_score(domain_true[mask], pred[mask]))
    return pred, ari, nmi


def run_one_trial(
    mod,
    adata,
    k_neighbors: int,
    train_seed: int,
    cluster_seed: int,
    n_iterations: int,
    n_top_genes: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    cfg = {
        "tag": f"k_{k_neighbors}",
        "n_top_genes": int(n_top_genes),
        "module_dim": 30,
        "hidden_dim": 96,
        "n_modules": 8,
        "n_glayers": 6,
        "n_enc_layer": 3,
        "edge_clip": None,
        "max_neighbors": int(k_neighbors),
        "n_iterations": int(n_iterations),
        "sample_rate": 1.0,
        "lr": 1.8e-3,
        "beta": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip_norm": 2.0,
        "train_seed": int(train_seed),
        "cluster_seed": int(cluster_seed),
        "use_bem": True,
    }
    rec, model = mod.run_step_trial(adata, cfg, return_model=True)
    assert model is not None
    emb_spatial = np.asarray(model.adata.obsm["X_smoothed"], dtype=np.float64)
    domain_true = model.adata.obs["spatial_domain"].astype(str).to_numpy()
    pred, domain_ari, domain_nmi = evaluate_domain_labels(
        emb_spatial=emb_spatial,
        domain_true=domain_true,
        cluster_seed=cluster_seed,
    )
    out = {
        "k": int(k_neighbors),
        "domain_ari": domain_ari,
        "domain_nmi": domain_nmi,
        "domain_neighbor_agreement": float(rec["domain_neighbor_agreement"]),
        "batch_silhouette": float(rec["batch_silhouette"]),
        "coherence_spearman": float(rec["coherence_spearman"]),
        "elapsed_sec": float(rec["elapsed_sec"]),
    }
    model.model.cpu()
    return out, pred, domain_true


def parse_k_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("k list is empty")
    if any(v <= 0 for v in vals):
        raise ValueError("k values must be positive")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="k-sweep for irregular/image-based spatial-domain robustness.")
    parser.add_argument("--k-list", type=str, default="6,8,10,12,15")
    parser.add_argument("--ref-k", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=80)
    parser.add_argument("--n-top-genes", type=int, default=3000, choices=[2000, 3000])
    parser.add_argument("--grid-rows", type=int, default=22)
    parser.add_argument("--grid-cols", type=int, default=22)
    parser.add_argument("--n-genes", type=int, default=2200)
    parser.add_argument("--n-domains", type=int, default=6)
    parser.add_argument("--n-batches", type=int, default=2)
    parser.add_argument("--batch-effect", type=float, default=0.8)
    parser.add_argument("--jitter-scale", type=float, default=0.45)
    parser.add_argument("--rotation-max", type=float, default=0.45)
    parser.add_argument("--warp-x-amp", type=float, default=0.9)
    parser.add_argument("--warp-x-period", type=float, default=4.5)
    parser.add_argument("--warp-y-amp", type=float, default=0.7)
    parser.add_argument("--warp-y-period", type=float, default=5.0)
    parser.add_argument("--scenario", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="workflows/image_based_k_sweep")
    args = parser.parse_args()

    ks = parse_k_list(args.k_list)
    if args.ref_k not in ks:
        raise ValueError(f"ref-k ({args.ref_k}) must be in k-list ({ks})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mod = load_ablation_module()

    records: list[dict[str, Any]] = []
    ref_pairs: list[dict[str, Any]] = []

    for repeat in range(args.n_repeats):
        sim_seed = int(args.seed + repeat * 97)
        sim = mod.SpatialCoherenceSimulator(
            grid_size=(args.grid_rows, args.grid_cols),
            n_genes=args.n_genes,
            n_domains=args.n_domains,
            n_batches=args.n_batches,
            batch_effect_strength=args.batch_effect,
            random_seed=sim_seed,
        )
        adata = irregularize_coords(
            sim.simulate(),
            seed=args.seed + 500 + repeat,
            scale=args.jitter_scale,
            rotation_max=args.rotation_max,
            warp_x_amp=args.warp_x_amp,
            warp_x_period=args.warp_x_period,
            warp_y_amp=args.warp_y_amp,
            warp_y_period=args.warp_y_period,
        )

        per_k_pred: dict[int, np.ndarray] = {}
        per_k_true: dict[int, np.ndarray] = {}
        for k in ks:
            train_seed = int(args.seed + 1000 * repeat + 13 * k)
            cluster_seed = int(args.seed + 2000 * repeat + 17)
            rec, pred, truth = run_one_trial(
                mod=mod,
                adata=adata,
                k_neighbors=k,
                train_seed=train_seed,
                cluster_seed=cluster_seed,
                n_iterations=args.n_iterations,
                n_top_genes=args.n_top_genes,
            )
            rec["repeat"] = repeat + 1
            rec["sim_seed"] = sim_seed
            records.append(rec)
            per_k_pred[k] = pred
            per_k_true[k] = truth

        ref_pred = per_k_pred[args.ref_k]
        ref_truth = per_k_true[args.ref_k]
        ref_mask = valid_gt_mask(ref_truth)
        for k in ks:
            ari_vs_ref = float(adjusted_rand_score(ref_pred[ref_mask], per_k_pred[k][ref_mask]))
            ref_pairs.append(
                {
                    "repeat": repeat + 1,
                    "k": int(k),
                    "ref_k": int(args.ref_k),
                    "ari_vs_ref_k": ari_vs_ref,
                }
            )

    df = pd.DataFrame(records)
    rf = pd.DataFrame(ref_pairs)
    merged = df.merge(rf, on=["repeat", "k"], how="left")

    summary = []
    for k, sub in merged.groupby("k"):
        summary.append(
            {
                "k": int(k),
                "domain_ari_mean": float(sub["domain_ari"].mean()),
                "domain_ari_std": float(sub["domain_ari"].std(ddof=0)),
                "domain_nmi_mean": float(sub["domain_nmi"].mean()),
                "domain_nmi_std": float(sub["domain_nmi"].std(ddof=0)),
                "ari_vs_ref_k_mean": float(sub["ari_vs_ref_k"].mean()),
                "ari_vs_ref_k_std": float(sub["ari_vs_ref_k"].std(ddof=0)),
                "domain_neighbor_agreement_mean": float(sub["domain_neighbor_agreement"].mean()),
                "domain_neighbor_agreement_std": float(sub["domain_neighbor_agreement"].std(ddof=0)),
                "batch_silhouette_mean": float(sub["batch_silhouette"].mean()),
                "batch_silhouette_std": float(sub["batch_silhouette"].std(ddof=0)),
                "coherence_spearman_mean": float(sub["coherence_spearman"].mean()),
                "coherence_spearman_std": float(sub["coherence_spearman"].std(ddof=0)),
            }
        )

    result = {
        "arguments": vars(args),
        "scenario": args.scenario,
        "k_values": ks,
        "records": records,
        "ref_pairwise": ref_pairs,
        "summary_by_k": sorted(summary, key=lambda x: x["k"]),
    }

    out_json = out_dir / f"image_based_k_sweep_{args.scenario}_{ts}.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(out_json), "k_values": ks}, indent=2))


if __name__ == "__main__":
    main()
