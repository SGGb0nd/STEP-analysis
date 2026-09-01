#!/usr/bin/env python3
"""Run matched simulations for the BBM, BEM, and SpM ablation study."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn

METRIC_KEYS = [
    "domain_ari",
    "domain_nmi",
    "celltype_ari",
    "celltype_nmi",
    "batch_silhouette",
    "batch_asw",
    "batch_ilisi",
    "batch_silhouette_spatial",
    "batch_asw_spatial",
    "batch_ilisi_spatial",
    "domain_neighbor_agreement",
    "spatial_chaos",
    "spatial_pas",
]


def to_dense_array(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


class IdentitySmoother(nn.Module):
    def forward(self, g, x):
        return x


class SpatialCoherenceSimulator:
    def __init__(
        self,
        grid_size: tuple[int, int] = (30, 30),
        n_genes: int = 2400,
        n_domains: int = 6,
        n_cell_types: int = 3,
        n_batches: int = 2,
        batch_effect_strength: float = 0.8,
        pattern_mode: str = "complex",
        random_seed: int = 42,
    ) -> None:
        self.grid_size = grid_size
        self.n_cells_per_batch = grid_size[0] * grid_size[1]
        self.n_genes = int(n_genes)
        self.n_domains = int(n_domains)
        self.n_cell_types = int(n_cell_types)
        self.n_batches = int(n_batches)
        self.random_seed = int(random_seed)
        self.batch_effect_strength = float(batch_effect_strength)
        self.pattern_mode = pattern_mode
        self.rng = np.random.RandomState(random_seed)

        if self.pattern_mode not in {"simple", "complex"}:
            raise ValueError("pattern_mode must be 'simple' or 'complex'")

        self.domain_names = [f"Domain{i}" for i in range(self.n_domains)]
        self.cell_type_names = [f"CT{i}" for i in range(self.n_cell_types)]

        self.domain_markers_per_domain = 15
        self.celltype_markers_per_type = 20
        self.n_ccc_targets = 8

        self.domain_marker_total = self.n_domains * self.domain_markers_per_domain
        self.celltype_marker_total = self.n_cell_types * self.celltype_markers_per_type
        self.ligand_gene_idx = self.domain_marker_total + self.celltype_marker_total
        self.receptor_gene_idx = self.ligand_gene_idx + 1
        self.target_start_idx = self.receptor_gene_idx + 1
        self.target_end_idx = self.target_start_idx + self.n_ccc_targets
        self.background_start_idx = self.target_end_idx

        min_genes = self.background_start_idx + 50
        if self.n_genes < min_genes:
            raise ValueError(
                f"n_genes must be >= {min_genes} for current marker layout "
                f"(domains={self.n_domains}, cell_types={self.n_cell_types})."
            )

        self.batch_domain_ct_probs = np.stack(
            [self._build_domain_celltype_probs(batch_idx) for batch_idx in range(self.n_batches)],
            axis=0,
        )

    def _generate_coordinates(self) -> np.ndarray:
        x, y = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), indexing="ij")
        return np.column_stack([x.ravel(), y.ravel()]).astype(float)

    def _build_domain_celltype_probs(self, batch_idx: int) -> np.ndarray:
        probs = np.zeros((self.n_domains, self.n_cell_types), dtype=np.float32)
        for d in range(self.n_domains):
            dominant = (d + batch_idx) % self.n_cell_types
            alpha = np.ones(self.n_cell_types, dtype=np.float32) * 0.9
            alpha[dominant] = 1.9
            secondary = (dominant + 1 + (d % 2)) % self.n_cell_types
            alpha[secondary] += 0.4
            probs[d] = self.rng.dirichlet(alpha)
        return probs

    def _create_domains_simple(self, coords_local: np.ndarray) -> np.ndarray:
        domain_labels = np.zeros(len(coords_local), dtype=int)
        x = coords_local[:, 0]
        x_norm = x / max(float(x.max()), 1.0)
        edges = np.linspace(0, 1, self.n_domains + 1)
        for i in range(self.n_domains):
            mask = (x_norm >= edges[i]) & (x_norm <= edges[i + 1] if i == self.n_domains - 1 else x_norm < edges[i + 1])
            domain_labels[mask] = i
        return domain_labels

    def _create_domains_complex(self, coords_local: np.ndarray, batch_idx: int) -> np.ndarray:
        x = coords_local[:, 0] / max(self.grid_size[0] - 1, 1)
        y = coords_local[:, 1] / max(self.grid_size[1] - 1, 1)
        x_shift = 0.11 * batch_idx
        y_shift = 0.07 * batch_idx
        xw = np.mod(x + x_shift, 1.0)
        yw = np.mod(y + y_shift, 1.0)
        feats = np.column_stack(
            [
                xw,
                yw,
                np.sin(2 * np.pi * xw),
                np.cos(2 * np.pi * yw),
                np.sin(2 * np.pi * (xw + yw + 0.15 * batch_idx)),
                np.cos(4 * np.pi * xw * yw + 0.35 * batch_idx),
            ]
        )
        km = KMeans(
            n_clusters=self.n_domains,
            random_state=int(self.random_seed + 101 * batch_idx + 17),
            n_init=20,
        )
        return km.fit_predict(feats).astype(int)

    def _create_domains(self, coords_local: np.ndarray, batch_idx: int) -> np.ndarray:
        if self.pattern_mode == "simple":
            return self._create_domains_simple(coords_local)
        return self._create_domains_complex(coords_local, batch_idx=batch_idx)

    def _assign_cell_types(self, domain_labels: np.ndarray, batch_idx: int) -> np.ndarray:
        cell_types = np.zeros(len(domain_labels), dtype=int)
        for i in range(len(domain_labels)):
            d = int(domain_labels[i])
            probs = self.batch_domain_ct_probs[batch_idx, d]
            cell_types[i] = int(self.rng.choice(self.n_cell_types, p=probs))
        return cell_types

    def _generate_expression(self, cell_types: np.ndarray, domain_labels: np.ndarray) -> np.ndarray:
        expression = np.zeros((len(cell_types), self.n_genes), dtype=np.float32)
        for i in range(len(cell_types)):
            ct = int(cell_types[i])
            domain = int(domain_labels[i])

            dropout = self.rng.rand(self.n_genes) < 0.7
            vals = self.rng.negative_binomial(2, 0.3, size=self.n_genes).astype(np.float32)
            vals[dropout] = 0
            expression[i] = vals

            d_start = domain * self.domain_markers_per_domain
            d_end = d_start + self.domain_markers_per_domain
            domain_boost = self.rng.negative_binomial(4, 0.24, size=self.domain_markers_per_domain) + 14
            expression[i, d_start:d_end] += domain_boost.astype(np.float32)

            ct_start = self.domain_marker_total + ct * self.celltype_markers_per_type
            ct_end = ct_start + self.celltype_markers_per_type
            ct_boost = self.rng.negative_binomial(5, 0.22, size=self.celltype_markers_per_type) + 20
            expression[i, ct_start:ct_end] += ct_boost.astype(np.float32)

            if ct == 0:
                expression[i, self.ligand_gene_idx] += float(self.rng.negative_binomial(5, 0.2) + 30)
            if ct == 1:
                expression[i, self.receptor_gene_idx] += float(self.rng.negative_binomial(5, 0.2) + 25)

        return expression

    def _get_grid_neighbors(self, k_hop: int = 3) -> np.ndarray:
        n = self.n_cells_per_batch
        gs = self.grid_size
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            row_i, col_i = divmod(i, gs[1])
            for j in range(n):
                if i == j:
                    continue
                row_j, col_j = divmod(j, gs[1])
                if abs(row_i - row_j) + abs(col_i - col_j) <= k_hop:
                    adj[i, j] = True
        return adj

    def _apply_ccc_effects(self, expression: np.ndarray, cell_types: np.ndarray, diffusion_hops: int = 3) -> np.ndarray:
        modified_expr = expression.copy()
        adj = self._get_grid_neighbors(k_hop=diffusion_hops)
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        kernel = adj.astype(float) / row_sums
        ligand_conc = kernel @ expression[:, self.ligand_gene_idx]
        target_idx = list(range(self.target_start_idx, self.target_end_idx))
        for i in range(len(cell_types)):
            if int(cell_types[i]) == 1:
                receptor_level = expression[i, self.receptor_gene_idx]
                signal = ligand_conc[i] * receptor_level / 100.0
                for t_idx in target_idx:
                    if signal > 0.3:
                        modified_expr[i, t_idx] += float(signal * self.rng.uniform(15, 40))
        return modified_expr

    def _calculate_oracles(self, expression: np.ndarray, cell_types: np.ndarray, coords_local: np.ndarray) -> dict[str, np.ndarray]:
        nbrs = NearestNeighbors(n_neighbors=min(11, len(coords_local))).fit(coords_local)
        _, idx = nbrs.kneighbors(coords_local)
        neigh = idx[:, 1:]
        ct_heterogeneity = np.zeros(len(cell_types), dtype=np.float32)
        for i in range(len(cell_types)):
            if neigh.shape[1] == 0:
                continue
            ct_heterogeneity[i] = 1.0 - float(np.mean(cell_types[neigh[i]] == cell_types[i]))
        ct_match = 1.0 - ct_heterogeneity

        target_idx = list(range(self.target_start_idx, self.target_end_idx))
        ccc_activity = expression[:, target_idx].mean(axis=1)
        if float(ccc_activity.max()) > 0:
            ccc_activity = ccc_activity / float(ccc_activity.max())
        return {
            "ct_heterogeneity": ct_heterogeneity,
            "ct_match": ct_match,
            "ccc_activity": ccc_activity.astype(np.float32),
        }

    def simulate(self) -> ad.AnnData:
        obs_frames: list[pd.DataFrame] = []
        xs: list[np.ndarray] = []
        coords_all: list[np.ndarray] = []

        for batch_idx in range(self.n_batches):
            coords_local = self._generate_coordinates()
            domain_labels = self._create_domains(coords_local, batch_idx=batch_idx)
            cell_types = self._assign_cell_types(domain_labels, batch_idx=batch_idx)
            expression = self._generate_expression(cell_types, domain_labels)
            expression = self._apply_ccc_effects(expression, cell_types)
            oracles = self._calculate_oracles(expression, cell_types, coords_local)

            batch_name = f"b{batch_idx}"
            y_offset = batch_idx * (self.grid_size[1] + 4)
            coords_shifted = coords_local.copy()
            coords_shifted[:, 1] += y_offset

            obs_frames.append(
                pd.DataFrame(
                    {
                        "batch": batch_name,
                        "cell_type_id": cell_types,
                        "spatial_domain_id": domain_labels,
                        "cell_type": [self.cell_type_names[i] for i in cell_types],
                        "spatial_domain": [self.domain_names[i] for i in domain_labels],
                        "oracle_ct_heterogeneity": oracles["ct_heterogeneity"],
                        "oracle_ct_match": oracles["ct_match"],
                        "oracle_ccc_activity": oracles["ccc_activity"],
                        "array_row": coords_shifted[:, 0],
                        "array_col": coords_shifted[:, 1],
                    }
                )
            )
            xs.append(expression)
            coords_all.append(coords_shifted)

        obs = pd.concat(obs_frames, ignore_index=True)
        x = np.vstack(xs).astype(np.float32)
        coords = np.vstack(coords_all).astype(np.float32)

        gene_names: list[str] = []
        for d in range(self.n_domains):
            gene_names.extend([f"Domain{d}_marker_{i}" for i in range(self.domain_markers_per_domain)])
        for ct in range(self.n_cell_types):
            gene_names.extend([f"CT{ct}_marker_{i}" for i in range(self.celltype_markers_per_type)])
        gene_names.append("Ligand")
        gene_names.append("Receptor")
        gene_names.extend([f"Target_{i}" for i in range(self.n_ccc_targets)])
        gene_names.extend([f"Gene_{i}" for i in range(self.n_genes - len(gene_names))])

        adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=gene_names), obsm={"spatial": coords})
        adata.layers["counts"] = x.copy()
        return adata


def permute_expression_within_batch(adata: ad.AnnData, batch_key: str, seed: int) -> ad.AnnData:
    out = adata.copy()
    x = to_dense_array(out.X)
    rng = np.random.RandomState(seed)
    batches = out.obs[batch_key].astype(str).to_numpy()
    for batch in np.unique(batches):
        mask = batches == batch
        sub = x[mask].copy()
        for g in range(sub.shape[1]):
            rng.shuffle(sub[:, g])
        x[mask] = sub
    out.X = x
    return out


def _cal_chaos(coords: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    if len(coords) < 3:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    neigh = idx[:, 1:]
    chaos_vals = []
    for i in range(len(coords)):
        same = neigh[i][labels[neigh[i]] == labels[i]]
        if len(same) == 0:
            chaos_vals.append(1.0)
            continue
        d = np.linalg.norm(coords[same] - coords[i], axis=1)
        chaos_vals.append(float(np.mean(d)))
    raw = float(np.mean(chaos_vals))
    return raw / (raw + 1.0)


def _cal_pas(coords: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    if len(coords) < 3:
        return 1.0
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    neigh = idx[:, 1:]
    scores = []
    for i in range(len(coords)):
        scores.append(float(np.mean(labels[neigh[i]] == labels[i])))
    return float(np.mean(scores))


def _cal_batch_asw(emb: np.ndarray, batch_true: np.ndarray) -> float:
    if len(np.unique(batch_true)) < 2:
        return 1.0
    try:
        sil = silhouette_score(emb, batch_true, metric="euclidean")
    except Exception:
        return 0.0
    return float((1.0 - sil) / 2.0)


def _cal_batch_ilisi(emb: np.ndarray, batch_true: np.ndarray, k: int = 30) -> float:
    if len(np.unique(batch_true)) < 2:
        return 1.0
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(emb))).fit(emb)
    _, idx = nbrs.kneighbors(emb)
    neigh = idx[:, 1:]
    batch_levels = np.unique(batch_true)
    ilis = []
    for i in range(len(emb)):
        neigh_batches = batch_true[neigh[i]]
        p = np.array([np.mean(neigh_batches == b) for b in batch_levels], dtype=float)
        denom = float(np.sum(p ** 2))
        ilis.append(1.0 / denom if denom > 0 else 1.0)
    return float(np.mean(ilis))


def _cal_batch_silhouette(emb: np.ndarray, batch_true: np.ndarray) -> float:
    if len(np.unique(batch_true)) < 2:
        return 0.0
    try:
        return float(silhouette_score(emb, batch_true, metric="euclidean"))
    except Exception:
        return 0.0


def neighbor_label_agreement(labels: np.ndarray, coords: np.ndarray, k: int = 10) -> float:
    if len(coords) < 3:
        return 1.0
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    neigh = idx[:, 1:]
    return float(np.mean([np.mean(labels[neigh[i]] == labels[i]) for i in range(len(labels))]))


def run_step_trial(adata: ad.AnnData, cfg: dict[str, Any], return_model: bool = False) -> tuple[dict[str, Any], Any | None]:
    from step import stModel
    from step.utils.misc import set_seed

    t0 = time.time()
    train_seed = int(cfg.get("train_seed", 42))
    set_seed(train_seed)

    model_kwargs: dict[str, Any] = {
        "adata": adata.copy(),
        "coord_keys": ("array_row", "array_col"),
        "n_top_genes": int(cfg.get("n_top_genes", 2000)),
        "module_dim": int(cfg.get("module_dim", 30)),
        "hidden_dim": int(cfg.get("hidden_dim", 96)),
        "n_modules": int(cfg.get("n_modules", 8)),
        "decoder_type": "zinb",
        "variational": bool(cfg.get("variational", False)),
        "n_glayers": int(cfg.get("n_glayers", 6)),
        "n_enc_layer": int(cfg.get("n_enc_layer", 3)),
    }
    if "batch" in adata.obs.columns and len(np.unique(adata.obs["batch"])) > 1 and bool(cfg.get("use_bem", True)):
        model_kwargs["batch_key"] = "batch"
    edge_clip_cfg = cfg.get("edge_clip", 1.5)
    if edge_clip_cfg is not None:
        model_kwargs["edge_clip"] = float(edge_clip_cfg)
    if cfg.get("max_neighbors") is not None:
        model_kwargs["max_neighbors"] = int(cfg["max_neighbors"])

    model = stModel(**model_kwargs)
    if str(cfg.get("tag", "")) == "ablate_spm" and hasattr(model, "model") and hasattr(model.model, "smoother"):
        model.model.smoother = IdentitySmoother()

    sample_rate_cfg = cfg.get("sample_rate", 1.0)
    run_kwargs: dict[str, Any] = {
        "n_iterations": int(cfg.get("n_iterations", 120)),
        "sampling": "saint",
        "contrast": bool(cfg.get("contrast", False)),
        "e2e": True,
        "lr": float(cfg.get("lr", 1.8e-3)),
        "beta": float(cfg.get("beta", 1e-4)),
    }
    if cfg.get("graph_batch_size") is not None:
        run_kwargs["graph_batch_size"] = int(cfg["graph_batch_size"])
    if sample_rate_cfg is not None:
        if isinstance(sample_rate_cfg, int):
            run_kwargs["sample_rate"] = int(sample_rate_cfg)
        else:
            run_kwargs["sample_rate"] = float(sample_rate_cfg)
    if cfg.get("graph_loss_mode") is not None:
        run_kwargs["graph_loss_mode"] = str(cfg["graph_loss_mode"])
    model.run(**run_kwargs)

    if "X_rep" not in model.adata.obsm_keys():
        model.adata.obsm["X_rep"] = np.asarray(model.embed(), dtype=np.float32)
    emb_individual = np.asarray(model.adata.obsm["X_rep"], dtype=np.float32)
    emb_spatial = np.asarray(model.adata.obsm["X_smoothed"], dtype=np.float32)
    if str(cfg.get("tag", "")) == "ablate_spm":
        emb_spatial = emb_individual

    domain_true = model.adata.obs["spatial_domain"].astype(str).to_numpy()
    ct_true = model.adata.obs["cell_type"].astype(str).to_numpy()
    batch_true = model.adata.obs["batch"].astype(str).to_numpy() if "batch" in model.adata.obs else np.array(["b0"] * model.adata.n_obs)
    coords = model.adata.obs[["array_row", "array_col"]].to_numpy(dtype=float)

    n_domain = len(np.unique(domain_true))
    n_ct = len(np.unique(ct_true))
    pred_domain = KMeans(n_clusters=n_domain, random_state=int(cfg.get("cluster_seed", 42)), n_init=10).fit_predict(emb_spatial)
    pred_ct = KMeans(n_clusters=n_ct, random_state=int(cfg.get("cluster_seed", 42)) + 11, n_init=10).fit_predict(emb_individual)

    batch_sil = _cal_batch_silhouette(emb_individual, batch_true)
    batch_sil_spatial = _cal_batch_silhouette(emb_spatial, batch_true)
    spatial_chaos = _cal_chaos(coords, pred_domain, k=10)
    spatial_pas = _cal_pas(coords, pred_domain, k=10)

    rec: dict[str, Any] = dict(cfg)
    rec.update(
        {
            "domain_ari": float(adjusted_rand_score(domain_true, pred_domain)),
            "domain_nmi": float(normalized_mutual_info_score(domain_true, pred_domain)),
            "celltype_ari": float(adjusted_rand_score(ct_true, pred_ct)),
            "celltype_nmi": float(normalized_mutual_info_score(ct_true, pred_ct)),
            "batch_silhouette": batch_sil,
            "batch_asw": _cal_batch_asw(emb_individual, batch_true),
            "batch_ilisi": _cal_batch_ilisi(emb_individual, batch_true),
            "batch_silhouette_spatial": batch_sil_spatial,
            "batch_asw_spatial": _cal_batch_asw(emb_spatial, batch_true),
            "batch_ilisi_spatial": _cal_batch_ilisi(emb_spatial, batch_true),
            "domain_neighbor_agreement": neighbor_label_agreement(pred_domain, coords, k=10),
            "spatial_chaos": spatial_chaos,
            "spatial_pas": spatial_pas,
            "elapsed_sec": time.time() - t0,
        }
    )

    model.adata.obs[f"pred_domain_{cfg['tag']}"] = pd.Categorical(pred_domain.astype(str))
    model.adata.obs[f"pred_cell_type_{cfg['tag']}"] = pd.Categorical(pred_ct.astype(str))

    if return_model:
        return rec, model
    model.model.cpu()
    return rec, None


def aggregate_by_tag(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    df = pd.DataFrame(records)
    out: dict[str, dict[str, float]] = {}
    for tag, sub in df.groupby("tag"):
        row: dict[str, float] = {"n": float(len(sub))}
        for key in METRIC_KEYS:
            if key in sub.columns:
                row[f"{key}_mean"] = float(sub[key].mean())
                row[f"{key}_std"] = float(sub[key].std(ddof=0))
        out[str(tag)] = row
    return out


def paired_diffs(records: list[dict[str, Any]], ref_tag: str, other_tag: str) -> dict[str, float]:
    ref = {int(r["repeat"]): r for r in records if r["tag"] == ref_tag}
    other = {int(r["repeat"]): r for r in records if r["tag"] == other_tag}
    common = sorted(set(ref) & set(other))
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = [float(ref[i][key]) - float(other[i][key]) for i in common if key in ref[i] and key in other[i]]
        arr = np.array(vals, dtype=float)
        out[f"{key}_mean_diff"] = float(arr.mean()) if len(arr) else 0.0
        out[f"{key}_std_diff"] = float(arr.std(ddof=0)) if len(arr) else 0.0
        out[f"{key}_positive_frac"] = float((arr > 0).mean()) if len(arr) else 0.0
    out["n_pairs"] = float(len(common))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid-rows", type=int, default=24)
    p.add_argument("--grid-cols", type=int, default=24)
    p.add_argument("--n-genes", type=int, default=2400)
    p.add_argument("--n-domains", type=int, default=6)
    p.add_argument("--n-cell-types", type=int, default=3)
    p.add_argument("--n-batches", type=int, default=2)
    p.add_argument("--pattern-mode", type=str, choices=["simple", "complex"], default="complex")
    p.add_argument("--batch-effect", type=float, default=0.8)
    p.add_argument("--n-iterations", type=int, default=120)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--module-dim", type=int, default=30)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--n-modules", type=int, default=8)
    p.add_argument("--variational", action="store_true")
    p.add_argument("--n-glayers", type=int, default=6)
    p.add_argument("--n-enc-layer", type=int, default=3)
    p.add_argument("--edge-clip", type=float, default=1.5)
    p.add_argument("--graph-loss-mode", type=str, default="legacy")
    p.add_argument("--sample-rate", type=float, default=1.0)
    p.add_argument("--graph-batch-size", type=int, default=1)
    p.add_argument("--contrast", action="store_true")
    p.add_argument("--lr", type=float, default=1.8e-3)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("workflows/module_ablation"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_jsonl = out_dir / f"step_module_ablation_{ts}.jsonl"
    sim_paths: list[str] = []
    records: list[dict[str, Any]] = []
    with out_jsonl.open("w") as f:
        for repeat in range(args.n_repeats):
            sim_seed = int(args.seed) + repeat
            train_seed = int(args.seed) + 11 + repeat
            sim = SpatialCoherenceSimulator(
                grid_size=(args.grid_rows, args.grid_cols),
                n_genes=args.n_genes,
                n_domains=args.n_domains,
                n_cell_types=args.n_cell_types,
                n_batches=args.n_batches,
                batch_effect_strength=args.batch_effect,
                pattern_mode=args.pattern_mode,
                random_seed=sim_seed,
            )
            adata = sim.simulate()
            sim_path = out_dir / f"step_module_ablation_{ts}_repeat{repeat + 1}.h5ad"
            adata.write_h5ad(sim_path)
            sim_paths.append(str(sim_path))

            common = {
                "n_top_genes": args.n_top_genes,
                "module_dim": args.module_dim,
                "hidden_dim": args.hidden_dim,
                "n_modules": args.n_modules,
                "variational": args.variational,
                "n_glayers": args.n_glayers,
                "n_enc_layer": args.n_enc_layer,
                "edge_clip": args.edge_clip,
                "graph_loss_mode": args.graph_loss_mode,
                "n_iterations": args.n_iterations,
                "sample_rate": args.sample_rate,
                "graph_batch_size": args.graph_batch_size,
                "contrast": args.contrast,
                "lr": args.lr,
                "beta": args.beta,
                "train_seed": train_seed,
                "cluster_seed": 42,
                "use_bem": True,
            }

            full_cfg = {"tag": "full_bbm_bem_spm", **common}
            full_rec, full_model = run_step_trial(adata, full_cfg, return_model=True)
            full_rec.update({"repeat": repeat + 1, "trial": 1, "sim_seed": sim_seed, "sim_path": str(sim_path), "data_mode": "original"})
            records.append(full_rec)
            f.write(json.dumps(full_rec) + "\n")
            f.flush()

            if full_model is not None:
                full_model.model.cpu()

            trial_specs: list[tuple[ad.AnnData, dict[str, Any]]] = [
                (adata, {"tag": "ablate_spm", **common, "n_glayers": 0, "data_mode": "original"}),
                (adata, {"tag": "ablate_bem", **common, "use_bem": False, "data_mode": "original"}),
                (
                    adata,
                    {
                        "tag": "ablate_bbm_lowcap",
                        **common,
                        "n_modules": 1,
                        "module_dim": 8,
                        "hidden_dim": 16,
                        "n_enc_layer": 1,
                        "data_mode": "original",
                    },
                ),
            ]

            for offset, (trial_adata, trial_cfg) in enumerate(trial_specs, start=2):
                run_cfg = {k: v for k, v in trial_cfg.items() if k != "data_mode"}
                rec, _ = run_step_trial(trial_adata, run_cfg, return_model=False)
                rec.update({"repeat": repeat + 1, "trial": offset, "sim_seed": sim_seed, "sim_path": str(sim_path), "data_mode": str(trial_cfg["data_mode"])})
                records.append(rec)
                f.write(json.dumps(rec) + "\n")
                f.flush()

    serializable_args = {
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "n_genes": args.n_genes,
        "n_domains": args.n_domains,
        "n_cell_types": args.n_cell_types,
        "n_batches": args.n_batches,
        "pattern_mode": args.pattern_mode,
        "batch_effect": args.batch_effect,
        "n_iterations": args.n_iterations,
        "n_repeats": args.n_repeats,
        "n_top_genes": args.n_top_genes,
        "module_dim": args.module_dim,
        "hidden_dim": args.hidden_dim,
        "n_modules": args.n_modules,
        "variational": args.variational,
        "n_glayers": args.n_glayers,
        "n_enc_layer": args.n_enc_layer,
        "edge_clip": args.edge_clip,
        "graph_loss_mode": args.graph_loss_mode,
        "sample_rate": args.sample_rate,
        "graph_batch_size": args.graph_batch_size,
        "contrast": args.contrast,
        "lr": args.lr,
        "beta": args.beta,
        "seed": args.seed,
        "out_dir": str(args.out_dir),
    }

    by_tag = aggregate_by_tag(records)
    deltas = {
        "full_vs_ablate_spm": paired_diffs(records, "full_bbm_bem_spm", "ablate_spm"),
        "full_vs_ablate_bem": paired_diffs(records, "full_bbm_bem_spm", "ablate_bem"),
        "full_vs_ablate_bbm_lowcap": paired_diffs(records, "full_bbm_bem_spm", "ablate_bbm_lowcap"),
    }

    spm_delta = deltas["full_vs_ablate_spm"]
    bem_delta = deltas["full_vs_ablate_bem"]
    bbm_delta = deltas["full_vs_ablate_bbm_lowcap"]

    checks = {
        "full_beats_no_spm_on_spatial_continuity": spm_delta["spatial_pas_mean_diff"] > 0.0 and spm_delta["spatial_chaos_mean_diff"] < 0.0,
        "full_beats_no_bem_on_batch_mixing": (
            bem_delta["batch_asw_mean_diff"] > 0.0
            and bem_delta["batch_ilisi_mean_diff"] > 0.0
            and bem_delta["batch_asw_spatial_mean_diff"] > 0.0
            and bem_delta["batch_ilisi_spatial_mean_diff"] > 0.0
        ),
        "bbm_capacity_required_for_celltype_structure": (
            bbm_delta["celltype_ari_mean_diff"] > 0.15
            and bbm_delta["celltype_ari_positive_frac"] >= 0.67
        ),
    }

    summary = {
        "arguments": serializable_args,
        "sim_data_files": sim_paths,
        "result_jsonl": str(out_jsonl),
        "records": records,
        "aggregate_by_tag": by_tag,
        "paired_deltas": deltas,
        "hypothesis_checks": checks,
    }
    summary_path = out_dir / f"step_module_ablation_{ts}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
