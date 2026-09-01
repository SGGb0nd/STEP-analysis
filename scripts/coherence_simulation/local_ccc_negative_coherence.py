"""Simulate local communication-driven expression changes and validate spatial coherence."""

from __future__ import annotations

import warnings
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    vmax = float(np.max(x)) if x.size else 0.0
    if vmax <= 0:
        return np.zeros_like(x, dtype=float)
    return x / vmax


@dataclass
class CCCValidationResults:
    pearson_total_ccc: float
    spearman_total_ccc: float
    pearson_ct1_ct1: float
    spearman_ct1_ct1: float
    pearson_ct1_ct2: float
    spearman_ct1_ct2: float
    pearson_incoherence_prog: float
    spearman_incoherence_prog: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "pearson_total_ccc": self.pearson_total_ccc,
            "spearman_total_ccc": self.spearman_total_ccc,
            "pearson_ct1_ct1": self.pearson_ct1_ct1,
            "spearman_ct1_ct1": self.spearman_ct1_ct1,
            "pearson_ct1_ct2": self.pearson_ct1_ct2,
            "spearman_ct1_ct2": self.spearman_ct1_ct2,
            "pearson_incoherence_prog": self.pearson_incoherence_prog,
            "spearman_incoherence_prog": self.spearman_incoherence_prog,
        }


class LocalCCCSimulator:
    """
    Local CCC-driven simulation in which the primary validation target is a
    continuous negative correlation between STEP coherence and CCC strength.
    """

    def __init__(self, grid_size: Tuple[int, int] = (50, 50), n_genes: int = 240, random_seed: int = 42) -> None:
        if n_genes < 240:
            raise ValueError("LocalCCCSimulator requires n_genes >= 240 because the oracle modules use fixed gene-index blocks.")
        self.grid_size = grid_size
        self.n_cells = grid_size[0] * grid_size[1]
        self.n_genes = n_genes
        self.rng = np.random.RandomState(random_seed)

        self.cell_state_names = [
            "ct1_silent",
            "ct1_ligand",
            "ct1_receptor",
            "ct2_silent",
            "ct2_ligand",
            "ct2_receptor",
        ]

        self.ct1_prog_idx = list(range(0, 30))
        self.ct2_prog_idx = list(range(30, 60))
        self.ct1_ligand_idx = list(range(60, 65))
        self.ct2_ligand_idx = list(range(65, 70))
        self.ct1_receptor_idx = list(range(70, 75))
        self.ct2_receptor_idx = list(range(75, 80))
        self.ct1_ct1_response_idx = list(range(80, 120))
        self.ct1_ct2_response_idx = list(range(120, 160))
        self.shared_noise_prog_idx = list(range(160, 200))
        self.bg_idx = list(range(200, n_genes))

    def generate_coordinates(self) -> np.ndarray:
        x, y = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), indexing="ij")
        return np.column_stack([x.ravel(), y.ravel()]).astype(float)

    def build_radius_adjacency(self, coords: np.ndarray, radius: float = 2.5) -> np.ndarray:
        dist = cdist(coords, coords)
        return (dist > 0) & (dist <= radius)

    def assign_cell_states(self) -> np.ndarray:
        probs = np.array([0.22, 0.10, 0.13, 0.22, 0.10, 0.23])
        return self.rng.choice(6, size=self.n_cells, p=probs)

    def inject_local_ccc_hotspots(self, states: np.ndarray, coords: np.ndarray, n_hotspots: int = 5, hotspot_radius: float = 4.5) -> Tuple[np.ndarray, np.ndarray]:
        states = states.copy()
        hotspot_score = np.zeros(self.n_cells, dtype=float)

        gsx, gsy = self.grid_size
        centers = [
            (self.rng.uniform(0.15 * gsx, 0.85 * gsx), self.rng.uniform(0.15 * gsy, 0.85 * gsy))
            for _ in range(n_hotspots)
        ]

        for cx, cy in centers:
            d = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
            patch_idx = np.where(d <= hotspot_radius)[0]
            hotspot_score[patch_idx] += 1 - d[patch_idx] / hotspot_radius

            for idx in patch_idx:
                base_ct = 0 if states[idx] in [0, 1, 2] else 1
                r = self.rng.rand()
                if base_ct == 0:
                    if r < 0.40:
                        states[idx] = 1
                    elif r < 0.75:
                        states[idx] = 2
                    else:
                        states[idx] = 0
                else:
                    if r < 0.45:
                        states[idx] = 5
                    elif r < 0.60:
                        states[idx] = 4
                    else:
                        states[idx] = 3

        return states, np.clip(hotspot_score, 0, 1)

    def generate_base_expression(self, states: np.ndarray) -> np.ndarray:
        expr = np.zeros((self.n_cells, self.n_genes), dtype=np.float32)
        for i, st in enumerate(states):
            bg = self.rng.negative_binomial(1, 0.55, size=self.n_genes).astype(np.float32)
            bg[self.rng.rand(self.n_genes) < 0.82] = 0
            expr[i] = bg

            if st in [0, 1, 2]:
                expr[i, self.ct1_prog_idx] += self.rng.negative_binomial(3, 0.30, size=len(self.ct1_prog_idx)) + 4
            else:
                expr[i, self.ct2_prog_idx] += self.rng.negative_binomial(3, 0.30, size=len(self.ct2_prog_idx)) + 4

            if st == 1:
                expr[i, self.ct1_ligand_idx] += self.rng.negative_binomial(6, 0.22, size=len(self.ct1_ligand_idx)) + 30
            elif st == 2:
                expr[i, self.ct1_receptor_idx] += self.rng.negative_binomial(6, 0.22, size=len(self.ct1_receptor_idx)) + 28
            elif st == 4:
                expr[i, self.ct2_ligand_idx] += self.rng.negative_binomial(6, 0.22, size=len(self.ct2_ligand_idx)) + 30
            elif st == 5:
                expr[i, self.ct2_receptor_idx] += self.rng.negative_binomial(6, 0.22, size=len(self.ct2_receptor_idx)) + 28
        return expr

    def apply_local_ccc(self, expr: np.ndarray, states: np.ndarray, coords: np.ndarray, radius: float = 2.5, effect_strength: float = 1.2, incoherence_strength: float = 1.5) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        expr2 = expr.copy()
        adj = self.build_radius_adjacency(coords, radius=radius)

        is_ct1_lig = states == 1
        is_ct1_rec = states == 2
        is_ct2_rec = states == 5
        ligand_strength = expr[:, self.ct1_ligand_idx].mean(axis=1)
        ct1_rec_strength = expr[:, self.ct1_receptor_idx].mean(axis=1)
        ct2_rec_strength = expr[:, self.ct2_receptor_idx].mean(axis=1)

        recv_ct1_ct1 = np.zeros(self.n_cells, dtype=float)
        recv_ct1_ct2 = np.zeros(self.n_cells, dtype=float)
        send_ct1_ct1 = np.zeros(self.n_cells, dtype=float)
        send_ct1_ct2 = np.zeros(self.n_cells, dtype=float)

        for i in range(self.n_cells):
            nbrs = np.where(adj[i])[0]
            if len(nbrs) == 0:
                continue

            if is_ct1_rec[i]:
                donor_idx = nbrs[is_ct1_lig[nbrs]]
                if len(donor_idx) > 0:
                    nearby_lig = ligand_strength[donor_idx].sum()
                    rec_strength = expr[i, self.ct1_receptor_idx].mean()
                    recv_ct1_ct1[i] = nearby_lig * rec_strength

            if is_ct2_rec[i]:
                donor_idx = nbrs[is_ct1_lig[nbrs]]
                if len(donor_idx) > 0:
                    nearby_lig = ligand_strength[donor_idx].sum()
                    rec_strength = expr[i, self.ct2_receptor_idx].mean()
                    recv_ct1_ct2[i] = nearby_lig * rec_strength

            if is_ct1_lig[i]:
                ct1_neighbors = nbrs[is_ct1_rec[nbrs]]
                ct2_neighbors = nbrs[is_ct2_rec[nbrs]]
                lig_strength = expr[i, self.ct1_ligand_idx].mean()
                if len(ct1_neighbors) > 0:
                    send_ct1_ct1[i] = lig_strength * ct1_rec_strength[ct1_neighbors].sum()
                if len(ct2_neighbors) > 0:
                    send_ct1_ct2[i] = lig_strength * ct2_rec_strength[ct2_neighbors].sum()

        ccc_ct1_ct1 = _normalize(recv_ct1_ct1 + send_ct1_ct1)
        ccc_ct1_ct2 = _normalize(recv_ct1_ct2 + send_ct1_ct2)
        ccc_total = np.clip(ccc_ct1_ct1 + ccc_ct1_ct2, 0, 1)

        modules_ct1_ct1 = np.array_split(self.ct1_ct1_response_idx, 4)
        modules_ct1_ct2 = np.array_split(self.ct1_ct2_response_idx, 4)

        for i in np.where(ccc_ct1_ct1 > 0)[0]:
            strength = ccc_ct1_ct1[i]
            n_mod = max(1, int(np.ceil(strength * 4)))
            chosen = self.rng.choice(len(modules_ct1_ct1), size=n_mod, replace=False)
            for mod_idx, genes in enumerate(modules_ct1_ct1):
                scale = self.rng.uniform(15, 40, size=len(genes)) if mod_idx in chosen else self.rng.uniform(0, 8, size=len(genes))
                expr2[i, genes] += scale * strength * effect_strength

        for i in np.where(ccc_ct1_ct2 > 0)[0]:
            strength = ccc_ct1_ct2[i]
            n_mod = max(1, int(np.ceil(strength * 4)))
            chosen = self.rng.choice(len(modules_ct1_ct2), size=n_mod, replace=False)
            for mod_idx, genes in enumerate(modules_ct1_ct2):
                scale = self.rng.uniform(18, 45, size=len(genes)) if mod_idx in chosen else self.rng.uniform(0, 8, size=len(genes))
                expr2[i, genes] += scale * strength * effect_strength

        noise_genes = self.ct1_ct1_response_idx + self.ct1_ct2_response_idx + self.shared_noise_prog_idx
        for i in np.where(ccc_total > 0)[0]:
            strength = ccc_total[i]
            role_scale = 1.0 if states[i] in [2, 5] else 0.8
            if states[i] in [0, 1, 2]:
                native_prog = self.ct1_prog_idx
                opposite_prog = self.ct2_prog_idx
            else:
                native_prog = self.ct2_prog_idx
                opposite_prog = self.ct1_prog_idx

            native_scale = max(0.08, 1.0 - 0.90 * strength * incoherence_strength * role_scale)
            expr2[i, native_prog] *= native_scale
            expr2[i, opposite_prog] += self.rng.uniform(12, 30, size=len(opposite_prog)) * strength * incoherence_strength * role_scale

            chosen = self.rng.choice(self.shared_noise_prog_idx, size=max(3, int(len(self.shared_noise_prog_idx) * 0.25 * strength + 3)), replace=False)
            expr2[i, chosen] += self.rng.uniform(10, 35, size=len(chosen)) * strength * incoherence_strength
            for genes in (self.ct1_ct1_response_idx, self.ct1_ct2_response_idx):
                expr2[i, genes] += self.rng.uniform(28, 80, size=len(genes)) * strength * incoherence_strength * role_scale
            expr2[i, noise_genes] += self.rng.normal(loc=0, scale=(2.0 + 14.0 * strength * incoherence_strength * role_scale), size=len(noise_genes)).astype(np.float32)

        expr2 = np.clip(expr2, 0, None)
        delta = np.clip(expr2 - expr, 0, None)
        oracles = {
            "oracle_ccc_ct1_ct1": ccc_ct1_ct1,
            "oracle_ccc_ct1_ct2": ccc_ct1_ct2,
            "oracle_ccc_total": ccc_total,
            "oracle_response_ct1_ct1": _normalize(delta[:, self.ct1_ct1_response_idx].mean(axis=1)),
            "oracle_response_ct1_ct2": _normalize(delta[:, self.ct1_ct2_response_idx].mean(axis=1)),
            "oracle_incoherence_prog": _normalize(delta[:, self.shared_noise_prog_idx].mean(axis=1)),
        }
        return expr2, oracles

    def simulate(self, ccc_radius: float = 2.5, effect_strength: float = 1.2, incoherence_strength: float = 1.5, n_hotspots: int = 5, hotspot_radius: float = 4.5) -> ad.AnnData:
        coords = self.generate_coordinates()
        states = self.assign_cell_states()
        states, hotspot_score = self.inject_local_ccc_hotspots(states, coords, n_hotspots=n_hotspots, hotspot_radius=hotspot_radius)
        expr0 = self.generate_base_expression(states)
        expr, oracles = self.apply_local_ccc(expr0, states, coords, radius=ccc_radius, effect_strength=effect_strength, incoherence_strength=incoherence_strength)

        gene_names = [f"Gene_{i}" for i in range(self.n_genes)]
        for i, idx in enumerate(self.ct1_ligand_idx):
            gene_names[idx] = f"CT1_Ligand_{i}"
        for i, idx in enumerate(self.ct2_ligand_idx):
            gene_names[idx] = f"CT2_Ligand_{i}"
        for i, idx in enumerate(self.ct1_receptor_idx):
            gene_names[idx] = f"CT1_Receptor_{i}"
        for i, idx in enumerate(self.ct2_receptor_idx):
            gene_names[idx] = f"CT2_Receptor_{i}"
        for i, idx in enumerate(self.ct1_ct1_response_idx):
            gene_names[idx] = f"Resp_CT1_CT1_{i}"
        for i, idx in enumerate(self.ct1_ct2_response_idx):
            gene_names[idx] = f"Resp_CT1_CT2_{i}"
        for i, idx in enumerate(self.shared_noise_prog_idx):
            gene_names[idx] = f"Incoh_{i}"

        adata = ad.AnnData(
            X=expr,
            obs=pd.DataFrame(
                {
                    "cell_state": [self.cell_state_names[s] for s in states],
                    "cell_state_id": states,
                    "oracle_ccc_ct1_ct1": oracles["oracle_ccc_ct1_ct1"],
                    "oracle_ccc_ct1_ct2": oracles["oracle_ccc_ct1_ct2"],
                    "oracle_ccc_total": oracles["oracle_ccc_total"],
                    "oracle_response_ct1_ct1": oracles["oracle_response_ct1_ct1"],
                    "oracle_response_ct1_ct2": oracles["oracle_response_ct1_ct2"],
                    "oracle_incoherence_prog": oracles["oracle_incoherence_prog"],
                    "hotspot_score": hotspot_score,
                }
            ),
            var=pd.DataFrame(index=gene_names),
            obsm={"spatial": coords},
        )
        adata.layers["counts"] = expr.copy()
        adata.uns["simulation_params"] = {
            "grid_size": list(self.grid_size),
            "ccc_radius": ccc_radius,
            "effect_strength": effect_strength,
            "incoherence_strength": incoherence_strength,
            "n_hotspots": n_hotspots,
            "hotspot_radius": hotspot_radius,
            "type": "local_ccc_negative_coherence",
        }
        return adata


def run_step_ccc(adata: ad.AnnData, n_glayers: int = 3):
    from step import stModel

    adata = adata.copy()
    adata.obs[["array_row", "array_col"]] = adata.obsm["spatial"]
    stepc = stModel(
        adata=adata,
        n_top_genes=None,
        geneset_to_use=adata.var_names.to_list(),
        layer_key=None,
        n_modules=12,
        decoder_type="zinb",
        coord_keys=("array_col", "array_row"),
        n_glayers=n_glayers,
        edge_clip=1,
    )
    stepc.run()
    adata = stepc.adata
    adata.obsm["X_rep"] = stepc.embed()
    rep_norm = np.linalg.norm(adata.obsm["X_rep"], axis=1)
    smooth_norm = np.linalg.norm(adata.obsm["X_smoothed"], axis=1)
    adata.obs["step_coherence"] = (adata.obsm["X_rep"] * adata.obsm["X_smoothed"]).sum(1) / (rep_norm * smooth_norm + 1e-8)
    return adata, stepc


def validate_ccc_simulation(adata: ad.AnnData) -> Dict[str, float]:
    total_p = pearsonr(adata.obs["step_coherence"], adata.obs["oracle_ccc_total"])[0]
    total_s = spearmanr(adata.obs["step_coherence"], adata.obs["oracle_ccc_total"])[0]
    ct11_p = pearsonr(adata.obs["step_coherence"], adata.obs["oracle_ccc_ct1_ct1"])[0]
    ct11_s = spearmanr(adata.obs["step_coherence"], adata.obs["oracle_ccc_ct1_ct1"])[0]
    ct12_p = pearsonr(adata.obs["step_coherence"], adata.obs["oracle_ccc_ct1_ct2"])[0]
    ct12_s = spearmanr(adata.obs["step_coherence"], adata.obs["oracle_ccc_ct1_ct2"])[0]
    incoh_p = pearsonr(adata.obs["step_coherence"], adata.obs["oracle_incoherence_prog"])[0]
    incoh_s = spearmanr(adata.obs["step_coherence"], adata.obs["oracle_incoherence_prog"])[0]
    return CCCValidationResults(
        pearson_total_ccc=float(total_p),
        spearman_total_ccc=float(total_s),
        pearson_ct1_ct1=float(ct11_p),
        spearman_ct1_ct1=float(ct11_s),
        pearson_ct1_ct2=float(ct12_p),
        spearman_ct1_ct2=float(ct12_s),
        pearson_incoherence_prog=float(incoh_p),
        spearman_incoherence_prog=float(incoh_s),
    ).as_dict()


def visualize_ccc_validation(adata: ad.AnnData, save_path: str | Path | None = None):
    fig = plt.figure(figsize=(16, 10))
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)]

    sc.pl.embedding(adata, basis="spatial", color="cell_state", ax=axes[0], show=False, title="Cell states", size=25)
    sc.pl.embedding(adata, basis="spatial", color="oracle_ccc_total", ax=axes[1], show=False, title="Continuous CCC strength", size=25, cmap="magma")
    sc.pl.embedding(adata, basis="spatial", color="step_coherence", ax=axes[2], show=False, title="STEP coherence", size=25, cmap="RdYlBu")

    for ax, key, title in [
        (axes[3], "oracle_ccc_total", "Coherence vs total CCC"),
        (axes[4], "oracle_ccc_ct1_ct1", "Coherence vs ct1->ct1 CCC"),
        (axes[5], "oracle_ccc_ct1_ct2", "Coherence vs ct1->ct2 CCC"),
    ]:
        sns.regplot(data=adata.obs, x=key, y="step_coherence", scatter_kws={"alpha": 0.25, "s": 6}, line_kws={"color": "#c62828"}, ax=ax)
        r = pearsonr(adata.obs["step_coherence"], adata.obs[key])[0]
        rho = spearmanr(adata.obs["step_coherence"], adata.obs[key])[0]
        ax.set_title(f"{title}\nPearson r={r:.3f}, Spearman rho={rho:.3f}")
        ax.set_ylabel("STEP coherence")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local CCC-driven spatial coherence simulation.")
    parser.add_argument("--out-dir", type=Path, default=Path("workflows/coherence_simulation"))
    parser.add_argument("--grid-size", type=int, nargs=2, default=(50, 50), metavar=("N_ROWS", "N_COLS"))
    parser.add_argument("--n-genes", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ccc-radius", type=float, default=2.5)
    parser.add_argument("--effect-strength", type=float, default=1.2)
    parser.add_argument("--incoherence-strength", type=float, default=1.8)
    parser.add_argument("--n-hotspots", type=int, default=5)
    parser.add_argument("--hotspot-radius", type=float, default=4.5)
    parser.add_argument("--skip-step", action="store_true", help="Generate only the oracle simulation object without fitting STEP.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    simulator = LocalCCCSimulator(grid_size=tuple(args.grid_size), n_genes=args.n_genes, random_seed=args.seed)
    adata = simulator.simulate(
        ccc_radius=args.ccc_radius,
        effect_strength=args.effect_strength,
        incoherence_strength=args.incoherence_strength,
        n_hotspots=args.n_hotspots,
        hotspot_radius=args.hotspot_radius,
    )
    adata.write(out_dir / "simulation_local_ccc_negative_coherence.h5ad")
    if args.skip_step:
        settings = {
            "grid_size": list(args.grid_size),
            "n_genes": args.n_genes,
            "seed": args.seed,
            "ccc_radius": args.ccc_radius,
            "effect_strength": args.effect_strength,
            "incoherence_strength": args.incoherence_strength,
            "n_hotspots": args.n_hotspots,
            "hotspot_radius": args.hotspot_radius,
            "step_run": False,
        }
        (out_dir / "ccc_simulation_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        return

    adata, _ = run_step_ccc(adata, n_glayers=3)
    results = validate_ccc_simulation(adata)
    visualize_ccc_validation(adata, save_path=out_dir / "ccc_validation.png")
    adata.write(out_dir / "simulation_local_ccc_negative_coherence_with_step.h5ad")
    pd.Series(results).to_json(out_dir / "ccc_validation_metrics.json", indent=2)
    settings = {
        "grid_size": list(args.grid_size),
        "n_genes": args.n_genes,
        "seed": args.seed,
        "ccc_radius": args.ccc_radius,
        "effect_strength": args.effect_strength,
        "incoherence_strength": args.incoherence_strength,
        "n_hotspots": args.n_hotspots,
        "hotspot_radius": args.hotspot_radius,
        "step_run": True,
        "metrics": results,
    }
    (out_dir / "ccc_simulation_settings.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
