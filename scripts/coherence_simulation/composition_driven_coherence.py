"""Simulate composition-driven spatial coherence and validate STEP outputs."""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============== Part 1: Simulation ==============

class SpatialCoherenceSimulator:

    def __init__(
        self,
        grid_size: Tuple[int, int] = (50, 50),
        n_genes: int = 200,
        random_seed: int = 42
    ):
        self.grid_size = grid_size
        self.n_cells = grid_size[0] * grid_size[1]
        self.n_genes = n_genes
        self.rng = np.random.RandomState(random_seed)

        self.domain_names = ['Active', 'Transition', 'Quiescent']
        self.cell_type_names = ['Sender', 'Receiver', 'Neutral']

        # Gene indices
        self.domain_marker_idx = {
            0: list(range(0, 20)),
            1: list(range(20, 40)),
            2: list(range(40, 60)),
        }
        self.ct_marker_idx = {
            0: list(range(60, 80)),
            1: list(range(80, 100)),
            2: list(range(100, 120)),
        }
        self.ligand_idx = 120
        self.receptor_idx = 121
        self.target_idx = list(range(122, 130))

    def generate_coordinates(self) -> np.ndarray:
        x, y = np.meshgrid(
            np.arange(self.grid_size[0]),
            np.arange(self.grid_size[1])
        )
        return np.column_stack([x.ravel(), y.ravel()]).astype(float)

    def get_grid_neighbors(self, k_hop: int = 3) -> np.ndarray:
        """Grid-based neighbors (Manhattan distance)"""
        gs = self.grid_size
        n_cells = self.n_cells
        adj = np.zeros((n_cells, n_cells), dtype=bool)

        for i in range(n_cells):
            row_i, col_i = i // gs[1], i % gs[1]
            for j in range(n_cells):
                if i == j:
                    continue
                row_j, col_j = j // gs[1], j % gs[1]
                manhattan_dist = abs(row_i - row_j) + abs(col_i - col_j)
                if manhattan_dist <= k_hop:
                    adj[i, j] = True
        return adj

    def create_domains(self) -> np.ndarray:
        domain_labels = np.zeros(self.n_cells, dtype=int)
        gs = self.grid_size

        for i in range(self.n_cells):
            row = i // gs[1]
            col = i % gs[1]

            if col < gs[1] * 0.4:
                domain_labels[i] = 0  # Active
            elif col > gs[1] * 0.6:
                domain_labels[i] = 2  # Quiescent
            else:
                domain_labels[i] = 1  # Transition

        return domain_labels

    def assign_cell_types(self, domain_labels: np.ndarray) -> np.ndarray:
        cell_types = np.zeros(self.n_cells, dtype=int)

        distributions = {
            0: ([0, 1], [0.5, 0.5]),
            1: ([0, 1, 2], [0.2, 0.2, 0.6]),
            2: ([2], [1.0]),
        }

        for i in range(self.n_cells):
            types, probs = distributions[domain_labels[i]]
            cell_types[i] = self.rng.choice(types, p=probs)

        return cell_types

    def generate_expression(self, cell_types: np.ndarray, domain_labels: np.ndarray) -> np.ndarray:
        expression = np.zeros((self.n_cells, self.n_genes), dtype=np.float32)

        for i in range(self.n_cells):
            ct = cell_types[i]
            domain = domain_labels[i]

            # Background
            for g in range(self.n_genes):
                if self.rng.random() < 0.7:
                    expression[i, g] = 0
                else:
                    expression[i, g] = self.rng.negative_binomial(2, 0.3)

            # Domain markers
            for g in self.domain_marker_idx[domain]:
                if self.rng.random() < 0.1:
                    expression[i, g] = 0
                else:
                    expression[i, g] = self.rng.negative_binomial(5, 0.2) + 20

            # Cell type markers
            for g in self.ct_marker_idx[ct]:
                if self.rng.random() < 0.15:
                    expression[i, g] = 0
                else:
                    expression[i, g] = self.rng.negative_binomial(4, 0.25) + 15

            # Ligand (Sender)
            if ct == 0:
                expression[i, self.ligand_idx] = self.rng.negative_binomial(5, 0.2) + 30

            # Receptor (Receiver)
            if ct == 1:
                expression[i, self.receptor_idx] = self.rng.negative_binomial(5, 0.2) + 25

        return expression

    def apply_ccc_effects(
        self,
        expression: np.ndarray,
        cell_types: np.ndarray,
        diffusion_hops: int = 3
    ) -> np.ndarray:
        modified_expr = expression.copy()

        adj = self.get_grid_neighbors(k_hop=diffusion_hops)
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        kernel = adj.astype(float) / row_sums

        ligand_conc = kernel @ expression[:, self.ligand_idx]

        for i in range(self.n_cells):
            if cell_types[i] == 1:  # Receiver
                receptor_level = expression[i, self.receptor_idx]
                signal = ligand_conc[i] * receptor_level / 100

                for t_idx in self.target_idx:
                    if signal > 0.3:
                        modified_expr[i, t_idx] += signal * self.rng.uniform(15, 40)

        return modified_expr

    def calculate_oracles(
        self,
        expression: np.ndarray,
        cell_types: np.ndarray,
        k_hop: int = 3
    ) -> Dict[str, np.ndarray]:
        adj = self.get_grid_neighbors(k_hop=k_hop)
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        kernel = adj.astype(float) / row_sums

        # CT Heterogeneity
        ct_heterogeneity = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            if adj[i].sum() > 0:
                neighbor_types = cell_types[adj[i]]
                ct_heterogeneity[i] = 1 - np.mean(neighbor_types == cell_types[i])

        # CT Match
        ct_match = 1 - ct_heterogeneity

        # CCC Activity
        ligand_conc = kernel @ expression[:, self.ligand_idx]
        receptor_expr = expression[:, self.receptor_idx]
        ccc_signal = ligand_conc * receptor_expr
        target_expr = expression[:, self.target_idx].mean(axis=1)

        if ccc_signal.max() > 0:
            ccc_signal = ccc_signal / ccc_signal.max()
        if target_expr.max() > 0:
            target_expr = target_expr / target_expr.max()

        ccc_activity = 0.5 * ccc_signal + 0.5 * target_expr

        return {
            'ct_heterogeneity': ct_heterogeneity,
            'ct_match': ct_match,
            'ccc_activity': ccc_activity
        }

    def simulate(self, diffusion_hops: int = 3, oracle_hops: int = 3) -> ad.AnnData:
        print("1. Generating coordinates...")
        coords = self.generate_coordinates()

        print("2. Creating domains...")
        domain_labels = self.create_domains()

        print("3. Assigning cell types...")
        cell_types = self.assign_cell_types(domain_labels)

        print("4. Generating expression...")
        base_expr = self.generate_expression(cell_types, domain_labels)

        print("5. Applying CCC effects...")
        expression = self.apply_ccc_effects(base_expr, cell_types, diffusion_hops)

        print("6. Calculating oracles...")
        oracles = self.calculate_oracles(expression, cell_types, oracle_hops)

        # Gene names
        gene_names = [f'Gene_{i}' for i in range(self.n_genes)]
        gene_names[self.ligand_idx] = 'Ligand'
        gene_names[self.receptor_idx] = 'Receptor'
        for i, idx in enumerate(self.target_idx):
            gene_names[idx] = f'Target_{i}'

        adata = ad.AnnData(
            X=expression,
            obs=pd.DataFrame({
                'cell_type': [self.cell_type_names[ct] for ct in cell_types],
                'cell_type_id': cell_types,
                'spatial_domain': [self.domain_names[d] for d in domain_labels],
                'domain_id': domain_labels,
                'oracle_ct_heterogeneity': oracles['ct_heterogeneity'],
                'oracle_ct_match': oracles['ct_match'],
                'oracle_ccc_activity': oracles['ccc_activity'],
            }),
            var=pd.DataFrame(index=gene_names),
            obsm={'spatial': coords}
        )

        adata.layers['counts'] = expression.copy()
        # adata.uns['simulation_params'] = {
        #     'diffusion_hops': diffusion_hops,
        #     'oracle_hops': oracle_hops,
        #     'grid_size': self.grid_size
        # }

        print("Simulation complete.")
        return adata


# ============== Part 2: Run STEP ==============

def run_step(adata, n_glayers=3, n_clusters_domain=3, n_clusters_ct=3):
    """Run STEP on simulated data"""
    from step import stModel

    # Setup coordinates
    adata.obs[['array_row', 'array_col']] = adata.obsm['spatial']

    print("\n" + "="*50)
    print("Running STEP")
    print("="*50)

    # Initialize STEP
    stepc = stModel(
        adata=adata,
        n_top_genes=None,
        geneset_to_use=adata.var_names.to_list(),
        layer_key=None,
        n_modules=10,
        decoder_type='zinb',
        coord_keys=("array_col", "array_row"),
        n_glayers=n_glayers,
        variational=True,
        edge_clip=1,
    )

    # Run STEP
    stepc.run()

    # Cluster for domains
    stepc.cluster(n_clusters=n_clusters_domain)
    adata = stepc.adata

    # Get embeddings
    adata.obsm['X_rep'] = stepc.embed()  # Individual embedding

    # Calculate spatial coherence
    adata.obs['step_coherence'] = (
        adata.obsm['X_rep'] * adata.obsm['X_smoothed']
    ).sum(1) / (
        np.linalg.norm(adata.obsm['X_rep'], axis=1) *
        np.linalg.norm(adata.obsm['X_smoothed'], axis=1)
    )

    # Cluster for cell types (using individual embedding)
    stepc.cluster(adata=adata, n_clusters=n_clusters_ct, use_rep='X_rep', key_added='step_ct')

    print("STEP fitting complete.")
    return adata, stepc


# ============== Part 3: Validation ==============

def validate_step(adata):
    """Validate STEP results against ground truth and oracles"""

    print("\n" + "="*50)
    print("STEP Validation Results")
    print("="*50)

    results = {}

    # 1. Domain identification accuracy
    ari_domain = ari(adata.obs['domain_id'], adata.obs['domain'])
    print(f"\n1. Domain Identification:")
    print(f"   ARI = {ari_domain:.4f}")
    results['ari_domain'] = ari_domain

    # 2. Cell type identification accuracy
    ari_ct = ari(adata.obs['cell_type_id'], adata.obs['step_ct'])
    print(f"\n2. Cell Type Identification:")
    print(f"   ARI = {ari_ct:.4f}")
    results['ari_ct'] = ari_ct

    # 3. Spatial coherence vs CT Heterogeneity (expected: NEGATIVE)
    r_het, p_het = pearsonr(adata.obs['step_coherence'], adata.obs['oracle_ct_heterogeneity'])
    rho_het, _ = spearmanr(adata.obs['step_coherence'], adata.obs['oracle_ct_heterogeneity'])
    print(f"\n3. Coherence vs CT Heterogeneity:")
    print(f"   Pearson r = {r_het:.4f} (p = {p_het:.2e})")
    print(f"   Spearman rho = {rho_het:.4f}")
    print(f"   Expected negative association: {'PASS' if r_het < 0 else 'FAIL'}")
    results['r_ct_heterogeneity'] = r_het
    results['rho_ct_heterogeneity'] = rho_het

    # 4. Spatial coherence vs CT Match (expected: POSITIVE)
    r_match, p_match = pearsonr(adata.obs['step_coherence'], adata.obs['oracle_ct_match'])
    rho_match, _ = spearmanr(adata.obs['step_coherence'], adata.obs['oracle_ct_match'])
    print(f"\n4. Coherence vs CT Composition Match:")
    print(f"   Pearson r = {r_match:.4f} (p = {p_match:.2e})")
    print(f"   Spearman rho = {rho_match:.4f}")
    print(f"   Expected positive association: {'PASS' if r_match > 0 else 'FAIL'}")
    results['r_ct_match'] = r_match
    results['rho_ct_match'] = rho_match

    # 5. Spatial coherence vs CCC Activity
    r_ccc, p_ccc = pearsonr(adata.obs['step_coherence'], adata.obs['oracle_ccc_activity'])
    rho_ccc, _ = spearmanr(adata.obs['step_coherence'], adata.obs['oracle_ccc_activity'])
    print(f"\n5. Coherence vs CCC Activity:")
    print(f"   Pearson r = {r_ccc:.4f} (p = {p_ccc:.2e})")
    print(f"   Spearman rho = {rho_ccc:.4f}")
    results['r_ccc_activity'] = r_ccc
    results['rho_ccc_activity'] = rho_ccc

    # 6. Coherence by domain
    print(f"\n6. Mean Coherence by Domain:")
    for domain in adata.obs['spatial_domain'].unique():
        mask = adata.obs['spatial_domain'] == domain
        mean_coh = adata.obs.loc[mask, 'step_coherence'].mean()
        std_coh = adata.obs.loc[mask, 'step_coherence'].std()
        print(f"   {domain}: {mean_coh:.4f} +/- {std_coh:.4f}")

    print("\n" + "="*50)

    return results


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    try:
        from .plot_coherence_validation import visualize_from_h5ad
    except ImportError:
        from plot_coherence_validation import visualize_from_h5ad

    parser = argparse.ArgumentParser(
        description="Run the known-label spatial coherence simulation."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("workflows/coherence_simulation"),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        default=(50, 50),
        metavar=("N_ROWS", "N_COLS"),
    )
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-hops", type=int, default=3)
    parser.add_argument("--oracle-hops", type=int, default=3)
    parser.add_argument("--n-glayers", type=int, default=3)
    parser.add_argument(
        "--skip-step",
        action="store_true",
        help="Generate only the oracle simulation object.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    simulator = SpatialCoherenceSimulator(
        grid_size=tuple(args.grid_size),
        n_genes=args.n_genes,
        random_seed=args.seed,
    )
    adata = simulator.simulate(
        diffusion_hops=args.diffusion_hops,
        oracle_hops=args.oracle_hops,
    )
    oracle_path = args.out_dir / "simulation_grid_based.h5ad"
    adata.write(oracle_path)

    settings = {
        "grid_size": list(args.grid_size),
        "n_genes": args.n_genes,
        "seed": args.seed,
        "diffusion_hops": args.diffusion_hops,
        "oracle_hops": args.oracle_hops,
        "n_glayers": args.n_glayers,
        "step_run": False,
    }

    if not args.skip_step:
        adata, _ = run_step(
            adata,
            n_glayers=args.n_glayers,
            n_clusters_domain=3,
            n_clusters_ct=3,
        )
        results = validate_step(adata)
        result_path = args.out_dir / "simulation_with_step_results.h5ad"
        adata.write(result_path)
        pd.Series(results).to_json(
            args.out_dir / "composition_validation_metrics.json",
            indent=2,
        )
        figure, _ = visualize_from_h5ad(
            result_path,
            save_path=args.out_dir / "coherence_validation.png",
        )

        plt.close(figure)
        settings["step_run"] = True
        settings["metrics"] = {
            key: float(value)
            if isinstance(value, (int, float, np.floating))
            else value
            for key, value in results.items()
        }

    (args.out_dir / "composition_simulation_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
