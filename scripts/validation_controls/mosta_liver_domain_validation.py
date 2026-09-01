#!/usr/bin/env python
"""Validate transcriptional and spatial structure within MOSTA Liver annotations."""


import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, rankdata


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SOURCE = Path("data/mosta_e16_test.h5ad")
DEFAULT_RAW_DIR = Path("data/stomics-mosta")
DEFAULT_OUTPUT = Path("workflows/mosta_liver_validation/domain_validation")
SECTION_ORDER = (
    "E16.5_E2S1.MOSTA",
    "E16.5_E2S4.MOSTA",
    "E16.5_E2S7.MOSTA",
    "E16.5_E2S10.MOSTA",
    "E16.5_E2S13.MOSTA",
)
SECTION_DISTANCE_UM = {
    "E16.5_E2S1.MOSTA": -2120,
    "E16.5_E2S4.MOSTA": -1150,
    "E16.5_E2S7.MOSTA": 0,
    "E16.5_E2S10.MOSTA": 740,
    "E16.5_E2S13.MOSTA": 1770,
}
MARKER_GROUPS = {
    "Hepatic": (
        "Afp",
        "Alb",
        "Apoa1",
        "Apoa2",
        "Ttr",
        "Dlk1",
        "Apom",
        "Ahsg",
        "Trf",
        "Rbp4",
        "Hnf4a",
        "Hnf1a",
    ),
    "Erythroid": (
        "Hba-a1",
        "Hba-a2",
        "Hba-x",
        "Hbb-bs",
        "Hbb-bt",
        "Hbb-y",
        "Hbb-bh1",
        "Hbb-bh2",
        "Slc4a1",
        "Car1",
        "Car2",
        "Alas2",
        "Gata1",
        "Klf1",
        "Nfe2",
        "Ahsp",
        "Epb42",
    ),
    "Immune/endothelial": (
        "Ptprc",
        "Lyz2",
        "Tyrobp",
        "Spi1",
        "C1qa",
        "C1qb",
        "C1qc",
        "Kdr",
        "Pecam1",
        "Cdh5",
        "Emcn",
        "Eng",
        "Esam",
    ),
    "Mesenchymal/myogenic": (
        "Col1a1",
        "Col1a2",
        "Col3a1",
        "Dcn",
        "Lum",
        "Sparc",
        "Lgals1",
        "Acta1",
        "Acta2",
        "Actc1",
        "Mylpf",
        "Myl1",
        "Myh3",
        "Myh8",
        "Tnnt3",
        "Tnni2",
    ),
}
DOMAIN_MARKER_GENES = {
    "2": (
        "Ttr",
        "Ahsg",
        "Apoa2",
        "Apoa1",
        "Alb",
        "Trf",
        "Afp",
        "Rbp4",
    ),
    "15": (
        "Col1a2",
        "Col3a1",
        "Acta1",
        "Mylpf",
        "Actc1",
        "Col1a1",
        "Dlk1",
        "H19",
    ),
    "26": (
        "Peak1",
        "Nnat",
        "Map1b",
        "Serpinf2",
        "Uchl1",
        "Sox11",
        "Dcx",
        "Igf2",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-radius", type=float, default=1.51)
    parser.add_argument("--min-component-size", type=int, default=20)
    parser.add_argument("--top-genes", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-patch-domain-fraction", type=float, default=0.05)
    parser.add_argument("--min-patch-domains", type=int, default=2)
    return parser.parse_args()


def normalize_counts(
    counts: sparse.spmatrix, total_counts: np.ndarray
) -> sparse.csr_matrix:
    matrix = normalize_counts_linear(counts, total_counts)
    matrix.data = np.log1p(matrix.data)
    return matrix


def normalize_counts_linear(
    counts: sparse.spmatrix, total_counts: np.ndarray
) -> sparse.csr_matrix:
    matrix = counts.tocsr().astype(np.float32)
    matrix = matrix.multiply(
        (1e4 / np.maximum(total_counts.astype(np.float32), 1.0))[:, None]
    ).tocsr()
    return matrix


def score_gene_sets(
    normalized_counts: sparse.csr_matrix,
    gene_names: pd.Index,
    gene_sets: dict[str, tuple[str, ...]],
    seed: int,
    score_prefix: str,
    ctrl_size: int = 50,
    n_bins: int = 25,
) -> dict[str, np.ndarray]:
    """Compute standard Scanpy marker scores on log-normalized counts."""
    gene_names = pd.Index(gene_names.astype(str))
    score_adata = ad.AnnData(
        X=normalized_counts,
        var=pd.DataFrame(index=gene_names),
    )
    scores = {}
    for offset, (name, genes) in enumerate(gene_sets.items()):
        present = [gene for gene in genes if gene in gene_names]
        if not present:
            raise ValueError(f"no genes found for marker set: {name}")
        score_name = f"{score_prefix}_{offset}_score"
        sc.tl.score_genes(
            score_adata,
            gene_list=present,
            ctrl_as_ref=True,
            ctrl_size=ctrl_size,
            gene_pool=gene_names,
            n_bins=n_bins,
            score_name=score_name,
            random_state=seed + offset,
            use_raw=False,
        )
        scores[name] = score_adata.obs[score_name].to_numpy(
            dtype=np.float32
        )
    return scores


def component_metrics(
    labels: np.ndarray,
    pairs: np.ndarray,
    domain: str,
    min_component_size: int,
) -> dict[str, float | int | str]:
    selected = np.flatnonzero(labels == domain)
    if len(selected) == 0:
        return {
            "domain": domain,
            "n_spots": 0,
            "n_components": 0,
            "largest_component_fraction": np.nan,
            "n_components_at_least_min_size": 0,
        }

    inverse = np.full(len(labels), -1, dtype=np.int64)
    inverse[selected] = np.arange(len(selected))
    domain_pairs = pairs[
        (labels[pairs[:, 0]] == domain) & (labels[pairs[:, 1]] == domain)
    ]
    if len(domain_pairs):
        rows = inverse[domain_pairs[:, 0]]
        columns = inverse[domain_pairs[:, 1]]
        graph = coo_matrix(
            (
                np.ones(len(domain_pairs) * 2, dtype=np.uint8),
                (np.r_[rows, columns], np.r_[columns, rows]),
            ),
            shape=(len(selected), len(selected)),
        ).tocsr()
    else:
        graph = sparse.csr_matrix((len(selected), len(selected)))

    n_components, component_labels = connected_components(graph, directed=False)
    sizes = np.bincount(component_labels)
    return {
        "domain": domain,
        "n_spots": int(len(selected)),
        "n_components": int(n_components),
        "largest_component_fraction": float(sizes.max() / len(selected)),
        "n_components_at_least_min_size": int(
            np.count_nonzero(sizes >= min_component_size)
        ),
    }


def load_raw_liver_expression(
    raw_dir: Path,
    model_obs_names: pd.Index,
    sections: np.ndarray,
    annotations: np.ndarray,
    domains: np.ndarray,
    liver_indices: np.ndarray,
) -> ad.AnnData:
    """Load and align all Liver spots with the full raw gene space."""
    section_datasets = []
    liver_positions = np.full(len(sections), -1, dtype=np.int64)
    liver_positions[liver_indices] = np.arange(len(liver_indices))

    for section in SECTION_ORDER:
        global_indices = np.flatnonzero(
            (sections == section) & (annotations == "Liver")
        )
        raw_path = raw_dir / f"{section}.h5ad"
        raw = ad.read_h5ad(raw_path, backed="r")
        raw_indices = raw.obs_names.get_indexer(model_obs_names[global_indices])
        if np.any(raw_indices < 0):
            raise AssertionError(f"failed to align raw spots for {section}")
        counts = raw.layers["count"][raw_indices, :]
        if not sparse.issparse(counts):
            counts = sparse.csr_matrix(counts)
        normalized = normalize_counts(
            counts, raw.obs["total_counts"].to_numpy()[raw_indices]
        )
        spot_names = model_obs_names[global_indices].astype(str)
        section_obs = pd.DataFrame(
            {
                "section": section,
                "domain": domains[global_indices].astype(str),
                "output_position": liver_positions[global_indices],
            },
            index=pd.Index(
                [f"{section}:{spot}" for spot in spot_names],
                name="spot_id",
            ),
        )
        section_datasets.append(
            ad.AnnData(
                X=normalized,
                obs=section_obs,
                var=pd.DataFrame(index=raw.var_names.astype(str)),
            )
        )
        raw.file.close()

    combined = ad.concat(
        section_datasets,
        axis=0,
        join="outer",
        merge="same",
        fill_value=0,
    )
    order = np.argsort(combined.obs["output_position"].to_numpy())
    combined = combined[order].copy()
    expected = np.arange(len(liver_indices))
    if not np.array_equal(
        combined.obs["output_position"].to_numpy(), expected
    ):
        raise AssertionError("failed to restore Liver spot order")
    return combined


def summarize_marker_gene_expression(
    expression: ad.AnnData,
) -> pd.DataFrame:
    genes = list(
        dict.fromkeys(
            gene
            for marker_set in (*MARKER_GROUPS.values(), *DOMAIN_MARKER_GENES.values())
            for gene in marker_set
            if gene in expression.var_names
        )
    )
    selected = expression[:, genes]
    rows = []
    for section in SECTION_ORDER:
        section_mask = selected.obs["section"].to_numpy() == section
        section_domains = selected.obs["domain"].to_numpy()[section_mask]
        section_values = selected.X[section_mask]
        for domain in np.unique(section_domains):
            domain_mask = section_domains == domain
            means = np.asarray(section_values[domain_mask].mean(axis=0)).ravel()
            for gene, mean in zip(genes, means, strict=True):
                rows.append(
                    {
                        "section": section,
                        "domain": str(domain),
                        "gene": gene,
                        "n_spots": int(domain_mask.sum()),
                        "mean_log_normalized_expression": float(mean),
                    }
                )
    return pd.DataFrame(rows)


def full_transcriptome_de(
    expression: ad.AnnData,
    target_domains: tuple[str, ...],
    top_genes: int,
) -> pd.DataFrame:
    """Run pooled full-transcriptome Wilcoxon tests within Liver spots."""
    expression.obs["domain"] = pd.Categorical(
        expression.obs["domain"].astype(str)
    )
    sc.tl.rank_genes_groups(
        expression,
        groupby="domain",
        groups=list(target_domains),
        reference="rest",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        n_genes=top_genes,
        rankby_abs=False,
        pts=True,
        use_raw=False,
    )
    rows = []
    for domain in target_domains:
        frame = sc.get.rank_genes_groups_df(expression, group=domain)
        frame = frame.rename(columns={"names": "gene"})
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
        frame.insert(0, "domain", domain)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def program_statistics(
    marker_scores: dict[str, np.ndarray],
    sections: np.ndarray,
    domains: np.ndarray,
    liver_indices: np.ndarray,
    liver_domains: tuple[str, ...],
    patch_sections: tuple[str, ...],
    min_domain_fraction: float,
) -> pd.DataFrame:
    liver_sections = sections[liver_indices]
    liver_domain_labels = domains[liver_indices]
    rows = []
    for section in patch_sections:
        section_mask = liver_sections == section
        for domain in liver_domains:
            domain_mask = section_mask & (liver_domain_labels == domain)
            other_mask = section_mask & (liver_domain_labels != domain)
            if domain_mask.sum() / section_mask.sum() < min_domain_fraction:
                continue
            if domain_mask.sum() < 20 or other_mask.sum() < 20:
                continue
            for program, values in marker_scores.items():
                domain_values = values[domain_mask]
                other_values = values[other_mask]
                test = mannwhitneyu(
                    domain_values,
                    other_values,
                    alternative="two-sided",
                    method="asymptotic",
                )
                rank_biserial = (
                    2.0 * float(test.statistic)
                    / (len(domain_values) * len(other_values))
                    - 1.0
                )
                rows.append(
                    {
                        "section": section,
                        "domain": domain,
                        "program": program,
                        "n_domain": int(len(domain_values)),
                        "n_other_liver": int(len(other_values)),
                        "domain_mean": float(domain_values.mean()),
                        "other_liver_mean": float(other_values.mean()),
                        "mean_difference": float(
                            domain_values.mean() - other_values.mean()
                        ),
                        "rank_biserial": rank_biserial,
                        "p_value": float(test.pvalue),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["fdr_bh"] = bh_adjust(frame["p_value"].to_numpy())
    return frame


def logistic_odds_ratio_per_sd(
    target: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    """Estimate the domain-membership odds ratio per score standard deviation."""
    score_sd = float(score.std(ddof=0))
    if not np.isfinite(score_sd) or score_sd <= 0:
        return {
            "logistic_beta_per_sd": np.nan,
            "odds_ratio_per_sd": np.nan,
            "odds_ratio_ci025": np.nan,
            "odds_ratio_ci975": np.nan,
            "logistic_p_value": np.nan,
        }
    standardized = (score - score.mean()) / score_sd
    design = sm.add_constant(standardized, has_constant="add")
    fit = sm.GLM(
        target.astype(np.float64),
        design,
        family=sm.families.Binomial(),
    ).fit()
    beta = float(fit.params[1])
    standard_error = float(fit.bse[1])
    lower = beta - 1.96 * standard_error
    upper = beta + 1.96 * standard_error
    return {
        "logistic_beta_per_sd": beta,
        "odds_ratio_per_sd": float(np.exp(np.clip(beta, -700, 700))),
        "odds_ratio_ci025": float(np.exp(np.clip(lower, -700, 700))),
        "odds_ratio_ci975": float(np.exp(np.clip(upper, -700, 700))),
        "logistic_p_value": float(fit.pvalues[1]),
    }


def domain_marker_score_statistics(
    marker_scores: dict[str, np.ndarray],
    sections: np.ndarray,
    domains: np.ndarray,
    liver_indices: np.ndarray,
    target_domains: tuple[str, ...],
    eligible_sections: tuple[str, ...],
    n_permutations: int,
    rng: np.random.Generator,
    min_spots: int = 20,
) -> pd.DataFrame:
    """Contrast every Scanpy marker score across Liver domains."""
    liver_sections = sections[liver_indices]
    liver_domain_labels = domains[liver_indices]
    rows = []

    for section in SECTION_ORDER:
        section_mask = liver_sections == section
        section_eligible = section in eligible_sections
        for assigned_domain in target_domains:
            target_mask = section_mask & (
                liver_domain_labels == assigned_domain
            )
            background_mask = section_mask & (
                liver_domain_labels != assigned_domain
            )
            n_target = int(target_mask.sum())
            n_background = int(background_mask.sum())
            tested = (
                section_eligible
                and n_target >= min_spots
                and n_background >= min_spots
            )
            section_target = (
                liver_domain_labels[section_mask] == assigned_domain
            )
            for marker_set_domain in target_domains:
                row: dict[str, float | int | str | bool] = {
                    "section": section,
                    "assigned_domain": assigned_domain,
                    "marker_set_domain": marker_set_domain,
                    "n_target": n_target,
                    "n_background": n_background,
                    "domain_fraction": n_target / (n_target + n_background),
                    "tested": tested,
                }
                if not tested:
                    rows.append(row)
                    continue

                section_values = np.asarray(
                    marker_scores[marker_set_domain][section_mask],
                    dtype=np.float64,
                )
                target_values = section_values[section_target]
                background_values = section_values[~section_target]
                target_mean = float(target_values.mean())
                background_mean = float(background_values.mean())
                score_difference = target_mean - background_mean
                test = mannwhitneyu(
                    target_values,
                    background_values,
                    alternative="two-sided",
                    method="asymptotic",
                )
                rank_biserial = (
                    2.0 * float(test.statistic)
                    / (n_target * n_background)
                    - 1.0
                )

                section_ranks = rankdata(section_values, method="average")
                null_rank_biserial = np.empty(
                    n_permutations, dtype=np.float64
                )
                for permutation_index in range(n_permutations):
                    sampled = rng.choice(
                        len(section_values), size=n_target, replace=False
                    )
                    rank_sum = float(section_ranks[sampled].sum())
                    null_u = rank_sum - n_target * (n_target + 1) / 2.0
                    null_rank_biserial[permutation_index] = (
                        2.0 * null_u / (n_target * n_background) - 1.0
                    )
                null_center = float(null_rank_biserial.mean())

                row.update(
                    {
                        "target_mean_score": target_mean,
                        "other_liver_mean_score": background_mean,
                        "score_difference": score_difference,
                        "rank_biserial": rank_biserial,
                        "mannwhitney_p_value": float(test.pvalue),
                        "permutation_mean_rank_biserial": null_center,
                        "permutation_rank_biserial_q025": float(
                            np.quantile(null_rank_biserial, 0.025)
                        ),
                        "permutation_rank_biserial_q975": float(
                            np.quantile(null_rank_biserial, 0.975)
                        ),
                        "permutation_two_sided_p_value": float(
                            (
                                1
                                + np.count_nonzero(
                                    np.abs(
                                        null_rank_biserial - null_center
                                    )
                                    >= abs(rank_biserial - null_center)
                                )
                            )
                            / (n_permutations + 1)
                        ),
                    }
                )
                row.update(
                    logistic_odds_ratio_per_sd(
                        section_target,
                        section_values,
                    )
                )
                rows.append(row)

    table = pd.DataFrame(rows)
    tested = table["tested"].astype(bool)
    table["mannwhitney_fdr_bh"] = np.nan
    table["permutation_two_sided_fdr_bh"] = np.nan
    table.loc[tested, "mannwhitney_fdr_bh"] = bh_adjust(
        table.loc[tested, "mannwhitney_p_value"].to_numpy()
    )
    table.loc[tested, "permutation_two_sided_fdr_bh"] = bh_adjust(
        table.loc[tested, "permutation_two_sided_p_value"].to_numpy()
    )
    return table


def _rank_biserial_matrix(
    indicators: np.ndarray,
    rank_quantiles: np.ndarray,
    valid_domains: np.ndarray,
) -> np.ndarray:
    n_spots = len(rank_quantiles)
    n_target = indicators.sum(axis=0).astype(np.float64)
    n_background = n_spots - n_target
    target_mean = indicators.T @ rank_quantiles
    target_mean = np.divide(
        target_mean,
        n_target[:, None],
        out=np.full_like(target_mean, np.nan, dtype=np.float64),
        where=n_target[:, None] > 0,
    )
    background_mean = (~indicators).T @ rank_quantiles
    background_mean = np.divide(
        background_mean,
        n_background[:, None],
        out=np.full_like(background_mean, np.nan, dtype=np.float64),
        where=n_background[:, None] > 0,
    )
    effect = 2.0 * (target_mean - background_mean)
    effect[~valid_domains, :] = np.nan
    return effect


def marker_domain_spatial_colocalization(
    marker_scores: dict[str, np.ndarray],
    sections: np.ndarray,
    domains: np.ndarray,
    coords: np.ndarray,
    liver_indices: np.ndarray,
    target_domains: tuple[str, ...],
    eligible_sections: tuple[str, ...],
    n_permutations: int,
    rng: np.random.Generator,
    min_spots: int = 20,
) -> pd.DataFrame:
    """Test marker-map and domain-mask co-localization using spatial shifts."""
    liver_sections = sections[liver_indices]
    liver_domains = domains[liver_indices]
    liver_coords = coords[liver_indices]
    rows = []

    for section in SECTION_ORDER:
        section_mask = liver_sections == section
        section_domains = liver_domains[section_mask]
        section_coords = np.asarray(liver_coords[section_mask], dtype=np.float64)
        values = np.column_stack(
            [marker_scores[domain][section_mask] for domain in target_domains]
        ).astype(np.float64)
        rank_quantiles = np.column_stack(
            [
                rankdata(values[:, column], method="average") / len(values)
                for column in range(values.shape[1])
            ]
        )
        indicators = np.column_stack(
            [section_domains == domain for domain in target_domains]
        )
        n_target = indicators.sum(axis=0)
        valid_domains = (
            (section in eligible_sections)
            & (n_target >= min_spots)
            & (len(section_domains) - n_target >= min_spots)
        )
        observed = _rank_biserial_matrix(
            indicators, rank_quantiles, valid_domains
        )

        minimum = section_coords.min(axis=0)
        span = np.maximum(np.ptp(section_coords, axis=0), 1.0)
        unit_coords = (section_coords - minimum) / span
        unit_coords = np.minimum(unit_coords, np.nextafter(1.0, 0.0))
        tree = cKDTree(unit_coords, boxsize=1.0)
        null = np.full(
            (
                n_permutations,
                len(target_domains),
                len(target_domains),
            ),
            np.nan,
            dtype=np.float64,
        )
        for permutation_index in range(n_permutations):
            shift = rng.uniform(0.0, 1.0, size=2)
            while np.linalg.norm(np.minimum(shift, 1.0 - shift)) < 0.10:
                shift = rng.uniform(0.0, 1.0, size=2)
            query_coords = (unit_coords - shift) % 1.0
            shifted_indices = tree.query(query_coords, k=1)[1]
            null[permutation_index] = _rank_biserial_matrix(
                indicators,
                rank_quantiles[shifted_indices],
                valid_domains,
            )

        for row_index, assigned_domain in enumerate(target_domains):
            for column_index, marker_set_domain in enumerate(target_domains):
                tested = bool(valid_domains[row_index])
                row: dict[str, float | int | str | bool] = {
                    "section": section,
                    "assigned_domain": assigned_domain,
                    "marker_set_domain": marker_set_domain,
                    "n_target": int(n_target[row_index]),
                    "n_background": int(
                        len(section_domains) - n_target[row_index]
                    ),
                    "tested": tested,
                }
                if tested:
                    null_values = null[:, row_index, column_index]
                    observed_value = float(observed[row_index, column_index])
                    null_mean = float(null_values.mean())
                    null_sd = float(null_values.std(ddof=1))
                    row.update(
                        {
                            "spatial_rank_biserial": observed_value,
                            "shift_null_mean": null_mean,
                            "shift_null_q025": float(
                                np.quantile(null_values, 0.025)
                            ),
                            "shift_null_q975": float(
                                np.quantile(null_values, 0.975)
                            ),
                            "shift_z_score": float(
                                (observed_value - null_mean)
                                / max(null_sd, 1e-12)
                            ),
                            "spatial_shift_p_value": float(
                                (
                                    1
                                    + np.count_nonzero(
                                        np.abs(null_values - null_mean)
                                        >= abs(observed_value - null_mean)
                                    )
                                )
                                / (n_permutations + 1)
                            ),
                        }
                    )
                rows.append(row)

    table = pd.DataFrame(rows)
    tested = table["tested"].astype(bool)
    table["spatial_shift_fdr_bh"] = np.nan
    table.loc[tested, "spatial_shift_fdr_bh"] = bh_adjust(
        table.loc[tested, "spatial_shift_p_value"].to_numpy()
    )
    return table


def permutation_edge_agreement(
    labels: np.ndarray,
    pairs: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    _, codes = np.unique(labels, return_inverse=True)
    observed = float(np.mean(codes[pairs[:, 0]] == codes[pairs[:, 1]]))
    null = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        shuffled = rng.permutation(codes)
        null[index] = np.mean(
            shuffled[pairs[:, 0]] == shuffled[pairs[:, 1]]
        )
    null_std = float(null.std(ddof=1))
    return {
        "observed_edge_agreement": observed,
        "permutation_mean": float(null.mean()),
        "permutation_sd": null_std,
        "permutation_q025": float(np.quantile(null, 0.025)),
        "permutation_q975": float(np.quantile(null, 0.975)),
        "permutation_p_value": float(
            (1 + np.count_nonzero(null >= observed)) / (n_permutations + 1)
        ),
        "z_score": float((observed - null.mean()) / null_std),
        "n_permutations": int(n_permutations),
    }


def plot_validation(
    output_path: Path,
    sections: np.ndarray,
    annotations: np.ndarray,
    domains: np.ndarray,
    coords: np.ndarray,
    liver_indices: np.ndarray,
    liver_domains: tuple[str, ...],
    domain_colors: dict[str, str],
    marker_scores: dict[str, np.ndarray],
) -> None:
    plot_coords = coords.copy()
    plot_coords[:, 1] *= -1
    liver = annotations == "Liver"
    score_by_global_index = {
        name: dict(zip(liver_indices, values, strict=True))
        for name, values in marker_scores.items()
    }
    score_limits = {
        name: max(
            abs(float(np.quantile(values, 0.02))),
            abs(float(np.quantile(values, 0.98))),
            0.1,
        )
        for name, values in marker_scores.items()
    }

    figure = plt.figure(figsize=(18.2, 13.0), constrained_layout=True)
    grid = figure.add_gridspec(
        4,
        6,
        width_ratios=(0.09, 1, 1, 1, 1, 1),
        wspace=0.015,
        hspace=0.02,
    )
    row_names = (
        "Manual Liver",
        "STEP domains",
        "Hepatic program",
        "Mesenchymal/myogenic program",
    )
    continuous_rows: dict[str, list[plt.Axes]] = {
        "Hepatic": [],
        "Mesenchymal/myogenic": [],
    }
    mappables: dict[str, object] = {}

    for row, row_name in enumerate(row_names):
        row_axis = figure.add_subplot(grid[row, 0])
        row_axis.text(
            0.5,
            0.5,
            row_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        row_axis.axis("off")

        for column, section in enumerate(SECTION_ORDER, start=1):
            axis = figure.add_subplot(grid[row, column])
            section_mask = sections == section
            section_indices = np.flatnonzero(section_mask)
            liver_section = section_mask & liver
            liver_section_indices = np.flatnonzero(liver_section)

            axis.scatter(
                plot_coords[section_indices, 0],
                plot_coords[section_indices, 1],
                c="#D9D9D9",
                s=0.20,
                linewidths=0,
                rasterized=True,
            )
            if row == 0:
                axis.scatter(
                    plot_coords[liver_section_indices, 0],
                    plot_coords[liver_section_indices, 1],
                    c="#2F6B4F",
                    s=0.45,
                    linewidths=0,
                    rasterized=True,
                )
            elif row == 1:
                for domain in liver_domains:
                    mask = liver_section & (domains == domain)
                    axis.scatter(
                        plot_coords[mask, 0],
                        plot_coords[mask, 1],
                        c=domain_colors[domain],
                        s=0.45,
                        linewidths=0,
                        rasterized=True,
                    )
            else:
                score_name = row_names[row].replace(" program", "")
                values = np.asarray(
                    [
                        score_by_global_index[score_name][index]
                        for index in liver_section_indices
                    ]
                )
                score_limit = score_limits[score_name]
                mappable = axis.scatter(
                    plot_coords[liver_section_indices, 0],
                    plot_coords[liver_section_indices, 1],
                    c=values,
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(
                        vmin=-score_limit,
                        vcenter=0.0,
                        vmax=score_limit,
                    ),
                    s=0.45,
                    linewidths=0,
                    rasterized=True,
                )
                continuous_rows[score_name].append(axis)
                mappables[score_name] = mappable

            axis.set_aspect("equal")
            axis.axis("off")
            if row == 0:
                axis.set_title(section.removesuffix(".MOSTA"), fontsize=12)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=domain_colors[domain],
            markeredgecolor="none",
            markersize=7,
            label=f"Domain {domain}",
        )
        for domain in liver_domains
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=10,
    )
    for score_name, axes in continuous_rows.items():
        colorbar = figure.colorbar(
            mappables[score_name],
            ax=axes,
            location="right",
            fraction=0.018,
            pad=0.01,
        )
        colorbar.ax.tick_params(labelsize=8)

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, resolution=300.0)


def domain_interface_segments(
    coords: np.ndarray,
    labels: np.ndarray,
    displayed_domains: tuple[str, ...],
    edge_radius: float,
) -> np.ndarray:
    """Return local edges crossing displayed STEP-domain assignments."""
    pairs = cKDTree(coords).query_pairs(
        r=edge_radius,
        output_type="ndarray",
    )
    if len(pairs) == 0:
        return np.empty((0, 2, 2), dtype=np.float32)
    displayed = np.isin(labels, displayed_domains)
    keep = (
        displayed[pairs[:, 0]]
        & displayed[pairs[:, 1]]
        & (labels[pairs[:, 0]] != labels[pairs[:, 1]])
    )
    return coords[pairs[keep]]


def connected_liver_display_mask(
    coords: np.ndarray,
    edge_radius: float,
    min_component_size: int = 20,
) -> np.ndarray:
    """Exclude isolated annotation points from the Liver-only zoom viewport."""
    pairs = cKDTree(coords).query_pairs(
        r=edge_radius,
        output_type="ndarray",
    )
    if len(pairs) == 0:
        return np.ones(len(coords), dtype=bool)
    graph = coo_matrix(
        (
            np.ones(len(pairs) * 2, dtype=np.uint8),
            (
                np.r_[pairs[:, 0], pairs[:, 1]],
                np.r_[pairs[:, 1], pairs[:, 0]],
            ),
        ),
        shape=(len(coords), len(coords)),
    ).tocsr()
    _, component_labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(component_labels)
    keep = component_sizes[component_labels] >= min_component_size
    return keep if keep.any() else np.ones(len(coords), dtype=bool)


def plot_spatial_evidence_story(
    output_path: Path,
    sections: np.ndarray,
    annotations: np.ndarray,
    domains: np.ndarray,
    coords: np.ndarray,
    liver_indices: np.ndarray,
    liver_domains: tuple[str, ...],
    domain_colors: dict[str, str],
    marker_scores: dict[str, np.ndarray],
    edge_radius: float,
) -> None:
    """Connect coarse Liver annotation to STEP domains and expression in space."""
    plot_coords = coords.copy()
    plot_coords[:, 1] *= -1
    liver = annotations == "Liver"
    score_by_global_index = {
        name: dict(zip(liver_indices, values, strict=True))
        for name, values in marker_scores.items()
    }
    score_limits = {
        name: max(
            abs(float(np.quantile(values, 0.02))),
            abs(float(np.quantile(values, 0.98))),
            0.1,
        )
        for name, values in marker_scores.items()
    }

    figure = plt.figure(figsize=(18.2, 11.8), constrained_layout=True)
    grid = figure.add_gridspec(
        4,
        6,
        width_ratios=(0.10, 1, 1, 1, 1, 1),
        height_ratios=(1.18, 1, 1, 1),
        wspace=0.02,
        hspace=0.025,
    )
    row_names = (
        "Manual Liver",
        "STEP domains",
        "Hepatic markers",
        "ECM/myogenic markers",
    )
    score_names = {
        2: "Hepatic",
        3: "Mesenchymal/myogenic",
    }
    continuous_axes: dict[str, list[plt.Axes]] = {
        "Hepatic": [],
        "Mesenchymal/myogenic": [],
    }
    mappables: dict[str, object] = {}

    for row, row_name in enumerate(row_names):
        row_axis = figure.add_subplot(grid[row, 0])
        row_axis.text(
            0.5,
            0.5,
            row_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        row_axis.axis("off")

        for column, section in enumerate(SECTION_ORDER, start=1):
            axis = figure.add_subplot(grid[row, column])
            section_mask = sections == section
            section_indices = np.flatnonzero(section_mask)
            liver_section = section_mask & liver
            liver_section_indices = np.flatnonzero(liver_section)
            liver_coords = plot_coords[liver_section_indices]
            liver_labels = domains[liver_section_indices]
            display_mask = connected_liver_display_mask(
                liver_coords,
                edge_radius,
            )
            display_indices = liver_section_indices[display_mask]
            display_coords = liver_coords[display_mask]
            display_labels = liver_labels[display_mask]

            if row == 0:
                axis.scatter(
                    plot_coords[section_indices, 0],
                    plot_coords[section_indices, 1],
                    c="#E1E1E1",
                    s=0.28,
                    linewidths=0,
                    rasterized=True,
                )
                axis.scatter(
                    liver_coords[:, 0],
                    liver_coords[:, 1],
                    c="#2F6B4F",
                    s=0.60,
                    linewidths=0,
                    rasterized=True,
                )
            elif row == 1:
                axis.scatter(
                    display_coords[:, 0],
                    display_coords[:, 1],
                    c="#C8C8C8",
                    s=1.20,
                    linewidths=0,
                    rasterized=True,
                )
                for domain in liver_domains:
                    mask = display_labels == domain
                    axis.scatter(
                        display_coords[mask, 0],
                        display_coords[mask, 1],
                        c=domain_colors[domain],
                        s=1.20,
                        linewidths=0,
                        rasterized=True,
                    )
            else:
                score_name = score_names[row]
                values = np.asarray(
                    [
                        score_by_global_index[score_name][index]
                        for index in display_indices
                    ]
                )
                score_limit = score_limits[score_name]
                mappable = axis.scatter(
                    display_coords[:, 0],
                    display_coords[:, 1],
                    c=values,
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(
                        vmin=-score_limit,
                        vcenter=0.0,
                        vmax=score_limit,
                    ),
                    s=1.20,
                    linewidths=0,
                    rasterized=True,
                )
                interfaces = domain_interface_segments(
                    display_coords,
                    display_labels,
                    liver_domains,
                    edge_radius,
                )
                if len(interfaces):
                    axis.add_collection(
                        LineCollection(
                            interfaces,
                            colors="#161616",
                            linewidths=0.16,
                            alpha=0.62,
                            rasterized=True,
                        )
                    )
                continuous_axes[score_name].append(axis)
                mappables[score_name] = mappable

            if row > 0:
                span = np.ptp(display_coords, axis=0)
                padding = max(float(span.max()) * 0.06, 2.0)
                axis.set_xlim(
                    float(display_coords[:, 0].min() - padding),
                    float(display_coords[:, 0].max() + padding),
                )
                axis.set_ylim(
                    float(display_coords[:, 1].min() - padding),
                    float(display_coords[:, 1].max() + padding),
                )

            axis.set_aspect("equal")
            axis.axis("off")
            if row == 0:
                short = section.removeprefix("E16.5_").removesuffix(".MOSTA")
                distance = SECTION_DISTANCE_UM[section]
                axis.set_title(f"{short} ({distance:+d} um)", fontsize=12)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=domain_colors[domain],
            markeredgecolor="none",
            markersize=7,
            label=f"Domain {domain}",
        )
        for domain in liver_domains
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#161616",
            linewidth=1.0,
            label="Domain interface",
        )
    )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=10,
    )
    for score_name, axes in continuous_axes.items():
        colorbar = figure.colorbar(
            mappables[score_name],
            ax=axes,
            location="right",
            fraction=0.018,
            pad=0.01,
        )
        colorbar.ax.tick_params(labelsize=8)

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, resolution=300.0)


def plot_domain_marker_alignment(
    output_path: Path,
    sections: np.ndarray,
    annotations: np.ndarray,
    domains: np.ndarray,
    coords: np.ndarray,
    liver_indices: np.ndarray,
    liver_domains: tuple[str, ...],
    domain_colors: dict[str, str],
    domain_marker_scores: dict[str, np.ndarray],
    edge_radius: float,
) -> None:
    """Show each Liver domain beside its own within-Liver marker pattern."""
    target_domains = tuple(
        domain for domain in liver_domains if domain in domain_marker_scores
    )
    if len(target_domains) != 3:
        raise ValueError("expected marker sets for three Liver domains")

    plot_coords = coords.copy()
    plot_coords[:, 1] *= -1
    liver = annotations == "Liver"
    score_by_global_index = {
        domain: dict(zip(liver_indices, values, strict=True))
        for domain, values in domain_marker_scores.items()
    }
    score_limits = {}
    for domain in target_domains:
        lower, upper = np.quantile(
            domain_marker_scores[domain], (0.02, 0.98)
        )
        score_limits[domain] = max(abs(float(lower)), abs(float(upper)), 0.1)
    representative_genes = {
        "2": "Ttr / Alb / Afp",
        "15": "Col1a2 / Col3a1 / Acta1",
        "26": "Peak1 / Nnat / Map1b",
    }
    row_names = ["Manual Liver", "STEP domains"] + [
        f"Domain {domain} score\n{representative_genes[domain]}"
        for domain in target_domains
    ]

    figure = plt.figure(figsize=(19.2, 13.4), constrained_layout=True)
    grid = figure.add_gridspec(
        5,
        6,
        width_ratios=(0.24, 1, 1, 1, 1, 1),
        height_ratios=(1.15, 1, 1, 1, 1),
        wspace=0.018,
        hspace=0.025,
    )
    continuous_axes: dict[str, list[plt.Axes]] = {
        domain: [] for domain in target_domains
    }
    mappables: dict[str, object] = {}

    for row, row_name in enumerate(row_names):
        row_axis = figure.add_subplot(grid[row, 0])
        row_axis.text(
            0.96,
            0.5,
            row_name,
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            linespacing=1.35,
        )
        row_axis.axis("off")

        for column, section in enumerate(SECTION_ORDER, start=1):
            axis = figure.add_subplot(grid[row, column])
            section_mask = sections == section
            section_indices = np.flatnonzero(section_mask)
            liver_section_indices = np.flatnonzero(
                section_mask & liver
            )
            liver_coords = plot_coords[liver_section_indices]
            liver_labels = domains[liver_section_indices]
            display_mask = connected_liver_display_mask(
                liver_coords,
                edge_radius,
            )
            display_indices = liver_section_indices[display_mask]
            display_coords = liver_coords[display_mask]
            display_labels = liver_labels[display_mask]

            if row == 0:
                axis.scatter(
                    plot_coords[section_indices, 0],
                    plot_coords[section_indices, 1],
                    c="#E1E1E1",
                    s=0.28,
                    linewidths=0,
                    rasterized=True,
                )
                axis.scatter(
                    liver_coords[:, 0],
                    liver_coords[:, 1],
                    c="#2F6B4F",
                    s=0.60,
                    linewidths=0,
                    rasterized=True,
                )
            elif row == 1:
                axis.scatter(
                    display_coords[:, 0],
                    display_coords[:, 1],
                    c="#C8C8C8",
                    s=1.20,
                    linewidths=0,
                    rasterized=True,
                )
                for domain in target_domains:
                    mask = display_labels == domain
                    axis.scatter(
                        display_coords[mask, 0],
                        display_coords[mask, 1],
                        c=domain_colors[domain],
                        s=1.20,
                        linewidths=0,
                        rasterized=True,
                    )
            else:
                target_domain = target_domains[row - 2]
                values = np.asarray(
                    [
                        score_by_global_index[target_domain][index]
                        for index in display_indices
                    ]
                )
                score_limit = score_limits[target_domain]
                marker_cmap = plt.get_cmap("RdBu_r")
                marker_norm = TwoSlopeNorm(
                    vmin=-score_limit,
                    vcenter=0.0,
                    vmax=score_limit,
                )
                axis.scatter(
                    display_coords[:, 0],
                    display_coords[:, 1],
                    c=values,
                    cmap=marker_cmap,
                    norm=marker_norm,
                    s=1.25,
                    linewidths=0,
                    alpha=1.0,
                    rasterized=True,
                )
                continuous_axes[target_domain].append(axis)
                mappables[target_domain] = plt.cm.ScalarMappable(
                    norm=marker_norm,
                    cmap=marker_cmap,
                )

            if row > 0:
                span = np.ptp(display_coords, axis=0)
                padding = max(float(span.max()) * 0.06, 2.0)
                axis.set_xlim(
                    float(display_coords[:, 0].min() - padding),
                    float(display_coords[:, 0].max() + padding),
                )
                axis.set_ylim(
                    float(display_coords[:, 1].min() - padding),
                    float(display_coords[:, 1].max() + padding),
                )

            axis.set_aspect("equal")
            axis.axis("off")
            if row == 0:
                short = section.removeprefix("E16.5_").removesuffix(".MOSTA")
                distance = SECTION_DISTANCE_UM[section]
                axis.set_title(f"{short} ({distance:+d} um)", fontsize=12)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=domain_colors[domain],
            markeredgecolor="none",
            markersize=7,
            label=f"Domain {domain}",
        )
        for domain in target_domains
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=10,
    )
    for domain, axes in continuous_axes.items():
        colorbar = figure.colorbar(
            mappables[domain],
            ax=axes,
            location="right",
            fraction=0.018,
            pad=0.01,
        )
        colorbar.ax.tick_params(labelsize=8)

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, resolution=300.0)


def plot_domain_marker_enrichment(
    output_path: Path,
    enrichment_table: pd.DataFrame,
    target_domains: tuple[str, ...],
    sections_to_plot: tuple[str, ...],
) -> None:
    """Plot score-domain odds ratios for each section."""
    indexed = enrichment_table.set_index(
        ["section", "assigned_domain", "marker_set_domain"]
    )

    section_matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    all_values = []
    for section in sections_to_plot:
        odds_ratio = np.full(
            (len(target_domains), len(target_domains)), np.nan
        )
        fdr = np.full_like(odds_ratio, np.nan)
        permutation_fdr = np.full_like(odds_ratio, np.nan)
        for row_index, assigned_domain in enumerate(target_domains):
            for column_index, marker_set_domain in enumerate(target_domains):
                row = indexed.loc[
                    (section, assigned_domain, marker_set_domain)
                ]
                if not bool(row["tested"]):
                    continue
                odds_ratio[row_index, column_index] = float(
                    row["odds_ratio_per_sd"]
                )
                fdr[row_index, column_index] = float(
                    row["mannwhitney_fdr_bh"]
                )
                permutation_fdr[row_index, column_index] = float(
                    row["permutation_two_sided_fdr_bh"]
                )
        section_matrices[section] = (odds_ratio, fdr, permutation_fdr)
        finite_odds = odds_ratio[np.isfinite(odds_ratio)]
        all_values.extend(np.log2(finite_odds))

    stacked = np.stack(
        [section_matrices[section][0] for section in sections_to_plot]
    )
    median_odds_ratio = np.exp2(np.nanmedian(np.log2(stacked), axis=0))
    finite = np.abs(np.asarray(all_values, dtype=np.float64))
    color_limit = max(float(finite.max(initial=0.0)), 0.10)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")
    image = None
    panels = [
        (section, *section_matrices[section]) for section in sections_to_plot
    ]
    panels.append(("Median", median_odds_ratio, None, None))
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(14.2, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (title, odds_ratio, fdr, permutation_fdr) in zip(
        axes.flat, panels
    ):
        log2_odds_ratio = np.log2(odds_ratio)
        image = axis.imshow(
            log2_odds_ratio,
            cmap=cmap,
            vmin=-color_limit,
            vmax=color_limit,
            aspect="equal",
        )
        short_title = (
            title.removeprefix("E16.5_").removesuffix(".MOSTA")
            if title != "Median"
            else "Median across sections"
        )
        axis.set_title(short_title, fontsize=11)
        axis.set_xticks(
            np.arange(len(target_domains)),
            [f"Score {domain}" for domain in target_domains],
        )
        axis.set_yticks(
            np.arange(len(target_domains)),
            [f"Domain {domain}" for domain in target_domains],
        )
        axis.tick_params(length=0, labelsize=9)
        for spine in axis.spines.values():
            spine.set_visible(False)

        for row_index in range(len(target_domains)):
            for column_index in range(len(target_domains)):
                value = odds_ratio[row_index, column_index]
                if not np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#666666",
                    )
                    continue
                significance = ""
                if fdr is not None and permutation_fdr is not None:
                    q_value = fdr[row_index, column_index]
                    random_q = permutation_fdr[row_index, column_index]
                    if q_value < 0.001 and random_q < 0.05:
                        significance = "***"
                    elif q_value < 0.01 and random_q < 0.05:
                        significance = "**"
                    elif q_value < 0.05 and random_q < 0.05:
                        significance = "*"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}x{significance}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold" if significance else "normal",
                    color=(
                        "white"
                        if abs(log2_odds_ratio[row_index, column_index])
                        > color_limit * 0.55
                        else "#202020"
                    ),
                )
                if row_index == column_index:
                    axis.add_patch(
                        Rectangle(
                            (column_index - 0.48, row_index - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor="#151515",
                            linewidth=1.25,
                        )
                    )

    figure.supxlabel("Marker score", fontsize=11)
    figure.supylabel("Assigned STEP domain", fontsize=11)
    if image is None:
        raise AssertionError("failed to create enrichment heatmap")
    colorbar = figure.colorbar(
        image, ax=axes, fraction=0.018, pad=0.015, location="right"
    )
    colorbar.set_label("log2 odds ratio per 1-SD score")
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image_file:
        image_file.convert("RGB").save(pdf_path, resolution=300.0)


def plot_marker_domain_colocalization(
    output_path: Path,
    colocalization_table: pd.DataFrame,
    target_domains: tuple[str, ...],
    sections_to_plot: tuple[str, ...],
) -> None:
    """Plot marker-score and domain-mask spatial co-localization."""
    indexed = colocalization_table.set_index(
        ["section", "assigned_domain", "marker_set_domain"]
    )
    section_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_values = []
    for section in sections_to_plot:
        correlation = np.full(
            (len(target_domains), len(target_domains)), np.nan
        )
        fdr = np.full_like(correlation, np.nan)
        for row_index, assigned_domain in enumerate(target_domains):
            for column_index, marker_set_domain in enumerate(target_domains):
                row = indexed.loc[
                    (section, assigned_domain, marker_set_domain)
                ]
                if not bool(row["tested"]):
                    continue
                correlation[row_index, column_index] = float(
                    row["spatial_rank_biserial"]
                )
                fdr[row_index, column_index] = float(
                    row["spatial_shift_fdr_bh"]
                )
        section_matrices[section] = (correlation, fdr)
        all_values.extend(correlation[np.isfinite(correlation)])

    stacked = np.stack(
        [section_matrices[section][0] for section in sections_to_plot]
    )
    median_correlation = np.nanmedian(stacked, axis=0)
    color_limit = max(
        float(np.abs(np.asarray(all_values)).max(initial=0.0)), 0.10
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")
    panels = [
        (section, *section_matrices[section]) for section in sections_to_plot
    ]
    panels.append(("Median", median_correlation, None))
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(14.2, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    image = None
    for axis, (title, correlation, fdr) in zip(axes.flat, panels):
        image = axis.imshow(
            correlation,
            cmap=cmap,
            vmin=-color_limit,
            vmax=color_limit,
            aspect="equal",
        )
        axis.set_title(
            title.removeprefix("E16.5_").removesuffix(".MOSTA")
            if title != "Median"
            else "Median across sections",
            fontsize=11,
        )
        axis.set_xticks(
            np.arange(len(target_domains)),
            [f"Score {domain}" for domain in target_domains],
        )
        axis.set_yticks(
            np.arange(len(target_domains)),
            [f"Domain {domain}" for domain in target_domains],
        )
        axis.tick_params(length=0, labelsize=9)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for row_index in range(len(target_domains)):
            for column_index in range(len(target_domains)):
                value = correlation[row_index, column_index]
                if not np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#666666",
                    )
                    continue
                significance = ""
                if fdr is not None:
                    q_value = fdr[row_index, column_index]
                    significance = (
                        "***"
                        if q_value < 0.001
                        else "**"
                        if q_value < 0.01
                        else "*"
                        if q_value < 0.05
                        else ""
                    )
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}{significance}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold" if significance else "normal",
                    color=(
                        "white"
                        if abs(value) > color_limit * 0.55
                        else "#202020"
                    ),
                )
                if row_index == column_index:
                    axis.add_patch(
                        Rectangle(
                            (column_index - 0.48, row_index - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor="#151515",
                            linewidth=1.25,
                        )
                    )

    figure.supxlabel("Composite marker score", fontsize=11)
    figure.supylabel("Assigned STEP domain", fontsize=11)
    if image is None:
        raise AssertionError("failed to create co-localization heatmap")
    colorbar = figure.colorbar(
        image, ax=axes, fraction=0.018, pad=0.015, location="right"
    )
    colorbar.set_label("Spatial rank-biserial correlation")
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image_file:
        image_file.convert("RGB").save(pdf_path, resolution=300.0)


def plot_quantitative_validation(
    output_path: Path,
    marker_table: pd.DataFrame,
    gene_expression_table: pd.DataFrame,
    permutation_table: pd.DataFrame,
    liver_domains: tuple[str, ...],
    domain_colors: dict[str, str],
    patch_sections: tuple[str, ...],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    short_sections = [
        section.removeprefix("E16.5_").removesuffix(".MOSTA")
        for section in patch_sections
    ]
    section_positions = np.arange(len(patch_sections))

    for axis, program in zip(
        axes[0],
        ("Hepatic", "Mesenchymal/myogenic"),
        strict=True,
    ):
        for domain in liver_domains:
            domain_table = (
                marker_table.loc[marker_table["domain"] == domain]
                .set_index("section")
                .reindex(patch_sections)
            )
            values = domain_table[program].where(
                domain_table["patch_domain_eligible"]
            ).to_numpy()
            axis.plot(
                section_positions,
                values,
                color=domain_colors[domain],
                marker="o",
                linewidth=1.8,
                markersize=5.5,
                label=f"Domain {domain}",
            )
        axis.set_xticks(section_positions, short_sections)
        axis.set_ylabel("Mean marker-program score")
        axis.set_title(program)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(frameon=False, ncol=len(liver_domains), fontsize=9)

    representative_genes = (
        "Ttr",
        "Alb",
        "Afp",
        "Apoa1",
        "Apoa2",
        "Ahsg",
        "Col1a1",
        "Col1a2",
        "Col3a1",
        "Acta1",
        "Mylpf",
        "Myh3",
    )
    eligible_domain_sections = marker_table.loc[
        marker_table["patch_domain_eligible"], ["section", "domain"]
    ]
    expression = gene_expression_table.loc[
        gene_expression_table["domain"].isin(liver_domains)
        & gene_expression_table["gene"].isin(representative_genes)
    ].merge(eligible_domain_sections, on=["section", "domain"], how="inner")
    gene_domain = (
        expression.groupby(["gene", "domain"], observed=True)[
            "mean_log_normalized_expression"
        ]
        .mean()
        .unstack("domain")
        .reindex(index=representative_genes, columns=liver_domains)
    )
    row_mean = gene_domain.mean(axis=1)
    row_sd = gene_domain.std(axis=1, ddof=0).replace(0, np.nan)
    gene_z = gene_domain.sub(row_mean, axis=0).div(row_sd, axis=0)
    heatmap = axes[1, 0].imshow(
        gene_z.to_numpy(),
        cmap="RdBu_r",
        vmin=-1.4,
        vmax=1.4,
        aspect="auto",
    )
    axes[1, 0].set_xticks(
        np.arange(len(liver_domains)),
        [f"Domain {domain}" for domain in liver_domains],
    )
    axes[1, 0].set_yticks(
        np.arange(len(representative_genes)), representative_genes
    )
    axes[1, 0].set_title("Representative marker genes")
    colorbar = figure.colorbar(heatmap, ax=axes[1, 0], fraction=0.045, pad=0.03)
    colorbar.set_label("Within-gene z score")

    permutation = permutation_table.set_index("section").reindex(patch_sections)
    axes[1, 1].errorbar(
        section_positions,
        permutation["permutation_mean"],
        yerr=np.vstack(
            [
                permutation["permutation_mean"] - permutation["permutation_q025"],
                permutation["permutation_q975"] - permutation["permutation_mean"],
            ]
        ),
        fmt="o",
        color="#8A8A8A",
        ecolor="#B7B7B7",
        capsize=3,
        markersize=5,
        label="Permuted labels",
    )
    axes[1, 1].plot(
        section_positions,
        permutation["observed_edge_agreement"],
        color="#202020",
        marker="o",
        linewidth=1.8,
        markersize=5.5,
        label="Observed",
    )
    axes[1, 1].set_xticks(section_positions, short_sections)
    axes[1, 1].set_ylabel("Same-domain local-edge fraction")
    axes[1, 1].set_title("Local spatial agreement")
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, resolution=300.0)


def plot_serial_section_context(
    output_path: Path,
    composition: pd.DataFrame,
    patch_eligibility: pd.DataFrame,
    liver_domains: tuple[str, ...],
    domain_colors: dict[str, str],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))
    positions = np.arange(len(SECTION_ORDER))
    labels = [
        f"{section.removeprefix('E16.5_').removesuffix('.MOSTA')}\n"
        f"{SECTION_DISTANCE_UM[section]:+d} um"
        for section in SECTION_ORDER
    ]

    bottom = np.zeros(len(SECTION_ORDER), dtype=np.float64)
    for domain in liver_domains:
        values = (
            composition.reindex(index=SECTION_ORDER)[domain]
            if domain in composition.columns
            else np.zeros(len(SECTION_ORDER))
        )
        values = np.asarray(values, dtype=np.float64)
        axes[0].bar(
            positions,
            values,
            bottom=bottom,
            color=domain_colors[domain],
            width=0.72,
            label=f"Domain {domain}",
        )
        bottom += values
    axes[0].bar(
        positions,
        np.maximum(1.0 - bottom, 0.0),
        bottom=bottom,
        color="#BDBDBD",
        width=0.72,
        label="Other",
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of manual Liver spots")
    axes[0].set_title("STEP-domain composition")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    section_context = patch_eligibility.set_index("section").reindex(
        SECTION_ORDER
    )
    axes[1].plot(
        positions,
        section_context["n_liver_spots"],
        color="#2F6B4F",
        marker="o",
        linewidth=1.8,
        markersize=6,
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Manual Liver spots")
    axes[1].set_title("Liver cross-section size")
    axes[1].spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, resolution=300.0)


def write_validation_report(
    output_path: Path,
    domain_table: pd.DataFrame,
    marker_table: pd.DataFrame,
    program_table: pd.DataFrame,
    permutation_table: pd.DataFrame,
    liver_domains: tuple[str, ...],
    patch_eligibility: pd.DataFrame,
    patch_sections: tuple[str, ...],
    top_gene_table: pd.DataFrame,
) -> None:
    lines = [
        "# MOSTA Liver domain validation",
        "",
        "## Domain correspondence",
        "",
    ]
    for domain in liver_domains:
        row = domain_table.loc[domain_table["domain"] == domain].iloc[0]
        lines.append(
            f"- Domain {domain}: {int(row['liver_spots']):,} Liver spots; "
            f"Liver purity {row['liver_purity']:.3f}."
        )

    lines.extend(["", "## Patch-analysis scope", ""])
    for row in patch_eligibility.itertuples(index=False):
        short_section = row.section.removesuffix(".MOSTA")
        if row.patch_eligible:
            lines.append(
                f"- {short_section}: included; {row.n_patch_domains} dominant "
                "Liver-associated domains each occupy at least "
                f"{row.min_domain_fraction:.0%} of Liver spots."
            )
        else:
            lines.append(
                f"- {short_section}: excluded from patch-level statistics; "
                f"the leading domain occupies {row.leading_domain_fraction:.1%} "
                "of Liver spots and no second dominant domain reaches "
                f"{row.min_domain_fraction:.0%}."
            )

    s1_section = SECTION_ORDER[0]
    s1_domain = str(
        patch_eligibility.set_index("section").loc[
            s1_section, "leading_domain"
        ]
    )
    s1_programs = marker_table.loc[
        (marker_table["section"] == s1_section)
        & (marker_table["domain"] == s1_domain)
    ].iloc[0]
    domain_2_top = set(
        top_gene_table.loc[
            top_gene_table["domain"] == liver_domains[0], "gene"
        ]
    )
    domain_15_top = set(
        top_gene_table.loc[
            top_gene_table["domain"] == liver_domains[1], "gene"
        ]
    )
    domain_2_genes = [
        gene
        for gene in (
            "Ttr",
            "Ahsg",
            "Apoa2",
            "Apoa1",
            "Alb",
            "Trf",
            "Afp",
            "Apoc2",
            "Rbp4",
            "Apom",
        )
        if gene in domain_2_top
    ]
    domain_15_genes = [
        gene
        for gene in (
            "Col1a2",
            "Col3a1",
            "Acta1",
            "Mylpf",
            "Actc1",
            "Dlk1",
            "Col1a1",
            "Cdkn1c",
            "H19",
            "Myh3",
        )
        if gene in domain_15_top
    ]
    lines.extend(
        [
            "",
            "## Serial-section developmental context",
            "",
            "All five samples are sagittal sections from the same E16.5 embryo, "
            "not a developmental time series. Their reported distances from the "
            "embryonic midline are -2120, -1150, 0, +740, and +1770 um for S1, "
            "S4, S7, S10, and S13, respectively.",
            "",
            f"S1 is the most lateral selected plane and contains the smallest "
            f"manual Liver cross-section ({int(s1_programs['n_spots']):,} spots "
            f"in its leading Domain {s1_domain}; 4,755 Liver spots in total). "
            f"This dominant state retains both hepatic and erythroid expression "
            f"(marker-program scores {s1_programs['Hepatic']:.3f} and "
            f"{s1_programs['Erythroid']:.3f}), so its near-uniform assignment is "
            "not evidence of a non-liver or random region.",
            "",
            f"Across the serial sections, Domain {liver_domains[0]} is marked by "
            f"hepatic genes including {', '.join(domain_2_genes)}, whereas "
            f"Domain {liver_domains[1]} is shifted toward fetal/ECM and myogenic "
            f"genes including {', '.join(domain_15_genes)}. A biologically "
            "plausible interpretation is therefore that the lateral S1 plane "
            "intersects a comparatively uniform liver lobe/state, while more "
            "central and contralateral sections intersect different lobes and "
            "additional hepatic-stromal interfaces.",
            "",
            "This interpretation is consistent with the established left-right "
            "identity and lobe organization of the E16.5 mouse liver and with "
            "the mixed hepatic, hematopoietic, and stromal composition of fetal "
            "liver. The present data do not assign an exact lobe identity to "
            "each section and do not establish each inferred patch as a "
            "separately validated functional unit.",
        ]
    )
    lines.extend(["", "## Marker-program evidence", ""])
    target_programs = (
        (liver_domains[0], "Hepatic"),
        (liver_domains[1], "Mesenchymal/myogenic"),
    )
    for domain, program in target_programs:
        rows = program_table.loc[
            (program_table["domain"] == domain)
            & (program_table["program"] == program)
        ].sort_values("section")
        positive = int((rows["mean_difference"] > 0).sum())
        lines.append(
            f"- Domain {domain}, {program}: higher than the other Liver "
            f"domains in {positive}/{len(rows)} sections; rank-biserial effect "
            f"range {rows['rank_biserial'].min():.3f} to "
            f"{rows['rank_biserial'].max():.3f}; BH-FDR range "
            f"{rows['fdr_bh'].min():.3g} to {rows['fdr_bh'].max():.3g}."
        )

    lines.extend(["", "## Spatial non-randomness", ""])
    for row in permutation_table.itertuples(index=False):
        short_section = row.section.removesuffix(".MOSTA")
        lines.append(
            f"- {short_section}: observed same-domain local-edge fraction "
            f"{row.observed_edge_agreement:.3f}; permutation mean "
            f"{row.permutation_mean:.3f} (95% interval "
            f"{row.permutation_q025:.3f}-{row.permutation_q975:.3f}); "
            f"empirical P={row.permutation_p_value:.4g}; z={row.z_score:.1f}."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The dominant STEP assignments within the coarse manual Liver "
            "annotation are transcriptionally distinguishable and more locally "
            "contiguous than expected after section-wise label permutation. "
            "This supports structured intra-liver heterogeneity. It does not, "
            "without additional functional validation, establish every spatial "
            "patch as a separate functional unit.",
            "",
            "Program scores are means of log1p library-size-normalized raw-count "
            "expression for the genes listed in summary.json. Per-section "
            "domain-versus-other-Liver comparisons use two-sided Mann-Whitney U "
            "tests with BH correction across all reported domain-program tests. "
            "Spatial null distributions preserve each section's domain counts "
            "and randomly permute labels over the observed Liver coordinates. "
            "Patch-level comparisons are restricted to the sections listed as "
            f"eligible above ({', '.join(section.removesuffix('.MOSTA') for section in patch_sections)}).",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.source, backed="r")
    annotations = adata.obs["annotation"].astype(str).to_numpy()
    domains = adata.obs["domain"].astype(str).to_numpy()
    sections = adata.obs["batch"].astype(str).to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    liver_indices = np.flatnonzero(annotations == "Liver")
    liver_domain_counts = pd.Series(domains[liver_indices]).value_counts()
    liver_domains = tuple(str(value) for value in liver_domain_counts.index[:3])

    patch_eligibility_rows = []
    for section in SECTION_ORDER:
        labels = domains[(sections == section) & (annotations == "Liver")]
        proportions = pd.Series(labels).value_counts(normalize=True)
        selected_proportions = np.asarray(
            [float(proportions.get(domain, 0.0)) for domain in liver_domains]
        )
        n_patch_domains = int(
            np.count_nonzero(
                selected_proportions >= args.min_patch_domain_fraction
            )
        )
        patch_eligibility_rows.append(
            {
                "section": section,
                "distance_from_midline_um": SECTION_DISTANCE_UM[section],
                "n_liver_spots": int(len(labels)),
                "leading_domain": str(proportions.index[0]),
                "leading_domain_fraction": float(proportions.iloc[0]),
                "n_patch_domains": n_patch_domains,
                "min_domain_fraction": args.min_patch_domain_fraction,
                "patch_eligible": n_patch_domains >= args.min_patch_domains,
            }
        )
    patch_eligibility = pd.DataFrame(patch_eligibility_rows)
    patch_eligibility.to_csv(
        args.output_dir / "section_patch_eligibility.csv", index=False
    )
    patch_sections = tuple(
        patch_eligibility.loc[patch_eligibility["patch_eligible"], "section"]
    )

    categories = [str(value) for value in adata.obs["domain"].cat.categories]
    colors = [str(value) for value in adata.uns["domain_colors"]]
    domain_colors = dict(zip(categories, colors, strict=True))

    raw_liver_expression = load_raw_liver_expression(
        args.raw_dir,
        adata.obs_names,
        sections,
        annotations,
        domains,
        liver_indices,
    )
    marker_scores = score_gene_sets(
        raw_liver_expression.X,
        pd.Index(raw_liver_expression.var_names),
        MARKER_GROUPS,
        args.seed + 100,
        "program",
    )
    genes_used = {
        name: [
            gene for gene in genes if gene in raw_liver_expression.var_names
        ]
        for name, genes in MARKER_GROUPS.items()
    }
    domain_marker_scores = score_gene_sets(
        raw_liver_expression.X,
        pd.Index(raw_liver_expression.var_names),
        DOMAIN_MARKER_GENES,
        args.seed,
        "domain",
    )
    gene_expression_table = summarize_marker_gene_expression(
        raw_liver_expression
    )
    gene_expression_table.to_csv(
        args.output_dir / "gene_expression_by_domain.csv", index=False
    )
    top_gene_table = full_transcriptome_de(
        raw_liver_expression,
        liver_domains,
        args.top_genes,
    )
    top_gene_table.to_csv(
        args.output_dir / "top_genes_within_liver.csv", index=False
    )

    domain_rows = []
    for domain, liver_count in liver_domain_counts.items():
        domain_mask = domains == str(domain)
        domain_rows.append(
            {
                "domain": str(domain),
                "liver_spots": int(liver_count),
                "all_domain_spots": int(domain_mask.sum()),
                "liver_purity": float(
                    np.mean(annotations[domain_mask] == "Liver")
                ),
            }
        )
    domain_table = pd.DataFrame(domain_rows)
    domain_table.to_csv(args.output_dir / "domain_liver_purity.csv", index=False)

    composition = pd.crosstab(
        pd.Series(sections[liver_indices], name="section"),
        pd.Series(domains[liver_indices], name="domain"),
        normalize="index",
    )
    composition.to_csv(args.output_dir / "section_domain_composition.csv")

    marker_rows = []
    for section in SECTION_ORDER:
        section_liver_mask = sections[liver_indices] == section
        n_section_liver = int(section_liver_mask.sum())
        for domain in liver_domains:
            mask = section_liver_mask & (domains[liver_indices] == domain)
            if not mask.any():
                continue
            domain_fraction = float(mask.sum() / n_section_liver)
            row: dict[str, float | int | str] = {
                "section": section,
                "domain": domain,
                "n_spots": int(mask.sum()),
                "fraction_of_liver": domain_fraction,
                "patch_domain_eligible": (
                    section in patch_sections
                    and domain_fraction >= args.min_patch_domain_fraction
                ),
            }
            for name, values in marker_scores.items():
                row[name] = float(values[mask].mean())
            marker_rows.append(row)
    marker_table = pd.DataFrame(marker_rows)
    marker_table.to_csv(args.output_dir / "marker_program_scores.csv", index=False)
    program_table = program_statistics(
        marker_scores,
        sections,
        domains,
        liver_indices,
        liver_domains,
        patch_sections,
        args.min_patch_domain_fraction,
    )
    program_table.to_csv(
        args.output_dir / "program_statistics.csv", index=False
    )
    domain_marker_programs = {
        f"Domain {domain} markers": values
        for domain, values in domain_marker_scores.items()
    }
    all_domain_marker_statistics = program_statistics(
        domain_marker_programs,
        sections,
        domains,
        liver_indices,
        liver_domains,
        patch_sections,
        args.min_patch_domain_fraction,
    )
    matched_domain_marker_statistics = pd.concat(
        [
            all_domain_marker_statistics.loc[
                (all_domain_marker_statistics["domain"] == domain)
                & (
                    all_domain_marker_statistics["program"]
                    == f"Domain {domain} markers"
                )
            ]
            for domain in liver_domains
        ],
        ignore_index=True,
    )
    matched_domain_marker_statistics.to_csv(
        args.output_dir / "domain_marker_statistics.csv",
        index=False,
    )
    marker_score_table = domain_marker_score_statistics(
        domain_marker_scores,
        sections,
        domains,
        liver_indices,
        liver_domains,
        patch_sections,
        args.n_permutations,
        np.random.default_rng(args.seed + 1),
    )
    marker_score_table.to_csv(
        args.output_dir / "domain_marker_score_contrasts.csv",
        index=False,
    )
    marker_score_table.loc[
        :,
        [
            "section",
            "assigned_domain",
            "marker_set_domain",
            "n_target",
            "n_background",
            "domain_fraction",
            "tested",
            "odds_ratio_per_sd",
            "odds_ratio_ci025",
            "odds_ratio_ci975",
            "logistic_beta_per_sd",
            "logistic_p_value",
        ],
    ].to_csv(
        args.output_dir / "domain_marker_odds_ratios.csv",
        index=False,
    )
    marker_colocalization_table = marker_domain_spatial_colocalization(
        domain_marker_scores,
        sections,
        domains,
        coords,
        liver_indices,
        liver_domains,
        patch_sections,
        args.n_permutations,
        np.random.default_rng(args.seed + 2),
    )
    marker_colocalization_table.to_csv(
        args.output_dir / "domain_marker_spatial_colocalization.csv",
        index=False,
    )
    spatial_rows = []
    permutation_rows = []
    rng = np.random.default_rng(args.seed)
    for section in SECTION_ORDER:
        section_indices = np.flatnonzero(
            (sections == section) & (annotations == "Liver")
        )
        labels = domains[section_indices]
        pairs = cKDTree(coords[section_indices]).query_pairs(
            r=args.edge_radius, output_type="ndarray"
        )
        same_domain_fraction = float(
            np.mean(labels[pairs[:, 0]] == labels[pairs[:, 1]])
        )
        proportions = pd.Series(labels).value_counts(normalize=True).to_numpy()
        random_expectation = float(np.square(proportions).sum())
        patch_eligible = section in patch_sections
        if patch_eligible:
            permutation_row = permutation_edge_agreement(
                labels,
                pairs,
                args.n_permutations,
                rng,
            )
            permutation_row.update(
                {
                    "section": section,
                    "n_liver_spots": int(len(section_indices)),
                    "n_liver_edges": int(len(pairs)),
                }
            )
            permutation_rows.append(permutation_row)
        for domain in liver_domains:
            row = component_metrics(
                labels,
                pairs,
                domain,
                args.min_component_size,
            )
            row.update(
                {
                    "section": section,
                    "n_liver_spots": int(len(section_indices)),
                    "n_liver_edges": int(len(pairs)),
                    "same_domain_edge_fraction": same_domain_fraction,
                    "random_label_expectation": random_expectation,
                    "edge_agreement_enrichment": float(
                        same_domain_fraction / random_expectation
                    ),
                    "patch_eligible": patch_eligible,
                }
            )
            spatial_rows.append(row)
    spatial_table = pd.DataFrame(spatial_rows)
    spatial_table.to_csv(args.output_dir / "spatial_structure.csv", index=False)
    permutation_table = pd.DataFrame(permutation_rows)
    permutation_table.to_csv(
        args.output_dir / "spatial_permutation_test.csv", index=False
    )

    plot_validation(
        args.output_dir / "mosta_liver_domain_validation",
        sections,
        annotations,
        domains,
        coords,
        liver_indices,
        liver_domains,
        domain_colors,
        marker_scores,
    )
    plot_spatial_evidence_story(
        args.output_dir / "mosta_liver_spatial_evidence_story",
        sections,
        annotations,
        domains,
        coords,
        liver_indices,
        liver_domains,
        domain_colors,
        marker_scores,
        args.edge_radius,
    )
    plot_domain_marker_alignment(
        args.output_dir / "mosta_liver_domain_marker_alignment",
        sections,
        annotations,
        domains,
        coords,
        liver_indices,
        liver_domains,
        domain_colors,
        domain_marker_scores,
        args.edge_radius,
    )
    plot_domain_marker_enrichment(
        args.output_dir / "mosta_liver_domain_marker_enrichment",
        marker_score_table,
        liver_domains,
        patch_sections,
    )
    plot_marker_domain_colocalization(
        args.output_dir / "mosta_liver_marker_domain_colocalization",
        marker_colocalization_table,
        liver_domains,
        patch_sections,
    )
    plot_quantitative_validation(
        args.output_dir / "mosta_liver_patch_quantitative",
        marker_table,
        gene_expression_table,
        permutation_table,
        liver_domains,
        domain_colors,
        patch_sections,
    )
    plot_serial_section_context(
        args.output_dir / "mosta_liver_serial_section_context",
        composition,
        patch_eligibility,
        liver_domains,
        domain_colors,
    )
    write_validation_report(
        args.output_dir / "validation_report.md",
        domain_table,
        marker_table,
        program_table,
        permutation_table,
        liver_domains,
        patch_eligibility,
        patch_sections,
        top_gene_table,
    )
    summary = {
        "source": str(args.source),
        "raw_count_source_dir": str(args.raw_dir),
        "n_liver_spots": int(len(liver_indices)),
        "liver_domains": list(liver_domains),
        "patch_sections": list(patch_sections),
        "domain_liver_purity": {
            row["domain"]: row["liver_purity"]
            for row in domain_rows
            if row["domain"] in liver_domains
        },
        "mean_edge_agreement_enrichment": float(
            spatial_table.drop_duplicates("section")[
                "edge_agreement_enrichment"
            ].mean()
        ),
        "spatial_permutation_tests": {
            row["section"]: {
                "observed": row["observed_edge_agreement"],
                "null_mean": row["permutation_mean"],
                "p_value": row["permutation_p_value"],
                "z_score": row["z_score"],
            }
            for row in permutation_rows
        },
        "marker_genes": genes_used,
        "domain_marker_genes": DOMAIN_MARKER_GENES,
        "marker_score_method": {
            "implementation": "scanpy.tl.score_genes",
            "expression": "log1p CP10K from raw full-transcriptome counts",
            "n_gene_pool": int(raw_liver_expression.n_vars),
            "ctrl_size": 50,
            "n_bins": 25,
            "ctrl_as_ref": True,
        },
        "domain_marker_score_contrasts": marker_score_table.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "domain_marker_spatial_colocalization": (
            marker_colocalization_table.replace({np.nan: None}).to_dict(
                orient="records"
            )
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "figure_legend.md").write_text(
        "Gray indicates spots outside the manual Liver annotation, and green "
        "indicates manual Liver. Red, blue, and cyan indicate STEP Domains 2, "
        "15, and 26, respectively. Continuous colors indicate Scanpy marker "
        "scores.\n",
        encoding="utf-8",
    )
    adata.file.close()


if __name__ == "__main__":
    main()
