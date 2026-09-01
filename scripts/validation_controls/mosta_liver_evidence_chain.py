#!/usr/bin/env python
"""Evaluate biological evidence for MOSTA liver domains."""

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import gseapy as gp
import matplotlib
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image
from scipy import sparse
from scipy.stats import fisher_exact, mannwhitneyu, rankdata
from sklearn.metrics import average_precision_score
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

import mosta_liver_domain_validation as liver_validation


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_DIR = Path("workflows/mosta_liver_validation/biological_evidence")
GO_LIBRARY = "GO_Biological_Process_2025"
DOMAIN_ORDER = ("2", "15", "26")
DOMAIN_COLORS = {"2": "#D9486E", "15": "#4C78E8", "26": "#13BFB5"}
UNINFORMATIVE_GENE_NAME = re.compile(
    r"^(?:Gm\d+|Rpl|Rps|mt-|AC\d|D\d+.*Rik)|Rik$",
    flags=re.IGNORECASE,
)


# Each gene below is explicitly supported by the cited primary source. These
# programs are independent biological priors and are not selected from STEP.
LITERATURE_PROGRAMS = {
    "Fetal hepatocyte": (
        "Hnf4a",
        "Afp",
        "Alb",
        "Ahsg",
        "Apoa1",
        "Apoa2",
        "Trf",
        "Rbp4",
        "Fgb",
        "Fxyd1",
        "Gjb1",
    ),
    "Stellate/stromal": (
        "Pdgfra",
        "Ncam1",
        "Dcn",
        "Hgf",
        "Lum",
        "Pdgfrb",
        "Col6a1",
        "Col6a2",
    ),
    "Primitive erythroid": (
        "Kit",
        "Gata1",
        "Bpgm",
        "Gypa",
        "Hba-a1",
        "Hba-x",
        "Hbb-y",
        "Hbb-bh1",
    ),
    "Endothelial": (
        "Kdr",
        "Pecam1",
        "Cdh5",
        "Emcn",
        "Eng",
        "Esam",
        "Flt4",
    ),
    "Mesothelial": (
        "Wt1",
        "Upk3b",
        "Pdpn",
        "Krt7",
    ),
    "Kupffer/macrophage": (
        "Cd68",
        "C1qa",
        "C1qb",
        "C1qc",
    ),
}


SOURCE_RECORDS = (
    {
        "source_id": "Chen2022MOSTA",
        "title": "Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays",
        "journal": "Cell",
        "year": 2022,
        "doi": "10.1016/j.cell.2022.04.003",
        "url": "https://doi.org/10.1016/j.cell.2022.04.003",
        "role": "Original MOSTA data source",
    },
    {
        "source_id": "Wang2020FetalLiver",
        "title": "Comparative analysis of cell lineage differentiation during hepatogenesis in humans and mice at the single-cell transcriptome level",
        "journal": "Cell Research",
        "year": 2020,
        "doi": "10.1038/s41422-020-0378-6",
        "url": "https://doi.org/10.1038/s41422-020-0378-6",
        "role": "Independent mouse fetal-liver lineage markers",
    },
    {
        "source_id": "Lu2021FetalLiverMERFISH",
        "title": "Spatial transcriptome profiling by MERFISH reveals fetal liver hematopoietic stem cell niche architecture",
        "journal": "Cell Discovery",
        "year": 2021,
        "doi": "10.1038/s41421-021-00266-1",
        "url": "https://doi.org/10.1038/s41421-021-00266-1",
        "role": "Independent spatial precedent for marker-defined fetal-liver cell types and DEG-GO validation",
    },
    {
        "source_id": "Subramanian2005GSEA",
        "title": "Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles",
        "journal": "Proceedings of the National Academy of Sciences",
        "year": 2005,
        "doi": "10.1073/pnas.0506580102",
        "url": "https://doi.org/10.1073/pnas.0506580102",
        "role": "GSEA method",
    },
    {
        "source_id": "GeneOntology2023",
        "title": "The Gene Ontology knowledgebase in 2023",
        "journal": "Genetics",
        "year": 2023,
        "doi": "10.1093/genetics/iyad031",
        "url": "https://doi.org/10.1093/genetics/iyad031",
        "role": "Gene Ontology biological-process knowledgebase",
    },
    {
        "source_id": "Kuleshov2016Enrichr",
        "title": "Enrichr: a comprehensive gene set enrichment analysis web server 2016 update",
        "journal": "Nucleic Acids Research",
        "year": 2016,
        "doi": "10.1093/nar/gkw377",
        "url": "https://doi.org/10.1093/nar/gkw377",
        "role": "Source of the versioned mouse GO Biological Process library",
    },
)


def marker_provenance() -> pd.DataFrame:
    rows = []
    wang_evidence = {
        "Hnf4a": "Mouse hepatoblast/hepatocyte marker, Fig. 1",
        "Afp": "Mouse hepatoblast/hepatocyte marker, Fig. 1; co-expression validation at E14.5",
        "Alb": "Co-expression validation in E14.5 fetal-liver hepatoblasts",
        "Ahsg": "Mouse hepatocyte marker, Supplementary Table S2",
        "Apoa1": "Mouse hepatocyte marker, Supplementary Table S2",
        "Apoa2": "Mouse hepatocyte marker, Supplementary Table S2",
        "Trf": "Mouse hepatocyte marker, Supplementary Table S2",
        "Rbp4": "Mouse hepatocyte marker, Supplementary Table S2",
        "Fgb": "Fetal hepatoblast marker validated by lineage tracing and E14.5 single-cell RT-qPCR",
        "Fxyd1": "Hepatic lineage marker validated with HNF4A in E17.5 mouse liver",
        "Gjb1": "Hepatic lineage marker validated with HNF4A in E17.5 mouse liver",
        "Pdgfra": "Septum transversum cell marker, Fig. 1",
        "Ncam1": "Septum transversum cell marker, Fig. 1",
        "Dcn": "Hepatic stellate-cell marker, Fig. 1",
        "Hgf": "Hepatic stellate-cell marker, Fig. 1",
        "Lum": "Mouse hepatic stellate-cell marker, Supplementary Table S2",
        "Pdgfrb": "Mouse hepatic stellate-cell marker, Supplementary Table S2",
        "Col6a1": "Mouse hepatic stellate-cell marker, Supplementary Table S2",
        "Col6a2": "Mouse hepatic stellate-cell marker, Supplementary Table S2",
        "Pdpn": "Mesothelial-cell marker, Fig. 1",
        "Kit": "Erythroid-progenitor marker, Fig. 1",
        "Gata1": "Erythroid-progenitor marker, Fig. 1",
        "Bpgm": "Early erythroid-lineage marker, Fig. 1",
        "Gypa": "Early erythrocyte marker, Fig. 1",
        "Hba-a1": "Mouse fetal erythrocyte marker, Supplementary Fig. S1",
        "Hba-x": "Primitive erythrocyte marker, Fig. 1",
        "Hbb-y": "Mouse primitive erythrocyte marker, Supplementary Table S2",
        "Hbb-bh1": "Mouse primitive erythrocyte marker, Supplementary Table S2",
        "Kdr": "Mouse fetal-liver endothelial marker, Supplementary Table S2",
        "Pecam1": "Mouse fetal-liver endothelial marker, Supplementary Table S2",
        "Cdh5": "Mouse fetal-liver endothelial marker, Supplementary Table S2",
        "Emcn": "Mouse fetal-liver endothelial marker, Supplementary Table S2",
        "Eng": "Mouse fetal-liver endothelial marker, Supplementary Table S2",
        "Esam": "Mouse fetal-liver endothelial marker, Supplementary Table S2",
        "Flt4": "Fetal-liver endothelial marker, Fig. 1",
        "Wt1": "Mouse fetal-liver mesothelial marker, Supplementary Table S2",
        "Upk3b": "Mouse fetal-liver mesothelial marker, Supplementary Table S2",
        "Krt7": "Mouse fetal-liver mesothelial marker, Supplementary Table S2",
        "Cd68": "Fetal-liver Kupffer-cell marker, Fig. 1",
        "C1qa": "Mouse fetal-liver Kupffer-cell marker, Supplementary Table S2",
        "C1qb": "Mouse fetal-liver Kupffer-cell marker, Supplementary Table S2",
        "C1qc": "Mouse fetal-liver Kupffer-cell marker, Supplementary Table S2",
    }
    for program in LITERATURE_PROGRAMS:
        for gene in LITERATURE_PROGRAMS[program]:
            rows.append(
                {
                    "program": program,
                    "gene": gene,
                    "source_id": "Wang2020FetalLiver",
                    "source_location_and_evidence": wang_evidence[gene],
                    "doi": "10.1038/s41422-020-0378-6",
                    "url": "https://doi.org/10.1038/s41422-020-0378-6",
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=liver_validation.DEFAULT_SOURCE)
    parser.add_argument("--raw-dir", type=Path, default=liver_validation.DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-spots", type=int, default=20)
    parser.add_argument("--marker-high-fraction", type=float, default=0.15)
    parser.add_argument("--min-display-domain-fraction", type=float, default=0.05)
    parser.add_argument("--gsea-permutations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--skip-gsea", action="store_true")
    return parser.parse_args()


def save_raster_pdf(figure: plt.Figure, output_path: Path) -> None:
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, resolution=300.0)


def score_literature_programs(expression: ad.AnnData, seed: int) -> dict[str, np.ndarray]:
    return liver_validation.score_gene_sets(
        expression.X,
        pd.Index(expression.var_names),
        LITERATURE_PROGRAMS,
        seed,
        "literature_program",
    )


def marker_associations(
    expression: ad.AnnData,
    scores: dict[str, np.ndarray],
    min_spots: int,
    marker_high_fraction: float,
) -> pd.DataFrame:
    sections = expression.obs["section"].astype(str).to_numpy()
    domains = expression.obs["domain"].astype(str).to_numpy()
    rows = []
    for section in liver_validation.SECTION_ORDER:
        section_mask = sections == section
        section_domains = domains[section_mask]
        for domain in DOMAIN_ORDER:
            target = section_domains == domain
            background = ~target
            tested = int(target.sum()) >= min_spots and int(background.sum()) >= min_spots
            for program, all_values in scores.items():
                row = {
                    "section": section,
                    "domain": domain,
                    "program": program,
                    "n_domain": int(target.sum()),
                    "n_background": int(background.sum()),
                    "domain_fraction": float(target.mean()),
                    "tested": tested,
                }
                if tested:
                    values = all_values[section_mask]
                    n_high = max(1, int(np.ceil(marker_high_fraction * len(values))))
                    high = np.zeros(len(values), dtype=bool)
                    high[np.argsort(values, kind="stable")[-n_high:]] = True
                    overlap = int(np.count_nonzero(target & high))
                    high_background = int(np.count_nonzero(background & high))
                    low_domain = int(np.count_nonzero(target & ~high))
                    low_background = int(np.count_nonzero(background & ~high))
                    expected_overlap = float(n_high * target.mean())
                    table = np.asarray(
                        [[overlap, high_background], [low_domain, low_background]],
                        dtype=np.int64,
                    )
                    fisher = fisher_exact(table, alternative="two-sided")
                    test = mannwhitneyu(
                        values[target],
                        values[background],
                        alternative="two-sided",
                        method="asymptotic",
                    )
                    rank_biserial = (
                        2.0 * float(test.statistic)
                        / float(target.sum() * background.sum())
                        - 1.0
                    )
                    row.update(
                        {
                            "domain_mean": float(values[target].mean()),
                            "background_mean": float(values[background].mean()),
                            "mean_difference": float(
                                values[target].mean() - values[background].mean()
                            ),
                            "rank_biserial": rank_biserial,
                            "score_mannwhitney_p_value": float(test.pvalue),
                            "marker_high_fraction_requested": marker_high_fraction,
                            "n_marker_high": n_high,
                            "marker_high_fraction_observed": float(high.mean()),
                            "n_marker_high_in_domain": overlap,
                            "expected_marker_high_in_domain": expected_overlap,
                            "observed_expected_enrichment": float(
                                overlap / expected_overlap
                            ),
                            "fraction_marker_high_in_domain": float(
                                overlap / target.sum()
                            ),
                            "fraction_domain_among_marker_high": float(
                                overlap / n_high
                            ),
                            "fisher_odds_ratio": float(fisher.statistic),
                            "fisher_p_value": float(fisher.pvalue),
                            "average_precision": float(
                                average_precision_score(target, values)
                            ),
                            "average_precision_lift": float(
                                average_precision_score(target, values)
                                / target.mean()
                            ),
                            "table_high_domain": overlap,
                            "table_high_background": high_background,
                            "table_low_domain": low_domain,
                            "table_low_background": low_background,
                        }
                    )
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["score_mannwhitney_fdr_bh"] = np.nan
    frame["fisher_fdr_bh"] = np.nan
    tested = frame["tested"].astype(bool)
    frame.loc[tested, "score_mannwhitney_fdr_bh"] = multipletests(
        frame.loc[tested, "score_mannwhitney_p_value"].to_numpy(),
        method="fdr_bh",
    )[1]
    frame.loc[tested, "fisher_fdr_bh"] = multipletests(
        frame.loc[tested, "fisher_p_value"].to_numpy(), method="fdr_bh"
    )[1]
    return frame


def summarize_marker_enrichment(associations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tested = associations.loc[associations["tested"]].copy()
    for (domain, program), group in tested.groupby(["domain", "program"]):
        tables = []
        for row in group.itertuples(index=False):
            tables.append(
                np.asarray(
                    [
                        [row.table_high_domain, row.table_high_background],
                        [row.table_low_domain, row.table_low_background],
                    ],
                    dtype=np.float64,
                )
            )
        stratified = StratifiedTable(
            np.stack(tables, axis=2), shift_zeros=True
        )
        odds_ratio_ci_low, odds_ratio_ci_high = (
            stratified.oddsratio_pooled_confint()
        )
        observed = float(group["n_marker_high_in_domain"].sum())
        expected = float(group["expected_marker_high_in_domain"].sum())
        rows.append(
            {
                "domain": str(domain),
                "program": program,
                "tested_sections": int(group["section"].nunique()),
                "sections_enrichment_above_one": int(
                    (group["observed_expected_enrichment"] > 1.0).sum()
                ),
                "total_marker_high_spots": int(group["n_marker_high"].sum()),
                "total_observed_marker_high_in_domain": int(observed),
                "total_expected_marker_high_in_domain": expected,
                "observed_expected_enrichment": observed / expected,
                "mantel_haenszel_common_odds_ratio": float(
                    stratified.oddsratio_pooled
                ),
                "mantel_haenszel_odds_ratio_ci95_low": float(
                    odds_ratio_ci_low
                ),
                "mantel_haenszel_odds_ratio_ci95_high": float(
                    odds_ratio_ci_high
                ),
                "mantel_haenszel_p_value": float(
                    stratified.test_null_odds().pvalue
                ),
                "median_average_precision_lift": float(
                    group["average_precision_lift"].median()
                ),
                "median_rank_biserial": float(group["rank_biserial"].median()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["mantel_haenszel_fdr_bh"] = multipletests(
        summary["mantel_haenszel_p_value"].to_numpy(), method="fdr_bh"
    )[1]
    return summary.sort_values(
        ["domain", "observed_expected_enrichment"], ascending=[True, False]
    )


def _masked_triangulation(coords: np.ndarray) -> mtri.Triangulation | None:
    if len(coords) < 3:
        return None
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1])
    triangles = tri.triangles
    points = coords[triangles]
    lengths = np.linalg.norm(points - np.roll(points, 1, axis=1), axis=2)
    tri.set_mask(lengths.max(axis=1) > 1.55)
    return tri


def plot_spatial_evidence(
    output_path: Path,
    sections: np.ndarray,
    domains: np.ndarray,
    coords: np.ndarray,
    scores: dict[str, np.ndarray],
    program_domains: dict[str, str],
    min_domain_fraction: float,
) -> None:
    rows = ((None, "STEP domains"),) + tuple((name, name) for name in scores)
    plot_coords = coords.astype(np.float64).copy()
    plot_coords[:, 1] *= -1
    figure = plt.figure(figsize=(17.5, 15.0), constrained_layout=True)
    grid = figure.add_gridspec(
        len(rows),
        len(liver_validation.SECTION_ORDER) + 1,
        width_ratios=(0.13, 1, 1, 1, 1, 1),
        wspace=0.015,
        hspace=0.018,
    )
    score_mappable = None
    score_axes = []
    for row_index, (program, label) in enumerate(rows):
        label_axis = figure.add_subplot(grid[row_index, 0])
        label_axis.text(
            0.5,
            0.5,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
        label_axis.axis("off")
        for column, section in enumerate(liver_validation.SECTION_ORDER, start=1):
            axis = figure.add_subplot(grid[row_index, column])
            section_mask = sections == section
            full_section_domains = domains[section_mask]
            displayed_domains = {
                domain
                for domain in DOMAIN_ORDER
                if np.mean(full_section_domains == domain) >= min_domain_fraction
            }
            keep = liver_validation.connected_liver_display_mask(
                coords[section_mask], edge_radius=1.51, min_component_size=20
            )
            section_coords = plot_coords[section_mask][keep]
            section_domains = domains[section_mask][keep]
            if program is None:
                axis.scatter(
                    section_coords[:, 0],
                    section_coords[:, 1],
                    c="#E7E7E7",
                    s=0.75,
                    linewidths=0,
                    alpha=1.0,
                    rasterized=True,
                )
                for domain in DOMAIN_ORDER:
                    if domain not in displayed_domains:
                        continue
                    selected = section_domains == domain
                    axis.scatter(
                        section_coords[selected, 0],
                        section_coords[selected, 1],
                        c=DOMAIN_COLORS[domain],
                        s=0.9,
                        linewidths=0,
                        alpha=1.0,
                        rasterized=True,
                    )
            else:
                values = scores[program][section_mask][keep]
                percentiles = rankdata(values, method="average") / len(values) * 100.0
                score_mappable = axis.scatter(
                    section_coords[:, 0],
                    section_coords[:, 1],
                    c=percentiles,
                    cmap="magma",
                    vmin=0.0,
                    vmax=100.0,
                    s=0.9,
                    linewidths=0,
                    alpha=1.0,
                    rasterized=True,
                )
                score_axes.append(axis)
                triangulation = _masked_triangulation(section_coords)
                if triangulation is not None and program in program_domains:
                    domain = program_domains[program]
                    indicator = (section_domains == domain).astype(float)
                    if (
                        domain in displayed_domains
                        and indicator.min() != indicator.max()
                    ):
                        axis.tricontour(
                            triangulation,
                            indicator,
                            levels=[0.5],
                            colors=[DOMAIN_COLORS[domain]],
                            linewidths=1.25,
                            alpha=1.0,
                        )
            axis.set_aspect("equal")
            axis.axis("off")
            if row_index == 0:
                axis.set_title(
                    section.removeprefix("E16.5_").removesuffix(".MOSTA"),
                    fontsize=11,
                )
    handles = [
        Line2D(
            [0],
            [0],
            color=DOMAIN_COLORS[domain],
            linewidth=2.0,
            label=f"Domain {domain}",
        )
        for domain in DOMAIN_ORDER
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    if score_mappable is not None:
        colorbar = figure.colorbar(
            score_mappable,
            ax=score_axes,
            location="right",
            fraction=0.012,
            pad=0.008,
        )
        colorbar.set_label("Within-section marker-score percentile")
        colorbar.set_ticks([0, 25, 50, 75, 100])
    save_raster_pdf(figure, output_path)


def plot_marker_effects(output_path: Path, associations: pd.DataFrame) -> None:
    tested = associations.loc[associations["tested"]].copy()
    domain_fractions = (
        associations[["section", "domain", "domain_fraction"]]
        .drop_duplicates()
        .set_index(["domain", "section"])["domain_fraction"]
    )
    figure, axes = plt.subplots(
        1, len(DOMAIN_ORDER), figsize=(15.4, 4.6), sharey=True, constrained_layout=True
    )
    tested["log2_observed_expected"] = np.log2(
        tested["observed_expected_enrichment"].clip(lower=1e-6)
    )
    limit = max(float(tested["log2_observed_expected"].abs().max()), 0.25)
    image = None
    for axis, domain in zip(axes, DOMAIN_ORDER, strict=True):
        values = np.full((len(LITERATURE_PROGRAMS), len(liver_validation.SECTION_ORDER)), np.nan)
        q_values = np.full_like(values, np.nan)
        selected = tested.loc[tested["domain"] == domain].set_index(
            ["program", "section"]
        )
        for row, program in enumerate(LITERATURE_PROGRAMS):
            for column, section in enumerate(liver_validation.SECTION_ORDER):
                key = (program, section)
                if key not in selected.index:
                    continue
                values[row, column] = selected.loc[key, "log2_observed_expected"]
                q_values[row, column] = selected.loc[key, "fisher_fdr_bh"]
        image = axis.imshow(
            values,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        axis.set_title(f"Domain {domain}", color=DOMAIN_COLORS[domain], fontweight="bold")
        axis.set_xticks(
            np.arange(len(liver_validation.SECTION_ORDER)),
            [
                (
                    f"{section.removeprefix('E16.5_').removesuffix('.MOSTA')}\n"
                    f"{100.0 * domain_fractions.loc[(domain, section)]:.1f}%"
                )
                for section in liver_validation.SECTION_ORDER
            ],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(np.arange(len(LITERATURE_PROGRAMS)), list(LITERATURE_PROGRAMS))
        axis.tick_params(length=0, labelsize=9)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if not np.isfinite(value):
                    label = "NA"
                else:
                    fold = 2.0**value
                    label = f"{fold:.2f}x{'*' if q_values[row, column] < 0.05 else ''}"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if np.isfinite(value) and abs(value) > 0.55 * limit else "#202020",
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
    if image is None:
        raise AssertionError("no marker effects were plotted")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("log2(observed / expected overlap)")
    save_raster_pdf(figure, output_path)


def plot_domain_composition(output_path: Path, composition: pd.DataFrame) -> None:
    indexed = composition.set_index(["section", "domain"])["domain_fraction"]
    sections = list(liver_validation.SECTION_ORDER)
    values = {
        domain: np.asarray(
            [indexed.get((section, domain), 0.0) for section in sections]
        )
        for domain in DOMAIN_ORDER
    }
    assigned = np.sum(np.vstack(list(values.values())), axis=0)
    values["Other"] = np.maximum(0.0, 1.0 - assigned)

    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(len(sections))
    bottom = np.zeros(len(sections), dtype=float)
    colors = {**DOMAIN_COLORS, "Other": "#D9D9D9"}
    for domain in (*DOMAIN_ORDER, "Other"):
        axis.bar(
            x,
            100.0 * values[domain],
            bottom=100.0 * bottom,
            color=colors[domain],
            width=0.72,
            linewidth=0,
            label=f"Domain {domain}" if domain != "Other" else domain,
        )
        bottom += values[domain]
    axis.set_xticks(
        x,
        [
            section.removeprefix("E16.5_").removesuffix(".MOSTA")
            for section in sections
        ],
    )
    axis.set_ylim(0, 100)
    axis.set_ylabel("Liver spots (%)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    save_raster_pdf(figure, output_path)


def plot_marker_enrichment_summary(
    output_path: Path, summary: pd.DataFrame
) -> None:
    indexed = summary.set_index(["program", "domain"])
    values = np.full((len(LITERATURE_PROGRAMS), len(DOMAIN_ORDER)), np.nan)
    q_values = np.full_like(values, np.nan)
    labels = np.empty(values.shape, dtype=object)
    for row, program in enumerate(LITERATURE_PROGRAMS):
        for column, domain in enumerate(DOMAIN_ORDER):
            result = indexed.loc[(program, domain)]
            odds_ratio = float(result["mantel_haenszel_common_odds_ratio"])
            values[row, column] = np.log2(max(odds_ratio, 1e-6))
            q_values[row, column] = float(result["mantel_haenszel_fdr_bh"])
            labels[row, column] = (
                f"{odds_ratio:.2f}"
                f"{'*' if q_values[row, column] < 0.05 else ''}"
            )
    limit = max(float(np.nanmax(np.abs(values))), 0.25)
    figure, axis = plt.subplots(figsize=(6.7, 4.4), constrained_layout=True)
    image = axis.imshow(
        values,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(len(DOMAIN_ORDER)),
        [f"Domain {domain}" for domain in DOMAIN_ORDER],
    )
    axis.set_yticks(np.arange(len(LITERATURE_PROGRAMS)), list(LITERATURE_PROGRAMS))
    axis.tick_params(length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                labels[row, column],
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(values[row, column]) > 0.55 * limit else "#202020",
            )
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.025)
    colorbar.set_label("log2 common odds ratio")
    save_raster_pdf(figure, output_path)


def run_section_de(
    expression: ad.AnnData,
    min_spots: int,
) -> pd.DataFrame:
    rows = []
    sections = expression.obs["section"].astype(str)
    for section in liver_validation.SECTION_ORDER:
        section_data = expression[sections == section].copy()
        sc.pp.filter_genes(section_data, min_cells=3)
        matrix = section_data.X.tocsr()
        means = np.asarray(matrix.mean(axis=0)).ravel()
        squared = matrix.copy().astype(np.float64)
        squared.data **= 2
        variances = np.asarray(squared.mean(axis=0)).ravel() - means**2
        section_data = section_data[:, variances > 1e-12].copy()
        section_data.obs["domain"] = section_data.obs["domain"].astype(str).astype("category")
        counts = section_data.obs["domain"].value_counts()
        groups = [
            domain
            for domain in DOMAIN_ORDER
            if counts.get(domain, 0) >= min_spots
            and section_data.n_obs - counts.get(domain, 0) >= min_spots
        ]
        if not groups:
            continue
        sc.tl.rank_genes_groups(
            section_data,
            groupby="domain",
            groups=groups,
            reference="rest",
            method="wilcoxon",
            corr_method="benjamini-hochberg",
            tie_correct=True,
            rankby_abs=False,
            pts=True,
            n_genes=section_data.n_vars,
            use_raw=False,
        )
        for domain in groups:
            frame = sc.get.rank_genes_groups_df(section_data, group=domain)
            frame = frame.rename(
                columns={
                    "names": "gene",
                    "logfoldchanges": "log2_fold_change",
                    "pvals": "p_value",
                    "pvals_adj": "fdr_bh",
                    "pct_nz_group": "fraction_expressing_domain",
                    "pct_nz_reference": "fraction_expressing_rest",
                }
            )
            frame.insert(0, "domain", domain)
            frame.insert(0, "section", section)
            rows.append(frame)
        del section_data
    return pd.concat(rows, ignore_index=True)


def summarize_de(de: pd.DataFrame) -> pd.DataFrame:
    frame = de.copy()
    frame["positive"] = frame["scores"] > 0
    frame["significant_positive"] = frame["positive"] & (frame["fdr_bh"] < 0.05)
    available_sections = frame.groupby("domain")["section"].nunique()
    summary = (
        frame.groupby(["domain", "gene"], as_index=False)
        .agg(
            tested_sections=("section", "nunique"),
            positive_sections=("positive", "sum"),
            significant_positive_sections=("significant_positive", "sum"),
            median_wilcoxon_score=("scores", "median"),
            median_log2_fold_change=("log2_fold_change", "median"),
            best_fdr_bh=("fdr_bh", "min"),
        )
    )
    summary["available_sections"] = summary["domain"].map(available_sections)
    summary["repeat_supported"] = (
        (summary["tested_sections"] == summary["available_sections"])
        & (summary["positive_sections"] == summary["available_sections"])
        & (
            summary["significant_positive_sections"]
            >= np.ceil(summary["available_sections"] / 2.0)
        )
        & (summary["median_log2_fold_change"] > 0.25)
    )
    return summary.sort_values(
        ["domain", "repeat_supported", "significant_positive_sections", "median_wilcoxon_score"],
        ascending=[True, False, False, False],
    )


def build_deterministic_prerank(contrast: pd.DataFrame) -> pd.DataFrame:
    ranking = (
        contrast[["gene", "scores", "log2_fold_change"]]
        .dropna(subset=["gene", "scores"])
        .drop_duplicates("gene")
        .sort_values(
            ["scores", "log2_fold_change", "gene"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    metric = ranking["scores"].to_numpy(dtype=np.float64, copy=True)
    starts = np.r_[0, np.flatnonzero(metric[1:] != metric[:-1]) + 1]
    ends = np.r_[starts[1:], len(metric)]
    for start, end in zip(starts, ends, strict=True):
        if end - start < 2:
            continue
        value = metric[start]
        for index in range(end - 2, start - 1, -1):
            metric[index] = np.nextafter(metric[index + 1], np.inf)
        if start > 0 and metric[start] >= metric[start - 1]:
            gap = ranking.loc[start - 1, "scores"] - value
            metric[start:end] = value + np.linspace(
                0.25 * gap,
                0.0,
                end - start,
                endpoint=True,
            )
    if not np.all(metric[:-1] > metric[1:]):
        raise AssertionError("deterministic GSEA ranking is not strictly ordered")
    return pd.DataFrame({"gene": ranking["gene"], "metric": metric})


def save_go_library(output_dir: Path) -> dict[str, list[str]]:
    library = gp.get_library(name=GO_LIBRARY, organism="Mouse")
    path = output_dir / f"{GO_LIBRARY}.Mouse.gmt"
    with path.open("w", encoding="utf-8") as handle:
        for term, genes in library.items():
            handle.write("\t".join((term, "Enrichr", *genes)) + "\n")
    return library


def load_go_library(path: Path) -> dict[str, list[str]]:
    library = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                library[fields[0]] = fields[2:]
    if not library:
        raise ValueError(f"no gene sets found in {path}")
    return library


def run_gsea(
    de: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    permutations: int,
    threads: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for contrast_index, ((section, domain), contrast) in enumerate(
        de.groupby(["section", "domain"], sort=False)
    ):
        ranking = build_deterministic_prerank(contrast)
        result = gp.prerank(
            rnk=ranking,
            gene_sets=gene_sets,
            organism="Mouse",
            min_size=10,
            max_size=500,
            permutation_num=permutations,
            threads=threads,
            no_plot=True,
            seed=seed + contrast_index,
            verbose=False,
        ).res2d.copy()
        result.insert(0, "domain", domain)
        result.insert(0, "section", section)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def summarize_gsea(gsea: pd.DataFrame) -> pd.DataFrame:
    renamed = gsea.rename(
        columns={
            "Term": "term",
            "ES": "enrichment_score",
            "NES": "normalized_enrichment_score",
            "NOM p-val": "nominal_p_value",
            "FDR q-val": "fdr_q_value",
            "Lead_genes": "leading_edge_genes",
        }
    ).copy()
    for column in (
        "enrichment_score",
        "normalized_enrichment_score",
        "nominal_p_value",
        "fdr_q_value",
    ):
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed["positive"] = renamed["normalized_enrichment_score"] > 0
    renamed["significant_positive"] = renamed["positive"] & (renamed["fdr_q_value"] < 0.05)
    available_sections = renamed.groupby("domain")["section"].nunique()
    summary = (
        renamed.groupby(["domain", "term"], as_index=False)
        .agg(
            tested_sections=("section", "nunique"),
            positive_sections=("positive", "sum"),
            significant_positive_sections=("significant_positive", "sum"),
            median_nes=("normalized_enrichment_score", "median"),
            median_fdr_q_value=("fdr_q_value", "median"),
            best_fdr_q_value=("fdr_q_value", "min"),
        )
    )
    summary["available_sections"] = summary["domain"].map(available_sections)
    summary["recurrent_positive"] = (
        (summary["tested_sections"] == summary["available_sections"])
        & (summary["positive_sections"] == summary["available_sections"])
        & (
            summary["significant_positive_sections"]
            >= np.ceil(summary["available_sections"] / 2.0)
        )
    )
    return renamed, summary.sort_values(
        ["domain", "recurrent_positive", "significant_positive_sections", "median_nes"],
        ascending=[True, False, False, False],
    )


def run_go_overrepresentation(
    de_summary: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    background: list[str],
) -> pd.DataFrame:
    rows = []
    for domain in DOMAIN_ORDER:
        genes = de_summary.loc[
            (de_summary["domain"] == domain) & de_summary["repeat_supported"], "gene"
        ].tolist()
        if not genes:
            continue
        result = gp.enrich(
            gene_list=genes,
            gene_sets=gene_sets,
            background=background,
            cutoff=1.0,
            no_plot=True,
            verbose=False,
        ).results.copy()
        result.insert(0, "n_repeat_supported_genes", len(genes))
        result.insert(0, "domain", domain)
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_de_heatmap(output_path: Path, de: pd.DataFrame, summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(
        1, len(DOMAIN_ORDER), figsize=(14.8, 6.2), constrained_layout=True
    )
    image = None
    limit = 0.0
    selected_genes = {}
    for domain in DOMAIN_ORDER:
        candidates = summary.loc[
            (summary["domain"] == domain)
            & summary["repeat_supported"]
            & ~summary["gene"].astype(str).str.contains(
                UNINFORMATIVE_GENE_NAME,
                regex=True,
            )
        ].copy()
        candidates = candidates.sort_values(
            ["significant_positive_sections", "median_wilcoxon_score"],
            ascending=[False, False],
        )
        selected_genes[domain] = candidates.head(10)["gene"].tolist()
        values = de.loc[
            (de["domain"] == domain) & de["gene"].isin(selected_genes[domain]),
            "log2_fold_change",
        ].to_numpy()
        if len(values):
            limit = max(limit, float(np.nanquantile(np.abs(values), 0.98)))
    limit = max(limit, 0.5)
    for axis, domain in zip(axes, DOMAIN_ORDER, strict=True):
        genes = selected_genes[domain]
        values = np.full((len(genes), len(liver_validation.SECTION_ORDER)), np.nan)
        q_values = np.full_like(values, np.nan)
        indexed = de.loc[
            (de["domain"] == domain) & de["gene"].isin(genes)
        ].set_index(["gene", "section"])
        for row, gene in enumerate(genes):
            for column, section in enumerate(liver_validation.SECTION_ORDER):
                key = (gene, section)
                if key not in indexed.index:
                    continue
                values[row, column] = indexed.loc[key, "log2_fold_change"]
                q_values[row, column] = indexed.loc[key, "fdr_bh"]
        image = axis.imshow(
            values,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        axis.set_title(f"Domain {domain}", color=DOMAIN_COLORS[domain], fontweight="bold")
        axis.set_xticks(
            np.arange(len(liver_validation.SECTION_ORDER)),
            [s.removeprefix("E16.5_").removesuffix(".MOSTA") for s in liver_validation.SECTION_ORDER],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(np.arange(len(genes)), genes)
        axis.tick_params(length=0, labelsize=9)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                if np.isfinite(values[row, column]) and q_values[row, column] < 0.05:
                    axis.text(column, row, "*", ha="center", va="center", fontsize=9)
        for spine in axis.spines.values():
            spine.set_visible(False)
    if image is None:
        raise AssertionError("no DEG heatmap was plotted")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("Domain-vs-rest log2 fold change")
    save_raster_pdf(figure, output_path)


def _clean_go_term(term: str) -> str:
    return re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term))


def plot_gsea_summary(output_path: Path, summary: pd.DataFrame) -> None:
    plotted_domains = [
        domain
        for domain in DOMAIN_ORDER
        if summary.loc[
            (summary["domain"] == domain) & summary["recurrent_positive"]
        ].shape[0]
    ]
    if not plotted_domains:
        return
    figure, axes = plt.subplots(
        1,
        len(plotted_domains),
        figsize=(7.2 * len(plotted_domains), 6.5),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, domain in zip(axes.ravel(), plotted_domains, strict=True):
        selected = summary.loc[
            (summary["domain"] == domain) & summary["recurrent_positive"]
        ].copy()
        selected = selected.sort_values(
            ["recurrent_positive", "significant_positive_sections", "median_nes"],
            ascending=[False, False, False],
        ).head(8)
        selected = selected.sort_values("median_nes")
        y = np.arange(len(selected))
        sizes = 35.0 + 30.0 * selected["significant_positive_sections"].to_numpy()
        colors = -np.log10(
            np.maximum(selected["median_fdr_q_value"].to_numpy(), 1e-12)
        )
        scatter = axis.scatter(
            selected["median_nes"],
            y,
            s=sizes,
            c=colors,
            cmap="YlGnBu",
            edgecolor="#303030",
            linewidth=0.4,
        )
        axis.set_yticks(y, [_clean_go_term(term) for term in selected["term"]])
        axis.set_title(f"Domain {domain}", color=DOMAIN_COLORS[domain], fontweight="bold")
        axis.set_xlabel("Median NES across sections")
        axis.grid(axis="x", color="#E3E3E3", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, labelsize=8.5)
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.04, pad=0.02)
        colorbar.set_label("Median -log10(FDR)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    save_raster_pdf(figure, output_path)


def write_manifest(args: argparse.Namespace, expression: ad.AnnData) -> None:
    manifest = {
        "source_model_output": str(args.source),
        "raw_count_directory": str(args.raw_dir),
        "sections": list(liver_validation.SECTION_ORDER),
        "n_liver_spots": int(expression.n_obs),
        "n_full_transcriptome_genes": int(expression.n_vars),
        "expression_preprocessing": "Per-section raw layers[count] -> CP10K -> log1p",
        "de_method": "scanpy.tl.rank_genes_groups(method='wilcoxon', tie_correct=True), domain versus remaining Liver spots within each section",
        "multiple_testing": "Benjamini-Hochberg within each section-domain genome-wide contrast",
        "marker_scoring": "scanpy.tl.score_genes on literature-defined genes; marker genes were not selected from STEP domains",
        "marker_enrichment_definition": (
            "Within each section, marker-high spots are the top requested score "
            "fraction. Expected overlap equals n_marker_high multiplied by the "
            "observed domain spot proportion in that section."
        ),
        "marker_high_fraction": args.marker_high_fraction,
        "minimum_display_domain_fraction": args.min_display_domain_fraction,
        "cross_section_marker_summary": (
            "Mantel-Haenszel common odds ratio over section-specific 2x2 tables; "
            "section counts are reported only as directional replication."
        ),
        "gsea_method": (
            "gseapy.prerank using per-section signed Wilcoxon scores; exact "
            "score ties are ordered by log2 fold change and gene name with a "
            "strictly order-preserving floating-point perturbation"
        ),
        "repeat_supported_deg_definition": (
            "Tested and positive in every section where the domain is present, "
            "BH-FDR below 0.05 in at least half of those sections, and median "
            "log2 fold change above 0.25"
        ),
        "recurrent_positive_gsea_definition": (
            "Positive NES in every section where the domain is present and "
            "GSEA FDR below 0.05 in at least half of those sections"
        ),
        "go_library": GO_LIBRARY,
        "gsea_permutations": args.gsea_permutations,
        "source_records": list(SOURCE_RECORDS),
    }
    with (args.output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(args.source, backed="r")
    annotations = source.obs["annotation"].astype(str).to_numpy()
    domains = source.obs["domain"].astype(str).to_numpy()
    sections = source.obs["batch"].astype(str).to_numpy()
    coords = np.asarray(source.obsm["spatial"], dtype=np.float64)
    liver_indices = np.flatnonzero(annotations == "Liver")
    expression = liver_validation.load_raw_liver_expression(
        args.raw_dir,
        source.obs_names,
        sections,
        annotations,
        domains,
        liver_indices,
    )
    source.file.close()

    pd.DataFrame(SOURCE_RECORDS).to_csv(
        args.output_dir / "reference_sources.csv", index=False
    )
    marker_provenance().to_csv(
        args.output_dir / "literature_marker_provenance.csv", index=False
    )
    write_manifest(args, expression)

    scores = score_literature_programs(expression, args.seed)
    marker_stats = marker_associations(
        expression,
        scores,
        args.min_spots,
        args.marker_high_fraction,
    )
    marker_stats.to_csv(
        args.output_dir / "literature_marker_domain_associations.csv", index=False
    )
    composition = marker_stats[
        ["section", "domain", "n_domain", "n_background", "domain_fraction"]
    ].drop_duplicates()
    composition["n_liver_spots"] = (
        composition["n_domain"] + composition["n_background"]
    )
    composition["domain_percent"] = 100.0 * composition["domain_fraction"]
    composition.to_csv(
        args.output_dir / "section_domain_composition.csv", index=False
    )
    plot_domain_composition(
        args.output_dir / "figure2a_domain_composition", composition
    )
    marker_summary = summarize_marker_enrichment(marker_stats)
    marker_summary.to_csv(
        args.output_dir / "literature_marker_enrichment_summary.csv", index=False
    )
    positive_marker_summary = marker_summary.loc[
        (marker_summary["mantel_haenszel_common_odds_ratio"] > 1.0)
        & (marker_summary["mantel_haenszel_fdr_bh"] < 0.05)
    ]
    program_domains = (
        positive_marker_summary.sort_values(
            ["program", "mantel_haenszel_common_odds_ratio"],
            ascending=[True, False],
        )
        .groupby("program", sort=False)
        .first()["domain"]
        .astype(str)
        .to_dict()
    )
    plot_spatial_evidence(
        args.output_dir / "figure1_literature_marker_spatial_evidence",
        sections[liver_indices],
        domains[liver_indices],
        coords[liver_indices],
        scores,
        program_domains,
        args.min_display_domain_fraction,
    )
    plot_marker_effects(
        args.output_dir / "figure2_literature_marker_effects", marker_stats
    )
    plot_marker_enrichment_summary(
        args.output_dir / "figure2b_literature_marker_common_odds_ratio",
        marker_summary,
    )

    de = run_section_de(expression, args.min_spots)
    de.to_csv(args.output_dir / "section_domain_wilcoxon_deg.csv.gz", index=False)
    de_summary = summarize_de(de)
    de_summary.to_csv(args.output_dir / "cross_section_deg_summary.csv", index=False)
    plot_de_heatmap(args.output_dir / "figure3_repeat_supported_deg", de, de_summary)

    gene_set_path = args.output_dir / f"{GO_LIBRARY}.Mouse.gmt"
    if args.skip_gsea:
        gene_sets = load_go_library(gene_set_path)
        saved_gsea = pd.read_csv(args.output_dir / "section_domain_gsea.csv.gz")
        saved_gsea["domain"] = saved_gsea["domain"].astype(str)
        _, gsea_summary = summarize_gsea(saved_gsea)
        gsea_summary.to_csv(
            args.output_dir / "cross_section_gsea_summary.csv", index=False
        )
    else:
        gene_sets = save_go_library(args.output_dir)
        gsea = run_gsea(
            de,
            gene_sets,
            args.gsea_permutations,
            args.threads,
            args.seed,
        )
        gsea_all, gsea_summary = summarize_gsea(gsea)
        gsea_all.to_csv(args.output_dir / "section_domain_gsea.csv.gz", index=False)
        gsea_summary.to_csv(args.output_dir / "cross_section_gsea_summary.csv", index=False)
    ora = run_go_overrepresentation(
        de_summary,
        gene_sets,
        expression.var_names.astype(str).tolist(),
    )
    ora.to_csv(args.output_dir / "repeat_supported_deg_go_overrepresentation.csv", index=False)
    plot_gsea_summary(args.output_dir / "figure4_recurrent_gsea", gsea_summary)


if __name__ == "__main__":
    main()
