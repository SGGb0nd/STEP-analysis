#!/usr/bin/env python
"""Write a publication-style report for the MOSTA liver evidence chain."""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_EVIDENCE_DIR = Path("workflows/mosta_liver_validation/biological_evidence")
DEFAULT_TRANSFER_DIR = Path("workflows/mosta_liver_validation/cross_section_transfer")
DOMAIN_ORDER = ("2", "15", "26")
SECTION_ORDER = (
    "E16.5_E2S1.MOSTA",
    "E16.5_E2S4.MOSTA",
    "E16.5_E2S7.MOSTA",
    "E16.5_E2S10.MOSTA",
    "E16.5_E2S13.MOSTA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    return parser.parse_args()


def format_probability(value: float) -> str:
    if value == 0:
        return "< 1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def short_section(section: str) -> str:
    return section.removeprefix("E16.5_").removesuffix(".MOSTA")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def composition_table(composition: pd.DataFrame) -> pd.DataFrame:
    table = composition.pivot(
        index="section", columns="domain", values="domain_percent"
    ).reindex(
        index=SECTION_ORDER,
        columns=[int(domain) for domain in DOMAIN_ORDER],
    )
    table.index = [short_section(section) for section in table.index]
    table = table.rename(
        columns={
            int(domain): f"Domain {domain} (%)" for domain in DOMAIN_ORDER
        }
    )
    table.insert(0, "Section", table.index)
    for column in table.columns[1:]:
        table[column] = table[column].map(lambda value: f"{value:.2f}")
    return table.reset_index(drop=True)


def marker_result_table(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary.loc[
        (summary["mantel_haenszel_common_odds_ratio"] > 1.0)
        & (summary["mantel_haenszel_fdr_bh"] < 0.05)
    ].copy()
    selected = selected.sort_values(
        ["domain", "mantel_haenszel_common_odds_ratio"],
        ascending=[True, False],
    )
    selected["Domain"] = "Domain " + selected["domain"].astype(str)
    selected["Program"] = selected["program"]
    selected["Observed / expected"] = selected[
        "observed_expected_enrichment"
    ].map(lambda value: f"{value:.2f}")
    selected["Common OR (95% CI)"] = selected.apply(
        lambda row: (
            f"{row['mantel_haenszel_common_odds_ratio']:.2f} "
            f"({row['mantel_haenszel_odds_ratio_ci95_low']:.2f}-"
            f"{row['mantel_haenszel_odds_ratio_ci95_high']:.2f})"
        ),
        axis=1,
    )
    selected["BH-FDR"] = selected["mantel_haenszel_fdr_bh"].map(
        format_probability
    )
    selected["Sections above expectation"] = selected.apply(
        lambda row: (
            f"{int(row['sections_enrichment_above_one'])}/"
            f"{int(row['tested_sections'])}"
        ),
        axis=1,
    )
    return selected[
        [
            "Domain",
            "Program",
            "Observed / expected",
            "Common OR (95% CI)",
            "BH-FDR",
            "Sections above expectation",
        ]
    ]


def gsea_result_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(path)
    rows = []
    for domain in DOMAIN_ORDER:
        selected = summary.loc[
            (summary["domain"].astype(str) == domain)
            & summary["recurrent_positive"]
        ].copy()
        selected = selected.sort_values(
            ["recurrent_positive", "significant_positive_sections", "median_nes"],
            ascending=[False, False, False],
        ).head(6)
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "Domain": f"Domain {domain}",
                    "GO biological process": str(row.term).rsplit(" (GO:", 1)[0],
                    "Median NES": f"{row.median_nes:.2f}",
                    "Significant sections": (
                        f"{int(row.significant_positive_sections)}/"
                        f"{int(row.tested_sections)}"
                    ),
                    "Median FDR": format_probability(
                        float(row.median_fdr_q_value)
                    ),
                }
            )
    return pd.DataFrame(rows)


def marker_provenance_table(provenance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for program, group in provenance.groupby("program", sort=False):
        rows.append(
            {
                "Program": program,
                "Genes": ", ".join(group["gene"].astype(str)),
                "Primary source": ", ".join(
                    dict.fromkeys(group["doi"].astype(str))
                ),
            }
        )
    return pd.DataFrame(rows)


def transfer_section_table(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.copy()
    table["Section"] = table["section"].map(short_section)
    table["Spots"] = table["n_spots"].map(lambda value: f"{int(value):,}")
    table["Balanced accuracy"] = table["balanced_accuracy"].map(
        lambda value: f"{value:.3f}"
    )
    table["Macro F1"] = table["macro_f1_present_domains"].map(
        lambda value: f"{value:.3f}"
    )
    return table[["Section", "Spots", "Balanced accuracy", "Macro F1"]]


def transfer_domain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    tested = metrics.loc[metrics["tested"]].copy()
    rows = []
    for domain in DOMAIN_ORDER:
        selected = tested.loc[tested["domain"].astype(str) == domain]
        rows.append(
            {
                "Domain": f"Domain {domain}",
                "Tested sections": str(len(selected)),
                "Median AUROC": f"{selected['auroc'].median():.3f}",
                "Median AUPRC": f"{selected['auprc'].median():.3f}",
                "Median AUPRC / prevalence": (
                    f"{selected['auprc_over_prevalence'].median():.2f}"
                ),
                "Median recall": f"{selected['recall'].median():.3f}",
            }
        )
    return pd.DataFrame(rows)


def weight_stability_table(summary: dict) -> pd.DataFrame:
    rows = []
    for domain in DOMAIN_ORDER:
        result = summary["per_domain"][domain]
        marker_genes = result["stable_domain_supported_marker_genes"]
        significant_marker_programs = [
            row["program"]
            for row in result["marker_program_tests"]
            if row["median_weight_rank_fdr_bh"] < 0.05
            or row["stable_gene_fisher_fdr_bh"] < 0.05
        ]
        rows.append(
            {
                "Domain": f"Domain {domain}",
                "Median fold-pair rho": (
                    f"{result['median_pairwise_weight_spearman_rho']:.3f}"
                ),
                "Pairwise rho range": (
                    f"{result['minimum_pairwise_weight_spearman_rho']:.3f}-"
                    f"{result['maximum_pairwise_weight_spearman_rho']:.3f}"
                ),
                "Stable top-weight genes": str(
                    result["stable_top_positive_weight_genes"]
                ),
                "Supported marker enrichment": (
                    ", ".join(significant_marker_programs)
                    if significant_marker_programs
                    else "None"
                ),
                "Stable supported marker genes": (
                    ", ".join(marker_genes) if marker_genes else "None"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    evidence_dir: Path,
    transfer_dir: Path,
) -> Path:
    composition = pd.read_csv(evidence_dir / "section_domain_composition.csv")
    marker_summary = pd.read_csv(
        evidence_dir / "literature_marker_enrichment_summary.csv"
    )
    marker_provenance = pd.read_csv(
        evidence_dir / "literature_marker_provenance.csv"
    )
    with (transfer_dir / "analysis_manifest.json").open(encoding="utf-8") as handle:
        transfer_manifest = json.load(handle)
    transfer_sections = pd.read_csv(transfer_dir / "loso_section_metrics.csv")
    transfer_domains = pd.read_csv(transfer_dir / "loso_per_domain_metrics.csv")
    mean_balanced_accuracy = transfer_manifest["summary"][
        "mean_section_balanced_accuracy"
    ]
    label_mapping_null = transfer_manifest["summary"]["label_mapping_null"]
    weight_summary = transfer_manifest["summary"]["classifier_weight_stability"]

    marker_table = marker_result_table(marker_summary)
    gsea_table = gsea_result_table(evidence_dir / "cross_section_gsea_summary.csv")
    lines = [
        "# Biological validation of STEP-resolved E16.5 liver domains",
        "",
        (
            "STEP resolved Domains 2, 15, and 26 within the manual Liver "
            "annotation of five MOSTA E16.5 sagittal sections. We evaluated "
            "whether these assignments carry reproducible transcriptomic "
            "signals across sections and whether their biological associations "
            "exceed the overlap expected from each domain's spot proportion."
        ),
        "",
        "## Domain composition across sagittal sections",
        "",
        markdown_table(composition_table(composition)),
        "",
        (
            "![Liver-domain composition across sections.]"
            "(./figure2a_domain_composition.png)"
        ),
        "",
        (
            "**Figure 1 | Section-specific composition of STEP domains within "
            "the manual Liver annotation.** Bars show the percentage of Liver "
            "spots assigned to Domains 2, 15, 26, or other STEP domains. The "
            "composition varies markedly among sagittal sections; E2S1 is "
            "nearly uniform, with 98.07% of Liver spots assigned to Domain 15."
        ),
        "",
        "## Literature-defined fetal-liver marker programs",
        "",
        (
            "Marker programs were fixed from primary fetal-liver studies before "
            "testing their association with STEP domains. Scores were computed "
            "with scanpy.tl.score_genes over the full measured gene set. Within "
            "each section, marker-high spots were defined as the top 15% of "
            "scores. The expected overlap with a domain was the number of "
            "marker-high spots multiplied by that domain's observed Liver-spot "
            "proportion."
        ),
        "",
        markdown_table(marker_provenance_table(marker_provenance)),
        "",
        "![Spatial marker evidence.](./figure1_literature_marker_spatial_evidence.png)",
        "",
        (
            "**Figure 2 | Spatial correspondence between STEP domains and "
            "literature-defined fetal-liver marker programs.** The first row "
            "shows STEP assignments within the manual Liver annotation. "
            "Subsequent rows show within-section percentiles of scanpy marker "
            "scores. Colored outlines visualize the domain-program pairs with "
            "a significant positive pooled association in Figure 4. Domain "
            "assignments occupying less than 5% of Liver spots in a section "
            "are left gray in the first row and are not outlined."
        ),
        "",
        "![Section-level marker enrichment.](./figure2_literature_marker_effects.png)",
        "",
        (
            "**Figure 3 | Section-level marker enrichment after accounting for "
            "domain spot proportions.** Each cell reports observed marker-high "
            "overlap divided by its composition-based expectation. The domain "
            "percentage is printed below each section. Asterisks denote "
            "two-sided Fisher tests passing BH-FDR across all tested "
            "section-domain-program combinations."
        ),
        "",
        (
            "![Pooled marker odds ratios.]"
            "(./figure2b_literature_marker_common_odds_ratio.png)"
        ),
        "",
        (
            "**Figure 4 | Cross-section marker associations.** Values are "
            "Mantel-Haenszel common odds ratios from section-specific 2 x 2 "
            "tables. Asterisks denote BH-FDR below 0.05 across the 18 pooled "
            "domain-program tests."
        ),
        "",
        markdown_table(marker_table),
        "",
        (
            "Domain 2 showed the dominant fetal-hepatocyte association (common "
            "OR 4.11). Domain 15 was enriched for primitive-erythroid and "
            "stellate/stromal signals, whereas Domain 26 was enriched for "
            "stellate/stromal and mesothelial signals. For E2S1 Domain 15, "
            "which comprised 98.07% of Liver spots, observed-to-expected "
            "marker overlaps ranged from 0.974 to 1.010 after accounting for "
            "that domain proportion."
        ),
        "",
        "## Full-transcriptome differential expression and gene-set enrichment",
        "",
        (
            "Within each section, each domain was compared with the remaining "
            "Liver spots using a tie-corrected Wilcoxon rank-sum test over all "
            "genes detected in at least three spots; no highly variable-gene "
            "restriction was applied. Counts were normalized separately in "
            "each section to 10,000 counts per spot and log1p transformed. "
            "Gene-level P values were corrected by Benjamini-Hochberg within "
            "each section-domain contrast."
        ),
        "",
        "![Repeated domain-associated genes.](./figure3_repeat_supported_deg.png)",
        "",
        (
            "**Figure 5 | Repeated domain-associated expression differences "
            "across sections.** Values are domain-versus-rest log2 fold "
            "changes from the full-transcriptome tests. Asterisks denote "
            "within-contrast BH-FDR below 0.05. Up to ten repeat-supported "
            "genes per domain are displayed, ordered by the number of "
            "significant sections and then the median Wilcoxon score."
        ),
    ]
    if not gsea_table.empty:
        lines.extend(
            [
                "",
                "![Recurrent GO enrichment.](./figure4_recurrent_gsea.png)",
                "",
                (
                    "**Figure 6 | Recurrent GO Biological Process enrichment.** "
                    "GSEA used signed Wilcoxon statistics with 1,000 gene-set "
                    "permutations per section-domain contrast. Exact Wilcoxon "
                    "ties were ordered deterministically by log2 fold change "
                    "and gene name without crossing adjacent Wilcoxon scores. "
                    "Terms qualify when enrichment is positive in every "
                    "available section and FDR is below 0.05 in at least half "
                    "of those sections. Up to eight qualifying terms per "
                    "domain are plotted, ordered by the number of significant "
                    "sections and then median NES; the table lists the first six."
                ),
                "",
                markdown_table(gsea_table),
                "",
                (
                    "Recurrent positive GO enrichment was detected for Domain 2, "
                    "including cellular respiration, electron transport, ATP "
                    "synthesis, heme metabolism, and lipid transport. Domains 15 "
                    "and 26 did not meet this recurrent GSEA criterion. Their "
                    "biological annotations are supported by the literature-defined "
                    "marker-program associations above; cross-section transcriptome "
                    "transfer is evaluated separately below."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Cross-section transcriptome consistency",
            "",
            (
                "We tested whether the numerical domain labels correspond to "
                "transcriptomic identities that transfer between sections. In "
                "each of five folds, one complete section was excluded from "
                "feature selection and classifier fitting. Stable "
                "domain-associated genes were selected using only the other "
                "four sections, and a section- and class-balanced multinomial "
                "logistic-regression classifier was then applied to all "
                "held-out Liver spots assigned by STEP to Domains 2, 15, or "
                "26. Coordinates, STEP "
                "embeddings, and labels from the held-out section were not used "
                "for feature selection or model fitting."
            ),
            "",
            markdown_table(transfer_section_table(transfer_sections)),
            "",
            markdown_table(transfer_domain_table(transfer_domains)),
            "",
            (
                "Across the 45,980 out-of-fold spots, mean section-level "
                f"balanced accuracy was {mean_balanced_accuracy:.3f}. "
                "Median one-versus-rest AUROCs were 0.835, 0.804, and 0.897 "
                "for Domains 2, 15, and 26, respectively. In all "
                f"{transfer_manifest['summary']['tested_section_domain_combinations']} "
                "section-domain combinations that were present, more than half "
                "of the assigned spots were classified as the same domain using "
                "only the other sections."
            ),
            "",
            (
                "As a label-alignment diagnostic, we enumerated all six "
                "one-to-one mappings between the domain identities present in "
                "each section and the three classifier outputs. The identity "
                "mapping was the unique best mapping in all five sections. Its "
                "mean balanced score was the unique maximum among all "
                f"{label_mapping_null['n_joint_label_mappings']:,} joint "
                "section-level mappings (exact P = "
                f"{label_mapping_null['exact_p_value']:.2e}). This null permutes "
                "domain identities at the section level; it does not shuffle "
                "spots or use spatial coordinates."
            ),
            "",
            (
                "![Held-out-section transcriptome-transfer metrics.]"
                "(../cross_section_transfer/"
                "figure1_loso_transfer_metrics.png)"
            ),
            "",
            (
                "**Figure 7 | Transcriptome-only transfer of STEP Liver-domain "
                "identities to held-out sections.** The confusion matrix "
                "summarizes within-domain classification fractions averaged "
                "equally across sections. Bars report balanced accuracy and "
                "macro F1 for each held-out section. Curves report one-versus-rest "
                "AUROC and AUPRC relative to each domain's prevalence. The test "
                "includes 45,980 Liver spots assigned by STEP to Domains 2, 15, "
                "or 26; 199 Liver spots assigned to other domains are outside "
                "this three-class evaluation."
            ),
            "",
            (
                "![Spatial distribution of held-out transcriptome-transfer "
                "predictions.](../cross_section_transfer/"
                "figure2_loso_spatial_transfer.png)"
            ),
            "",
            (
                "**Figure 8 | Spatial diagnostic of transcriptome-only "
                "held-out predictions.** The first two rows compare STEP "
                "assignments with labels predicted from the other four sections. "
                "The remaining rows show held-out class probabilities. Spatial "
                "coordinates are used only for this visualization and were not "
                "available to the classifier. For display, isolated connected "
                "components containing fewer than 20 Liver spots are omitted; "
                "all 45,980 spots in the three-class evaluation remain included "
                "in the reported metrics."
            ),
            "",
            "## Classifier-weight stability and interpretation",
            "",
            (
                "The held-out predictions above test whether domain identities "
                "transfer between sections. We separately examined whether the "
                "genes used by the five LOSO fits were stable and "
                "biologically interpretable. Coefficient vectors were aligned "
                "over the union of eligible model features; a gene absent from "
                "a fold-specific selected feature set was assigned weight zero. "
                "Across the ten "
                "fold pairs per domain, median Spearman correlations were "
                f"{weight_summary['per_domain']['2']['median_pairwise_weight_spearman_rho']:.3f}, "
                f"{weight_summary['per_domain']['15']['median_pairwise_weight_spearman_rho']:.3f}, "
                "and "
                f"{weight_summary['per_domain']['26']['median_pairwise_weight_spearman_rho']:.3f} "
                "for Domains 2, 15, and 26. Because each pair of fits shares "
                "three training sections, these values measure stability to "
                "leaving out one section and are not independent replications."
            ),
            "",
            markdown_table(weight_stability_table(weight_summary)),
            "",
            (
                "A stable top-weight gene was defined as having a positive "
                "top-100 coefficient in at least four of five models. This "
                "identified 59, 39, and 36 genes for Domains 2, 15, and 26. "
                "Against the complete eligible Liver-gene background, the "
                "Domain 2 coefficient ranking was enriched for the "
                "literature-defined fetal-hepatocyte program and its stable set "
                "contained Alb, "
                "Apoa1, Ahsg, Trf, Afp, Fgb, and Apoa2. The Domain 15 stable set "
                "contained the primitive-erythroid genes Hbb-y and Hba-x, while "
                "the full Domain 15 ranking was enriched for the stellate/stromal "
                "program. For Domain 26, neither prespecified marker program was "
                "significantly enriched in the coefficient ranking after BH "
                "correction, and neither overlapped the stable top-weight set. GO "
                "over-representation of stable weights was tested separately "
                "against all eligible Liver genes; after BH correction across all "
                "tested terms within each domain, Domain 26 was enriched for "
                "negative regulation of blood coagulation."
            ),
            "",
            (
                "![Classifier-weight stability and biological enrichment.]"
                "(../cross_section_transfer/"
                "figure3_loso_weight_stability.png)"
            ),
            "",
            (
                "**Figure 9 | Classifier-weight stability and interpretation "
                "across leave-one-section-out fits.** The upper-left "
                "heatmap reports fold-pair Spearman correlations over the union "
                "of eligible model features, assigning zero to genes absent from "
                "a fold-specific feature set. Bars count genes with a positive "
                "top-100 coefficient in at least four folds. Marker-program rank "
                "AUC quantifies whether literature-defined programs with pooled "
                "domain associations in Figures 2-4 occur above other eligible "
                "genes in the median "
                "coefficient ranking; asterisks denote BH-FDR below 0.05 across "
                "all domain-program tests. The lower heatmaps show standardized "
                "coefficients for the 12 largest stable weights in each domain. "
                "Gene-label asterisks mark literature-defined marker genes from "
                "programs associated with that domain. This panel interprets "
                "the transfer classifier and is not an additional biological "
                "validation dataset."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The evidence supports biologically structured, section-varying "
                "transcriptomic organization within the coarse Liver annotation. "
                "Domain 2 has a strong fetal-hepatocyte identity. Domain 15 is "
                "associated with primitive-erythroid and weaker stromal signals, "
                "whereas Domain 26 is associated with stromal and mesothelial "
                "signals. Literature-defined marker programs provide biological "
                "annotation, and full-transcriptome differential expression and "
                "GSEA provide broader expression and pathway context. The strict "
                "leave-one-section-out transcriptome transfer directly tests "
                "whether the same STEP domain labels are supported across "
                "sections. The weight analysis interprets the transfer models "
                "and shows partly stable gene rankings with the clearest "
                "literature-marker correspondence for Domain 2. The transfer is substantial "
                "but not perfect, with "
                "the largest residual mixing involving Domains 2 and 15."
            ),
            "",
            "## Primary sources",
            "",
            (
                "- Chen et al., original MOSTA atlas: "
                "<https://doi.org/10.1016/j.cell.2022.04.003>"
            ),
            (
                "- Wang et al., mouse fetal-liver lineage markers: "
                "<https://doi.org/10.1038/s41422-020-0378-6>"
            ),
            (
                "- Lu et al., spatial fetal-liver marker and DEG-GO precedent: "
                "<https://doi.org/10.1038/s41421-021-00266-1>"
            ),
            (
                "- Subramanian et al., GSEA: "
                "<https://doi.org/10.1073/pnas.0506580102>"
            ),
            (
                "- Gene Ontology Consortium: "
                "<https://doi.org/10.1093/genetics/iyad031>"
            ),
            (
                "- Kuleshov et al., Enrichr: "
                "<https://doi.org/10.1093/nar/gkw377>"
            ),
        ]
    )
    content = "\n".join(lines) + "\n"
    output = evidence_dir / "MOSTA_liver_biological_evidence.md"
    output.write_text(content, encoding="utf-8")

    return output


def main() -> None:
    args = parse_args()
    output = write_report(args.evidence_dir, args.transfer_dir)
    print(output)


if __name__ == "__main__":
    main()
