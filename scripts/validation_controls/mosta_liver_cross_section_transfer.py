#!/usr/bin/env python
"""Test whether MOSTA Liver domain identities transfer across held-out sections."""

import argparse
import itertools
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from scipy import sparse
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

import mosta_liver_evidence_chain as evidence
import mosta_liver_domain_validation as liver_validation


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DOMAIN_ORDER = ("2", "15", "26")
DOMAIN_COLORS = {"2": "#D9486E", "15": "#4C78E8", "26": "#13BFB5"}
DEFAULT_OUTPUT = Path("workflows/mosta_liver_validation/cross_section_transfer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=liver_validation.DEFAULT_SOURCE)
    parser.add_argument("--raw-dir", type=Path, default=liver_validation.DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features-per-domain", type=int, default=200)
    parser.add_argument("--min-stratum-spots", type=int, default=20)
    parser.add_argument("--min-detection-fraction", type=float, default=0.005)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def biological_gene_mask(gene_names: pd.Index) -> np.ndarray:
    names = gene_names.astype(str)
    return np.asarray(
        [
            not (
                name.lower().startswith("mt-")
                or name.startswith("Rpl")
                or name.startswith("Rps")
            )
            for name in names
        ],
        dtype=bool,
    )


def mean_profile(matrix: sparse.csr_matrix, mask: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[mask].mean(axis=0)).ravel().astype(np.float32)


def select_training_features(
    matrix: sparse.csr_matrix,
    sections: np.ndarray,
    labels: np.ndarray,
    genes: pd.Index,
    training_sections: tuple[str, ...],
    features_per_domain: int,
    min_stratum_spots: int,
    min_detection_fraction: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Select cross-section-consistent domain genes without target data."""
    training_mask = np.isin(sections, training_sections)
    minimum_detected = max(
        20, int(np.ceil(training_mask.sum() * min_detection_fraction))
    )
    detected = np.asarray(matrix[training_mask].getnnz(axis=0)).ravel()
    eligible_gene = (
        biological_gene_mask(genes) & (detected >= minimum_detected)
    )
    feature_rows = []
    selected_by_domain = {}

    for domain in DOMAIN_ORDER:
        effects = []
        contributing_sections = []
        for section in training_sections:
            section_mask = sections == section
            positive = section_mask & (labels == domain)
            negative = section_mask & (labels != domain)
            if (
                positive.sum() < min_stratum_spots
                or negative.sum() < min_stratum_spots
            ):
                continue
            effects.append(
                mean_profile(matrix, positive)
                - mean_profile(matrix, negative)
            )
            contributing_sections.append(section)
        if not effects:
            raise ValueError(f"no training contrast available for Domain {domain}")

        stacked = np.vstack(effects)
        median_effect = np.median(stacked, axis=0)
        positive_fraction = np.mean(stacked > 0, axis=0)
        minimum_positive_fraction = 0.5
        stable = (
            eligible_gene
            & (median_effect > 0)
            & (positive_fraction >= minimum_positive_fraction)
        )
        candidates = np.flatnonzero(stable)
        if len(candidates) < features_per_domain:
            candidates = np.flatnonzero(eligible_gene & (median_effect > 0))
        order = candidates[np.argsort(median_effect[candidates])[::-1]]
        selected = order[:features_per_domain]
        if len(selected) == 0:
            raise ValueError(f"no genes selected for Domain {domain}")
        selected_by_domain[domain] = selected
        for rank, gene_index in enumerate(selected, start=1):
            feature_rows.append(
                {
                    "domain": domain,
                    "rank": rank,
                    "gene": str(genes[gene_index]),
                    "gene_index": int(gene_index),
                    "median_training_effect": float(median_effect[gene_index]),
                    "positive_training_section_fraction": float(
                        positive_fraction[gene_index]
                    ),
                    "n_training_section_contrasts": len(contributing_sections),
                    "training_sections": ";".join(contributing_sections),
                }
            )

    selected_indices = np.asarray(
        sorted(
            set(
                np.concatenate(list(selected_by_domain.values())).tolist()
            )
        ),
        dtype=np.int64,
    )
    return selected_indices, pd.DataFrame(feature_rows)


def section_domain_weights(
    sections: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Give every domain and each available section within it equal weight."""
    weights = np.zeros(len(labels), dtype=np.float64)
    for domain in DOMAIN_ORDER:
        domain_mask = labels == domain
        domain_sections = np.unique(sections[domain_mask])
        domain_total_weight = 1.0 / len(DOMAIN_ORDER)
        stratum_total_weight = domain_total_weight / len(domain_sections)
        for section in domain_sections:
            stratum = domain_mask & (sections == section)
            weights[stratum] = stratum_total_weight / stratum.sum()
    weights *= len(weights) / weights.sum()
    return weights


def evaluate_fold(
    target_section: str,
    labels: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict, list[dict], list[dict]]:
    present_domains = tuple(
        domain for domain in DOMAIN_ORDER if np.any(labels == domain)
    )
    overall = {
        "section": target_section,
        "n_spots": int(len(labels)),
        "n_true_domains": len(present_domains),
        "true_domains": ";".join(present_domains),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(
            np.mean(
                [
                    np.mean(predicted[labels == domain] == domain)
                    for domain in present_domains
                ]
            )
        ),
        "macro_f1_present_domains": float(
            f1_score(labels, predicted, labels=list(present_domains), average="macro")
        ),
        "multiclass_log_loss": float(
            -np.mean(
                np.log(
                    np.clip(
                        probabilities[
                            np.arange(len(labels)),
                            [DOMAIN_ORDER.index(label) for label in labels],
                        ],
                        1e-15,
                        1.0,
                    )
                )
            )
        ),
    }

    class_rows = []
    for column, domain in enumerate(DOMAIN_ORDER):
        binary = labels == domain
        prevalence = float(binary.mean())
        tested = bool(binary.any() and (~binary).any())
        row = {
            "section": target_section,
            "domain": domain,
            "n_positive": int(binary.sum()),
            "n_negative": int((~binary).sum()),
            "prevalence": prevalence,
            "tested": tested,
            "auroc": np.nan,
            "auprc": np.nan,
            "auprc_over_prevalence": np.nan,
            "recall": np.nan,
        }
        if tested:
            auprc = average_precision_score(binary, probabilities[:, column])
            row.update(
                {
                    "auroc": float(
                        roc_auc_score(binary, probabilities[:, column])
                    ),
                    "auprc": float(auprc),
                    "auprc_over_prevalence": float(auprc / prevalence),
                    "recall": float(np.mean(predicted[binary] == domain)),
                }
            )
        class_rows.append(row)

    counts = confusion_matrix(labels, predicted, labels=list(DOMAIN_ORDER))
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=row_totals > 0,
    )
    confusion_rows = []
    for true_index, true_domain in enumerate(DOMAIN_ORDER):
        for predicted_index, predicted_domain in enumerate(DOMAIN_ORDER):
            confusion_rows.append(
                {
                    "section": target_section,
                    "true_domain": true_domain,
                    "predicted_domain": predicted_domain,
                    "count": int(counts[true_index, predicted_index]),
                    "fraction_within_true_domain": float(
                        normalized[true_index, predicted_index]
                    ),
                }
            )
    return overall, class_rows, confusion_rows


def exact_label_mapping_null(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Test whether same-name domain mapping is optimal across sections."""
    rows = []
    section_score_options = []
    identity_scores = []

    for section in liver_validation.SECTION_ORDER:
        frame = predictions.loc[predictions["section"] == section]
        present_domains = tuple(
            domain
            for domain in DOMAIN_ORDER
            if (frame["true_domain"] == domain).any()
        )
        section_rows = []
        for assigned_domains in itertools.permutations(
            DOMAIN_ORDER, len(present_domains)
        ):
            mapping = dict(zip(present_domains, assigned_domains, strict=True))
            recalls = [
                float(
                    (
                        frame.loc[
                            frame["true_domain"] == true_domain,
                            "predicted_domain",
                        ]
                        == mapping[true_domain]
                    ).mean()
                )
                for true_domain in present_domains
            ]
            score = float(np.mean(recalls))
            is_identity = all(
                mapping[domain] == domain for domain in present_domains
            )
            section_rows.append(
                {
                    "section": section,
                    "mapping": ";".join(
                        f"{domain}->{mapping[domain]}"
                        for domain in present_domains
                    ),
                    "balanced_mapping_score": score,
                    "identity_mapping": is_identity,
                }
            )
        section_frame = pd.DataFrame(section_rows)
        identity_score = float(
            section_frame.loc[
                section_frame["identity_mapping"], "balanced_mapping_score"
            ].iloc[0]
        )
        section_frame["identity_score"] = identity_score
        section_frame["identity_rank"] = 1 + int(
            (
                section_frame["balanced_mapping_score"]
                > identity_score + 1e-12
            ).sum()
        )
        section_frame["identity_is_unique_best"] = bool(
            np.isclose(
                section_frame["balanced_mapping_score"],
                identity_score,
                rtol=0.0,
                atol=1e-12,
            ).sum()
            == 1
        )
        rows.append(section_frame)
        section_score_options.append(
            section_frame["balanced_mapping_score"].to_numpy()
        )
        identity_scores.append(identity_score)

    joint_null = np.asarray(
        [
            np.mean(scores)
            for scores in itertools.product(*section_score_options)
        ],
        dtype=np.float64,
    )
    observed = float(np.mean(identity_scores))
    at_least_observed = int((joint_null >= observed - 1e-12).sum())
    summary = {
        "observed_mean_balanced_mapping_score": observed,
        "null_mean_balanced_mapping_score": float(joint_null.mean()),
        "n_joint_label_mappings": int(len(joint_null)),
        "joint_mappings_at_least_observed": at_least_observed,
        "exact_p_value": float(at_least_observed / len(joint_null)),
        "identity_mapping_best_in_sections": int(
            sum(frame["identity_rank"].iloc[0] == 1 for frame in rows)
        ),
        "identity_mapping_unique_best_in_sections": int(
            sum(frame["identity_is_unique_best"].iloc[0] for frame in rows)
        ),
        "n_sections": len(rows),
    }
    return pd.concat(rows, ignore_index=True), summary


def classifier_coefficient_rows(
    target_section: str,
    selected_features: np.ndarray,
    genes: pd.Index,
    classifier: LogisticRegression,
    scaler: StandardScaler,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    selected_for = (
        feature_frame.groupby("gene", sort=False)["domain"]
        .agg(lambda values: ";".join(dict.fromkeys(values.astype(str))))
        .to_dict()
    )
    rows = []
    for domain in DOMAIN_ORDER:
        class_index = int(np.flatnonzero(classifier.classes_ == domain)[0])
        coefficients = classifier.coef_[class_index].astype(np.float64)
        for feature_position, gene_index in enumerate(selected_features):
            gene = str(genes[gene_index])
            rows.append(
                {
                    "held_out_section": target_section,
                    "domain": domain,
                    "gene": gene,
                    "gene_index": int(gene_index),
                    "selected_for_domains": selected_for.get(gene, ""),
                    "standardized_weight": float(coefficients[feature_position]),
                    "weight_per_log_normalized_unit": float(
                        coefficients[feature_position]
                        / max(float(scaler.scale_[feature_position]), 1e-12)
                    ),
                }
            )
    return pd.DataFrame(rows)


def annotate_literature_programs(genes: pd.Series) -> pd.Series:
    program_by_gene = {}
    for program, members in evidence.LITERATURE_PROGRAMS.items():
        for gene in members:
            program_by_gene.setdefault(gene, []).append(program)
    return genes.map(
        lambda gene: ";".join(program_by_gene.get(str(gene), ()))
    )


def supported_programs_by_domain(
    marker_summary: pd.DataFrame,
) -> dict[str, set[str]]:
    supported = marker_summary.loc[
        (marker_summary["mantel_haenszel_common_odds_ratio"] > 1.0)
        & (marker_summary["mantel_haenszel_fdr_bh"] < 0.05)
    ].copy()
    return (
        supported.groupby(supported["domain"].astype(str))["program"]
        .agg(set)
        .to_dict()
    )


def summarize_weight_stability(
    coefficients: pd.DataFrame,
    gene_universe: pd.Index,
    top_k: int = 100,
    minimum_top_folds: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_frames = []
    correlation_rows = []
    fold_order = list(liver_validation.SECTION_ORDER)
    for domain in DOMAIN_ORDER:
        pivot = (
            coefficients.loc[coefficients["domain"] == domain]
            .pivot(
                index="gene",
                columns="held_out_section",
                values="standardized_weight",
            )
            .reindex(index=gene_universe, columns=fold_order)
            .fillna(0.0)
        )
        model_feature_union = pivot.ne(0).any(axis=1)
        union_weights = pivot.loc[model_feature_union]
        for left_index, left in enumerate(fold_order):
            for right in fold_order[left_index + 1 :]:
                correlation_rows.append(
                    {
                        "domain": domain,
                        "fold_a": left,
                        "fold_b": right,
                        "n_union_features": int(len(union_weights)),
                        "spearman_rho": float(
                            spearmanr(
                                union_weights[left], union_weights[right]
                            ).statistic
                        ),
                    }
                )
        ranks = pivot.rank(axis=0, ascending=False, method="min")
        top_positive = (pivot > 0) & (ranks <= top_k)
        frame = pd.DataFrame(
            {
                "domain": domain,
                "gene": gene_universe,
                "folds_with_nonzero_weight": pivot.ne(0).sum(axis=1).to_numpy(),
                "positive_weight_folds": (pivot > 0).sum(axis=1).to_numpy(),
                "top_positive_weight_folds": top_positive.sum(axis=1).to_numpy(),
                "median_standardized_weight": pivot.median(axis=1).to_numpy(),
                "mean_standardized_weight": pivot.mean(axis=1).to_numpy(),
                "minimum_standardized_weight": pivot.min(axis=1).to_numpy(),
                "maximum_standardized_weight": pivot.max(axis=1).to_numpy(),
            }
        )
        frame["stable_top_positive_weight"] = (
            frame["top_positive_weight_folds"] >= minimum_top_folds
        ) & (frame["median_standardized_weight"] > 0)
        frame["literature_program"] = annotate_literature_programs(frame["gene"])
        gene_frames.append(frame)
    genes = pd.concat(gene_frames, ignore_index=True).sort_values(
        ["domain", "stable_top_positive_weight", "median_standardized_weight"],
        ascending=[True, False, False],
    )
    return genes, pd.DataFrame(correlation_rows)


def annotate_domain_supported_programs(
    genes: pd.DataFrame,
    programs_by_domain: dict[str, set[str]],
) -> pd.Series:
    programs_by_gene = {}
    for program, members in evidence.LITERATURE_PROGRAMS.items():
        for gene in members:
            programs_by_gene.setdefault(gene, set()).add(program)
    return genes.apply(
        lambda row: ";".join(
            sorted(
                programs_by_gene.get(str(row["gene"]), set())
                & programs_by_domain.get(str(row["domain"]), set())
            )
        ),
        axis=1,
    )


def marker_weight_enrichment(
    weight_genes: pd.DataFrame,
    programs_by_domain: dict[str, set[str]],
) -> pd.DataFrame:
    rows = []
    for domain in DOMAIN_ORDER:
        frame = weight_genes.loc[weight_genes["domain"] == domain].copy()
        stable = frame["stable_top_positive_weight"].astype(bool)
        background = set(frame["gene"])
        for program in evidence.LITERATURE_PROGRAMS:
            members = set(evidence.LITERATURE_PROGRAMS[program]) & background
            if not members:
                continue
            member_mask = frame["gene"].isin(members)
            member_weights = frame.loc[member_mask, "median_standardized_weight"]
            other_weights = frame.loc[~member_mask, "median_standardized_weight"]
            rank_test = mannwhitneyu(
                member_weights,
                other_weights,
                alternative="greater",
            )
            rank_auc = float(
                rank_test.statistic / (len(member_weights) * len(other_weights))
            )
            stable_members = int((stable & member_mask).sum())
            table = [
                [stable_members, int((stable & ~member_mask).sum())],
                [
                    int((~stable & member_mask).sum()),
                    int((~stable & ~member_mask).sum()),
                ],
            ]
            stable_test = fisher_exact(table, alternative="greater")
            rows.append(
                {
                    "domain": domain,
                    "program": program,
                    "independent_domain_association": bool(
                        program in programs_by_domain.get(domain, set())
                    ),
                    "n_program_genes_measured": int(len(members)),
                    "median_weight_rank_auc": rank_auc,
                    "median_weight_rank_p_value": float(rank_test.pvalue),
                    "stable_gene_overlap": stable_members,
                    "stable_gene_odds_ratio": float(stable_test.statistic),
                    "stable_gene_fisher_p_value": float(stable_test.pvalue),
                    "stable_genes": ";".join(
                        sorted(set(frame.loc[stable, "gene"]) & members)
                    ),
                }
            )
    result = pd.DataFrame(rows)
    for source, target in (
        ("median_weight_rank_p_value", "median_weight_rank_fdr_bh"),
        ("stable_gene_fisher_p_value", "stable_gene_fisher_fdr_bh"),
    ):
        result[target] = multipletests(result[source], method="fdr_bh")[1]
    return result


def weight_go_enrichment(
    weight_genes: pd.DataFrame,
    gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for domain in DOMAIN_ORDER:
        frame = weight_genes.loc[weight_genes["domain"] == domain]
        background = set(frame["gene"].str.upper())
        selected = set(
            frame.loc[
                frame["stable_top_positive_weight"], "gene"
            ].str.upper()
        )
        for term, raw_members in gene_sets.items():
            members = set(gene.upper() for gene in raw_members) & background
            if not 5 <= len(members) <= 500:
                continue
            overlap = selected & members
            table = [
                [len(overlap), len(selected - members)],
                [len(members - selected), len(background - selected - members)],
            ]
            test = fisher_exact(table, alternative="greater")
            rows.append(
                {
                    "domain": domain,
                    "term": term,
                    "n_stable_genes": len(selected),
                    "n_background_genes": len(background),
                    "n_term_genes_in_background": len(members),
                    "overlap": len(overlap),
                    "odds_ratio": float(test.statistic),
                    "fisher_p_value": float(test.pvalue),
                    "genes": ";".join(sorted(overlap)),
                }
            )
    result = pd.DataFrame(rows)
    result["fisher_fdr_bh"] = np.nan
    for domain in DOMAIN_ORDER:
        selected = result["domain"] == domain
        result.loc[selected, "fisher_fdr_bh"] = multipletests(
            result.loc[selected, "fisher_p_value"], method="fdr_bh"
        )[1]
    return result.sort_values(["fisher_fdr_bh", "fisher_p_value"])


def weight_stability_summary(
    weight_genes: pd.DataFrame,
    correlations: pd.DataFrame,
    marker_enrichment: pd.DataFrame,
    go_enrichment: pd.DataFrame,
) -> dict:
    domain_summary = {}
    for domain in DOMAIN_ORDER:
        domain_correlations = correlations.loc[
            correlations["domain"] == domain, "spearman_rho"
        ]
        stable = weight_genes.loc[
            (weight_genes["domain"] == domain)
            & weight_genes["stable_top_positive_weight"]
        ]
        marker_rows = marker_enrichment.loc[
            (marker_enrichment["domain"] == domain)
            & marker_enrichment["independent_domain_association"]
        ]
        go_rows = go_enrichment.loc[
            (go_enrichment["domain"] == domain)
            & (go_enrichment["fisher_fdr_bh"] < 0.05)
        ]
        domain_summary[domain] = {
            "median_pairwise_weight_spearman_rho": float(
                domain_correlations.median()
            ),
            "minimum_pairwise_weight_spearman_rho": float(
                domain_correlations.min()
            ),
            "maximum_pairwise_weight_spearman_rho": float(
                domain_correlations.max()
            ),
            "stable_top_positive_weight_genes": int(len(stable)),
            "stable_domain_supported_marker_genes": stable.loc[
                stable["domain_supported_literature_program"].ne(""), "gene"
            ].tolist(),
            "marker_program_tests": marker_rows.to_dict("records"),
            "significant_go_terms": go_rows.head(10).to_dict("records"),
        }
    return {
        "n_folds": len(liver_validation.SECTION_ORDER),
        "pairwise_comparisons_per_domain": 10,
        "stable_gene_definition": (
            "positive top-100 coefficient in at least four of five folds"
        ),
        "per_domain": domain_summary,
    }


def save_raster_pdf(figure: plt.Figure, output_path: Path) -> None:
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, resolution=300.0)


def short_section(section: str) -> str:
    return section.removeprefix("E16.5_").removesuffix(".MOSTA")


def plot_transfer_metrics(
    output_path: Path,
    overall: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
) -> None:
    figure = plt.figure(figsize=(14.5, 7.3), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.1, 1, 1))
    axis = figure.add_subplot(grid[:, 0])
    section_confusions = []
    for section in liver_validation.SECTION_ORDER:
        table = confusion.loc[confusion["section"] == section].pivot(
            index="true_domain",
            columns="predicted_domain",
            values="fraction_within_true_domain",
        )
        section_confusions.append(
            table.reindex(index=DOMAIN_ORDER, columns=DOMAIN_ORDER).to_numpy()
        )
    average_confusion = np.nanmean(np.stack(section_confusions), axis=0)
    image = axis.imshow(average_confusion, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(3), [f"Domain {domain}" for domain in DOMAIN_ORDER])
    axis.set_yticks(range(3), [f"Domain {domain}" for domain in DOMAIN_ORDER])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("STEP assignment")
    axis.tick_params(length=0)
    for row in range(3):
        for column in range(3):
            value = average_confusion[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#202020",
                fontsize=11,
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04).set_label(
        "Mean within-domain fraction"
    )

    axis = figure.add_subplot(grid[0, 1:])
    x = np.arange(len(liver_validation.SECTION_ORDER))
    metric_specs = (
        ("balanced_accuracy", "Balanced accuracy", "#345995"),
        ("macro_f1_present_domains", "Macro F1", "#D1495B"),
    )
    width = 0.34
    indexed = overall.set_index("section")
    for offset, (column, label, color) in enumerate(metric_specs):
        values = indexed.reindex(liver_validation.SECTION_ORDER)[column].to_numpy()
        axis.bar(
            x + (offset - 0.5) * width,
            values,
            width=width,
            color=color,
            label=label,
        )
    axis.set_ylim(0, 1)
    axis.set_xticks(x, [short_section(section) for section in liver_validation.SECTION_ORDER])
    axis.set_ylabel("Score")
    axis.legend(frameon=False, ncol=2, loc="upper center")
    axis.spines[["top", "right"]].set_visible(False)

    tested = per_class.loc[per_class["tested"]].copy()
    for plot_index, (metric, ylabel) in enumerate(
        (("auroc", "AUROC"), ("auprc_over_prevalence", "AUPRC / prevalence"))
    ):
        axis = figure.add_subplot(grid[1, plot_index + 1])
        for domain in DOMAIN_ORDER:
            selected = tested["domain"] == domain
            domain_frame = tested.loc[selected].set_index("section")
            values = domain_frame.reindex(liver_validation.SECTION_ORDER)[metric].to_numpy()
            axis.plot(
                x,
                values,
                color=DOMAIN_COLORS[domain],
                marker="o",
                linewidth=1.5,
                markersize=5,
                label=f"Domain {domain}",
            )
        axis.axhline(
            0.5 if metric == "auroc" else 1.0,
            color="#666666",
            linewidth=1,
            linestyle="--",
        )
        if metric == "auroc":
            axis.set_ylim(0, 1)
        else:
            finite = tested[metric].dropna().to_numpy()
            axis.set_ylim(0, max(2.0, float(finite.max()) * 1.08))
        axis.set_xticks(
            x,
            [short_section(section) for section in liver_validation.SECTION_ORDER],
            rotation=30,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        if plot_index == 1:
            axis.legend(frameon=False, fontsize=9)

    save_raster_pdf(figure, output_path)


def scatter_domains(
    axis: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
) -> None:
    axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c="#D9D9D9",
        s=1.2,
        linewidths=0,
        rasterized=True,
    )
    for domain in DOMAIN_ORDER:
        selected = labels == domain
        axis.scatter(
            coords[selected, 0],
            coords[selected, 1],
            c=DOMAIN_COLORS[domain],
            s=1.35,
            linewidths=0,
            rasterized=True,
        )


def plot_spatial_transfer(
    output_path: Path,
    predictions: pd.DataFrame,
) -> None:
    plot_coords = predictions[["x", "y"]].to_numpy(dtype=np.float64)
    plot_coords[:, 1] *= -1
    row_labels = (
        "STEP domains",
        "Transcriptome transfer",
        "P(Domain 2)",
        "P(Domain 15)",
        "P(Domain 26)",
    )
    figure = plt.figure(figsize=(17.5, 13.2), constrained_layout=True)
    grid = figure.add_gridspec(
        len(row_labels),
        len(liver_validation.SECTION_ORDER) + 1,
        width_ratios=(0.16, 1, 1, 1, 1, 1),
        wspace=0.015,
        hspace=0.015,
    )
    probability_axes = []
    probability_image = None
    section_values = predictions["section"].to_numpy()
    true_labels = predictions["true_domain"].to_numpy()
    predicted_labels = predictions["predicted_domain"].to_numpy()

    for row, label in enumerate(row_labels):
        label_axis = figure.add_subplot(grid[row, 0])
        label_axis.text(
            0.95,
            0.5,
            label,
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
        label_axis.axis("off")
        for column, section in enumerate(liver_validation.SECTION_ORDER, start=1):
            axis = figure.add_subplot(grid[row, column])
            selected = section_values == section
            section_coords = plot_coords[selected]
            display = liver_validation.connected_liver_display_mask(
                section_coords,
                edge_radius=1.51,
                min_component_size=20,
            )
            section_coords = section_coords[display]
            section_true_labels = true_labels[selected][display]
            section_predicted_labels = predicted_labels[selected][display]
            if row == 0:
                scatter_domains(axis, section_coords, section_true_labels)
            elif row == 1:
                scatter_domains(axis, section_coords, section_predicted_labels)
            else:
                domain = DOMAIN_ORDER[row - 2]
                probability_image = axis.scatter(
                    section_coords[:, 0],
                    section_coords[:, 1],
                    c=predictions.loc[
                        selected, f"probability_domain_{domain}"
                    ].to_numpy()[display],
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    s=1.3,
                    linewidths=0,
                    rasterized=True,
                )
                probability_axes.append(axis)
            axis.set_aspect("equal")
            axis.axis("off")
            if row == 0:
                axis.set_title(short_section(section), fontsize=11)

    if probability_image is not None:
        colorbar = figure.colorbar(
            probability_image,
            ax=probability_axes,
            location="right",
            fraction=0.012,
            pad=0.008,
        )
        colorbar.set_label("Held-out probability")
    save_raster_pdf(figure, output_path)


def plot_weight_stability(
    output_path: Path,
    correlations: pd.DataFrame,
    weight_genes: pd.DataFrame,
    marker_enrichment: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    stable = weight_genes.loc[
        weight_genes["stable_top_positive_weight"]
    ].copy()
    figure = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(0.8, 1.2))

    axis = figure.add_subplot(grid[0, 0])
    pair_order = list(
        dict.fromkeys(
            zip(
                correlations["fold_a"],
                correlations["fold_b"],
                strict=False,
            )
        )
    )
    pair_labels = [
        f"{short_section(left)} / {short_section(right)}"
        for left, right in pair_order
    ]
    correlation_table = (
        correlations.assign(
            pair=list(
                zip(
                    correlations["fold_a"],
                    correlations["fold_b"],
                    strict=False,
                )
            )
        )
        .pivot(index="pair", columns="domain", values="spearman_rho")
        .reindex(index=pair_order, columns=DOMAIN_ORDER)
    )
    image = axis.imshow(
        correlation_table.to_numpy(),
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axis.set_xticks(
        range(len(DOMAIN_ORDER)),
        [f"Domain {domain}" for domain in DOMAIN_ORDER],
    )
    axis.set_yticks(range(len(pair_labels)), pair_labels)
    axis.set_title("Fold-to-fold weight Spearman rho", fontsize=11)
    axis.tick_params(length=0, labelsize=8.5)
    for row in range(len(pair_order)):
        for column in range(len(DOMAIN_ORDER)):
            value = correlation_table.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.68 else "#202020",
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)

    count_axis = figure.add_subplot(grid[0, 1])
    stable_counts = stable.groupby("domain").size().reindex(
        DOMAIN_ORDER, fill_value=0
    )
    count_axis.bar(
        np.arange(len(DOMAIN_ORDER)),
        stable_counts.to_numpy(),
        color=[DOMAIN_COLORS[domain] for domain in DOMAIN_ORDER],
    )
    count_axis.set_xticks(
        range(len(DOMAIN_ORDER)),
        [f"Domain {domain}" for domain in DOMAIN_ORDER],
    )
    count_axis.set_ylabel("Stable top-weight genes")
    count_axis.set_title("Positive top-100 in at least 4/5 folds", fontsize=11)
    count_axis.spines[["top", "right"]].set_visible(False)

    marker_axis = figure.add_subplot(grid[0, 2])
    marker_frame = marker_enrichment.loc[
        marker_enrichment["independent_domain_association"]
    ].sort_values(
        ["domain", "program"]
    ).reset_index(drop=True)
    marker_y = np.arange(len(marker_frame))
    marker_axis.scatter(
        marker_frame["median_weight_rank_auc"],
        marker_y,
        s=40 + 22 * marker_frame["stable_gene_overlap"],
        c=[DOMAIN_COLORS[str(domain)] for domain in marker_frame["domain"]],
        edgecolor="white",
        linewidth=0.7,
    )
    marker_axis.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    marker_axis.set_yticks(
        marker_y,
        [
            f"Domain {row.domain}: {row.program}"
            for row in marker_frame.itertuples(index=False)
        ],
    )
    marker_axis.set_xlim(0.3, 1.0)
    marker_axis.set_xlabel("Median-weight rank AUC")
    marker_axis.set_title("Marker-program rank AUC", fontsize=11)
    marker_axis.spines[["top", "right"]].set_visible(False)
    for index, row in marker_frame.iterrows():
        if row["median_weight_rank_fdr_bh"] < 0.05:
            marker_axis.text(
                min(float(row["median_weight_rank_auc"]) + 0.025, 0.98),
                index,
                "*",
                va="center",
                fontsize=12,
            )

    displayed_by_domain = {}
    for domain in DOMAIN_ORDER:
        displayed_by_domain[domain] = (
            stable.loc[stable["domain"] == domain]
            .sort_values("median_standardized_weight", ascending=False)
            .head(12)
        )
    fold_columns = list(liver_validation.SECTION_ORDER)
    coefficient_pivot = coefficients.pivot(
        index=["domain", "gene"],
        columns="held_out_section",
        values="standardized_weight",
    ).reindex(columns=fold_columns)
    displayed_values = []
    for domain, frame in displayed_by_domain.items():
        if frame.empty:
            continue
        index = pd.MultiIndex.from_arrays(
            [np.repeat(domain, len(frame)), frame["gene"]]
        )
        displayed_values.append(
            coefficient_pivot.reindex(index).fillna(0).to_numpy().ravel()
        )
    shared_limit = max(
        float(np.nanmax(np.abs(np.concatenate(displayed_values))))
        if displayed_values
        else 0.0,
        0.1,
    )
    shared_norm = TwoSlopeNorm(
        vmin=-shared_limit,
        vcenter=0,
        vmax=shared_limit,
    )
    gene_images = []
    gene_axes = []
    for column, domain in enumerate(DOMAIN_ORDER):
        axis = figure.add_subplot(grid[1, column])
        selected = displayed_by_domain[domain]
        index = pd.MultiIndex.from_arrays(
            [np.repeat(domain, len(selected)), selected["gene"]]
        )
        values = coefficient_pivot.reindex(index).fillna(0).to_numpy()
        if len(selected):
            image = axis.imshow(
                values,
                cmap="RdBu_r",
                norm=shared_norm,
                aspect="auto",
            )
            gene_images.append(image)
            gene_axes.append(axis)
            labels = [
                f"{row.gene}*"
                if row.domain_supported_literature_program
                else str(row.gene)
                for row in selected.itertuples(index=False)
            ]
            axis.set_yticks(range(len(selected)), labels)
            for row in range(len(selected)):
                for col in range(len(fold_columns)):
                    value = values[row, col]
                    axis.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white"
                        if abs(value) > 0.62 * shared_limit
                        else "#202020",
                    )
        else:
            axis.set_yticks([])
        axis.set_xticks(
            range(len(fold_columns)),
            [short_section(section) for section in fold_columns],
            rotation=35,
            ha="right",
        )
        axis.set_title(f"Domain {domain}", color=DOMAIN_COLORS[domain])
        axis.tick_params(length=0, labelsize=9)
    if gene_images:
        figure.colorbar(
            gene_images[-1],
            ax=gene_axes,
            location="bottom",
            fraction=0.035,
            pad=0.08,
            label="Standardized coefficient",
        )
    save_raster_pdf(figure, output_path)


def build_summary(
    overall: pd.DataFrame,
    per_class: pd.DataFrame,
    classifier_iterations: dict[str, int],
    label_mapping_summary: dict,
    weight_stability: dict,
) -> dict:
    tested = per_class.loc[per_class["tested"]]
    domain_summary = {}
    for domain in DOMAIN_ORDER:
        selected = tested["domain"] == domain
        frame = tested.loc[selected]
        domain_summary[domain] = {
            "tested_sections": int(len(frame)),
            "median_auroc": float(frame["auroc"].median()),
            "median_auprc": float(frame["auprc"].median()),
            "median_auprc_over_prevalence": float(
                frame["auprc_over_prevalence"].median()
            ),
            "median_recall": float(frame["recall"].median()),
        }
    return {
        "n_sections": len(liver_validation.SECTION_ORDER),
        "n_out_of_fold_spots": int(overall["n_spots"].sum()),
        "mean_section_balanced_accuracy": float(
            overall["balanced_accuracy"].mean()
        ),
        "median_section_balanced_accuracy": float(
            overall["balanced_accuracy"].median()
        ),
        "mean_section_macro_f1": float(
            overall["macro_f1_present_domains"].mean()
        ),
        "section_domain_combinations_with_majority_correct": int(
            (tested["recall"] > 0.5).sum()
        ),
        "tested_section_domain_combinations": int(len(tested)),
        "label_mapping_null": label_mapping_summary,
        "classifier_weight_stability": weight_stability,
        "classifier_iterations": classifier_iterations,
        "per_domain": domain_summary,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.source, backed="r")
    annotations = source.obs["annotation"].astype(str).to_numpy()
    all_domains = source.obs["domain"].astype(str).to_numpy()
    all_sections = source.obs["batch"].astype(str).to_numpy()
    all_coords = np.asarray(source.obsm["spatial"], dtype=np.float32)
    liver_indices = np.flatnonzero(annotations == "Liver")
    expression = liver_validation.load_raw_liver_expression(
        args.raw_dir,
        source.obs_names,
        all_sections,
        annotations,
        all_domains,
        liver_indices,
    )
    source.file.close()

    target_mask = expression.obs["domain"].astype(str).isin(DOMAIN_ORDER).to_numpy()
    expression = expression[target_mask].copy()
    matrix = expression.X.tocsr()
    sections = expression.obs["section"].astype(str).to_numpy()
    labels = expression.obs["domain"].astype(str).to_numpy()
    coords = all_coords[liver_indices][target_mask]
    genes = pd.Index(expression.var_names.astype(str))

    prediction_frames = []
    overall_rows = []
    class_rows = []
    confusion_rows = []
    feature_frames = []
    coefficient_frames = []
    classifier_iterations = {}

    for fold_index, target_section in enumerate(liver_validation.SECTION_ORDER):
        training_sections = tuple(
            section for section in liver_validation.SECTION_ORDER if section != target_section
        )
        selected_features, feature_frame = select_training_features(
            matrix,
            sections,
            labels,
            genes,
            training_sections,
            args.features_per_domain,
            args.min_stratum_spots,
            args.min_detection_fraction,
        )
        feature_frame.insert(0, "held_out_section", target_section)
        feature_frames.append(feature_frame)

        training = sections != target_section
        target = sections == target_section
        training_values = matrix[training][:, selected_features].toarray()
        target_values = matrix[target][:, selected_features].toarray()
        scaler = StandardScaler()
        training_values = scaler.fit_transform(training_values)
        target_values = scaler.transform(target_values)

        classifier = LogisticRegression(
            C=args.regularization_c,
            solver="lbfgs",
            max_iter=args.max_iterations,
            random_state=args.seed + fold_index,
        )
        training_weights = section_domain_weights(
            sections[training], labels[training]
        )
        classifier.fit(
            training_values,
            labels[training],
            sample_weight=training_weights,
        )
        fold_probabilities = classifier.predict_proba(target_values)
        probability_columns = {
            domain: int(np.flatnonzero(classifier.classes_ == domain)[0])
            for domain in DOMAIN_ORDER
        }
        fold_probabilities = np.column_stack(
            [
                fold_probabilities[:, probability_columns[domain]]
                for domain in DOMAIN_ORDER
            ]
        )
        fold_prediction = np.asarray(DOMAIN_ORDER)[
            np.argmax(fold_probabilities, axis=1)
        ]
        classifier_iterations[target_section] = int(classifier.n_iter_.max())
        coefficients = classifier_coefficient_rows(
            target_section,
            selected_features,
            genes,
            classifier,
            scaler,
            feature_frame,
        )
        coefficient_frames.append(coefficients)

        target_indices = np.flatnonzero(target)
        frame = pd.DataFrame(
            {
                "spot_id": expression.obs_names[target_indices].astype(str),
                "section": target_section,
                "true_domain": labels[target],
                "predicted_domain": fold_prediction,
                "probability_domain_2": fold_probabilities[:, 0],
                "probability_domain_15": fold_probabilities[:, 1],
                "probability_domain_26": fold_probabilities[:, 2],
                "x": coords[target, 0],
                "y": coords[target, 1],
                "n_selected_features": len(selected_features),
            }
        )
        prediction_frames.append(frame)
        overall, classes, confusion = evaluate_fold(
            target_section,
            labels[target],
            fold_prediction,
            fold_probabilities,
        )
        overall["n_selected_features"] = len(selected_features)
        overall["classifier_iterations"] = int(classifier.n_iter_.max())
        overall_rows.append(overall)
        class_rows.extend(classes)
        confusion_rows.extend(confusion)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(class_rows)
    confusion = pd.DataFrame(confusion_rows)
    features = pd.concat(feature_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    minimum_detected = max(
        20, int(np.ceil(matrix.shape[0] * args.min_detection_fraction))
    )
    eligible_genes = biological_gene_mask(genes) & (
        np.asarray(matrix.getnnz(axis=0)).ravel() >= minimum_detected
    )
    weight_genes, weight_correlations = summarize_weight_stability(
        coefficients,
        genes[eligible_genes],
    )
    marker_summary_path = (
        evidence.OUTPUT_DIR / "literature_marker_enrichment_summary.csv"
    )
    if not marker_summary_path.exists():
        raise FileNotFoundError(marker_summary_path)
    marker_summary = pd.read_csv(marker_summary_path)
    marker_summary["domain"] = marker_summary["domain"].astype(str)
    programs_by_domain = supported_programs_by_domain(marker_summary)
    weight_genes["domain_supported_literature_program"] = (
        annotate_domain_supported_programs(
            weight_genes,
            programs_by_domain,
        )
    )
    marker_enrichment = marker_weight_enrichment(
        weight_genes,
        programs_by_domain,
    )
    gene_set_path = evidence.OUTPUT_DIR / f"{evidence.GO_LIBRARY}.Mouse.gmt"
    gene_sets = evidence.load_go_library(gene_set_path)
    go_enrichment = weight_go_enrichment(weight_genes, gene_sets)
    label_mapping_null, label_mapping_summary = exact_label_mapping_null(
        predictions
    )
    weight_summary = weight_stability_summary(
        weight_genes,
        weight_correlations,
        marker_enrichment,
        go_enrichment,
    )

    predictions.to_csv(
        args.output_dir / "loso_transcriptome_predictions.csv.gz", index=False
    )
    overall.to_csv(args.output_dir / "loso_section_metrics.csv", index=False)
    per_class.to_csv(args.output_dir / "loso_per_domain_metrics.csv", index=False)
    confusion.to_csv(args.output_dir / "loso_confusion_matrix.csv", index=False)
    features.to_csv(args.output_dir / "loso_training_features.csv", index=False)
    coefficients.to_csv(
        args.output_dir / "loso_classifier_coefficients.csv.gz", index=False
    )
    weight_genes.to_csv(
        args.output_dir / "loso_weight_gene_stability.csv", index=False
    )
    weight_genes.loc[weight_genes["stable_top_positive_weight"]].to_csv(
        args.output_dir / "loso_stable_top_weight_genes.csv", index=False
    )
    weight_correlations.to_csv(
        args.output_dir / "loso_weight_pairwise_correlations.csv", index=False
    )
    marker_enrichment.to_csv(
        args.output_dir / "loso_weight_marker_enrichment.csv", index=False
    )
    go_enrichment.to_csv(
        args.output_dir / "loso_weight_go_enrichment.csv", index=False
    )
    with (args.output_dir / "loso_weight_stability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(weight_summary, handle, indent=2)
    label_mapping_null.to_csv(
        args.output_dir / "loso_label_mapping_null.csv", index=False
    )
    with (args.output_dir / "loso_label_mapping_null_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(label_mapping_summary, handle, indent=2)

    plot_transfer_metrics(
        args.output_dir / "figure1_loso_transfer_metrics",
        overall,
        per_class,
        confusion,
    )
    plot_spatial_transfer(
        args.output_dir / "figure2_loso_spatial_transfer",
        predictions,
    )
    plot_weight_stability(
        args.output_dir / "figure3_loso_weight_stability",
        weight_correlations,
        weight_genes,
        marker_enrichment,
        coefficients,
    )

    summary = build_summary(
        overall,
        per_class,
        classifier_iterations,
        label_mapping_summary,
        weight_summary,
    )
    manifest = {
        "source": str(args.source),
        "raw_dir": str(args.raw_dir),
        "target_annotation": "Liver",
        "target_domains": list(DOMAIN_ORDER),
        "evaluation_spots": (
            "Liver spots assigned by STEP to Domains 2, 15, or 26; Liver "
            "spots assigned to other domains are outside this three-class test"
        ),
        "normalization": "per-spot CP10K followed by log1p",
        "validation": "leave one complete section out",
        "feature_selection": (
            "training sections only; top positive median within-section "
            "domain-versus-other-domain effects"
        ),
        "classifier": "multinomial L2 logistic regression",
        "training_weights": (
            "equal total weight per domain and equal weight per available "
            "section within each domain"
        ),
        "coordinates_used_for_model": False,
        "step_embedding_used_for_model": False,
        "held_out_labels_used_for_training_or_feature_selection": False,
        "weight_stability": (
            "fold-to-fold coefficient rank correlations over the union of "
            "eligible model features, with genes absent from a fold-specific "
            "selected feature set assigned weight zero"
        ),
        "eligible_weight_gene_background": (
            "non-mitochondrial and non-ribosomal genes detected in at least "
            f"{minimum_detected} of the evaluated Liver spots"
        ),
        "stable_top_weight_gene_definition": (
            "positive top-100 standardized coefficient in at least four of "
            "five leave-one-section-out models"
        ),
        "features_per_domain": args.features_per_domain,
        "min_stratum_spots": args.min_stratum_spots,
        "min_detection_fraction": args.min_detection_fraction,
        "regularization_c": args.regularization_c,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "summary": summary,
    }
    manifest_path = args.output_dir / "analysis_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
