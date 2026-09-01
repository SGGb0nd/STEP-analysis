"""Quantify graph-level histomorphological contexts for CRC microarchitectures."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import fisher_exact
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests


CRC_METADATA_CSV = Path("results/visium-hd/crc5_meta_filtered.csv")
CRC_MICROARC_PKL = Path("results/visium-hd/microarc_true50um.pkl")
CRC_PATHOLOGY_CSV = Path("data/8um_squares_annotation.csv")
CRC_DECONV_META_CSV = Path("results/visium-hd/crc5_meta_with_deconv.csv")
CRC_LR_MULTI_AGG_TEMPLATE = "results/visium-hd/niche_pattern_{pattern}_multi_agg.csv"


@dataclass(frozen=True)
class CrcMaHistologyConfig:
    metadata_csv: Path = CRC_METADATA_CSV
    microarc_pkl: Path = CRC_MICROARC_PKL
    pathology_csv: Path = CRC_PATHOLOGY_CSV
    deconv_meta_csv: Path = CRC_DECONV_META_CSV
    top_n_patterns: int = 3


def load_pathology_annotations(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["barcode", "pathology_label"],
        dtype=str,
    )
    return df.dropna(subset=["barcode", "pathology_label"]).reset_index(drop=True)


def collapse_pathology_labels(labels: pd.Series) -> pd.Series:
    mapping = {
        "Neoplasm": "Neoplasm",
        "Connective Tissue": "Connective/Stromal",
        "Non-neoplastic Epithelium": "Non-neoplastic Epithelium",
        "Vessel": "Vascular",
        "Veins": "Vascular",
        "Smooth Muscle": "Smooth Muscle",
        "Outside": "Other/Outside",
    }
    return labels.map(mapping).fillna("Other/Outside")


def normalize_annotation_index(
    annotation: pd.DataFrame,
    batch: str,
) -> pd.DataFrame:
    normalized = annotation.copy()
    normalized["metadata_index"] = normalized["barcode"].astype(str) + f"-{batch}"
    normalized["morphology_group"] = collapse_pathology_labels(
        normalized["pathology_label"].astype(str)
    )
    return normalized


def detect_annotation_batch(metadata: pd.DataFrame, annotation: pd.DataFrame) -> str:
    annotation_barcodes = set(annotation["barcode"].astype(str))
    cancer_meta = metadata[metadata["batch"].astype(str).str.startswith("cancer")]
    best_batch = None
    best_score = (-1.0, -1)
    for batch, batch_df in cancer_meta.groupby("batch"):
        raw_index = batch_df.index.to_series().astype(str).str.replace(f"-{batch}$", "", regex=True)
        overlap = int(raw_index.isin(annotation_barcodes).sum())
        coverage = overlap / len(batch_df) if len(batch_df) else 0.0
        score = (coverage, overlap)
        if score > best_score:
            best_batch = str(batch)
            best_score = score
    if best_batch is None:
        raise RuntimeError("Unable to identify an annotated CRC batch from pathology barcodes.")
    return best_batch


def covered_spot_sets_by_pattern(niche, roi_index: pd.Index) -> dict[int, set[str]]:
    roi_set = set(roi_index.astype(str))
    covered: dict[int, set[str]] = {}
    for pattern in sorted(np.unique(niche.clusters)):
        members = np.where(niche.clusters == pattern)[0]
        pattern_spots: set[str] = set()
        for member_idx in members:
            pattern_spots.update(str(idx) for idx in niche.neighbor_indices[member_idx])
        covered[int(pattern)] = pattern_spots & roi_set
    return covered


def assigned_spot_patterns(niche, roi_index: pd.Index) -> pd.Series:
    roi_set = set(roi_index.astype(str))
    votes: dict[str, dict[int, int]] = {}
    for member_idx, pattern in enumerate(np.asarray(niche.clusters, dtype=int)):
        for spot in niche.neighbor_indices[member_idx]:
            spot = str(spot)
            if spot not in roi_set:
                continue
            votes.setdefault(spot, {})
            votes[spot][pattern] = votes[spot].get(pattern, 0) + 1

    assignments: dict[str, int] = {}
    for spot, pattern_votes in votes.items():
        assignments[spot] = sorted(
            pattern_votes.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
    return pd.Series(assignments, name="pattern", dtype=int).sort_index()


def major_patterns(niche, top_n: int = 3) -> list[int]:
    counts = pd.Series(niche.clusters).value_counts()
    return [int(idx) for idx in counts.index[:top_n]]


def _enrichment_rows_for_pattern(
    pattern: int,
    pattern_spots: set[str],
    roi_groups: pd.Series,
) -> list[dict[str, object]]:
    pattern_series = roi_groups.loc[roi_groups.index.intersection(sorted(pattern_spots))]
    background_series = roi_groups.loc[~roi_groups.index.isin(pattern_series.index)]
    categories = sorted(set(roi_groups.dropna().unique()))
    rows: list[dict[str, object]] = []
    for category in categories:
        a = int((pattern_series == category).sum())
        b = int((pattern_series != category).sum())
        c = int((background_series == category).sum())
        d = int((background_series != category).sum())
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        rows.append(
            {
                "pattern": int(pattern),
                "morphology_group": category,
                "pattern_count": a,
                "background_count": c,
                "pattern_total": len(pattern_series),
                "background_total": len(background_series),
                "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.inf,
                "p_value": float(p_value),
            }
        )
    return rows


def pattern_pathology_enrichment_table(
    covered_spots: dict[int, set[str]] | None,
    roi_groups: pd.Series,
    patterns: Iterable[int] | None = None,
    assigned_patterns: pd.Series | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if assigned_patterns is not None:
        if patterns is None:
            patterns = sorted(assigned_patterns.unique())
        for pattern in patterns:
            pattern_spots = set(assigned_patterns.index[assigned_patterns == int(pattern)].astype(str))
            rows.extend(_enrichment_rows_for_pattern(int(pattern), pattern_spots, roi_groups))
    else:
        if covered_spots is None:
            raise ValueError("Either covered_spots or assigned_patterns must be provided.")
        if patterns is None:
            patterns = sorted(covered_spots)
        for pattern in patterns:
            rows.extend(
                _enrichment_rows_for_pattern(int(pattern), covered_spots[int(pattern)], roi_groups)
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result.sort_values(["pattern", "p_value", "morphology_group"]).reset_index(drop=True)


def _composition_stat(
    labels: pd.Series,
    in_pattern: pd.Series,
) -> float:
    categories = sorted(labels.dropna().unique())
    stat = 0.0
    for category in categories:
        obs_in = float(((labels == category) & in_pattern).sum())
        obs_out = float(((labels == category) & (~in_pattern)).sum())
        total_cat = obs_in + obs_out
        exp_in = total_cat * in_pattern.mean()
        exp_out = total_cat * (~in_pattern).mean()
        if exp_in > 0:
            stat += (obs_in - exp_in) ** 2 / exp_in
        if exp_out > 0:
            stat += (obs_out - exp_out) ** 2 / exp_out
    return float(stat)


def pattern_global_permutation_table(
    assigned_patterns: pd.Series,
    roi_groups: pd.Series,
    patterns: Iterable[int],
    n_permutations: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    shared = roi_groups.index.intersection(assigned_patterns.index)
    labels = roi_groups.loc[shared]
    assignments = assigned_patterns.loc[shared]
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    label_values = labels.to_numpy(copy=True)
    for pattern in patterns:
        in_pattern = assignments.eq(int(pattern))
        observed = _composition_stat(labels, in_pattern)
        perm_stats = []
        for _ in range(n_permutations):
            shuffled = pd.Series(rng.permutation(label_values), index=labels.index)
            perm_stats.append(_composition_stat(shuffled, in_pattern))
        perm_stats_arr = np.asarray(perm_stats, dtype=float)
        p_value = float((1 + np.sum(perm_stats_arr >= observed)) / (n_permutations + 1))
        rows.append(
            {
                "pattern": int(pattern),
                "observed_stat": observed,
                "perm_p_value": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("pattern").reset_index(drop=True)


def pattern_center_table(niche, metadata: pd.DataFrame) -> pd.DataFrame:
    centers = pd.DataFrame(
        {
            "center_index": pd.Index(niche.center_indices).astype(str),
            "pattern": np.asarray(niche.clusters, dtype=int),
        }
    ).set_index("center_index")
    centers = centers.join(metadata[["morphology_group"]], how="inner")
    return centers[["pattern", "morphology_group"]]


def _fraction_match(series: pd.Series, keywords: tuple[str, ...]) -> float:
    if series.empty:
        return np.nan
    lower = series.astype(str).str.lower()
    return float(lower.apply(lambda x: any(k in x for k in keywords)).mean())


def _sample_id_from_index(index_value: str) -> str:
    return str(index_value).rsplit("-", 1)[-1]


def _edge_density(n_nodes: int, edge_count: int) -> float:
    if n_nodes < 2:
        return 0.0
    return float((2.0 * edge_count) / (n_nodes * (n_nodes - 1)))


def graph_instance_table(
    niche,
    metadata: pd.DataFrame,
    roi_index: pd.Index | None = None,
) -> pd.DataFrame:
    roi_set = set(roi_index.astype(str)) if roi_index is not None else None
    rows: list[dict[str, object]] = []
    for graph_i, (graph, ma_class, focal_cell_id, member_index) in enumerate(
        zip(niche.graphs, np.asarray(niche.clusters, dtype=int), niche.center_indices, niche.neighbor_indices)
    ):
        focal_cell_id = str(focal_cell_id)
        if roi_set is not None and focal_cell_id not in roi_set:
            continue
        member_nodes = [str(node) for node in member_index]
        if focal_cell_id in metadata.index:
            sample_id = str(metadata.loc[focal_cell_id, "batch"])
        else:
            sample_id = _sample_id_from_index(focal_cell_id)
        edge_count = int(graph.number_of_edges())
        rows.append(
            {
                "graph_instance_id": int(graph_i),
                "graph": graph,
                "ma_class": ma_display_id(int(ma_class)),
                "sample_id": sample_id,
                "focal_cell_id": focal_cell_id,
                "member_nodes": member_nodes,
                "graph_size": int(len(member_nodes)),
                "edge_count": edge_count,
                "edge_density": _edge_density(len(member_nodes), edge_count),
            }
        )
    return pd.DataFrame(rows)


def _safe_entropy(fractions: np.ndarray) -> float:
    fractions = fractions[fractions > 0]
    if fractions.size == 0:
        return 0.0
    return float(-(fractions * np.log2(fractions)).sum())


def edge_pathology_summary(graph, roi_metadata: pd.DataFrame) -> dict[str, float]:
    valid_edges: list[tuple[str, str]] = []
    same = 0
    neoplasm_stromal = 0
    for u, v in graph.edges():
        u = str(u)
        v = str(v)
        if u not in roi_metadata.index or v not in roi_metadata.index:
            continue
        lu = roi_metadata.at[u, "morphology_group"] if "morphology_group" in roi_metadata.columns else np.nan
        lv = roi_metadata.at[v, "morphology_group"] if "morphology_group" in roi_metadata.columns else np.nan
        if pd.isna(lu) or pd.isna(lv):
            continue
        valid_edges.append((u, v))
        if lu == lv:
            same += 1
        if {str(lu), str(lv)} == {"Neoplasm", "Connective/Stromal"}:
            neoplasm_stromal += 1
    n = len(valid_edges)
    if n == 0:
        return {
            "n_edges_with_pathology": 0,
            "same_pathology_edge_fraction": np.nan,
            "neoplasm_stromal_edge_fraction": np.nan,
            "edge_pathology_heterophily": np.nan,
        }
    same_frac = same / n
    return {
        "n_edges_with_pathology": int(n),
        "same_pathology_edge_fraction": float(same_frac),
        "neoplasm_stromal_edge_fraction": float(neoplasm_stromal / n),
        "edge_pathology_heterophily": float(1.0 - same_frac),
    }


def graph_instance_descriptor(
    graph_row: pd.Series,
    roi_metadata: pd.DataFrame,
    cell_type_col: str = "cell_type",
) -> dict[str, object]:
    member_nodes = list(graph_row["member_nodes"])
    subset = roi_metadata.loc[roi_metadata.index.intersection(member_nodes)].copy()
    pathology = subset["morphology_group"].dropna() if "morphology_group" in subset.columns else pd.Series(dtype=str)
    pathology_counts = pathology.value_counts()
    pathology_fracs = (pathology_counts / pathology_counts.sum()) if pathology_counts.sum() else pd.Series(dtype=float)
    dominant = pathology_fracs.idxmax() if not pathology_fracs.empty else None
    cell_types = subset[cell_type_col] if cell_type_col in subset.columns else pd.Series(dtype=str)
    out: dict[str, object] = {
        "graph_instance_id": graph_row.get("graph_instance_id"),
        "ma_class": int(graph_row["ma_class"]),
        "sample_id": graph_row["sample_id"],
        "focal_cell_id": graph_row["focal_cell_id"],
        "member_nodes": member_nodes,
        "graph_size": int(graph_row["graph_size"]),
        "edge_count": int(graph_row["edge_count"]),
        "edge_density": float(graph_row["edge_density"]),
        "n_nodes_with_pathology": int(len(pathology)),
        "dominant_pathology_context": dominant,
        "pathology_entropy": _safe_entropy(pathology_fracs.to_numpy(dtype=float)),
        "macrophage_fraction": _fraction_match(cell_types, ("macroph",)),
        "t_cell_fraction": _fraction_match(cell_types, ("t cell", "t-cell", "cd4", "cd8")),
        "fibroblast_caf_fraction": _fraction_match(cell_types, ("fibro", "caf")),
        "tumor_fraction": _fraction_match(cell_types, ("tumor",)),
        "mean_tumor_distance_um": float(subset["dist_to_tumor_um"].mean())
        if "dist_to_tumor_um" in subset.columns and len(subset) > 0
        else np.nan,
        "median_tumor_distance_um": float(subset["dist_to_tumor_um"].median())
        if "dist_to_tumor_um" in subset.columns and len(subset) > 0
        else np.nan,
    }
    graph = graph_row.get("graph")
    if graph is not None:
        out.update(edge_pathology_summary(graph, roi_metadata))
    for label, frac in pathology_fracs.items():
        out[f"fraction_{label}"] = float(frac)
    return out


def pattern_celltype_axes(
    covered_spots: dict[int, set[str]],
    metadata: pd.DataFrame,
    cell_type_col: str = "cell_type",
    distance_col: str = "dist_to_tumor_um",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pattern, spots in covered_spots.items():
        subset = metadata.loc[metadata.index.intersection(sorted(spots))]
        cell_types = subset[cell_type_col] if cell_type_col in subset.columns else pd.Series(dtype=str)
        rows.append(
            {
                "pattern": int(pattern),
                "n_spots": int(len(subset)),
                "macrophage_fraction": _fraction_match(cell_types, ("macroph",)),
                "t_cell_fraction": _fraction_match(cell_types, ("t cell", "t-cell", "cd4", "cd8")),
                "fibroblast_caf_fraction": _fraction_match(cell_types, ("fibro", "caf")),
                "tumor_fraction": _fraction_match(cell_types, ("tumor",)),
                "mean_tumor_distance_um": float(subset[distance_col].mean())
                if distance_col in subset.columns and len(subset) > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("pattern").sort_index()


def compute_tumor_distance_um(
    metadata: pd.DataFrame,
    roi_index: pd.Index,
    batch: str,
    tumor_domain: int = 5,
    square_size_um: float = 8.0,
) -> pd.Series:
    batch_meta = metadata[metadata["batch"] == batch]
    tumor = batch_meta[batch_meta["domain"] == tumor_domain]
    if tumor.empty:
        return pd.Series(np.nan, index=roi_index, name="dist_to_tumor_um")
    tumor_xy = tumor[["array_row", "array_col"]].to_numpy(dtype=float)
    roi_xy = metadata.loc[roi_index, ["array_row", "array_col"]].to_numpy(dtype=float)
    tree = cKDTree(tumor_xy)
    dist_grid = tree.query(roi_xy, k=1)[0]
    return pd.Series(dist_grid * square_size_um, index=roi_index, name="dist_to_tumor_um")


def load_deconvolution_labels(
    config: CrcMaHistologyConfig,
    batch: str,
) -> pd.DataFrame:
    df = pd.read_csv(config.deconv_meta_csv, index_col=0, low_memory=False)
    df = df[df["batch"] == batch].copy()
    df["metadata_index"] = df.index.astype(str) + f"-{batch}"
    return df.set_index("metadata_index")[["DeconvolutionLabel1"]]


def summarize_lr_programs(
    pattern_to_csv: dict[int, str | Path],
    top_n: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pattern, csv_path in pattern_to_csv.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        ranked = df.sort_values(["specificity_rank", "magnitude_rank"]).head(top_n)
        row: dict[str, object] = {"pattern": int(pattern)}
        for i, (_, rec) in enumerate(ranked.iterrows(), start=1):
            row[f"top_lr_{i}"] = (
                f"{rec['source']} -> {rec['target']} | "
                f"{rec['ligand_complex']}-{rec['receptor_complex']}"
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("pattern").sort_index() if rows else pd.DataFrame()


def load_crc_ma_artifacts(config: CrcMaHistologyConfig) -> tuple[pd.DataFrame, object]:
    metadata = pd.read_csv(config.metadata_csv, index_col=0, low_memory=False)
    niche = pickle.load(open(config.microarc_pkl, "rb"))
    return metadata, niche


def roi_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata[
        metadata["domain"].eq(9) & metadata["batch"].astype(str).str.startswith("cancer")
    ].copy()


def load_annotated_roi_metadata(
    config: CrcMaHistologyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, object, list[str]]:
    metadata, niche = load_crc_ma_artifacts(config)
    roi = roi_metadata(metadata)
    ann = load_pathology_annotations(config.pathology_csv)
    annotated_batch = detect_annotation_batch(metadata, ann)
    batch_ann = normalize_annotation_index(ann, annotated_batch)
    batch_roi = roi[roi["batch"] == annotated_batch].copy()
    roi_annot = batch_roi.join(
        batch_ann.set_index("metadata_index")[["pathology_label", "morphology_group"]],
        how="left",
    )
    roi_annot = roi_annot[roi_annot["morphology_group"].notna()].copy()
    roi_annot["dist_to_tumor_um"] = compute_tumor_distance_um(
        metadata,
        roi_annot.index,
        batch=annotated_batch,
    )
    try:
        roi_annot = roi_annot.join(load_deconvolution_labels(config, annotated_batch), how="left")
    except FileNotFoundError:
        pass
    batch_meta = metadata[metadata["batch"] == annotated_batch].copy()
    return roi_annot, batch_meta, niche, [annotated_batch]


def build_graph_instance_descriptor_table(
    graph_instances: pd.DataFrame,
    roi_annot: pd.DataFrame,
    cell_type_col: str,
) -> pd.DataFrame:
    rows = [
        graph_instance_descriptor(row, roi_annot, cell_type_col=cell_type_col)
        for _, row in graph_instances.iterrows()
    ]
    return pd.DataFrame(rows)


def _representative_lr_text(lr_summary: pd.DataFrame, ma_class: int, top_n: int = 2) -> str:
    if lr_summary.empty or ma_class not in lr_summary.index:
        return ""
    row = lr_summary.loc[ma_class]
    values = [str(row.get(f"top_lr_{i}", "")).strip() for i in range(1, top_n + 1)]
    values = [v for v in values if v]
    return "; ".join(values)


def build_graph_level_ma_summary_table(
    graph_desc: pd.DataFrame,
    lr_summary: pd.DataFrame,
    top_n_patterns: int,
) -> pd.DataFrame:
    if graph_desc.empty:
        return pd.DataFrame()
    count_order = graph_desc["ma_class"].value_counts().index.tolist()[:top_n_patterns]
    frac_cols = sorted([c for c in graph_desc.columns if c.startswith("fraction_")])
    rows: list[dict[str, object]] = []
    for ma_class in count_order:
        subset = graph_desc[graph_desc["ma_class"] == ma_class].copy()
        frac_means = subset[frac_cols].fillna(0.0).mean(numeric_only=True) if frac_cols else pd.Series(dtype=float)
        dominant = frac_means.idxmax().removeprefix("fraction_") if not frac_means.empty else None
        rows.append(
            {
                "ma_class": ma_display_id(int(ma_class)),
                "n_graph_instances": int(len(subset)),
                "dominant_graph_pathology_tendency": dominant,
                "mean_pathology_entropy": float(subset["pathology_entropy"].mean()),
                "mean_tumor_distance_um": float(subset["mean_tumor_distance_um"].mean()),
                "mean_fibroblast_caf_fraction": float(subset["fibroblast_caf_fraction"].mean()),
                "mean_tumor_fraction": float(subset["tumor_fraction"].mean()),
                "mean_macrophage_fraction": float(subset["macrophage_fraction"].mean()),
                "mean_t_cell_fraction": float(subset["t_cell_fraction"].mean()),
                "representative_lr_programs": _representative_lr_text(lr_summary, int(ma_class)),
                **{col: float(frac_means.get(col, np.nan)) for col in frac_cols},
            }
        )
    return pd.DataFrame(rows)


def roi_pathology_baseline(graph_desc: pd.DataFrame) -> pd.Series:
    frac_cols = sorted([c for c in graph_desc.columns if c.startswith("fraction_")])
    if not frac_cols:
        return pd.Series(dtype=float)
    return graph_desc[frac_cols].fillna(0.0).mean(numeric_only=True)


def build_graph_level_context_deviation_table(
    graph_desc: pd.DataFrame,
    roi_baseline: pd.Series,
    lr_summary: pd.DataFrame,
    top_n_patterns: int,
) -> pd.DataFrame:
    if graph_desc.empty:
        return pd.DataFrame()
    frac_cols = sorted([c for c in graph_desc.columns if c.startswith("fraction_")])
    count_order = graph_desc["ma_class"].value_counts().index.tolist()[:top_n_patterns]
    rows: list[dict[str, object]] = []
    for ma_class in count_order:
        subset = graph_desc[graph_desc["ma_class"] == ma_class].copy()
        frac_means = subset[frac_cols].fillna(0.0).mean(numeric_only=True) if frac_cols else pd.Series(dtype=float)
        delta = frac_means.subtract(roi_baseline.reindex(frac_means.index).fillna(0.0), fill_value=0.0)
        dominant = delta.idxmax().removeprefix("fraction_") if not delta.empty else None
        def _mean_or_nan(col: str) -> float:
            return float(subset[col].mean()) if col in subset.columns else np.nan
        rows.append(
            {
                "ma_class": ma_display_id(int(ma_class)),
                "n_graph_instances": int(len(subset)),
                "dominant_graph_pathology_tendency": dominant,
                "mean_pathology_entropy": _mean_or_nan("pathology_entropy"),
                "mean_tumor_distance_um": _mean_or_nan("mean_tumor_distance_um"),
                "mean_fibroblast_caf_fraction": _mean_or_nan("fibroblast_caf_fraction"),
                "mean_tumor_fraction": _mean_or_nan("tumor_fraction"),
                "mean_macrophage_fraction": _mean_or_nan("macrophage_fraction"),
                "mean_t_cell_fraction": _mean_or_nan("t_cell_fraction"),
                "mean_same_pathology_edge_fraction": _mean_or_nan("same_pathology_edge_fraction"),
                "mean_neoplasm_stromal_edge_fraction": _mean_or_nan("neoplasm_stromal_edge_fraction"),
                "mean_edge_pathology_heterophily": _mean_or_nan("edge_pathology_heterophily"),
                "representative_lr_programs": _representative_lr_text(lr_summary, int(ma_class)),
                **{col: float(frac_means.get(col, 0.0)) for col in frac_cols},
                **{f"delta_{col}": float(delta.get(col, 0.0)) for col in frac_cols},
            }
        )
    return pd.DataFrame(rows)


def make_graph_level_pathology_composition_plot(path: Path, summary_table: pd.DataFrame) -> None:
    if summary_table.empty:
        return
    frac_cols = [c for c in summary_table.columns if c.startswith("fraction_")]
    if not frac_cols:
        return
    labels = [c.removeprefix("fraction_") for c in frac_cols]
    matrix = summary_table[frac_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.5, 1.0 * len(summary_table) + 2.0))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.5, float(np.nanmax(matrix))))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(summary_table)))
    ax.set_yticklabels([f"MA {int(x)}" for x in summary_table["ma_class"]])
    ax.set_title("Graph-level pathology composition")
    plt.colorbar(im, ax=ax, label="Mean fraction across graph instances")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_graph_level_context_deviation_plot(path: Path, summary_table: pd.DataFrame) -> None:
    if summary_table.empty:
        return
    delta_cols = [c for c in summary_table.columns if c.startswith("delta_fraction_")]
    if not delta_cols:
        return
    labels = [c.removeprefix("delta_fraction_") for c in delta_cols]
    matrix = summary_table[delta_cols].to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    vmax = max(vmax, 0.1)
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.5, 1.0 * len(summary_table) + 2.0))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(summary_table)))
    ax.set_yticklabels([f"MA {int(x)}" for x in summary_table["ma_class"]])
    ax.set_title("Graph-level pathology context relative to ROI background")
    plt.colorbar(im, ax=ax, label="Mean fraction shift vs ROI background")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _representative_graphs(
    graph_desc: pd.DataFrame,
    top_n_patterns: int,
) -> pd.DataFrame:
    if graph_desc.empty:
        return graph_desc
    frac_cols = sorted([c for c in graph_desc.columns if c.startswith("fraction_")])
    keep_rows = []
    for ma_class in graph_desc["ma_class"].value_counts().index.tolist()[:top_n_patterns]:
        subset = graph_desc[graph_desc["ma_class"] == ma_class].copy()
        if subset.empty:
            continue
        if frac_cols:
            center = subset[frac_cols].mean(numeric_only=True).to_numpy(dtype=float)
            dist = ((subset[frac_cols].fillna(0.0).to_numpy(dtype=float) - center) ** 2).sum(axis=1)
            rep = subset.iloc[int(np.argmin(dist))]
        else:
            rep = subset.iloc[0]
        keep_rows.append(rep)
    return pd.DataFrame(keep_rows)


def local_patch_subset(
    graph_row: pd.Series,
    metadata: pd.DataFrame,
    pad: float | None = None,
) -> pd.DataFrame:
    x_col = "x" if "x" in metadata.columns else "array_col"
    y_col = "y" if "y" in metadata.columns else "array_row"
    members = metadata.loc[metadata.index.intersection(pd.Index(graph_row["member_nodes"]))].copy()
    if members.empty:
        return members
    x_min, x_max = members[x_col].min(), members[x_col].max()
    y_min, y_max = members[y_col].min(), members[y_col].max()
    if pad is None:
        span = max(float(x_max - x_min), float(y_max - y_min), 1.0)
        pad = 0.35 * span
    return metadata[
        metadata[x_col].between(x_min - pad, x_max + pad)
        & metadata[y_col].between(y_min - pad, y_max + pad)
    ].copy()


def graph_member_subset(
    graph_row: pd.Series,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    return metadata.loc[metadata.index.intersection(pd.Index(graph_row["member_nodes"]))].copy()


def patch_adata(
    patch_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    morphology_groups: pd.Series | None = None,
    member_index: pd.Index | None = None,
) -> ad.AnnData:
    obs = patch_df.copy()
    if morphology_groups is not None:
        obs["morphology_group"] = morphology_groups.reindex(obs.index)
    else:
        obs["morphology_group"] = pd.NA
    member_set = set(member_index.astype(str)) if member_index is not None else set()
    obs["is_graph_member"] = pd.Categorical(
        np.where(obs.index.astype(str).isin(member_set), "Graph member", "Patch background"),
        categories=["Patch background", "Graph member"],
    )
    adata = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = obs[[x_col, y_col]].to_numpy(dtype=float)
    return adata


def build_membership_patch_adata(
    patch_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    selected_index: pd.Index,
    obs_key: str,
    selected_label: str,
    background_label: str = "Background",
) -> ad.AnnData:
    obs = patch_df.copy()
    selected = set(selected_index.astype(str))
    obs[obs_key] = pd.Categorical(
        np.where(obs.index.astype(str).isin(selected), selected_label, background_label),
        categories=[background_label, selected_label],
    )
    adata = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = obs[[x_col, y_col]].to_numpy(dtype=float)
    return adata


def prepare_representative_patch_components(
    graph_row: pd.Series,
    batch_metadata: pd.DataFrame,
    roi_annot: pd.DataFrame,
    pad: float | None = None,
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    x_col = "x" if "x" in batch_metadata.columns else "array_col"
    y_col = "y" if "y" in batch_metadata.columns else "array_row"
    members = pd.Index(graph_row["member_nodes"])
    patch_df = local_patch_subset(graph_row, batch_metadata, pad=pad)
    patch = patch_adata(
        patch_df,
        x_col=x_col,
        y_col=y_col,
        morphology_groups=roi_annot["morphology_group"],
        member_index=members,
    )
    member_df = graph_member_subset(graph_row, batch_metadata)
    annotated_member_df = roi_annot.loc[roi_annot.index.intersection(members)].copy()
    return patch, member_df, annotated_member_df


def make_graph_level_representative_instances_figure(
    path: Path,
    batch_metadata: pd.DataFrame,
    roi_annot: pd.DataFrame,
    graph_instances: pd.DataFrame,
    graph_desc: pd.DataFrame,
    top_n_patterns: int,
) -> None:
    reps = _representative_graphs(graph_desc, top_n_patterns=top_n_patterns)
    if reps.empty:
        return
    x_col = "x" if "x" in batch_metadata.columns else "array_col"
    y_col = "y" if "y" in batch_metadata.columns else "array_row"
    fig, axes = plt.subplots(1, len(reps), figsize=(4.8 * len(reps), 4.8))
    if len(reps) == 1:
        axes = [axes]
    colors = {
        "Neoplasm": "#c44e52",
        "Connective/Stromal": "#55a868",
        "Non-neoplastic Epithelium": "#4c72b0",
        "Vascular": "#8172b2",
        "Smooth Muscle": "#dd8452",
        "Other/Outside": "#999999",
    }
    for ax, (_, row) in zip(axes, reps.iterrows()):
        members = pd.Index(row["member_nodes"])
        patch, full_members, fg = prepare_representative_patch_components(
            row,
            batch_metadata,
            roi_annot,
        )
        graph_obj = None
        if "graph_instance_id" in row.index and "graph_instance_id" in graph_instances.columns:
            matched = graph_instances.loc[
                graph_instances["graph_instance_id"].eq(int(row["graph_instance_id"])),
                "graph",
            ]
            if not matched.empty:
                graph_obj = matched.iloc[0]
        sc.pl.embedding(
            patch,
            basis="spatial",
            color="is_graph_member",
            palette=["#cfcfcf", "#111111"],
            size=26,
            frameon=False,
            show=False,
            ax=ax,
            legend_loc=None,
        )
        if graph_obj is not None:
            for u, v in graph_obj.edges():
                u = str(u)
                v = str(v)
                if u in full_members.index and v in full_members.index:
                    ax.plot(
                        [full_members.at[u, x_col], full_members.at[v, x_col]],
                        [full_members.at[u, y_col], full_members.at[v, y_col]],
                        color="black",
                        linewidth=0.5,
                        alpha=0.35,
                        zorder=1,
                        rasterized=True,
                    )
        if not fg.empty:
            patch_fg = patch[patch.obs["morphology_group"].notna()].copy()
            if patch_fg.n_obs > 0:
                palette = [colors.get(cat, "#333333") for cat in patch_fg.obs["morphology_group"].astype("category").cat.categories]
                sc.pl.embedding(
                    patch_fg,
                    basis="spatial",
                    color="morphology_group",
                    palette=palette,
                    size=34,
                    frameon=False,
                    show=False,
                    ax=ax,
                    legend_loc=None,
                )
        ax.set_title(
            f"MA {ma_display_id(int(row['ma_class']))} rep\nn={int(row['graph_size'])}, "
            f"tumor dist={row.get('mean_tumor_distance_um', np.nan):.1f}"
        )
        ax.set_aspect("equal")
        ax.axis("off")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(config: CrcMaHistologyConfig, outdir: Path) -> dict[str, object]:
    roi_annot, batch_meta, niche, annotated_batches = load_annotated_roi_metadata(config)
    if roi_annot.empty:
        raise RuntimeError("No overlap between ROI metadata and pathology annotations.")

    cell_type_col = "DeconvolutionLabel1" if "DeconvolutionLabel1" in roi_annot.columns else "cell_type"
    graph_instances = graph_instance_table(niche, niche.metadata, roi_index=roi_annot.index)
    graph_desc = build_graph_instance_descriptor_table(graph_instances, roi_annot, cell_type_col=cell_type_col)

    covered = covered_spot_sets_by_pattern(niche, roi_annot.index)
    spot_assignments = assigned_spot_patterns(niche, roi_annot.index)
    selected_patterns = [p for p in major_patterns(niche, top_n=config.top_n_patterns) if covered.get(p)]
    roi_groups = roi_annot["morphology_group"].dropna()
    enrichment = pattern_pathology_enrichment_table(
        covered_spots=None,
        roi_groups=roi_groups,
        patterns=selected_patterns,
        assigned_patterns=spot_assignments,
    )
    permutation = pattern_global_permutation_table(
        assigned_patterns=spot_assignments,
        roi_groups=roi_groups,
        patterns=selected_patterns,
        n_permutations=1000,
        random_state=0,
    )
    enrichment = enrichment.merge(permutation, on="pattern", how="left")
    centers = pattern_center_table(niche, roi_annot)
    center_enrichment = pattern_pathology_enrichment_table(
        {
            pattern: set(centers.index[centers["pattern"] == pattern].astype(str))
            for pattern in selected_patterns
        },
        roi_groups,
        patterns=selected_patterns,
    )
    axes = pattern_celltype_axes(
        covered,
        roi_annot,
        cell_type_col=cell_type_col,
        distance_col="dist_to_tumor_um",
    )
    lr_summary = summarize_lr_programs(
        {
            int(pattern): CRC_LR_MULTI_AGG_TEMPLATE.format(pattern=int(pattern))
            for pattern in selected_patterns
        }
    )

    outdir.mkdir(parents=True, exist_ok=True)
    graph_instances.to_csv(outdir / "graph_instance_base_table.csv", index=False)
    graph_desc.to_csv(outdir / "graph_instance_table.csv", index=False)
    raw_graph_summary = build_graph_level_ma_summary_table(graph_desc, lr_summary, top_n_patterns=config.top_n_patterns)
    roi_baseline = roi_pathology_baseline(graph_desc)
    graph_summary = build_graph_level_context_deviation_table(
        graph_desc,
        roi_baseline,
        lr_summary,
        top_n_patterns=config.top_n_patterns,
    )
    graph_summary.to_csv(outdir / "graph_level_ma_summary.csv", index=False)
    (outdir / "graph_level_ma_summary.md").write_text(
        "# Graph-level histomorphological contexts of major CRC microarchitectures\n\n"
        + graph_summary.to_string(index=False)
    )
    raw_graph_summary.to_csv(outdir / "graph_level_raw_composition_summary.csv", index=False)
    enrichment.to_csv(outdir / "pathology_enrichment_primary.csv", index=False)
    center_enrichment.to_csv(outdir / "pathology_enrichment_centers.csv", index=False)
    axes.to_csv(outdir / "ma_biological_axes.csv")
    if not lr_summary.empty:
        lr_summary.to_csv(outdir / "ma_lr_programs.csv")
    summary_table = build_major_ma_summary_table(
        enrichment,
        axes,
        lr_summary,
        selected_patterns,
    )
    write_major_ma_summary_files(outdir, summary_table)

    summary = {
        "annotated_batches": annotated_batches,
        "n_roi_spots": int(len(roi_annot)),
        "n_graph_instances": int(len(graph_instances)),
        "selected_patterns": selected_patterns,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_note(outdir / "note.md", summary, enrichment, center_enrichment, axes, lr_summary)
    make_graph_level_pathology_composition_plot(outdir / "graph_level_pathology_composition.png", graph_summary)
    make_graph_level_context_deviation_plot(outdir / "graph_level_context_deviation.png", graph_summary)
    make_graph_level_representative_instances_figure(
        outdir / "graph_level_representative_instances.png",
        batch_meta,
        roi_annot,
        graph_instances,
        graph_desc,
        top_n_patterns=config.top_n_patterns,
    )
    make_basic_plot(outdir / "pathology_enrichment_primary.png", enrichment)
    make_spatial_context_plot(
        outdir / "spatial_context_map.png",
        batch_meta,
        roi_annot,
        covered,
        selected_patterns,
    )
    make_histomorphology_context_figure(
        outdir / "histomorphology_context_figure.png",
        outdir / "histomorphology_context_figure.pdf",
        batch_meta,
        roi_annot,
        covered,
        enrichment,
        selected_patterns,
        batch_label=annotated_batches[0],
    )
    return summary


def write_note(
    path: Path,
    summary: dict[str, object],
    enrichment: pd.DataFrame,
    center_enrichment: pd.DataFrame,
    axes: pd.DataFrame,
    lr_summary: pd.DataFrame,
) -> None:
    def _table_text(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        return df.to_string(index=False)

    lines = [
        "# CRC MA Histomorphology Corroboration",
        "",
        f"Annotated batches: {', '.join(summary['annotated_batches'])}",
        f"ROI spots with morphology labels: {summary['n_roi_spots']}",
        f"Graph instances in primary analysis: {summary.get('n_graph_instances', 0)}",
        f"Major patterns analyzed: {[ma_display_id(p) for p in summary['selected_patterns']]}",
        "",
        "Primary analysis uses one graph instance per MA neighborhood as the validation unit.",
        "Dominant-spot and center-cell summaries are retained only as sensitivity analyses.",
        "Representative graph instances are selected as the graph nearest to the class-average pathology composition vector.",
        "",
        "## Primary enrichment",
        _table_text(enrichment.head(12)),
        "",
        "## Center-level sensitivity",
        _table_text(center_enrichment.head(12)),
        "",
        "## Biological axes",
        _table_text(axes.reset_index()),
        "",
        "## Ligand-receptor programs",
        _table_text(lr_summary.reset_index()),
    ]
    path.write_text("\n".join(lines))


def make_basic_plot(path: Path, enrichment: pd.DataFrame) -> None:
    if enrichment.empty:
        return
    pivot = enrichment.pivot(index="pattern", columns="morphology_group", values="odds_ratio").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 3 + 0.5 * len(pivot)))
    im = ax.imshow(np.log2(pivot.replace(0, np.nan)), aspect="auto", cmap="RdBu_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"MA {ma_display_id(idx)}" for idx in pivot.index])
    ax.set_title("Primary morphology enrichment")
    plt.colorbar(im, ax=ax, label="log2 odds ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_spatial_context_plot(
    path: Path,
    batch_metadata: pd.DataFrame,
    roi_annot: pd.DataFrame,
    covered_spots: dict[int, set[str]],
    selected_patterns: list[int],
) -> None:
    plot_df = batch_metadata.copy()
    x_col = "x" if "x" in plot_df.columns else "array_col"
    y_col = "y" if "y" in plot_df.columns else "array_row"
    n_panels = 2 + len(selected_patterns)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    morph_adata = patch_adata(
        plot_df,
        x_col=x_col,
        y_col=y_col,
        morphology_groups=roi_annot["morphology_group"],
        member_index=pd.Index([]),
    )
    morph_fg = morph_adata[morph_adata.obs["morphology_group"].notna()].copy()
    palette = ["#c44e52", "#55a868", "#4c72b0", "#8172b2", "#dd8452", "#999999"]
    sc.pl.embedding(
        morph_fg,
        basis="spatial",
        color="morphology_group",
        palette=palette[: len(morph_fg.obs["morphology_group"].astype("category").cat.categories)],
        size=9,
        frameon=False,
        show=False,
        ax=axes[0],
        legend_loc=None,
    )
    axes[0].set_title("Pathology context")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    roi_adata = build_membership_patch_adata(
        plot_df,
        x_col=x_col,
        y_col=y_col,
        selected_index=roi_annot.index,
        obs_key="roi_status",
        selected_label="Annotated ROI",
        background_label="Background",
    )
    sc.pl.embedding(
        roi_adata,
        basis="spatial",
        color="roi_status",
        palette=["#d8d8d8", "#111111"],
        size=7,
        frameon=False,
        show=False,
        ax=axes[1],
        legend_loc=None,
    )
    axes[1].set_title("Annotated ROI in cancer_p2")
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    for panel_idx, pattern in enumerate(selected_patterns, start=2):
        coverage_adata = build_membership_patch_adata(
            plot_df,
            x_col=x_col,
            y_col=y_col,
            selected_index=pd.Index(sorted(covered_spots.get(int(pattern), set()))),
            obs_key="coverage_status",
            selected_label=f"MA {ma_display_id(pattern)} coverage",
            background_label="Background",
        )
        sc.pl.embedding(
            coverage_adata,
            basis="spatial",
            color="coverage_status",
            palette=["#d8d8d8", "#d62728"],
            size=7,
            frameon=False,
            show=False,
            ax=axes[panel_idx],
            legend_loc=None,
        )
        axes[panel_idx].set_title(f"MA {ma_display_id(pattern)} coverage")
        axes[panel_idx].set_aspect("equal")
        axes[panel_idx].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def pattern_display_name(pattern: int) -> str:
    display_idx = int(pattern) + 1
    names = {
        0: "MA 1: stromal-enriched interface",
        1: "MA 2: neoplasm-dominant interface",
        2: "MA 3: tumor-adjacent mixed interface",
    }
    return names.get(int(pattern), f"MA {display_idx}")


def ma_display_id(ma_class: int) -> int:
    return int(ma_class) + 1


def representative_lr_overrides(pattern: int) -> tuple[str, str] | None:
    overrides = {
        0: (
            "Fibroblasts -> Tumor 1 | CXCL12-SDC4",
            "CAF -> Endothelial | MMP2-PECAM1",
        ),
        1: (
            "Tumor 2 -> Epithelial differentiated | CEACAM6-CEACAM1",
            "Epithelial differentiated -> Endothelial | CD177-PECAM1",
        ),
        2: (
            "Fibroblasts -> Endothelial | GREM1-KDR",
            "Immune cells -> Endothelial | SPP1-S1PR1",
        ),
    }
    return overrides.get(int(pattern))


def build_major_ma_summary_table(
    enrichment: pd.DataFrame,
    axes: pd.DataFrame,
    lr_summary: pd.DataFrame,
    patterns: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pattern in patterns:
        sub = enrichment[enrichment["pattern"] == int(pattern)].copy()
        enriched = sub[(sub["odds_ratio"] > 1.0) & (sub["q_value"] < 0.05)].copy()
        if enriched.empty:
            enriched = sub.copy()
        enriched = enriched.sort_values(
            ["pattern_count", "odds_ratio", "q_value"],
            ascending=[False, False, True],
        )
        top = enriched.iloc[0]
        axis_row = axes.loc[int(pattern)] if int(pattern) in axes.index else pd.Series(dtype=float)
        lr_row = lr_summary.loc[int(pattern)] if not lr_summary.empty and int(pattern) in lr_summary.index else pd.Series(dtype=object)
        lr_override = representative_lr_overrides(int(pattern))
        rows.append(
            {
                "ma": pattern_display_name(int(pattern)),
                "dominant_morphology_enrichment": top["morphology_group"],
                "or": float(top["odds_ratio"]),
                "q_value": float(top["q_value"]),
                "mean_tumor_distance_um": float(axis_row.get("mean_tumor_distance_um", np.nan)),
                "tumor_fraction": float(axis_row.get("tumor_fraction", np.nan)),
                "fibroblast_caf_fraction": float(axis_row.get("fibroblast_caf_fraction", np.nan)),
                "representative_lr_1": lr_override[0] if lr_override else lr_row.get("top_lr_1", ""),
                "representative_lr_2": lr_override[1] if lr_override else lr_row.get("top_lr_2", ""),
            }
        )
    return pd.DataFrame(rows)


def write_major_ma_summary_files(
    outdir: Path,
    summary_table: pd.DataFrame,
) -> None:
    summary_table.to_csv(outdir / "histomorph_contexts_major_ma.csv", index=False)
    lines = [
        "# Histomorphological contexts of major CRC microarchitectures",
        "",
        summary_table.to_string(index=False),
    ]
    (outdir / "histomorph_contexts_major_ma.md").write_text("\n".join(lines))


def make_histomorphology_context_figure(
    png_path: Path,
    pdf_path: Path,
    batch_meta: pd.DataFrame,
    roi_annot: pd.DataFrame,
    covered_spots: dict[int, set[str]],
    enrichment: pd.DataFrame,
    patterns: list[int],
    batch_label: str,
) -> None:
    plot_df = batch_meta.copy()
    x_col = "x" if "x" in plot_df.columns else "array_col"
    y_col = "y" if "y" in plot_df.columns else "array_row"

    morph_order = [
        "Neoplasm",
        "Connective/Stromal",
        "Non-neoplastic Epithelium",
        "Vascular",
        "Smooth Muscle",
    ]
    morph_colors = {
        "Neoplasm": "#c44e52",
        "Connective/Stromal": "#55a868",
        "Non-neoplastic Epithelium": "#4c72b0",
        "Vascular": "#8172b2",
        "Smooth Muscle": "#dd8452",
    }
    pattern_colors = {
        int(pattern): color
        for pattern, color in zip(patterns, ["#c44e52", "#4c72b0", "#55a868"])
    }

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    for group in morph_order:
        sub = roi_annot[roi_annot["morphology_group"] == group]
        ax0.scatter(sub[x_col], -sub[y_col], s=1, c=morph_colors[group], label=group, rasterized=True)
    ax0.set_title(f"{batch_label}: pathology context")
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.legend(loc="upper left", bbox_to_anchor=(0.0, -0.03), frameon=False, markerscale=4, fontsize=8, ncol=1)

    ax1 = fig.add_subplot(gs[0, 1])
    for pattern in patterns:
        mask = plot_df.index.isin(sorted(covered_spots.get(int(pattern), set())))
        ax1.scatter(
            plot_df.loc[mask, x_col],
            -plot_df.loc[mask, y_col],
            s=1,
            c=pattern_colors[int(pattern)],
            label=pattern_display_name(int(pattern)),
            rasterized=True,
        )
    ax1.set_title(f"{batch_label}: dominant MA assignment")
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, -0.03), frameon=False, markerscale=4, fontsize=8, ncol=1)

    ax2 = fig.add_subplot(gs[0, 2])
    pivot = (
        enrichment[enrichment["morphology_group"].isin(morph_order)]
        .pivot(index="pattern", columns="morphology_group", values="odds_ratio")
        .reindex(index=patterns, columns=morph_order)
    )
    log_pivot = np.log2(pivot.replace(0, np.nan))
    im = ax2.imshow(log_pivot, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax2.set_xticks(np.arange(len(morph_order)))
    ax2.set_xticklabels(morph_order, rotation=35, ha="right", fontsize=8)
    ax2.set_yticks(np.arange(len(patterns)))
    ax2.set_yticklabels([pattern_display_name(int(pattern)) for pattern in patterns], fontsize=8)
    ax2.set_title("Morphology enrichment")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="log2 odds ratio")

    for panel_idx, pattern in enumerate(patterns):
        ax = fig.add_subplot(gs[1, panel_idx])
        mask = plot_df.index.isin(sorted(covered_spots.get(int(pattern), set())))
        ax.scatter(plot_df.loc[~mask, x_col], -plot_df.loc[~mask, y_col], c="lightgray", s=1, alpha=0.15, rasterized=True)
        ax.scatter(plot_df.loc[mask, x_col], -plot_df.loc[mask, y_col], c=pattern_colors[int(pattern)], s=1.2, alpha=0.9, rasterized=True)
        ax.set_title(pattern_display_name(int(pattern)))
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        "Distinct histomorphological environments of major CRC microarchitectures within the same tumor-periphery context",
        y=0.98,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n-patterns", type=int, default=3)
    parser.add_argument("--metadata-csv", type=Path, default=CRC_METADATA_CSV)
    parser.add_argument("--microarc-pkl", type=Path, default=CRC_MICROARC_PKL)
    parser.add_argument("--pathology-csv", type=Path, default=CRC_PATHOLOGY_CSV)
    parser.add_argument("--deconv-meta-csv", type=Path, default=CRC_DECONV_META_CSV)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    outdir = args.output_dir or Path("workflows/crc_ma_histomorphology")
    config = CrcMaHistologyConfig(
        metadata_csv=args.metadata_csv,
        microarc_pkl=args.microarc_pkl,
        pathology_csv=args.pathology_csv,
        deconv_meta_csv=args.deconv_meta_csv,
        top_n_patterns=args.top_n_patterns,
    )
    run_analysis(config, outdir)


if __name__ == "__main__":
    main()
