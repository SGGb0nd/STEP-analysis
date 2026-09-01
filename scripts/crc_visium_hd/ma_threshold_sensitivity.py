"""Evaluate CRC and prostate microarchitecture stability across coherence thresholds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from step.models.niche import MicroArc


@dataclass(frozen=True)
class DatasetProfile:
    dataset_key: str
    dataset_name: str
    metadata_csv: str
    label_col: str
    batch_col: str
    query_mode: str
    radius: float
    n_clusters: int
    clustering_method: str
    query_domain: int | None = None
    cancer_prefix: str | None = None
    query_samples: tuple[str, ...] | None = None
    query_zonation: tuple[str, ...] | None = None
    legacy_absolute_threshold: float | None = None
    n_iter: int = 5
    y_flip: bool = False


CRC_NOTEBOOK_CONFIG = DatasetProfile(
    dataset_key="crc",
    dataset_name="crc_visium_hd_domain9_cancer_true50um",
    metadata_csv="./results/visium-hd/crc5_meta_filtered.csv",
    label_col="cell_type",
    batch_col="batch",
    query_mode="crc_domain",
    radius=50.0,
    n_clusters=3,
    clustering_method="approx",
    query_domain=9,
    cancer_prefix="cancer",
    legacy_absolute_threshold=0.8,
    y_flip=True,
)

SLIDE_SEQ_NOTEBOOK_CONFIG = DatasetProfile(
    dataset_key="slide_seq",
    dataset_name="slide_seq_tumor_microarchitecture",
    metadata_csv="./results/slide-seq/prostate_microarchitecture_metadata.csv",
    label_col="cell1",
    batch_col="sample",
    query_mode="sample_zonation",
    radius=50.0,
    n_clusters=5,
    clustering_method="full",
    query_samples=("Tumor02", "Tumor08"),
    query_zonation=("Tumor",),
    legacy_absolute_threshold=0.4,
    y_flip=False,
)

CRC_SCALE_H5AD = "./results/visium-hd/cancer_tumor_regions.h5ad"


def get_dataset_profile(dataset_key: str) -> DatasetProfile:
    profiles = {
        "crc": CRC_NOTEBOOK_CONFIG,
        "slide_seq": SLIDE_SEQ_NOTEBOOK_CONFIG,
    }
    try:
        return profiles[dataset_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset_key: {dataset_key}") from exc


def bottom_fraction_threshold(values: pd.Series, fraction: float) -> float:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    return float(values.quantile(fraction))


def focal_labels(niche: MicroArc) -> pd.Series:
    return pd.Series(niche.clusters, index=pd.Index(niche.center_indices, name="center"))


def overlap_label_metrics(reference: pd.Series, other: pd.Series) -> dict[str, float | int | None]:
    ref_index = set(reference.index)
    other_index = set(other.index)
    overlap = sorted(ref_index & other_index)
    union = ref_index | other_index

    metrics: dict[str, float | int | None] = {
        "reference_count": len(reference),
        "other_count": len(other),
        "overlap_count": len(overlap),
        "focal_jaccard": (len(overlap) / len(union)) if union else 1.0,
        "ari": None,
        "nmi": None,
    }
    if overlap:
        ref_labels = reference.loc[overlap].tolist()
        other_labels = other.loc[overlap].tolist()
        metrics["ari"] = float(adjusted_rand_score(ref_labels, other_labels))
        metrics["nmi"] = float(normalized_mutual_info_score(ref_labels, other_labels))
    return metrics


def pattern_profile_table(pattern_summaries: dict[int, dict]) -> pd.DataFrame:
    rows = {}
    for pattern_idx, summary in pattern_summaries.items():
        node_sizes = pd.Series(summary["node_sizes"], dtype=float)
        total = float(node_sizes.sum())
        if total > 0:
            node_sizes = node_sizes / total
        rows[pattern_idx] = node_sizes
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()


def pattern_summaries(niche: MicroArc) -> dict[int, dict]:
    return {
        pattern_idx: niche.get_average_pattern(
            pattern_idx,
            return_summary=True,
        )
        for pattern_idx in sorted(np.unique(niche.clusters))
    }


def raw_pattern_profile_table(niche: MicroArc) -> pd.DataFrame:
    rows = {}
    for pattern_idx in sorted(np.unique(niche.clusters)):
        pattern_members = np.where(niche.clusters == pattern_idx)[0]
        neighbor_ind = np.concatenate([niche.neighbor_indices[idx] for idx in pattern_members])
        neighbors = niche.metadata.loc[neighbor_ind]
        freq = neighbors[niche.label_col].value_counts(normalize=True).astype(float)
        rows[int(pattern_idx)] = freq
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()


def dominant_cell_types(profiles: pd.DataFrame, top_n: int = 3) -> dict[int, list[str]]:
    result = {}
    for idx, row in profiles.iterrows():
        ranked = row.sort_values(ascending=False)
        result[int(idx)] = [str(name) for name in ranked.head(top_n).index if ranked.loc[name] > 0]
    return result


def matched_profile_similarity(reference: pd.DataFrame, other: pd.DataFrame) -> dict[str, object]:
    if reference.empty or other.empty:
        return {
            "mean_cosine_similarity": None,
            "matched_pairs": [],
        }
    shared_cols = sorted(set(reference.columns) | set(other.columns))
    ref = reference.reindex(columns=shared_cols, fill_value=0.0)
    oth = other.reindex(columns=shared_cols, fill_value=0.0)
    sim = cosine_similarity(ref.values, oth.values)
    row_ind, col_ind = linear_sum_assignment(-sim)
    pairs = []
    scores = []
    for r, c in zip(row_ind, col_ind):
        score = float(sim[r, c])
        scores.append(score)
        pairs.append(
            {
                "reference_pattern": int(ref.index[r]),
                "other_pattern": int(oth.index[c]),
                "cosine_similarity": score,
            }
        )
    return {
        "mean_cosine_similarity": float(np.mean(scores)) if scores else None,
        "matched_pairs": pairs,
    }


def load_dataset(config: DatasetProfile) -> tuple[pd.DataFrame, pd.Index]:
    metadata = pd.read_csv(config.metadata_csv, index_col=0, low_memory=False)
    if config.query_mode == "crc_domain":
        query = metadata[
            metadata["domain"].eq(config.query_domain)
            & metadata[config.batch_col].str.startswith(config.cancer_prefix)
        ].index
    elif config.query_mode == "sample_zonation":
        query = metadata[
            metadata[config.batch_col].isin(config.query_samples)
            & metadata["zonation"].isin(config.query_zonation)
        ].index
    else:
        raise ValueError(f"Unsupported query_mode: {config.query_mode}")
    return metadata, query


def load_scale_map(h5ad_path: str) -> dict[str, float]:
    adata = sc.read_h5ad(h5ad_path, backed="r")
    try:
        return {
            batch: float(payload["scalefactors"]["microns_per_pixel"])
            for batch, payload in adata.uns["spatial"].items()
        }
    finally:
        adata.file.close()


def convert_metadata_to_um(meta: pd.DataFrame, microns_per_pixel: dict[str, float]) -> pd.DataFrame:
    out = meta.copy()
    out["microns_per_pixel"] = out["batch"].map(microns_per_pixel)
    if out["microns_per_pixel"].isna().any():
        missing = sorted(out.loc[out["microns_per_pixel"].isna(), "batch"].astype(str).unique())
        raise ValueError(f"Missing microns_per_pixel for batches: {missing}")
    out["x"] = out["x"] * out["microns_per_pixel"]
    out["y"] = out["y"] * out["microns_per_pixel"]
    return out


def legacy_threshold_quantile(
    metadata: pd.DataFrame,
    query: pd.Index,
    config: DatasetProfile,
) -> float | None:
    if config.legacy_absolute_threshold is None:
        return None
    query_values = metadata.loc[query, "cosine"]
    return float((query_values < config.legacy_absolute_threshold).mean())


def analyze_threshold(
    metadata: pd.DataFrame,
    query: pd.Index,
    fraction: float,
    config: DatasetProfile,
) -> dict[str, object]:
    query_values = metadata.loc[query, "cosine"]
    threshold = bottom_fraction_threshold(query_values, fraction)
    candidate_indices = metadata.loc[query_values[query_values < threshold].index].index

    niche = MicroArc(
        metadata=metadata,
        label_col=config.label_col,
        batch_col=config.batch_col,
        coherence_threshold=threshold,
        radius=config.radius,
        n_clusters=config.n_clusters,
        n_iter=config.n_iter,
    )
    niche.create_graphs(query).compute_kernel_and_cluster(method=config.clustering_method)

    focal = focal_labels(niche)
    profiles = raw_pattern_profile_table(niche)
    return {
        "fraction": fraction,
        "fraction_percent": int(round(fraction * 100)),
        "coherence_threshold": threshold,
        "candidate_count": int(len(candidate_indices)),
        "ma_graph_count": int(len(niche.graphs)),
        "ma_class_count": int(len(np.unique(niche.clusters))),
        "focal_labels": focal,
        "pattern_profiles": profiles,
        "dominant_cell_types": dominant_cell_types(profiles),
        "candidate_indices": candidate_indices,
    }


def _plot_y(values: pd.Series, y_flip: bool) -> pd.Series:
    return -values if y_flip else values


def _patterns_by_position(centers: pd.DataFrame) -> list[int]:
    centroids = (
        centers.groupby("pattern")[["x", "y"]]
        .mean()
        .reset_index()
        .sort_values(["x", "y"], kind="mergesort")
    )
    return [int(v) for v in centroids["pattern"].tolist()]


def _match_patterns_by_overlap(reference: pd.Series, other: pd.Series) -> dict[int, int]:
    overlap = sorted(set(reference.index) & set(other.index))
    if not overlap:
        return {}

    ref_patterns = sorted(pd.Index(reference.loc[overlap]).unique().tolist())
    other_patterns = sorted(pd.Index(other.loc[overlap]).unique().tolist())
    if not ref_patterns or not other_patterns:
        return {}

    scores = np.zeros((len(other_patterns), len(ref_patterns)), dtype=float)
    for i, other_pattern in enumerate(other_patterns):
        other_mask = other.loc[overlap] == other_pattern
        for j, ref_pattern in enumerate(ref_patterns):
            ref_mask = reference.loc[overlap] == ref_pattern
            scores[i, j] = float((other_mask & ref_mask).sum())

    row_ind, col_ind = linear_sum_assignment(-scores)
    mapping: dict[int, int] = {}
    for row_idx, col_idx in zip(row_ind, col_ind):
        if scores[row_idx, col_idx] <= 0:
            continue
        mapping[int(other_patterns[row_idx])] = int(ref_patterns[col_idx])
    return mapping


def _match_patterns_by_profile(reference: pd.DataFrame, other: pd.DataFrame) -> dict[int, int]:
    if reference.empty or other.empty:
        return {}
    shared_cols = sorted(set(reference.columns) | set(other.columns))
    ref = reference.reindex(columns=shared_cols, fill_value=0.0)
    oth = other.reindex(columns=shared_cols, fill_value=0.0)
    sim = cosine_similarity(oth.values, ref.values)
    row_ind, col_ind = linear_sum_assignment(-sim)
    mapping: dict[int, int] = {}
    for row_idx, col_idx in zip(row_ind, col_ind):
        if sim[row_idx, col_idx] <= 0:
            continue
        mapping[int(oth.index[row_idx])] = int(ref.index[col_idx])
    return mapping


def _reference_aligned_color_maps(
    metadata: pd.DataFrame,
    analyses: dict[int, dict[str, object]],
    reference_percent: int = 20,
) -> dict[int, dict[int, np.ndarray]]:
    reference_labels = analyses[reference_percent]["focal_labels"]
    reference_order = sorted(pd.Index(reference_labels).unique().tolist())
    color_bank = list(plt.cm.tab10(np.linspace(0, 1, 10)))
    color_maps: dict[int, dict[int, np.ndarray]] = {}
    reference_color_map = {
        pattern: color_bank[idx % len(color_bank)]
        for idx, pattern in enumerate(reference_order)
    }
    color_maps[reference_percent] = reference_color_map

    return color_maps


def _analysis_cache_dir(output_dir: Path) -> Path:
    return output_dir / "cache"


def _analysis_cache_paths(cache_dir: Path, percent: int) -> dict[str, Path]:
    stem = f"threshold_{percent}"
    return {
        "meta": cache_dir / f"{stem}_meta.json",
        "focal": cache_dir / f"{stem}_focal_labels.csv",
        "profiles": cache_dir / f"{stem}_pattern_profiles.csv",
        "candidates": cache_dir / f"{stem}_candidate_indices.csv",
    }


def save_analysis_cache(cache_dir: Path, analysis: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    percent = int(analysis["fraction_percent"])
    paths = _analysis_cache_paths(cache_dir, percent)
    meta = {
        "fraction": float(analysis["fraction"]),
        "fraction_percent": percent,
        "coherence_threshold": float(analysis["coherence_threshold"]),
        "candidate_count": int(analysis["candidate_count"]),
        "ma_graph_count": int(analysis["ma_graph_count"]),
        "ma_class_count": int(analysis["ma_class_count"]),
        "dominant_cell_types": analysis["dominant_cell_types"],
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True))
    focal = analysis["focal_labels"].rename("pattern").rename_axis("center").reset_index()
    focal.to_csv(paths["focal"], index=False)
    profiles = analysis["pattern_profiles"].copy()
    profiles.index.name = "pattern"
    profiles.reset_index().to_csv(paths["profiles"], index=False)
    pd.DataFrame({"index": pd.Index(analysis["candidate_indices"]).astype(str)}).to_csv(paths["candidates"], index=False)


def load_analysis_cache(cache_dir: Path, percent: int) -> dict[str, Any] | None:
    paths = _analysis_cache_paths(cache_dir, percent)
    if not all(path.exists() for path in paths.values()):
        return None
    meta = json.loads(paths["meta"].read_text())
    focal_df = pd.read_csv(paths["focal"])
    focal = pd.Series(
        focal_df["pattern"].astype(int).to_numpy(),
        index=pd.Index(focal_df["center"].astype(str).tolist(), name="center"),
    )
    profiles_df = pd.read_csv(paths["profiles"])
    profiles = profiles_df.set_index("pattern")
    profiles.index = profiles.index.astype(int)
    candidate_indices = pd.Index(pd.read_csv(paths["candidates"])["index"].astype(str).tolist())
    return {
        "fraction": float(meta["fraction"]),
        "fraction_percent": int(meta["fraction_percent"]),
        "coherence_threshold": float(meta["coherence_threshold"]),
        "candidate_count": int(meta["candidate_count"]),
        "ma_graph_count": int(meta["ma_graph_count"]),
        "ma_class_count": int(meta["ma_class_count"]),
        "dominant_cell_types": meta["dominant_cell_types"],
        "focal_labels": focal,
        "pattern_profiles": profiles,
        "candidate_indices": candidate_indices,
    }


def plot_candidate_maps(
    metadata: pd.DataFrame,
    query: pd.Index,
    analyses: dict[int, dict[str, object]],
    batch_col: str,
    y_flip: bool,
    output_path: Path,
) -> None:
    query_meta = metadata.loc[query]
    batches = list(query_meta[batch_col].unique())
    thresholds = sorted(analyses)
    fig, axes = plt.subplots(len(thresholds), len(batches), figsize=(5 * len(batches), 4 * len(thresholds)))
    axes = np.atleast_2d(axes)
    for row_idx, threshold in enumerate(thresholds):
        candidate_idx = analyses[threshold]["candidate_indices"]
        for col_idx, batch in enumerate(batches):
            ax = axes[row_idx, col_idx]
            batch_query = query_meta[query_meta[batch_col] == batch]
            batch_candidates = metadata.loc[candidate_idx]
            batch_candidates = batch_candidates[batch_candidates[batch_col] == batch]
            ax.scatter(
                batch_query["x"],
                _plot_y(batch_query["y"], y_flip),
                s=0.2,
                c="lightgray",
                alpha=0.2,
                rasterized=True,
            )
            ax.scatter(
                batch_candidates["x"],
                _plot_y(batch_candidates["y"], y_flip),
                s=1.0,
                c="crimson",
                alpha=0.7,
                rasterized=True,
            )
            ax.set_title(f"{batch} | {threshold}% candidates")
            ax.set_aspect("equal")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_final_maps(
    metadata: pd.DataFrame,
    analyses: dict[int, dict[str, object]],
    batch_col: str,
    y_flip: bool,
    output_path: Path,
) -> None:
    thresholds = sorted(analyses)
    batches = list(metadata.loc[analyses[thresholds[0]]["focal_labels"].index, batch_col].unique())
    fig, axes = plt.subplots(len(thresholds), len(batches), figsize=(5 * len(batches), 4 * len(thresholds)))
    axes = np.atleast_2d(axes)
    color_maps = _reference_aligned_color_maps(metadata=metadata, analyses=analyses)
    for row_idx, threshold in enumerate(thresholds):
        focal = analyses[threshold]["focal_labels"]
        centers = metadata.loc[focal.index].copy()
        centers["pattern"] = focal.values
        ordered_patterns = _patterns_by_position(centers)
        if threshold != 20:
            mapping = _match_patterns_by_profile(
                analyses[20]["pattern_profiles"],
                analyses[threshold]["pattern_profiles"],
            )
            if len(mapping) < len(ordered_patterns):
                overlap_mapping = _match_patterns_by_overlap(analyses[20]["focal_labels"], focal)
                for pattern, ref_pattern in overlap_mapping.items():
                    mapping.setdefault(pattern, ref_pattern)
            base_map = color_maps[20]
            color_bank = list(plt.cm.tab10(np.linspace(0, 1, 10)))
            color_map = {}
            for pattern in ordered_patterns:
                if pattern in mapping:
                    color_map[pattern] = base_map[mapping[pattern]]
            fallback_offset = len(base_map)
            for fallback_idx, pattern in enumerate(ordered_patterns):
                if pattern in color_map:
                    continue
                color_map[pattern] = color_bank[(fallback_offset + fallback_idx) % len(color_bank)]
            color_maps[threshold] = color_map
        else:
            color_map = color_maps[20]
        for col_idx, batch in enumerate(batches):
            ax = axes[row_idx, col_idx]
            batch_centers = centers[centers[batch_col] == batch]
            background = metadata[metadata[batch_col] == batch]
            ax.scatter(
                background["x"],
                _plot_y(background["y"], y_flip),
                s=0.2,
                c="lightgray",
                alpha=0.08,
                rasterized=True,
            )
            batch_patterns = set(batch_centers["pattern"].unique())
            for pattern in ordered_patterns:
                if pattern not in batch_patterns:
                    continue
                subset = batch_centers[batch_centers["pattern"] == pattern]
                ax.scatter(
                    subset["x"],
                    _plot_y(subset["y"], y_flip),
                    s=1.2,
                    c=[color_map[pattern]],
                    alpha=0.8,
                    rasterized=True,
                    label=f"Pattern {pattern}",
                )
            ax.set_title(f"{batch} | {threshold}% MAs")
            ax.set_aspect("equal")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_metadata_for_profile(metadata: pd.DataFrame, config: DatasetProfile) -> pd.DataFrame:
    if config.query_mode == "crc_domain":
        return metadata.loc[metadata[config.batch_col].str.startswith(config.cancer_prefix)]
    if config.query_mode == "sample_zonation":
        return metadata.loc[metadata[config.batch_col].isin(config.query_samples)]
    raise ValueError(f"Unsupported query_mode: {config.query_mode}")


def run_threshold_sensitivity(
    dataset_key: str,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3),
    output_dir: Path | None = None,
    crc_true50um: bool = False,
    scale_h5ad: str = CRC_SCALE_H5AD,
) -> Path:
    config = get_dataset_profile(dataset_key)
    metadata, query = load_dataset(config)
    if dataset_key == "crc" and crc_true50um:
        scale_map = load_scale_map(scale_h5ad)
        metadata = convert_metadata_to_um(metadata, scale_map)
        query = metadata[
            metadata["domain"].eq(config.query_domain)
            & metadata[config.batch_col].str.startswith(config.cancer_prefix)
        ].index
        config = DatasetProfile(
            dataset_key=config.dataset_key,
            dataset_name=f"{config.dataset_name}_true50um",
            metadata_csv=config.metadata_csv,
            label_col=config.label_col,
            batch_col=config.batch_col,
            query_mode=config.query_mode,
            radius=50.0,
            n_clusters=config.n_clusters,
            clustering_method=config.clustering_method,
            query_domain=config.query_domain,
            cancer_prefix=config.cancer_prefix,
            query_samples=config.query_samples,
            query_zonation=config.query_zonation,
            legacy_absolute_threshold=config.legacy_absolute_threshold,
            n_iter=config.n_iter,
            y_flip=config.y_flip,
        )
    query_meta = metadata.loc[query]
    analyses_by_percent: dict[int, dict[str, object]] = {}

    if output_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("workflows") / f"ma_threshold_{dataset_key}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _analysis_cache_dir(output_dir)

    for fraction in fractions:
        percent = int(round(fraction * 100))
        analysis = load_analysis_cache(cache_dir, percent)
        if analysis is None:
            analysis = analyze_threshold(metadata, query, fraction, config)
            save_analysis_cache(cache_dir, analysis)
        analyses_by_percent[percent] = analysis

    reference = analyses_by_percent[20]
    summary = {
        "dataset": config.dataset_name,
        "config": asdict(config),
        "query_size": int(len(query)),
        "query_batches": query_meta[config.batch_col].value_counts().to_dict(),
        "legacy_absolute_threshold": config.legacy_absolute_threshold,
        "legacy_absolute_threshold_equivalent_quantile": legacy_threshold_quantile(
            metadata, query, config
        ),
        "fractions": {},
    }

    for percent, analysis in analyses_by_percent.items():
        focal = analysis["focal_labels"]
        candidate_jaccard = None
        if percent != 20:
            ref_candidates = set(reference["candidate_indices"])
            cur_candidates = set(analysis["candidate_indices"])
            candidate_union = ref_candidates | cur_candidates
            candidate_jaccard = (len(ref_candidates & cur_candidates) / len(candidate_union)) if candidate_union else 1.0

        profile_similarity = matched_profile_similarity(
            reference["pattern_profiles"],
            analysis["pattern_profiles"],
        )
        summary["fractions"][str(percent)] = {
            "coherence_threshold": analysis["coherence_threshold"],
            "candidate_count": analysis["candidate_count"],
            "ma_graph_count": analysis["ma_graph_count"],
            "ma_class_count": analysis["ma_class_count"],
            "dominant_cell_types": analysis["dominant_cell_types"],
            "candidate_jaccard_vs_20": candidate_jaccard,
            "focal_metrics_vs_20": (
                overlap_label_metrics(reference["focal_labels"], focal)
                if percent != 20
                else overlap_label_metrics(focal, focal)
            ),
            "profile_similarity_vs_20": profile_similarity,
        }

    plot_candidate_maps(
        metadata=metadata,
        query=query,
        analyses=analyses_by_percent,
        batch_col=config.batch_col,
        y_flip=config.y_flip,
        output_path=output_dir / "candidate_maps.png",
    )
    plot_final_maps(
        metadata=plot_metadata_for_profile(metadata, config),
        analyses=analyses_by_percent,
        batch_col=config.batch_col,
        y_flip=config.y_flip,
        output_path=output_dir / "ma_maps.png",
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["crc", "slide_seq"], default="crc")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--crc-true50um", action="store_true")
    parser.add_argument("--scale-h5ad", default=CRC_SCALE_H5AD)
    args = parser.parse_args()

    out = run_threshold_sensitivity(
        dataset_key=args.dataset,
        output_dir=args.output_dir,
        crc_true50um=args.crc_true50um,
        scale_h5ad=args.scale_h5ad,
    )
    print(out)


if __name__ == "__main__":
    main()
