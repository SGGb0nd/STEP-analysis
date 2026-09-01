"""Load, align, cluster, and plot external-method results for manuscript datasets."""

from __future__ import annotations

import argparse
import math
import json
import time
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy.external as sce
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans, MiniBatchKMeans

try:
    import harmonypy as hm
except ImportError:  # pragma: no cover - optional dependency
    hm = None


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_RESULTS_DIR = REPO_ROOT / "external_results"
OUTDIR = REPO_ROOT / "workflows" / "external_method_cluster_plots"
BANKSY_RECLUSTER_OUTDIR = REPO_ROOT / "workflows" / "banksy_recluster"
STACKED_OUTDIR = REPO_ROOT / "workflows" / "stacked_external_method_figures"
STEP_MOSTA_DEFAULT_H5AD = (
    REPO_ROOT / "data" / "mosta_e16_test_slim.h5ad"
)

CRC_ORDER = ["cancer_p1", "cancer_p2", "cancer_p5", "normal_p3", "normal_p5"]
MOSTA_ORDER = ["E16.5_E2S1", "E16.5_E2S4", "E16.5_E2S7", "E16.5_E2S10", "E16.5_E2S13"]
SLIDE_ORDER = ["Benign04", "Tumor08", "Tumor02", "HP1"]


def _spatial_to_numpy(spatial) -> np.ndarray:
    if isinstance(spatial, pd.DataFrame):
        return spatial.to_numpy()
    return np.asarray(spatial)


def _point_size(max_points: int) -> float:
    if max_points > 400_000:
        return 0.08
    if max_points > 150_000:
        return 0.12
    if max_points > 60_000:
        return 0.18
    if max_points > 20_000:
        return 0.32
    return 1.2


def _flip_y(data_by_sample: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sample, df in data_by_sample.items():
        frame = df.copy()
        frame["y"] = -frame["y"].to_numpy(dtype=float)
        out[sample] = frame
    return out


def _swap_xy(data_by_sample: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sample, df in data_by_sample.items():
        frame = df.copy()
        old_x = frame["x"].to_numpy(dtype=float)
        old_y = frame["y"].to_numpy(dtype=float)
        frame["x"] = old_y
        frame["y"] = old_x
        out[sample] = frame
    return out


def _make_palette(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    labels = sorted(labels, key=lambda x: (not str(x).isdigit(), str(x)))
    if len(labels) <= 20:
        colors = sns.color_palette("tab20", len(labels))
    elif len(labels) <= 40:
        colors = list(sns.color_palette("tab20", 20)) + list(sns.color_palette("tab20b", len(labels) - 20))
    else:
        colors = sns.color_palette("husl", len(labels))
    return {label: colors[i] for i, label in enumerate(labels)}


def _recluster_frame(
    coords: np.ndarray,
    embedding: np.ndarray,
    n_clusters: int,
    sample_values: np.ndarray,
    sample_order: list[str],
    *,
    random_state: int = 42,
    batch_size: int = 4096,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    labels = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        batch_size=batch_size,
    ).fit_predict(np.asarray(embedding, dtype=float)) + 1
    frame = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": labels.astype(str),
            "sample": sample_values.astype(str),
        }
    )
    return (
        {
            sample: frame.loc[frame["sample"] == sample, ["x", "y", "cluster"]].copy()
            for sample in sample_order
            if sample in frame["sample"].unique()
        },
        sample_order,
    )


def _plot_panels(
    data_by_sample: dict[str, pd.DataFrame],
    sample_order: list[str],
    title: str,
    output_prefix: Path,
    *,
    ncols_override: int | None = None,
    use_legend_panel: bool = False,
) -> None:
    samples = [sample for sample in sample_order if sample in data_by_sample]
    if not samples:
        return

    labels = sorted(
        {
            str(label)
            for sample in samples
            for label in data_by_sample[sample]["cluster"].astype(str).unique().tolist()
        },
        key=lambda x: (not x.isdigit(), x),
    )
    palette = _make_palette(labels)
    max_points = max(len(data_by_sample[sample]) for sample in samples)

    n_panels = len(samples)
    ncols = ncols_override if ncols_override is not None else (3 if n_panels >= 5 else 2 if n_panels >= 3 else n_panels)
    nrows = math.ceil((n_panels + (1 if use_legend_panel else 0)) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, sample in zip(axes, samples):
        df = data_by_sample[sample]
        colors = [palette[str(label)] for label in df["cluster"].astype(str)]
        ax.scatter(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            c=colors,
            s=_point_size(len(df)),
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(sample, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_frame_on(False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5, color=palette[label], label=str(label))
        for label in labels
    ]
    fig.suptitle(title, y=0.995, fontsize=14)
    if use_legend_panel:
        legend_ax = axes[n_panels]
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles,
            loc="center",
            frameon=False,
            ncol=1,
            fontsize=10,
            title="Cluster",
        )
        for ax in axes[n_panels + 1 :]:
            ax.axis("off")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    else:
        for ax in axes[n_panels:]:
            ax.axis("off")
        legend_y = -0.03 if (ncols_override == 5 or ncols == 5) else -0.01
        bottom_pad = 0.10 if (ncols_override == 5 or ncols == 5) else 0.06
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(6, max(3, len(handles) // 2 or 1)),
            frameon=False,
            bbox_to_anchor=(0.5, legend_y),
            fontsize=9,
            title="Cluster",
        )
        fig.tight_layout(rect=(0, bottom_pad, 1, 0.97))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_mosta_method_grid(method_data: dict[str, dict[str, pd.DataFrame]], output_prefix: Path) -> None:
    preferred_order = ["Ground truth", "STEP", "BANKSY", "HERGAST", "NicheCompass"]
    methods = [method for method in preferred_order if method in method_data]
    if not methods:
        return
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    samples = MOSTA_ORDER
    height_ratios: list[float] = []
    for _ in methods:
        height_ratios.extend([1.0, 0.30])

    fig = plt.figure(figsize=(17.5, 3.35 * len(methods)))
    gs = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=len(samples),
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.03,
    )

    for method_idx, method in enumerate(methods):
        data_by_sample = method_data[method]
        labels = sorted(
            {
                str(label)
                for sample in samples
                if sample in data_by_sample
                for label in data_by_sample[sample]["cluster"].astype(str).unique().tolist()
            },
            key=lambda x: (not x.isdigit(), x),
        )
        palette = _make_palette(labels)
        row = method_idx * 2
        for col_idx, sample in enumerate(samples):
            ax = fig.add_subplot(gs[row, col_idx])
            df = data_by_sample.get(sample, pd.DataFrame(columns=["x", "y", "cluster"]))
            if not df.empty:
                colors = [palette[str(label)] for label in df["cluster"].astype(str)]
                ax.scatter(
                    df["x"].to_numpy(dtype=float),
                    df["y"].to_numpy(dtype=float),
                    c=colors,
                    s=_point_size(len(df)),
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_frame_on(False)
            if row == 0:
                ax.set_title(sample, fontsize=11)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    method,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    fontsize=12,
                )

        legend_ax = fig.add_subplot(gs[row + 1, :])
        legend_ax.axis("off")
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=5, color=palette[label], label=str(label))
            for label in labels
        ]
        legend_ax.legend(
            handles=handles,
            loc="center",
            frameon=False,
            ncol=min(9, max(4, math.ceil(len(handles) / 3))),
            fontsize=8,
            title=None,
            columnspacing=0.9,
            handletextpad=0.4,
        )

    fig.savefig(output_prefix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def _load_step_mosta(
    cluster_col: str = "domain",
    path: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    if path is None:
        path = STEP_MOSTA_DEFAULT_H5AD
    if not path.exists():
        return None

    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    if cluster_col not in obs.columns:
        raise KeyError(f"Expected {cluster_col} labels in {path}")

    coords = _spatial_to_numpy(adata.obsm["spatial"])
    samples = obs["batch"].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    labels = obs[cluster_col].astype(str).to_numpy()
    frame = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": labels,
            "sample": samples,
        }
    )
    data_by_sample = {
        sample: frame.loc[frame["sample"] == sample, ["x", "y", "cluster"]].copy()
        for sample in MOSTA_ORDER
        if sample in frame["sample"].unique()
    }
    return _flip_y(data_by_sample), MOSTA_ORDER


def _mosta_ground_truth_cluster_count() -> int:
    path = STEP_MOSTA_DEFAULT_H5AD
    adata = ad.read_h5ad(path, backed="r")
    try:
        if "annotation" in adata.obs.columns:
            return int(adata.obs["annotation"].astype(str).nunique())
        if "domain" in adata.obs.columns:
            return int(adata.obs["domain"].astype(str).nunique())
        raise KeyError(f"No annotation/domain column found in {path}")
    finally:
        adata.file.close()


def _load_banksy_harmony_mosta_reclustered(n_clusters: int) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    base = EXTERNAL_RESULTS_DIR / "banksy_output" / "mosta"
    emb_path = EXTERNAL_RESULTS_DIR / "banksy_output" / "mosta_harmony" / "harmony_embeddings.csv"
    if not emb_path.exists():
        return None

    coord_parts = []
    for sample in MOSTA_ORDER:
        coord_path = base / sample / "spatial_coords.csv"
        if not coord_path.exists():
            continue
        coords = pd.read_csv(coord_path)
        coord_parts.append(
            pd.DataFrame(
                {
                    "barcode": coords["barcode"].astype(str),
                    "sample": coords["sample_id"].astype(str),
                    "x": coords["array_row"].to_numpy(dtype=float),
                    "y": coords["array_col"].to_numpy(dtype=float),
                }
            )
        )
    if not coord_parts:
        return None

    coords_df = pd.concat(coord_parts, ignore_index=True)
    emb = pd.read_csv(emb_path)
    emb["barcode"] = emb["barcode"].astype(str)
    emb["sample"] = emb["sample"].astype(str)
    merged = coords_df.merge(emb, on=["barcode", "sample"], how="inner", validate="one_to_one")
    merged["sample"] = pd.Categorical(merged["sample"], categories=MOSTA_ORDER, ordered=True)
    merged = merged.sort_values(["sample", "barcode"], kind="mergesort").reset_index(drop=True)
    value_cols = [c for c in merged.columns if c.startswith("PC")]
    labels = (
        MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10, batch_size=4096)
        .fit_predict(merged[value_cols].to_numpy(dtype=float))
        + 1
    )
    merged["cluster"] = labels.astype(str)
    data_by_sample = {
        sample: merged.loc[merged["sample"].astype(str) == sample, ["x", "y", "cluster"]].copy()
        for sample in MOSTA_ORDER
        if sample in set(merged["sample"].astype(str))
    }
    return data_by_sample, MOSTA_ORDER


def _load_hergast_harmony_mosta_reclustered(n_clusters: int) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    path = EXTERNAL_RESULTS_DIR / "hergast_harmony" / "MOSTA.harmony.h5ad"
    if not path.exists():
        return None
    adata = ad.read_h5ad(path, backed="r")
    coords = _spatial_to_numpy(adata.obsm["spatial"])
    emb = _spatial_to_numpy(adata.obsm["HERGAST_harmony"])
    samples = adata.obs["sample"].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    adata.file.close()
    return _recluster_frame(coords, emb, n_clusters, samples, MOSTA_ORDER)


def _load_nichecompass_mosta_reclustered(n_clusters: int) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    path = EXTERNAL_RESULTS_DIR / "nichecompass-test" / "mosta_e16_5_adata.h5ad"
    if not path.exists():
        return None
    adata = ad.read_h5ad(path, backed="r")
    coords = _spatial_to_numpy(adata.obsm["spatial"])
    emb = np.asarray(adata.obsm["nichecompass_latent"], dtype=float)
    samples = adata.obs["batch"].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    adata.file.close()
    return _recluster_frame(coords, emb, n_clusters, samples, MOSTA_ORDER)


def build_mosta_stacked(
    output_prefix: Path,
    *,
    use_harmony_recluster: bool = False,
    step_mosta_h5ad: Path | None = None,
) -> None:
    ground_truth = _load_step_mosta(cluster_col="annotation", path=step_mosta_h5ad)
    step_mosta = _load_step_mosta(path=step_mosta_h5ad)
    if use_harmony_recluster:
        n_clusters = _mosta_ground_truth_cluster_count()
        banksy_mosta = _load_banksy_harmony_mosta_reclustered(n_clusters=n_clusters)
        hergast_mosta = _load_hergast_harmony_mosta_reclustered(n_clusters=n_clusters)
        niche_mosta = _load_nichecompass_mosta_reclustered(n_clusters=n_clusters)
    else:
        banksy_mosta = _load_banksy("mosta")
        hergast_mosta = _load_hergast("mosta")
        niche_mosta = _load_nichecompass("mosta")
    hergast_mosta_data = hergast_mosta[0] if hergast_mosta is not None else {}
    if ground_truth is None or step_mosta is None or banksy_mosta is None or niche_mosta is None or not hergast_mosta_data:
        raise RuntimeError("Missing MOSTA data for stacked external-method grid")
    _plot_mosta_method_grid(
        {
            "Ground truth": ground_truth[0],
            "STEP": step_mosta[0],
            "BANKSY": banksy_mosta[0],
            "HERGAST": hergast_mosta_data,
            "NicheCompass": niche_mosta[0],
        },
        output_prefix,
    )


def build_mosta_ground_truth_step(
    output_prefix: Path,
    *,
    step_mosta_h5ad: Path | None = None,
) -> None:
    ground_truth = _load_step_mosta(cluster_col="annotation", path=step_mosta_h5ad)
    step_mosta = _load_step_mosta(path=step_mosta_h5ad)
    if ground_truth is None or step_mosta is None:
        raise RuntimeError("Missing MOSTA data for Ground truth vs STEP comparison")
    _plot_mosta_method_grid(
        {
            "Ground truth": ground_truth[0],
            "STEP": step_mosta[0],
        },
        output_prefix,
    )


def recluster_embeddings_per_sample(
    embeddings_by_sample: dict[str, np.ndarray],
    n_clusters: int,
    *,
    random_state: int = 42,
    batch_size: int = 4096,
) -> dict[str, np.ndarray]:
    labels_by_sample: dict[str, np.ndarray] = {}
    for sample, emb in embeddings_by_sample.items():
        labels = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            batch_size=batch_size,
        ).fit_predict(np.asarray(emb, dtype=float))
        labels_by_sample[sample] = labels + 1
    return labels_by_sample


def harmony_recluster_embeddings(
    embeddings_by_sample: dict[str, np.ndarray],
    n_clusters: int,
    *,
    random_state: int = 42,
    batch_size: int = 4096,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    samples = list(embeddings_by_sample)
    arrays = [np.asarray(embeddings_by_sample[sample], dtype=float) for sample in samples]
    counts = [arr.shape[0] for arr in arrays]
    emb_all = np.vstack(arrays)
    batch_labels = np.concatenate([[sample] * count for sample, count in zip(samples, counts)])
    corrected = None
    if hm is not None:
        harmony_out = hm.run_harmony(emb_all, pd.DataFrame({"sample": batch_labels}), ["sample"])
        corrected = np.asarray(harmony_out.Z_corr, dtype=float)
    else:  # pragma: no cover - fallback path
        adata = ad.AnnData(np.zeros((emb_all.shape[0], 1), dtype=float), obs=pd.DataFrame({"sample": batch_labels}))
        adata.obsm["X_banksy"] = emb_all
        sce.pp.harmony_integrate(adata, key="sample", basis="X_banksy", adjusted_basis="X_banksy_harmony")
        corrected = np.asarray(adata.obsm["X_banksy_harmony"], dtype=float)
    labels = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        batch_size=batch_size,
    ).fit_predict(corrected) + 1

    labels_by_sample: dict[str, np.ndarray] = {}
    start = 0
    for sample, count in zip(samples, counts):
        labels_by_sample[sample] = labels[start : start + count]
        start += count
    return labels_by_sample, corrected


def _load_banksy_embeddings(dataset: str) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], list[str]] | None:
    if dataset == "crc":
        base = EXTERNAL_RESULTS_DIR / "banksy_output" / "008um"
        sample_order = CRC_ORDER
    elif dataset == "slide_seq":
        base = EXTERNAL_RESULTS_DIR / "banksy_output" / "slideseq"
        sample_order = SLIDE_ORDER
    else:
        return None

    coords_by_sample: dict[str, pd.DataFrame] = {}
    embeddings_by_sample: dict[str, np.ndarray] = {}
    for sample in sample_order:
        emb_path = base / sample / "banksy_embeddings.csv"
        coord_path = base / sample / "spatial_coords.csv"
        if not emb_path.exists() or not coord_path.exists():
            continue
        emb = pd.read_csv(emb_path).select_dtypes(include=[np.number]).to_numpy()
        coords = pd.read_csv(coord_path)
        if len(emb) != len(coords):
            raise ValueError(f"Row mismatch for BANKSY {sample}: {len(emb)} embeddings vs {len(coords)} coords")
        coords_by_sample[sample] = pd.DataFrame(
            {
                "x": coords.iloc[:, 0].to_numpy(),
                "y": coords.iloc[:, 1].to_numpy(),
            }
        )
        embeddings_by_sample[sample] = emb
    return coords_by_sample, embeddings_by_sample, sample_order


def _labels_to_plot_frames(
    coords_by_sample: dict[str, pd.DataFrame],
    labels_by_sample: dict[str, np.ndarray],
    sample_order: list[str],
) -> dict[str, pd.DataFrame]:
    data_by_sample: dict[str, pd.DataFrame] = {}
    for sample in sample_order:
        if sample not in coords_by_sample or sample not in labels_by_sample:
            continue
        coords = coords_by_sample[sample]
        labels = np.asarray(labels_by_sample[sample])
        if len(coords) != len(labels):
            raise ValueError(f"Row mismatch after clustering for {sample}: {len(coords)} coords vs {len(labels)} labels")
        df = coords.copy()
        df["cluster"] = labels.astype(int).astype(str)
        data_by_sample[sample] = df
    return data_by_sample


def write_plot_csv(
    data_by_sample: dict[str, pd.DataFrame],
    sample_order: list[str],
    output_csv: Path,
) -> None:
    combined = []
    for sample in sample_order:
        if sample not in data_by_sample:
            continue
        df = data_by_sample[sample].copy()
        df["sample"] = sample
        combined.append(df[["sample", "x", "y", "cluster"]])
    if not combined:
        raise ValueError(f"No sample data available for {output_csv}")
    pd.concat(combined, ignore_index=True).to_csv(output_csv, index=False)


def _read_plot_csv(input_csv: Path, sample_order: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    if not input_csv.exists():
        return None
    frame = pd.read_csv(input_csv)
    return (
        {
            sample: frame.loc[frame["sample"] == sample, ["x", "y", "cluster"]].copy()
            for sample in sample_order
            if sample in frame["sample"].unique()
        },
        sample_order,
    )


def _run_banksy_recluster_mode(
    dataset: str,
    n_clusters: int,
    mode: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]] | None:
    loaded = _load_banksy_embeddings(dataset)
    if loaded is None:
        return None
    coords_by_sample, embeddings_by_sample, sample_order = loaded
    if not embeddings_by_sample:
        return None

    start = time.time()
    corrected_dim = None
    if mode == "single_slice":
        labels_by_sample = recluster_embeddings_per_sample(embeddings_by_sample, n_clusters=n_clusters)
    elif mode == "harmony_all_slice":
        labels_by_sample, corrected = harmony_recluster_embeddings(embeddings_by_sample, n_clusters=n_clusters)
        corrected_dim = int(corrected.shape[1])
    else:
        raise ValueError(f"Unknown BANKSY mode: {mode}")
    elapsed_sec = time.time() - start

    data_by_sample = _labels_to_plot_frames(coords_by_sample, labels_by_sample, sample_order)
    summary = {
        "dataset": dataset,
        "mode": mode,
        "n_clusters": n_clusters,
        "sample_order": [sample for sample in sample_order if sample in data_by_sample],
        "sample_sizes": {sample: int(len(data_by_sample[sample])) for sample in data_by_sample},
        "unique_clusters_per_sample": {
            sample: int(pd.Index(data_by_sample[sample]["cluster"]).nunique()) for sample in data_by_sample
        },
        "unique_clusters_global": int(
            pd.Index(
                pd.concat([data_by_sample[sample]["cluster"] for sample in data_by_sample], ignore_index=True)
            ).nunique()
        ),
        "total_spots": int(sum(len(df) for df in data_by_sample.values())),
        "latent_dim": int(next(iter(embeddings_by_sample.values())).shape[1]),
        "corrected_dim": corrected_dim,
        "elapsed_sec": float(elapsed_sec),
        "elapsed_min": float(elapsed_sec / 60.0),
    }
    return data_by_sample, summary


def _write_banksy_recluster_outputs() -> None:
    BANKSY_RECLUSTER_OUTDIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    configs = [
        ("crc", 12, "single_slice", "BANKSY single-slice clusters on CRC Visium HD", "banksy_crc_single_slice_k12"),
        ("slide_seq", 5, "single_slice", "BANKSY single-slice clusters on slide-seq", "banksy_slide_seq_single_slice_k5"),
        ("crc", 12, "harmony_all_slice", "BANKSY Harmony-integrated clusters on CRC Visium HD", "banksy_crc_harmony_k12"),
        ("slide_seq", 5, "harmony_all_slice", "BANKSY Harmony-integrated clusters on slide-seq", "banksy_slide_seq_harmony_k5"),
    ]

    for dataset, n_clusters, mode, title, prefix in configs:
        result = _run_banksy_recluster_mode(dataset, n_clusters, mode)
        if result is None:
            continue
        data_by_sample, summary = result
        _plot_panels(
            data_by_sample,
            summary["sample_order"],
            title,
            BANKSY_RECLUSTER_OUTDIR / prefix,
            ncols_override=3 if dataset == "crc" else None,
        )
        write_plot_csv(data_by_sample, summary["sample_order"], BANKSY_RECLUSTER_OUTDIR / f"{prefix}.csv")
        summaries.append(summary)

    (BANKSY_RECLUSTER_OUTDIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    readme = (
        "BANKSY reclustered comparison outputs.\n\n"
        "Modes:\n"
        "- single_slice: samplewise MiniBatchKMeans on existing BANKSY embeddings\n"
        "- harmony_all_slice: all-slice Harmony integration with sample as batch, then global MiniBatchKMeans\n\n"
        "Target cluster counts:\n"
        "- CRC Visium HD: 12\n"
        "- slide-seq: 5\n\n"
        "Note:\n"
        "The Harmony-integrated outputs enforce the target cluster count globally across all slices. "
        "A given slice may display fewer than the global count if one joint cluster is absent from that slice.\n"
    )
    (BANKSY_RECLUSTER_OUTDIR / "README.md").write_text(readme)


def _load_nichecompass(dataset: str) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    base = EXTERNAL_RESULTS_DIR / "nichecompass-test"
    if dataset == "crc":
        adata_path = base / "visium_hd_crc_adata.h5ad"
        cluster_path = base / "visium_hd_crc_clusters_k10.csv"
        sample_order = CRC_ORDER
    elif dataset == "mosta":
        adata_path = base / "mosta_e16_5_adata.h5ad"
        cluster_path = base / "mosta_e16_5_clusters_k21.csv"
        sample_order = MOSTA_ORDER
    else:
        return None

    adata = ad.read_h5ad(adata_path, backed="r")
    coords = _spatial_to_numpy(adata.obsm["spatial"])
    obs_names = pd.Index(adata.obs_names.astype(str))
    batches = adata.obs["batch"].astype(str).to_numpy()

    clusters = pd.read_csv(cluster_path).set_index("barcode")
    labels = clusters.reindex(obs_names)["cluster"].astype("Int64")
    valid = labels.notna().to_numpy()
    frame = pd.DataFrame(
        {
            "x": coords[valid, 0],
            "y": coords[valid, 1],
            "cluster": labels[valid].astype(int).astype(str).to_numpy(),
            "sample": np.asarray(batches)[valid],
        }
    )
    return {sample: frame.loc[frame["sample"] == sample, ["x", "y", "cluster"]].copy() for sample in sample_order if sample in frame["sample"].unique()}, sample_order


def _load_nichecompass_crc_reclustered(n_clusters: int = 12) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    path = EXTERNAL_RESULTS_DIR / "nichecompass-test" / "visium_hd_crc_adata.h5ad"
    if not path.exists():
        return None
    adata = ad.read_h5ad(path, backed="r")
    coords = _spatial_to_numpy(adata.obsm["spatial"])
    emb = _spatial_to_numpy(adata.obsm["nichecompass_latent"])
    samples = adata.obs["batch"].astype(str).to_numpy()
    return _recluster_frame(coords, emb, n_clusters, samples, CRC_ORDER)


def _load_nichecompass_slide_seq_reclustered(n_clusters: int = 5) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    path = EXTERNAL_RESULTS_DIR / "nichecompass-test" / "slideseq_adata.h5ad"
    if not path.exists():
        return None

    adata = ad.read_h5ad(path, backed="r")
    samples = adata.obs["sample"].astype(str).to_numpy()
    mask = np.isin(samples, SLIDE_ORDER)
    if not np.any(mask):
        return None

    coords = _spatial_to_numpy(adata.obsm["spatial"])[mask]
    emb = _spatial_to_numpy(adata.obsm["nichecompass_latent"])[mask]
    sample_values = samples[mask]
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10, batch_size=4096).fit_predict(emb) + 1

    frame = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": labels.astype(str),
            "sample": sample_values,
        }
    )
    return {
        sample: frame.loc[frame["sample"] == sample, ["x", "y", "cluster"]].copy()
        for sample in SLIDE_ORDER
        if sample in frame["sample"].unique()
    }, SLIDE_ORDER


def _load_banksy(dataset: str) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    if dataset == "crc":
        base = EXTERNAL_RESULTS_DIR / "banksy_output" / "008um"
        sample_order = CRC_ORDER
        kmeans_name = "kmeans_clusters_k15.csv"
    elif dataset == "mosta":
        base = EXTERNAL_RESULTS_DIR / "banksy_output" / "mosta"
        sample_order = MOSTA_ORDER
        kmeans_name = "kmeans_clusters_k15.csv"
    elif dataset == "slide_seq":
        base = EXTERNAL_RESULTS_DIR / "banksy_output" / "slideseq"
        sample_order = SLIDE_ORDER
        kmeans_name = "kmeans_clusters_k10.csv"
    else:
        return None

    data_by_sample: dict[str, pd.DataFrame] = {}
    for sample in sample_order:
        cluster_path = base / sample / kmeans_name
        coord_path = base / sample / "spatial_coords.csv"
        if not cluster_path.exists() or not coord_path.exists():
            continue
        coords = pd.read_csv(coord_path)
        clusters = pd.read_csv(cluster_path)
        if len(coords) != len(clusters):
            raise ValueError(f"Row mismatch for BANKSY {sample}: {len(coords)} coords vs {len(clusters)} clusters")
        df = pd.DataFrame(
            {
                "x": coords.iloc[:, 0].to_numpy(),
                "y": coords.iloc[:, 1].to_numpy(),
                "cluster": clusters["cluster"].astype(int).astype(str).to_numpy(),
            }
        )
        data_by_sample[sample] = df
    return data_by_sample, sample_order


def _load_banksy_slide_seq_reclustered(n_clusters: int = 5) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    base = EXTERNAL_RESULTS_DIR / "banksy_output" / "slideseq"
    sample_frames: list[pd.DataFrame] = []
    embeddings: list[np.ndarray] = []

    for sample in SLIDE_ORDER:
        emb_path = base / sample / "banksy_embeddings.csv"
        coord_path = base / sample / "spatial_coords.csv"
        if not emb_path.exists() or not coord_path.exists():
            continue
        emb = pd.read_csv(emb_path).select_dtypes(include=[np.number])
        coords = pd.read_csv(coord_path)
        if len(emb) != len(coords):
            raise ValueError(f"Row mismatch for BANKSY {sample}: {len(emb)} embeddings vs {len(coords)} coords")
        sample_frames.append(
            pd.DataFrame(
                {
                    "x": coords.iloc[:, 0].to_numpy(),
                    "y": coords.iloc[:, 1].to_numpy(),
                    "sample": sample,
                }
            )
        )
        embeddings.append(emb.to_numpy())

    if not sample_frames:
        return None

    frame = pd.concat(sample_frames, ignore_index=True)
    emb_all = np.vstack(embeddings)
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10, batch_size=4096).fit_predict(emb_all) + 1
    frame["cluster"] = labels.astype(str)
    return {
        sample: frame.loc[frame["sample"] == sample, ["x", "y", "cluster"]].copy()
        for sample in SLIDE_ORDER
        if sample in frame["sample"].unique()
    }, SLIDE_ORDER


def _load_hergast(dataset: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if dataset == "crc":
        base = EXTERNAL_RESULTS_DIR / "hergast_slim" / "VisiumHD"
        sample_order = CRC_ORDER
        suffix = "_square_008um.HERGAST.slim.h5ad"
    elif dataset == "mosta":
        base = EXTERNAL_RESULTS_DIR / "hergast_slim" / "MOSTA"
        sample_order = MOSTA_ORDER
        suffix = ".HERGAST.slim.h5ad"
    else:
        base = EXTERNAL_RESULTS_DIR / "HERGAST" / "HERGAST_SlideSeq_4samples" / "22022026_155127"
        sample_order = SLIDE_ORDER
        suffix = ".HERGAST.full_kmeans.h5ad"

    data_by_sample: dict[str, pd.DataFrame] = {}
    for sample in sample_order:
        path = base / f"{sample}{suffix}"
        if not path.exists():
            continue
        adata = ad.read_h5ad(path, backed="r")
        coords = _spatial_to_numpy(adata.obsm["spatial"])
        if "kmeans" not in adata.obs.columns:
            raise KeyError(f"Expected kmeans labels in {path}")
        labels = adata.obs["kmeans"].astype(int).astype(str).to_numpy()
        data_by_sample[sample] = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    return data_by_sample, sample_order


def _load_hergast_crc_reclustered(n_clusters: int = 12) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    base = EXTERNAL_RESULTS_DIR / "hergast_slim" / "VisiumHD"
    frames = []
    embeddings = []
    for sample in CRC_ORDER:
        path = base / f"{sample}_square_008um.HERGAST.slim.h5ad"
        if not path.exists():
            continue
        adata = ad.read_h5ad(path, backed="r")
        coords = _spatial_to_numpy(adata.obsm["spatial"])
        emb = _spatial_to_numpy(adata.obsm["HERGAST"])
        frames.append(pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "sample": sample}))
        embeddings.append(emb)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    emb_all = np.vstack(embeddings)
    return _recluster_frame(
        frame[["x", "y"]].to_numpy(dtype=float),
        emb_all,
        n_clusters,
        frame["sample"].to_numpy(),
        CRC_ORDER,
    )


def _load_hergast_slide_seq_reclustered(n_clusters: int = 5) -> tuple[dict[str, pd.DataFrame], list[str]] | None:
    base = EXTERNAL_RESULTS_DIR / "HERGAST" / "HERGAST_SlideSeq_4samples" / "22022026_155127"
    frames: list[pd.DataFrame] = []
    embeddings: list[np.ndarray] = []

    for sample in SLIDE_ORDER:
        path = base / f"{sample}.HERGAST.full_kmeans.h5ad"
        if not path.exists():
            continue
        adata = ad.read_h5ad(path, backed="r")
        coords = _spatial_to_numpy(adata.obsm["spatial"])
        emb = _spatial_to_numpy(adata.obsm["HERGAST"])
        frames.append(pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "sample": sample}))
        embeddings.append(emb)

    if not frames:
        return None

    frame = pd.concat(frames, ignore_index=True)
    emb_all = np.vstack(embeddings)
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10, batch_size=4096).fit_predict(emb_all) + 1
    frame["cluster"] = labels.astype(str)
    return {
        sample: frame.loc[frame["sample"] == sample, ["x", "y", "cluster"]].copy()
        for sample in SLIDE_ORDER
        if sample in frame["sample"].unique()
    }, SLIDE_ORDER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["all", "mosta_stacked", "mosta_gt_step"],
        default="all",
    )
    parser.add_argument(
        "--mosta-output-prefix",
        type=Path,
        default=STACKED_OUTDIR / "mosta_external_methods_stacked",
    )
    parser.add_argument(
        "--mosta-gt-step-output-prefix",
        type=Path,
        default=STACKED_OUTDIR / "mosta_ground_truth_step",
    )
    parser.add_argument("--mosta-harmony-recluster", action="store_true")
    parser.add_argument("--step-mosta-h5ad", type=Path, default=None)
    args = parser.parse_args()

    if args.target == "mosta_stacked":
        build_mosta_stacked(
            args.mosta_output_prefix,
            use_harmony_recluster=args.mosta_harmony_recluster,
            step_mosta_h5ad=args.step_mosta_h5ad,
        )
        return
    if args.target == "mosta_gt_step":
        build_mosta_ground_truth_step(
            args.mosta_gt_step_output_prefix,
            step_mosta_h5ad=args.step_mosta_h5ad,
        )
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)
    STACKED_OUTDIR.mkdir(parents=True, exist_ok=True)

    niche_crc = _load_nichecompass_crc_reclustered(n_clusters=12)
    if niche_crc is not None:
        niche_crc_data = _flip_y(niche_crc[0])
        _plot_panels(niche_crc_data, niche_crc[1], "NicheCompass clusters on CRC Visium HD", OUTDIR / "nichecompass_crc_clusters")

    banksy_crc = _load_banksy("crc")
    if banksy_crc is not None:
        _plot_panels(banksy_crc[0], banksy_crc[1], "BANKSY clusters on CRC Visium HD", OUTDIR / "banksy_crc_clusters")

    banksy_slide = _load_banksy_slide_seq_reclustered(n_clusters=5)
    if banksy_slide is not None:
        write_plot_csv(banksy_slide[0], banksy_slide[1], OUTDIR / "banksy_slide_seq_clusters.csv")

    niche_slide = _load_nichecompass_slide_seq_reclustered(n_clusters=5)
    if niche_slide is not None:
        write_plot_csv(niche_slide[0], niche_slide[1], OUTDIR / "nichecompass_slide_seq_clusters.csv")

    hergast_crc = _load_hergast_crc_reclustered(n_clusters=12)
    if hergast_crc is not None:
        hergast_crc_data = _flip_y(hergast_crc[0])
        _plot_panels(hergast_crc_data, hergast_crc[1], "HERGAST clusters on CRC Visium HD", OUTDIR / "hergast_crc_clusters")

    hergast_slide = _load_hergast_slide_seq_reclustered(n_clusters=5)
    if hergast_slide is not None:
        write_plot_csv(hergast_slide[0], hergast_slide[1], OUTDIR / "hergast_slide_seq_clusters.csv")

    build_mosta_stacked(
        args.mosta_output_prefix,
        use_harmony_recluster=args.mosta_harmony_recluster,
        step_mosta_h5ad=args.step_mosta_h5ad,
    )
    build_mosta_ground_truth_step(
        args.mosta_gt_step_output_prefix,
        step_mosta_h5ad=args.step_mosta_h5ad,
    )

    banksy_crc_single = _read_plot_csv(BANKSY_RECLUSTER_OUTDIR / "banksy_crc_single_slice_k12.csv", CRC_ORDER)
    if banksy_crc_single is not None:
        banksy_crc_single_data = _swap_xy(banksy_crc_single[0])
        _plot_panels(
            banksy_crc_single_data,
            banksy_crc_single[1],
            "BANKSY single-slice clusters on CRC Visium HD",
            BANKSY_RECLUSTER_OUTDIR / "banksy_crc_single_slice_k12",
            ncols_override=3,
        )

    banksy_crc_harmony = _read_plot_csv(BANKSY_RECLUSTER_OUTDIR / "banksy_crc_harmony_k12.csv", CRC_ORDER)
    if banksy_crc_harmony is not None:
        banksy_crc_harmony_data = _swap_xy(banksy_crc_harmony[0])
        _plot_panels(
            banksy_crc_harmony_data,
            banksy_crc_harmony[1],
            "BANKSY Harmony-integrated clusters on CRC Visium HD",
            BANKSY_RECLUSTER_OUTDIR / "banksy_crc_harmony_k12",
            ncols_override=3,
        )


if __name__ == "__main__":
    main()
