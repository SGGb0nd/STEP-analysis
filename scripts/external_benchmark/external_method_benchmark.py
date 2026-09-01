"""Compute external-method quality metrics for the CRC and MOSTA comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anndata as ad
import numpy as np
import pandas as pd
from scib_metrics import ilisi_knn
from scib_metrics.metrics._silhouette import silhouette_batch
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

try:
    from .plot_external_method_clusters import (
        BANKSY_RECLUSTER_OUTDIR,
        CRC_ORDER,
        MOSTA_ORDER,
        _mosta_ground_truth_cluster_count,
        _load_banksy,
        _load_banksy_embeddings,
        _load_hergast,
        _load_nichecompass,
        _read_plot_csv,
    )
except ImportError:
    from plot_external_method_clusters import (
        BANKSY_RECLUSTER_OUTDIR,
        CRC_ORDER,
        MOSTA_ORDER,
        _mosta_ground_truth_cluster_count,
        _load_banksy,
        _load_banksy_embeddings,
        _load_hergast,
        _load_nichecompass,
        _read_plot_csv,
    )

from scripts.module_ablation.step_module_ablation_sim import (
    _cal_chaos,
    _cal_pas,
)


def cal_batch_asw(embed: np.ndarray, batch: np.ndarray, bio: np.ndarray) -> float:
    """Compute scIB silhouette batch using biological labels and batch labels."""
    return float(silhouette_batch(embed, bio, batch))


def cal_batch_ilisi(embed: np.ndarray, batch: np.ndarray, n_neighbors: int = 90) -> float:
    """Compute scaled iLISI from the integrated embedding."""
    neighbors = pynndescent(
        embed,
        n_neighbors=min(n_neighbors, len(embed) - 1),
        random_state=0,
        n_jobs=1,
    )
    return float(ilisi_knn(neighbors, batch, scale=True))


def cal_chaos(labels: np.ndarray, locs: np.ndarray) -> float:
    return float(_cal_chaos(locs, labels))


def cal_pas(labels: np.ndarray, locs: np.ndarray, k: int = 10) -> float:
    return float(_cal_pas(locs, labels, k=k))


CRC_META = REPO_ROOT / "results" / "visium-hd" / "crc5_meta_filtered.csv"
MOSTA_REF = REPO_ROOT / "external_results" / "nichecompass-test" / "mosta_e16_5_adata.h5ad"
NICHE_CRC = REPO_ROOT / "external_results" / "nichecompass-test" / "visium_hd_crc_adata.h5ad"
NICHE_CRC_CLUSTERS = REPO_ROOT / "external_results" / "nichecompass-test" / "visium_hd_crc_clusters_k10.csv"
BANKSY_HARMONY_DIR = REPO_ROOT / "external_results" / "banksy_output"
HERGAST_HARMONY_DIR = REPO_ROOT / "external_results" / "hergast_harmony"
SCIB_BATCH_SCRIPT = Path(__file__).with_name("compute_mosta_scib_batch_metrics.py")


def _round_frame(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    out["x_round"] = out["x"].round(ndigits)
    out["y_round"] = out["y"].round(ndigits)
    return out


def best_matching_cluster(cluster_labels: pd.Series | np.ndarray, ref_mask: pd.Series | np.ndarray) -> dict[str, float | str]:
    labels = pd.Series(np.asarray(cluster_labels).astype(str))
    mask = pd.Series(np.asarray(ref_mask).astype(bool))
    best: dict[str, float | str] | None = None
    for cluster in labels.unique():
        pred = labels == cluster
        tp = int((pred & mask).sum())
        fp = int((pred & ~mask).sum())
        fn = int((~pred & mask).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        jaccard = tp / int((pred | mask).sum()) if int((pred | mask).sum()) else 0.0
        cand = {
            "cluster": str(cluster),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "jaccard": float(jaccard),
        }
        if best is None or (cand["f1"], cand["jaccard"], cand["precision"]) > (
            best["f1"],
            best["jaccard"],
            best["precision"],
        ):
            best = cand
    assert best is not None
    return best


def _nearest_distance(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if len(src) == 0 or len(dst) == 0:
        return np.full(len(src), np.inf, dtype=float)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(dst)
    dist, _ = nn.kneighbors(src)
    return dist[:, 0]


def proximity_match_cluster(frame: pd.DataFrame, anchors: pd.DataFrame, radius: float) -> dict[str, float | str]:
    best: dict[str, float | str] | None = None
    for cluster in frame["cluster"].astype(str).unique():
        sub = frame.loc[frame["cluster"].astype(str) == cluster, ["sample", "x", "y"]].copy()
        cluster_dists = []
        anchor_dists = []
        for sample, anchor_sub in anchors.groupby("sample"):
            cluster_sub = sub.loc[sub["sample"].astype(str) == str(sample), ["x", "y"]].to_numpy(dtype=float)
            anchor_coords = anchor_sub[["x", "y"]].to_numpy(dtype=float)
            if len(cluster_sub):
                cluster_dists.append(_nearest_distance(cluster_sub, anchor_coords))
            if len(anchor_coords):
                anchor_dists.append(_nearest_distance(anchor_coords, cluster_sub))
        cluster_nn = np.concatenate(cluster_dists) if cluster_dists else np.array([], dtype=float)
        anchor_nn = np.concatenate(anchor_dists) if anchor_dists else np.array([], dtype=float)
        cluster_within = float(np.mean(cluster_nn <= radius)) if len(cluster_nn) else 0.0
        anchor_coverage = float(np.mean(anchor_nn <= radius)) if len(anchor_nn) else 0.0
        mean_cluster_distance = float(np.mean(cluster_nn)) if len(cluster_nn) else float("inf")
        mean_anchor_distance = float(np.mean(anchor_nn)) if len(anchor_nn) else float("inf")
        score = (
            2 * cluster_within * anchor_coverage / (cluster_within + anchor_coverage)
            if (cluster_within + anchor_coverage)
            else 0.0
        )
        cand = {
            "cluster": str(cluster),
            "cluster_within_r": cluster_within,
            "anchor_coverage_within_r": anchor_coverage,
            "mean_cluster_anchor_distance": mean_cluster_distance,
            "mean_anchor_cluster_distance": mean_anchor_distance,
            "score": float(score),
        }
        if best is None or (
            cand["score"],
            cand["anchor_coverage_within_r"],
            cand["cluster_within_r"],
            -cand["mean_anchor_cluster_distance"],
        ) > (
            best["score"],
            best["anchor_coverage_within_r"],
            best["cluster_within_r"],
            -best["mean_anchor_cluster_distance"],
        ):
            best = cand
    assert best is not None
    return best


def cancer_fraction_for_cluster(frame: pd.DataFrame, cluster: str) -> float:
    sub = frame.loc[frame["cluster"].astype(str) == str(cluster)]
    if len(sub) == 0:
        return float("nan")
    return float(sub["sample"].astype(str).str.startswith("cancer").mean())


def crc_domain_match_row(method: str, frame: pd.DataFrame, ref: pd.DataFrame, radius: float) -> dict[str, float | str]:
    merged = _merge_with_reference(frame, ref)
    if merged.empty:
        return {"method": method}
    row: dict[str, float | str] = {"method": method}
    for context in ["tumor", "tumor_periphery", "lamina_propria"]:
        anchor = merged.loc[merged["reference"] == context, ["sample", "x", "y"]].copy()
        best = proximity_match_cluster(merged[["sample", "x", "y", "cluster"]], anchor, radius=radius)
        row[f"{context}_cluster"] = best["cluster"]
        row[f"{context}_score"] = best["score"]
        row[f"{context}_cluster_within_r"] = best["cluster_within_r"]
        row[f"{context}_anchor_coverage_within_r"] = best["anchor_coverage_within_r"]
        row[f"{context}_mean_cluster_anchor_distance"] = best["mean_cluster_anchor_distance"]
        row[f"{context}_mean_anchor_cluster_distance"] = best["mean_anchor_cluster_distance"]
        if context in {"tumor", "tumor_periphery"}:
            row[f"{context}_cancer_fraction"] = cancer_fraction_for_cluster(merged, str(best["cluster"]))
    return row


def continuity_metrics(cluster_labels: pd.Series | np.ndarray, coords: np.ndarray) -> dict[str, float]:
    labels = np.asarray(cluster_labels).astype(str)
    locs = np.asarray(coords, dtype=float)
    locs = locs - locs.mean(axis=0, keepdims=True)
    scale = locs.std(axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    locs = locs / scale
    return {
        "pas": float(cal_pas(labels, locs, k=10)),
        "chaos": float(cal_chaos(labels, locs)),
    }


def mean_cluster_sample_entropy(frame: pd.DataFrame) -> float:
    entropies = []
    for _, sub in frame.groupby("cluster"):
        probs = sub["sample"].astype(str).value_counts(normalize=True).to_numpy(dtype=float)
        if len(probs) <= 1:
            entropies.append(0.0)
            continue
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs)))
        entropies.append(entropy)
    return float(np.mean(entropies)) if entropies else float("nan")


def _embedding_frame_from_arrays(sample_order: list[str], coords_by_sample: dict[str, pd.DataFrame], arrays_by_sample: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for sample in sample_order:
        if sample not in coords_by_sample or sample not in arrays_by_sample:
            continue
        coords = coords_by_sample[sample].reset_index(drop=True)
        emb = np.asarray(arrays_by_sample[sample], dtype=float)
        emb_df = pd.DataFrame(emb)
        frame = pd.concat([coords.reset_index(drop=True), emb_df], axis=1)
        frame["sample"] = sample
        rows.append(frame[["sample", "x", "y"] + list(emb_df.columns)])
    return pd.concat(rows, ignore_index=True)


def _merge_with_reference(method_frame: pd.DataFrame, ref_frame: pd.DataFrame, *, ndigits: int = 4) -> pd.DataFrame:
    left = _round_frame(method_frame, ndigits=ndigits)
    right = _round_frame(ref_frame, ndigits=ndigits)
    merged = left.merge(
        right.drop(columns=["x", "y"]),
        on=["sample", "x_round", "y_round"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def _merge_with_embedding(frame: pd.DataFrame, emb_frame: pd.DataFrame, *, ndigits: int = 4) -> pd.DataFrame:
    left = _round_frame(frame, ndigits=ndigits)
    right = _round_frame(emb_frame, ndigits=ndigits)
    return left.merge(
        right.drop(columns=["x", "y"]),
        on=["sample", "x_round", "y_round"],
        how="inner",
        validate="one_to_one",
    )


def _stack_samples(data_by_sample: dict[str, pd.DataFrame], sample_order: list[str]) -> pd.DataFrame:
    parts = []
    for sample in sample_order:
        if sample not in data_by_sample:
            continue
        df = data_by_sample[sample].copy()
        df["sample"] = sample
        parts.append(df[["sample", "x", "y", "cluster"]])
    if not parts:
        return pd.DataFrame(columns=["sample", "x", "y", "cluster"])
    return pd.concat(parts, ignore_index=True)


def _load_crc_meta() -> pd.DataFrame:
    meta = pd.read_csv(CRC_META, index_col=0)
    meta["sample"] = meta["batch"].astype(str)
    return meta


def _load_crc_reference() -> pd.DataFrame:
    meta = _load_crc_meta()
    keep = meta["domain"].astype(str).isin(["4", "5", "9"])
    ref = meta.loc[keep, ["batch", "x", "y", "domain"]].copy()
    ref = ref.rename(columns={"batch": "sample"})
    mapping = {"4": "lamina_propria", "5": "tumor", "9": "tumor_periphery"}
    ref["reference"] = ref["domain"].astype(str).map(mapping)
    return ref[["sample", "x", "y", "reference"]]


def _load_step_crc_labels() -> pd.DataFrame:
    meta = _load_crc_meta()
    return pd.DataFrame(
        {
            "sample": meta["sample"].astype(str).to_numpy(),
            "x": meta["x"].to_numpy(),
            "y": meta["y"].to_numpy(),
            "cluster": meta["domain"].astype(str).to_numpy(),
        }
    )


def _attach_crc_reference_coords(frame: pd.DataFrame) -> pd.DataFrame:
    meta = _load_crc_meta()[["sample", "x", "y"]].copy()
    meta["spot_id"] = meta.index.astype(str)
    return frame.merge(meta, on=["spot_id", "sample"], how="inner", validate="one_to_one")


def _crc_spot_id_from_banksy(barcode: pd.Series, sample: pd.Series) -> pd.Series:
    return barcode.astype(str) + "-" + sample.astype(str)


def _crc_spot_id_from_nichecompass(obs_names: pd.Index) -> pd.Index:
    return pd.Index([f"{v.split('__', 1)[1]}-{v.split('__', 1)[0]}" for v in obs_names.astype(str)])


def _crc_spot_id_from_hergast(obs_names: pd.Index) -> pd.Index:
    out = []
    for v in obs_names.astype(str):
        sample, barcode = v.split("__", 1)
        out.append(f"{barcode}-{sample.replace('_square_008um', '')}")
    return pd.Index(out)


def _load_mosta_reference() -> pd.DataFrame:
    adata = ad.read_h5ad(MOSTA_REF, backed="r")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    obs = adata.obs
    batch_col = "batch" if "batch" in obs.columns else "sample"
    sample = obs[batch_col].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    ref = pd.DataFrame(
        {
            "sample": sample,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "annotation": obs["annotation"].astype(str).to_numpy(),
        }
    )
    adata.file.close()
    return ref


def _load_nichecompass_mosta_labels() -> pd.DataFrame:
    adata = ad.read_h5ad(MOSTA_REF, backed="r")
    obs = adata.obs
    batch_col = "batch" if "batch" in obs.columns else "sample"
    batches = obs[batch_col].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    clusters = pd.read_csv(REPO_ROOT / "external_results" / "nichecompass-test" / "mosta_e16_5_clusters_k21.csv").set_index("barcode")
    labels = clusters.reindex(adata.obs_names)["cluster"].astype("Int64")
    frame = pd.DataFrame(
        {
            "sample": batches,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": labels.astype(int).astype(str).to_numpy(),
        }
    )
    adata.file.close()
    return frame


def _mbk_labels(embedding: np.ndarray, n_clusters: int, batch_size: int = 4096) -> np.ndarray:
    return (
        MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            batch_size=batch_size,
        ).fit_predict(np.asarray(embedding, dtype=float))
        + 1
    )


def _load_banksy_harmony_mosta_reclustered(n_clusters: int) -> pd.DataFrame:
    base = BANKSY_HARMONY_DIR / "mosta"
    sample_order = MOSTA_ORDER
    coord_parts: list[pd.DataFrame] = []
    for sample in sample_order:
        coord_path = base / sample / "spatial_coords.csv"
        if not coord_path.exists():
            continue
        coords = pd.read_csv(coord_path)
        coord_parts.append(
            {
                "barcode": coords["barcode"].astype(str).to_numpy(),
                "sample": np.full(len(coords), sample, dtype=object),
                "x": coords["array_row"].to_numpy(dtype=float),
                "y": coords["array_col"].to_numpy(dtype=float),
            }
        )
    if not coord_parts:
        raise RuntimeError("BANKSY MOSTA coordinates not found")
    coords_df = pd.concat([pd.DataFrame(part) for part in coord_parts], ignore_index=True)
    emb = pd.read_csv(BANKSY_HARMONY_DIR / "mosta_harmony" / "harmony_embeddings.csv")
    emb["barcode"] = emb["barcode"].astype(str)
    emb["sample"] = emb["sample"].astype(str)
    frame = coords_df.merge(emb, on=["barcode", "sample"], how="inner", validate="one_to_one")
    value_cols = [c for c in frame.columns if c.startswith("PC")]
    frame["cluster"] = _mbk_labels(frame[value_cols].to_numpy(dtype=float), n_clusters).astype(str)
    frame["sample"] = pd.Categorical(frame["sample"], categories=sample_order, ordered=True)
    frame = frame.sort_values(["sample", "barcode"], kind="mergesort").reset_index(drop=True)
    return frame[["sample", "x", "y", "cluster"]].copy()


def _load_banksy_harmony_mosta_reclustered_with_embedding(n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = BANKSY_HARMONY_DIR / "mosta"
    sample_order = MOSTA_ORDER
    coord_parts: list[pd.DataFrame] = []
    for sample in sample_order:
        coord_path = base / sample / "spatial_coords.csv"
        if not coord_path.exists():
            continue
        coords = pd.read_csv(coord_path)
        coord_parts.append(
            pd.DataFrame(
                {
                    "barcode": coords["barcode"].astype(str).to_numpy(),
                    "sample": np.full(len(coords), sample, dtype=object),
                    "x": coords["array_row"].to_numpy(dtype=float),
                    "y": coords["array_col"].to_numpy(dtype=float),
                }
            )
        )
    if not coord_parts:
        raise RuntimeError("BANKSY MOSTA coordinates not found")
    coords_df = pd.concat(coord_parts, ignore_index=True)
    emb = pd.read_csv(BANKSY_HARMONY_DIR / "mosta_harmony" / "harmony_embeddings.csv")
    emb["barcode"] = emb["barcode"].astype(str)
    emb["sample"] = emb["sample"].astype(str)
    frame = coords_df.merge(emb, on=["barcode", "sample"], how="inner", validate="one_to_one")
    value_cols = [c for c in frame.columns if c.startswith("PC")]
    frame["cluster"] = _mbk_labels(frame[value_cols].to_numpy(dtype=float), n_clusters).astype(str)
    frame["sample"] = pd.Categorical(frame["sample"], categories=sample_order, ordered=True)
    frame = frame.sort_values(["sample", "barcode"], kind="mergesort").reset_index(drop=True)
    labels = frame[["sample", "x", "y", "cluster"]].copy()
    emb_frame = frame[["sample", "x", "y"] + value_cols].copy()
    return labels, emb_frame


def _load_hergast_harmony_mosta_reclustered(n_clusters: int) -> pd.DataFrame:
    path = HERGAST_HARMONY_DIR / "MOSTA.harmony.h5ad"
    adata = ad.read_h5ad(path, backed="r")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    emb = np.asarray(adata.obsm["HERGAST_harmony"], dtype=float)
    samples = adata.obs["sample"].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    labels = _mbk_labels(emb, n_clusters).astype(str)
    adata.file.close()
    return pd.DataFrame({"sample": samples, "x": coords[:, 0], "y": coords[:, 1], "cluster": labels})


def _load_hergast_harmony_mosta_reclustered_with_embedding(n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = HERGAST_HARMONY_DIR / "MOSTA.harmony.h5ad"
    adata = ad.read_h5ad(path, backed="r")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    emb = np.asarray(adata.obsm["HERGAST_harmony"], dtype=float)
    samples = adata.obs["sample"].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    labels = _mbk_labels(emb, n_clusters).astype(str)
    adata.file.close()
    labels_frame = pd.DataFrame({"sample": samples, "x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    emb_frame = pd.DataFrame(emb)
    emb_frame.insert(0, "y", coords[:, 1])
    emb_frame.insert(0, "x", coords[:, 0])
    emb_frame.insert(0, "sample", samples)
    return labels_frame, emb_frame


def _load_nichecompass_mosta_reclustered(n_clusters: int) -> pd.DataFrame:
    adata = ad.read_h5ad(MOSTA_REF, backed="r")
    obs = adata.obs
    batch_col = "batch" if "batch" in obs.columns else "sample"
    samples = obs[batch_col].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    emb = np.asarray(adata.obsm["nichecompass_latent"], dtype=float)
    labels = _mbk_labels(emb, n_clusters).astype(str)
    adata.file.close()
    return pd.DataFrame({"sample": samples, "x": coords[:, 0], "y": coords[:, 1], "cluster": labels})


def _load_nichecompass_mosta_reclustered_with_embedding(n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    adata = ad.read_h5ad(MOSTA_REF, backed="r")
    obs = adata.obs
    batch_col = "batch" if "batch" in obs.columns else "sample"
    samples = obs[batch_col].astype(str).str.replace(".MOSTA", "", regex=False).to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    emb = np.asarray(adata.obsm["nichecompass_latent"], dtype=float)
    labels = _mbk_labels(emb, n_clusters).astype(str)
    adata.file.close()
    labels_frame = pd.DataFrame({"sample": samples, "x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    emb_frame = pd.DataFrame(emb)
    emb_frame.insert(0, "y", coords[:, 1])
    emb_frame.insert(0, "x", coords[:, 0])
    emb_frame.insert(0, "sample", samples)
    return labels_frame, emb_frame


def _load_nichecompass_crc_embedding_frame() -> pd.DataFrame:
    adata = ad.read_h5ad(NICHE_CRC, backed="r")
    emb = np.asarray(adata.obsm["nichecompass_latent"], dtype=float)
    spot_ids = _crc_spot_id_from_nichecompass(pd.Index(adata.obs_names))
    frame = pd.DataFrame(emb)
    frame.insert(0, "sample", adata.obs["batch"].astype(str).to_numpy())
    frame.insert(0, "spot_id", np.asarray(spot_ids))
    frame = _attach_crc_reference_coords(frame)
    adata.file.close()
    return frame.drop(columns=["spot_id"])


def _banksy_frame_from_precomputed_blocks(
    *,
    sample_order: list[str],
    coords_by_sample: dict[str, pd.DataFrame],
    embedding_df: pd.DataFrame,
    cluster_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    parts = []
    start = 0
    value_cols = [c for c in embedding_df.columns if c not in {"barcode", "sample"}]
    total_rows = 0
    for sample in sample_order:
        if sample not in coords_by_sample:
            continue
        coords = coords_by_sample[sample].reset_index(drop=True)
        n = len(coords)
        emb_block = embedding_df.iloc[start : start + n].reset_index(drop=True)
        if len(emb_block) != n:
            raise ValueError(f"Embedding row mismatch for {sample}: expected {n}, got {len(emb_block)}")
        if "sample" in emb_block.columns:
            block_samples = emb_block["sample"].astype(str).unique().tolist()
            if block_samples and block_samples != [str(sample)]:
                raise ValueError(f"Unexpected sample block order for {sample}: {block_samples}")
        frame = coords.copy()
        frame.insert(0, "sample", sample)
        if cluster_df is not None:
            cluster_block = cluster_df.iloc[start : start + n].reset_index(drop=True)
            if len(cluster_block) != n:
                raise ValueError(f"Cluster row mismatch for {sample}: expected {n}, got {len(cluster_block)}")
            frame["cluster"] = cluster_block["cluster"].astype(int).astype(str).to_numpy()
        for col in value_cols:
            frame[col] = emb_block[col].to_numpy()
        parts.append(frame)
        start += n
        total_rows += n
    if start != len(embedding_df):
        raise ValueError(f"Unused embedding rows: consumed {start}, total {len(embedding_df)}")
    if cluster_df is not None and total_rows != len(cluster_df):
        raise ValueError(f"Cluster table length mismatch: consumed {total_rows}, total {len(cluster_df)}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _load_banksy_harmony_crc() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = BANKSY_HARMONY_DIR / "008um"
    coords_by_sample: dict[str, pd.DataFrame] = {}
    sample_order = CRC_ORDER
    for sample in sample_order:
        coord_path = base / sample / "spatial_coords.csv"
        if not coord_path.exists():
            continue
        coords = pd.read_csv(coord_path)
        coords_by_sample[sample] = pd.DataFrame(
            {
                "barcode": coords["barcode"].astype(str).to_numpy(),
                "x": coords["array_row"].to_numpy(),
                "y": coords["array_col"].to_numpy(),
            }
        )
    emb = pd.read_csv(BANKSY_HARMONY_DIR / "visium_hd_harmony" / "harmony_embeddings.csv")
    clusters = pd.read_csv(BANKSY_HARMONY_DIR / "visium_hd_harmony" / "kmeans_clusters_k12.csv")
    frame = _banksy_frame_from_precomputed_blocks(
        sample_order=sample_order,
        coords_by_sample=coords_by_sample,
        embedding_df=emb,
        cluster_df=clusters,
    )
    frame["spot_id"] = _crc_spot_id_from_banksy(frame["barcode"], frame["sample"])
    frame = _attach_crc_reference_coords(frame[["spot_id", "sample", "cluster"] + [c for c in frame.columns if c.startswith("PC")]])
    labels = frame[["sample", "x", "y", "cluster"]].copy()
    emb_frame = frame.drop(columns=["cluster", "spot_id"]).copy()
    return labels, emb_frame


def _load_banksy_harmony_mosta() -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded = _load_banksy_embeddings("mosta")
    if loaded is None:
        raise RuntimeError("BANKSY MOSTA coordinates not found")
    coords_by_sample, _, sample_order = loaded
    emb = pd.read_csv(BANKSY_HARMONY_DIR / "mosta_harmony" / "harmony_embeddings.csv")
    clusters = pd.read_csv(BANKSY_HARMONY_DIR / "mosta_harmony" / "kmeans_clusters_k21.csv").drop(columns=["barcode"], errors="ignore")
    frame = _banksy_frame_from_precomputed_blocks(
        sample_order=sample_order,
        coords_by_sample=coords_by_sample,
        embedding_df=emb,
        cluster_df=clusters,
    )
    labels = frame[["sample", "x", "y", "cluster"]].copy()
    emb_frame = frame.drop(columns=["cluster"]).copy()
    return labels, emb_frame


def _load_hergast_harmony(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "crc":
        path = HERGAST_HARMONY_DIR / "VisiumHD.harmony.h5ad"
        sample_order = CRC_ORDER
    elif dataset == "mosta":
        path = HERGAST_HARMONY_DIR / "MOSTA.harmony.h5ad"
        sample_order = MOSTA_ORDER
    else:
        raise ValueError(f"Unsupported HERGAST harmony dataset: {dataset}")
    adata = ad.read_h5ad(path, backed="r")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    emb = np.asarray(adata.obsm["HERGAST_harmony"], dtype=float)
    samples = adata.obs["sample"].astype(str).to_numpy()
    labels = adata.obs["kmeans_harmony"].astype(int).astype(str).to_numpy()
    frame = pd.DataFrame({"sample": samples, "x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    if dataset == "mosta":
        frame["sample"] = frame["sample"].str.replace(".MOSTA", "", regex=False)
    elif dataset == "crc":
        frame = pd.DataFrame(
            {
                "spot_id": _crc_spot_id_from_hergast(pd.Index(adata.obs_names)),
                "sample": samples,
                "cluster": labels,
            }
        )
        emb_frame = pd.DataFrame(emb)
        emb_frame.insert(0, "sample", samples)
        emb_frame.insert(0, "spot_id", _crc_spot_id_from_hergast(pd.Index(adata.obs_names)))
        frame = _attach_crc_reference_coords(frame)
        emb_frame = _attach_crc_reference_coords(emb_frame)
        adata.file.close()
        return frame[["sample", "x", "y", "cluster"]], emb_frame.drop(columns=["spot_id"])
    emb_frame = pd.DataFrame(emb)
    emb_frame.insert(0, "y", frame["y"].to_numpy())
    emb_frame.insert(0, "x", frame["x"].to_numpy())
    emb_frame.insert(0, "sample", frame["sample"].to_numpy())
    labels_frame = pd.concat(
        [frame.loc[frame["sample"] == sample, ["sample", "x", "y", "cluster"]] for sample in sample_order if sample in set(frame["sample"])],
        ignore_index=True,
    )
    emb_frame = pd.concat(
        [emb_frame.loc[emb_frame["sample"] == sample] for sample in sample_order if sample in set(emb_frame["sample"])],
        ignore_index=True,
    )
    adata.file.close()
    return labels_frame, emb_frame


def _load_nichecompass_crc_labels() -> pd.DataFrame:
    adata = ad.read_h5ad(NICHE_CRC, backed="r")
    samples = adata.obs["batch"].astype(str).to_numpy()
    spot_ids = _crc_spot_id_from_nichecompass(pd.Index(adata.obs_names))
    if NICHE_CRC_CLUSTERS.exists():
        clusters = pd.read_csv(NICHE_CRC_CLUSTERS).set_index("barcode")
        labels = clusters.reindex(pd.Index(adata.obs_names.astype(str)))["cluster"].astype("Int64")
        valid = labels.notna().to_numpy()
        frame = _attach_crc_reference_coords(
            pd.DataFrame(
            {
                "spot_id": np.asarray(spot_ids)[valid],
                "sample": samples[valid],
                "cluster": labels[valid].astype(int).astype(str).to_numpy(),
            })
        )
    else:
        loaded = _read_plot_csv(BANKSY_RECLUSTER_OUTDIR / "nichecompass_crc_clusters.csv", CRC_ORDER)
        if loaded is None:
            raise RuntimeError("NicheCompass CRC clusters not found")
        frame = _stack_samples(*loaded)
    adata.file.close()
    return frame


def benchmark_crc() -> pd.DataFrame:
    domain_df = benchmark_crc_domain_matching()
    quality_df = benchmark_crc_quality()
    if domain_df.empty:
        return quality_df
    if quality_df.empty:
        return domain_df
    return domain_df.merge(quality_df, on="method", how="left", validate="one_to_one")


def benchmark_crc_domain_matching() -> pd.DataFrame:
    ref = _load_crc_reference()
    radius = 1.5
    methods: list[tuple[str, pd.DataFrame]] = [
        ("STEP", _load_step_crc_labels()),
        ("BANKSY Harmony-integrated", _load_banksy_harmony_crc()[0]),
        ("HERGAST Harmony-integrated", _load_hergast_harmony("crc")[0]),
        ("NicheCompass", _load_nichecompass_crc_labels()),
    ]
    rows = []
    for method, frame in methods:
        rows.append(crc_domain_match_row(method, frame, ref, radius=radius))
    return pd.DataFrame(rows)


def benchmark_crc_quality() -> pd.DataFrame:
    ref = _load_crc_reference()
    methods: list[tuple[str, pd.DataFrame, pd.DataFrame | None]] = [
        ("BANKSY Harmony-integrated", *_load_banksy_harmony_crc()),
        ("HERGAST Harmony-integrated", *_load_hergast_harmony("crc")),
        ("NicheCompass", _load_nichecompass_crc_labels(), _load_nichecompass_crc_embedding_frame()),
    ]
    rows = []
    for method, frame, emb_frame in methods:
        merged = _merge_with_reference(frame, ref)
        if merged.empty:
            continue
        row = {"method": method}
        row.update(benchmark_crc_continuity_for_frame(merged))
        row.update(benchmark_crc_batch_for_frame(merged, emb_frame))
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_crc_continuity_for_frame(merged: pd.DataFrame) -> dict[str, float]:
    coords = merged[["x", "y"]].to_numpy(dtype=float)
    cont = continuity_metrics(merged["cluster"], coords)
    return {
        "pas": cont["pas"],
        "chaos": cont["chaos"],
        "sample_entropy": mean_cluster_sample_entropy(merged[["sample", "cluster"]]),
        "cluster_sample_nmi": float(normalized_mutual_info_score(merged["sample"].astype(str), merged["cluster"].astype(str))),
    }


def benchmark_crc_batch_for_frame(merged: pd.DataFrame, emb_frame: pd.DataFrame | None) -> dict[str, float]:
    if emb_frame is None:
        return {"batch_asw": float("nan"), "batch_ilisi": float("nan"), "n_eval_cells": 0.0}
    emb_merged = _merge_with_embedding(merged[["sample", "x", "y", "cluster"]], emb_frame)
    max_cells = 100000
    if len(emb_merged) > max_cells:
        emb_merged = emb_merged.sample(max_cells, random_state=42)
    embed = emb_merged.drop(columns=["sample", "x", "y", "cluster", "x_round", "y_round"]).to_numpy(dtype=float)
    batch = emb_merged["sample"].astype(str).to_numpy()
    bio = emb_merged["cluster"].astype(str).to_numpy()
    return {
        "batch_asw": float(cal_batch_asw(embed, batch, bio)),
        "batch_ilisi": float(cal_batch_ilisi(embed, batch, n_neighbors=30)),
        "n_eval_cells": float(len(emb_merged)),
    }


def benchmark_crc_continuity() -> pd.DataFrame:
    ref = _load_crc_reference()
    methods: list[tuple[str, pd.DataFrame]] = [
        ("BANKSY Harmony-integrated", _load_banksy_harmony_crc()[0]),
        ("HERGAST Harmony-integrated", _load_hergast_harmony("crc")[0]),
        ("NicheCompass", _load_nichecompass_crc_labels()),
    ]
    rows = []
    for method, frame in methods:
        merged = _merge_with_reference(frame, ref)
        if merged.empty:
            continue
        row: dict[str, float | str] = {"method": method}
        row.update(benchmark_crc_continuity_for_frame(merged))
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_crc_batch() -> pd.DataFrame:
    ref = _load_crc_reference()
    methods: list[tuple[str, pd.DataFrame, pd.DataFrame | None]] = [
        ("BANKSY Harmony-integrated", *_load_banksy_harmony_crc()),
        ("HERGAST Harmony-integrated", *_load_hergast_harmony("crc")),
        ("NicheCompass", _load_nichecompass_crc_labels(), _load_nichecompass_crc_embedding_frame()),
    ]
    rows = []
    for method, frame, emb_frame in methods:
        merged = _merge_with_reference(frame, ref)
        if merged.empty:
            continue
        row: dict[str, float | str] = {"method": method}
        row.update(benchmark_crc_batch_for_frame(merged, emb_frame))
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_mosta_scib_batch_metrics(batch_csv: Path | None = None) -> pd.DataFrame:
    if batch_csv is not None:
        return pd.read_csv(batch_csv)
    with tempfile.TemporaryDirectory(prefix="mosta_scib_batch_") as tmpdir:
        temp_batch_csv = Path(tmpdir) / "mosta_scib_batch_metrics.csv"
        cmd = [
            "uv",
            "run",
            "--with",
            "numpy>=2.1.0",
            "--with",
            "scib-metrics",
            "python",
            str(SCIB_BATCH_SCRIPT),
            "--output-csv",
            str(temp_batch_csv),
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return pd.read_csv(temp_batch_csv)


def benchmark_mosta(batch_csv: Path | None = None) -> pd.DataFrame:
    ref = _load_mosta_reference()
    n_clusters = _mosta_ground_truth_cluster_count()
    methods: list[tuple[str, pd.DataFrame]] = [
        ("BANKSY Harmony-integrated", _load_banksy_harmony_mosta_reclustered(n_clusters)),
        ("HERGAST Harmony-integrated", _load_hergast_harmony_mosta_reclustered(n_clusters)),
        ("NicheCompass", _load_nichecompass_mosta_reclustered(n_clusters)),
    ]

    rows = []
    for method, frame in methods:
        merged = _merge_with_reference(frame, ref)
        coords = merged[["x", "y"]].to_numpy(dtype=float)
        cont = continuity_metrics(merged["cluster"], coords)
        ari = float(adjusted_rand_score(merged["annotation"].astype(str), merged["cluster"].astype(str)))
        nmi = float(normalized_mutual_info_score(merged["annotation"].astype(str), merged["cluster"].astype(str)))
        row: dict[str, float | str] = {
            "method": method,
            "annotation_ari": ari,
            "annotation_nmi": nmi,
            "pas": cont["pas"],
            "chaos": cont["chaos"],
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    batch_df = _compute_mosta_scib_batch_metrics(batch_csv=batch_csv)
    return result.merge(batch_df, on="method", how="left", validate="one_to_one")


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("nan" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "crc", "mosta"], default="all")
    parser.add_argument("--mosta-batch-csv", type=Path, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "workflows" / f"external_method_benchmark_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, list[dict[str, object]]] = {}

    if args.only in {"all", "crc"}:
        crc = benchmark_crc()
        crc.to_csv(out_dir / "crc_benchmark.csv", index=False)
        write_markdown(crc, out_dir / "crc_benchmark.md")
        payload["crc"] = crc.to_dict(orient="records")

    if args.only in {"all", "mosta"}:
        mosta = benchmark_mosta(batch_csv=args.mosta_batch_csv)
        mosta.to_csv(out_dir / "mosta_benchmark.csv", index=False)
        write_markdown(mosta, out_dir / "mosta_benchmark.md")
        payload["mosta"] = mosta.to_dict(orient="records")

    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    note = (
        "CRC benchmark uses STEP notebook reference domains 4, 5, and 9 as Lamina Propria, Tumor, and Tumor Periphery.\n"
        "MOSTA benchmark uses the manual annotation field from the merged reference object.\n"
        "Continuity uses PAS and CHAOS on the final spatial clustering labels.\n"
        "CRC batch-mixing summaries use the local embedding-based proxy metrics defined in this script.\n"
        "MOSTA batch-mixing summaries use the agg_metrics_multi.ipynb-matched scib_metrics Benchmarker pathway.\n"
    )
    (out_dir / "README.md").write_text(note)


if __name__ == "__main__":
    main()
