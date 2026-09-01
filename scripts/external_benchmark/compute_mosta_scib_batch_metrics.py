"""Compute dataset-level MOSTA ASW and iLISI with the scib-metrics functional API."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from scib_metrics import ilisi_knn
from scib_metrics.metrics._silhouette import silhouette_batch
from scib_metrics.nearest_neighbors import pynndescent

try:
    from .external_method_benchmark import (
        _load_banksy_harmony_mosta_reclustered_with_embedding,
        _load_hergast_harmony_mosta_reclustered_with_embedding,
        _load_mosta_reference,
        _load_nichecompass_mosta_reclustered_with_embedding,
        _merge_with_embedding,
        _mosta_ground_truth_cluster_count,
    )
except ImportError:
    from external_method_benchmark import (
        _load_banksy_harmony_mosta_reclustered_with_embedding,
        _load_hergast_harmony_mosta_reclustered_with_embedding,
        _load_mosta_reference,
        _load_nichecompass_mosta_reclustered_with_embedding,
        _merge_with_embedding,
        _mosta_ground_truth_cluster_count,
    )


METHOD_LOADERS = {
    "BANKSY Harmony-integrated": _load_banksy_harmony_mosta_reclustered_with_embedding,
    "HERGAST Harmony-integrated": _load_hergast_harmony_mosta_reclustered_with_embedding,
    "NicheCompass": _load_nichecompass_mosta_reclustered_with_embedding,
}


def _compute_batch_metrics(emb_frame: pd.DataFrame, ref: pd.DataFrame) -> dict[str, float]:
    merged = _merge_with_embedding(ref[["sample", "x", "y", "annotation"]], emb_frame)
    embed = merged.drop(columns=["sample", "x", "y", "annotation", "x_round", "y_round"]).to_numpy(dtype=float)
    labels = merged["annotation"].astype(str).to_numpy()
    batch = merged["sample"].astype(str).to_numpy()
    # Use the scib-metrics functional API so the ASW and iLISI definitions are
    # explicit and shared across every section of the merged dataset.
    asw = float(silhouette_batch(embed, labels, batch))
    neighbors = pynndescent(embed, n_neighbors=90, random_state=0, n_jobs=1)
    ilisi = float(ilisi_knn(neighbors, batch, scale=True))
    return {
        "batch_asw": asw,
        "batch_ilisi": ilisi,
        "n_eval_cells": float(len(merged)),
    }


def compute_mosta_scib_batch_metrics(methods: list[str] | None = None) -> pd.DataFrame:
    ref = _load_mosta_reference()
    n_clusters = _mosta_ground_truth_cluster_count()
    selected_methods = methods if methods is not None else list(METHOD_LOADERS)
    rows: list[dict[str, float | str]] = []
    for method in selected_methods:
        print(f"[compute_mosta_scib_batch_metrics] starting {method}", flush=True)
        emb_frame = METHOD_LOADERS[method](n_clusters)[1]
        row: dict[str, float | str] = {"method": method}
        row.update(_compute_batch_metrics(emb_frame, ref))
        rows.append(row)
        print(f"[compute_mosta_scib_batch_metrics] finished {method}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        choices=list(METHOD_LOADERS),
        help="Compute only the specified method. Repeatable.",
    )
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = compute_mosta_scib_batch_metrics(methods=args.method)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "output_csv": str(args.output_csv.resolve()),
        "methods": df.to_dict(orient="records"),
        "metric_source": "agg_metrics_multi.ipynb notebook semantic path: official scib_metrics silhouette_batch plus ilisi_knn on the integrated embedding, with annotation labels and sample batches",
        "metric_note": "ASW and iLISI are computed once on the complete integrated dataset with the scib-metrics functional API.",
    }
    args.output_csv.with_name(args.output_csv.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
