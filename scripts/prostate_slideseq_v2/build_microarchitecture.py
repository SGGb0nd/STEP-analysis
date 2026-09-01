#!/usr/bin/env python3
"""Build the prostate Slide-seq V2 microarchitecture intermediates.

The input is the fitted STEP object written by
``prostate_slideseq_v2_representation.ipynb``. The script identifies the
bottom 20% low-coherence beads in the tumor regions as graph centers, builds
50-um local graphs, and clusters them into the five MAs reported in the
manuscript.
"""

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from step.models.niche import MicroArc


DEFAULT_SAMPLES = ("Tumor02", "Tumor08")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=Path("results/slide-seq/prostate_slideseq_step.h5ad"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/slide-seq"),
    )
    parser.add_argument("--low-coherence-fraction", type=float, default=0.20)
    parser.add_argument("--radius-um", type=float, default=50.0)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    return parser.parse_args()


def ensure_analysis_columns(adata: ad.AnnData) -> None:
    if "cosine" not in adata.obs:
        intrinsic = np.asarray(adata.obsm["X_rep"])
        spatial = np.asarray(adata.obsm["X_smoothed"])
        denominator = np.linalg.norm(intrinsic, axis=1) * np.linalg.norm(
            spatial, axis=1
        )
        adata.obs["cosine"] = np.divide(
            np.sum(intrinsic * spatial, axis=1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

    if "zonation" not in adata.obs:
        zonation = {
            "1": "Epi",
            "2": "Boundary",
            "3": "Tumor",
            "4": "Epi2",
            "5": "Fibro",
        }
        adata.obs["zonation"] = adata.obs["domain"].astype(str).map(zonation)


def serialize_results(microarc: MicroArc, metadata: pd.DataFrame) -> dict:
    graphs = []
    for graph, center_index, neighbor_index in zip(
        microarc.graphs,
        microarc.center_indices,
        microarc.neighbor_indices,
        strict=True,
    ):
        graphs.append(
            {
                "graph": graph,
                "center": metadata.loc[center_index].copy(),
                "neighbors": metadata.loc[neighbor_index].copy(),
            }
        )

    return {
        "clusters": np.asarray(microarc.clusters),
        "niche_graphs": graphs,
        "kernel_matrix": np.asarray(microarc.kernel_matrix),
        "centers": metadata.loc[microarc.center_indices].copy(),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.low_coherence_fraction < 1:
        raise ValueError("--low-coherence-fraction must be between 0 and 1")

    adata = ad.read_h5ad(args.input_h5ad)
    ensure_analysis_columns(adata)
    metadata = adata.obs.copy()

    required = {"sample", "x", "y", "cell1", "cosine", "zonation"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise KeyError(f"Missing required observation columns: {missing}")

    query_mask = metadata["sample"].isin(args.samples) & metadata["zonation"].eq(
        "Tumor"
    )
    query = metadata.index[query_mask]
    if query.empty:
        raise ValueError("No tumor-region beads were found for the requested samples")

    coherence_threshold = float(
        metadata.loc[query, "cosine"].quantile(args.low_coherence_fraction)
    )
    microarc = MicroArc(
        metadata=metadata,
        label_col="cell1",
        batch_col="sample",
        coherence_threshold=coherence_threshold,
        radius=args.radius_um,
        n_clusters=args.n_clusters,
    )
    microarc.create_graphs(query).compute_kernel_and_cluster(
        method="full",
        save_kernel_matrix=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = serialize_results(microarc, metadata)
    results_path = args.output_dir / "prostate_microarchitecture_results.pkl"
    with results_path.open("wb") as handle:
        pickle.dump(results, handle)

    metadata["is_center"] = False
    metadata["center_pattern"] = np.nan
    metadata["microarchitecture"] = pd.Series(
        pd.NA,
        index=metadata.index,
        dtype="Int64",
    )
    centers = pd.Index(microarc.center_indices)
    metadata.loc[centers, "is_center"] = True
    metadata.loc[centers, "center_pattern"] = microarc.clusters
    metadata.loc[centers, "microarchitecture"] = microarc.clusters + 1
    metadata_path = args.output_dir / "prostate_microarchitecture_metadata.csv"
    metadata.to_csv(metadata_path)

    settings = {
        "input_h5ad": str(args.input_h5ad),
        "samples": list(args.samples),
        "query_zonation": "Tumor",
        "low_coherence_fraction": args.low_coherence_fraction,
        "coherence_threshold": coherence_threshold,
        "radius_um": args.radius_um,
        "n_clusters": args.n_clusters,
        "n_graphs": len(microarc.graphs),
        "results": str(results_path),
        "metadata": str(metadata_path),
    }
    (args.output_dir / "prostate_microarchitecture_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
