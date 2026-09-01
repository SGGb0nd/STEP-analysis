"""Build the canonical 50-micron CRC microarchitecture intermediates."""

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from step.models.niche import MicroArc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("results/visium-hd/crc5_meta_filtered.csv"),
    )
    parser.add_argument(
        "--scale-h5ad",
        type=Path,
        default=Path("results/visium-hd/cancer_tumor_regions.h5ad"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/visium-hd"),
    )
    parser.add_argument("--low-coherence-fraction", type=float, default=0.20)
    parser.add_argument("--radius-um", type=float, default=50.0)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--kernel-iterations", type=int, default=5)
    return parser.parse_args()


def load_microns_per_pixel(path: Path) -> dict[str, float]:
    scale_adata = ad.read_h5ad(path, backed="r")
    try:
        return {
            str(batch): float(payload["scalefactors"]["microns_per_pixel"])
            for batch, payload in scale_adata.uns["spatial"].items()
        }
    finally:
        scale_adata.file.close()


def convert_coordinates_to_microns(
    metadata: pd.DataFrame,
    microns_per_pixel: dict[str, float],
) -> pd.DataFrame:
    result = metadata.copy()
    result["microns_per_pixel"] = result["batch"].astype(str).map(microns_per_pixel)
    if result["microns_per_pixel"].isna().any():
        missing = sorted(
            result.loc[result["microns_per_pixel"].isna(), "batch"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Missing microns_per_pixel for batches: {missing}")
    result["x_pixel"] = result["x"]
    result["y_pixel"] = result["y"]
    result["x"] = result["x_pixel"] * result["microns_per_pixel"]
    result["y"] = result["y_pixel"] * result["microns_per_pixel"]
    return result


def add_pattern_assignments(
    metadata: pd.DataFrame,
    microarc: MicroArc,
) -> pd.DataFrame:
    if microarc.clusters is None:
        raise ValueError("Microarchitecture clusters have not been computed")
    result = metadata.copy()
    result["niche_pattern"] = "None"
    for graph_index, neighbor_indices in enumerate(microarc.neighbor_indices):
        pattern = int(microarc.clusters[graph_index])
        result.loc[neighbor_indices, "niche_pattern"] = f"Pattern_{pattern}"
    return result


def main() -> None:
    args = parse_args()
    if not 0 < args.low_coherence_fraction < 1:
        raise ValueError("--low-coherence-fraction must be between 0 and 1")

    metadata = pd.read_csv(args.metadata_csv, index_col=0, low_memory=False)
    required = {"batch", "domain", "cell_type", "cosine", "x", "y"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise KeyError(f"Missing metadata columns: {missing}")

    metadata_um = convert_coordinates_to_microns(
        metadata,
        load_microns_per_pixel(args.scale_h5ad),
    )
    query_mask = metadata_um["domain"].eq(9) & metadata_um["batch"].astype(
        str
    ).str.startswith("cancer")
    query = metadata_um.index[query_mask]
    if query.empty:
        raise ValueError("No CRC tumor-periphery spots were found")
    threshold = float(
        metadata_um.loc[query, "cosine"].quantile(args.low_coherence_fraction)
    )

    microarc = MicroArc(
        metadata=metadata_um,
        label_col="cell_type",
        batch_col="batch",
        coherence_threshold=threshold,
        radius=args.radius_um,
        n_clusters=args.n_clusters,
        n_iter=args.kernel_iterations,
    )
    microarc.create_graphs(query).compute_kernel_and_cluster(method="approx")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "microarc_true50um.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump(microarc, handle)

    metadata_with_patterns = add_pattern_assignments(metadata, microarc)
    metadata_path = args.output_dir / "metadata_with_patterns_true50um.csv"
    metadata_with_patterns.to_csv(metadata_path)

    settings = {
        "metadata_csv": str(args.metadata_csv),
        "scale_h5ad": str(args.scale_h5ad),
        "query_domain": 9,
        "query_sections": "batch starts with cancer",
        "low_coherence_fraction": args.low_coherence_fraction,
        "coherence_threshold": threshold,
        "radius_um": args.radius_um,
        "n_clusters": args.n_clusters,
        "kernel_method": "approx",
        "kernel_iterations": args.kernel_iterations,
        "n_candidate_spots": int(len(query)),
        "n_graphs": int(len(microarc.graphs)),
        "cluster_counts": {
            str(index + 1): int(count)
            for index, count in enumerate(
                np.bincount(np.asarray(microarc.clusters), minlength=args.n_clusters)
            )
        },
        "checkpoint": str(checkpoint),
        "metadata_with_patterns": str(metadata_path),
    }
    (args.output_dir / "microarc_true50um_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    print(json.dumps(settings, indent=2), flush=True)


if __name__ == "__main__":
    main()
