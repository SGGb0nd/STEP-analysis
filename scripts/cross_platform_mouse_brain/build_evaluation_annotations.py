"""Build biological labels for the official 70-section evaluation subset."""

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CELL_TYPE_MAP = SCRIPT_DIR / "broad_cell_type_map.tsv"


def numeric_file_key(path: Path) -> int:
    match = re.search(r"batch(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_cell_type_map(path: Path) -> dict[tuple[str, str], str]:
    table = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {"technology", "original_label", "broad_cell_type"}
    if set(table.columns) != required:
        raise ValueError(f"Unexpected columns in {path}: {list(table.columns)}")
    if table.duplicated(["technology", "original_label"]).any():
        raise ValueError(f"Duplicate entries in {path}")
    return {
        (str(row.technology), str(row.original_label)): str(row.broad_cell_type)
        for row in table.itertuples(index=False)
    }


def read_annotations(
    path: Path,
    technology: str,
    cell_type_map: dict[tuple[str, str], str],
) -> pd.DataFrame:
    source = ad.read_h5ad(path, backed="r")
    try:
        if technology == "MERFISH":
            section_key = "brain_section_label"
            cell_type_key = "cell_type"
            tissue_region = pd.Series("", index=source.obs_names, dtype=object)
        else:
            section_key = "batch"
            cell_type_key = "Main_molecular_cell_type"
            tissue_region = source.obs["Main_molecular_tissue_region"].astype(str)

        section_values = source.obs[section_key].astype(str)
        if section_values.nunique() != 1:
            raise ValueError(f"Expected one section in {path}")
        original = source.obs[cell_type_key].astype(str)
        broad = original.map(
            lambda value: cell_type_map.get((technology, value), "unmatched")
        )
        return pd.DataFrame(
            {
                "source_obs_name": source.obs_names.astype(str),
                "technology": technology,
                "section": str(section_values.iloc[0]),
                "cell_type_original": original.to_numpy(),
                "broad_cell_type": broad.to_numpy(),
                "tissue_region_original": tissue_region.to_numpy(),
                "source_file": path.name,
            }
        )
    finally:
        source.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labels for the public 50 MERFISH + 20 STARmap evaluation subset."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--cell-type-map", type=Path, default=DEFAULT_CELL_TYPE_MAP)
    args = parser.parse_args()

    cell_type_map = load_cell_type_map(args.cell_type_map)
    sources = {
        "MERFISH": sorted(
            (args.input_root / "merfish_mouse_brain").glob("*.h5ad"),
            key=numeric_file_key,
        ),
        "STARmap PLUS": sorted(
            (args.input_root / "starmap_plus_mouse_cns").glob("*.h5ad"),
            key=numeric_file_key,
        ),
    }
    expected = {"MERFISH": 50, "STARmap PLUS": 20}
    frames = []
    for technology, paths in sources.items():
        if len(paths) != expected[technology]:
            raise ValueError(
                f"Expected {expected[technology]} {technology} files, found {len(paths)}"
            )
        frames.extend(read_annotations(path, technology, cell_type_map) for path in paths)

    annotations = pd.concat(frames, ignore_index=True)
    if annotations["source_obs_name"].duplicated().any():
        raise ValueError("Evaluation cell identifiers are not globally unique")
    annotations["technology"] = pd.Categorical(
        annotations["technology"], categories=["MERFISH", "STARmap PLUS"]
    )
    annotations["section"] = annotations["section"].astype("category")
    annotations["broad_cell_type"] = annotations["broad_cell_type"].astype("category")
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_parquet(args.output_parquet, index=False)

    summary = {
        "output": str(args.output_parquet.resolve()),
        "n_cells": int(len(annotations)),
        "technology_counts": {
            str(key): int(value)
            for key, value in annotations["technology"].value_counts().items()
        },
        "n_sections": int(annotations["section"].nunique()),
        "broad_cell_type_counts": {
            str(key): int(value)
            for key, value in annotations["broad_cell_type"].value_counts().items()
        },
    }
    args.output_parquet.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
