"""Build anatomical-domain labels for the public cross-technology subset."""

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import nrrd
import numpy as np
import pandas as pd


CCF_MAJOR_REGION_ANCESTORS = [
    ("Isocortex", "Isocortex"),
    ("Olfactory areas", "OLF"),
    ("Hippocampal formation", "HPF"),
    ("Cortical subplate", "CTXsp"),
    ("Striatum", "STR"),
    ("Pallidum", "PAL"),
    ("Thalamus", "TH"),
    ("Hypothalamus", "HY"),
    ("Midbrain", "MB"),
    ("Hindbrain", "P"),
    ("Hindbrain", "MY"),
    ("Cerebellum", "CB"),
    ("Fiber tracts", "fiber tracts"),
    ("Ventricular systems", "VS"),
]

# Composite STARmap regions are intentionally left unmapped.
STARMAP_SHARED_REGIONS = {
    "CTX_1": "Isocortex",
    "CTX_2": "Isocortex",
    "ENTm": "Isocortex",
    "OB_1": "Olfactory areas",
    "OB_2": "Olfactory areas",
    "DG": "Hippocampal formation",
    "HPF_CA": "Hippocampal formation",
    "STR": "Striatum",
    "TH": "Thalamus",
    "HY": "Hypothalamus",
    "MYdp": "Hindbrain",
    "CB_1": "Cerebellum",
    "CB_2": "Cerebellum",
    "FbTrt": "Fiber tracts",
}


def numeric_file_key(path: Path) -> int:
    match = re.search(r"batch(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_structure_graph(path: Path) -> dict[int, dict[str, object]]:
    root = json.loads(path.read_text())["msg"][0]
    nodes: dict[int, dict[str, object]] = {}

    def walk(node: dict[str, object], parent_path: tuple[int, ...]) -> None:
        node_id = int(node["id"])
        path_ids = parent_path + (node_id,)
        nodes[node_id] = {
            "acronym": str(node["acronym"]),
            "name": str(node["name"]),
            "path": path_ids,
        }
        for child in node.get("children", []):
            walk(child, path_ids)

    walk(root, ())
    return nodes


def find_structure_id(nodes: dict[int, dict[str, object]], term: str) -> int:
    term = term.lower()
    matches = [
        node_id
        for node_id, node in nodes.items()
        if str(node["acronym"]).lower() == term
        or str(node["name"]).lower() == term
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one CCF structure for {term!r}, found {matches}")
    return matches[0]


def make_structure_lookup(
    nodes: dict[int, dict[str, object]],
) -> dict[int, tuple[str, str, str]]:
    major_ancestors = [
        (label, find_structure_id(nodes, term))
        for label, term in CCF_MAJOR_REGION_ANCESTORS
    ]
    lookup = {}
    for node_id, node in nodes.items():
        path_ids = set(node["path"])
        major_label = next(
            (label for label, major_id in major_ancestors if major_id in path_ids),
            "unassigned",
        )
        lookup[node_id] = (
            str(node["acronym"]),
            str(node["name"]),
            major_label,
        )
    return lookup


def assign_ccf_regions(
    coordinates: np.ndarray,
    annotation: np.ndarray,
    voxel_size_um: float,
    structure_lookup: dict[int, tuple[str, str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    finite = np.isfinite(coordinates).all(axis=1)
    voxels = np.zeros_like(coordinates, dtype=np.int64)
    voxels[finite] = np.floor(coordinates[finite] / voxel_size_um).astype(np.int64)
    inside = finite & np.all(voxels >= 0, axis=1) & np.all(
        voxels < np.asarray(annotation.shape), axis=1
    )
    structure_ids = np.zeros(len(coordinates), dtype=np.int64)
    structure_ids[inside] = annotation[tuple(voxels[inside].T)]

    acronyms = np.full(len(coordinates), "unassigned", dtype=object)
    names = np.full(len(coordinates), "unassigned", dtype=object)
    major = np.full(len(coordinates), "unassigned", dtype=object)
    for structure_id in np.unique(structure_ids):
        if structure_id == 0:
            continue
        acronym, name, major_label = structure_lookup.get(
            int(structure_id), ("unassigned", "unassigned", "unassigned")
        )
        mask = structure_ids == structure_id
        acronyms[mask] = acronym
        names[mask] = name
        major[mask] = major_label
    return structure_ids, acronyms, names, major


def read_merfish(
    path: Path,
    annotation: np.ndarray,
    voxel_size_um: float,
    structure_lookup: dict[int, tuple[str, str, str]],
) -> pd.DataFrame:
    source = ad.read_h5ad(path, backed="r")
    try:
        section = source.obs["brain_section_label"].astype(str)
        if section.nunique() != 1:
            raise ValueError(f"Expected one section in {path}")
        coordinates = np.asarray(source.obsm["X_CCF"])
        structure_ids, acronyms, names, major = assign_ccf_regions(
            coordinates,
            annotation,
            voxel_size_um,
            structure_lookup,
        )
        return pd.DataFrame(
            {
                "source_obs_name": source.obs_names.astype(str),
                "technology": "MERFISH",
                "section": str(section.iloc[0]),
                "domain_label": major,
                "shared_domain_label": major,
                "domain_source": "Allen CCFv3 major region",
                "domain_detail": names,
                "domain_acronym": acronyms,
                "ccf_structure_id": structure_ids,
                "source_file": path.name,
            }
        )
    finally:
        source.file.close()


def read_starmap(path: Path) -> pd.DataFrame:
    source = ad.read_h5ad(path, backed="r")
    try:
        section = source.obs["batch"].astype(str)
        if section.nunique() != 1:
            raise ValueError(f"Expected one section in {path}")
        region = source.obs["Main_molecular_tissue_region"].astype(str)
        shared = region.map(STARMAP_SHARED_REGIONS).fillna("unassigned")
        return pd.DataFrame(
            {
                "source_obs_name": source.obs_names.astype(str),
                "technology": "STARmap PLUS",
                "section": str(section.iloc[0]),
                "domain_label": region.to_numpy(),
                "shared_domain_label": shared.to_numpy(),
                "domain_source": "STARmap PLUS Main_molecular_tissue_region",
                "domain_detail": region.to_numpy(),
                "domain_acronym": region.to_numpy(),
                "ccf_structure_id": np.zeros(source.n_obs, dtype=np.int64),
                "source_file": path.name,
            }
        )
    finally:
        source.file.close()


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build external anatomical-domain labels for MERFISH and STARmap."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ccf-annotation", type=Path, required=True)
    parser.add_argument("--ccf-structure-graph", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    args = parser.parse_args()

    annotation, header = nrrd.read(args.ccf_annotation)
    directions = np.asarray(header["space directions"], dtype=float)
    voxel_sizes = np.linalg.norm(directions, axis=1)
    if not np.allclose(voxel_sizes, voxel_sizes[0]):
        raise ValueError(f"Expected isotropic CCF voxels, found {voxel_sizes}")
    voxel_size_um = float(voxel_sizes[0])
    structure_lookup = make_structure_lookup(
        load_structure_graph(args.ccf_structure_graph)
    )

    merfish_paths = sorted(
        (args.input_root / "merfish_mouse_brain").glob("*.h5ad"),
        key=numeric_file_key,
    )
    starmap_paths = sorted(
        (args.input_root / "starmap_plus_mouse_cns").glob("*.h5ad"),
        key=numeric_file_key,
    )
    if len(merfish_paths) != 50 or len(starmap_paths) != 20:
        raise ValueError(
            f"Expected 50 MERFISH and 20 STARmap files, found "
            f"{len(merfish_paths)} and {len(starmap_paths)}"
        )

    frames = [
        read_merfish(path, annotation, voxel_size_um, structure_lookup)
        for path in merfish_paths
    ]
    frames.extend(read_starmap(path) for path in starmap_paths)
    domains = pd.concat(frames, ignore_index=True)
    if domains["source_obs_name"].duplicated().any():
        raise ValueError("Domain annotation identifiers are not globally unique")
    for column in [
        "technology",
        "section",
        "domain_label",
        "shared_domain_label",
        "domain_source",
        "domain_detail",
        "domain_acronym",
    ]:
        domains[column] = domains[column].astype("category")

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    domains.to_parquet(args.output_parquet, index=False)
    summary = {
        "output": str(args.output_parquet.resolve()),
        "ccf_annotation": str(args.ccf_annotation.resolve()),
        "ccf_structure_graph": str(args.ccf_structure_graph.resolve()),
        "ccf_voxel_size_um": voxel_size_um,
        "n_cells": int(len(domains)),
        "n_sections": int(domains["section"].nunique()),
        "technology_counts": value_counts(domains, "technology"),
        "domain_counts_by_technology": {
            technology: value_counts(frame, "domain_label")
            for technology, frame in domains.groupby("technology", observed=True)
        },
        "shared_domain_counts_by_technology": {
            technology: value_counts(frame, "shared_domain_label")
            for technology, frame in domains.groupby("technology", observed=True)
        },
    }
    args.output_parquet.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
