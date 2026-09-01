"""Prepare the exact published NicheCompass mouse-brain reference for STEP."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


EXPECTED_SHAPE = (9_471_568, 432)
EXPECTED_DATASET_COUNTS = {"merfish": 8_380_288, "starmap": 1_091_280}
EXPECTED_SECTIONS = 259


def normalize_technology(values: pd.Series) -> pd.Categorical:
    mapped = values.astype(str).str.lower().map(
        {"merfish": "MERFISH", "starmap": "STARmap PLUS"}
    )
    if mapped.isna().any():
        unknown = sorted(values.loc[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unknown technology labels: {unknown}")
    return pd.Categorical(mapped, categories=["MERFISH", "STARmap PLUS"])


def select_smoke_rows(
    obs: pd.DataFrame,
    sections_per_technology: int | None,
    cells_per_section: int | None,
    seed: int,
) -> np.ndarray | None:
    if sections_per_technology is None and cells_per_section is None:
        return None

    rng = np.random.default_rng(seed)
    technology = normalize_technology(obs["dataset"])
    section = obs["section"].astype(str).to_numpy()
    selected = []
    for current_technology in ["MERFISH", "STARmap PLUS"]:
        technology_mask = np.asarray(technology == current_technology)
        sections = list(dict.fromkeys(section[technology_mask].tolist()))
        if sections_per_technology is not None:
            sections = sections[:sections_per_technology]
        for current_section in sections:
            rows = np.flatnonzero(technology_mask & (section == current_section))
            if cells_per_section is not None and len(rows) > cells_per_section:
                rows = rng.choice(rows, cells_per_section, replace=False)
            selected.append(np.sort(rows))
    if not selected:
        raise ValueError("Smoke-test selection produced no cells")
    return np.sort(np.concatenate(selected))


def validate_reference(source: ad.AnnData) -> None:
    if source.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected reference shape {EXPECTED_SHAPE}, found {source.shape}")
    required_obs = {"dataset", "section", "sample"}
    missing_obs = sorted(required_obs.difference(source.obs.columns))
    if missing_obs:
        raise ValueError(f"Reference is missing obs columns: {missing_obs}")
    counts = {
        str(key): int(value)
        for key, value in source.obs["dataset"].value_counts().items()
    }
    if counts != EXPECTED_DATASET_COUNTS:
        raise ValueError(f"Unexpected dataset counts: {counts}")
    if source.obs["section"].nunique() != EXPECTED_SECTIONS:
        raise ValueError(
            f"Expected {EXPECTED_SECTIONS} sections, found "
            f"{source.obs['section'].nunique()}"
        )
    if "spatial" not in source.obsm:
        raise ValueError("Reference is missing obsm['spatial']")


def load_reference_for_step(
    reference_h5ad: Path,
    sections_per_technology: int | None = None,
    cells_per_section: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    source = ad.read_h5ad(reference_h5ad, backed="r")
    try:
        validate_reference(source)
        rows = select_smoke_rows(
            source.obs,
            sections_per_technology,
            cells_per_section,
            seed,
        )
        if rows is None:
            counts = source.X[:]
            spatial = np.asarray(source.obsm["spatial"], dtype=np.float32)
            source_obs = source.obs
            source_names = source.obs_names
        else:
            counts = source.X[rows]
            spatial = np.asarray(source.obsm["spatial"][rows], dtype=np.float32)
            source_obs = source.obs.iloc[rows]
            source_names = source.obs_names[rows]

        if sparse.issparse(counts):
            counts = counts.tocsr().astype(np.float32)
        else:
            counts = np.asarray(counts, dtype=np.float32)
        nonzero = counts.data if sparse.issparse(counts) else counts[counts != 0]
        if nonzero.size and not np.allclose(nonzero, np.rint(nonzero)):
            raise ValueError("Reference X is not integer-valued count data")

        technology = normalize_technology(source_obs["dataset"])
        section = source_obs["section"].astype(str).to_numpy(dtype=str)
        technology_text = np.asarray(technology.astype(str), dtype=str)
        batch = np.char.add(np.char.add(technology_text, "::"), section)
        obs = pd.DataFrame(
            {
                "source_obs_name": source_names.astype(str),
                "technology": technology,
                "section": pd.Categorical(section),
                "batch": pd.Categorical(batch),
                "nichecompass_sample": source_obs["sample"].astype(str).to_numpy(),
                "x": spatial[:, 0],
                "y": spatial[:, 1],
            },
            index=pd.Index(source_names.astype(str), name=source.obs_names.name),
        )
        result = ad.AnnData(
            X=counts,
            obs=obs,
            var=source.var.copy(),
            obsm={"spatial": spatial},
        )
        result.uns["source_reference"] = {
            "path": str(reference_h5ad.resolve()),
            "formal_reference_shape": list(EXPECTED_SHAPE),
            "selection": "all cells" if rows is None else "smoke-test subset",
        }
        return result
    finally:
        source.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the exact 259-section published NicheCompass reference."
    )
    parser.add_argument("--reference-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--sections-per-technology", type=int)
    parser.add_argument("--cells-per-section", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prepared = load_reference_for_step(
        args.reference_h5ad,
        sections_per_technology=args.sections_per_technology,
        cells_per_section=args.cells_per_section,
        seed=args.seed,
    )
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(args.output_h5ad, compression="gzip")
    summary = {
        "source": str(args.reference_h5ad.resolve()),
        "output": str(args.output_h5ad.resolve()),
        "shape": [int(prepared.n_obs), int(prepared.n_vars)],
        "technology_counts": {
            str(key): int(value)
            for key, value in prepared.obs["technology"].value_counts().items()
        },
        "n_sections": int(prepared.obs["batch"].nunique()),
    }
    args.output_h5ad.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
