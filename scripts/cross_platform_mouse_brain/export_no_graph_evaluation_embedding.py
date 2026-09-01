"""Export no-graph STEP embeddings for a matched evaluation subset."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from step import stModel

from prepare_cross_platform_input import normalize_technology, validate_reference


def take_rows(matrix, rows: np.ndarray):
    order = np.argsort(rows)
    sorted_rows = rows[order]
    values = matrix[sorted_rows]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return values[inverse]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export no-graph embeddings from a saved cross-technology STEP model."
    )
    parser.add_argument("--reference-h5ad", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--evaluation-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    evaluation = pd.read_parquet(args.evaluation_cells)
    required = {"source_obs_name", "nichecompass_index"}
    missing = sorted(required.difference(evaluation.columns))
    if missing:
        raise ValueError(f"Evaluation table is missing columns: {missing}")
    evaluation_rows = evaluation["nichecompass_index"].to_numpy(dtype=np.int64)

    source = ad.read_h5ad(args.reference_h5ad, backed="r")
    try:
        validate_reference(source)
        technology = normalize_technology(source.obs["dataset"])
        section = source.obs["section"].astype(str).to_numpy(dtype=str)
        technology_text = np.asarray(technology.astype(str), dtype=str)
        full_batch = np.char.add(np.char.add(technology_text, "::"), section)
        batch_categories = pd.Categorical(full_batch).categories

        evaluation_batches = set(full_batch[evaluation_rows])
        extra_rows = []
        for batch in batch_categories:
            if batch not in evaluation_batches:
                extra_rows.append(int(np.flatnonzero(full_batch == batch)[0]))
        model_rows = np.concatenate(
            [evaluation_rows, np.asarray(extra_rows, dtype=np.int64)]
        )

        counts = take_rows(source.X, model_rows)
        if sparse.issparse(counts):
            counts = counts.tocsr().astype(np.float32)
        else:
            counts = np.asarray(counts, dtype=np.float32)
        spatial = np.asarray(take_rows(source.obsm["spatial"], model_rows), dtype=np.float32)
        selected_names = source.obs_names[model_rows].astype(str)
        expected_names = evaluation["source_obs_name"].astype(str).to_numpy()
        if not np.array_equal(selected_names[: len(evaluation)], expected_names):
            raise ValueError("Evaluation identifiers do not match reference rows")

        selected_batch = pd.Categorical(
            full_batch[model_rows], categories=batch_categories
        )
        selected_technology = pd.Categorical(
            technology_text[model_rows], categories=["MERFISH", "STARmap PLUS"]
        )
        selected_section = pd.Categorical(section[model_rows])
        obs = pd.DataFrame(
            {
                "source_obs_name": selected_names,
                "technology": selected_technology,
                "section": selected_section,
                "batch": selected_batch,
                "x": spatial[:, 0],
                "y": spatial[:, 1],
            },
            index=pd.Index(selected_names, name=source.obs_names.name),
        )
        input_adata = ad.AnnData(
            X=counts,
            obs=obs,
            var=source.var.copy(),
            obsm={"spatial": spatial},
        )
    finally:
        source.file.close()

    model = stModel.load(args.model_dir, adata=input_adata)
    embedding = model.embed(
        tsfmr_out=False,
        as_numpy=True,
        batch_size=args.batch_size,
    )[: len(evaluation)].astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output_dir / "step_no_graph_evaluation_embedding.npy"
    index_path = args.output_dir / "step_no_graph_evaluation_cells.parquet"
    np.save(embedding_path, embedding)
    evaluation[["source_obs_name"]].to_parquet(index_path, index=False)
    summary = {
        "reference_h5ad": str(args.reference_h5ad.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "evaluation_cells": str(args.evaluation_cells.resolve()),
        "n_evaluation_cells": int(len(evaluation)),
        "n_model_cells": int(input_adata.n_obs),
        "n_batches": int(input_adata.obs["batch"].nunique()),
        "embedding_shape": list(embedding.shape),
        "embedding": str(embedding_path.resolve()),
        "index": str(index_path.resolve()),
    }
    (args.output_dir / "step_no_graph_evaluation_embedding.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
