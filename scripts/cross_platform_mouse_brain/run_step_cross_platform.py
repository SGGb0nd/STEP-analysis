"""Train STEP jointly on the MERFISH and STARmap PLUS mouse-brain atlas."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch

from step import stModel
from prepare_cross_platform_input import load_reference_for_step


MODEL_CONFIG = {
    "module_dim": 30,
    "hidden_dim": 64,
    "n_modules": 32,
    "n_dec_hid_layers": 2,
    "batch_injection_mode": "scale",
    "n_glayers": 4,
    "edge_clip": None,
    "max_neighbors": 11,
    "variational": True,
    "dispersion": "batch-gene",
}

TRAINING_CONFIG = {
    "n_iterations": 3000,
    "graph_batch_size": 1,
    "sample_rate": 2048,
    "sampling": "saint",
    "e2e": True,
    "contrast": True,
    "beta": 1e-3,
    "lr": 1e-3,
    "batch_inference": False,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_state(path: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    try:
        return {
            "root": str(path.resolve()),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"root": str(path.resolve()), "commit": None, "dirty": None}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_step_source_snapshot(step_source: Path, output_dir: Path) -> dict[str, object]:
    candidates = ["step", "pyproject.toml", "uv.lock"]
    members = [name for name in candidates if (step_source / name).exists()]
    if "step" not in members:
        raise FileNotFoundError(f"STEP package source is missing under {step_source}")
    output = output_dir / "step_source_snapshot.tar.zst"
    subprocess.run(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "-I",
            "zstd -T0 -10",
            "-cf",
            str(output),
            *members,
        ],
        cwd=step_source,
        check=True,
    )
    return {
        "path": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train STEP on the published 259-section mouse-brain atlas."
    )
    parser.add_argument("--reference-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-source", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sections-per-technology", type=int)
    parser.add_argument("--cells-per-section", type=int)
    parser.add_argument("--n-iterations", type=int, default=TRAINING_CONFIG["n_iterations"])
    parser.add_argument("--sample-size", type=int, default=TRAINING_CONFIG["sample_rate"])
    parser.add_argument(
        "--spatial-neighbors",
        type=int,
        default=MODEL_CONFIG["max_neighbors"] - 1,
        help="Number of non-self spatial nearest neighbors within each section.",
    )
    parser.add_argument(
        "--export-no-graph",
        action="store_true",
        help="Also export the no-graph embedding after durable spatial outputs are saved.",
    )
    args = parser.parse_args()
    if args.spatial_neighbors < 1:
        parser.error("--spatial-neighbors must be positive")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_adata = load_reference_for_step(
        args.reference_h5ad,
        sections_per_technology=args.sections_per_technology,
        cells_per_section=args.cells_per_section,
        seed=args.seed,
    )
    input_adata.obs["batch"] = input_adata.obs["batch"].astype("category")

    model_config = dict(MODEL_CONFIG)
    model_config["max_neighbors"] = args.spatial_neighbors + 1
    training_config = dict(TRAINING_CONFIG)
    training_config["n_iterations"] = args.n_iterations
    training_config["sample_rate"] = args.sample_size

    started = time.time()
    model = stModel(
        adata=input_adata,
        batch_key="batch",
        coord_keys=("x", "y"),
        layer_key=None,
        filtered=True,
        log_transformed=False,
        n_top_genes=None,
        geneset_to_use=input_adata.var_names.to_list(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        **model_config,
    )
    initialized = time.time()
    model.run(**training_config)
    trained = time.time()

    model_dir = args.output_dir / "model"
    model.save(model_dir, save_adata=False)
    np.save(
        args.output_dir / "step_spatial_embedding.npy",
        np.asarray(model.adata.obsm["X_smoothed"], dtype=np.float32),
    )
    model.adata.obs.to_parquet(args.output_dir / "observations.parquet")
    pd.Series(model.adata.var_names, name="gene").to_csv(
        args.output_dir / "genes.tsv", sep="\t", index=False
    )
    source_snapshot = save_step_source_snapshot(args.step_source, args.output_dir)
    durable_outputs = time.time()

    settings = {
        "reference_h5ad": str(args.reference_h5ad.resolve()),
        "input_shape": [int(model.adata.n_obs), int(model.adata.n_vars)],
        "technology_counts": {
            str(key): int(value)
            for key, value in model.adata.obs["technology"].value_counts().items()
        },
        "n_sections": int(model.adata.obs["batch"].nunique()),
        "seed": args.seed,
        "model": model_config,
        "training": training_config,
        "graph_note": (
            f"max_neighbors={model_config['max_neighbors']} includes the query cell "
            f"in STEP's KDTree call and therefore yields "
            f"{args.spatial_neighbors} non-self nearest neighbors per section."
        ),
        "step_source": git_state(args.step_source),
        "torch_version": torch.__version__,
        "dgl_version": dgl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timing_seconds": {
            "model_initialization": initialized - started,
            "training_and_spatial_inference": trained - initialized,
            "durable_output_export": durable_outputs - trained,
            "no_graph_embedding_export": None,
            "total_before_serialization": durable_outputs - started,
        },
        "outputs": {
            "model": "model/",
            "spatial_embedding": "step_spatial_embedding.npy",
            "observations": "observations.parquet",
            "genes": "genes.tsv",
            "step_source_snapshot": source_snapshot,
        },
        "no_graph_embedding_requested": args.export_no_graph,
    }
    write_json(args.output_dir / "settings.json", settings)

    if args.export_no_graph:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        no_graph_embedding = model.embed(
            tsfmr_out=False,
            as_numpy=True,
            batch_size=64,
        ).astype(np.float32)
        np.save(args.output_dir / "step_no_graph_embedding.npy", no_graph_embedding)
        embedded = time.time()
        settings["timing_seconds"]["no_graph_embedding_export"] = (
            embedded - durable_outputs
        )
        settings["timing_seconds"]["total_before_serialization"] = embedded - started
        settings["outputs"]["no_graph_embedding"] = "step_no_graph_embedding.npy"
        write_json(args.output_dir / "settings.json", settings)
    print(json.dumps(settings, indent=2), flush=True)


if __name__ == "__main__":
    main()
