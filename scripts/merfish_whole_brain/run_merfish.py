"""Run STEP on the 8.4-million-spot whole-brain MERFISH dataset."""

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")

import scanpy as sc


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parents[1]
STEP_ROOT = REPO_ROOT
DEFAULT_INPUT_DIR = Path("/storage/yangjianLab/lilounan/data/wo_imputation").resolve()
DEFAULT_MERGED_PATH = DEFAULT_INPUT_DIR / "WB_MERFISH_wo_imputation_merged.h5ad"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "merfish_wo_imputation_merged"
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "merfish_wo_imputation_merged"
DEFAULT_SLURM_DIR = REPO_ROOT / "slurm"
DEFAULT_PROCESSED_PATH = DEFAULT_OUTPUT_DIR / "merfish_processed.h5ad"


def configure_torch_cuda_workarounds() -> dict[str, bool]:
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)

    return {
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled()
        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
        else False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run STEP on the merged MERFISH wo_imputation dataset. "
            "Without --execute, this script submits a Slurm job."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the training job inside a Slurm allocation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Slurm script instead of submitting it.",
    )
    parser.add_argument(
        "--force-merge",
        action="store_true",
        help="Rebuild the merged h5ad even if a cache already exists.",
    )
    parser.add_argument(
        "--plot-all-sections",
        action="store_true",
        help="Export one spatial plot per section after clustering.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the source h5ad files.",
    )
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=DEFAULT_MERGED_PATH,
        help="Cache path for the merged h5ad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model outputs.",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help="Path to save the processed AnnData after training.",
    )
    parser.add_argument(
        "--batch-key",
        default="brain_section_label",
        help="obs column used to separate sections.",
    )
    parser.add_argument(
        "--label-key",
        default="major_brain_region",
        help="obs column used for the reference plot/export.",
    )
    parser.add_argument(
        "--ignore-label-value",
        action="append",
        default=[],
        help="Label values in label-key to ignore when inferring n_clusters and computing metrics. Repeatable.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of spatial domains. Defaults to the non-null class count of label-key in the merged h5ad.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3000,
        help="Training iterations passed to STEP.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=2048, help="Sample rate passed to STEP."
    )
    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=1,
        help="graph_batch_size passed to STEP.",
    )
    parser.add_argument(
        "--max-neighbors", type=int, default=20, help="max_neighbors passed to STEP."
    )
    parser.add_argument(
        "--device", default="cuda", help="Execution device passed to STEP."
    )
    parser.add_argument(
        "--partition", default="a800,l40s", help="Slurm GPU partition list."
    )
    parser.add_argument("--qos", default="gpu-huge", help="Slurm QoS.")
    parser.add_argument("--gres", default="gpu:1", help="Slurm GPU request.")
    parser.add_argument(
        "--cpus-per-task", type=int, default=16, help="Slurm CPU request."
    )
    parser.add_argument("--mem", default="500G", help="Slurm memory request.")
    parser.add_argument("--time", default="48:00:00", help="Slurm walltime.")
    parser.add_argument(
        "--job-name", default="step_merfish_merged", help="Slurm job name."
    )
    return parser.parse_args()


def discover_input_files(input_dir: Path, merged_path: Path) -> list[Path]:
    merged_resolved = merged_path.resolve(strict=False)
    candidates = sorted(path.resolve() for path in input_dir.glob("*.h5ad"))
    files = [
        path
        for path in candidates
        if path.resolve(strict=False) != merged_resolved
        and "merged" not in path.stem.lower()
        and path.name.startswith("WB_MERFISH_")
    ]
    if not files:
        files = [
            path
            for path in candidates
            if path.resolve(strict=False) != merged_resolved
            and "merged" not in path.stem.lower()
        ]
    if not files:
        raise FileNotFoundError(f"No source h5ad files found under {input_dir}.")
    return files


def merged_cache_is_fresh(merged_path: Path, input_files: list[Path]) -> bool:
    if not merged_path.exists():
        return False
    merged_mtime = merged_path.stat().st_mtime
    return all(merged_mtime >= path.stat().st_mtime for path in input_files)


def detect_spatial_key(adata: ad.AnnData, dataset_name: str) -> str:
    for key in ("spatial", "X_spatial_coords", "X_spatial"):
        if key in adata.obsm and adata.obsm[key].shape[1] >= 2:
            return key
    raise KeyError(
        f"{dataset_name} does not contain spatial, X_spatial_coords, or X_spatial in adata.obsm."
    )


def _coerce_plain_index(index: pd.Index) -> pd.Index:
    values = index.astype(str).to_numpy(dtype=object)
    return pd.Index(values, dtype=object, name=index.name)


def _coerce_plain_categorical(series: pd.Series) -> pd.Series:
    cat = series.cat
    categories = pd.Index(
        cat.categories.astype(str).to_numpy(dtype=object), dtype=object
    )
    coerced = pd.Categorical.from_codes(
        cat.codes, categories=categories, ordered=cat.ordered
    )
    return pd.Series(coerced, index=series.index, name=series.name)


def sanitize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    cols_to_rename = {"_index": "original_index"} if "_index" in frame.columns else {}
    if cols_to_rename:
        frame.rename(columns=cols_to_rename, inplace=True)

    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            frame[column] = _coerce_plain_categorical(series)
        elif pd.api.types.is_string_dtype(series.dtype):
            frame[column] = series.astype(object)

    frame.index = _coerce_plain_index(frame.index)
    return frame


def sanitize_index_fields(adata: ad.AnnData) -> ad.AnnData:
    sanitize_dataframe(adata.obs)
    sanitize_dataframe(adata.var)

    if adata.raw is not None:
        sanitize_dataframe(adata.raw._var)

    return adata


def normalize_adata(adata: ad.AnnData, dataset_name: str, batch_key: str) -> ad.AnnData:
    spatial_key = detect_spatial_key(adata, dataset_name)
    coords = adata.obsm[spatial_key][:, :2].copy()
    adata.obs_names_make_unique()
    adata.obsm["spatial"] = coords
    adata.obs[["x", "y"]] = coords
    if batch_key not in adata.obs.columns:
        raise KeyError(f"{dataset_name} is missing required obs column {batch_key!r}.")
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    for key in ("X_spatial", "X_spatial_coords"):
        if key in adata.obsm:
            del adata.obsm[key]
    return sanitize_index_fields(adata)


def build_merged_adata(input_files: list[Path], batch_key: str) -> ad.AnnData:
    datasets: list[ad.AnnData] = []
    for path in input_files:
        print(f"Loading {path} ...", flush=True)
        adata = sc.read_h5ad(path)
        datasets.append(normalize_adata(adata, path.stem, batch_key=batch_key))

    merged = ad.concat(
        datasets,
        join="inner",
        label="source_h5ad",
        keys=[path.stem for path in input_files],
        index_unique="__",
        merge="same",
        uns_merge="same",
    )
    merged.obs_names_make_unique()
    merged.obs[batch_key] = merged.obs[batch_key].astype("category")
    if "source_h5ad" in merged.obs.columns:
        merged.obs["source_h5ad"] = merged.obs["source_h5ad"].astype("category")
    return sanitize_index_fields(merged)


def load_or_build_merged_adata(
    args: argparse.Namespace,
) -> tuple[ad.AnnData, list[Path]]:
    input_files = discover_input_files(args.input_dir, args.merged_path)
    args.merged_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.force_merge and merged_cache_is_fresh(args.merged_path, input_files):
        print(f"Reusing merged dataset: {args.merged_path}", flush=True)
        adata = sc.read_h5ad(args.merged_path)
        return normalize_adata(
            adata, args.merged_path.stem, batch_key=args.batch_key
        ), input_files

    print("Building merged MERFISH dataset ...", flush=True)
    merged = build_merged_adata(input_files, batch_key=args.batch_key)
    tmp_path = args.merged_path.with_suffix(".tmp.h5ad")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"Writing merged cache to {args.merged_path}", flush=True)
    merged.write_h5ad(tmp_path, compression="gzip")
    tmp_path.replace(args.merged_path)
    return merged, input_files


def write_run_metadata(
    args: argparse.Namespace, adata: ad.AnnData, input_files: list[Path]
) -> None:
    metadata = {
        "input_files": [str(path) for path in input_files],
        "merged_h5ad": str(args.merged_path),
        "processed_h5ad": str(args.processed_path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_sections": int(adata.obs[args.batch_key].nunique()),
        "batch_key": args.batch_key,
        "label_key": args.label_key,
        "ignore_label_values": args.ignore_label_value,
        "n_clusters": args.n_clusters,
        "iterations": args.iterations,
        "sample_rate": args.sample_rate,
        "graph_batch_size": args.graph_batch_size,
        "max_neighbors": args.max_neighbors,
        "device": args.device,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def export_domain_tables(args: argparse.Namespace, adata: ad.AnnData) -> pd.DataFrame:
    export_cols = [args.batch_key, "source_h5ad", "domain", "x", "y"]
    if args.label_key in adata.obs.columns and args.label_key not in export_cols:
        export_cols.append(args.label_key)

    export_frame = adata.obs.loc[:, export_cols].copy()
    export_frame.to_csv(
        args.output_dir / "domain_assignments.csv.gz", compression="gzip"
    )

    domain_counts = (
        adata.obs.groupby("domain", observed=True)
        .size()
        .rename("n_obs")
        .sort_values(ascending=False)
        .reset_index()
    )
    domain_counts.to_csv(args.output_dir / "domain_counts.csv", index=False)

    section_domain_counts = (
        adata.obs.groupby([args.batch_key, "domain"], observed=True)
        .size()
        .rename("n_obs")
        .reset_index()
    )
    section_domain_counts.to_csv(
        args.output_dir / "section_domain_counts.csv.gz",
        index=False,
        compression="gzip",
    )
    return export_frame


def export_section_plots(args: argparse.Namespace, adata: ad.AnnData) -> None:
    import matplotlib.pyplot as plt

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_keys = ["domain"]
    if args.label_key in adata.obs.columns:
        plot_keys.append(args.label_key)

    batches = adata.obs[args.batch_key].cat.categories
    for color in plot_keys:
        for batch in batches:
            subset = adata[adata.obs[args.batch_key] == batch].copy()
            ax = sc.pl.embedding(
                subset,
                basis="spatial",
                color=color,
                title=f"{batch} {color}",
                show=False,
            )
            figure = ax.figure if hasattr(ax, "figure") else plt.gcf()
            output_path = figures_dir / f"{batch}_{color}.png"
            figure.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(figure)


def save_processed_adata(args: argparse.Namespace, adata: ad.AnnData) -> None:
    args.processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed = sanitize_index_fields(adata)
    print(f"Writing processed AnnData to {args.processed_path}", flush=True)
    tmp_path = args.processed_path.with_suffix(".tmp.h5ad")
    if tmp_path.exists():
        tmp_path.unlink()
    processed.write_h5ad(tmp_path, compression="gzip")
    tmp_path.replace(args.processed_path)


def render_summary_outputs(
    assignments: pd.DataFrame, output_dir: Path, ignore_label_values: list[str]
) -> None:
    from plot_merfish_results import render_outputs

    print("Rendering summary plots and metrics ...", flush=True)
    render_outputs(assignments, output_dir, ignore_label_values)


def infer_n_clusters(
    adata: ad.AnnData,
    label_key: str,
    requested_n_clusters: int | None,
    ignore_label_values: list[str],
) -> int:
    if label_key not in adata.obs.columns:
        raise KeyError(
            f"Cannot infer n_clusters: {label_key!r} not found in adata.obs."
        )

    labels = adata.obs[label_key].dropna()
    if ignore_label_values:
        labels = labels[~labels.astype(str).isin({str(v) for v in ignore_label_values})]
    inferred = int(labels.nunique())
    if requested_n_clusters is None:
        print(f"Using n_clusters={inferred} inferred from {label_key}", flush=True)
        return inferred

    if requested_n_clusters != inferred:
        raise ValueError(
            f"n_clusters must match the non-null class count of {label_key!r} in the merged h5ad. "
            f"Requested {requested_n_clusters}, inferred {inferred}."
        )
    return requested_n_clusters


def run_analysis(args: argparse.Namespace) -> None:
    from step import stModel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        print(
            {"torch_cuda_workarounds": configure_torch_cuda_workarounds()}, flush=True
        )
    sc.set_figure_params(dpi=300, figsize=(6, 4.5), frameon=False)
    sc.settings.figdir = str(args.output_dir / "figures")

    adata, input_files = load_or_build_merged_adata(args)
    if args.label_key in adata.obs.columns:
        adata.obs[args.label_key] = adata.obs[args.label_key].astype("category")
    args.n_clusters = infer_n_clusters(
        adata, args.label_key, args.n_clusters, args.ignore_label_value
    )

    print(
        f"Merged dataset ready: {adata.n_obs} cells x {adata.n_vars} genes across "
        f"{adata.obs[args.batch_key].nunique()} sections.",
        flush=True,
    )
    write_run_metadata(args, adata, input_files)

    model = stModel(
        adata=adata,
        n_top_genes=None,
        batch_key=args.batch_key,
        edge_clip=None,
        max_neighbors=args.max_neighbors,
        device=args.device,
        filtered=True,
        coord_keys=("x", "y"),
    )
    model.run(
        n_iterations=args.iterations,
        sample_rate=args.sample_rate,
        graph_batch_size=args.graph_batch_size,
    )
    model.cluster(n_clusters=args.n_clusters, seed=0)

    save_processed_adata(args, model.adata)
    assignments = export_domain_tables(args, model.adata)
    render_summary_outputs(assignments, args.output_dir, args.ignore_label_value)
    if args.plot_all_sections:
        export_section_plots(args, model.adata)

    print(f"Finished. Outputs written to {args.output_dir}", flush=True)


def build_execute_command(args: argparse.Namespace) -> list[str]:
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise RuntimeError("uv was not found in PATH.")

    command = [
        uv_bin,
        "run",
        "--directory",
        str(STEP_ROOT),
        "python",
        str(SCRIPT_PATH),
        "--execute",
        "--input-dir",
        str(args.input_dir),
        "--merged-path",
        str(args.merged_path),
        "--output-dir",
        str(args.output_dir),
        "--processed-path",
        str(args.processed_path),
        "--batch-key",
        args.batch_key,
        "--label-key",
        args.label_key,
        *[
            item
            for value in args.ignore_label_value
            for item in ("--ignore-label-value", value)
        ],
        "--iterations",
        str(args.iterations),
        "--sample-rate",
        str(args.sample_rate),
        "--graph-batch-size",
        str(args.graph_batch_size),
        "--max-neighbors",
        str(args.max_neighbors),
        "--device",
        args.device,
    ]
    if args.n_clusters is not None:
        command.extend(["--n-clusters", str(args.n_clusters)])
    if args.force_merge:
        command.append("--force-merge")
    if args.plot_all_sections:
        command.append("--plot-all-sections")
    return command


def render_slurm_script(args: argparse.Namespace) -> str:
    command = " ".join(shlex.quote(part) for part in build_execute_command(args))
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH -J {args.job_name}",
            f"#SBATCH -D {SCRIPT_DIR}",
            f"#SBATCH -o {DEFAULT_LOG_DIR / (args.job_name + '_%j.out')}",
            f"#SBATCH -e {DEFAULT_LOG_DIR / (args.job_name + '_%j.err')}",
            f"#SBATCH -p {args.partition}",
            f"#SBATCH --qos={args.qos}",
            f"#SBATCH --gres={args.gres}",
            f"#SBATCH --mem={args.mem}",
            f"#SBATCH -c {args.cpus_per_task}",
            f"#SBATCH --time={args.time}",
            "",
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(DEFAULT_LOG_DIR))}",
            f"mkdir -p {shlex.quote(str(args.output_dir))}",
            f"cd {shlex.quote(str(SCRIPT_DIR))}",
            'echo "Job started on $(hostname) at $(date)"',
            command,
            'echo "Job finished at $(date)"',
            "",
        ]
    )


def submit_job(args: argparse.Namespace) -> None:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_SLURM_DIR.mkdir(parents=True, exist_ok=True)
    slurm_script_path = DEFAULT_SLURM_DIR / "run_merfish_merged.sbatch"
    script_text = render_slurm_script(args)

    if args.dry_run:
        print(script_text)
        return

    slurm_script_path.write_text(script_text, encoding="utf-8")
    result = subprocess.run(
        ["sbatch", "--parsable", str(slurm_script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Submitted Slurm job {result.stdout.strip()} via {slurm_script_path}")


def main() -> None:
    args = parse_args()
    if args.execute:
        run_analysis(args)
        return
    submit_job(args)


if __name__ == "__main__":
    main()
