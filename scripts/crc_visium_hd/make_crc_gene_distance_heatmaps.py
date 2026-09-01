import argparse
import inspect
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import sparse
from scipy.spatial import KDTree


H5AD_PATH = Path("results/visium-hd/cancer_tumor_regions.h5ad")
OUTDIR = Path("workflows/crc_gene_distance_heatmaps")
PERIPHERY_GENES = ["VIM", "COL3A1", "COL1A1", "SPARC", "COL1A2"]
TUMOR_GENES = ["CEACAM5", "EPCAM", "CEACAM6", "PHGR1", "KRT8"]

ORIGINAL_RC_PARAMS = {
    "axes.grid": True,
    "axes.labelsize": 14.0,
    "axes.titlesize": 14.0,
    "figure.dpi": 80.0,
    "figure.figsize": [6.0, 4.5],
    "figure.subplot.bottom": 0.15,
    "figure.subplot.left": 0.18,
    "figure.subplot.right": 0.96,
    "figure.subplot.top": 0.91,
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ],
    "font.size": 14.0,
    "image.interpolation": "antialiased",
    "legend.fontsize": 12.88,
    "legend.handlelength": 0.5,
    "legend.handletextpad": 0.4,
    "savefig.dpi": 300.0,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
}


def compatible_plot_style() -> dict[str, object]:
    """Return the original style using keys supported by this Matplotlib."""
    return {
        key: value
        for key, value in ORIGINAL_RC_PARAMS.items()
        if key in matplotlib.rcParams
    }


def heatmap_image_kwargs(expression: np.ndarray) -> dict[str, object]:
    """Keep color scaling fixed while tolerating older Matplotlib releases."""
    kwargs: dict[str, object] = {
        "aspect": "auto",
        "cmap": "Spectral_r",
        "interpolation": "antialiased",
        "vmin": float(np.nanmin(expression)),
        "vmax": float(np.quantile(expression, 0.95)),
    }
    if "interpolation_stage" in inspect.signature(Axes.imshow).parameters:
        kwargs["interpolation_stage"] = "data"
    return kwargs


def recover_normalized_expression(
    adata: ad.AnnData,
    rows: np.ndarray,
    gene_indices: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    matrix = adata.X[rows, :]
    if sparse.issparse(matrix):
        counts = matrix.tocsr(copy=True)
        counts.data = np.rint(np.expm1(counts.data.astype(np.float64)))
        library_size = np.asarray(counts.sum(axis=1)).ravel()
        gene_counts = counts[:, gene_indices].toarray()
    else:
        counts = np.rint(np.expm1(np.asarray(matrix, dtype=np.float64)))
        library_size = counts.sum(axis=1)
        gene_counts = counts[:, gene_indices]

    normalized = np.divide(
        gene_counts * target_sum,
        library_size[:, None],
        out=np.zeros_like(gene_counts, dtype=np.float64),
        where=library_size[:, None] > 0,
    )
    return np.log1p(normalized)


def sample_scale_summary(adata: ad.AnnData) -> pd.DataFrame:
    rows = []
    for batch, spatial_metadata in adata.uns["spatial"].items():
        scale_factors = spatial_metadata["scalefactors"]
        rows.append(
            {
                "batch": batch,
                "microns_per_pixel": float(scale_factors["microns_per_pixel"]),
                "spot_diameter_fullres_px": float(
                    scale_factors["spot_diameter_fullres"]
                ),
                "bin_size_um": float(scale_factors["bin_size_um"]),
            }
        )
    return pd.DataFrame(rows)


def export_batch_figure(
    adata: ad.AnnData,
    batch: str,
    all_genes: list[str],
    gene_indices: np.ndarray,
    threshold_px: float,
    target_sum: float,
    outdir: Path,
) -> dict[str, object] | None:
    batch_values = adata.obs["batch"].astype(str).to_numpy()
    zonation_values = adata.obs["zonation"].astype(str).to_numpy()
    periphery_rows = np.flatnonzero(
        (zonation_values == "Tumor Periphery (domain 9)")
        & (batch_values == batch)
    )
    tumor_rows = np.flatnonzero(
        (zonation_values == "Tumor (domain 5)") & (batch_values == batch)
    )
    if len(periphery_rows) == 0 or len(tumor_rows) == 0:
        return None

    spatial = np.asarray(adata.obsm["spatial"])
    distances_px = KDTree(spatial[tumor_rows]).query(
        spatial[periphery_rows], k=1
    )[0]
    keep = distances_px <= threshold_px
    if not np.any(keep):
        return None

    selected_rows = periphery_rows[keep]
    expression = recover_normalized_expression(
        adata=adata,
        rows=selected_rows,
        gene_indices=gene_indices,
        target_sum=target_sum,
    )

    microns_per_pixel = float(
        adata.uns["spatial"][batch]["scalefactors"]["microns_per_pixel"]
    )
    distances_um = distances_px[keep] * microns_per_pixel
    order = np.argsort(distances_um, kind="stable")
    sorted_distances_um = distances_um[order]
    expression = expression[order]

    rounded_distances = np.round(sorted_distances_um)
    distance_labels, shell_starts = np.unique(
        rounded_distances, return_index=True
    )

    with matplotlib.rc_context(rc=compatible_plot_style()):
        fig, axis = plt.subplots(figsize=(15, 4))
        axis.grid(False)
        image = axis.imshow(
            expression.T,
            **heatmap_image_kwargs(expression),
        )
        colorbar = plt.colorbar(
            image, label="Library-size-normalized log expression"
        )
        colorbar.ax.tick_params(labelsize=9)

        axis.set_yticks(np.arange(len(all_genes)))
        axis.set_yticklabels(all_genes, rotation=0, ha="right")
        axis.axhline(
            y=len(PERIPHERY_GENES) - 0.5,
            color="red",
            linestyle="-",
            linewidth=1,
        )
        for shell_start in shell_starts:
            axis.axvline(
                x=shell_start,
                color="black",
                linestyle="--",
                alpha=0.9,
            )

        axis.set_xticks(shell_starts)
        axis.set_xticklabels(
            [f"{distance:.0f}" + r"$\mu$m" for distance in distance_labels],
            ha="center",
        )
        axis.set_title(
            f"Periphery cells vs distance to tumor - "
            f"{batch.split('_')[1].capitalize()}"
        )
        plt.tight_layout()

        pdf_path = outdir / (
            f"{batch}_periphery_vs_tumor_distance_corrected.pdf"
        )
        png_path = outdir / (
            f"{batch}_periphery_vs_tumor_distance_corrected.png"
        )
        fig.savefig(pdf_path, dpi=300)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

    return {
        "batch": batch,
        "n_periphery_total": int(len(periphery_rows)),
        "n_tumor_total": int(len(tumor_rows)),
        "n_kept": int(np.sum(keep)),
        "threshold_px": float(threshold_px),
        "threshold_um": float(threshold_px * microns_per_pixel),
        "microns_per_pixel": microns_per_pixel,
        "distance_um_min": float(sorted_distances_um.min()),
        "distance_um_max": float(sorted_distances_um.max()),
        "distance_shells_um": distance_labels.tolist(),
        "pdf_path": str(pdf_path),
        "png_path": str(png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, default=H5AD_PATH)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--threshold-px", type=float, default=100.0)
    parser.add_argument("--target-sum", type=float, default=1e5)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.h5ad, backed="r")
    all_genes = PERIPHERY_GENES + TUMOR_GENES
    gene_indices = adata.var_names.get_indexer(all_genes)
    if np.any(gene_indices < 0):
        missing = np.asarray(all_genes)[gene_indices < 0].tolist()
        raise ValueError(f"Missing genes: {missing}")

    summaries = []
    for batch in ["cancer_p1", "cancer_p2", "cancer_p5"]:
        summary = export_batch_figure(
            adata=adata,
            batch=batch,
            all_genes=all_genes,
            gene_indices=gene_indices,
            threshold_px=args.threshold_px,
            target_sum=args.target_sum,
            outdir=args.outdir,
        )
        if summary is not None:
            summaries.append(summary)

    sample_scale_summary(adata).to_csv(
        args.outdir / "scale_summary.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        args.outdir / "per_batch_summary.csv", index=False
    )
    (args.outdir / "gene_list.json").write_text(
        json.dumps(
            {
                "periphery_genes": PERIPHERY_GENES,
                "tumor_genes": TUMOR_GENES,
                "normalization": (
                    "expm1(log1p_counts), per-spot library-size normalization, "
                    "log1p"
                ),
                "target_sum": args.target_sum,
            },
            indent=2,
        )
    )
    (args.outdir / "plot_environment.json").write_text(
        json.dumps(
            {
                "matplotlib": matplotlib.__version__,
                "backend": matplotlib.get_backend(),
                "rc_params": compatible_plot_style(),
                "imshow_interpolation_stage_supported": (
                    "interpolation_stage"
                    in inspect.signature(Axes.imshow).parameters
                ),
            },
            indent=2,
        )
    )
    adata.file.close()


if __name__ == "__main__":
    main()
