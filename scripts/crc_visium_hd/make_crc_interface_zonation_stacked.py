#!/usr/bin/env python3
"""Render distance-dependent CRC interface expression profiles across biological replicates."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from PIL import Image
from scipy.spatial import KDTree

matplotlib.use("Agg")


MICRONS_PER_PIXEL = 0.27380817798463214
STROMAL_GENES = ["VIM", "COL3A1", "COL1A1", "SPARC", "COL1A2"]
EPITHELIAL_GENES = ["CEACAM5", "EPCAM", "CEACAM6", "PHGR1", "KRT8"]
ZONE_COLORS = ["red", "green", "blue"]
ZONE_LABELS = ["Inner", "Core", "Outer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/visium-hd/cancer_tumor_regions.h5ad"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workflows/crc_interface_zonation/crc_interface_zonation_stacked.png"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_data(path: Path):
    adata = sc.read_h5ad(path)
    return adata


def _expr_vector(adata, mask, gene: str) -> np.ndarray:
    x = adata[mask, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).reshape(-1)


def analyze_interface_zones(adata, batch: str):
    periphery_mask = (
        (adata.obs["zonation"] == "Tumor Periphery (domain 9)")
        & (adata.obs["batch"] == batch)
    ).to_numpy()
    tumor_mask = (
        (adata.obs["zonation"] == "Tumor (domain 5)")
        & (adata.obs["batch"] == batch)
    ).to_numpy()

    periphery_coords = np.asarray(adata.obsm["spatial"][periphery_mask])
    tumor_coords = np.asarray(adata.obsm["spatial"][tumor_mask])
    tumor_tree = KDTree(tumor_coords)
    dist_to_tumor = tumor_tree.query(periphery_coords, k=1)[0] * MICRONS_PER_PIXEL

    inner_thresh = np.percentile(dist_to_tumor, 33)
    outer_thresh = np.percentile(dist_to_tumor, 67)

    inner_mask = dist_to_tumor <= inner_thresh
    outer_mask = dist_to_tumor >= outer_thresh
    core_mask = ~(inner_mask | outer_mask)
    zone_masks = [inner_mask, core_mask, outer_mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    zone_codes = np.zeros(len(periphery_coords))
    zone_codes[inner_mask] = 0
    zone_codes[core_mask] = 1
    zone_codes[outer_mask] = 2
    cmap = plt.cm.colors.ListedColormap(ZONE_COLORS)
    ax.scatter(
        periphery_coords[:, 0] * MICRONS_PER_PIXEL,
        periphery_coords[:, 1] * MICRONS_PER_PIXEL,
        c=zone_codes,
        cmap=cmap,
        s=1,
        alpha=0.5,
        vmin=0,
        vmax=2,
    )
    ax.scatter([], [], c="red", label=f"Inner (<{inner_thresh:.0f}um)")
    ax.scatter([], [], c="green", label="Core")
    ax.scatter([], [], c="blue", label=f"Outer (>{outer_thresh:.0f}um)")
    ax.legend()
    ax.set_aspect("equal")
    ax.set_title("Periphery zones")
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.invert_yaxis()

    ax = axes[1]
    x = np.arange(len(STROMAL_GENES))
    width = 0.25
    for zone_idx, (zone_name, zone_mask) in enumerate(zip(ZONE_LABELS, zone_masks)):
        means = []
        for gene in STROMAL_GENES:
            expr = _expr_vector(adata, periphery_mask, gene)
            means.append(expr[zone_mask].mean())
        ax.bar(
            x + zone_idx * width,
            means,
            width,
            label=zone_name,
            color=ZONE_COLORS[zone_idx],
            alpha=0.7,
        )
    ax.set_xticks(x + width)
    ax.set_xticklabels(STROMAL_GENES, rotation=45, ha="right")
    ax.set_ylabel("Mean expression")
    ax.set_title("Stromal markers by zone")
    ax.legend()

    ax = axes[2]
    x = np.arange(len(EPITHELIAL_GENES))
    for zone_idx, (zone_name, zone_mask) in enumerate(zip(ZONE_LABELS, zone_masks)):
        means = []
        for gene in EPITHELIAL_GENES:
            expr = _expr_vector(adata, periphery_mask, gene)
            means.append(expr[zone_mask].mean())
        ax.bar(
            x + zone_idx * width,
            means,
            width,
            label=zone_name,
            color=ZONE_COLORS[zone_idx],
            alpha=0.7,
        )
    ax.set_xticks(x + width)
    ax.set_xticklabels(EPITHELIAL_GENES, rotation=45, ha="right")
    ax.set_ylabel("Mean expression")
    ax.set_title("Epithelial markers by zone")
    ax.legend()

    plt.suptitle(f"{batch}: Inner vs Core vs Outer periphery", fontsize=14)
    plt.tight_layout()
    return fig


def stack_images(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    args = parse_args()
    adata = load_data(args.input)
    temp_dir = Path(tempfile.mkdtemp(prefix="crc_interface_zonation_"))
    panel_paths: list[Path] = []
    for batch in adata.obs["batch"].unique().tolist():
        fig = analyze_interface_zones(adata, batch)
        panel_path = temp_dir / f"{batch}_interface_zones.png"
        fig.savefig(panel_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        panel_paths.append(panel_path)
    stack_images(panel_paths, args.output)


if __name__ == "__main__":
    main()
