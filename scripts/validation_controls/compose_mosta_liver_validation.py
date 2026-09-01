#!/usr/bin/env python
"""Arrange the five MOSTA Liver validation panels into one appendix figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "workflows/mosta_liver_validation/biological_evidence"
TRANSFER_DIR = REPO_ROOT / "workflows/mosta_liver_validation/cross_section_transfer"
OUTPUT_DIR = REPO_ROOT / "workflows/mosta_liver_validation/composite"
PANELS = {
    "a": EVIDENCE_DIR / "figure1_literature_marker_spatial_evidence.png",
    "b": EVIDENCE_DIR / "figure2a_domain_composition.png",
    "c": EVIDENCE_DIR / "figure2b_literature_marker_common_odds_ratio.png",
    "d": TRANSFER_DIR / "figure1_loso_transfer_metrics.png",
    "e": TRANSFER_DIR / "figure2_loso_spatial_transfer.png",
}


def crop_white(image: Image.Image, padding: int = 12) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    bounds = ImageChops.difference(rgb, background).getbbox()
    if bounds is None:
        return rgb
    left, top, right, bottom = bounds
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgb.width, right + padding)
    bottom = min(rgb.height, bottom + padding)
    return rgb.crop((left, top, right, bottom))


def place(axis: plt.Axes, image: Image.Image, label: str) -> None:
    axis.imshow(image)
    axis.set_title(label, loc="left", fontsize=15, fontweight="bold", pad=2)
    axis.axis("off")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = {label: crop_white(Image.open(path)) for label, path in PANELS.items()}

    figure = plt.figure(figsize=(12.0, 10.0), constrained_layout=True)
    outer = figure.add_gridspec(2, 1, height_ratios=(1.22, 1.0))
    top = outer[0, 0].subgridspec(1, 2, width_ratios=(1.45, 1.0))
    top_right = top[0, 1].subgridspec(2, 1, height_ratios=(1, 1))
    bottom = outer[1, 0].subgridspec(1, 2, width_ratios=(1.35, 1.0))
    axes = {
        "a": figure.add_subplot(top[0, 0]),
        "b": figure.add_subplot(top_right[0, 0]),
        "c": figure.add_subplot(top_right[1, 0]),
        "d": figure.add_subplot(bottom[0, 0]),
        "e": figure.add_subplot(bottom[0, 1]),
    }
    for label in ("a", "b", "c", "d", "e"):
        place(axes[label], images[label], label)

    png = OUTPUT_DIR / "mosta_liver_validation_composite.png"
    pdf = OUTPUT_DIR / "mosta_liver_validation_composite.pdf"
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(pdf, dpi=300, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
