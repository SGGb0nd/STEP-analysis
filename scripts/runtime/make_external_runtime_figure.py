"""Render the large-dataset runtime comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "workflows" / "external_benchmark_runtime"


def hhmmss_to_minutes(text: str) -> float:
    h, m, s = [int(x) for x in text.split(":")]
    return h * 60 + m + s / 60.0


def build_model_stage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"method": "STEP", "dataset": "Visium HD", "minutes": 392 / 60.0, "note": "edge-clip grid training"},
            {"method": "STEP", "dataset": "MOSTA", "minutes": 255 / 60.0, "note": "edge-clip grid training"},
            {"method": "STEP", "dataset": "slide-seq", "minutes": 142.7917747820029 / 60.0, "note": "node-sampled graph training"},
            {"method": "HERGAST", "dataset": "Visium HD", "minutes": 67.1, "note": "all 5 samples"},
            {"method": "HERGAST", "dataset": "MOSTA", "minutes": 30.5, "note": "all 5 samples"},
            {"method": "HERGAST", "dataset": "slide-seq", "minutes": 1.9, "note": "all 4 samples; reduced params"},
            {"method": "NicheCompass", "dataset": "Visium HD", "minutes": 382.0, "note": "400 epochs; merged 5-sample training"},
            {"method": "NicheCompass", "dataset": "MOSTA", "minutes": 24.7, "note": "400 epochs; training only"},
            {"method": "NicheCompass", "dataset": "slide-seq", "minutes": 50.0, "note": "400 epochs; training only"},
            {"method": "BANKSY", "dataset": "Visium HD", "minutes": sum(map(hhmmss_to_minutes, ["00:47:25", "00:43:44", "00:40:10", "00:45:22", "00:27:37"])), "note": "all 5 full-data samples"},
            {"method": "BANKSY", "dataset": "MOSTA", "minutes": sum(map(hhmmss_to_minutes, ["00:03:06", "00:04:26", "00:06:10", "00:04:06", "00:03:07"])), "note": "all 5 embedding runs"},
            {"method": "BANKSY", "dataset": "slide-seq", "minutes": sum(map(hhmmss_to_minutes, ["00:01:10", "00:01:19", "00:01:36", "00:01:20"])), "note": "all 4 samples; full spot counts"},
        ]
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    model_stage = build_model_stage()

    datasets = ["Visium HD", "MOSTA", "slide-seq"]
    methods = ["STEP", "HERGAST", "NicheCompass", "BANKSY"]
    colors = {"STEP": "#9467bd", "HERGAST": "#1f77b4", "NicheCompass": "#d62728", "BANKSY": "#2ca02c"}

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.8))
    x = np.arange(len(datasets))
    width = 0.18
    for idx, method in enumerate(methods):
        vals = []
        for dataset in datasets:
            row = model_stage[(model_stage["method"] == method) & (model_stage["dataset"] == dataset)]
            vals.append(float(row["minutes"].iloc[0]) if not row.empty else np.nan)
        xoff = x + (idx - (len(methods) - 1) / 2) * width
        ax.bar(xoff, vals, width=width, color=colors[method], label=method)
        for xi, val in zip(xoff, vals):
            if not np.isnan(val):
                ax.text(xi, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Minutes")
    ax.set_title("Runtime")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR / "external_benchmark_runtime.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTDIR / "external_benchmark_runtime.pdf", dpi=320, bbox_inches="tight")


if __name__ == "__main__":
    main()
