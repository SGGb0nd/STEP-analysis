#!/usr/bin/env python3
"""Render the matched STEP module-ablation metric panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TAG_ORDER = ["full_bbm_bem_spm", "ablate_spm", "ablate_bem", "ablate_bbm_signal"]
TAG_LABELS = {
    "full_bbm_bem_spm": "STEP",
    "ablate_spm": "No SpM",
    "ablate_bem": "No BEM",
    "ablate_bbm_signal": "No BBM",
}
METRICS = [
    ("celltype_ari_mean", "CT ARI", True),
    ("celltype_nmi_mean", "CT NMI", True),
    ("domain_ari_mean", "Dom ARI", True),
    ("domain_nmi_mean", "Dom NMI", True),
    ("spatial_pas_mean", "PAS", True),
    ("spatial_chaos_mean", "CHAOS", False),
    ("batch_asw_mean", "ASW z", True),
    ("batch_ilisi_mean", "iLISI z", True),
    ("batch_asw_spatial_mean", "ASW z~", True),
    ("batch_ilisi_spatial_mean", "iLISI z~", True),
]


def load_panel(summary_path: Path, regime_label: str) -> pd.DataFrame:
    payload = json.loads(summary_path.read_text())
    rows = []
    for tag in TAG_ORDER:
        rec = payload["aggregate_by_tag"][tag]
        row = {"tag": tag, "label": TAG_LABELS[tag], "regime": regime_label}
        for key, _, _ in METRICS:
            row[key] = float(rec[key])
        rows.append(row)
    return pd.DataFrame(rows)


def score_for_col(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    arr = values.astype(float)
    if not higher_is_better:
        arr = -arr
    lo = float(arr.min())
    hi = float(arr.max())
    if np.isclose(lo, hi):
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def draw_heatmap(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    raw = np.column_stack([df[key].to_numpy(dtype=float) for key, _, _ in METRICS])
    scaled = np.column_stack([score_for_col(df[key].to_numpy(dtype=float), higher) for key, _, higher in METRICS])
    ax.imshow(scaled, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([label for _, label, _ in METRICS], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["label"])
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(1.5, color="#444444", lw=1.5)
    ax.axvline(5.5, color="#444444", lw=1.5)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the module-ablation metric panel from paired simulation summaries.")
    parser.add_argument("--no-batch-summary", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("workflows/module_ablation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df0 = load_panel(args.no_batch_summary, "No batch effect")
    df1 = load_panel(args.batch_summary, "Batch effect")
    pd.concat([df0, df1], ignore_index=True).to_csv(args.out_dir / "ablation_metric_panel.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.3), constrained_layout=True)
    draw_heatmap(axes[0], df0, "No batch effect")
    draw_heatmap(axes[1], df1, "Batch effect")
    fig.savefig(args.out_dir / "ablation_metric_panel.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out_dir / "ablation_metric_panel.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
