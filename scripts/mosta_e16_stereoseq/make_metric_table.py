#!/usr/bin/env python3
"""Aggregate MOSTA metrics and render the comparison table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP_METRICS = (
    REPO_ROOT / "workflows" / "mosta_e16_step" / "step_mosta_metrics.csv"
)
DEFAULT_BATCH_COUNTS = (
    REPO_ROOT / "workflows" / "mosta_e16_step" / "batch_counts.csv"
)
DEFAULT_EXTERNAL_METRICS = (
    REPO_ROOT / "workflows" / "mosta_external_benchmark" / "mosta_benchmark.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "workflows" / "mosta_metric_summary"

_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"
_METRIC_GROUP = "Metric"

METRIC_SPECS = [
    ("annotation_ari", "ARI", True),
    ("annotation_nmi", "NMI", True),
    ("pas", "PAS", True),
    ("chaos", "CHAOS", False),
    ("batch_asw", "ASW", True),
    ("batch_ilisi", "iLISI", True),
]

METHOD_COL_WIDTH = 2.4


def _display_method_name(method: str) -> str:
    return re.sub(
        r"^(BANKSY|HERGAST)\s+Harmony[- ]integrated$",
        r"\1 (Harmony)",
        str(method),
    )


def load_step_weighted_metrics(
    step_metrics_csv: Path,
    batch_counts_csv: Path,
) -> pd.DataFrame:
    """Aggregate section-level STEP metrics using the final section sizes."""
    metrics = pd.read_csv(step_metrics_csv).copy()
    counts = pd.read_csv(batch_counts_csv).copy()
    metrics["batch"] = metrics["batch"].astype(str)
    counts["batch"] = counts["batch"].astype(str)

    weighted = metrics.merge(counts, on="batch", how="left", validate="one_to_one")
    if weighted["n_cells"].isna().any():
        missing = weighted.loc[weighted["n_cells"].isna(), "batch"].tolist()
        raise ValueError(f"Missing STEP batch counts for: {missing}")

    ilisi_col = "iLISI" if "iLISI" in weighted.columns else "iLICI"
    weights = weighted["n_cells"].to_numpy(dtype=float)
    row = {
        "method": "STEP",
        "annotation_ari": float(np.average(weighted["ARI"], weights=weights)),
        "annotation_nmi": float(np.average(weighted["NMI"], weights=weights)),
        "pas": float(np.average(weighted["PAS"], weights=weights)),
        "chaos": float(np.average(weighted["CHAOS"], weights=weights)),
        "batch_asw": float(np.average(weighted["ASW"], weights=weights)),
        "batch_ilisi": float(np.average(weighted[ilisi_col], weights=weights)),
        "n_eval_cells": int(weights.sum()),
    }
    return pd.DataFrame([row])


def _normalized_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for key, _, higher_better in METRIC_SPECS:
        series = frame[key].astype(float)
        min_v = float(series.min())
        max_v = float(series.max())
        if np.isclose(min_v, max_v):
            out[key] = 1.0
        elif higher_better:
            out[key] = (series - min_v) / (max_v - min_v)
        else:
            out[key] = (max_v - series) / (max_v - min_v)
    return out


def _build_plot_results_frame(metrics_df: pd.DataFrame) -> pd.DataFrame:
    values = metrics_df.copy().set_index("method")
    display_names = {key: label for key, label, _ in METRIC_SPECS}
    plot_df = values[[key for key, _, _ in METRIC_SPECS]].rename(
        columns=display_names
    )
    plot_df[_AGGREGATE_SCORE] = _normalized_scores(values).mean(axis=1)

    metric_types = {label: _METRIC_GROUP for _, label, _ in METRIC_SPECS}
    metric_types[_AGGREGATE_SCORE] = _AGGREGATE_SCORE
    plot_df.loc[_METRIC_TYPE] = pd.Series(metric_types)
    return plot_df


def render_results_table(metrics_df: pd.DataFrame, output_prefix: Path) -> None:
    """Render the MOSTA metric table."""
    frame = _build_plot_results_frame(metrics_df)
    num_methods = len(frame.index) - 1
    plot_df = frame.drop(_METRIC_TYPE, axis=0)
    plot_df = plot_df.sort_values(by=_AGGREGATE_SCORE, ascending=False).astype(
        np.float64
    )
    plot_df["Method"] = plot_df.index

    score_cols = frame.columns[frame.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
    metric_cols = frame.columns[frame.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
    cmap_fn = lambda values: normed_cmap(  # noqa: E731
        values,
        cmap=mpl.cm.PRGn,
        num_stds=2.5,
    )

    column_definitions = [
        ColumnDefinition(
            "Method",
            width=METHOD_COL_WIDTH,
            textprops={"ha": "left", "weight": "bold"},
        )
    ]
    for col in metric_cols:
        cmap_values = -plot_df[col] if col == "CHAOS" else plot_df[col]
        column_definitions.append(
            ColumnDefinition(
                col,
                title=col.replace(" ", "\n", 1),
                width=1,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.25},
                },
                cmap=cmap_fn(cmap_values),
                group=frame.loc[_METRIC_TYPE, col],
                formatter="{:.2f}",
            )
        )
    for index, col in enumerate(score_cols):
        column_definitions.append(
            ColumnDefinition(
                col,
                width=1,
                title=col.replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": mpl.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=frame.loc[_METRIC_TYPE, col],
                border="left" if index == 0 else None,
            )
        )

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(
            figsize=(len(frame.columns) * 1.25 + 1.0, 3 + 0.3 * num_methods)
        )
        Table(
            plot_df,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
        ax.set_title("MOSTA Metrics Table")
        for suffix in (".svg", ".png", ".pdf"):
            fig.savefig(
                output_prefix.with_suffix(suffix),
                facecolor=ax.get_facecolor(),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate MOSTA metrics and render the comparison table."
    )
    parser.add_argument(
        "--step-metrics-csv",
        type=Path,
        default=DEFAULT_STEP_METRICS,
    )
    parser.add_argument(
        "--batch-counts-csv",
        type=Path,
        default=DEFAULT_BATCH_COUNTS,
    )
    parser.add_argument(
        "--external-metrics-csv",
        type=Path,
        default=DEFAULT_EXTERNAL_METRICS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_df = load_step_weighted_metrics(
        args.step_metrics_csv,
        args.batch_counts_csv,
    )
    external_df = pd.read_csv(args.external_metrics_csv).copy()
    combined = pd.concat([step_df, external_df], ignore_index=True)
    combined["method"] = combined["method"].map(_display_method_name)

    combined_path = args.output_dir / "mosta_metrics_combined.csv"
    combined.to_csv(combined_path, index=False)
    render_results_table(
        combined,
        args.output_dir / "mosta_metrics_results_table",
    )

    summary = {
        "inputs": {
            "step_metrics": "../mosta_e16_step/step_mosta_metrics.csv",
            "batch_counts": "../mosta_e16_step/batch_counts.csv",
            "external_metrics": "../mosta_external_benchmark/mosta_benchmark.csv",
        },
        "combined_metrics": combined_path.name,
        "step_aggregation": "cell-count-weighted mean across the five embryo sections",
        "batch_metric_scope": "one ASW and one iLISI value for the complete five-section dataset",
        "methods": combined.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
