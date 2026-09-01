"""Render the composition-driven spatial coherence validation panel from a saved AnnData object."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def _corr_title(y, x, label: str) -> str:
    r, p = pearsonr(y, x)
    rho, _ = spearmanr(y, x)
    return f"{label}\nPearson r={r:.3f}, p={p:.1e}; Spearman rho={rho:.3f}"


def visualize_from_h5ad(
    h5ad_path: str | Path,
    size: int = 80,
    save_path: str | Path | None = None,
):
    adata = sc.read_h5ad(h5ad_path)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    for ax, color, title in [
        (axes[0, 0], "cell_type", "GT: Cell Types"),
        (axes[0, 1], "spatial_domain", "GT: Domains"),
        (axes[0, 2], "step_ct", "STEP: Cell Types"),
        (axes[0, 3], "domain", "STEP: Domains"),
        (axes[1, 0], "oracle_ct_heterogeneity", "Oracle: CT Heterogeneity"),
        (axes[1, 1], "oracle_ct_match", "Oracle: CT Match"),
        (axes[1, 2], "step_coherence", "STEP: Coherence"),
    ]:
        kwargs = {}
        if color == "oracle_ct_heterogeneity":
            kwargs["cmap"] = "Reds"
        elif color == "oracle_ct_match":
            kwargs["cmap"] = "Blues"
        elif color == "step_coherence":
            kwargs["cmap"] = "RdYlBu"
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=color,
            ax=ax,
            show=False,
            title=title,
            size=size,
            **kwargs,
        )

    ax = axes[1, 3]
    sns.regplot(
        data=adata.obs,
        x="oracle_ct_heterogeneity",
        y="step_coherence",
        scatter_kws={"alpha": 0.3, "s": size},
        line_kws={"color": "#c62828"},
        ax=ax,
    )
    ax.set_title(
        _corr_title(
            adata.obs["step_coherence"].to_numpy(),
            adata.obs["oracle_ct_heterogeneity"].to_numpy(),
            "Coherence vs CT Heterogeneity",
        )
    )
    ax.set_xlabel("CT Heterogeneity")
    ax.set_ylabel("Spatial Coherence")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, adata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot STEP coherence validation from a saved simulation h5ad.")
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("workflows/coherence_simulation/coherence_validation.png"))
    parser.add_argument("--point-size", type=int, default=80)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"figure.dpi": 300}):
        fig, _ = visualize_from_h5ad(h5ad_path=args.input_h5ad, size=args.point_size, save_path=args.output)
        plt.close(fig)


if __name__ == "__main__":
    main()
