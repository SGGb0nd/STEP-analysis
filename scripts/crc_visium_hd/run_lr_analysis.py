"""Run the canonical CRC MicroArchitecture ligand-receptor analysis."""

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from liana.method import cellphonedb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import liana as li
from step.utils.misc import read_visium_hd


CRC_CELL_TYPES = [
    "Interface Fibroblasts",
    "Immune cells",
    "Endothelial",
    "CAF",
    "Tumor 1",
    "Tumor 2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("results/visium-hd/crc5_meta_filtered.csv"),
    )
    parser.add_argument(
        "--microarc-pickle",
        type=Path,
        default=Path("results/visium-hd/microarc_true50um.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/visium-hd/lr-analyses-true50um"),
    )
    parser.add_argument(
        "--crc-data-root",
        type=Path,
        default=Path("data/visium-hd-crc"),
    )
    parser.add_argument(
        "--crc-p2-root",
        type=Path,
        default=Path("data/visium-hd/human-coloretal-cancer"),
    )
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    return parser.parse_args()


def read_section(name: str, path: Path):
    adata = read_visium_hd(path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    spatial_key = next(iter(adata.uns["spatial"]))
    adata.uns["spatial"][name] = adata.uns["spatial"].pop(spatial_key)
    return adata


def load_expression(args: argparse.Namespace, metadata: pd.DataFrame):
    roots = {
        "cancer_p1": args.crc_data_root
        / "cancer_p1/binned_outputs/square_008um",
        "cancer_p2": args.crc_p2_root / "square_008um",
        "cancer_p5": args.crc_data_root
        / "cancer_p5/binned_outputs/square_008um",
        "normal_p3": args.crc_data_root
        / "normal_p3/binned_outputs/square_008um",
        "normal_p5": args.crc_data_root
        / "normal_p5/binned_outputs/square_008um",
    }
    missing = [str(path) for path in roots.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CRC Visium HD inputs: {missing}")

    sections = {name: read_section(name, path) for name, path in roots.items()}
    adata = ad.concat(sections, label="batch", uns_merge="unique")
    adata.obs.index = adata.obs_names + "-" + adata.obs["batch"].astype(str)
    missing_spots = metadata.index.difference(adata.obs_names)
    if len(missing_spots):
        raise ValueError(f"Expression data are missing {len(missing_spots)} metadata spots")
    adata = adata[metadata.index].copy()
    adata.obs["cell_type"] = metadata["cell_type"].astype(str)
    sc.pp.normalize_total(adata, target_sum=1e5)
    sc.pp.log1p(adata)
    return adata


def add_bh_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    p_values = pd.to_numeric(
        result["cellphone_pvals"], errors="coerce"
    ).to_numpy(float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p_values))
    if len(valid):
        order = valid[np.argsort(p_values[valid])]
        ranked = p_values[order]
        ranked_adjusted = ranked * len(order) / np.arange(1, len(order) + 1)
        ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
        adjusted[order] = np.clip(ranked_adjusted, 0.0, 1.0)
    result["bh_fdr"] = adjusted
    return result


def collect_ma_expression(adata, microarc, ma_index: int):
    graph_indices = np.flatnonzero(np.asarray(microarc.clusters) == ma_index)
    spot_names = pd.Index(
        np.concatenate(
            [microarc.neighbor_indices[index] for index in graph_indices]
        )
    ).unique()
    missing = spot_names.difference(adata.obs_names)
    if len(missing):
        raise ValueError(f"MA {ma_index + 1} has {len(missing)} unmatched spots")
    return adata[spot_names].copy()


def run_cellphonedb(adata, microarc, output_dir: Path) -> dict[str, object]:
    results = {}
    for ma_index in range(microarc.n_clusters):
        ma_adata = collect_ma_expression(adata, microarc, ma_index)
        if ma_adata.n_obs < 3:
            raise ValueError(
                f"MicroArchitecture {ma_index + 1} has insufficient spots"
            )
        cellphonedb(
            ma_adata,
            groupby="cell_type",
            resource_name="consensus",
            expr_prop=0.1,
            key_added="cpdb_res",
            use_raw=False,
            verbose=True,
        )
        result = add_bh_fdr(ma_adata.uns["cpdb_res"])
        result.to_csv(
            output_dir
            / f"niche_pattern_{ma_index}_cellphonedb_bh_fdr.csv",
            index=False,
        )
        ma_adata.uns["cpdb_res"] = result
        results[f"MA_{ma_index + 1}"] = ma_adata
    return results


def write_dotplots_and_tileplots(
    ma_results: dict[str, object], output_dir: Path
) -> None:
    for ma_index, ma_adata in enumerate(ma_results.values()):
        result = ma_adata.uns["cpdb_res"]
        sources = [name for name in CRC_CELL_TYPES if name in set(result["source"])]
        targets = [name for name in CRC_CELL_TYPES if name in set(result["target"])]
        dotplot = li.pl.dotplot(
            liana_res=result,
            colour="lr_means",
            size="bh_fdr",
            inverse_size=True,
            source_labels=sources,
            target_labels=targets,
            filter_fun=lambda frame: frame["bh_fdr"] <= 0.05,
            figure_size=(18, 24),
        )
        dotplot.save(
            output_dir / f"cpdb_microarc_pattern_{ma_index}_dotplot.pdf",
            verbose=False,
        )
        tileplot = li.pl.tileplot(
            liana_res=result,
            fill="means",
            label="props",
            label_fun=lambda value: "*" if value >= 0.5 else np.nan,
            top_n=20,
            orderby="bh_fdr",
            orderby_ascending=True,
            source_labels=sources,
            target_labels=targets,
            source_title="Ligand",
            target_title="Receptor",
            figure_size=(8, 7),
        )
        tileplot.save(
            output_dir / f"cpdb_microarc_pattern_{ma_index}_tileplot.pdf",
            verbose=False,
        )


def prepare_network(result: pd.DataFrame, threshold: float):
    retained = result[
        result["bh_fdr"].lt(threshold)
        & result["source"].isin(CRC_CELL_TYPES)
        & result["target"].isin(CRC_CELL_TYPES)
    ].copy()
    edges = (
        retained.groupby(["source", "target"])
        .agg(pair_count=("bh_fdr", "size"), mean_bh_fdr=("bh_fdr", "mean"))
        .reset_index()
    )
    node_counts = pd.concat([retained["source"], retained["target"]]).value_counts()
    return retained, edges, node_counts


def write_network_stack(
    ma_results: dict[str, object],
    microarc,
    output_dir: Path,
    threshold: float,
) -> dict[str, dict[str, int | float | None]]:
    network_data = {
        name: prepare_network(ma_adata.uns["cpdb_res"], threshold)
        for name, ma_adata in ma_results.items()
    }
    max_node_count = max(
        (counts.max() for _, _, counts in network_data.values() if len(counts)),
        default=1,
    )
    max_edge_count = max(
        (edges["pair_count"].max() for _, edges, _ in network_data.values() if len(edges)),
        default=1,
    )
    score_arrays = [
        -np.log10(edges["mean_bh_fdr"].to_numpy() + 1e-300)
        for _, edges, _ in network_data.values()
        if len(edges)
    ]
    scores = np.concatenate(score_arrays) if score_arrays else np.array([])
    score_min = float(scores.min()) if len(scores) else 0.0
    score_max = float(scores.max()) if len(scores) else 1.0
    if score_max <= score_min:
        score_max = score_min + 1.0
    score_norm = Normalize(vmin=score_min, vmax=score_max)

    all_cell_types = sorted(microarc.global_positions)
    colors = dict(
        zip(all_cell_types, plt.cm.tab20(np.linspace(0, 1, len(all_cell_types))))
    )
    fig, axes = plt.subplots(1, microarc.n_clusters, figsize=(18, 6))
    axes = np.atleast_1d(axes)
    summary = {}
    for ma_index, (name, (retained, edges, node_counts)) in enumerate(
        network_data.items()
    ):
        ax = axes[ma_index]
        graph = nx.DiGraph()
        graph.add_nodes_from(CRC_CELL_TYPES)
        for row in edges.itertuples(index=False):
            graph.add_edge(
                row.source,
                row.target,
                pair_count=row.pair_count,
                mean_bh_fdr=row.mean_bh_fdr,
            )
        positions = {
            node: microarc.global_positions[node] for node in graph.nodes
        }
        sizes = [
            np.interp(node_counts.get(node, 0), [0, max_node_count], [100, 1000])
            for node in graph.nodes
        ]
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_size=sizes,
            node_color=[colors[node] for node in graph.nodes],
            alpha=0.7,
        )
        for source, target, attributes in graph.edges(data=True):
            width = np.interp(
                attributes["pair_count"], [0, max_edge_count], [1, 10]
            )
            score = -np.log10(attributes["mean_bh_fdr"] + 1e-300)
            arrow = patches.FancyArrowPatch(
                positions[source],
                positions[target],
                connectionstyle=patches.ConnectionStyle.Arc3(rad=0.4),
                arrowstyle="-|>",
                linewidth=width,
                alpha=0.65,
                color=plt.cm.Spectral_r(score_norm(score)),
                mutation_scale=20,
                shrinkA=10,
                shrinkB=10,
                zorder=1,
            )
            ax.add_patch(arrow)
        label_positions = {
            node: (position[0], position[1] + 0.05)
            for node, position in positions.items()
        }
        nx.draw_networkx_labels(
            graph,
            label_positions,
            ax=ax,
            font_size=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )
        ax.set_title(f"MicroArchitecture {ma_index + 1}", fontsize=14, pad=18)
        ax.margins(x=0.2, y=0.2)
        ax.axis("off")
        summary[name] = {
            "n_spots": int(ma_results[name].n_obs),
            "n_bh_fdr_pairs": int(len(retained)),
            "n_directed_cell_type_edges": int(len(edges)),
            "minimum_bh_fdr": (
                float(ma_results[name].uns["cpdb_res"]["bh_fdr"].min())
                if len(ma_results[name].uns["cpdb_res"])
                else None
            ),
        }

    colorbar = fig.colorbar(
        ScalarMappable(norm=score_norm, cmap="Spectral_r"),
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.12,
        shrink=0.45,
    )
    colorbar.set_label("-log10(BH-FDR)")
    fig.savefig(
        output_dir / "microarc_mole_patterns_bh_fdr_stacked.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "microarc_mole_patterns_bh_fdr_stacked.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    if not 0 < args.fdr_threshold < 1:
        raise ValueError("--fdr-threshold must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv, index_col=0, low_memory=False)
    with args.microarc_pickle.open("rb") as handle:
        microarc = pickle.load(handle)
    adata = load_expression(args, metadata)
    ma_results = run_cellphonedb(adata, microarc, args.output_dir)
    write_dotplots_and_tileplots(ma_results, args.output_dir)
    summary = write_network_stack(
        ma_results,
        microarc,
        args.output_dir,
        args.fdr_threshold,
    )
    payload = {
        "metadata_csv": str(args.metadata_csv),
        "microarc_pickle": str(args.microarc_pickle),
        "normalization": "library-size 1e5 followed by log1p",
        "cellphonedb_resource": "consensus",
        "cellphonedb_expression_proportion": 0.1,
        "bh_fdr_scope": "complete CellPhoneDB table within each MicroArchitecture",
        "bh_fdr_threshold": args.fdr_threshold,
        "microarchitectures": summary,
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
