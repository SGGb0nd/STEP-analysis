# Notebooks

The notebooks are grouped by role. Files under `main/` contain the primary
dataset-specific analyses. The `validation/`, `support/`, and
`summary_panels/` directories contain additional analyses and outputs reported
in the manuscript. Files under `tutorials/` explain the STEP workflow with
user-facing narrative and runnable examples.

## `main/`

- `dlpfc_visium.ipynb`
- `merfish_hypothalamus_analysis.ipynb`
- `starmap_mpfc.ipynb`
- `crc_visium_hd.ipynb`
- `crc_gene_distance_heatmaps.ipynb`
- `prostate_slideseq_v2.ipynb`
- `liver_visium_BA_normal.ipynb`

## `tutorials/`

- `DLFPC.ipynb`
- `liver_visium_BA_normal.ipynb`
- `MERFISH.ipynb`
- `crc_visium_hd_representation.ipynb`
- `crc_visium_hd_microarchitecture.ipynb`
- `human_colorectal_cancer.ipynb`
- `mouse_small_intestine.ipynb`
- `prostate_slideseq_v2_representation.ipynb`
- `prostate_slideseq_v2_microarchitecture.ipynb`
- `scRNA-seq.ipynb`

## `validation/`

- `spatial_coherence_simulation.ipynb`

## `support/` and `summary_panels/`

- `support/scrna_integration.ipynb`
- `summary_panels/liver_zonation_celltype_heatmap.ipynb`
- `summary_panels/spatial_benchmark_summary.ipynb`
- `summary_panels/spatial_domain_comparison_panels.ipynb`

The `summary_panels/` notebooks assemble selected cross-dataset summaries from
analysis outputs. Dataset-specific notebooks also produce figures directly
during their analysis workflows.

## Conventions

Saved outputs document the main analysis steps and results. Place input data
under `../data/` and write generated files under `../results/`. See
`../docs/analysis_index.md` for the analysis-to-file map.
