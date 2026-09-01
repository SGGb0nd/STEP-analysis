# Analysis index

This page connects each study analysis to the code that runs it and the result
files included in this repository.

## Dataset analyses

| Analysis | Code | Results |
|---|---|---|
| DLPFC spatial domains | `notebooks/main/dlpfc_visium.ipynb` | Notebook outputs |
| MERFISH hypothalamus spatial domains | `notebooks/main/merfish_hypothalamus_analysis.ipynb` | `results/MERFISH/processed.h5ad`; comparison assembly in `notebooks/summary_panels/spatial_domain_comparison_panels.ipynb` |
| STARmap MPFC spatial domains | `notebooks/main/starmap_mpfc.ipynb` | Notebook outputs |
| MOSTA E16.5 Stereo-seq | `scripts/mosta_e16_stereoseq/run_mosta_e16_step.py` | `workflows/mosta_e16_step/` |
| MOSTA external-method comparison | `scripts/external_benchmark/` | `workflows/mosta_external_benchmark/` and `workflows/mosta_metric_summary/` |
| CRC and prostate external-method spatial-domain comparisons | `scripts/external_benchmark/plot_external_method_clusters.py` and `scripts/external_benchmark/external_method_benchmark.py` | Figures and aligned cluster tables generated under `workflows/external_method_cluster_plots/` and `workflows/stacked_external_method_figures/` |
| CRC Visium HD representations and spatial domains | `notebooks/tutorials/crc_visium_hd_representation.ipynb` | `results/visium-hd/crc_8um_5slices_domain.h5ad` |
| CRC tumor periphery analysis | `notebooks/main/crc_visium_hd.ipynb` | Notebook outputs |
| CRC tumor-periphery gene-distance heatmaps | `notebooks/main/crc_gene_distance_heatmaps.ipynb` and `scripts/crc_visium_hd/make_crc_gene_distance_heatmaps.py` | `workflows/crc_gene_distance_heatmaps/` |
| CRC microarchitectures | `scripts/crc_visium_hd/build_microarchitecture.py` and `notebooks/tutorials/crc_visium_hd_microarchitecture.ipynb` | `results/visium-hd/microarc_true50um.pkl`, center assignments, and notebook outputs |
| Prostate Slide-seq V2 representations and spatial domains | `notebooks/tutorials/prostate_slideseq_v2_representation.ipynb` | `results/slide-seq/prostate_slideseq_step.h5ad` |
| Prostate microarchitectures and ligand-receptor programs | `scripts/prostate_slideseq_v2/build_microarchitecture.py` and `notebooks/tutorials/prostate_slideseq_v2_microarchitecture.ipynb` | `results/slide-seq/prostate_microarchitecture_results.pkl`, center assignments, and notebook outputs |
| Liver spatial domains, zonation, and deconvolution | `notebooks/main/liver_visium_BA_normal.ipynb` | Notebook outputs |
| Whole-brain 8.4-million-spot MERFISH | `scripts/merfish_whole_brain/run_merfish.py` | `workflows/merfish_whole_brain_8_4m/` |
| MERFISH and STARmap PLUS cross-technology integration | `scripts/cross_platform_mouse_brain/` | `workflows/cross_platform_mouse_brain/` and `results/cross_platform_mouse_brain/` |

## Validation analyses

| Analysis | Code | Results |
|---|---|---|
| Spatial coherence simulations | `notebooks/validation/spatial_coherence_simulation.ipynb` and `scripts/coherence_simulation/` | `workflows/coherence_simulation/` |
| STEP module ablation | `scripts/module_ablation/` | `workflows/module_ablation/` |
| GraphSAINT sampling | `scripts/graphsaint_sampling/` | `workflows/graphsaint_sampling/` |
| Image-based graph k sensitivity | `scripts/image_based_k_sweep/` | `workflows/image_based_k_sweep/` |
| CRC MA threshold sensitivity | `scripts/crc_visium_hd/ma_threshold_sensitivity.py` | `workflows/crc_ma_threshold_sensitivity/` |
| Prostate MA threshold sensitivity | `scripts/crc_visium_hd/ma_threshold_sensitivity.py` with prostate inputs | `workflows/prostate_ma_threshold_sensitivity/` |
| CRC interface expression | `scripts/crc_visium_hd/make_crc_interface_zonation_stacked.py` | `workflows/crc_interface_zonation/` |
| CRC distance-ranked expression heatmaps | `notebooks/main/crc_gene_distance_heatmaps.ipynb` and `scripts/crc_visium_hd/make_crc_gene_distance_heatmaps.py` | `workflows/crc_gene_distance_heatmaps/` |
| CRC MA histomorphology | `scripts/crc_visium_hd/crc_ma_histomorphology.py` | `workflows/crc_ma_histomorphology/` |
| CRC periphery continuity | `scripts/crc_visium_hd/plot_periphery_continuity.R` | `workflows/crc_periphery_continuity/` |
| Runtime comparison | `scripts/runtime/make_external_runtime_figure.py` | `workflows/external_benchmark_runtime/` |
| STEP complete-neighborhood controls | `scripts/validation_controls/step_fullgraph_prostate_crc.py` | `workflows/step_complete_neighborhood_controls/` |
| MOSTA Cavity method comparison | `scripts/validation_controls/mosta_common_spot_cavity_benchmark.py` | `workflows/mosta_cavity_controls/method_comparison/` |
| MOSTA Cavity local-signal control | `scripts/validation_controls/mosta_cavity_local_signal_control.py` | `workflows/mosta_cavity_controls/local_signal/` |
| MOSTA Liver biological and cross-section validation | `scripts/validation_controls/mosta_liver_domain_validation.py`, `scripts/validation_controls/mosta_liver_evidence_chain.py`, and `scripts/validation_controls/mosta_liver_cross_section_transfer.py` | `workflows/mosta_liver_validation/` |
| MERFISH and DLPFC graph-training controls | `scripts/validation_controls/advanced_graph_training_modes.py` and `scripts/validation_controls/dlpfc_method_graph_modes.py` | `workflows/graph_training_controls/` |

The aggregate cross-platform metric panel is assembled by
`notebooks/summary_panels/spatial_benchmark_summary.ipynb`. The DLPFC,
MERFISH, and STARmap spatial comparison panels are assembled separately by
`notebooks/summary_panels/spatial_domain_comparison_panels.ipynb`.

## User tutorials

| Tutorial | Notebook |
|---|---|
| DLPFC spatial-domain identification | `notebooks/tutorials/DLFPC.ipynb` |
| Human liver spatial domains, zonation, and deconvolution | `notebooks/tutorials/liver_visium_BA_normal.ipynb` |
| MERFISH cell-state and spatial-domain analysis | `notebooks/tutorials/MERFISH.ipynb` |
| Human colorectal-cancer Visium HD analysis | `notebooks/tutorials/human_colorectal_cancer.ipynb` |
| Mouse small-intestine Visium HD analysis | `notebooks/tutorials/mouse_small_intestine.ipynb` |
| Single-cell RNA-seq integration | `notebooks/tutorials/scRNA-seq.ipynb` |
| CRC Visium HD representation | `notebooks/tutorials/crc_visium_hd_representation.ipynb` |
| CRC Visium HD microarchitectures and ligand-receptor programs | `notebooks/tutorials/crc_visium_hd_microarchitecture.ipynb` |
| Prostate Slide-seq V2 representation | `notebooks/tutorials/prostate_slideseq_v2_representation.ipynb` |
| Prostate Slide-seq V2 microarchitectures and ligand-receptor programs | `notebooks/tutorials/prostate_slideseq_v2_microarchitecture.ipynb` |

## MOSTA E16.5

The MOSTA analysis integrates five E16.5 embryo sections containing 504,861
spots and 2,000 genes. STEP uses the raw count layer and all genes in the input
object.

```bash
uv run python scripts/mosta_e16_stereoseq/run_mosta_e16_step.py \
  --input-h5ad data/mosta_e16_test.h5ad \
  --output-dir results/mosta_e16_step
```

The model uses two decoder hidden layers. Each hidden linear transformation is
followed by batch-aware feature scaling before normalization and activation.
The complete model and training settings are listed in
`workflows/mosta_e16_step/settings.json`.

The script writes:

- `mosta_e16_step.h5ad` with STEP representations and spatial domains.
- `step_mosta_metrics.csv` with ARI, NMI, PAS, and CHAOS for each section.
- `step_batch_metrics.json` with dataset-level ASW and iLISI.
- `settings.json` with model and training parameters.

## Input layout

Place analysis inputs under `data/` and generated outputs under `results/`.
Notebook paths follow the same layout. Large H5AD files can be linked into
`data/` from an external storage location.
