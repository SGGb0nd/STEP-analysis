# Zenodo dataset-to-result map

This inventory links each packaged input or external data reference to its analysis-ready intermediate files and workflow outputs. Sizes are the actual staged sizes used for the Zenodo archives; `link only` data are not included in archive totals.

| Dataset or analysis family | Raw/input | Intermediate results | Workflow outputs | Total staged | Link-only inputs |
|---|---:|---:|---:|---:|---:|
| CRC Visium HD | 0.0 B | 14.6 GiB | 19.2 MiB | 14.6 GiB | 1 |
| Prostate Slide-seq V2 | 376.7 MiB | 1.2 GiB | 4.2 MiB | 1.6 GiB | 0 |
| DLPFC Visium | 1.2 GiB | 1.8 GiB | 0.0 B | 3.1 GiB | 0 |
| MERFISH hypothalamus | 20.7 MiB | 62.0 MiB | 0.0 B | 82.7 MiB | 0 |
| STARmap MPFC | 2.4 MiB | 9.1 MiB | 0.0 B | 11.5 MiB | 0 |
| MOSTA E16.5 | 7.3 GiB | 0.0 B | 10.9 MiB | 7.3 GiB | 0 |
| Deconvolution simulation | 4.5 GiB | 0.0 B | 0.0 B | 4.5 GiB | 0 |
| Human liver | 320.1 MiB | 2.7 GiB | 0.0 B | 3.0 GiB | 0 |
| Whole-brain MERFISH | 0.0 B | 399.4 MiB | 23.3 MiB | 422.7 MiB | 1 |
| Cross-platform mouse brain | 0.0 B | 1.8 GiB | 23.6 MiB | 1.8 GiB | 1 |
| Spatial coherence simulation | 0.0 B | 23.5 MiB | 8.5 MiB | 32.0 MiB | 0 |
| Multi-dataset runtime benchmark | 0.0 B | 0.0 B | 127.9 KiB | 127.9 KiB | 0 |
| Synthetic graph-sampling control | 0.0 B | 0.0 B | 546.8 KiB | 546.8 KiB | 0 |
| Image-based graph sensitivity | 0.0 B | 0.0 B | 3.1 MiB | 3.1 MiB | 0 |
| Synthetic module ablation | 0.0 B | 0.0 B | 24.6 MiB | 24.6 MiB | 0 |
| CRC, prostate, and MOSTA external benchmarks | 0.0 B | 0.0 B | 103.5 MiB | 103.5 MiB | 0 |
| CRC and prostate graph-training controls | 0.0 B | 0.0 B | 38.4 MiB | 38.4 MiB | 0 |
| MOSTA E16.5 Liver validation | 0.0 B | 0.0 B | 76.0 MiB | 76.0 MiB | 0 |
| MERFISH and DLPFC graph-training controls | 0.0 B | 0.0 B | 90.6 MiB | 90.6 MiB | 0 |

The raw archive contains 13.7 GiB of staged inputs, the intermediate archive 22.6 GiB of analysis-ready results, and the output archive 426.6 MiB of workflow outputs. The compressed archive files are 4.1 GiB, 6.9 GiB, and 408.1 MiB, respectively.

## CRC Visium HD

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | CRC Visium HD | link only | external | `raw/README_crc_visium_hd.md` | 10x Genomics dataset portal; accession link in docs/data_accessions.md |
| intermediate | CRC histopathology | included | 18.3 MiB | `intermediate/crc/8um_squares_annotation.csv` | Pathology annotations consumed by scripts/crc_visium_hd/crc_ma_histomorphology.py |
| intermediate | CRC Visium HD | included | 8.8 GiB | `intermediate/crc/crc_8um_5slices_domain.h5ad` | notebooks/tutorials/crc_visium_hd_representation.ipynb |
| intermediate | CRC Visium HD | included | 141.3 MiB | `intermediate/crc/crc5_meta.csv` | Manuscript marker-cluster metadata |
| intermediate | CRC Visium HD | included | 236.3 MiB | `intermediate/crc/crc5_meta_filtered.csv` | Canonical-marker cell-type annotations consumed by the MA workflow |
| intermediate | CRC Visium HD | included | 4.2 GiB | `intermediate/crc/cancer_tumor_regions.h5ad` | notebooks/main/crc_visium_hd.ipynb |
| intermediate | CRC Visium HD | included | 65.5 KiB | `intermediate/crc/periphery_distances.csv` | notebooks/main/crc_visium_hd.ipynb |
| intermediate | CRC Visium HD | included | 838.3 MiB | `intermediate/crc/microarc_true50um.pkl` | notebooks/tutorials/crc_visium_hd_microarchitecture.ipynb |
| intermediate | CRC Visium HD | included | 247.9 MiB | `intermediate/crc/metadata_with_patterns_true50um.csv` | scripts/crc_visium_hd/build_microarchitecture.py |
| intermediate | CRC Visium HD | included | 629.0 B | `intermediate/crc/microarc_true50um_settings.json` | scripts/crc_visium_hd/build_microarchitecture.py |
| intermediate | CRC ligand-receptor source tables | included | 1.2 MiB | `intermediate/crc/lr_source_tables/` | Original CellPhoneDB result tables; BH-FDR postprocessing is implemented in scripts/crc_visium_hd/run_lr_analysis.py |
| intermediate | CRC Visium HD | included | 156.7 MiB | `intermediate/crc/crc5_meta_with_deconv.csv` | Consumed by scripts/crc_visium_hd/crc_ma_histomorphology.py |
| output | CRC gene-distance heatmaps | included | 9.5 MiB | `outputs/workflows/crc_gene_distance_heatmaps/` | scripts/crc_visium_hd/make_crc_gene_distance_heatmaps.py |
| output | CRC interface zonation | included | 2.0 MiB | `outputs/workflows/crc_interface_zonation/` | scripts/crc_visium_hd/make_crc_interface_zonation_stacked.py |
| output | CRC MA histomorphology | included | 174.1 KiB | `outputs/workflows/crc_ma_histomorphology/` | scripts/crc_visium_hd/crc_ma_histomorphology.py |
| output | CRC MA threshold sensitivity | included | 7.2 MiB | `outputs/workflows/crc_ma_threshold_sensitivity/` | scripts/crc_visium_hd/ma_threshold_sensitivity.py |
| output | CRC periphery continuity | included | 354.7 KiB | `outputs/workflows/crc_periphery_continuity/` | scripts/crc_visium_hd/plot_periphery_continuity.R |

## Prostate Slide-seq V2

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | Prostate Slide-seq V2 | included | 376.7 MiB | `raw/prostate_slideseq_v2/` | GEO GSE181294 |
| intermediate | Prostate Slide-seq V2 | included | 1.1 GiB | `intermediate/prostate/prostate_slideseq_step.h5ad` | notebooks/tutorials/prostate_slideseq_v2_representation.ipynb |
| intermediate | Prostate Slide-seq V2 | included | 113.7 MiB | `intermediate/prostate/prostate_microarchitecture_results.pkl` | scripts/prostate_slideseq_v2/build_microarchitecture.py |
| intermediate | Prostate Slide-seq V2 | included | 9.5 MiB | `intermediate/prostate/prostate_microarchitecture_metadata.csv` | scripts/prostate_slideseq_v2/build_microarchitecture.py |
| intermediate | Prostate Slide-seq V2 | included | 2.4 MiB | `intermediate/prostate/lr_tables/` | notebooks/tutorials/prostate_slideseq_v2_microarchitecture.ipynb |
| output | Prostate MA threshold sensitivity | included | 4.2 MiB | `outputs/workflows/prostate_ma_threshold_sensitivity/` | scripts/crc_visium_hd/ma_threshold_sensitivity.py |

## DLPFC Visium

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | DLPFC Visium | included | 1.2 GiB | `raw/dlpfc_visium/` | spatialLIBD HumanPilot |
| intermediate | DLPFC Visium | included | 1.8 GiB | `intermediate/dlpfc/` | notebooks/main/dlpfc_visium.ipynb |

## MERFISH hypothalamus

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | MERFISH hypothalamus | included | 20.7 MiB | `raw/merfish_hypothalamus/` | Dryad doi:10.5061/dryad.8t8s248 |
| intermediate | MERFISH hypothalamus | included | 62.0 MiB | `intermediate/merfish_hypothalamus/processed.h5ad` | notebooks/main/merfish_hypothalamus_analysis.ipynb |

## STARmap MPFC

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | STARmap MPFC | included | 2.4 MiB | `raw/starmap_mpfc/merged3.h5ad` | Analysis-ready four-section object assembled from STARmap resources |
| intermediate | STARmap MPFC | included | 9.1 MiB | `intermediate/starmap_mpfc/processed.h5ad` | notebooks/main/starmap_mpfc.ipynb |

## MOSTA E16.5

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | MOSTA E16.5 analysis-ready input and STEP result | included | 7.3 GiB | `raw/mosta_e16/mosta_e16_test.h5ad` | Canonical object containing count layers, STEP representations, spatial domains, and manual annotations |
| output | MOSTA E16.5 STEP | included | 1.8 KiB | `outputs/workflows/mosta_e16_step/` | scripts/mosta_e16_stereoseq/run_mosta_e16_step.py |
| output | MOSTA external benchmark | included | 1.5 KiB | `outputs/workflows/mosta_external_benchmark/` | scripts/external_benchmark/external_method_benchmark.py |
| output | MOSTA metric summary | included | 286.2 KiB | `outputs/workflows/mosta_metric_summary/` | scripts/mosta_e16_stereoseq/make_metric_table.py |
| output | MOSTA Cavity controls | included | 10.6 MiB | `outputs/workflows/mosta_cavity_controls/` | scripts/validation_controls/mosta_common_spot_cavity_benchmark.py; scripts/validation_controls/mosta_cavity_local_signal_control.py |

## Deconvolution simulation

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | Deconvolution simulation spatial input | included | 808.4 MiB | `raw/deconvolution_simulation/simulated_spatial_ground_truth.h5ad` | Synthetic spatial ground truth used by the deconvolution simulation notebook |
| raw | Deconvolution simulation single-cell reference | included | 3.7 GiB | `raw/deconvolution_simulation/single_cell_reference.h5ad` | Single-cell reference used by the deconvolution simulation notebook |

## Human liver

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | Liver Visium and snRNA-seq | included | 320.1 MiB | `raw/liver/` | Transcriptomics technical optimization repository and Figshare doi:10.6084/m9.figshare.22321447.v1 |
| intermediate | Liver spatial domains and zonation | included | 717.8 MiB | `intermediate/liver/spatial_domains_and_zonation.h5ad` | notebooks/main/liver_visium_BA_normal.ipynb |
| intermediate | Liver joint co-embedding | included | 1.4 GiB | `intermediate/liver/joint_coembedding.h5ad` | notebooks/main/liver_visium_BA_normal.ipynb |
| intermediate | Liver spatial deconvolution | included | 600.0 MiB | `intermediate/liver/spatial_deconvolution.h5ad` | notebooks/main/liver_visium_BA_normal.ipynb |
| intermediate | Liver deconvolution estimates | included | 1.8 MiB | `intermediate/liver/deconvolution_estimates.csv` | notebooks/main/liver_visium_BA_normal.ipynb |

## Whole-brain MERFISH

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | Whole-brain MERFISH | link only | external | `raw/README_whole_brain_merfish.md` | Zhang et al. doi:10.1038/s41586-023-06808-9 |
| intermediate | Whole-brain MERFISH | included | 399.4 MiB | `intermediate/whole_brain_merfish/merged_and_step_outputs/` | scripts/merfish_whole_brain/run_merfish.py |
| output | Whole-brain MERFISH | included | 23.3 MiB | `outputs/workflows/merfish_whole_brain_8_4m/` | scripts/merfish_whole_brain/run_merfish.py |

## Cross-platform mouse brain

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| raw | Cross-platform mouse brain | link only | external | `raw/README_cross_platform_mouse_brain.md` | Published NicheCompass MERFISH and STARmap PLUS reference |
| intermediate | Cross-platform mouse brain annotations | included | 35.1 MiB | `intermediate/cross_platform_mouse_brain/evaluation_annotations.parquet` | scripts/cross_platform_mouse_brain/build_evaluation_annotations.py |
| intermediate | Cross-platform mouse brain annotation summary | included | 866.0 B | `intermediate/cross_platform_mouse_brain/evaluation_annotations.summary.json` | scripts/cross_platform_mouse_brain/build_evaluation_annotations.py |
| intermediate | Cross-platform mouse brain STEP | included | 1.7 GiB | `intermediate/cross_platform_mouse_brain/step_9471568_k10/` | scripts/cross_platform_mouse_brain/run_step_cross_platform.py |
| output | Cross-platform mouse brain comparison | included | 23.6 MiB | `outputs/workflows/cross_platform_mouse_brain/` | scripts/cross_platform_mouse_brain/compare_step_nichecompass.py; scripts/cross_platform_mouse_brain/compare_step_nichecompass_domains.py |

## Spatial coherence simulation

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| intermediate | Spatial coherence simulations | included | 23.5 MiB | `intermediate/validation/coherence_simulation/` | Canonical simulated inputs and STEP outputs consumed by the coherence validation workflow |
| output | Spatial coherence simulations | included | 8.5 MiB | `outputs/workflows/coherence_simulation/` | scripts/coherence_simulation/ and notebooks/validation/spatial_coherence_simulation.ipynb |

## Multi-dataset runtime benchmark

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| output | External benchmark runtime | included | 127.9 KiB | `outputs/workflows/external_benchmark_runtime/` | scripts/runtime/make_external_runtime_figure.py |

## Synthetic graph-sampling control

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| output | GraphSAINT sampling | included | 546.8 KiB | `outputs/workflows/graphsaint_sampling/` | scripts/graphsaint_sampling/simulate_square_grid_node_sampling.py |

## Image-based graph sensitivity

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| output | Image-based graph k sweep | included | 3.1 MiB | `outputs/workflows/image_based_k_sweep/` | scripts/image_based_k_sweep/ |

## Synthetic module ablation

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| output | STEP module ablation | included | 24.6 MiB | `outputs/workflows/module_ablation/` | scripts/module_ablation/ |

## CRC, prostate, and MOSTA external benchmarks

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| output | External spatial-domain cluster plots | included | 103.5 MiB | `outputs/workflows/external_spatial_domain_plots/` | scripts/external_benchmark/plot_external_method_clusters.py |

## CRC and prostate graph-training controls

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| output | STEP complete-neighborhood controls | included | 38.4 MiB | `outputs/workflows/step_complete_neighborhood_controls/` | scripts/validation_controls/step_fullgraph_prostate_crc.py |

## MOSTA E16.5 Liver validation

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| output | MOSTA Liver biological and cross-section validation | included | 76.0 MiB | `outputs/workflows/mosta_liver_validation/` | scripts/validation_controls/mosta_liver_evidence_chain.py; scripts/validation_controls/mosta_liver_cross_section_transfer.py |

## MERFISH and DLPFC graph-training controls

| Tier | Item | Packaging | Size | Archive path | Source or producer |
|---|---|---:|---:|---|---|
| output | MERFISH and DLPFC graph-training controls | included | 90.6 MiB | `outputs/workflows/graph_training_controls/` | scripts/validation_controls/advanced_graph_training_modes.py; scripts/validation_controls/dlpfc_method_graph_modes.py |
