# Analysis scripts

This directory contains simulations, benchmarks, and figure-generation scripts
used in the manuscript analyses.

The scripts are organized by dataset or analysis topic. Supply datasets and
output directories through their command-line arguments.

## Contents

- `graphsaint_sampling/simulate_square_grid_node_sampling.py`: GraphSAINT node-sampling simulation on a square grid.
- `coherence_simulation/composition_driven_coherence.py`: composition-driven spatial coherence simulation.
- `coherence_simulation/local_ccc_negative_coherence.py`: local CCC-driven coherence simulation.
- `coherence_simulation/plot_coherence_validation.py`: plot STEP coherence validation from a saved simulation object.
- `image_based_k_sweep/k_sweep_image_based.py`: simulation-based kNN sensitivity analysis for irregular spatial graphs.
- `image_based_k_sweep/make_image_based_k_sweep_figure.py`: summary figure generation for the image-based kNN sensitivity analysis.
- `crc_visium_hd/plot_periphery_continuity.R`: CRC tumor-periphery singular-point distance plot.
- `crc_visium_hd/ma_threshold_sensitivity.py`: CRC and Slide-seq MA threshold sensitivity analysis.
- `crc_visium_hd/make_crc_interface_zonation_stacked.py`: CRC tumor-periphery distance-zonation marker summary.
- `crc_visium_hd/make_crc_gene_distance_heatmaps.py`: distance-ranked CRC tumor-periphery expression heatmaps with physical-distance axes.
- `crc_visium_hd/crc_ma_histomorphology.py`: graph-level CRC MA histomorphological context analysis.
- `external_benchmark/external_method_benchmark.py`: CRC/MOSTA external-method benchmark calculation.
- `external_benchmark/compute_mosta_scib_batch_metrics.py`: MOSTA batch-mixing metric calculation.
- `external_benchmark/plot_external_method_clusters.py`: CRC, MOSTA, and prostate Slide-seq V2 external-method cluster visualization and data-loading utilities.
- `mosta_e16_stereoseq/make_metric_table.py`: aggregate STEP and external-method metrics and render the MOSTA comparison table.
- `mosta_e16_stereoseq/run_mosta_e16_step.py`: train STEP and evaluate spatial domains across five MOSTA E16.5 sections.
- `merfish_whole_brain/run_merfish.py`: merge the raw whole-brain MERFISH sections, train STEP across 239 sections, and export the 8.4-million-spot representation and spatial-domain results.
- `cross_platform_mouse_brain/run_step_cross_platform.py`: train STEP jointly on the exact 239-section MERFISH and 20-section STARmap PLUS reference used by NicheCompass.
- `cross_platform_mouse_brain/compare_step_nichecompass.py`: align the two embeddings by observation ID and compute matched biological-conservation and technology-mixing metrics.
- `crc_visium_hd/run_lr_analysis.py`: reproduce the per-MicroArchitecture CellPhoneDB analysis, within-MA BH-FDR correction, and CRC communication summaries.
- `prostate_slideseq_v2/build_microarchitecture.py`: generate the five prostate MA assignments and graph intermediates from the fitted Slide-seq V2 STEP object.
- `module_ablation/step_module_ablation_sim.py`: STEP module-ablation simulation and metric aggregation.
- `module_ablation/make_ablation_metric_panel.py`: BBM/BEM/SpM module-ablation metric panel from paired simulation summaries.
- `module_ablation/make_ablation_case_figures.py`: matched representative spatial cases for the module-ablation study.
- `runtime/make_external_runtime_figure.py`: runtime comparison figure for the large-scale benchmarks.
- `validation_controls/step_fullgraph_prostate_crc.py`: matched sampled and complete-neighborhood STEP controls on CRC and prostate data.
- `validation_controls/advanced_graph_training_modes.py`: whole-graph, complete-neighborhood, and sampled GraphST/STAGATE controls on MERFISH.
- `validation_controls/dlpfc_method_graph_modes.py`: matched graph-training and graph-inference controls on DLPFC sections.
- `validation_controls/mosta_common_spot_cavity_benchmark.py`: common-spot Cavity comparison across STEP and external methods.
- `validation_controls/mosta_cavity_local_signal_control.py`: finite-hop local-signal control for Cavity recovery.
- `validation_controls/mosta_liver_domain_validation.py`: shared MOSTA Liver domain, marker-score, and spatial validation utilities.
- `validation_controls/mosta_liver_evidence_chain.py`: fetal-liver marker, DEG, and pathway evidence across MOSTA sections.
- `validation_controls/mosta_liver_cross_section_transfer.py`: leave-one-section-out transcriptomic validation of MOSTA Liver domains.
- `deposition/build_zenodo_archives.py`: validate the deposition manifest and build deterministic tiered archives.
- `deposition/upload_zenodo_draft.py`: create or update a Zenodo draft record without publishing it.
- `deposition/finalize_zenodo_draft.py`: verify the uploaded file set and publish a complete open-access record.

## Metric conventions

MOSTA batch ASW and iLISI are dataset-level metrics computed on the complete
five-section embedding with the `scib-metrics` functional API. ARI, NMI, PAS,
and CHAOS are computed for each section.
