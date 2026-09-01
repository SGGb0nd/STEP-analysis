# Validation controls

These scripts reproduce the graph-training, MOSTA Cavity, and MOSTA Liver
validation analyses included with the study.

## Graph-training controls

- `step_fullgraph_prostate_crc.py` compares sampled STEP training with matched
  complete-neighborhood mini-batches while retaining complete-section graph
  inference.
- `advanced_graph_training_modes.py` runs whole-graph,
  complete-neighborhood, and sampled GraphST or STAGATE controls on MERFISH.
- `dlpfc_method_graph_modes.py` applies the matched graph-training controls to
  one DLPFC section. Run it over the 12 section identifiers listed in
  `dlpfc_tutorial_methods.py` to reproduce the aggregate panel.
- `plot_merfish_training_controls.py` and
  `plot_dlpfc_graph_training_controls.py` render the compact summary tables in
  `workflows/graph_training_controls/summary/`.

## MOSTA Cavity controls

- `mosta_common_spot_cavity_benchmark.py` evaluates STEP, NicheCompass,
  BANKSY, GraphST, and STAGATE on matched spots within each section.
- `mosta_cavity_local_signal_control.py` tests finite-hop Gaussian smoothing
  across controlled no-graph embedding dispersions.
- `plot_mosta_cavity_all_sections.py` assembles the section-level Cavity
  assignments and metrics.

The NicheCompass and BANKSY inputs are supplied through
`--nichecompass-result` and `--banksy-embedding`. Their compact assignments,
metrics, and figures are included in the workflow output; full external-method
objects remain available from the corresponding method resources.

## MOSTA Liver validation

- `mosta_liver_domain_validation.py` provides the shared domain, marker-score,
  and spatial-summary utilities used by the liver analyses.
- `mosta_liver_evidence_chain.py` evaluates literature-supported fetal-liver
  programs and domain associations across five sections.
- `mosta_liver_cross_section_transfer.py` performs leave-one-section-out
  transcriptome-based domain prediction.
- `compose_mosta_liver_validation.py` combines the spatial, enrichment, and
  cross-section panels.

All commands accept repository-relative input and output paths. Run them from
the repository root so that `data/`, `results/`, and `workflows/` resolve
consistently.
