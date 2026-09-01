# Graph-training controls

This workflow compares whole-graph, complete-neighborhood mini-batch, and
sampled training for GraphST and STAGATE.

`merfish_runs/` contains compact assignments and metrics for five sagittal
MERFISH sections. `dlpfc_runs/` contains the spatial-graph training and
inference controls for all 12 DLPFC sections. `summary/` provides the aggregate
tables and figures included in this result bundle.

The complete-neighborhood mode expands each target mini-batch to the exact
finite-hop computation graph required by that method. Final embeddings are
inferred on the complete section graph in every training mode.
