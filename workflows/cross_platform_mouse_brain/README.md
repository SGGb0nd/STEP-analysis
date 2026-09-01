# Cross-technology mouse-brain atlas

This workflow compares STEP with the published NicheCompass joint model on the
same MERFISH mouse-brain and STARmap PLUS mouse-CNS sections.

The published NicheCompass reference contains 239 MERFISH sections and 20
STARmap PLUS sections, totaling 9,471,568 cells measured over 432 shared genes.
Its matrix contains integer-valued counts. STEP reads those exact cells, genes,
section labels, and coordinates. Each section retains an independent spatial
graph. STEP uses ten non-self nearest neighbors within each section.

An optional compact STEP-only input can be written with:

```bash
uv run python scripts/cross_platform_mouse_brain/prepare_cross_platform_input.py \
  --reference-h5ad data/nichecompass_reference/anndata_umap_with_clusters.h5ad \
  --output-h5ad results/cross_platform_mouse_brain/input_432_genes.h5ad
```

Train STEP with:

```bash
uv run python scripts/cross_platform_mouse_brain/run_step_cross_platform.py \
  --reference-h5ad data/nichecompass_reference/anndata_umap_with_clusters.h5ad \
  --output-dir results/cross_platform_mouse_brain/step_9471568_k10 \
  --spatial-neighbors 10
```

The metric table and figures are generated after aligning the STEP embedding
with the published NicheCompass model by technology, section, and original
observation identifier.

Build the biological annotation sidecar for the public 50 MERFISH + 20 STARmap
evaluation subset:

```bash
uv run python scripts/cross_platform_mouse_brain/build_evaluation_annotations.py \
  --input-root data/nichecompass_cross_platform \
  --output-parquet results/cross_platform_mouse_brain/evaluation_annotations.parquet
```

Run the matched comparison with:

```bash
uv run python scripts/cross_platform_mouse_brain/compare_step_nichecompass.py \
  --step-output-dir results/cross_platform_mouse_brain/step_9471568_k10 \
  --nichecompass-h5ad data/nichecompass_reference/anndata_umap_with_clusters.h5ad \
  --annotation-parquet results/cross_platform_mouse_brain/evaluation_annotations.parquet \
  --step-evaluation-embedding results/cross_platform_mouse_brain/comparison_k10/step_no_graph_evaluation_embedding.npy \
  --step-evaluation-index results/cross_platform_mouse_brain/comparison_k10/step_no_graph_evaluation_cells.parquet \
  --output-dir results/cross_platform_mouse_brain/comparison_k10_no_graph
```

The comparison aligns STEP and NicheCompass by unchanged observation
identifiers. Biological annotations from the public source atlases are joined
as evaluation metadata and are never used for training. Evaluation is balanced
within each shared broad cell type and technology; a cell type must contain at
least 100 cells in each technology to enter the matched comparison.

Plot both STEP representation levels with:

```bash
uv run python scripts/cross_platform_mouse_brain/make_dual_embedding_umap.py \
  --celltype-comparison-dir results/cross_platform_mouse_brain/comparison_k10_no_graph \
  --domain-comparison-dir results/cross_platform_mouse_brain/domain_comparison_k10 \
  --step-output-dir results/cross_platform_mouse_brain/step_9471568_k10 \
  --output-dir results/cross_platform_mouse_brain/dual_embedding_k10
```
