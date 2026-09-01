# MOSTA E16.5 STEP analysis

This workflow integrates five MOSTA E16.5 Stereo-seq sections with STEP and
evaluates spatial-domain recovery, spatial continuity, and batch mixing.

Run the analysis with:

```bash
uv run python scripts/mosta_e16_stereoseq/run_mosta_e16_step.py \
  --input-h5ad data/mosta_e16_test.h5ad \
  --output-dir results/mosta_e16_step
```

## Settings

- Raw counts: `layers["count"]`
- Genes: all 2,000 genes in the input object
- Decoder hidden layers: 2
- Decoder batch adjustment: `BatchAwareScale` after each hidden linear layer
- Graph layers: 2
- Graph edge cutoff: 1.5 coordinate units
- Graph batch size: 2
- Node sample size: 2,048
- Spatial domains: 27

The complete settings are recorded in `settings.json`.

## Results

- `batch_counts.csv`: spot count for each embryo section.
- `step_mosta_metrics.csv`: section-level ARI, NMI, CHAOS, and PAS.
- `step_batch_metrics.json`: dataset-level ASW and iLISI.
