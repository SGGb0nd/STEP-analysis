# CRC gene-distance heatmaps

These heatmaps show expression in tumor-periphery spots ordered by their
nearest-tumor distance for the three CRC sections. The analysis retains spots
within 100 full-resolution image pixels and converts the sorted distances to
microns with the section-specific scale factors stored in the AnnData object.

The first five rows are stromal/periphery-associated genes and the final five
rows are epithelial/tumor-associated genes. Expression is recovered from the
saved log-transformed matrix, normalized per spot to a library size of 100,000,
and log transformed before plotting.

Run:

```bash
python scripts/crc_visium_hd/make_crc_gene_distance_heatmaps.py \
  --h5ad results/visium-hd/cancer_tumor_regions.h5ad \
  --outdir workflows/crc_gene_distance_heatmaps
```

The script isolates the original plotting parameters with
`matplotlib.rc_context`, fixes the colormap and color limits explicitly, and
omits `interpolation_stage` on Matplotlib versions that do not support it.
