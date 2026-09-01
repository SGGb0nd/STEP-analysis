# Matched STEP and NicheCompass comparison

All metrics are calculated on the same observation IDs, balanced within
shared broad cell types and technologies. Higher values are better.

| Method | Cell-type ASW | Cell-type ARI | Cell-type NMI | kNN cell-type purity | Batch ASW | Batch iLISI | Cross-technology cell-type agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| STEP | 0.5087 | 0.2935 | 0.4305 | 0.5815 | 0.7292 | 0.1232 | 0.4169 |
| NicheCompass | 0.4970 | 0.1142 | 0.2498 | 0.3346 | 0.8418 | 0.0117 | 0.2167 |

Both embeddings retain broad cell-type organization, while the low
technology iLISI values and the technology-colored UMAPs show that
MERFISH and STARmap PLUS remain substantially separated. STEP has higher
biological ASW, cell-type ARI and NMI, local cell-type purity,
technology iLISI, and
cross-technology cell-type agreement, while NicheCompass has higher
technology ASW.
