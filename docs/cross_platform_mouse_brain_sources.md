# Cross-platform mouse-brain reference

The matched STEP and NicheCompass comparison uses the published NicheCompass
mouse-brain atlas without reconstructing or relabeling its observations. The
reference combines 8,380,288 MERFISH cells from 239 sections with 1,091,280
STARmap PLUS cells from 20 sections. The shared matrix contains 9,471,568 cells
and 432 genes.

The processed NicheCompass archive is distributed by the NicheCompass
reproducibility project through Google Drive:

- File: `mouse_brain_atlas.zip`
- Google Drive file ID: `1HsUpk2Lwtl36pMrgW0lh0ONQy-v__iRO`
- Size: 10,393,216,682 bytes
- SHA-256: `373c8dbf3032a9159075d9fc936917330bf791808f32281e6c1705b6ed348609`
- H5AD member: `trained_model_umaps:v2/anndata_umap_with_clusters.h5ad`

The underlying public datasets are the
[whole-brain MERFISH CELLxGENE collection](https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09)
and the [STARmap PLUS mouse CNS atlas](https://doi.org/10.5281/zenodo.8327576).
The processed reference is documented by the
[NicheCompass reproducibility repository](https://github.com/Lotfollahi-lab/nichecompass-reproducibility).

The official archive is referenced rather than duplicated in the STEP Zenodo
deposit. STEP model outputs, aligned evaluation annotations, metrics, and plots
are included in the deposit.
