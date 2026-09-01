# Data sources and accessions

The analyses use the public datasets listed in the manuscript Data
Availability statement. Large source datasets are not duplicated in this
repository. The table below connects each source to its expected local input
location.

| Dataset | Public source | Expected local input |
|---|---|---|
| Human colorectal cancer, 10x Visium HD | [10x Genomics datasets](https://www.10xgenomics.com/resources/datasets) | `data/visium-hd-crc/` and `data/visium-hd/human-coloretal-cancer/` |
| Mouse small intestine, 10x Visium HD | [10x Genomics datasets](https://www.10xgenomics.com/resources/datasets) | Notebook-configured Visium HD input paths |
| Human prostate, Slide-seq V2 | [GEO GSE181294](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181294) | `data/slide-seq-tumor/` |
| scRNA-seq integration benchmarks | [scib-reproducibility](https://github.com/theislab/scib-reproducibility) | `data/scib_benchmark/` |
| Human DLPFC, 10x Visium | [spatialLIBD HumanPilot](https://github.com/LieberInstitute/HumanPilot) | `data/<section>/<section>_annotated.h5ad` |
| Mouse hypothalamus, MERFISH | [Dryad doi:10.5061/dryad.8t8s248](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) | `data/merfish_animal1.h5ad` and `data/merfish_anno/` |
| Mouse whole brain, MERFISH | [Zhang et al. whole-brain atlas](https://doi.org/10.1038/s41586-023-06808-9) | Section-level H5AD files supplied to `scripts/merfish_whole_brain/run_merfish.py` |
| Cross-technology mouse brain, MERFISH and STARmap PLUS | [NicheCompass reproducibility repository](https://github.com/Lotfollahi-lab/nichecompass-reproducibility); [MERFISH CELLxGENE collection](https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09); [STARmap PLUS Zenodo record](https://doi.org/10.5281/zenodo.8327576) | Published NicheCompass reference H5AD supplied to `scripts/cross_platform_mouse_brain/run_step_cross_platform.py` |
| Mouse medial prefrontal cortex, STARmap | [STARmap resources](https://www.starmapresources.org/data) | `data/mpfc_160/` |
| Mouse embryo E16.5, Stereo-seq | [CNGBdb CNP0001543](https://db.cngb.org/search/project/CNP0001543/) | `data/mosta_e16_test.h5ad` |
| Human normal and biliary-atresia liver, 10x Visium and snRNA-seq | [Transcriptomics technical optimization](https://github.com/julietusc/Transcriptomics_technical_optimization) | `data/Transcriptomics_technical_optimization/` |
| Human normal liver, 10x Visium | [Figshare doi:10.6084/m9.figshare.22321447.v1](https://doi.org/10.6084/m9.figshare.22321447.v1) | `data/liver_normal2/` |

The CRC and prostate MA workflows also use generated intermediate artifacts.
Their producer-consumer chain and Zenodo packaging status are listed in
[`zenodo_manifest.tsv`](zenodo_manifest.tsv).

The exact published cross-technology reference and its local checksum are
documented in [`cross_platform_mouse_brain_sources.md`](cross_platform_mouse_brain_sources.md).
