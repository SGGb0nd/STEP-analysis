# STEP analysis

This repository contains the notebooks, analysis scripts, and result figures
for **Decoding Spatial Microarchitectures in Complex Tissues**.

## Start here

- [`docs/analysis_index.md`](docs/analysis_index.md) maps each analysis to its code and results.
- [`docs/reproduction_guide.md`](docs/reproduction_guide.md) describes the repository layout and execution conventions.
- [`docs/environment.md`](docs/environment.md) defines the locked Python environment.
- [`docs/data_accessions.md`](docs/data_accessions.md) lists source accessions and expected local paths.
- [`docs/zenodo_manifest.tsv`](docs/zenodo_manifest.tsv) lists the raw, intermediate, and output artifacts for deposition.
- [`docs/zenodo_deposition.md`](docs/zenodo_deposition.md) describes deterministic packaging and draft upload.
- [`notebooks/`](notebooks/) contains dataset workflows and tutorials.
- [`scripts/`](scripts/) contains simulations, benchmarks, and figure generation.
- [`workflows/`](workflows/) contains compact figures, tables, and summary values.

## Analysis topics

- Spatial-domain benchmarks on 10x Visium, MERFISH, STARmap, Stereo-seq,
  Visium HD, and Slide-seq V2.
- CRC and prostate microarchitecture identification and interpretation.
- Liver zonation and cross-modal deconvolution.
- Spatial coherence simulations and STEP module ablation.
- Graph construction, sampling, threshold, and runtime analyses.
- Whole-brain MERFISH integration across 239 sections.
- Matched MERFISH and STARmap PLUS cross-technology mouse-brain integration.

## Data sources

| Dataset | Public source |
|---|---|
| Human colorectal cancer, 10x Visium HD | [10x Genomics datasets](https://www.10xgenomics.com/resources/datasets) |
| Human prostate, Slide-seq V2 | [GEO GSE181294](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181294) |
| Human DLPFC, 10x Visium | [spatialLIBD HumanPilot](https://github.com/LieberInstitute/HumanPilot) |
| Mouse hypothalamus, MERFISH | [Dryad doi:10.5061/dryad.8t8s248](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| Mouse whole brain, MERFISH | [Zhang et al. whole-brain atlas](https://doi.org/10.1038/s41586-023-06808-9) |
| Cross-technology mouse-brain reference | [NicheCompass reproducibility repository](https://github.com/Lotfollahi-lab/nichecompass-reproducibility) |
| Mouse medial prefrontal cortex, STARmap | [STARmap Resources](https://www.starmapresources.org/data) |
| Mouse embryo E16.5, Stereo-seq | [CNGBdb CNP0001543](https://db.cngb.org/search/project/CNP0001543/) |
| Human normal and biliary-atresia liver | [Transcriptomics technical optimization](https://github.com/julietusc/Transcriptomics_technical_optimization) |
| Human normal liver, 10x Visium | [Figshare doi:10.6084/m9.figshare.22321447.v1](https://doi.org/10.6084/m9.figshare.22321447.v1) |
| scRNA-seq integration benchmarks | [scib-reproducibility](https://github.com/theislab/scib-reproducibility) |
| Mouse small intestine, 10x Visium HD | [10x Genomics datasets](https://www.10xgenomics.com/resources/datasets) |

Expected local paths and artifact provenance are provided in
[`docs/data_accessions.md`](docs/data_accessions.md).

## Environment

The analyses use Python 3.11, `step-kit==0.3`, PyTorch 2.3.1 with CUDA
12.1, and DGL 2.5.0 built for PyTorch 2.3 and CUDA 12.1. Create the Python
environment with:

```bash
uv sync --extra benchmark --extra notebook --extra microarchitecture
```

See [`docs/environment.md`](docs/environment.md) for the resolved
package-source and build-system details.

Place input H5AD files under `data/` or link that directory to the local data
store. Each runnable script accepts explicit input and output paths.

Public source datasets are not duplicated in this repository. Their expected
locations are documented by the relevant notebook or command-line interface.
Generated data and model checkpoints are written under the ignored `results/`
directory.

## Results

Notebook outputs show the main analysis steps for each dataset. Workflow
directories provide compact result bundles with figures, metric tables, and
the settings needed to reproduce them.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The source
code and documentation in this repository are distributed under the Apache
License 2.0.
