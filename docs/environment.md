# Analysis environments

## Python

The canonical Python environment targets Linux x86-64 systems with an NVIDIA
GPU supported by CUDA 12.1. It uses Python 3.11, PyTorch 2.3.1 with CUDA 12.1,
and DGL 2.5.0 built against PyTorch 2.3 and CUDA 12.1.

```bash
uv sync --extra benchmark --extra notebook --extra microarchitecture
```

The package indexes for the PyTorch and DGL wheels are declared explicitly in
`pyproject.toml`, and their resolved wheel URLs and hashes are recorded in
`uv.lock`. The lockfile is the authoritative Python environment for these
analyses.

The public `step-kit==0.3` package was built from the earlier Poetry project.
Its `cu117` extra records package names and version constraints in the wheel,
but Poetry's supplemental package indexes are not part of published wheel
metadata. The analysis environment therefore declares PyTorch and DGL
directly and binds each package to its CUDA 12.1 index through uv. This is a
current analysis environment, not a reconstruction of the historical CUDA
11.7 Poetry environment.

## Build and dependency resolution

The public STEP 0.3 repository uses `poetry-core` to build the distribution.
Poetry also resolves its development environment and honors
`[[tool.poetry.source]]` while working from that repository. The built wheel
contains dependency names, version constraints, and extras, but not those
repository-specific source definitions.

STEP-analysis declares its environment in the PEP 621 `[project]` table and
sets `[tool.uv] package = false`. It is an analysis repository rather than a
Python distribution, so `uv sync` does not build or install STEP-analysis
itself. uv acts as the resolver and installer: it reads `[tool.uv.sources]`,
selects the explicit CUDA wheel indexes, and records the complete resolution
in `uv.lock`.

The current STEP development repository also uses PEP 621, Hatchling, and uv.
Its uv source configuration controls environments created from a source
checkout; as with Poetry sources, those uv-specific indexes are not embedded
in a wheel uploaded to PyPI. Published installation instructions must still
state how PyTorch and DGL are selected.

`pyarrow` is installed directly because the Visium HD readers consume Parquet
spatial-position files. `setuptools<81` remains pinned for the
`pkg_resources` import used by the public STEP 0.3 package.
