# Reproducing the analyses

Start with [`analysis_index.md`](analysis_index.md), which links each manuscript
analysis to its notebook or script and its result directory.

Create the locked Python environment as described in
[`environment.md`](environment.md).

## Notebooks

The notebooks contain sectioned workflows with saved outputs for direct
inspection. Paths under `data/` and `results/` can be connected to local data
storage before execution.

## Scripts

The scripts expose command-line input and output paths. Dataset settings,
metric definitions, and output files are described in the script docstrings
and the corresponding workflow README.

## Workflow results

Each directory under `workflows/` contains a focused result bundle. Figures
are accompanied by metric tables or JSON settings when those values are part
of the analysis.
