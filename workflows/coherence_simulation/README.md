# Spatial coherence simulations

This bundle contains the two simulation panels used to support interpretation of STEP spatial coherence.

Results:

- `coherence_validation.png`
- `ccc_validation.png`

Code:

- `../../scripts/coherence_simulation/composition_driven_coherence.py` generates the known-label simulation and STEP outputs.
- `../../scripts/coherence_simulation/plot_coherence_validation.py` renders the final known-label validation panel from the saved simulation object.
- `../../scripts/coherence_simulation/local_ccc_negative_coherence.py` generates and plots the local communication-driven simulation.

The first panel evaluates coherence against composition-driven local cell-type heterogeneity under known cell-state and spatial-domain structure. The second evaluates coherence in a local communication-driven simulation.

`../../notebooks/validation/spatial_coherence_simulation.ipynb` presents both
simulations and runs the functions defined in these scripts.
