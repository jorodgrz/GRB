# Test layout

Tests are organised so the folder a file sits in determines which marker
auto-applies (see ``pytest_collection_modifyitems`` in
[tests/conftest.py](conftest.py)). Run a slice with ``-m <marker>``.

```
tests/
├── conftest.py             # shared fixtures + auto-marker hook
├── unit/                   # @pytest.mark.unit
├── sections/               # @pytest.mark.section_<N>
├── anchors/                # @pytest.mark.anchors
└── integration/            # @pytest.mark.integration
```

## Folders

### `unit/` (fast, every PR)

Pure module-level invariants over `grb_*.py`. No `data/`, no
`compas_python_utils`, no notebook coupling. Runs in `make smoke` and
`make ci`.

| File | Module under test |
|---|---|
| `test_classify_asserts.py` | `grb_classify._resolve_m_thresh`, `classify_grid` ns_max validation |
| `test_grb_io.py` | `grb_io` loaders against synthetic HDF5 |
| `test_isco.py` | `grb_physics.r_isco` against Bardeen 1972 |
| `test_palette_and_rcparams.py` | `grb_plot_style` palette + canonical rcParams |
| `test_phase4_helpers.py` | `marginalize_bh_spin`, `channel_class_crosstab`, offsets vectorisation |
| `test_physics.py` | Foucart 2018, Kruger-Foucart 2020, MSSFR normalisation |
| `test_rates.py` | `dPdlogZ`, `compute_merger_rate`, `calibrate_mean_mass_evolved` |
| `test_beaming.py` | `beamed_rate`, `beamed_rate_mixed`, `CLASS_THETA_J` |
| `test_bns_jet_breakout.py` | `breakout_fraction_bns_eos` and the intrinsic-to-observable rate ladder |
| `test_eos_sensitivity.py` | `compute_eos_sensitivity` contract on one population |
| `test_eos_sensitivity_grid.py` | `compute_eos_sensitivity` over NS-mass treatment x EOS |
| `test_offsets_kick_velocity.py` | kick velocity sets the offset scale in `compute_offsets_mixed_hosts` |

### `sections/` (one file per notebook section)

Thin smoke tests aligned with the figures in
[grb_main.ipynb](../grb_main.ipynb). Run a single section with
`pytest -m section_<N>`.

| Section | File | Notebook figure |
|---|---|---|
| 1 | `test_section_01_mass_plane.py` | Mass Plane (BNS and BHNS) |
| 2 | `test_section_02_component_masses.py` | Component Mass Distributions by GRB Class |
| 3 | `test_section_03_delay_times.py` | Delay Time Distributions by GRB Class |
| 4 | `test_section_04_mssfr.py` | Cosmic Integration / MSSFR Grid Setup |
| 5 | `test_section_05_metallicity_efficiency.py` | Metallicity Dependence of GRB Formation Efficiency |
| 6 | `test_section_06_bns_rate.py`, `test_section_06_lvk_rate_comparison.py` (+ `06b` channel split and `06b` formation-channel classifier) | BNS Merger Rate R(z) per GRB Class |
| 7 | `test_section_07_bhns_spin.py`, `test_section_07_lvk_rate_comparison.py` (+ `07b` channel split) | BHNS Merger Rate R(z) with BH Spin Sensitivity |
| 8 | `test_section_08_offsets.py` | Physical Host-Galaxy Offset Distributions |
| 8a | `test_section_08a_free_drift.py` | Free-drift check: what the natal kick and drift time are worth with no potential; what the two panels are allowed to claim (the collinear two-leg bound, the isotropic projection factor, and why the joint density is binned over the sample rather than the frame); and the data provenance the figure rests on (the two loaders return the same rows, the drift-baseline identity holds, the columns carry no sentinel values, the assumed units survive a range check, and the file's kick directions are a selected anisotropic sample rather than a bug) |
| 8c | `test_section_08c_free_drift_grid.py` | Free drift across the 20-model grid: that a weighted median of a fraction keeps its zeros, that the first leg is negligible on the fiducial and dominant on EH, that exactly two models leave the BHNS disk class empty, and that the grid moves the drift time far more than the kick |
| 8b | `test_section_08b_host_routing.py` | Delay-time host assignment: the sign of the offset shift for prompt and slow populations, and that the elliptical fraction becomes a prediction rather than the assumed 0.25 |
| 11 | `test_section_11_model_grid_scan.py` | Full 20-model Broekgaarden+ 2021 grid scan (population-synthesis robustness) |

### `anchors/` (literature audit)

Every cited number that the pipeline consumes (M_TOV, K_THRESH_DEFAULT,
EOS table, Kroupa slopes, Planck 2015, Foucart 2018 coefficients,
Wanderman-Piran 2015, beaming bands) lives here as a paper-anchored
assertion. If a literal in `grb_*.py` drifts without updating the
paper citation, this is the file that will tell you.

### `integration/` (manual dispatch in CI)

Tests that need `data/`, `compas_python_utils`, or notebook execution.
Runs locally via `make test`; in CI only on manual `workflow_dispatch`.

| File | Why it is integration |
|---|---|
| `test_compas_pin.py` | Imports the upstream `compas_python_utils` pin |
| `test_comparison_notebook.py` | Reads `data/rastinejad_2024.csv` |
| `test_cosmic_integration_vs_compas.py` | Bin-for-bin cross-check with COMPAS |
| `test_grb_io_realdata.py` | Audits real Broekgaarden+ 2021 HDF5 archives |
| `test_grb_main_notebook.py` | Executes `grb_main.ipynb` via `nbclient` |
| `test_notebook_output_vs_literature.py` | Class fractions vs published bands |
| `test_rate_class_shape.py` | R(z) shape on real BNS-A data |

## Markers

Defined in [pyproject.toml](../pyproject.toml) under
`[tool.pytest.ini_options].markers`. Composable: `pytest -m "unit and not slow"`.

| Marker | Auto-applied by | Meaning |
|---|---|---|
| `unit` | folder | Module-level invariant |
| `anchors` | folder | Literature anchor |
| `integration` | folder | Heavy / data-bound / notebook |
| `section_1` ... `section_12` | filename | Notebook-section coupling |
| `slow` | manual | Wall-clock above ~5 s |
| `requires_data` | manual | Needs `data/COMPASCompactOutput_*.h5` |
| `requires_compas` | manual | Needs `compas_python_utils` importable |

## Running subsets

```bash
make smoke                                 # fast PR set
make coverage                              # unit + anchors with 70 percent floor
pytest -m section_5                        # Section 5 tests only
pytest -m "unit and not slow"              # all fast unit tests
pytest tests/integration/ -m requires_data # data-bound integration only
```
