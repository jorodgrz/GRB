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

| File | Covers |
|---|---|
| `test_grb_{classify,io,offsets,physics,rates}_rigor.py` | Behaviour and edge cases of each module |
| `test_physics.py` | Foucart 2018, Kruger and Foucart 2020, NS-mass remap support |
| `test_isco.py` | `r_isco` against Bardeen et al. (1972) |
| `test_ns_remap_and_hist_pdf.py` | Alsing et al. (2018) remap mechanics; `weighted_hist_pdf` normalisation |
| `test_classify_asserts.py` | Runtime asserts in `grb_classify` (`m_thresh`, `ns_max`) |
| `test_class_channel_split.py` | Per-class per-channel split invariants (Sections 6b, 7b) |
| `test_grb_io.py` | Loaders against synthetic HDF5 |
| `test_tng_sfrd.py` | TNG S(Z,z) tables as cosmic-integration input |
| `test_rates.py` | `dPdlogZ`, `compute_merger_rate`, `calibrate_mean_mass_evolved`, smoothing, bootstrap |
| `test_phase4_helpers.py` | `marginalize_bh_spin`, `channel_class_crosstab`, remap marginals |
| `test_beaming.py` | `beamed_rate`, `beamed_rate_mixed`, `CLASS_THETA_J` |
| `test_bns_jet_breakout.py` | `breakout_fraction_bns_eos` and the intrinsic-to-observable chain |
| `test_eos_sensitivity.py`, `test_eos_sensitivity_grid.py` | `compute_eos_sensitivity` on one population and over EOS x mass treatment |
| `test_palette_and_rcparams.py` | Palette and ApJ rcParams |
| `test_ballistic_offset.py` | Free-drift offsets |
| `test_birth_radius_truncation.py` | Truncated birth radii renormalise rather than clip |
| `test_offsets_kick_velocity.py` | Kick velocity sets the offset scale |
| `test_verlet_integration_duration.py` | Verlet integrator runs for each system's own delay time |

### `sections/` (one file per notebook section)

Thin smoke tests aligned with the figures in
[grb_main.ipynb](../grb_main.ipynb). Run a single section with
`pytest -m section_<N>`.

| Section | File | Notebook figure |
|---|---|---|
| 1 | `test_section_01_mass_plane.py` | Mass plane, BNS and BHNS |
| 3 | `test_section_03_delay_times.py` | Delay-time distributions by class |
| 3b | `test_section_03b_delay_by_channel.py`, `test_section_03b_channel_rate_shares.py` | Delay time split by channel; channel shares of R(z=0) |
| 3c | `test_section_03c_sbgrb_delay_grid.py` | sbGRB delay times across the 20 models |
| 4 | `test_section_04_mssfr.py` | Cosmic-integration setup |
| 4d | `test_section_04d_tng_data_vs_fit.py` | TNG tables against the Levina fits |
| 4e | `test_section_04e_tng100_SZz.py` | TNG100-1 S(Z,z) map |
| 5 | `test_section_05_metallicity_efficiency.py` | eta(Z) per class |
| 6 | `test_section_06_bns_rate.py`, `test_section_06_lvk_rate_comparison.py` | BNS R(z) per class, against LVK |
| 6, 7 | `test_section_06_07_rate_quote_smoothing.py` | Quoted R(z=0) come from unsmoothed arrays |
| 6b | `test_section_06b_bns_class_channel.py`, `test_section_06b_formation_channels.py` | BNS R(z) per class x channel; channel classifier |
| 6c | `test_section_06c_sbgrb_vs_swift.py` | sbGRB rate against Swift, Du, P&S, LVK |
| 6d | `test_section_06d_class_rate_grid.py` | BNS R(z) per class across 20 models |
| 7 | `test_section_07_bhns_spin.py`, `test_section_07_lvk_rate_comparison.py` | BHNS R(z) with spin sweep, against LVK |
| 7b | `test_section_07b_bhns_class_channel.py` | BHNS R(z) per class x channel |
| 7c | `test_section_07c_bhns_class_rate_grid.py` | BHNS R(z) per class across 20 models |
| 7d | `test_section_07d_rate_weighted_masses.py` | Merger-rate-weighted masses by class |
| 7e | `test_section_07e_mass_redshift_tng_sweep.py` | Rate-weighted masses across redshift and TNG box |
| 8 | `test_section_08_offsets.py` | Integrated host offsets |
| 8a | `test_section_08a_free_drift.py` | Free-drift check and its data provenance |
| 8b | `test_section_08b_host_routing.py` | Delay-time host routing |
| 8c | `test_section_08c_free_drift_grid.py` | Free drift across 20 models |
| 11 | `test_section_11_model_grid_scan.py` | 20-model grid scan |

`_model_cache.py` is a shared helper (per-model rate cache), not a test file.

### `anchors/` (literature audit)

Every cited number that the pipeline consumes (M_TOV, K_THRESH_DEFAULT,
EOS table, Kroupa slopes, Planck 2015, Foucart 2018 coefficients,
Wanderman-Piran 2015, beaming bands) lives here as a paper-anchored
assertion. If a literal in `grb_*.py` drifts without updating the
paper citation, this is the file that will tell you.

### `integration/` (manual dispatch in CI)

Tests that need `data/`, `compas_python_utils`, or notebook execution.
Runs locally via `make test`; in CI only on manual `workflow_dispatch`.

| File | What it checks |
|---|---|
| `test_broekgaarden21_channels_reproduced.py` | Channels match Broekgaarden+ 2021 system by system |
| `test_channel_rates_vs_broekgaarden21_w000.py` | Per-channel rates against the `w_000` anchors, 20 models |
| `test_compas_pin.py` | Imports the pinned `compas_python_utils` |
| `test_cosmic_integration_vs_compas.py` | Bin-for-bin check against COMPAS; `w_000` reproduction |
| `test_failed_ce_ejections.py` | Loaders drop failed CE ejections |
| `test_grb_io_realdata.py` | Audits the real HDF5 archives |
| `test_grb_main_notebook.py` | Executes `grb_main.ipynb` via `nbclient`; every figure regenerated |
| `test_model_grid_realdata.py` | `BROEKGAARDEN21_MODELS` against the files on disk |
| `test_notebook_output_vs_literature.py` | Notebook outputs against published bands |
| `test_offset_tail_provenance.py` | Origin of the wide offset tail |
| `test_rate_class_shape.py` | R(z) shape on real Model A data |
| `test_section_01_mass_plane_realdata.py` | Section 1 panels on real data |
| `test_sspc_cross_check.py` | R(z) against an independent SSPC convolution |

## Markers

Defined in [pyproject.toml](../pyproject.toml) under
`[tool.pytest.ini_options].markers`. Composable: `pytest -m "unit and not slow"`.

| Marker | Auto-applied by | Meaning |
|---|---|---|
| `unit` | folder | Module-level invariant |
| `anchors` | folder | Literature anchor |
| `integration` | folder | Heavy / data-bound / notebook |
| `section_<N>` | filename | Notebook-section coupling (one marker per file in `sections/`, registered in `pyproject.toml`) |
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
