"""Real-data audit for ``grb_io.py`` across the Broekgaarden+ 2021 grid.

Pins the COMPAS HDF5 contract against the actual Zenodo files in
``Data/``:

1. No NaN in ``w[mask]`` for every available model file.
2. ``_match_sn_to_dco`` returns NaN for at most 1 percent of the
   merging subset (or logs the count if the threshold trips).  Per
   physics variation; can be relaxed via ``_UNMATCHED_TOLERANCE`` if
   a kick or unstable-CE variant legitimately has more bookkeeping
   gaps in the supernovae group.
3. ``np.unique(Z[mask])`` is a subset of ``np.unique(METALLICITY_GRID)``
   to within float tolerance.
4. ``f['doubleCompactObjects'].keys()`` contains every column the
   loaders depend on.
5. ``np.all(out['m1'] >= out['m2'])`` after ``load_bns(...)`` /
   ``out['M_BH'] >= out['M_NS']`` after ``load_bhns(...)``.

Each test is parametrised over every entry in
``tools/embed_model_metadata.KNOWN_FILES`` (40 files: 20 physics
variations x 2 populations); per-file skip via the ``compas_file``
fixture means missing files are no-ops, so a partial download (Tier-1
only, or in-progress staged download) exercises only the variations
that have landed.

Run once per Zenodo download:

    pytest tests/integration/test_grb_io_realdata.py -v --tb=line
"""

from __future__ import annotations

import sys

import h5py as h5
import numpy as np
import pytest

# ``embed_model_metadata`` is imported via the sys.path entry that
# ``tests/conftest.py`` adds for ``tools/``; this matches the pattern
# in ``tools/download_compas_data.py``.
from embed_model_metadata import KNOWN_FILES  # type: ignore[import-not-found]

# ─────────────────────────────────────────────────────────────────────
# Parametrisation source-of-truth
# ─────────────────────────────────────────────────────────────────────
# Sort for deterministic test ordering.  Each entry is a project filename
# like ``COMPASCompactOutput_BNS_A.h5``; the ``compas_file`` indirect
# fixture in ``conftest.py`` per-file skips when the data is absent.
_AUDIT_FILES = sorted(KNOWN_FILES.keys())

# Population kind lookup keyed by filename, used to dispatch BNS vs
# BHNS loaders inside the mass-ordering test.
_KIND_BY_FILE = {name: KNOWN_FILES[name]["kind"] for name in _AUDIT_FILES}

# Per-model unmatched-seed tolerance for ``_match_sn_to_dco``.  Most
# variations satisfy the 1 percent default; certain physics knobs
# (kick variants, unstable-CE variants) may legitimately have a
# different supernovae-group geometry.  Add entries here only after
# confirming the unmatched count is physics-driven, not a join bug.
_UNMATCHED_TOLERANCE = {
    # filename -> max unmatched fraction (default 0.01 if absent)
}


# ─────────────────────────────────────────────────────────────────────
# 1. Weight integrity
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _AUDIT_FILES, indirect=True)
def test_no_nan_weights_after_mask(compas_file):
    """STROOPWAFEL ``weight`` must be NaN-free on every merging row.

    Downstream rates use ``np.average(x, weights=w)`` and
    ``np.histogram(weights=w)``, both of which silently propagate NaN
    to the science output.
    """
    with h5.File(compas_file, "r") as f:
        dco = f["doubleCompactObjects"]
        w = dco["weight"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()

    w_masked = w[mh == 1]
    n_nan = int(np.isnan(w_masked).sum())
    assert n_nan == 0, (
        f"{compas_file}: {n_nan} of {len(w_masked)} merging rows have "
        f"NaN STROOPWAFEL weight; downstream rates would be poisoned."
    )


# ─────────────────────────────────────────────────────────────────────
# 2. SN-to-DCO match completeness
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _AUDIT_FILES, indirect=True)
def test_match_sn_to_dco_no_nan_on_merging_subset(compas_file, capsys):
    """Less than 1 percent of merging systems may have unmatched seeds.

    A small unmatched fraction is tolerable (Hernquist offsets drop NaN
    rows via ``np.isfinite`` in ``compute_offsets_population``), but a
    > tolerance fraction means a paper-killer in the SN join rather
    than a benign edge-case.  Per-model overrides for legitimate
    physics-driven gaps live in ``_UNMATCHED_TOLERANCE`` at the top of
    this file.
    """
    import os as _os

    from grb_io import load_bhns_with_kicks, load_bns_with_kicks

    name = _os.path.basename(compas_file)
    kind = _KIND_BY_FILE[name]
    loader = load_bns_with_kicks if kind == "BNS" else load_bhns_with_kicks
    out = loader(path=compas_file)
    v_sys = out["v_sys"]
    n = out["n_merging"]
    n_nan = int(np.isnan(v_sys).sum())
    frac = n_nan / max(n, 1)
    tol = _UNMATCHED_TOLERANCE.get(name, 0.01)

    print(
        f"_match_sn_to_dco {name}: {n_nan} of {n} merging rows have "
        f"NaN v_sys ({100 * frac:.3f} percent, tol {100 * tol:.1f} percent)",
        file=sys.stderr,
    )
    assert frac < tol, (
        f"{name}: _match_sn_to_dco unmatched fraction "
        f"{100 * frac:.2f} percent exceeds the {100 * tol:.1f} percent "
        f"threshold; the lexsort/searchsorted join is suspect, or this "
        f"variation needs a per-model entry in _UNMATCHED_TOLERANCE."
    )


# ─────────────────────────────────────────────────────────────────────
# 3. METALLICITY_GRID provenance vs actual data
# ─────────────────────────────────────────────────────────────────────
def _grid_contains(values, grid, atol=1e-8):
    """True iff every entry of ``values`` matches a ``grid`` entry within ``atol``."""
    return all(np.any(np.isclose(grid, v, atol=atol)) for v in values)


@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _AUDIT_FILES, indirect=True)
def test_metallicity_grid_matches_data(compas_file):
    """Birth metallicities in the HDF5 must be a subset of ``METALLICITY_GRID``.

    Catches Zenodo schema drift in the metallicity discretisation.
    The literal grid in ``grb_io.py`` was already corrected once based
    on the Model A audit; this test fans the same check across the
    full 20-variation grid.
    """
    from grb_io import METALLICITY_GRID

    with h5.File(compas_file, "r") as f:
        dco = f["doubleCompactObjects"]
        Z = dco["Metallicity1"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()

    Z_unique = np.unique(Z[mh == 1])
    grid_unique = np.unique(METALLICITY_GRID)
    missing = [v for v in Z_unique if not np.any(np.isclose(grid_unique, v, atol=1e-8))]
    assert _grid_contains(Z_unique, grid_unique), (
        f"{compas_file}: data carries metallicities not present in "
        f"METALLICITY_GRID; first 5 missing values = {missing[:5]}.  "
        f"Either the literal grid in grb_io.py drifted or this "
        f"variation uses a different metallicity prior."
    )


# ─────────────────────────────────────────────────────────────────────
# 4. HDF5 schema audit
# ─────────────────────────────────────────────────────────────────────
_REQUIRED_DCO_COLS = {
    "M1",
    "M2",
    "weight",
    "Metallicity1",
    "mergesInHubbleTimeFlag",
    "tc",
    "tform",
    "stellarType1",
}


@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _AUDIT_FILES, indirect=True)
def test_hdf5_schema_pinned(compas_file):
    """``doubleCompactObjects`` must contain the columns every loader uses.

    Catches Zenodo schema drift on the next download.
    """
    with h5.File(compas_file, "r") as f:
        keys = set(f["doubleCompactObjects"].keys())

    missing = _REQUIRED_DCO_COLS - keys
    assert not missing, (
        f"{compas_file}: doubleCompactObjects schema drift detected. "
        f"Missing columns: {sorted(missing)}.  Review the Zenodo "
        f"changelog (5189849 / 5178777) and update the loaders."
    )


# ─────────────────────────────────────────────────────────────────────
# 5. m1 >= m2 invariant after default load
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _AUDIT_FILES, indirect=True)
def test_mass_ordering_invariant(compas_file):
    """Sanity-check ``sort_masses=True`` (BNS) / BH-NS split (BHNS).

    Fanned across the full Broekgaarden+ 2021 model grid so the
    default ordering is verified file-by-file rather than only on the
    fiducial Model A.
    """
    import os as _os

    from grb_io import load_bhns, load_bns

    name = _os.path.basename(compas_file)
    kind = _KIND_BY_FILE[name]

    if kind == "BNS":
        out = load_bns(path=compas_file)
        assert (out["m1"] >= out["m2"]).all(), (
            f"{name}: load_bns default did not enforce m1 >= m2; "
            f"downstream classifiers in grb_classify.py rely on this "
            f"invariant."
        )
        assert out["population"] == "BNS"
    else:
        out = load_bhns(path=compas_file)
        assert (out["M_BH"] >= out["M_NS"]).all(), (
            f"{name}: load_bhns produced M_BH < M_NS rows; the "
            f"stellarType1 == 14 routing is suspect on this variation."
        )
        assert out["population"] == "BHNS"
