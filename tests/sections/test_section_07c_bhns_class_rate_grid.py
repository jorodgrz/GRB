"""Section 7c: BHNS R(z) per GRB class across the 20-model grid.

Same cache contract as Section 6d, with two things 6d does not have: the
cache is stamped with the BH spin its classes were cut at, and the stored
rates are intrinsic while two of the four panels are drawn at the
misalignment factor.  Both are pinned here, because the failure mode is a
figure that renders correctly with the factor applied twice, or not at all,
or with classes cut at the wrong spin.

Source-level checks on ``grb_main.ipynb`` plus shape and content checks on
the cached npz; no COMPAS files needed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from grb_io import LOADER_SELECTION

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_NB_PATH = os.path.join(_REPO_ROOT, "grb_main.ipynb")
_CACHE = os.path.join(_REPO_ROOT, "plots", "rate_bhns_by_class_per_model.npz")

CLASS_KEYS = ["lbGRB + red KN (BHNS disk)", "Faint lbGRB (BHNS)", "No GRB"]
# The two classes that launch a jet, and so the only two the misalignment
# factor applies to.  "No GRB" swallows the NS.
MISALIGNED = {"lbGRB + red KN (BHNS disk)", "Faint lbGRB (BHNS)"}


def _cell_7c():
    with open(_NB_PATH) as fh:
        nb = json.load(fh)
    cells = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "_CLASS_RATE_CACHE_7c" in "".join(c["source"])
    ]
    assert len(cells) == 1, f"expected exactly one Section 7c code cell, found {len(cells)}"
    return cells[0]


@pytest.mark.unit
def test_section_7c_cache_guard_includes_the_bh_spin():
    """A BHNS class cut at a different a_BH is a different partition.

    The 6d guards (normalisation stamp, class keys, redshift grid) are not
    enough here: ``classify_bhns`` takes a spin, so a cache built at
    a_BH = 0.9 would pass all three and render as if it were the fiducial.
    """
    src = _cell_7c()
    assert "_c7c.get('a_bh'" in src and "== A_BH_FID" in src, (
        "Section 7c must reject a cache built at a different BH spin"
    )
    for guard in ("_CACHE_NORM_7c", "_c7c['class_keys']", "np.allclose(_c7c['redshifts']"):
        assert guard in src, f"Section 7c cache guard missing: {guard}"


@pytest.mark.unit
def test_section_7c_stores_intrinsic_and_applies_misalignment_at_plot_time():
    """The factor is applied once, on the display copy, to two classes only.

    Baking it into the cache would break the class sum against Section 7's
    intrinsic All-BHNS rate, and applying it to "No GRB" would suppress a
    population that launches no jet at all.
    """
    src = _cell_7c()
    assert "_MISALIGNED_7c = {'lbGRB + red KN (BHNS disk)', 'Faint lbGRB (BHNS)'}" in src
    assert "'No GRB'" not in src.split("_MISALIGNED_7c = ")[1].split("}")[0]
    # The factor multiplies the display copy inside the plotting loop, and
    # never the array that is cached.
    assert "_fac * smooth_rate_curve(_R[_i])" in src
    assert "MISALIGNMENT_SYSTEMATIC_FACTOR" not in src.split("np.savez(")[1].split(")")[0]


@pytest.mark.unit
def test_section_7c_cross_checks_model_a_against_section_7():
    """Three independent comparisons against Section 7, not one.

    The intrinsic sum pins the normalisation, and the two per-class checks
    pin the direction the 0.5x factor is applied in: an inverted factor
    would still satisfy the sum.
    """
    src = _cell_7c()
    assert "rate_bhns_all" in src
    assert "spin_rates_long[A_BH_FID]" in src and "spin_rates_short[A_BH_FID]" in src
    assert src.count("assert np.allclose") >= 2


@pytest.mark.unit
def test_section_7c_handles_an_empty_class():
    """Models E and L produce no long-lbGRB BHNS, so the spread and the peak
    redshift must be taken over populated models only.

    Taking argmin over the raw column would report a spread against zero.
    """
    src = _cell_7c()
    assert "np.where(_pos, _v, np.inf)" in src, (
        "Section 7c must exclude empty models from the minimum"
    )
    assert "_R[_pos].argmax(axis=1)" in src, (
        "peak redshift must be taken over populated models only"
    )


@pytest.mark.unit
@pytest.mark.skipif(not os.path.exists(_CACHE), reason="Section 7c cache not built")
def test_section_7c_cache_shape_and_content():
    """20 models x 3 classes x the Section 4 redshift grid, finite and
    non-negative, stamped with the normalisation and the fiducial spin."""
    from grb_physics import A_BH_FID

    d = np.load(_CACHE, allow_pickle=False)
    assert str(d["norm"]) == f"M_SF_SIM|TNG100-1 S(Z,z) data|{LOADER_SELECTION}"
    assert [str(k) for k in d["class_keys"]] == CLASS_KEYS
    assert float(d["a_bh"]) == A_BH_FID
    R = d["R_class"]
    assert R.shape == (20, 3, d["redshifts"].size)
    assert np.all(np.isfinite(R)) and np.all(R >= 0)
    dropped = d["dropped"]
    assert np.all(np.isfinite(dropped)) and np.all((dropped >= 0) & (dropped < 1))
    assert dropped.max() > 0


@pytest.mark.unit
@pytest.mark.skipif(not os.path.exists(_CACHE), reason="Section 7c cache not built")
def test_section_7c_records_the_foucart_extrapolation_fraction():
    """Every BHNS disk mass above Q = 7 is outside the Foucart (2018)
    calibration, so the weight fraction is carried per model and quoted.

    It is not a constant of the analysis: it spans a factor of six across the
    grid, so a single Model A number would misstate most of the models.  The
    low-mass-ratio edge is checked too, because Q < 1 would be a second and
    undocumented extrapolation.
    """
    d = np.load(_CACHE, allow_pickle=False)
    q, qmax = d["q_gt7"], d["q_max"]
    assert q.shape == (20,) and qmax.shape == (20,)
    assert np.all((q >= 0.0) & (q <= 1.0))
    assert q.max() > 0.3, "expected at least one model with >30 percent of weight above Q = 7"
    # Foucart's upper calibration edge is 7; every model exceeds it somewhere.
    assert np.all(qmax > 7.0)
    assert qmax.max() < 40.0, "max mass ratio outside the range the captions quote"


@pytest.mark.unit
@pytest.mark.skipif(not os.path.exists(_CACHE), reason="Section 7c cache not built")
def test_section_7c_empty_class_rows_are_exactly_zero():
    """A model with no system in a class stores a zero curve, not noise.

    The figure relies on this: an all-zero row disappears off the bottom of
    the log axis, which is the correct rendering for "this variation makes
    none of these".  A tiny non-zero row would draw a spurious curve.
    """
    d = np.load(_CACHE, allow_pickle=False)
    empty = d["n_raw"] == 0
    assert empty.any(), "expected at least one empty BHNS class in the grid"
    assert np.all(d["R_class"][empty] == 0.0)
    # And a populated class is never identically zero.
    assert np.all(d["R_class"][~empty].max(axis=-1) > 0)
