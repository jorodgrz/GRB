"""Section 6d: BNS R(z) per GRB class across the 20-model grid.

The panel is driven by a cache (``plots/rate_bns_by_class_per_model.npz``)
rather than by a live scan, so the failure mode is a stale npz surviving a
change to the rate setup.  Three things guard against that and are pinned
here: the cache carries the ``M_SF_SIM`` normalisation stamp and the class
key list, the cell re-checks its own Model A row against the Section 6
integration, and the All-BNS panel is the class sum rather than a sixth
stored array.

Source-level checks on ``grb_main.ipynb`` plus a shape and content check on
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
_CACHE = os.path.join(_REPO_ROOT, "plots", "rate_bns_by_class_per_model.npz")

CLASS_KEYS = [
    "sbGRB + blue KN",
    "lbGRB + red KN (HMNS)",
    "lbGRB + red KN (disk)",
    "Faint lbGRB",
]


def _cell_6d():
    with open(_NB_PATH) as fh:
        nb = json.load(fh)
    cells = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "_CLASS_RATE_CACHE_6d" in "".join(c["source"])
    ]
    assert len(cells) == 1, f"expected exactly one Section 6d code cell, found {len(cells)}"
    return cells[0]


@pytest.mark.unit
def test_section_6d_cell_guards_its_cache():
    """The cache is accepted only on the normalisation stamp, the class key
    list and the Section 4 redshift grid.

    Dropping any one of the three lets a cache built under a different
    M_SF, a different class partition, or a different redshift_step render
    silently on the figure.
    """
    src = _cell_6d()
    for guard in ("_CACHE_NORM_6d", "_c6d['class_keys']", "np.allclose(_c6d['redshifts']"):
        assert guard in src, f"Section 6d cache guard missing: {guard}"


@pytest.mark.unit
def test_section_6d_checks_model_a_against_section_6():
    """Model A is integrated twice, once per section, and the cell asserts
    they agree.  That assert is the only thing tying the cached grid to the
    live rate setup, so it is not optional."""
    src = _cell_6d()
    assert "merger_rates_BNS[_k]" in src and "assert np.allclose" in src, (
        "Section 6d must assert its cached Model A curves against the Section 6 integration"
    )


@pytest.mark.unit
def test_section_6d_all_bns_is_the_class_sum():
    """The four classes partition the sample, so the total is their sum.

    Storing or integrating a fifth All-BNS array would let the panel drift
    from the four above it when a classifier boundary moves.
    """
    src = _cell_6d()
    assert "_R_cls_6d.sum(axis=1)" in src


@pytest.mark.unit
def test_section_6d_cell_smooths_for_display_only():
    """Stored arrays are unsmoothed; smooth_rate_curve is applied at plot
    time.  The sigma=30 kernel reflects at z = 0 and lifts the local rate by
    ~30 percent, which would corrupt the printed spread table."""
    src = _cell_6d()
    assert "smooth_sigma=0" in src
    assert "smooth_rate_curve(_R[_i])" in src


@pytest.mark.unit
@pytest.mark.skipif(not os.path.exists(_CACHE), reason="Section 6d cache not built")
def test_section_6d_cache_shape_and_content():
    """20 models x 4 classes x the Section 4 redshift grid, all finite and
    non-negative, with the normalisation stamp and class keys the cell
    checks for."""
    d = np.load(_CACHE, allow_pickle=False)
    assert str(d["norm"]) == f"M_SF_SIM|TNG100-1 S(Z,z) data|{LOADER_SELECTION}"
    assert [str(k) for k in d["class_keys"]] == CLASS_KEYS
    R = d["R_class"]
    assert R.shape == (20, 4, d["redshifts"].size)
    assert np.all(np.isfinite(R)) and np.all(R >= 0)
    # Every model populates every class, so no panel is missing a curve.
    assert np.all(d["n_raw"] > 0)
    # The cold-stability cut fraction is recorded per model, not left at zero.
    # It grows with the run's M_NS_max, so a grid of exact zeros means the
    # cache was written by a cell that stubbed the field out.
    dropped = d["dropped"]
    assert dropped.shape == (20,)
    assert np.all(np.isfinite(dropped)) and np.all((dropped >= 0) & (dropped < 1))
    assert dropped.max() > 0, "per-model cold-stability weight fractions are all zero"


@pytest.mark.unit
@pytest.mark.skipif(not os.path.exists(_CACHE), reason="Section 6d cache not built")
def test_section_6d_model_spread_exceeds_within_model_class_spread():
    """The claim the section is built to make: across the grid a single
    class varies by more than the four classes vary inside one model.

    If this ever inverts, the section's conclusion (that a class ordering
    read off Model A is not population-synthesis-robust) no longer holds and
    the markdown has to change with it.
    """
    d = np.load(_CACHE, allow_pickle=False)
    R = d["R_class"]
    iz0 = int(np.argmin(np.abs(d["redshifts"])))
    v = R[:, :, iz0]
    across = v.max(axis=0) / v.min(axis=0)  # per class, over models
    within = v.max(axis=1) / v.min(axis=1)  # per model, over classes
    assert across.min() > np.median(within), (
        f"narrowest class spread across models ({across.min():.0f}x) no longer "
        f"exceeds the median within-model class spread ({np.median(within):.0f}x)"
    )
