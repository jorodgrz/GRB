"""Section 8 of grb_main.ipynb: BHNS Merger Rate R(z) with BH Spin Sensitivity.

Smoke-level invariants on ``marginalize_bh_spin`` and on the project's
fiducial BH-spin choice (``A_BH_FID = 0.5``).

The full integration-plus-marginalisation suite lives in
``tests/unit/test_phase4_helpers.py::test_marginalize_bh_spin_*``;
this section file is the thin synthetic wrapper for the figure.

Reference: Fragos et al. (2010), ApJL 719, L79; Kawaguchi et al.
(2015), ApJ 807, 95; Foucart et al. (2018), arXiv:1807.00011
(calibrated for chi in [-0.5, 0.9]).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from grb_classify import classify_bhns
from grb_rates import marginalize_bh_spin

A_BH_FID = inspect.signature(classify_bhns).parameters["a_BH"].default
"""Fiducial BH spin pulled from ``classify_bhns(a_BH=0.5)`` rather than
inlined as a literal, so this section test stays in lockstep with the
default the figure-generation code uses."""


def test_marginalize_bh_spin_dict_mode_sums_to_input_when_uniform():
    rate = {0.0: 1.0, 0.3: 1.0, 0.5: 1.0, 0.7: 1.0, 0.9: 1.0}
    p_chi = {0.0: 0.2, 0.3: 0.2, 0.5: 0.2, 0.7: 0.2, 0.9: 0.2}
    out = marginalize_bh_spin(rate, p_chi)
    assert out == pytest.approx(1.0, rel=1e-12), out


def test_marginalize_bh_spin_array_mode_with_array_weights():
    spin_grid = np.array([0.0, 0.3, 0.5, 0.7, 0.9])
    rate = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    weights = np.full_like(spin_grid, 1.0 / spin_grid.size)
    out = marginalize_bh_spin(rate, weights, spin_grid=spin_grid)
    assert out == pytest.approx(rate.mean(), rel=1e-12), out


def test_a_bh_fid_inside_foucart_calibration_window():
    """A_BH_FID = 0.5 sits inside Foucart 2018's calibrated chi range
    [-0.5, 0.9]; the project's reference sweep stays within this band."""
    assert -0.5 <= A_BH_FID <= 0.9, A_BH_FID
