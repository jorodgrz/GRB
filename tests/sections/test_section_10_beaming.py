"""Section 10 of grb_main.ipynb: Beaming Correction (Intrinsic vs Observable).

Smoke-level invariants on ``beamed_rate``, ``beamed_rate_mixed`` and the
project's per-class ``CLASS_THETA_J`` table.

The literature anchors for ``CLASS_THETA_J`` and Fong+ 2015 beaming
factors live in
``tests/anchors/test_literature_anchors.py``; this section file pins
the round-trip relations the figure relies on.

Reference: Fong et al. (2015), ApJ 815, 102; Beniamini and Nakar
(2019), MNRAS 482, 5430; Gottlieb et al. (2023), arXiv:2309.00038.
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_rates import CLASS_THETA_J, beamed_rate, beamed_rate_mixed


def test_beamed_rate_zero_opening_angle_is_zero():
    assert beamed_rate(100.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_beamed_rate_round_trip_for_sbgrb_fiducial():
    R_int = 100.0
    theta = CLASS_THETA_J["sbGRB"]["fid"]
    R_obs = beamed_rate(R_int, theta)
    f_beam = 1.0 - np.cos(np.radians(theta))
    assert R_obs == pytest.approx(R_int * f_beam, rel=1e-12)


def test_beamed_rate_mixed_matches_per_class_sum():
    rates = {"sbGRB": 50.0, "lbGRB": 25.0}
    thetas = {
        "sbGRB": CLASS_THETA_J["sbGRB"]["fid"],
        "lbGRB": CLASS_THETA_J["lbGRB"]["fid"],
    }
    expected = sum(beamed_rate(rates[c], thetas[c]) for c in rates)
    got = beamed_rate_mixed(rates, thetas)
    assert got == pytest.approx(expected, rel=1e-12)


def test_class_theta_j_lo_fid_hi_ordered():
    """Bands are ordered ``lo <= fid <= hi`` for both classes."""
    for cls_name in ("sbGRB", "lbGRB"):
        band = CLASS_THETA_J[cls_name]
        assert band["lo"] <= band["fid"] <= band["hi"], (cls_name, band)
