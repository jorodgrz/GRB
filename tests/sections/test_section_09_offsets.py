"""Section 9 of grb_main.ipynb: Physical Host-Galaxy Offset Distributions.

Smoke-level invariants on the Hernquist (1990) potential helpers and
the per-class weighted offset CDF that the figure plots.

End-to-end real-data offset KS tests live in
``tests/integration/test_notebook_output_vs_literature.py::test_offset_cdf_ks_pvalue_lbgrb_above_threshold``;
this section file is the thin synthetic wrapper.

Reference: Hernquist (1990), ApJ 356, 359; Fong and Berger (2013),
ApJ 776, 18; Fong et al. (2015), ApJ 815, 102.
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_offsets import (
    KPC_CM,
    MSUN_G,
    escape_velocity,
    hernquist_potential,
    hernquist_scale_radius,
    offset_cdf_by_class,
)


def test_hernquist_scale_radius_positive():
    a = hernquist_scale_radius(R_e=4.0)
    assert a > 0.0


def test_hernquist_potential_is_negative_and_rises_with_radius():
    M_gal = 1e10 * MSUN_G
    a = 5.0 * KPC_CM
    r = np.array([0.5, 2.0, 10.0]) * a
    phi = hernquist_potential(r, M_gal, a)
    assert (phi < 0.0).all(), phi
    assert phi[0] < phi[-1], "potential should rise toward zero with r"


def test_escape_velocity_decreases_with_radius():
    M_gal = 1e10 * MSUN_G
    a = 5.0 * KPC_CM
    r = np.array([0.5, 2.0, 10.0, 50.0]) * a
    vesc = escape_velocity(r, M_gal, a)
    assert (np.diff(vesc) < 0).all(), vesc


def test_offset_cdf_by_class_returns_one_entry_per_class():
    rng = np.random.default_rng(0)
    n = 1000
    offsets = rng.uniform(0.1, 30.0, n)
    weights = rng.uniform(0.0, 1.0, n)
    masks = {
        "A": rng.random(n) < 0.5,
        "B": rng.random(n) < 0.5,
    }
    cdfs = offset_cdf_by_class(offsets, weights, masks)
    assert set(cdfs.keys()) == {"A", "B"}
    for label, (sorted_o, cdf) in cdfs.items():
        assert sorted_o.shape == cdf.shape
        if cdf.size > 1:
            assert cdf[-1] == pytest.approx(1.0, rel=1e-9), (label, cdf[-1])
