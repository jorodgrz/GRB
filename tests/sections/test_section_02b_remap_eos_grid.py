"""Section 2b: BNS class-fraction sensitivity to the NS mass function and EOS (grb_main.ipynb).

Section 2b crosses two NS-mass treatments (raw COMPAS fiducial, Alsing+
2018 remap) with the four ``EOS_MODELS``, each column running
``compute_eos_sensitivity(..., require_cold_stable=True)``.  These
smoke tests pin the grid shape and the invariants the figure relies on:
per (treatment, EOS) the four class fractions are renormalised over the
cold-stable systems and sum to 1, and the discarded weight fraction is a
valid probability that vanishes when every component sits below the
softest EOS's M_TOV.
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_physics import EOS_MODELS, K_THRESH_DEFAULT, remap_ns_masses_double_gaussian
from grb_rates import compute_eos_sensitivity

_BNS_CLASS_KEYS = [
    "sbGRB + blue KN",
    "lbGRB + red KN (HMNS)",
    "lbGRB + red KN (disk)",
    "Faint lbGRB",
]


@pytest.fixture(scope="module")
def treatments():
    """Raw and Alsing-remapped views of one synthetic merging BNS sample."""
    rng = np.random.default_rng(2026)
    n = 4000
    m1_raw = rng.uniform(1.1, 2.5, n)
    m2_raw = rng.uniform(1.1, 2.5, n)
    m1_raw, m2_raw = np.maximum(m1_raw, m2_raw), np.minimum(m1_raw, m2_raw)
    w = rng.uniform(0.5, 1.5, n)
    m1_al, m2_al = remap_ns_masses_double_gaussian(
        m1_raw.copy(), m2_raw.copy(), weights=w, rng=np.random.default_rng(42)
    )
    return {
        "Raw (COMPAS)": (m1_raw, m2_raw),
        "Alsing+ 2018": (m1_al, m2_al),
    }, w


def _sweep(treatments):
    grids, w = treatments
    return {
        name: compute_eos_sensitivity(
            m1,
            m2,
            w,
            eos_models=EOS_MODELS,
            k_thresh=K_THRESH_DEFAULT,
            require_cold_stable=True,
        )
        for name, (m1, m2) in grids.items()
    }


def test_grid_shape_two_treatments_four_eos(treatments):
    sweep = _sweep(treatments)
    assert set(sweep) == {"Raw (COMPAS)", "Alsing+ 2018"}
    for df in sweep.values():
        assert set(df.index) == set(EOS_MODELS)
        assert df.shape[0] == len(EOS_MODELS)


def test_each_cell_class_fractions_sum_to_one(treatments):
    sweep = _sweep(treatments)
    for name, df in sweep.items():
        for eos in df.index:
            s = float(df.loc[eos, _BNS_CLASS_KEYS].sum())
            assert s == pytest.approx(1.0, rel=1e-9), f"{name}/{eos} sums to {s}"


def test_dropped_weight_frac_is_a_valid_probability(treatments):
    sweep = _sweep(treatments)
    for df in sweep.values():
        d = df["dropped_weight_frac"].to_numpy()
        assert np.all(d >= 0.0) and np.all(d <= 1.0)


def test_dropped_weight_frac_zero_when_all_below_softest_m_tov():
    """A sample capped below min(M_TOV) loses no weight under any EOS."""
    m_tov_min = min(e["M_TOV"] for e in EOS_MODELS.values())
    rng = np.random.default_rng(7)
    n = 500
    m1 = rng.uniform(1.1, m_tov_min - 0.05, n)
    m2 = rng.uniform(1.1, m_tov_min - 0.05, n)
    w = rng.uniform(0.5, 1.5, n)
    df = compute_eos_sensitivity(m1, m2, w, require_cold_stable=True)
    np.testing.assert_allclose(df["dropped_weight_frac"], 0.0, atol=1e-12)
