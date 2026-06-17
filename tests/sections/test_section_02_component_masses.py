"""Section 2 of grb_main.ipynb: Component Mass Distributions by GRB Class.

Smoke-level invariants on the Alsing, Silva and Berti (2018) double-Gaussian
NS-mass remap that produces the per-class component-mass histograms.  The
remap closes the Fryer 2012 baryonic-to-gravitational artifact near 1.7 Msun
(Eq. 12-13; present in both delayed and rapid engines, per Broekgaarden+
2021 footnote 3) and preserves the ``m1 >= m2`` invariant.

Reference: Alsing, Silva and Berti (2018), MNRAS 478, 1377 (Table 3 fit);
Mandel and Muller (2020), MNRAS 499, 3214 (gap motivation).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_physics import (
    M_TOV,
    NS_REMAP_M_MIN,
    NS_REMAP_W1,
    NS_REMAP_W2,
    remap_ns_masses_double_gaussian,
)
from grb_plot_style import weighted_hist_pdf


@pytest.fixture(scope="module")
def synthetic_pre_remap_ns_masses():
    """Two-population sample with an artificial deficit near 1.7 Msun.

    Mimics the Fryer 2012 Eq. 12-13 baryonic-to-gravitational mass
    artifact on a small N sample so the test is fast in CI; real-data
    behaviour is verified end-to-end in ``tests/anchors/test_literature_anchors.py``.
    """
    rng = np.random.default_rng(123)
    n = 5000
    a = rng.normal(1.30, 0.06, n // 2)
    b = rng.normal(1.95, 0.10, n // 2)
    pool = np.concatenate([a, b])
    pool = pool[(pool > NS_REMAP_M_MIN) & (pool < M_TOV)]
    rng.shuffle(pool)
    half = pool.size // 2
    m1 = pool[:half]
    m2 = pool[half : 2 * half]
    m_heavy = np.maximum(m1, m2)
    m_light = np.minimum(m1, m2)
    return m_heavy, m_light


def test_remap_preserves_m1_geq_m2(synthetic_pre_remap_ns_masses):
    m1, m2 = synthetic_pre_remap_ns_masses
    m1_new, m2_new = remap_ns_masses_double_gaussian(m1, m2)
    assert (m1_new >= m2_new).all(), "m1 >= m2 invariant broken by remap"


def test_remap_stays_within_truncation_window(synthetic_pre_remap_ns_masses):
    m1, m2 = synthetic_pre_remap_ns_masses
    m1_new, m2_new = remap_ns_masses_double_gaussian(m1, m2)
    for arr in (m1_new, m2_new):
        assert (arr >= NS_REMAP_M_MIN - 1e-9).all()
        assert (arr <= M_TOV + 1e-9).all()


def test_remap_closes_artificial_gap_near_1p7(synthetic_pre_remap_ns_masses):
    """Pre-remap pool has a deficit in [1.65, 1.80]; post-remap fills it
    with non-zero density (the whole point of the Alsing remap)."""
    m1, m2 = synthetic_pre_remap_ns_masses
    pool_pre = np.concatenate([m1, m2])
    pool_pre_in_gap = ((pool_pre >= 1.65) & (pool_pre <= 1.80)).sum()

    m1_new, m2_new = remap_ns_masses_double_gaussian(m1, m2)
    pool_post = np.concatenate([m1_new, m2_new])
    pool_post_in_gap = ((pool_post >= 1.65) & (pool_post <= 1.80)).sum()

    assert pool_post_in_gap > pool_pre_in_gap, (
        f"remap did not close the Fryer 2012 gap; pre={pool_pre_in_gap}, post={pool_post_in_gap}"
    )


def test_alsing_double_gauss_weights_sum_to_one():
    assert NS_REMAP_W1 + NS_REMAP_W2 == pytest.approx(1.0, rel=1e-6), (NS_REMAP_W1, NS_REMAP_W2)


# The block below pins the abundance-weighted normalization contract that
# the Section 2 component-mass histogram cell of grb_main.ipynb relies on:
# for mutually exclusive class masks sharing the same population total,
# the per-class curve areas equal the class weight fractions and jointly
# integrate to one (modulo out-of-range tails).  Catches both a silent
# change to the helper and a future bin-range regression that drops area.


def test_weighted_hist_pdf_abundance_normalization_sums_to_one():
    rng = np.random.default_rng(0)
    n = 2000
    data = rng.uniform(0.0, 10.0, n)
    mask_a = data < 4.0
    mask_b = ~mask_a
    w = rng.uniform(0.5, 2.0, n)
    w_total = w[mask_a | mask_b].sum()

    bins = np.linspace(0.0, 10.0, 41)
    binw = np.diff(bins)

    res_a = weighted_hist_pdf(data, mask_a, w, bins, w_total)
    res_b = weighted_hist_pdf(data, mask_b, w, bins, w_total)
    assert res_a is not None and res_b is not None

    area_a = (res_a[1] * binw).sum()
    area_b = (res_b[1] * binw).sum()

    assert area_a + area_b == pytest.approx(1.0, abs=1e-12)
    assert area_a == pytest.approx(w[mask_a].sum() / w_total, abs=1e-12)
    assert area_b == pytest.approx(w[mask_b].sum() / w_total, abs=1e-12)


def test_weighted_hist_pdf_clips_silently_when_bins_miss_support():
    """Documents that data outside [bins[0], bins[-1]] is dropped from the
    numerator but not from ``w_total``; the returned curve therefore
    integrates to *less than* 1.  Callers (e.g. the Section 2 BNS-q panel)
    are responsible for choosing bins that cover the full class support.
    """
    rng = np.random.default_rng(1)
    data = rng.uniform(0.0, 10.0, 2000)
    mask = np.ones_like(data, dtype=bool)
    w = np.ones_like(data)
    w_total = w[mask].sum()

    bins = np.linspace(0.0, 5.0, 41)
    res = weighted_hist_pdf(data, mask, w, bins, w_total)
    assert res is not None

    area = (res[1] * np.diff(bins)).sum()
    in_range_fraction = w[(data >= bins[0]) & (data <= bins[-1])].sum() / w_total
    assert area < 1.0
    assert area == pytest.approx(in_range_fraction, abs=1e-12)


def test_weighted_hist_pdf_empty_mask_returns_none():
    data = np.linspace(0.0, 1.0, 10)
    mask = np.zeros_like(data, dtype=bool)
    w = np.ones_like(data)
    bins = np.linspace(0.0, 1.0, 5)
    assert weighted_hist_pdf(data, mask, w, bins, w_total=1.0) is None
