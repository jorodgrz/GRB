"""Section 11 of grb_main.ipynb: EOS, Channels, Offset CDFs, Beaming Comparator.

Smoke-level invariants on ``compute_eos_sensitivity``: returned class
fractions sum to 1 per EOS row, ``M_thresh`` tracks ``M_TOV`` linearly,
and the EOS list comes from ``grb_physics.EOS_MODELS`` by default.

The literature anchors for ``M_TOV``, ``R_1p4`` and ``M_crit`` per EOS
live in ``tests/anchors/test_literature_anchors.py``; this section
file pins the helper contract.

Reference: Bauswein et al. (2013), arXiv:1302.6530; Read et al.
(2009), arXiv:0812.2163; Raaijmakers et al. (2021), arXiv:2105.06981.
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_physics import EOS_MODELS, K_THRESH_DEFAULT
from grb_rates import compute_eos_sensitivity


@pytest.fixture(scope="module")
def synthetic_bns_sample():
    rng = np.random.default_rng(2026)
    n = 5000
    m1 = rng.uniform(1.1, 2.4, n)
    m2 = rng.uniform(1.1, 2.4, n)
    weights = rng.uniform(0.0, 1.0, n)
    return m1, m2, weights


def test_eos_sensitivity_class_fractions_sum_to_one(synthetic_bns_sample):
    m1, m2, w = synthetic_bns_sample
    df = compute_eos_sensitivity(m1, m2, w)
    class_cols = [
        c
        for c in df.columns
        if c not in ("M_TOV", "R_1p4", "M_thresh", "hmns_split", "total_weight")
    ]
    sums = df[class_cols].sum(axis=1).values
    np.testing.assert_allclose(sums, 1.0, rtol=1e-9, atol=1e-12)


def test_eos_sensitivity_M_thresh_tracks_M_TOV(synthetic_bns_sample):
    m1, m2, w = synthetic_bns_sample
    df = compute_eos_sensitivity(m1, m2, w)
    np.testing.assert_allclose(
        df["M_thresh"].values,
        K_THRESH_DEFAULT * df["M_TOV"].values,
        rtol=1e-12,
    )


def test_eos_sensitivity_default_covers_all_eos_models(synthetic_bns_sample):
    m1, m2, w = synthetic_bns_sample
    df = compute_eos_sensitivity(m1, m2, w)
    assert set(df.index) == set(EOS_MODELS), df.index.tolist()
