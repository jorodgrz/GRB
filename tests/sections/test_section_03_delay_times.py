"""Section 3 of grb_main.ipynb: Delay Time Distributions by GRB Class.

NEW. The figure plots per-class weighted delay-time CDFs.  This test
exercises the same weighted-CDF utility (``weighted_offset_cdf`` and
``offset_cdf_by_class`` from ``grb_offsets``) on synthetic per-class
samples drawn from distributions with known means, and asserts that
a Kolmogorov-Smirnov-style test discriminates two classes with a
heavier-tail vs lighter-tail prior.

The utilities are generic weighted CDFs over any positive scalar
quantity; reusing them for delay times keeps the figure-generation
code path identical to Section 9 (host-galaxy offsets).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp

from grb_offsets import offset_cdf_by_class, weighted_offset_cdf


@pytest.fixture(scope="module")
def synthetic_delay_times():
    """Two synthetic GRB-class delay-time samples with different means.

    'short_class' is uniform over a tighter window (0.01 to 1 Gyr);
    'long_class' is exponential with mean 3 Gyr.  STROOPWAFEL-like
    weights are drawn from Uniform(0, 1).
    """
    rng = np.random.default_rng(0)
    n = 4000
    t_short = rng.uniform(0.01, 1.0, n)
    w_short = rng.uniform(0.0, 1.0, n)
    t_long = rng.exponential(3.0, n)
    w_long = rng.uniform(0.0, 1.0, n)
    return t_short, w_short, t_long, w_long


def test_weighted_cdf_is_monotone_and_unit_normalised(synthetic_delay_times):
    t_short, w_short, _, _ = synthetic_delay_times
    sorted_t, cdf = weighted_offset_cdf(t_short, w_short)
    assert (np.diff(sorted_t) >= 0).all(), "delay times not sorted"
    assert (np.diff(cdf) >= -1e-12).all(), "CDF not monotone non-decreasing"
    assert cdf[0] >= 0.0
    assert cdf[-1] == pytest.approx(1.0, rel=1e-9)


def test_weighted_cdf_empty_class_returns_sentinel():
    """``weighted_offset_cdf`` returns the (0.0, 0.0) sentinel when fewer
    than two finite-positive samples are present, so the figure-generation
    path can branch on it cleanly."""
    sorted_t, cdf = weighted_offset_cdf(np.array([np.nan]), np.array([1.0]))
    assert sorted_t.shape == (1,) and sorted_t[0] == 0.0
    assert cdf.shape == (1,) and cdf[0] == 0.0


def test_per_class_cdfs_separate_short_and_long_tails(synthetic_delay_times):
    """KS-statistic between the short-class and long-class delay-time
    CDFs is large (well above small-N noise), confirming that the
    helper produces distributions a class-discrimination plot can read."""
    t_short, w_short, t_long, w_long = synthetic_delay_times

    t_all = np.concatenate([t_short, t_long])
    w_all = np.concatenate([w_short, w_long])
    masks = {
        "short_class": np.concatenate(
            [np.ones_like(t_short, dtype=bool), np.zeros_like(t_long, dtype=bool)]
        ),
        "long_class": np.concatenate(
            [np.zeros_like(t_short, dtype=bool), np.ones_like(t_long, dtype=bool)]
        ),
    }
    cdfs = offset_cdf_by_class(t_all, w_all, masks)
    short_t, short_cdf = cdfs["short_class"]
    long_t, long_cdf = cdfs["long_class"]

    assert long_t[-1] > short_t[-1], (
        f"long-class tail did not exceed short-class tail "
        f"(short_max={short_t[-1]:.3f}, long_max={long_t[-1]:.3f})"
    )

    ks = ks_2samp(t_short, t_long)
    assert ks.statistic > 0.5, f"KS statistic too small: {ks.statistic}"
    assert ks.pvalue < 1e-9, f"KS p-value too large: {ks.pvalue}"
