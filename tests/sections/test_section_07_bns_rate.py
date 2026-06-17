"""Section 7 of grb_main.ipynb: BNS Merger Rate R(z) per GRB Class.

Smoke-level invariant on the per-class partition that the figure
relies on: the four ``classify_bns_2024`` boolean masks are mutually
exclusive and collectively exhaustive over the BNS sample.

The end-to-end rate-vs-z partition is exercised in
``tests/unit/test_rates.py::test_per_class_rates_partition_total_bns_2024``;
this section file is the synthetic, no-dependency smoke wrapper.

Reference: Gottlieb et al. (2024), arXiv:2411.13657 (four-class scheme).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import classify_bns_2024


def test_classify_bns_2024_masks_partition_sample():
    rng = np.random.default_rng(11)
    n = 5000
    m1 = rng.uniform(1.1, 3.5, n)
    m2 = rng.uniform(1.1, 3.5, n)
    masks = classify_bns_2024(m1, m2)

    assert set(masks.keys()) == {
        "sbGRB + blue KN",
        "lbGRB + red KN (HMNS)",
        "lbGRB + red KN (disk)",
        "Faint lbGRB",
    }

    stack = np.stack([np.asarray(v, dtype=bool) for v in masks.values()])
    overlap = stack.sum(axis=0)
    assert (overlap == 1).all(), (
        f"masks overlap or leave gaps: distribution of count = {np.bincount(overlap)}"
    )


def test_classify_bns_2024_weighted_fractions_sum_to_one():
    rng = np.random.default_rng(12)
    n = 5000
    m1 = rng.uniform(1.1, 3.5, n)
    m2 = rng.uniform(1.1, 3.5, n)
    weights = rng.uniform(0.0, 1.0, n)
    masks = classify_bns_2024(m1, m2)

    fractions = np.array([weights[m].sum() for m in masks.values()]) / weights.sum()
    assert fractions.sum() == pytest.approx(1.0, rel=1e-12), fractions
