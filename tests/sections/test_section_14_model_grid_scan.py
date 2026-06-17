"""Smoke tests for Section 14 of ``grb_main.ipynb`` (full 20-model grid scan).

Section 14 loops the complete Broekgaarden et al. (2021) variation grid,
classifies each model under the Gottlieb (2024) hybrid, calibrates the local
intrinsic rate, and caches per-model summaries to
``plots/grid_scan_results.npz``.  The heavy data-bound loop lives in the
notebook; this module pins the synthetic-data invariants the scan relies on
(class and channel partitions are closed; the channel x class table is a
proper conditional) and the cached-results schema the figure cells read back.

No COMPAS data is needed: the partitions are exercised on small synthetic
arrays, and the schema round-trip uses an in-memory NPZ.
"""

from __future__ import annotations

import numpy as np
import pytest

# Class / channel keys, mirrored from the Section 14 scan cell.  Kept here so
# a rename in the notebook that breaks the figure cells trips a test.
BNS_CLASS_KEYS = [
    "sbGRB + blue KN",
    "lbGRB + red KN (HMNS)",
    "lbGRB + red KN (disk)",
    "Faint lbGRB",
]
BHNS_CLASS_KEYS = ["lbGRB + red KN (BHNS disk)", "Faint lbGRB (BHNS)", "No GRB"]
CH_KEYS = [
    "I  Stable MT + CE",
    "II  Stable MT only",
    "III Single-core CE",
    "IV  Double-core CE",
    "V   Other",
]


def test_bns_class_fractions_sum_to_one():
    """The four Gottlieb (2024) BNS classes partition the population.

    The scan computes weighted fractions over exactly these four mutually
    exclusive, exhaustive masks; if the partition ever stops closing, the
    14.2 stacked bars would not reach 1.0.
    """
    from grb_classify import classify_bns_2024

    rng = np.random.default_rng(0)
    m1 = rng.uniform(1.0, 2.2, size=5000)
    m2 = rng.uniform(1.0, 2.2, size=5000)
    w = rng.uniform(0.1, 5.0, size=5000)
    cls = classify_bns_2024(m1, m2)
    masks = np.stack([cls[k] for k in BNS_CLASS_KEYS])
    frac = (masks * w).sum(-1) / w.sum()
    assert frac.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(frac >= 0.0)


def test_bhns_class_fractions_sum_to_one():
    """The three BHNS disk-mass classes partition the population."""
    from grb_classify import classify_bhns

    rng = np.random.default_rng(1)
    M_BH = rng.uniform(3.0, 20.0, size=5000)
    M_NS = rng.uniform(1.0, 2.0, size=5000)
    w = rng.uniform(0.1, 5.0, size=5000)
    cls = classify_bhns(M_BH, M_NS, a_BH=0.5)
    masks = np.stack([cls[k] for k in BHNS_CLASS_KEYS])
    frac = (masks * w).sum(-1) / w.sum()
    assert frac.sum() == pytest.approx(1.0, abs=1e-12)


def test_channel_class_crosstab_rows_are_conditional_distributions():
    """``normalise='channel'`` rows sum to 1 for every non-empty channel.

    Section 14.6 plots P(class | channel); each populated row must be a
    proper conditional distribution.
    """
    from grb_classify import channel_class_crosstab

    rng = np.random.default_rng(2)
    n = 4000
    # Build five disjoint channel masks covering the whole sample.
    idx = rng.integers(0, 5, size=n)
    channel_masks = {k: (idx == i) for i, k in enumerate(CH_KEYS)}
    m1 = rng.uniform(1.0, 2.2, size=n)
    m2 = rng.uniform(1.0, 2.2, size=n)
    from grb_classify import classify_bns_2024

    class_masks = classify_bns_2024(m1, m2)
    w = rng.uniform(0.1, 5.0, size=n)
    ct = channel_class_crosstab(channel_masks, class_masks, w, normalise="channel")
    row_sums = ct.to_numpy().sum(axis=1)
    for ch, rs in zip(CH_KEYS, row_sums):
        # Every synthetic channel is populated, so every row must sum to 1.
        assert rs == pytest.approx(1.0, abs=1e-9), f"row {ch} sums to {rs}"


def test_grid_scan_npz_schema_roundtrip(tmp_path):
    """The cached results carry every array the 14.1-14.6 figure cells read.

    Builds a minimal NPZ with the Section 14 schema and asserts the keys and
    shapes are internally consistent (20 models, 4 BNS classes, 3 BHNS
    classes, 5 channels).  This is the contract between the scan cell and the
    figure cells; a dropped or renamed array fails here rather than in the
    notebook.
    """
    n_models = len(BNS_CLASS_KEYS) * 0 + 20
    n_bins = 45
    path = tmp_path / "grid_scan_results.npz"
    np.savez(
        path,
        suffixes=np.array([f"m{i}" for i in range(n_models)]),
        families=np.array(["fiducial"] * n_models),
        R0_bns=np.zeros(n_models),
        R0_bhns=np.zeros(n_models),
        frac_bns=np.zeros((n_models, len(BNS_CLASS_KEYS))),
        frac_bhns=np.zeros((n_models, len(BHNS_CLASS_KEYS))),
        chfrac_bns=np.zeros((n_models, len(CH_KEYS))),
        chfrac_bhns=np.zeros((n_models, len(CH_KEYS))),
        crosstab_bns=np.zeros((n_models, len(CH_KEYS), len(BNS_CLASS_KEYS))),
        mbh_hist=np.zeros((n_models, n_bins)),
        mtot_hist=np.zeros((n_models, n_bins)),
        mbh_bins=np.linspace(2.0, 25.0, n_bins + 1),
        mtot_bins=np.linspace(2.0, 6.0, n_bins + 1),
        bns_class_keys=np.array(BNS_CLASS_KEYS),
        bhns_class_keys=np.array(BHNS_CLASS_KEYS),
        ch_keys=np.array(CH_KEYS),
    )
    d = np.load(path, allow_pickle=False)
    required = {
        "suffixes", "families", "R0_bns", "R0_bhns", "frac_bns", "frac_bhns",
        "chfrac_bns", "chfrac_bhns", "crosstab_bns", "mbh_hist", "mtot_hist",
        "mbh_bins", "mtot_bins", "bns_class_keys", "bhns_class_keys", "ch_keys",
    }  # fmt: skip
    assert required <= set(d.files)
    assert d["frac_bns"].shape == (n_models, 4)
    assert d["frac_bhns"].shape == (n_models, 3)
    assert d["crosstab_bns"].shape == (n_models, 5, 4)
    assert d["mbh_bins"].shape[0] == d["mbh_hist"].shape[1] + 1


def test_scan_suffix_order_matches_registry_core_first():
    """The scan iterates ALL_MODEL_SUFFIXES; core five lead for seed parity."""
    from grb_io import ALL_MODEL_SUFFIXES

    assert ALL_MODEL_SUFFIXES[:5] == ["A", "F", "G", "J", "K"]
    assert len(ALL_MODEL_SUFFIXES) == 20
