"""Section 10 of grb_main.ipynb: BNS Jet Breakout Efficiency.

Smoke-level invariants on the breakout chain the figure draws:
``breakout_fraction_bns_eos`` returns one band per Gottlieb (2024) class,
and the per-class rate ladder ``R_intrinsic -> apply_bns_jet_breakout ->
beamed_rate`` is monotonically non-increasing.

The Pais et al. (2025) constant anchors live in
``tests/anchors/test_literature_anchors.py`` and the function-level edge
cases in ``tests/unit/test_rates.py``; this file pins the section's
composition on a small synthetic population (no COMPAS data required).

Reference: Pais, Piran, Kiuchi and Shibata (2025), arXiv:2407.19002;
Gottlieb and Nakar (2022), arXiv:2106.03860.
"""

from __future__ import annotations

import numpy as np

from grb_rates import (
    CLASS_THETA_J,
    apply_bns_jet_breakout,
    beamed_rate,
    breakout_fraction_bns_eos,
)

# Canonical Gottlieb-2024 labels, matching classify_bns_2024 / the notebook.
CANON_CLASSES = (
    "sbGRB + blue KN",
    "lbGRB + red KN (HMNS)",
    "lbGRB + red KN (disk)",
    "Faint lbGRB",
)
THETA_KEY = {
    "sbGRB + blue KN": "sbGRB",
    "lbGRB + red KN (HMNS)": "lbGRB",
    "lbGRB + red KN (disk)": "lbGRB",
    "Faint lbGRB": "lbGRB",
}


def _toy_population(n=400, seed=21):
    """Small m1 >= m2 BNS sample partitioned across the four classes."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.15, 1.65, n)
    b = rng.uniform(1.10, 1.55, n)
    m1, m2 = np.maximum(a, b), np.minimum(a, b)
    w = rng.uniform(0.1, 1.0, n)
    q = np.quantile(m1, [0.25, 0.5, 0.75])
    masks = {
        "sbGRB + blue KN": m1 < q[0],
        "lbGRB + red KN (HMNS)": (m1 >= q[0]) & (m1 < q[1]),
        "lbGRB + red KN (disk)": (m1 >= q[1]) & (m1 < q[2]),
        "Faint lbGRB": m1 >= q[2],
    }
    return m1, m2, w, masks


def test_breakout_fraction_eos_covers_all_four_classes():
    m1, m2, w, masks = _toy_population()
    out = breakout_fraction_bns_eos(masks, m1, m2, w, n_draws=48, rng=np.random.default_rng(1))
    assert set(out) == set(CANON_CLASSES)
    for label, band in out.items():
        assert band["lo"] <= band["med"] <= band["hi"], label
        assert 0.0 <= band["lo"] <= 1.0 and 0.0 <= band["hi"] <= 1.0, label


def test_rate_ladder_is_monotonically_non_increasing():
    """R_obs_pred <= R_break <= R_intrinsic per class (the figure's ladder)."""
    m1, m2, w, masks = _toy_population()
    f_break = breakout_fraction_bns_eos(masks, m1, m2, w, n_draws=48, rng=np.random.default_rng(2))

    R_intrinsic = {
        "sbGRB + blue KN": 22.0,
        "lbGRB + red KN (HMNS)": 20.0,
        "lbGRB + red KN (disk)": 1.0,
        "Faint lbGRB": 12.0,
    }
    for label in CANON_CLASSES:
        R_int = R_intrinsic[label]
        R_break = float(apply_bns_jet_breakout(R_int, f_break[label]["med"]))
        theta = CLASS_THETA_J[THETA_KEY[label]]["fid"]
        R_beam = float(beamed_rate(R_break, theta))
        assert 0.0 <= R_break <= R_int + 1e-9, label
        assert 0.0 <= R_beam <= R_break + 1e-9, label


def test_hmns_classes_not_more_efficient_than_prompt_disk():
    """The two HMNS classes do not exceed the prompt-collapse disk class.

    Same underlying mass distribution across classes (quartile partition is
    statistically uniform in the other ejecta inputs), so the long HMNS launch
    delay plus the disk-wind obstruction can only lower their breakout.
    """
    m1, m2, w, masks = _toy_population()
    out = breakout_fraction_bns_eos(masks, m1, m2, w, n_draws=96, rng=np.random.default_rng(8))
    disk = out["lbGRB + red KN (disk)"]["med"]
    for hmns_label in ("sbGRB + blue KN", "lbGRB + red KN (HMNS)"):
        assert out[hmns_label]["med"] <= disk + 1e-9, hmns_label
