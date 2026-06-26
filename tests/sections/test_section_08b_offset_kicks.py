"""Smoke test for Section 8b of ``grb_main.ipynb`` (offset kick variations).

Section 8b compares projected host-galaxy offset CDFs across the
supernova-kick variations (fiducial Hobbs+ 2005, reduced ccSN dispersions
M / N, and the no-BH-kick run O).  The physical content is that a smaller
systemic velocity keeps binaries closer to their birth site, so the offset
distribution shifts to smaller radii.

This test pins that monotonicity through ``compute_offsets_mixed_hosts`` on
synthetic populations, so it needs neither COMPAS data nor
``compas_python_utils``.  The orbit integration is kept cheap (small sample,
small ``max_systems``).
"""

from __future__ import annotations

import numpy as np


def _weighted_median(offsets, weights):
    ok = np.isfinite(offsets) & (offsets > 0) & np.isfinite(weights)
    o, w = offsets[ok], weights[ok]
    order = np.argsort(o)
    o, w = o[order], w[order]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(0.5, c, o))


def test_lower_kick_velocity_gives_smaller_offsets():
    """A population with smaller v_sys has a smaller median projected offset.

    Two synthetic populations with identical delay times and weights but a
    factor-of-several difference in systemic-velocity scale: the low-velocity
    population must produce the smaller weighted-median offset through the
    Hernquist orbit integration, the trend Section 8b reads off the M / N
    kick variations.
    """
    from grb_offsets import compute_offsets_mixed_hosts

    rng = np.random.default_rng(7)
    n = 400
    t_delay = rng.uniform(50.0, 2000.0, size=n)
    weights = np.ones(n)

    v_low = rng.uniform(10.0, 60.0, size=n)
    v_high = rng.uniform(150.0, 400.0, size=n)

    res_low = compute_offsets_mixed_hosts(
        v_low,
        t_delay,
        weights=weights,
        max_systems=400,
        rng=np.random.default_rng(11),
    )
    res_high = compute_offsets_mixed_hosts(
        v_high,
        t_delay,
        weights=weights,
        max_systems=400,
        rng=np.random.default_rng(11),
    )

    med_low = _weighted_median(res_low["mixed_offsets"], res_low["mixed_weights"])
    med_high = _weighted_median(res_high["mixed_offsets"], res_high["mixed_weights"])
    assert med_low < med_high, (
        f"low-kick median offset {med_low:.3f} kpc is not below the "
        f"high-kick median {med_high:.3f} kpc; the Section 8b kick trend "
        f"is inverted."
    )


def test_observed_offset_overlays_are_available_and_positive():
    """Section 8b overlays the published SGRB and lGRB+KN offset samples."""
    from grb_offsets import OBSERVED_LGRB_KN_OFFSETS_KPC, OBSERVED_SGRB_OFFSETS_KPC

    assert OBSERVED_SGRB_OFFSETS_KPC.size > 0
    assert np.all(OBSERVED_SGRB_OFFSETS_KPC > 0)
    assert OBSERVED_LGRB_KN_OFFSETS_KPC.size > 0
    assert np.all(OBSERVED_LGRB_KN_OFFSETS_KPC > 0)
