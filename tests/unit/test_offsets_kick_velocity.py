"""Kick velocity sets the offset scale in ``compute_offsets_mixed_hosts``.

A smaller systemic velocity keeps a binary closer to its birth site, so the
projected offset distribution shifts to smaller radii.  This pins that
monotonicity on synthetic populations, so it needs neither COMPAS data nor
``compas_python_utils``; the orbit integration is kept cheap (small sample,
small ``max_systems``).  It also checks that the observed offset overlays
the Section 8 figure draws are present and positive.

The kick prescriptions this bounds are the fiducial Hobbs+ 2005 dispersion
and the reduced-ccSN and no-BH-kick variations of the Broekgaarden+ 2021
grid.
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
    Hernquist orbit integration, the trend the reduced-kick M / N
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
        f"high-kick median {med_high:.3f} kpc; the kick-velocity trend "
        f"is inverted."
    )


def test_observed_offset_overlays_are_available_and_positive():
    """The published sGRB and lGRB-KN offset overlays are loadable and positive."""
    from grb_offsets import OBSERVED_LGRB_KN_OFFSETS_KPC, OBSERVED_SGRB_OFFSETS_KPC

    assert OBSERVED_SGRB_OFFSETS_KPC.size > 0
    assert np.all(OBSERVED_SGRB_OFFSETS_KPC > 0)
    assert OBSERVED_LGRB_KN_OFFSETS_KPC.size > 0
    assert np.all(OBSERVED_LGRB_KN_OFFSETS_KPC > 0)
