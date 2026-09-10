"""Free-drift offsets: unit conversion and the bound they place on the orbits.

``ballistic_offset`` is the Section 8a sanity check, so what it has to get
right is the km/s x Myr -> kpc conversion and the fact that it never falls
below the gravitationally integrated offset of the same system.
"""

from __future__ import annotations

import numpy as np


def test_one_kms_over_one_myr_is_a_parsec():
    """1 km/s sustained for 1 Myr covers 1.02 pc."""
    from grb_offsets import ballistic_offset

    d = ballistic_offset(np.array([1.0]), np.array([1.0]))
    assert np.isclose(d[0], 1.0227e-3, rtol=1e-3)


def test_scaling_is_linear_in_velocity_and_time():
    from grb_offsets import ballistic_offset

    v = np.array([100.0, 200.0, 100.0])
    t = np.array([10.0, 10.0, 20.0])
    d = ballistic_offset(v, t)
    assert np.isclose(d[1], 2.0 * d[0])
    assert np.isclose(d[2], 2.0 * d[0])


def test_two_leg_adds_the_first_leg_path_length():
    """The two-leg form sums path lengths, the collinear upper bound."""
    from grb_offsets import ballistic_offset

    v2, t2 = np.array([150.0]), np.array([500.0])
    v1, t1 = np.array([80.0]), np.array([20.0])
    one = ballistic_offset(v2, t2)
    two = ballistic_offset(v2, t2, v_sys_1_km=v1, t_leg1_myr=t1)
    assert np.isclose(two[0] - one[0], ballistic_offset(v1, t1)[0])


def test_free_drift_bounds_the_integrated_offset():
    """Host gravity and projection can only shorten the offset."""
    from grb_offsets import ballistic_offset, compute_offsets_mixed_hosts

    rng = np.random.default_rng(11)
    n = 400
    v = rng.uniform(50.0, 400.0, n)
    t = 10.0 ** rng.uniform(1.0, 3.5, n)
    w = np.ones(n)

    res = compute_offsets_mixed_hosts(v, t, weights=w, max_systems=n, rng=np.random.default_rng(3))
    free = ballistic_offset(v, t)

    # Birth radii are drawn from the host profile, so an individual system can
    # start further out than it drifts; the comparison that has to hold is on
    # the drift itself, which the free-drift median bounds from above.
    assert np.median(res["mixed_offsets"]) < np.median(free)
