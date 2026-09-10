"""The composite Verlet integrator must advance a system for its own delay time.

``_advance_planar_composite`` buckets systems by ``2**ceil(log2(n_steps))`` and
runs that many steps, so the step has to divide the requested time by the bucket
count.  Sizing it by the unrounded step count instead marched every system whose
count was not a power of two past its merger time, by up to a factor of two; the
(10 kpc, 80 km/s, 1500 Myr) case below landed at 10.1 kpc instead of 6.5.  The
reference is scipy on the same equations of motion, so this is a black-box check
on the offset any Section 8 curve is built from.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reference_radius(host, r0, vr0, vt0, t):
    """High-accuracy 2-D Cartesian orbit integration of the same potential."""
    from scipy.integrate import solve_ivp

    from grb_offsets import total_acceleration

    def rhs(_t, y):
        x, yy, vx, vy = y
        rr = np.hypot(x, yy)
        a_rad = total_acceleration(rr, host)
        return [vx, vy, a_rad * x / rr, a_rad * yy / rr]

    sol = solve_ivp(rhs, (0.0, t), [r0, 0.0, vr0, vt0], rtol=1e-10, atol=1.0, method="DOP853")
    return float(np.hypot(sol.y[0, -1], sol.y[1, -1]))


def _host():
    from grb_offsets import KPC_CM, MSUN_G, make_host_potential

    return make_host_potential(10**10.5 * MSUN_G, 5.0 * KPC_CM, z=None, with_halo=True)


@pytest.mark.parametrize(
    ("r0_kpc", "v_kms", "t_myr"),
    [(10.0, 80.0, 1500.0), (5.0, 120.0, 500.0), (3.0, 60.0, 137.0), (2.0, 200.0, 300.0)],
)
def test_composite_step_advances_for_the_requested_time(r0_kpc, v_kms, t_myr):
    import grb_offsets as go
    from grb_offsets import KM_CM, KPC_CM, MYR_S

    host = _host()
    r0, t = r0_kpc * KPC_CM, t_myr * MYR_S
    vr0, vt0 = 0.6 * v_kms * KM_CM, 0.8 * v_kms * KM_CM

    got = go._advance_planar_composite(
        np.array([r0]),
        np.array([vr0]),
        np.array([vt0]),
        np.array([t]),
        host,
        np.random.default_rng(0),
        phase_cap=False,
    )[0]
    ref = _reference_radius(host, r0, vr0, vt0, t)
    assert abs(got - ref) / ref < 0.02, f"{got / KPC_CM:.3f} vs {ref / KPC_CM:.3f} kpc"


def test_bucket_rounding_does_not_bias_a_population_outward():
    """Across a random population the integrator neither over- nor under-shoots.

    The duration bug pushed every non-power-of-two system outward, so it showed
    up as a one-sided median error rather than as scatter.
    """
    import grb_offsets as go
    from grb_offsets import KM_CM, KPC_CM, MYR_S

    host = _host()
    rng = np.random.default_rng(0)
    err = []
    for _ in range(25):
        r0 = rng.uniform(1.0, 20.0) * KPC_CM
        v = rng.uniform(20.0, 300.0) * KM_CM
        t = 10 ** rng.uniform(1.3, 3.0) * MYR_S
        f = rng.uniform(0.1, 0.9)
        vr0, vt0 = f * v, np.sqrt(1.0 - f * f) * v
        got = go._advance_planar_composite(
            np.array([r0]),
            np.array([vr0]),
            np.array([vt0]),
            np.array([t]),
            host,
            rng,
            phase_cap=False,
        )[0]
        err.append(got / _reference_radius(host, r0, vr0, vt0, t) - 1.0)
    err = np.array(err)
    assert abs(np.median(err)) < 0.01, f"median relative error {np.median(err):+.4f}"
