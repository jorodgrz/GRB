"""Tabulated checks of grb_physics.r_isco against Bardeen et al. (1972) Eq. 2.21.

The ISCO sits at the base of the BHNS Foucart (2018) disk-mass formula,
so a 4 to 8 percent error here would propagate into every spin sweep.

Bardeen, Press and Teukolsky (1972) ApJ 178, 347, Eq. 2.21:

    Z1 = 1 + (1 - a^2)^(1/3) * [(1 + a)^(1/3) + (1 - a)^(1/3)]
    Z2 = sqrt(3 a^2 + Z1^2)
    r_isco / M = 3 + Z2 -+ sqrt((3 - Z1)(3 + Z1 + 2 Z2))

with the upper sign for prograde (a >= 0) and lower for retrograde (a < 0).
Closed-form anchors: r_isco(0) = 6, r_isco(+1) = 1, r_isco(-1) = 9.

Note on extremal Kerr.  grb_physics.r_isco clips inputs to +/- (1 - 1e-9)
to keep the Z1 cube root real.  The clamped values approach the analytic
limits to within numerical precision (within 0.5 percent for r_isco(+/- 1)).
The tests below either drive the function with the clamped inputs or
assert against the analytic limit at the clamping tolerance.
"""

import warnings

import numpy as np
import pytest

from grb_physics import r_isco


# Bardeen Eq. 2.21 closed-form values: (chi, r_isco_analytic).
# Values for non-extremal spins computed from the closed form below
# rather than transcribed from a table, so the test self-documents.
def _bardeen_closed_form(a):
    """Eq. 2.21 evaluated at |a| < 1 (no clipping)."""
    Z1 = 1.0 + (1.0 - a**2) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    Z2 = np.sqrt(3.0 * a**2 + Z1**2)
    sign = 1.0 if a >= 0 else -1.0
    return 3.0 + Z2 - sign * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))


def test_r_isco_against_bardeen_anchor_values():
    """Three textbook anchors: r_isco(0) = 6, r_isco(+1) = 1, r_isco(-1) = 9."""
    assert r_isco(0.0) == pytest.approx(6.0, rel=5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # |a|=1 inputs trigger the clip warning
        assert r_isco(+1.0) == pytest.approx(1.0, rel=5e-3)
        assert r_isco(-1.0) == pytest.approx(9.0, rel=5e-3)


@pytest.mark.parametrize("chi", [-0.9, -0.5, -0.3, 0.0, 0.3, 0.5, 0.9])
def test_r_isco_matches_bardeen_closed_form(chi):
    """At |chi| < 1, r_isco must match Eq. 2.21 to better than 0.5 percent."""
    expected = _bardeen_closed_form(chi)
    assert r_isco(chi) == pytest.approx(expected, rel=5e-3)


def test_r_isco_monotone_across_spin():
    """ISCO must decrease monotonically from chi = -1 to chi = +1.

    Catches a sign-handling bug at chi = 0 (the boundary in r_isco's
    np.where) that would fold the prograde and retrograde branches.
    """
    chi_grid = np.linspace(-0.95, 0.95, 39)
    r = r_isco(chi_grid)
    diffs = np.diff(r)
    assert (diffs <= 0).all(), (
        "r_isco(chi) is not monotonically decreasing across chi = 0; "
        f"max positive step = {diffs.max():.3e} at chi = "
        f"{chi_grid[np.argmax(diffs)]:.3f}"
    )


def test_r_isco_clipping_warns_above_one():
    """|a| >= 1 inputs are unphysical; r_isco emits a warning and clips."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = r_isco(np.array([0.9, 1.0, 1.5]))
    assert any("clipping" in str(item.message) for item in w), (
        "r_isco failed to warn about |a| >= 1 inputs"
    )
    # Clipped output approaches r_isco(+1) = 1 to within ~0.2 percent
    # (the clamp at |a| = 1 - 1e-9 is amplified by Bardeen's nonlinear
    # Z1, Z2 dependence near the extremal limit; absolute residual ~ 2e-3).
    assert out[1] == pytest.approx(1.0, abs=5e-3)
    assert out[2] == pytest.approx(1.0, abs=5e-3)


def test_r_isco_vector_input_matches_scalar_loop():
    """Vectorised call must agree with elementwise scalar evaluation."""
    chi_grid = np.array([-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9])
    vec = r_isco(chi_grid)
    loop = np.array([r_isco(c) for c in chi_grid])
    np.testing.assert_allclose(vec, loop, rtol=1e-12)
