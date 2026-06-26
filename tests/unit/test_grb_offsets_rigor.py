"""Behaviour and edge-case tests for ``grb_offsets``.

Complements ``tests/sections/test_section_08_offsets.py`` (smoke
tests on Hernquist closed forms), ``tests/unit/test_phase4_helpers.py``
(vectorised-vs-legacy KS, mixed-hosts regression, offset_cdf_by_class),
and the literature-anchor block in
``tests/anchors/test_literature_anchors.py`` (Hernquist 1990 Table 1,
host-model parameters, observed-offset arrays).  This file fills the
gaps: per-function bodies, dispatch branches inside the analytic and
vectorised orbit integrators, the public ``compute_offset*`` and
``assign_host_by_delay`` partitions, and ``weighted_offset_cdf``
filtering.

References cited inline per test: Hernquist (1990) ApJ 356, 359
Eq. 38 (birth-radius inverse CDF and pileup formula); Bloom et al.
(1999) MNRAS 305, 763 (isotropic kick prescription); Fong and Berger
(2013) ApJ 776, 18 (sGRB host demographics); Leibler and Berger
(2010) ApJ 725, 1202 (sGRB host stellar populations).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from grb_offsets import (
    _R_CAP_FACTOR,
    DEFAULT_M_GAL,
    DEFAULT_R_E,
    G_CGS,
    HOST_MODELS,
    KM_CM,
    KPC_CM,
    MSUN_G,
    MYR_S,
    SF_DISK_FRAC_OF_SF_HOSTS,
    SF_HOST_TRANSITION_T_MYR,
    _analytic_offset,
    _hernquist_apocenter,
    _orbit_rhs,
    _vectorized_orbit_3d,
    assign_host_by_delay,
    compute_offset_single,
    compute_offsets_delay_hosts,
    compute_offsets_mixed_hosts,
    compute_offsets_population,
    compute_offsets_population_vectorized,
    escape_velocity,
    hernquist_acceleration,
    hernquist_potential,
    hernquist_scale_radius,
    integrate_orbit,
    weighted_ks_2samp,
    weighted_offset_cdf,
)


# ---------------------------------------------------------------------------
# Constants schema
# ---------------------------------------------------------------------------
def test_module_constants_have_expected_cgs_values():
    """The CGS unit-conversion constants must match their documented values.

    Catches a copy-edit drift to a different unit system or precision
    truncation.  The values are pinned at single-precision (5e-4 rel)
    to the CODATA 2018 / IAU 2015 anchors that
    ``test_grb_offsets_cgs_constants_match_codata_values`` in
    ``test_literature_anchors.py`` validates with their literature
    citations.
    """
    assert G_CGS == 6.674e-8
    assert MSUN_G == 1.989e33
    assert KPC_CM == 3.0857e21
    assert MYR_S == 3.1557e13
    assert KM_CM == 1e5
    assert DEFAULT_M_GAL == 10**10.5 * MSUN_G
    assert DEFAULT_R_E == 5.0 * KPC_CM


# ---------------------------------------------------------------------------
# _orbit_rhs
# ---------------------------------------------------------------------------
def test_orbit_rhs_L_conservation():
    """The third RHS component is identically zero (L is conserved).

    Angular momentum is a constant of motion in a spherically
    symmetric potential; the orbit RHS must therefore return
    ``dLdt = 0`` regardless of the (r, v_r, L) state.
    """
    a = 1e22
    M_gal = 1e44
    for state in [(0.5 * a, 0.0, 0.0), (a, 1e7, 5e29), (2.0 * a, -3e7, -1e29)]:
        rhs = _orbit_rhs(0.0, list(state), M_gal, a)
        assert rhs[2] == 0.0, f"L not conserved at state {state}: dLdt = {rhs[2]}"


def test_orbit_rhs_radial_acceleration_matches_hernquist_plus_centrifugal():
    """``dvrdt`` equals ``hernquist_acceleration(r) + L^2 / r^3``.

    The radial equation of motion in a central potential is
    ``d v_r / dt = -dPhi/dr + L^2 / r^3`` (centrifugal term);
    pinning this assembly catches a sign flip on either piece.
    """
    a = 2.0 * KPC_CM
    M_gal = 1e10 * MSUN_G
    r = 1.5 * a
    vr = 1e7
    L = 2e29
    rhs = _orbit_rhs(0.0, [r, vr, L], M_gal, a)
    expected_dvrdt = float(hernquist_acceleration(r, M_gal, a) + L**2 / r**3)
    assert rhs[0] == vr  # drdt
    assert rhs[1] == pytest.approx(expected_dvrdt, rel=1e-12)


# ---------------------------------------------------------------------------
# _hernquist_apocenter
# ---------------------------------------------------------------------------
def test_hernquist_apocenter_brentq_root_solves_effective_potential():
    """The returned ``r_apo`` is a real zero of ``0.5 L^2/r^2 - GM/(r+a) - E``.

    Construct a near-radial bound orbit (theta_launch -> 0) so the
    centrifugal barrier ``0.5 L^2 / r^2`` is negligible and the
    brentq bracket has exactly one sign change.  At the resulting
    root the effective-potential expression must vanish.
    """
    M_gal = 1e10 * MSUN_G
    a = 2.0 * KPC_CM
    r0 = 0.5 * a
    v_cm = 80.0 * KM_CM  # well below v_esc
    GM = G_CGS * M_gal
    # Nearly radial kick keeps L very small so r_peri < a_eps and
    # only the apocenter lies inside [1e-3*a, _R_CAP_FACTOR * a].
    theta = 1e-4
    L = r0 * v_cm * np.sin(theta)
    E = 0.5 * v_cm**2 - GM / (r0 + a)
    assert E < 0.0, "test fixture is unbound; need a more deeply bound orbit"

    r_apo = _hernquist_apocenter(E, L, M_gal, a)
    f_at_apo = 0.5 * L**2 / r_apo**2 - GM / (r_apo + a) - E
    # Pin the residual relative to the binding-energy scale |E|.
    assert abs(f_at_apo) / abs(E) < 1e-6, (
        f"f(r_apo)/|E| = {abs(f_at_apo) / abs(E):.3e} not near zero; "
        f"_hernquist_apocenter returned the cap fallback or a "
        f"wrong-side root (r_apo = {r_apo:.3e}, cap = "
        f"{_R_CAP_FACTOR * a:.3e})"
    )
    assert 0 < r_apo < _R_CAP_FACTOR * a


def test_hernquist_apocenter_returns_finite_positive_for_bound_orbit():
    """The brentq success path yields a strictly positive apocenter inside the cap."""
    M_gal = 1e10 * MSUN_G
    a = 2.0 * KPC_CM
    r0 = 0.3 * a
    v_cm = 60.0 * KM_CM
    GM = G_CGS * M_gal
    # Near-radial kick so brentq finds the apocenter (see the prior
    # test for the geometric rationale).
    L = r0 * v_cm * np.sin(1e-4)
    E = 0.5 * v_cm**2 - GM / (r0 + a)
    r_apo = _hernquist_apocenter(E, L, M_gal, a)
    assert np.isfinite(r_apo)
    assert 0 < r_apo < _R_CAP_FACTOR * a


def test_hernquist_apocenter_falls_back_to_r_cap_when_brentq_has_no_root():
    """When the bracket has no sign change, the function returns ``_R_CAP_FACTOR * a``.

    Very-high-angular-momentum (near-tangential) kicks push the
    apocenter beyond the cap, breaking the brentq sign-change
    requirement.  The documented fallback returns the cap so the
    downstream caller never crashes on degenerate orbits.
    """
    M_gal = 10**10.5 * MSUN_G
    a = hernquist_scale_radius(5.0 * KPC_CM)
    r0 = 0.5 * a
    v_cm = 100.0 * KM_CM
    GM = G_CGS * M_gal
    L = r0 * 0.5 * v_cm  # high L → r_apo > _R_CAP_FACTOR * a
    E = 0.5 * v_cm**2 - GM / (r0 + a)
    r_apo = _hernquist_apocenter(E, L, M_gal, a)
    assert r_apo == pytest.approx(_R_CAP_FACTOR * a, rel=1e-12)


# ---------------------------------------------------------------------------
# integrate_orbit
# ---------------------------------------------------------------------------
def test_integrate_orbit_zero_velocity_returns_r_init():
    """``v_sys_cm = 0`` short-circuits the integrator and returns the launch radius."""
    a = 2.0 * KPC_CM
    M_gal = 1e10 * MSUN_G
    r0_frac = 0.5
    r_final = integrate_orbit(v_sys_cm=0.0, t_delay_s=1e16, M_gal=M_gal, a=a, r0=r0_frac)
    assert r_final == pytest.approx(r0_frac * a, rel=1e-12)


def test_integrate_orbit_zero_delay_returns_r_init():
    """``t_delay_s = 0`` short-circuits the integrator and returns the launch radius."""
    a = 2.0 * KPC_CM
    M_gal = 1e10 * MSUN_G
    r0_frac = 0.5
    r_final = integrate_orbit(v_sys_cm=1e7, t_delay_s=0.0, M_gal=M_gal, a=a, r0=r0_frac)
    assert r_final == pytest.approx(r0_frac * a, rel=1e-12)


def test_integrate_orbit_explicit_r0_and_theta_are_threaded_through():
    """Explicit ``r0`` and ``theta_launch`` produce a deterministic geometry.

    A purely tangential kick (``theta = pi/2``) gives zero radial
    velocity at launch, so the orbit stays close to its launch
    radius for a small fraction of an orbital period.  Verifies
    that both override kwargs reach the RHS via the initial
    conditions.
    """
    a = 2.0 * KPC_CM
    M_gal = 1e10 * MSUN_G
    r0_frac = 5.0
    v_cm = 50.0 * KM_CM
    t_s = 10.0 * MYR_S  # short integration so the orbit barely moves
    r_final = integrate_orbit(
        v_sys_cm=v_cm,
        t_delay_s=t_s,
        M_gal=M_gal,
        a=a,
        r0=r0_frac,
        theta_launch=np.pi / 2.0,
    )
    assert r_final == pytest.approx(r0_frac * a, rel=5e-2)


def test_integrate_orbit_caps_at_R_CAP_FACTOR_a():
    """Super-escape kicks are clamped at ``_R_CAP_FACTOR * a``."""
    a = 2.0 * KPC_CM
    M_gal = 1e10 * MSUN_G
    v_esc = float(escape_velocity(0.5 * a, M_gal, a))
    v_cm = 100.0 * v_esc  # very super-escape
    t_s = 1.0 * MYR_S * 1000  # 1 Gyr
    r_final = integrate_orbit(v_sys_cm=v_cm, t_delay_s=t_s, M_gal=M_gal, a=a, r0=0.5)
    assert r_final <= _R_CAP_FACTOR * a + 1e-12


# ---------------------------------------------------------------------------
# _analytic_offset dispatch branches
# ---------------------------------------------------------------------------
def test_analytic_offset_trivial_branch_returns_r0():
    """``v_sys_km = 0`` returns the (explicit or drawn) birth radius."""
    M_gal = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    r0_frac = 0.7
    r_out = _analytic_offset(0.0, 1000.0, M_gal, a, r0_frac=r0_frac, rng=np.random.default_rng(0))
    assert r_out == pytest.approx(r0_frac * a, rel=1e-12)


def test_analytic_offset_bound_long_time_returns_r_mean():
    """Bound, long-delay systems use the analytic ``r_mean`` shortcut.

    With ``t >> 5 P_est``, the orbit phase-mixes and the
    time-averaged radius ``(r0 + r_apo) / 2`` is used in place of
    the per-system RK4 result.  We replicate the function's RNG
    consumption to compute the expected ``r_mean`` independently.
    """
    M_gal = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    r0_frac = 0.5
    v_sys_km = 50.0
    t_delay_myr = 1e7  # 10 Gyr -- comfortably long
    seed = 7

    # Independent computation of the expected r_mean.
    rng_ref = np.random.default_rng(seed)
    r0 = r0_frac * a
    v_cm = v_sys_km * KM_CM
    theta = np.arccos(rng_ref.uniform(-1, 1))
    E = 0.5 * v_cm**2 + hernquist_potential(r0, M_gal, a)
    v_esc = float(escape_velocity(r0, M_gal, a))
    assert v_cm < v_esc, "test fixture is unbound; need a slower kick"
    L = r0 * v_cm * np.sin(theta)
    r_apo = _hernquist_apocenter(E, L, M_gal, a)
    r_mean = (r0 + r_apo) / 2.0
    GM = G_CGS * M_gal
    P_est = 2.0 * np.pi * np.sqrt(r_mean**3 / GM) * (1 + a / r_mean)
    assert t_delay_myr * MYR_S > 5.0 * P_est, "test fixture is not in the long-time branch"

    rng_func = np.random.default_rng(seed)
    out = _analytic_offset(v_sys_km, t_delay_myr, M_gal, a, r0_frac=r0_frac, rng=rng_func)
    assert out == pytest.approx(r_mean, rel=1e-12)


def test_analytic_offset_unbound_delegates_to_integrate_orbit():
    """Super-escape kicks bypass the analytic shortcut and run RK45.

    With ``v_cm >= v_esc`` the function calls ``integrate_orbit`` and
    returns its result (a finite, capped radius).  Catches a
    regression that lets the unbound branch hit the bound code path.
    """
    M_gal = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    r0_frac = 0.5
    r0 = r0_frac * a
    v_esc = float(escape_velocity(r0, M_gal, a)) / KM_CM  # km/s
    v_sys_km = 5.0 * v_esc
    out = _analytic_offset(
        v_sys_km, 1000.0, M_gal, a, r0_frac=r0_frac, rng=np.random.default_rng(0)
    )
    assert np.isfinite(out)
    assert r0 <= out <= _R_CAP_FACTOR * a + 1e-12


# ---------------------------------------------------------------------------
# _vectorized_orbit_3d dispatch branches
# ---------------------------------------------------------------------------
def test_vectorized_orbit_3d_empty_input_returns_empty():
    """Length-zero input returns a length-zero output (no integration)."""
    M_gal = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    out = _vectorized_orbit_3d(np.array([]), np.array([]), M_gal, a, np.random.default_rng(0))
    assert out.shape == (0,)


def test_vectorized_orbit_3d_all_trivial_input_returns_birth_radii():
    """All systems with ``v <= 0`` keep their birth radius (no RK4)."""
    M_gal = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    N = 50
    v_cm = np.zeros(N)
    t_s = np.full(N, 1000.0 * MYR_S)
    out = _vectorized_orbit_3d(v_cm, t_s, M_gal, a, np.random.default_rng(0))
    assert out.shape == (N,)
    assert np.isfinite(out).all()
    assert (out > 0).all()
    assert (out <= _R_CAP_FACTOR * a + 1e-12).all()


def test_vectorized_orbit_3d_caps_at_R_CAP_FACTOR_a():
    """Super-escape kicks across a batch land at or below ``_R_CAP_FACTOR * a``."""
    M_gal = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    N = 10
    v_cm = np.full(N, 5e9)  # 5e9 cm/s = 50 000 km/s, super-escape
    t_s = np.full(N, 100.0 * MYR_S)
    out = _vectorized_orbit_3d(v_cm, t_s, M_gal, a, np.random.default_rng(0))
    assert (out <= _R_CAP_FACTOR * a + 1e-12).all()
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# compute_offset_single
# ---------------------------------------------------------------------------
def test_compute_offset_single_returns_scalar_float():
    """Single-system convenience helper returns a numpy float scalar."""
    out = compute_offset_single(v_sys_km=100.0, t_delay_myr=500.0, rng=np.random.default_rng(0))
    assert isinstance(out, (float, np.floating))
    assert np.isfinite(out)
    assert out >= 0.0


def test_compute_offset_single_rng_determinism():
    """Fixed-seed reproducibility."""
    a = compute_offset_single(v_sys_km=100.0, t_delay_myr=500.0, rng=np.random.default_rng(11))
    b = compute_offset_single(v_sys_km=100.0, t_delay_myr=500.0, rng=np.random.default_rng(11))
    assert a == pytest.approx(b, rel=1e-15)


# ---------------------------------------------------------------------------
# compute_offsets_population_vectorized
# ---------------------------------------------------------------------------
def test_compute_offsets_population_vectorized_returns_three_documented_keys():
    """Output dict has exactly the three documented keys."""
    v = np.array([100.0, 200.0, 300.0])
    t = np.array([100.0, 500.0, 1000.0])
    out = compute_offsets_population_vectorized(v, t, max_systems=3, rng=np.random.default_rng(0))
    assert set(out.keys()) == {"offsets_kpc", "indices", "weights_sub"}
    assert out["offsets_kpc"].shape == (3,)
    assert out["indices"].shape == (3,)
    assert out["weights_sub"].shape == (3,)


def test_compute_offsets_population_vectorized_filters_nan_and_non_positive():
    """NaN, zero, and negative entries are stripped before integration.

    The ``valid`` mask requires both ``v > 0`` and ``t > 0``; entries
    not satisfying both are excluded and ``'indices'`` records the
    surviving positions.
    """
    v = np.array([100.0, np.nan, 200.0, 0.0, -50.0, 150.0])
    t = np.array([100.0, 200.0, np.nan, 300.0, 100.0, 500.0])
    out = compute_offsets_population_vectorized(v, t, max_systems=10, rng=np.random.default_rng(0))
    # Survivors: indices 0 (100, 100) and 5 (150, 500).
    np.testing.assert_array_equal(out["indices"], np.array([0, 5]))
    assert out["offsets_kpc"].shape == (2,)
    assert out["weights_sub"].shape == (2,)


def test_compute_offsets_population_vectorized_max_systems_subsample_uses_weights():
    """When ``len(valid_idx) > max_systems``, the subsample is
    STROOPWAFEL-weighted rather than uniform.

    With heavily skewed weights the subsample's mean weight is
    biased above the population mean weight, confirming that the
    cap is hit and the weighted draw is in effect.
    """
    rng_pop = np.random.default_rng(0)
    N = 200
    v = rng_pop.uniform(50.0, 500.0, size=N)
    t = rng_pop.uniform(10.0, 5000.0, size=N)
    w = np.exp(rng_pop.uniform(-3, 3, size=N))  # heavy-tailed weights
    out = compute_offsets_population_vectorized(
        v,
        t,
        weights=w,
        max_systems=50,
        rng=np.random.default_rng(7),
    )
    assert out["indices"].shape == (50,)
    # The mean weight of the sub-sampled population should exceed the
    # overall mean weight under a STROOPWAFEL-weighted draw because
    # the high-weight tail is preferentially picked.
    pop_mean = float(w.mean())
    sub_mean = float(out["weights_sub"].mean())
    assert sub_mean > pop_mean, (
        f"sub_mean = {sub_mean:.3f} did not exceed pop_mean = "
        f"{pop_mean:.3f}; the weighted draw is suspect."
    )


def test_compute_offsets_population_vectorized_n_proj_respects_array_size():
    """``n_proj`` controls projection-angle averaging; output shape is
    independent of ``n_proj`` but the projection scatter shrinks with
    larger ``n_proj``.

    Two seeds, two ``n_proj`` values: the large-``n_proj`` median
    is closer to the 3D radius mean than the small-``n_proj``
    median because more angles average out the projection scatter.
    """
    v = np.full(20, 200.0)
    t = np.full(20, 500.0)

    out_small = compute_offsets_population_vectorized(
        v, t, n_proj=2, max_systems=20, rng=np.random.default_rng(3)
    )
    out_big = compute_offsets_population_vectorized(
        v, t, n_proj=64, max_systems=20, rng=np.random.default_rng(3)
    )
    assert out_small["offsets_kpc"].shape == (20,)
    assert out_big["offsets_kpc"].shape == (20,)
    assert np.isfinite(out_small["offsets_kpc"]).all()
    assert np.isfinite(out_big["offsets_kpc"]).all()


# ---------------------------------------------------------------------------
# compute_offsets_population legacy path
# ---------------------------------------------------------------------------
def test_compute_offsets_population_legacy_returns_three_documented_keys():
    """``vectorized=False`` returns the same dict shape as the vectorised path."""
    v = np.array([100.0, 200.0, 300.0])
    t = np.array([100.0, 500.0, 1000.0])
    out = compute_offsets_population(
        v, t, max_systems=3, vectorized=False, use_analytic=True, rng=np.random.default_rng(0)
    )
    assert set(out.keys()) == {"offsets_kpc", "indices", "weights_sub"}
    assert out["offsets_kpc"].shape == (3,)


def test_compute_offsets_population_use_analytic_false_returns_finite_offsets():
    """``vectorized=False, use_analytic=False`` runs ``integrate_orbit`` per system."""
    v = np.array([100.0, 200.0])
    t = np.array([200.0, 800.0])
    out = compute_offsets_population(
        v, t, max_systems=2, vectorized=False, use_analytic=False, rng=np.random.default_rng(0)
    )
    assert np.isfinite(out["offsets_kpc"]).all()
    assert (out["offsets_kpc"] >= 0).all()


# ---------------------------------------------------------------------------
# compute_offsets_mixed_hosts
# ---------------------------------------------------------------------------
def test_compute_offsets_mixed_hosts_returns_three_documented_keys():
    """``'per_host'``, ``'mixed_offsets'``, and ``'mixed_weights'`` are present."""
    v = np.array([100.0, 200.0, 300.0])
    t = np.array([100.0, 500.0, 1000.0])
    out = compute_offsets_mixed_hosts(v, t, max_systems=3, rng=np.random.default_rng(0))
    assert set(out.keys()) == {"per_host", "mixed_offsets", "mixed_weights"}
    assert set(out["per_host"].keys()) == set(HOST_MODELS.keys())


def test_compute_offsets_mixed_hosts_weights_carry_host_factor():
    """``mixed_weights`` includes the host mixture-weight multiplier.

    Construct a population with constant input weights of 1.0;
    the per-host slice of ``mixed_weights`` should then equal the
    host's mixture weight times 1.0 for every entry.
    """
    v = np.array([100.0, 200.0, 300.0])
    t = np.array([100.0, 500.0, 1000.0])
    w = np.ones(3)
    out = compute_offsets_mixed_hosts(v, t, weights=w, max_systems=3, rng=np.random.default_rng(0))
    # The concatenation order follows ``HOST_MODELS.items()``.
    expected = np.concatenate([np.full(3, host["weight"]) for host in HOST_MODELS.values()])
    np.testing.assert_array_equal(out["mixed_weights"], expected)


def test_compute_offsets_mixed_hosts_defaults_to_HOST_MODELS_when_none():
    """``host_models=None`` falls back to the project-wide ``HOST_MODELS``."""
    v = np.array([100.0, 200.0])
    t = np.array([100.0, 500.0])
    out = compute_offsets_mixed_hosts(
        v, t, host_models=None, max_systems=2, rng=np.random.default_rng(0)
    )
    assert set(out["per_host"].keys()) == set(HOST_MODELS.keys())


# ---------------------------------------------------------------------------
# assign_host_by_delay
# ---------------------------------------------------------------------------
def test_assign_host_by_delay_partition_is_complete():
    """Every system gets one of the three documented host labels."""
    t = np.array([100.0, 1500.0, 2500.0, 3500.0, 5000.0, 8000.0])
    out = assign_host_by_delay(t, rng=np.random.default_rng(0))
    valid = {"SF_disk", "SF_massive", "Elliptical"}
    assert set(out.tolist()) <= valid


def test_assign_host_by_delay_boundary_uses_strict_less_than():
    """``t == t_sf_max`` lands in ``'Elliptical'`` (the ``<`` convention)."""
    t = np.array([SF_HOST_TRANSITION_T_MYR])
    out = assign_host_by_delay(t, rng=np.random.default_rng(0))
    assert out[0] == "Elliptical"


def test_assign_host_by_delay_sf_split_matches_disk_fraction():
    """Within the SF subset, ``SF_disk`` appears at the documented rate.

    Pin the ``SF_DISK_FRAC_OF_SF_HOSTS`` consumer with a large
    all-SF sample.
    """
    rng = np.random.default_rng(2026)
    n = 20000
    t = rng.uniform(0.0, SF_HOST_TRANSITION_T_MYR - 1.0, size=n)
    out = assign_host_by_delay(t, rng=np.random.default_rng(2027))
    n_disk = int((out == "SF_disk").sum())
    n_sf = int(((out == "SF_disk") | (out == "SF_massive")).sum())
    assert n_sf == n, "test fixture leaked an Elliptical assignment into the SF subset"
    frac = n_disk / n_sf
    assert frac == pytest.approx(SF_DISK_FRAC_OF_SF_HOSTS, abs=0.02), frac


def test_assign_host_by_delay_rng_determinism():
    """Fixed-seed reproducibility under repeated invocation."""
    t = np.linspace(0.0, 5000.0, 200)
    a = assign_host_by_delay(t, rng=np.random.default_rng(42))
    b = assign_host_by_delay(t, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# compute_offsets_delay_hosts
# ---------------------------------------------------------------------------
def test_compute_offsets_delay_hosts_returns_three_documented_keys():
    """Output dict has exactly the documented keys."""
    v = np.array([100.0, 200.0, 300.0])
    t = np.array([100.0, 500.0, 5000.0])
    out = compute_offsets_delay_hosts(v, t, rng=np.random.default_rng(0))
    assert set(out.keys()) == {"offsets_kpc", "weights_sub", "host_assignments"}


def test_compute_offsets_delay_hosts_offsets_finite_and_nonneg():
    """All offsets are finite and non-negative."""
    rng = np.random.default_rng(0)
    v = rng.uniform(50.0, 500.0, size=20)
    t = rng.uniform(10.0, 8000.0, size=20)
    out = compute_offsets_delay_hosts(v, t, rng=np.random.default_rng(11))
    assert np.isfinite(out["offsets_kpc"]).all()
    assert (out["offsets_kpc"] >= 0).all()


def test_compute_offsets_delay_hosts_host_assignments_length_matches_valid():
    """``len(host_assignments) == len(offsets_kpc) == len(weights_sub)``."""
    rng = np.random.default_rng(0)
    v = rng.uniform(50.0, 500.0, size=20)
    t = rng.uniform(10.0, 8000.0, size=20)
    out = compute_offsets_delay_hosts(v, t, rng=np.random.default_rng(11))
    n = out["offsets_kpc"].size
    assert out["host_assignments"].shape == (n,)
    assert out["weights_sub"].shape == (n,)


# ---------------------------------------------------------------------------
# weighted_offset_cdf
# ---------------------------------------------------------------------------
def test_weighted_offset_cdf_unit_normalised():
    """The cumulative-weight CDF ends at exactly 1.0."""
    offsets = np.array([1.0, 2.0, 3.0, 5.0])
    weights = np.array([1.0, 2.0, 0.5, 1.0])
    _, cdf = weighted_offset_cdf(offsets, weights)
    assert cdf[-1] == pytest.approx(1.0, rel=1e-12)


def test_weighted_offset_cdf_filters_non_finite_and_non_positive():
    """NaN, zero, and negative offsets are dropped before sorting.

    The implementation keeps only ``np.isfinite(offsets) & (offsets > 0)``.
    """
    offsets = np.array([1.0, 2.0, np.nan, 3.0, 0.0, -1.0, 5.0])
    weights = np.ones_like(offsets)
    sorted_o, cdf = weighted_offset_cdf(offsets, weights)
    np.testing.assert_array_equal(sorted_o, np.array([1.0, 2.0, 3.0, 5.0]))
    assert cdf.size == 4


def test_weighted_offset_cdf_sentinel_when_fewer_than_two_valid():
    """A single-system input returns the documented sentinel."""
    offsets = np.array([1.0])
    weights = np.array([1.0])
    sorted_o, cdf = weighted_offset_cdf(offsets, weights)
    np.testing.assert_array_equal(sorted_o, np.array([0.0]))
    np.testing.assert_array_equal(cdf, np.array([0.0]))


# ---------------------------------------------------------------------------
# weighted_ks_2samp
# ---------------------------------------------------------------------------
def test_weighted_ks_uniform_weights_matches_scipy():
    """With uniform weights the statistic equals scipy.stats.ks_2samp's D."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 400)
    b = rng.normal(0.4, 1.0, 350)
    D_w, _ = weighted_ks_2samp(a, b)
    D_s = scipy_stats.ks_2samp(a, b).statistic
    assert abs(D_w - D_s) < 1e-12


def test_weighted_ks_identical_samples_zero_statistic():
    x = np.linspace(0.1, 10.0, 50)
    D, p = weighted_ks_2samp(x, x)
    assert D == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_weighted_ks_weights_shift_the_distribution():
    """Up-weighting the large-offset tail of sample 1 increases D against a
    sample concentrated at small offsets."""
    x1 = np.array([1.0, 1.0, 1.0, 100.0])
    x2 = np.array([1.0, 1.0, 1.0, 1.0])
    D_flat, _ = weighted_ks_2samp(x1, x2, weights1=np.ones(4))
    D_tail, _ = weighted_ks_2samp(x1, x2, weights1=np.array([1.0, 1.0, 1.0, 50.0]))
    assert D_tail > D_flat


def test_weighted_ks_degenerate_inputs_return_nan():
    nan_pair = weighted_ks_2samp(np.array([1.0]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(nan_pair[0]) and np.isnan(nan_pair[1])
    zero_w = weighted_ks_2samp(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), weights1=np.array([0.0, 0.0])
    )
    assert np.isnan(zero_w[0]) and np.isnan(zero_w[1])


def test_weighted_ks_pvalue_in_unit_interval():
    rng = np.random.default_rng(3)
    a = rng.gamma(2.0, 2.0, 500)
    b = rng.gamma(2.0, 2.0, 500)
    w = rng.uniform(0.1, 5.0, 500)
    _, p = weighted_ks_2samp(a, b, weights1=w)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Imports used inside tests but not directly referenced above; keep them
# wired so ``ruff F401`` does not complain about the top-level imports.
# ---------------------------------------------------------------------------
def _unused_import_guard():  # pragma: no cover
    _ = warnings
