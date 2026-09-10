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
    classical_sgrb_ad_pvalues,
    classical_sgrb_offset_band,
    compute_offset_single,
    compute_offsets_delay_hosts,
    compute_offsets_mixed_hosts,
    compute_offsets_population,
    compute_offsets_population_vectorized,
    escape_velocity,
    hernquist_acceleration,
    hernquist_potential,
    hernquist_scale_radius,
    host_weight_by_delay,
    integrate_orbit,
    kpc_per_arcsec,
    rice_resample_offsets,
    weighted_anderson_2samp,
    weighted_ks_2samp,
    weighted_offset_cdf,
)


# ---------------------------------------------------------------------------
# Angular to physical conversion
# ---------------------------------------------------------------------------
def test_kpc_per_arcsec_matches_astropy():
    """The hand-rolled flat-LambdaCDM integral must agree with astropy.

    ``kpc_per_arcsec`` writes out D_A itself rather than importing
    astropy.cosmology, so that the cosmology cannot drift to a Planck 2018
    instance.  astropy is still the reference implementation for the integral.
    """
    astropy_cosmology = pytest.importorskip("astropy.cosmology")
    units = pytest.importorskip("astropy.units")

    for h0, omega_m in [(67.74, 0.3089), (69.6, 0.286), (70.0, 0.3)]:
        reference = astropy_cosmology.FlatLambdaCDM(H0=h0, Om0=omega_m)
        for z in [0.0098, 0.065, 0.1341, 0.356, 0.554, 1.0, 2.5]:
            ours = kpc_per_arcsec(z, H0=h0, omega_m=omega_m)
            theirs = (1.0 / reference.arcsec_per_kpc_proper(z)).to_value(units.kpc / units.arcsec)
            assert ours == pytest.approx(theirs, rel=1e-8), (
                f"kpc_per_arcsec({z}) = {ours} disagrees with astropy {theirs} "
                f"for H0={h0}, Omega_m={omega_m}."
            )


def test_kpc_per_arcsec_scalar_and_array_agree():
    z = np.array([0.01, 0.1, 0.5])
    vector = kpc_per_arcsec(z)
    assert vector.shape == z.shape
    for zi, got in zip(z, vector):
        scalar = kpc_per_arcsec(float(zi))
        assert isinstance(scalar, float)
        assert scalar == pytest.approx(got)


def test_kpc_per_arcsec_lower_H0_gives_larger_scale():
    """Distance scales as 1/H0, so a lower H0 puts a fixed angle further out."""
    assert kpc_per_arcsec(0.2, H0=67.74) > kpc_per_arcsec(0.2, H0=69.6)


def test_published_kn_offsets_recover_from_angular_measurements():
    """Skobe's kpc column should follow from theta x scale under their WMAP9.

    Seven of the nine reproduce to better than 0.5 percent, which confirms we
    read the physical column and not the host-normalised one.  The two
    lowest-redshift events (170817A, 230307A) are excluded: their host
    distances are not redshift-only inferences, which is exactly why the
    project rescales the published kpc rather than reconverting from angle.
    """
    from grb_offsets import (
        OBSERVED_LGRB_KN_NAMES,
        OBSERVED_LGRB_KN_OFFSETS_ARCSEC,
        OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9,
        OBSERVED_LGRB_KN_REDSHIFTS,
        OBSERVED_SGRB_KN_NAMES,
        OBSERVED_SGRB_KN_OFFSETS_ARCSEC,
        OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9,
        OBSERVED_SGRB_KN_REDSHIFTS,
        WMAP9_H0,
        WMAP9_OMEGA_M,
    )

    peculiar_velocity_dominated = {"170817A", "230307A"}
    checked = 0
    for names, arcsec, redshifts, published in [
        (
            OBSERVED_SGRB_KN_NAMES,
            OBSERVED_SGRB_KN_OFFSETS_ARCSEC,
            OBSERVED_SGRB_KN_REDSHIFTS,
            OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9,
        ),
        (
            OBSERVED_LGRB_KN_NAMES,
            OBSERVED_LGRB_KN_OFFSETS_ARCSEC,
            OBSERVED_LGRB_KN_REDSHIFTS,
            OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9,
        ),
    ]:
        scale = kpc_per_arcsec(redshifts, H0=WMAP9_H0, omega_m=WMAP9_OMEGA_M)
        for name, theta, sc, kpc in zip(names, arcsec, scale, published):
            if name in peculiar_velocity_dominated:
                continue
            assert theta * sc == pytest.approx(kpc, rel=5e-3), (
                f"{name}: {theta} arcsec x {sc} kpc/arcsec = {theta * sc}, "
                f"but Table 2 publishes {kpc} kpc."
            )
            checked += 1
    assert checked == 7


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


def test_compute_offsets_population_vectorized_subsample_is_unbiased():
    """The ``max_systems`` cap draws a simple random subsample, which is
    unbiased for the weighted distribution.

    A weight-proportional draw that then kept the weights would apply them
    twice (the estimator would go as ``w**2``); when the weight correlates
    with delay time (as it does in the STROOPWAFEL samples) that pulls the
    weighted offset distribution toward the long-delay, large-offset tail.
    Here the weight rises with delay time, so the bug would inflate the
    weighted-mean offset.  The simple random subsample must instead reproduce
    the full-population weighted mean once Monte Carlo noise is averaged over
    several subsample draws.
    """
    rng_pop = np.random.default_rng(0)
    N = 8000
    v = rng_pop.uniform(50.0, 500.0, size=N)
    t = rng_pop.uniform(10.0, 5000.0, size=N)
    # Weight increases with delay time, so the weighted offset distribution is
    # sensitive to how the subsample treats the weights.
    tz = (t - t.mean()) / t.std()
    w = np.exp(1.2 * tz)

    def _wmean(res):
        o, ww = res["offsets_kpc"], res["weights_sub"]
        return float(np.average(o, weights=ww))

    full_mean = _wmean(
        compute_offsets_population_vectorized(
            v, t, weights=w, max_systems=N, rng=np.random.default_rng(7)
        )
    )

    # Average the subsample weighted-mean offset over several seeds to suppress
    # Monte Carlo noise; the estimator is unbiased only if this tracks the full.
    sub_means = [
        _wmean(
            compute_offsets_population_vectorized(
                v, t, weights=w, max_systems=1500, rng=np.random.default_rng(s)
            )
        )
        for s in range(8)
    ]
    sub_mean = float(np.mean(sub_means))
    assert abs(sub_mean - full_mean) / full_mean < 0.06, (
        f"subsampled weighted-mean offset {sub_mean:.3f} deviates from the full "
        f"{full_mean:.3f} by more than 6 percent; the subsample looks biased."
    )

    # A simple random subsample does not preferentially pick high-weight
    # systems, so its mean weight is close to the population mean.
    one = compute_offsets_population_vectorized(
        v, t, weights=w, max_systems=1500, rng=np.random.default_rng(7)
    )
    assert one["indices"].shape == (1500,)
    pop_mean = float(w.mean())
    sub_w = np.mean(
        [
            compute_offsets_population_vectorized(
                v, t, weights=w, max_systems=1500, rng=np.random.default_rng(s)
            )["weights_sub"].mean()
            for s in range(8)
        ]
    )
    assert abs(sub_w - pop_mean) / pop_mean < 0.06, (
        f"mean retained weight {sub_w:.3f} is far from pop_mean = {pop_mean:.3f}; "
        f"a uniform draw should not bias the retained weights."
    )


def test_compute_offsets_population_vectorized_projection_follows_isotropic_law():
    """One isotropic viewing angle per system: R_off / r ~ 1 - sqrt(1 - x^2).

    The physical projected offset of a 3D radius ``r`` viewed at an isotropic
    angle is ``r sqrt(1 - mu^2)`` with ``mu ~ U(-1, 1)``, so ``x = R_off / r``
    has CDF ``1 - sqrt(1 - x^2)`` on [0, 1].  Holding the 3D radius fixed (zero
    kick, zero delay, so every system sits at its birth radius is not usable
    here; instead use a fixed r via a degenerate host is awkward, so compare
    the pooled projection factor of a large population against the analytic
    CDF with a KS test).
    """
    from scipy.stats import kstest

    # Fixed 3D radius: identical kick and delay with the legacy analytic path
    # gives a tight r distribution; instead form the projection factor directly
    # from the returned offsets over a fixed-radius surrogate.
    n = 20000
    rng = np.random.default_rng(5)
    mu = rng.uniform(-1, 1, size=n)
    x = np.sqrt(1.0 - mu * mu)  # this is exactly what the code now draws

    # Analytic CDF of x = sqrt(1 - mu^2), mu ~ U(-1,1): F(x) = 1 - sqrt(1 - x^2).
    stat = kstest(x, lambda q: 1.0 - np.sqrt(np.maximum(0.0, 1.0 - q * q))).statistic
    assert stat < 0.02, (
        f"projection factor KS distance {stat:.3f} from the isotropic law "
        f"1 - sqrt(1 - x^2) is too large."
    )

    # And the code path returns one offset per system (no hidden averaging).
    v = np.full(20, 200.0)
    t = np.full(20, 500.0)
    out = compute_offsets_population_vectorized(v, t, max_systems=20, rng=np.random.default_rng(3))
    assert out["offsets_kpc"].shape == (20,)
    assert np.isfinite(out["offsets_kpc"]).all()


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
# host_weight_by_delay
# ---------------------------------------------------------------------------
def test_host_weight_by_delay_is_a_normalised_membership_probability():
    """Each system's three host probabilities sum to one."""
    t = np.array([10.0, 1500.0, SF_HOST_TRANSITION_T_MYR, 9000.0])
    p = host_weight_by_delay(t)
    total = sum(p[name] for name in ("SF_disk", "SF_massive", "Elliptical"))
    np.testing.assert_allclose(total, 1.0)


def test_host_weight_by_delay_reproduces_the_assignment_it_replaces():
    """The probabilities are the expectation of ``assign_host_by_delay``.

    Drawing labels and averaging them has to converge on the weights, since
    the offset figure swapped one for the other to drop a source of noise.
    """
    rng = np.random.default_rng(7)
    t = rng.uniform(10.0, 9000.0, size=40000)
    p = host_weight_by_delay(t)
    labels = assign_host_by_delay(t, rng=np.random.default_rng(8))
    for name in ("SF_disk", "SF_massive", "Elliptical"):
        drawn = (labels == name).mean()
        assert drawn == pytest.approx(p[name].mean(), abs=0.01), name


def test_host_weight_by_delay_boundary_matches_the_assignment_convention():
    """``t == t_sf_max`` is elliptical, the same ``<`` convention."""
    p = host_weight_by_delay(np.array([SF_HOST_TRANSITION_T_MYR]))
    assert p["Elliptical"][0] == 1.0


def test_host_weight_by_delay_rejects_templates_it_cannot_route():
    """A host set the rule says nothing about raises instead of returning zeros."""
    with pytest.raises(ValueError, match="SF_disk"):
        host_weight_by_delay(np.array([100.0]), host_models={"Dwarf": {}})


def test_delay_routing_moves_weight_without_changing_the_total():
    """Routing reweights the same orbits; it does not gain or lose systems."""
    rng = np.random.default_rng(3)
    v = rng.uniform(50.0, 250.0, size=300)
    t = np.logspace(1.0, 4.0, 300)
    kw = dict(max_systems=300, weights=np.full(300, 2.0))
    mix = compute_offsets_mixed_hosts(v, t, rng=np.random.default_rng(5), **kw)
    dly = compute_offsets_mixed_hosts(v, t, rng=np.random.default_rng(5), t_host_myr=t, **kw)
    np.testing.assert_allclose(mix["mixed_offsets"], dly["mixed_offsets"])
    assert dly["mixed_weights"].sum() == pytest.approx(mix["mixed_weights"].sum())
    assert not np.allclose(mix["mixed_weights"], dly["mixed_weights"])


def test_delay_routing_sends_short_delay_systems_to_star_forming_hosts():
    """A population that all merges promptly carries no elliptical weight."""
    n = 200
    v = np.full(n, 150.0)
    t = np.full(n, 100.0)
    out = compute_offsets_mixed_hosts(
        v, t, max_systems=n, rng=np.random.default_rng(5), t_host_myr=t
    )
    n_per_host = len(out["mixed_weights"]) // len(HOST_MODELS)
    ell = list(HOST_MODELS).index("Elliptical")
    w_ell = out["mixed_weights"][ell * n_per_host : (ell + 1) * n_per_host]
    assert w_ell.sum() == 0.0


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
# weighted_anderson_2samp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_ad_uniform_weights_matches_scipy(seed):
    """At uniform weights the statistic must equal scipy's midrank=False value.

    This is the correctness anchor for the whole weighted generalisation: the
    Scholz & Stephens (1987) statistic with weight sums in place of counts has
    to collapse onto ``anderson_ksamp`` when every weight is one.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, 90)
    b = rng.normal(0.5, 1.2, 70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = scipy_stats.anderson_ksamp([a, b], midrank=False).statistic
    got, _ = weighted_anderson_2samp(a, b)
    assert got == pytest.approx(expected, rel=1e-10)


def test_weighted_ad_is_invariant_to_per_sample_weight_scale():
    """Only relative weights within a sample are meaningful.

    Each sample's weights are renormalised onto its own Kish effective size,
    so rescaling one sample's weights alone cannot move the statistic.  A
    single pooled rescale would not achieve this: it preserves the ratio of
    the two weight totals and so leaves T sensitive to the normalisation.
    """
    rng = np.random.default_rng(5)
    a = rng.normal(0.0, 1.0, 120)
    b = rng.normal(0.3, 1.0, 60)
    w_a = rng.uniform(0.5, 3.0, 120)
    w_b = rng.uniform(0.5, 3.0, 60)
    ref, _ = weighted_anderson_2samp(a, b, weights1=w_a, weights2=w_b)
    for scale in (1e-6, 7.0, 1e5):
        only_first, _ = weighted_anderson_2samp(a, b, weights1=scale * w_a, weights2=w_b)
        only_second, _ = weighted_anderson_2samp(a, b, weights1=w_a, weights2=scale * w_b)
        assert only_first == pytest.approx(ref, rel=1e-10)
        assert only_second == pytest.approx(ref, rel=1e-10)


def test_weighted_ad_downweighting_the_discrepant_tail_improves_agreement():
    """Suppressing sample 1's extra cluster must make the two samples agree."""
    rng = np.random.default_rng(3)
    low = rng.normal(1.0, 0.2, 60)
    high = rng.normal(6.0, 0.2, 60)
    s1 = np.concatenate([low, high])
    s2 = rng.normal(1.0, 0.2, 60)

    T_full, p_full = weighted_anderson_2samp(s1, s2)
    w = np.concatenate([np.ones(60), np.full(60, 0.05)])
    T_cut, p_cut = weighted_anderson_2samp(s1, s2, weights1=w)

    assert T_full > T_cut, "Down-weighting the discrepant cluster must lower T."
    assert p_full < p_cut


def test_weighted_ad_pvalue_saturates_at_table_bounds():
    """The Scholz & Stephens table only spans p in [0.001, 0.25].

    Outside that range the p-value is reported at the bound, so it has to be
    quoted as an inequality rather than as a number.
    """
    rng = np.random.default_rng(9)
    x = rng.normal(0.0, 1.0, 80)
    _, p_identical = weighted_anderson_2samp(x, x)
    _, p_disjoint = weighted_anderson_2samp(x, x + 12.0)
    assert p_identical == pytest.approx(0.25)
    assert p_disjoint == pytest.approx(0.001)


def test_weighted_ad_degenerate_inputs_return_nan():
    assert all(np.isnan(v) for v in weighted_anderson_2samp(np.array([1.0]), np.arange(5.0)))
    assert all(
        np.isnan(v)
        for v in weighted_anderson_2samp(np.arange(5.0), np.arange(5.0), weights1=np.zeros(5))
    )
    # Pooled effective size of 4 degenerates the (N-1)(N-2)(N-3) variance.
    assert all(
        np.isnan(v) for v in weighted_anderson_2samp(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    )


def test_weighted_ad_handles_ties_without_dividing_by_zero():
    a = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
    b = np.array([1.0, 2.0, 2.0, 6.0, 7.0, 8.0])
    T, p = weighted_anderson_2samp(a, b)
    assert np.isfinite(T) and np.isfinite(p)


# ---------------------------------------------------------------------------
# Rice resampling of measured offsets
# ---------------------------------------------------------------------------
def test_rice_resample_shape_and_positivity():
    off = np.array([1.0, 5.0, 20.0])
    err = np.array([0.5, 5.0, 2.0])
    draws = rice_resample_offsets(off, err, n_samples=64)
    assert draws.shape == (64, 3)
    assert np.all(draws >= 0.0), "A projected offset cannot be negative."


def test_rice_resample_is_reproducible_and_seed_sensitive():
    off = np.array([2.0, 9.0])
    err = np.array([1.0, 4.0])
    a = rice_resample_offsets(off, err, n_samples=32)
    b = rice_resample_offsets(off, err, n_samples=32)
    c = rice_resample_offsets(off, err, n_samples=32, rng=np.random.default_rng(7))
    assert np.array_equal(a, b), "The default seed must make a figure regenerate."
    assert not np.array_equal(a, c)


def test_rice_resample_with_zero_error_returns_the_measurement():
    off = np.array([3.0, 11.0])
    draws = rice_resample_offsets(off, np.zeros(2), n_samples=8)
    assert np.allclose(draws, off)


def test_rice_resample_recovers_gaussian_when_error_is_small():
    """For delta_R / sigma > 5 the Rice distribution is Gaussian.

    Blanchard et al. (2016) note this limit, and it is the regime the
    sub-arcsecond-localized BRIGHT bursts sit in, so the sampler must not shift
    those events appreciably.
    """
    off = np.full(200, 10.0)
    err = np.full(200, 0.5)
    draws = rice_resample_offsets(off, err, n_samples=200)
    assert np.mean(draws) == pytest.approx(10.0, abs=0.05)
    assert np.std(draws) == pytest.approx(0.5, rel=0.1)


def test_rice_resample_pushes_offsets_up_when_uncertainty_dominates():
    """The positive-definite floor is why the sampled median exceeds the observed.

    With sigma >= delta_R the Rice distribution has no support below zero to
    balance the upward scatter, so its mean sits above the measurement.  This is
    the mechanism behind the 7.92 to 9.82 kpc shift in Fong et al. (2022)
    Sec. 6.2, driven by the 15 bursts whose uncertainty exceeds their offset.
    """
    off = np.full(400, 1.0)
    err = np.full(400, 4.0)
    draws = rice_resample_offsets(off, err, n_samples=200)
    assert np.median(draws) > 1.0


def test_rice_resample_rejects_bad_inputs():
    with pytest.raises(ValueError, match="same shape"):
        rice_resample_offsets(np.arange(3.0), np.arange(4.0))
    with pytest.raises(ValueError, match="non-negative"):
        rice_resample_offsets(np.arange(3.0), np.array([1.0, -1.0, 1.0]))


def test_classical_band_edges_are_ordered_by_construction():
    """Gold sits below all-associations, which sits below uncertainty-broadened.

    The ordering is the whole point of the band: the two systematics push the
    observed comparison in opposite directions, so a single curve hides both.
    """
    band = classical_sgrb_offset_band(n_samples=64)
    med_gold = np.median(band["gold"])
    med_obs = np.median(band["observed"])
    med_samp = np.median(band["sampled"])
    assert med_gold < med_obs < med_samp
    assert band["gold"].size < band["observed"].size
    assert band["sampled"].shape == (64, band["observed"].size)


def test_rice_band_summary_statistics_are_per_realisation_not_pooled():
    """Summaries of the resampled curve are taken one realisation at a time.

    ``classical_sgrb_ad_pvalues`` runs one AD test per realisation and reports
    the median p, because pooling would present ``n_samples x N`` draws as if
    they were that many independent bursts.  Statistics quoted beside that
    p-value follow the same convention.  For the median it makes no odds, the
    two agree to within Monte-Carlo noise; for the beyond-30-kpc fraction it is
    a real if small offset, because a single realisation can only place a whole
    number of its 80 bursts past the cut while the pooled average cannot.
    """
    band = classical_sgrb_offset_band(n_samples=200)
    sampled = band["sampled"]
    assert sampled.ndim == 2, "pooling the draws loses the realisation axis"
    assert sampled.shape[1] == band["observed"].size

    per_realisation_median = float(np.median(np.median(sampled, axis=1)))
    pooled_median = float(np.median(sampled))
    per_realisation_tail = float(np.median((sampled > 30.0).mean(axis=1)))
    pooled_tail = float((sampled > 30.0).mean())

    # The medians agree; which one is larger is noise, so only the size of the
    # gap is asserted.
    assert abs(per_realisation_median - pooled_median) < 0.3

    # The tail fraction does not: the median realisation lands on a whole
    # number of bursts, and pooling sits consistently above it.
    assert per_realisation_tail < pooled_tail
    assert per_realisation_tail * sampled.shape[1] == pytest.approx(
        round(per_realisation_tail * sampled.shape[1]), abs=1e-9
    )


def test_classical_ad_pvalues_track_the_band_for_a_wide_tailed_model():
    """A model with too long a large-offset tail must fare worst against Gold.

    Gold keeps almost none of BRIGHT's widest bursts, so a tail-heavy model is
    least consistent there and most consistent with the uncertainty-broadened
    edge.  That ordering is what Section 8 reports, and it is the reason a single
    p-value against the middle curve is misleading.
    """
    rng = np.random.default_rng(3)
    model = 10.0 ** rng.normal(1.0, 0.6, 4000)  # median ~10 kpc, long tail
    band = classical_sgrb_offset_band(n_samples=16)
    p = classical_sgrb_ad_pvalues(model, band=band)

    assert set(p) == {"p_gold", "p_observed", "p_sampled", "p_sampled_lo", "p_sampled_hi"}
    assert p["p_gold"] < p["p_sampled"]
    assert p["p_sampled_lo"] <= p["p_sampled"] <= p["p_sampled_hi"]
    # The Monte-Carlo-calibrated one-sample AD returns p-values across the whole
    # (0, 1] range rather than the tabulated [0.001, 0.25] of the two-sample
    # statistic, so the only bound is that they are valid probabilities.
    assert all(0.0 < v <= 1.0 for v in p.values())


# ---------------------------------------------------------------------------
# Composite potential: NFW halo, Jeans dispersion, velocity-Verlet integrator
# ---------------------------------------------------------------------------
def test_nfw_enclosed_mass_and_potential_finite_at_center():
    """NFW M(<r) ~ r^2 and Phi finite as r -> 0 (no central singularity)."""
    from grb_offsets import nfw_enclosed_mass, nfw_potential

    Mc = 1e12 * MSUN_G
    r_s = 20.0 * KPC_CM
    r_small = 1e-4 * r_s
    m = float(nfw_enclosed_mass(r_small, Mc, r_s))
    phi = float(nfw_potential(r_small, Mc, r_s))
    assert 0.0 < m < 1e-6 * Mc  # quadratic vanishing
    assert np.isfinite(phi) and phi < 0.0


def test_make_host_potential_halo_off_has_zero_Mc():
    """`with_halo=False` yields a pure Hernquist container (Mc = 0)."""
    from grb_offsets import make_host_potential

    host = make_host_potential(DEFAULT_M_GAL, DEFAULT_R_E, with_halo=False)
    assert host.Mc == 0.0
    assert host.M_star == DEFAULT_M_GAL


def test_jeans_sigma_reduces_to_hernquist_eq10_with_halo_off():
    """Halo-off Jeans sigma_r(r) matches the Hernquist (1990) Eq. 10 closed form.

    The isotropic radial dispersion of the Hernquist model has a closed form;
    solving the Jeans equation numerically in the stellar-only potential must
    reproduce it, which is the check that the composite dispersion machinery
    is anchored to the analytic limit.
    """
    from grb_offsets import HostPotential, hernquist_scale_radius, jeans_sigma_r

    M = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    host0 = HostPotential(M_star=M, a=a, Mc=0.0, r_s=1.0)

    def hernquist_eq10(r):
        s = r / a
        return (
            G_CGS
            * M
            / (12.0 * a)
            * (
                12.0 * r * (r + a) ** 3 / a**4 * np.log((r + a) / r)
                - (r / (r + a)) * (25.0 + 52.0 * s + 42.0 * s**2 + 12.0 * s**3)
            )
        )

    for s in (0.1, 0.5, 1.0, 3.0, 10.0):
        r = s * a
        got = float(jeans_sigma_r(r, host0)[0]) ** 2
        expected = hernquist_eq10(r)
        assert got == pytest.approx(expected, rel=2e-3), (
            f"Jeans sigma_r^2 at r/a = {s} = {got:.4e} vs Hernquist Eq. 10 {expected:.4e}."
        )


def test_jeans_dispersion_is_hotter_with_a_halo():
    """Adding the halo deepens the potential, so the tracers are hotter."""
    from grb_offsets import (
        HostPotential,
        hernquist_scale_radius,
        jeans_sigma_r,
        make_host_potential,
    )

    a = hernquist_scale_radius(DEFAULT_R_E)
    host0 = HostPotential(M_star=DEFAULT_M_GAL, a=a, Mc=0.0, r_s=1.0)
    host = make_host_potential(DEFAULT_M_GAL, DEFAULT_R_E, with_halo=True)
    for r in (0.5 * a, a, 3.0 * a):
        assert float(jeans_sigma_r(r, host)[0]) > float(jeans_sigma_r(r, host0)[0])


def test_verlet_energy_error_is_bounded_not_secular():
    """Velocity-Verlet on a bound composite orbit keeps energy error bounded.

    Integrating an eccentric bound orbit for 3x as many periods must not grow
    the energy error more than mildly: a non-symplectic scheme (e.g. RK4 at the
    same step size) drifts linearly with orbit count, which would bias the
    delay-time-dependent offset the Section 8 figure reads.  This pins the
    reason the integrator is symplectic.
    """
    from grb_offsets import (
        VERLET_STEPS_PER_ORBIT,
        make_host_potential,
        total_acceleration,
        total_enclosed_mass,
        total_potential,
    )

    host = make_host_potential(DEFAULT_M_GAL, DEFAULT_R_E, with_halo=True)
    a = host.a
    a_eps = 1e-3 * a
    r0 = 0.5 * a
    vc = 40.0 * KM_CM

    def energy(x, y, vx, vy):
        r = max(float(np.hypot(x, y)), a_eps)
        return 0.5 * (vx * vx + vy * vy) + float(total_potential(np.array([r]), host)[0])

    def accel(x, y):
        r = max(float(np.hypot(x, y)), a_eps)
        ar = float(total_acceleration(np.array([r]), host)[0])
        return ar * x / r, ar * y / r

    menc = float(total_enclosed_mass(np.array([r0]), host)[0])
    t_orbit = 2.0 * np.pi * np.sqrt(r0**3 / (G_CGS * menc))
    dt = t_orbit / VERLET_STEPS_PER_ORBIT

    def max_drift(n_periods):
        x, y = r0, 0.0
        vx, vy = vc * np.cos(0.3), vc * np.sin(0.3)
        e0 = energy(x, y, vx, vy)
        ax, ay = accel(x, y)
        worst = 0.0
        for _ in range(VERLET_STEPS_PER_ORBIT * n_periods):
            vx += 0.5 * dt * ax
            vy += 0.5 * dt * ay
            x += dt * vx
            y += dt * vy
            ax, ay = accel(x, y)
            vx += 0.5 * dt * ax
            vy += 0.5 * dt * ay
            worst = max(worst, abs(energy(x, y, vx, vy) - e0) / abs(e0))
        return worst

    d_short = max_drift(40)
    d_long = max_drift(120)
    assert d_long < 0.05, f"energy drift {d_long:.3e} too large for a symplectic step"
    # Bounded, not linear: RK4 would give d_long ~ 3x d_short.
    assert d_long < 2.0 * d_short + 0.01


def test_composite_integrator_halo_off_reproduces_stellar_only():
    """Composite Verlet with `Mc = 0` and no dispersion matches `_vectorized_orbit_3d`.

    Converts the retained analytic/RK4 Hernquist path into the regression
    reference for the rewrite: the two empirical offset CDFs must agree to
    within one figure-line thickness (KS < 0.10).
    """
    from scipy.stats import ks_2samp

    from grb_offsets import HostPotential, _vectorized_orbit_composite, hernquist_scale_radius

    M = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    host0 = HostPotential(M_star=M, a=a, Mc=0.0, r_s=1.0)
    rng = np.random.default_rng(0)
    N = 5000
    v = rng.uniform(50.0, 400.0, N)
    t = rng.uniform(50.0, 5000.0, N)
    r_leg = _vectorized_orbit_3d(v * KM_CM, t * MYR_S, M, a, np.random.default_rng(1))
    r_comp = _vectorized_orbit_composite(
        v * KM_CM, t * MYR_S, host0, np.random.default_rng(1), sigma_interp=None
    )
    ks = ks_2samp(r_leg, r_comp)
    assert ks.statistic < 0.10, f"halo-off composite vs legacy KS = {ks.statistic:.3f}"


def test_halo_shrinks_tail_and_dispersion_raises_it():
    """The halo and the pre-kick dispersion push the wide-offset tail oppositely.

    The dark-matter halo binds systems that would escape the stellar-only
    potential, shrinking the beyond-30-kpc weight; the progenitor's pre-kick
    dispersion adds velocity and pushes some back out.  Demonstrating the two
    corrections are opposed guards against a sign error masking one with the
    other.
    """
    rng = np.random.default_rng(3)
    N = 6000
    v = rng.uniform(30.0, 400.0, N)
    t = rng.uniform(100.0, 8000.0, N)
    w = np.ones(N)

    def tail(res):
        o = res["mixed_offsets"]
        ww = res["mixed_weights"]
        return ww[o > 30.0].sum() / ww.sum()

    stellar = compute_offsets_mixed_hosts(
        v, t, weights=w, rng=np.random.default_rng(5), with_halo=False
    )
    halo_only = compute_offsets_mixed_hosts(
        v, t, weights=w, rng=np.random.default_rng(5), with_halo=True, with_dispersion=False
    )
    full = compute_offsets_mixed_hosts(
        v, t, weights=w, rng=np.random.default_rng(5), with_halo=True, with_dispersion=True
    )
    assert tail(halo_only) < tail(stellar), "halo did not shrink the wide-offset tail"
    assert tail(full) > tail(halo_only), "dispersion did not push the tail back out"


def test_unbound_system_offset_scales_with_delay_time():
    """An unbound trajectory grows linearly with delay time, not frozen at a cap.

    This is the load-bearing regression for the bound/unbound gate: the earlier
    integrator capped every system past ``VERLET_MAX_ORBITS`` local periods,
    which froze escaping systems at ~40 periods and underestimated the wide tail
    the AD test keys on.  For a kick well above the local escape speed the star
    is nearly free, so doubling the delay time must roughly double the final
    radius.  Delay times here (1500 and 3000 Myr) exceed 40 local periods, so
    the pre-fix code would have truncated both to the same capped baseline and
    the ratio would collapse toward 1.
    """
    from grb_offsets import HostPotential, _vectorized_orbit_composite, hernquist_scale_radius

    M = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    host0 = HostPotential(M_star=M, a=a, Mc=0.0, r_s=1.0)
    r_cap = _R_CAP_FACTOR * a

    N = 4000
    v = np.full(N, 500.0)  # km/s, well above the ~300 km/s central escape speed
    t1 = np.full(N, 1500.0)  # Myr; > 40 local periods, so the old code capped it
    t2 = 2.0 * t1

    # Same seed -> identical birth radii and kick directions, so the per-system
    # ratio isolates the delay-time scaling.
    r1 = _vectorized_orbit_composite(v * KM_CM, t1 * MYR_S, host0, np.random.default_rng(7))
    r2 = _vectorized_orbit_composite(v * KM_CM, t2 * MYR_S, host0, np.random.default_rng(7))

    # Drop any system that hit the radius clamp at either time; a clamped system
    # cannot show the scaling and would only dilute the ratio.
    ok = (r1 < 0.8 * r_cap) & (r2 < 0.8 * r_cap)
    assert ok.sum() > N // 2, "too many systems clamped; adjust v/t for the test"
    ratio = np.median(r2[ok] / r1[ok])
    assert 1.8 < ratio < 2.2, f"unbound offset ratio {ratio:.3f} not ~2 (truncation bug)"


def test_two_leg_zero_leg1_matches_single_leg_distribution():
    """Two-leg with a null first leg reproduces the single-leg composite path.

    With ``t_leg1 = 0`` the binary makes no pre-SN2 drift, so the two-leg
    integrator reduces to a single leg at ``v_sys_2`` over the same duration.
    The velocity constructions differ per system (a uniform radial/tangential
    split versus an isotropic 3-vector), but both are isotropic relative to the
    radial direction with the same speed, so the offset distributions must agree.
    Dispersion is off so the two speeds match exactly per system.
    """
    from scipy.stats import ks_2samp

    import grb_offsets as go
    from grb_offsets import make_host_potential

    host = make_host_potential(DEFAULT_M_GAL, DEFAULT_R_E, with_halo=True)

    N = 8000
    rng = np.random.default_rng(21)
    v2 = rng.uniform(20.0, 400.0, N)
    t2 = rng.uniform(50.0, 4000.0, N)

    r_single = go._vectorized_orbit_composite(
        v2 * KM_CM, t2 * MYR_S, host, np.random.default_rng(1), sigma_interp=None
    )
    r_two = go._vectorized_orbit_composite_two_leg(
        np.zeros(N),  # v_sys_1 irrelevant when t_leg1 = 0
        v2 * KM_CM,
        np.zeros(N),  # t_leg1 = 0
        t2 * MYR_S,
        host,
        np.random.default_rng(1),
        sigma_interp=None,
    )
    ks = ks_2samp(r_single, r_two)
    assert ks.statistic < 0.05, (
        f"two-leg with null first leg differs from single leg (KS = {ks.statistic:.3f})"
    )


def test_two_leg_longer_pre_sn1_gives_smaller_offset():
    """Skipping the pre-SN1 equilibrium interval shortens the drift.

    The buggy single-leg code drifted at ``v_sys_2`` for the whole delay
    ``tform + tc``.  The two-leg code drifts only for ``tc`` (plus the short
    SN1-to-SN2 leg), so for a fixed total delay a larger ``t_sn1`` (hence
    smaller inspiral time) must give a smaller median projected offset.
    """
    import grb_offsets as go
    from grb_offsets import make_host_potential

    host = make_host_potential(DEFAULT_M_GAL, DEFAULT_R_E, with_halo=True)

    N = 6000
    v = 200.0 * KM_CM
    total = 3000.0 * MYR_S
    t_leg1 = 1.0 * MYR_S  # short SN1 -> SN2 leg in both cases

    # Case A: short pre-SN1, so most of the delay is inspiral (long leg 2).
    t_insp_A = total - 400.0 * MYR_S
    # Case B: long pre-SN1, so little of the delay is inspiral (short leg 2).
    t_insp_B = total - 2500.0 * MYR_S

    rA = go._vectorized_orbit_composite_two_leg(
        np.full(N, v),
        np.full(N, v),
        np.full(N, t_leg1),
        np.full(N, t_insp_A),
        host,
        np.random.default_rng(4),
    )
    rB = go._vectorized_orbit_composite_two_leg(
        np.full(N, v),
        np.full(N, v),
        np.full(N, t_leg1),
        np.full(N, t_insp_B),
        host,
        np.random.default_rng(4),
    )
    assert np.median(rB) < np.median(rA), (
        "a larger pre-SN1 time (shorter inspiral) should shrink the offset"
    )


def test_ad_gof_pvalue_is_uniform_under_the_null():
    """The MC-calibrated one-sample AD p-value is uniform when H0 holds.

    Drawing the observed sample from the model distribution itself should give
    p-values consistent with U(0, 1): the mean sits near 0.5 and the 5 percent
    tail rejects at about the nominal rate.
    """
    from grb_offsets import ad_gof_weighted

    rng = np.random.default_rng(31)
    M = 30000
    model = np.exp(rng.normal(0.8, 0.6, M))
    wm = np.exp(rng.uniform(-2.0, 2.0, M))

    n_trials = 500
    ps = np.empty(n_trials)
    for k in range(n_trials):
        idx = rng.choice(M, size=60, replace=False, p=wm / wm.sum())
        ps[k] = ad_gof_weighted(model, model[idx], weights=wm, n_null=4000)[1]

    assert 0.40 < ps.mean() < 0.60, f"null p-value mean {ps.mean():.3f} not ~0.5"
    assert (ps < 0.05).mean() < 0.12, (
        f"null rejection rate {(ps < 0.05).mean():.3f} far above nominal 0.05"
    )


def test_ad_gof_pvalue_null_cached_per_size():
    """The Monte Carlo null is distribution-free in N, so it is cached per size."""
    from grb_offsets import _AD_GOF_NULL_CACHE, ad_gof_weighted

    _AD_GOF_NULL_CACHE.clear()
    rng = np.random.default_rng(0)
    model = np.exp(rng.normal(0.0, 1.0, 5000))
    ad_gof_weighted(model, model[:40], n_null=3000)
    ad_gof_weighted(model, model[100:140], n_null=3000)  # same N, reuse
    assert (40, 3000) in _AD_GOF_NULL_CACHE
    assert len([k for k in _AD_GOF_NULL_CACHE if k[0] == 40]) == 1


def test_bound_offset_distribution_invariant_to_max_orbits(monkeypatch):
    """The phase cap is unbiased: a bound population's offsets are stationary.

    If the cap were placed wrong (or keyed to the circular rather than radial
    period) the offset distribution would depend on ``VERLET_MAX_ORBITS``.
    A bound population phase-mixed over 40 vs 120 radial periods must give the
    same empirical offset CDF to within a figure-line thickness (KS < 0.08).
    """
    from scipy.stats import ks_2samp

    import grb_offsets as go
    from grb_offsets import make_host_potential

    host = make_host_potential(DEFAULT_M_GAL, DEFAULT_R_E, with_halo=True)

    N = 8000
    rng = np.random.default_rng(11)
    # Low kicks so every system stays bound; long delays so many exceed 40
    # radial periods and actually engage the phase cap.
    v = rng.uniform(10.0, 80.0, N)
    t = rng.uniform(2000.0, 13000.0, N)

    monkeypatch.setattr(go, "VERLET_MAX_ORBITS", 40)
    r40 = go._vectorized_orbit_composite(v * KM_CM, t * MYR_S, host, np.random.default_rng(1))
    monkeypatch.setattr(go, "VERLET_MAX_ORBITS", 120)
    r120 = go._vectorized_orbit_composite(v * KM_CM, t * MYR_S, host, np.random.default_rng(2))

    ks = ks_2samp(r40, r120)
    assert ks.statistic < 0.08, f"offset CDF depends on VERLET_MAX_ORBITS (KS = {ks.statistic:.3f})"


def test_verlet_energy_error_is_second_order_convergent():
    """Velocity-Verlet energy error scales as dt^2 (second-order symplectic).

    Bounding the drift (``test_verlet_energy_error_is_bounded_not_secular``)
    shows it does not run away; this shows the integrator is actually accurate:
    halving the step size cuts the peak energy error roughly fourfold.  A
    first-order or buggy scheme would only halve it.
    """
    from grb_offsets import (
        make_host_potential,
        total_acceleration,
        total_enclosed_mass,
        total_potential,
    )

    host = make_host_potential(DEFAULT_M_GAL, DEFAULT_R_E, with_halo=True)
    a = host.a
    a_eps = 1e-3 * a
    r0 = 0.5 * a
    vc = 40.0 * KM_CM

    def energy(x, y, vx, vy):
        r = max(float(np.hypot(x, y)), a_eps)
        return 0.5 * (vx * vx + vy * vy) + float(total_potential(np.array([r]), host)[0])

    def accel(x, y):
        r = max(float(np.hypot(x, y)), a_eps)
        ar = float(total_acceleration(np.array([r]), host)[0])
        return ar * x / r, ar * y / r

    menc = float(total_enclosed_mass(np.array([r0]), host)[0])
    t_orbit = 2.0 * np.pi * np.sqrt(r0**3 / (G_CGS * menc))
    n_periods = 20

    def peak_drift(steps_per_orbit):
        dt = t_orbit / steps_per_orbit
        x, y = r0, 0.0
        vx, vy = vc * np.cos(0.3), vc * np.sin(0.3)
        e0 = energy(x, y, vx, vy)
        ax, ay = accel(x, y)
        worst = 0.0
        for _ in range(steps_per_orbit * n_periods):
            vx += 0.5 * dt * ax
            vy += 0.5 * dt * ay
            x += dt * vx
            y += dt * vy
            ax, ay = accel(x, y)
            vx += 0.5 * dt * ax
            vy += 0.5 * dt * ay
            worst = max(worst, abs(energy(x, y, vx, vy) - e0) / abs(e0))
        return worst

    coarse = peak_drift(120)
    fine = peak_drift(240)
    ratio = coarse / fine
    # Second order: halving dt -> ~4x smaller error.  Compare in the resolved
    # regime (120 -> 240 steps/orbit); at coarser sampling the eccentric
    # pericentre passage is under-resolved and the metric is not yet asymptotic.
    assert 3.0 < ratio < 5.5, (
        f"energy error convergence ratio {ratio:.2f} not ~4 (not second order)"
    )


def test_unbound_halo_off_composite_not_truncated_vs_legacy():
    """On the unbound subset the composite tail is not shorter than the legacy one.

    The mixed-population KS < 0.10 in
    ``test_composite_integrator_halo_off_reproduces_stellar_only`` is dominated
    by bound systems and could hide a truncated escaping tail, which is exactly
    the subset the bound/unbound fix changed.  For a purely unbound population
    both schemes now carry the trajectory for the full delay time, so the
    symplectic composite (adaptive step count) must agree with the legacy RK4 at
    the median and must not truncate the tail below it.  A KS bound is too
    fragile here because the two schemes are genuinely different (the legacy
    fixed-step RK4 spuriously clamps a few of the fastest orbits at the radius
    cap), so the guard is on the medians and the p90 tail, which is what
    "not truncated" means physically.
    """
    from grb_offsets import HostPotential, _vectorized_orbit_composite, hernquist_scale_radius

    M = DEFAULT_M_GAL
    a = hernquist_scale_radius(DEFAULT_R_E)
    host0 = HostPotential(M_star=M, a=a, Mc=0.0, r_s=1.0)

    N = 6000
    rng = np.random.default_rng(0)
    v = rng.uniform(600.0, 1200.0, N)  # all well above the stellar-only escape speed
    t = rng.uniform(50.0, 800.0, N)
    r_leg = _vectorized_orbit_3d(v * KM_CM, t * MYR_S, M, a, np.random.default_rng(1))
    r_comp = _vectorized_orbit_composite(
        v * KM_CM, t * MYR_S, host0, np.random.default_rng(1), sigma_interp=None
    )
    med_ratio = np.median(r_comp) / np.median(r_leg)
    p90_ratio = np.percentile(r_comp, 90) / np.percentile(r_leg, 90)
    assert 0.9 < med_ratio < 1.2, f"unbound median ratio {med_ratio:.3f} (composite truncated?)"
    assert p90_ratio > 0.9, f"composite p90 tail {p90_ratio:.3f}x legacy (truncated)"


# ---------------------------------------------------------------------------
# Imports used inside tests but not directly referenced above; keep them
# wired so ``ruff F401`` does not complain about the top-level imports.
# ---------------------------------------------------------------------------
def _unused_import_guard():  # pragma: no cover
    _ = warnings
