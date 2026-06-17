"""Behaviour and edge-case tests for ``grb_physics``.

Complements ``tests/unit/test_physics.py`` (Foucart Eq. 4 by-hand,
KF2020 finite-and-nonneg, chirp-mass identity) and
``tests/unit/test_isco.py`` (Bardeen 1972 anchors).  This file fills
the gaps: NS-radius plateau and cubic suppression, the EOS-anchored
radius helper, ``mcrit_to_r14`` linear interpolation, compactness at
the GW170817 canonical, the truncated double-Gaussian CDF used by the
Alsing+ 2018 remap, Foucart 2018 ``clip_Q`` / ``clip_chi`` paths and
spin monotonicity, ``bns_disk_mass`` symmetry plus the documented
sharp transition around R_1.4 ~ 12 to 13 km, Gottlieb+ 2025 Eq. 11
scaling, the module-level deprecation hook, NaN propagation, and
vectorisation parity for every public BHNS / BNS mass helper.

References cited inline per test:
Bardeen, Press and Teukolsky (1972) ApJ 178, 347 (ISCO Eq. 2.21);
Foucart, Hinderer and Nissanke (2018) PRD 98, 081501 (BHNS remnant);
Kruger and Foucart (2020) PRD 101, 103002 (BHNS dyn ejecta, BNS disk
and BNS dyn ejecta); Read et al. (2009) PRD 79, 124032 (EOS Table III);
Raaijmakers et al. (2021) ApJL 918, L29 (M_TOV posterior);
Alsing, Silva and Berti (2018) MNRAS 478, 1377 (Galactic-NS double
Gaussian); Gottlieb et al. (2025) arXiv:2411.13657 (Eq. 11 disk-wind
ejecta).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from grb_physics import (
    _MEAN_MASS_EVOLVED_VALUE,
    EOS_MODELS,
    GOTTLIEB25_WIND_FRAC,
    M_TOV,
    NS_REMAP_M_MIN,
    _compactness,
    _truncated_double_gauss_cdf,
    bhns_dynamical_ejecta,
    bns_disk_mass,
    bns_dynamical_ejecta,
    effective_aligned_spin,
    foucart_disk_mass,
    foucart_remnant_mass,
    gottlieb25_eq11,
    hmns_wind_ejecta,
    mcrit_to_r14,
    ns_radius,
    ns_radius_from_eos,
    r_isco,
)


# ---------------------------------------------------------------------------
# ns_radius: plateau + cubic suppression + above-M_TOV NaN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M_NS", [0.5, 1.0, 1.2, 1.3, 1.399])
def test_ns_radius_plateau_below_1p4_msun(M_NS):
    """Below 1.4 Msun, ``ns_radius`` returns ``R_1p4_km`` exactly.

    Modeling choice documented in the ``ns_radius`` docstring: real
    R(M) sequences are nearly flat across 1.0 to 1.4 Msun for most
    EOSs, but the qualitative direction is EOS-dependent and small;
    the heuristic returns the reference radius unchanged below the
    1.4 Msun anchor.
    """
    R = float(ns_radius(M_NS, R_1p4_km=12.0))
    assert R == pytest.approx(12.0, rel=1e-12)


def test_ns_radius_at_1p4_msun_equals_R_1p4():
    """``ns_radius(1.4) = R_1p4`` exactly, the construction anchor."""
    assert float(ns_radius(1.4, R_1p4_km=12.0)) == pytest.approx(12.0, rel=1e-12)
    assert float(ns_radius(1.4, R_1p4_km=11.1)) == pytest.approx(11.1, rel=1e-12)


def test_ns_radius_at_M_TOV_drops_by_15_percent():
    """Cubic suppression hits ``0.85 * R_1p4`` at ``M_NS = M_TOV``.

    Docstring claim: ``R = R_1p4 * (1 - 0.15 * x**3)`` with
    ``x = (M - 1.4)/(M_TOV - 1.4)``, saturated at ``x = 1``.
    """
    R = float(ns_radius(M_TOV, R_1p4_km=12.0))
    assert R == pytest.approx(0.85 * 12.0, rel=1e-12)


def test_ns_radius_above_M_TOV_returns_nan():
    """No NS above ``M_TOV``; the helper must return NaN there."""
    R = ns_radius(np.array([M_TOV + 1e-3, M_TOV + 0.1]), R_1p4_km=12.0)
    assert np.isnan(R).all()


@pytest.mark.parametrize("eos_name", list(EOS_MODELS.keys()))
def test_ns_radius_from_eos_recovers_per_EOS_anchors(eos_name):
    """``ns_radius_from_eos(1.4, eos) == EOS_MODELS[eos]['R_1p4']``.

    The EOS-anchored helper forwards ``R_1p4`` and ``M_TOV`` from the
    named EOS to ``ns_radius``; at the 1.4 Msun anchor it must return
    the tabulated ``R_1p4``, and at ``M_TOV`` the value drops to
    ``0.85 * R_1p4`` per the cubic suppression.
    """
    eos = EOS_MODELS[eos_name]
    R_anchor = float(ns_radius_from_eos(1.4, eos_name))
    R_top = float(ns_radius_from_eos(eos["M_TOV"], eos_name))
    assert R_anchor == pytest.approx(eos["R_1p4"], rel=1e-12)
    assert R_top == pytest.approx(0.85 * eos["R_1p4"], rel=1e-12)


# ---------------------------------------------------------------------------
# mcrit_to_r14: linear interpolation between APR4 and DD2
# ---------------------------------------------------------------------------
def test_mcrit_to_r14_at_apr4_anchor():
    """At ``M_crit = APR4.M_crit``, return APR4 ``R_1p4`` exactly."""
    apr4 = EOS_MODELS["APR4"]
    assert float(mcrit_to_r14(apr4["M_crit"])) == pytest.approx(apr4["R_1p4"], rel=1e-12)


def test_mcrit_to_r14_at_dd2_anchor():
    """At ``M_crit = DD2.M_crit``, return DD2 ``R_1p4`` exactly."""
    dd2 = EOS_MODELS["DD2"]
    assert float(mcrit_to_r14(dd2["M_crit"])) == pytest.approx(dd2["R_1p4"], rel=1e-12)


def test_mcrit_to_r14_midpoint_is_average_of_anchors():
    """Midpoint of (APR4.M_crit, DD2.M_crit) maps to the average of
    ``R_1p4`` (linear interpolation, no curvature)."""
    apr4 = EOS_MODELS["APR4"]
    dd2 = EOS_MODELS["DD2"]
    mc_mid = 0.5 * (apr4["M_crit"] + dd2["M_crit"])
    R_mid = float(mcrit_to_r14(mc_mid))
    expected = 0.5 * (apr4["R_1p4"] + dd2["R_1p4"])
    assert R_mid == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# _compactness: GW170817 canonical + vectorisation
# ---------------------------------------------------------------------------
def test_compactness_gw170817_canonical_value():
    """``C(1.4 Msun, 12 km)`` evaluates to the closed-form value
    ``G * M_sun / (R * c**2) * 1.4`` and lands at ~ 0.172, inside the
    Raaijmakers+ 2021 NICER + GW + KN compactness band for the 1.4
    Msun NS."""
    C = float(_compactness(1.4, 12.0))
    G = 6.674e-11
    c = 2.99792458e8
    Msun = 1.989e30
    expected = G * 1.4 * Msun / (12.0 * 1e3 * c**2)
    assert C == pytest.approx(expected, rel=1e-12)
    assert 0.16 < C < 0.18, C


def test_compactness_vectorisation_parity():
    """Vectorised compactness matches an elementwise scalar loop."""
    M = np.linspace(1.0, 2.1, 12)
    R = np.linspace(10.5, 13.5, 12)
    vec = _compactness(M, R)
    loop = np.array([_compactness(float(m), float(r)) for m, r in zip(M, R)])
    np.testing.assert_allclose(vec, loop, rtol=1e-15)


# ---------------------------------------------------------------------------
# _truncated_double_gauss_cdf: boundary conditions and monotonicity
# ---------------------------------------------------------------------------
def test_truncated_double_gauss_cdf_zero_at_m_min():
    """The renormalised CDF must satisfy ``F(m_min) = 0``."""
    F = float(_truncated_double_gauss_cdf(NS_REMAP_M_MIN, NS_REMAP_M_MIN, M_TOV))
    assert F == pytest.approx(0.0, abs=1e-15)


def test_truncated_double_gauss_cdf_one_at_m_max():
    """The renormalised CDF must satisfy ``F(m_max) = 1``."""
    F = float(_truncated_double_gauss_cdf(M_TOV, NS_REMAP_M_MIN, M_TOV))
    assert F == pytest.approx(1.0, abs=1e-12)


def test_truncated_double_gauss_cdf_monotone_on_grid():
    """``F`` is strictly non-decreasing across ``[m_min, m_max]``."""
    grid = np.linspace(NS_REMAP_M_MIN, M_TOV, 256)
    F = _truncated_double_gauss_cdf(grid, NS_REMAP_M_MIN, M_TOV)
    diffs = np.diff(F)
    assert (diffs >= -1e-12).all()


# ---------------------------------------------------------------------------
# foucart_remnant_mass: clip paths, spin monotonicity, R_NS override,
# bulk-warning aggregation
# ---------------------------------------------------------------------------
def test_foucart_remnant_monotone_in_BH_spin_at_q3():
    """At ``Q = 3, M_NS = 1.35, R_NS = 12 km``, the Foucart (2018)
    remnant mass increases monotonically with ``a_BH`` in the
    calibration range [0, 0.9].  Catches sign-handling regressions in
    the ``r_isco`` plumbing or the bracket.
    """
    chi = np.array([0.3, 0.5, 0.7, 0.9])
    M_rem = np.array(
        [
            float(foucart_remnant_mass(M_BH=4.05, M_NS=1.35, a_BH=float(c), R_NS_km=12.0))
            for c in chi
        ]
    )
    assert (np.diff(M_rem) > 0).all(), M_rem


def test_foucart_remnant_clip_Q_zeros_high_Q_systems():
    """``clip_Q = 7`` zeros a ``Q = 10`` system but leaves a ``Q = 5``
    system untouched.  The Foucart+ 2018 fit is calibrated for Q in
    [1, 7]; ``clip_Q`` is the conservative-results knob."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = foucart_remnant_mass(
            M_BH=np.array([6.75, 13.5]),
            M_NS=np.array([1.35, 1.35]),
            a_BH=0.5,
            R_NS_km=12.0,
            clip_Q=7.0,
        )
    assert out[0] > 0.0
    assert out[1] == 0.0


def test_foucart_remnant_clip_chi_zeros_high_chi_systems():
    """``clip_chi = 0.9`` zeros a ``|chi| = 0.95`` system.

    The Foucart+ 2018 fit is calibrated for chi in [-0.5, 0.9]; the
    Table II data range extends to chi = 0.97 with larger residuals,
    so ``clip_chi`` is the strict-validity knob.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = foucart_remnant_mass(
            M_BH=6.0,
            M_NS=1.35,
            a_BH=np.array([0.5, 0.95]),
            R_NS_km=12.0,
            clip_chi=0.9,
        )
    assert out[0] > 0.0
    assert out[1] == 0.0


def test_foucart_remnant_R_NS_km_override_matches_default_at_1p4_msun():
    """Passing ``R_NS_km = R_1p4_km`` explicitly must match the
    default path at ``M_NS = 1.4`` (where ``ns_radius`` returns
    ``R_1p4_km`` exactly)."""
    args = dict(M_BH=6.75, M_NS=1.4, a_BH=0.5)
    M_default = float(foucart_remnant_mass(R_1p4_km=12.0, **args))
    M_override = float(foucart_remnant_mass(R_NS_km=12.0, **args))
    assert M_override == pytest.approx(M_default, rel=1e-12)


def test_foucart_remnant_bulk_warning_aggregated_per_call():
    """A bulk array call with three Q > 7 entries emits exactly one
    aggregated warning (count reported in the message).  Catches a
    refactor that switches to per-system warning emission."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        foucart_remnant_mass(
            M_BH=np.array([14.0, 15.0, 16.0]),
            M_NS=1.35,
            a_BH=0.5,
            R_NS_km=12.0,
        )
    q_warnings = [item for item in w if "Q > 7" in str(item.message)]
    assert len(q_warnings) == 1
    assert "3 systems" in str(q_warnings[0].message)


# ---------------------------------------------------------------------------
# foucart_disk_mass: invariant disk <= remnant, clip propagation
# ---------------------------------------------------------------------------
def test_foucart_disk_mass_never_exceeds_remnant_mass():
    """``M_disk = max(0, M_rem - M_dyn)`` must satisfy
    ``M_disk <= M_rem`` for any (Q, chi, M_NS) inside the calibration
    range.  Cross-function consistency between ``foucart_remnant_mass``
    and ``bhns_dynamical_ejecta``."""
    rng = np.random.default_rng(0)
    Q = rng.uniform(2.0, 6.0, size=12)
    chi = rng.uniform(0.0, 0.85, size=12)
    M_NS = rng.uniform(1.20, 1.60, size=12)
    M_BH = Q * M_NS
    M_rem = foucart_remnant_mass(M_BH=M_BH, M_NS=M_NS, a_BH=chi, R_NS_km=12.0)
    M_disk = foucart_disk_mass(M_BH=M_BH, M_NS=M_NS, a_BH=chi, R_NS_km=12.0)
    assert (M_disk <= M_rem + 1e-15).all()
    assert (M_disk >= 0.0).all()


def test_foucart_disk_mass_clip_propagation():
    """``clip_Q`` and ``clip_chi`` zero the disk through the same path
    as the remnant (the disk is built from the remnant)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_Q = foucart_disk_mass(
            M_BH=np.array([6.75, 13.5]),
            M_NS=np.array([1.35, 1.35]),
            a_BH=0.5,
            R_NS_km=12.0,
            clip_Q=7.0,
        )
        d_chi = foucart_disk_mass(
            M_BH=6.0,
            M_NS=1.35,
            a_BH=np.array([0.5, 0.95]),
            R_NS_km=12.0,
            clip_chi=0.9,
        )
    assert d_Q[1] == 0.0
    assert d_chi[1] == 0.0


# ---------------------------------------------------------------------------
# bns_disk_mass: symmetry, R override, and the documented KF2020
# sharp transition around R_1.4 ~ 12 to 13 km
# ---------------------------------------------------------------------------
def test_bns_disk_mass_symmetric_in_m1_m2():
    """KF2020 disk mass is built from the lighter NS compactness; the
    helper must therefore be symmetric under ``(M1, M2)`` swap."""
    M1, M2 = 1.46, 1.27
    a = float(bns_disk_mass(M1, M2, R_1p4_km=12.0))
    b = float(bns_disk_mass(M2, M1, R_1p4_km=12.0))
    assert a == pytest.approx(b, rel=1e-12)


def test_bns_disk_mass_explicit_radii_override_matches_default_at_1p4_msun():
    """At ``M1 >= M2 >= 1.4 Msun``, passing ``R1_km`` / ``R2_km``
    explicitly with ``ns_radius`` values matches the default path."""
    M1, M2 = 1.6, 1.5
    default = float(bns_disk_mass(M1, M2, R_1p4_km=12.0))
    R1 = float(ns_radius(M1, R_1p4_km=12.0))
    R2 = float(ns_radius(M2, R_1p4_km=12.0))
    explicit = float(bns_disk_mass(M1, M2, R1_km=R1, R2_km=R2))
    assert explicit == pytest.approx(default, rel=1e-12)


def test_bns_disk_mass_hits_floor_at_compact_light_ns():
    """At ``R_1.4 = 11 km`` (compact), the KF2020 ``a * C_1 + c``
    term is negative and the floor at ``5e-4 * M_b_tot`` kicks in.
    Docstring claim verified numerically for a GW170817-like binary.
    """
    M1, M2 = 1.46, 1.27
    M_disk = float(bns_disk_mass(M1, M2, R_1p4_km=11.0))
    assert 1e-3 < M_disk < 3e-3


def test_bns_disk_mass_sharp_transition_between_r14_12_and_13():
    """The KF2020 BNS disk fit jumps roughly two orders of magnitude
    as ``R_1.4`` crosses ~ 12.1 km at a GW170817-like binary; the
    warning in the docstring is locked in here."""
    M1, M2 = 1.46, 1.27
    low = float(bns_disk_mass(M1, M2, R_1p4_km=12.0))
    high = float(bns_disk_mass(M1, M2, R_1p4_km=13.0))
    assert low < 5e-3
    assert high > 0.1
    assert high / low > 100.0


# ---------------------------------------------------------------------------
# bns_dynamical_ejecta: symmetry, GW170817-band
# ---------------------------------------------------------------------------
def test_bns_dynamical_ejecta_symmetric_in_m1_m2():
    """KF2020 BNS dynamical ejecta is symmetric under ``(M1, M2)``
    swap (the two ``term1 / term2`` pieces switch roles)."""
    M1, M2 = 1.46, 1.27
    a = float(bns_dynamical_ejecta(M1, M2, R_1p4_km=12.0))
    b = float(bns_dynamical_ejecta(M2, M1, R_1p4_km=12.0))
    assert a == pytest.approx(b, rel=1e-12)


def test_bns_dynamical_ejecta_gw170817_finite_and_in_kf2020_band():
    """GW170817-like input (M1 = 1.46, M2 = 1.27, R_1p4 = 12 km) lies
    in the KF2020 sigma band ~ 0.003 to 0.010 Msun (docstring) and is
    well below the AT2017gfo total ejecta ~ 0.05 to 0.08 Msun
    (Rastinejad+ 2025).  Bounded sanity test rather than a single
    pinned value because the KF2020 residual is large."""
    d = float(bns_dynamical_ejecta(1.46, 1.27, R_1p4_km=12.0))
    assert np.isfinite(d)
    assert 1e-3 < d < 2e-2


# ---------------------------------------------------------------------------
# gottlieb25_eq11 and hmns_wind_ejecta
# ---------------------------------------------------------------------------
def test_gottlieb25_eq11_canonical_normalisation():
    """Eq. 11 normalisation at the reference point
    ``(T50 = 1 s, E_iso = 2e51 erg, alpha = 2, f_inv = 1)``
    returns exactly ``1e-3 Msun`` (the prefactor in the equation)."""
    M_ej = float(gottlieb25_eq11(T50=1.0, E_iso=2e51, alpha=2.0, f_inv=1.0))
    assert M_ej == pytest.approx(1e-3, rel=1e-12)


def test_gottlieb25_eq11_alpha_scaling_at_T50_4s():
    """At ``T50 = 4 s, alpha = 1.5``, the ``(T50)^(alpha - 1)`` factor
    is ``4 ** 0.5 = 2``; combined with the reference normalisation
    gives ``2e-3 Msun``.  Matches the inline ``_selftest_gottlieb25``."""
    M_ej = float(gottlieb25_eq11(T50=4.0, E_iso=2e51, alpha=1.5, f_inv=1.0))
    assert M_ej == pytest.approx(2e-3, rel=1e-12)


def test_gottlieb25_eq11_E_iso_linear_scaling():
    """``gottlieb25_eq11`` is linear in ``E_iso`` at fixed
    ``(T50, alpha, f_inv)``: doubling ``E_iso`` doubles ``M_ej``."""
    base = float(gottlieb25_eq11(T50=1.0, E_iso=2e51, alpha=2.0, f_inv=1.0))
    doubled = float(gottlieb25_eq11(T50=1.0, E_iso=4e51, alpha=2.0, f_inv=1.0))
    assert doubled == pytest.approx(2.0 * base, rel=1e-12)


def test_hmns_wind_ejecta_linear_in_disk_mass():
    """HMNS-engine ejecta is ``f_wind * M_d``.  Linearity must hold
    for both scalar and array inputs; the default
    ``f_wind = GOTTLIEB25_WIND_FRAC = 0.3``."""
    assert float(hmns_wind_ejecta(0.05)) == pytest.approx(GOTTLIEB25_WIND_FRAC * 0.05, rel=1e-12)
    arr = hmns_wind_ejecta(np.array([0.01, 0.05, 0.1]))
    np.testing.assert_allclose(arr, GOTTLIEB25_WIND_FRAC * np.array([0.01, 0.05, 0.1]), rtol=1e-12)
    assert float(hmns_wind_ejecta(0.05, wind_frac=0.25)) == pytest.approx(0.0125, rel=1e-12)


# ---------------------------------------------------------------------------
# effective_aligned_spin: edge angles and clipping at zero
# ---------------------------------------------------------------------------
def test_effective_aligned_spin_theta_zero_returns_a_BH():
    """Perfectly aligned (``theta = 0``) returns the full ``a_BH``."""
    assert float(effective_aligned_spin(0.7, 0.0)) == pytest.approx(0.7, rel=1e-12)


def test_effective_aligned_spin_theta_pi_over_two_returns_zero():
    """Perpendicular kick (``theta = pi/2``) returns zero
    (cos(pi/2) ~ 0).  Floating-point residual stays at machine
    precision."""
    assert float(effective_aligned_spin(0.7, np.pi / 2.0)) == pytest.approx(0.0, abs=5e-16)


def test_effective_aligned_spin_theta_pi_clipped_to_zero():
    """Anti-aligned (``theta = pi``) would give ``-a_BH``; the helper
    clips at zero to avoid a negative aligned-spin component being
    forwarded into Foucart 2018 (which is calibrated for ``chi >=
    -0.5`` but conventionally fed an aligned magnitude)."""
    assert float(effective_aligned_spin(0.7, np.pi)) == 0.0


# ---------------------------------------------------------------------------
# Module-level __getattr__ deprecation hook
# ---------------------------------------------------------------------------
def test_MEAN_MASS_EVOLVED_emits_deprecation_warning_and_returns_value():
    """``grb_physics.MEAN_MASS_EVOLVED`` is the deprecated attribute
    name for the cached calibration sentinel; access must emit a
    ``DeprecationWarning`` and return ``_MEAN_MASS_EVOLVED_VALUE``."""
    import grb_physics as gp

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = gp.MEAN_MASS_EVOLVED
    deprecation = [item for item in w if issubclass(item.category, DeprecationWarning)]
    assert len(deprecation) >= 1
    assert value == _MEAN_MASS_EVOLVED_VALUE


def test_unknown_attribute_raises_attribute_error():
    """Any other unknown attribute must raise ``AttributeError`` from
    the ``__getattr__`` fallback (not silently return None)."""
    import grb_physics as gp

    with pytest.raises(AttributeError):
        _ = gp.this_attribute_does_not_exist


# ---------------------------------------------------------------------------
# NaN propagation
# ---------------------------------------------------------------------------
def test_r_isco_nan_input_returns_nan():
    """``r_isco(np.nan)`` must propagate NaN through the
    ``(1 - a**2)**(1/3)`` chain rather than silently returning a
    real number."""
    assert np.isnan(float(r_isco(np.nan)))


def test_ns_radius_nan_input_documented_quirk_returns_plateau_value():
    """``ns_radius(np.nan)`` does *not* propagate NaN: ``np.nan >= 1.4``
    is False, so the function falls through the plateau branch and
    returns ``R_1p4_km``.  The quirk is documented here as a
    regression sentinel rather than fixed inline; callers that
    receive NaN-tainted COMPAS masses must guard upstream of
    ``ns_radius`` (the COMPAS loaders already do, so the live
    pipeline never hits this path).
    """
    R = ns_radius(np.array([np.nan, 1.4]), R_1p4_km=12.0)
    assert R[0] == pytest.approx(12.0, rel=1e-12)
    assert R[1] == pytest.approx(12.0, rel=1e-12)


def test_foucart_remnant_nan_M_BH_propagates_nan():
    """A NaN ``M_BH`` propagates NaN through the Q, eta, bracket chain."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = foucart_remnant_mass(
            M_BH=np.array([np.nan, 6.75]),
            M_NS=np.array([1.35, 1.35]),
            a_BH=0.5,
            R_NS_km=12.0,
        )
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


# ---------------------------------------------------------------------------
# Vectorisation parity: vectorised vs Python loop for the four mass helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vectorisation_sample():
    """Length-8 BHNS-like sample inside the Foucart 2018 calibration
    range.  Drawn deterministically so vectorisation parity tests are
    reproducible."""
    rng = np.random.default_rng(2026)
    n = 8
    M_NS = rng.uniform(1.20, 1.60, size=n)
    Q = rng.uniform(2.0, 6.0, size=n)
    M_BH = Q * M_NS
    chi = rng.uniform(0.0, 0.85, size=n)
    return M_BH, M_NS, chi


def test_foucart_remnant_vectorisation_parity(vectorisation_sample):
    """Vectorised ``foucart_remnant_mass`` matches an elementwise loop."""
    M_BH, M_NS, chi = vectorisation_sample
    vec = foucart_remnant_mass(M_BH=M_BH, M_NS=M_NS, a_BH=chi, R_NS_km=12.0)
    loop = np.array(
        [
            float(foucart_remnant_mass(M_BH=float(b), M_NS=float(n), a_BH=float(c), R_NS_km=12.0))
            for b, n, c in zip(M_BH, M_NS, chi)
        ]
    )
    np.testing.assert_allclose(vec, loop, rtol=1e-12, atol=0)


def test_bhns_dynamical_ejecta_vectorisation_parity(vectorisation_sample):
    """Vectorised KF2020 BHNS dyn ejecta matches an elementwise loop."""
    M_BH, M_NS, chi = vectorisation_sample
    vec = bhns_dynamical_ejecta(M_BH=M_BH, M_NS=M_NS, a_BH=chi, R_NS_km=12.0)
    loop = np.array(
        [
            float(bhns_dynamical_ejecta(M_BH=float(b), M_NS=float(n), a_BH=float(c), R_NS_km=12.0))
            for b, n, c in zip(M_BH, M_NS, chi)
        ]
    )
    np.testing.assert_allclose(vec, loop, rtol=1e-12, atol=0)


def test_bns_disk_mass_vectorisation_parity():
    """Vectorised KF2020 BNS disk mass matches an elementwise loop."""
    rng = np.random.default_rng(7)
    n = 8
    M1 = rng.uniform(1.30, 1.80, size=n)
    M2 = rng.uniform(1.15, 1.30, size=n)
    vec = bns_disk_mass(M1, M2, R_1p4_km=12.5)
    loop = np.array(
        [float(bns_disk_mass(float(a), float(b), R_1p4_km=12.5)) for a, b in zip(M1, M2)]
    )
    np.testing.assert_allclose(vec, loop, rtol=1e-12, atol=0)


def test_bns_dynamical_ejecta_vectorisation_parity():
    """Vectorised KF2020 BNS dyn ejecta matches an elementwise loop."""
    rng = np.random.default_rng(11)
    n = 8
    M1 = rng.uniform(1.30, 1.80, size=n)
    M2 = rng.uniform(1.15, 1.30, size=n)
    vec = bns_dynamical_ejecta(M1, M2, R_1p4_km=12.0)
    loop = np.array(
        [float(bns_dynamical_ejecta(float(a), float(b), R_1p4_km=12.0)) for a, b in zip(M1, M2)]
    )
    np.testing.assert_allclose(vec, loop, rtol=1e-12, atol=0)
