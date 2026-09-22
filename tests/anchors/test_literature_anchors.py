"""Strict literature-anchor tests for `grb_*.py` modules.

Each test pins a numerical constant or a function output to the
corresponding paper in ``papers/``.  Policy (``strict_paper``): any
in-code value that lies outside the paper-quoted central plus
uncertainty band fails this suite even when the docstring rationalizes
it.  Failures here are not regressions; they are scientific
discrepancies that should be resolved in a separate change.

Each test docstring states paper, equation or table number, and the
exact number being pinned, so the failure message is actionable.

Conventions:
- Pure-function tests; no ``data/`` access.  All run in well under one
  second per test.
- Where a paper quotes a 1-sigma band, the test asserts membership.
  Where it quotes a range (e.g. Bauswein 2013 ``k = M_thresh / M_TOV in
  [1.30, 1.70]``), the test asserts inclusion in that closed range.
- Where the project explicitly documents a heuristic that does not
  come from the cited paper (``HMNS_FACTOR_DEFAULT = 1.2``,
  ``MISALIGNMENT_SYSTEMATIC_FACTOR = 0.5``), the test pins the
  heuristic to the supporting-paper rationale band (e.g. Margalit and
  Metzger 2017, Kawaguchi 2015) since that is the closest thing to a
  paper-quoted bound.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────
# Bauswein, Baumgarte and Janka (2013) PRL 111, 131101 [papers/Bauswein_2013.pdf]
# Bauswein, Bastian, Blaschke et al. (2020) [papers/Bauswein_2020.pdf]
# Koppel, Bovard and Rezzolla (2019) ApJL 872, L16 [papers/Koppel_2019.pdf]
# ─────────────────────────────────────────────────────────────────────
def test_prompt_collapse_k_ratio_in_bauswein_2020_surveyed_range():
    """`K_THRESH_DEFAULT` lies inside the k = M_thres / M_max range Bauswein (2020) surveys.

    This test used to be a strict xfail against the [1.30, 1.70] band quoted
    in the Bauswein, Baumgarte and Janka (2013) PRL 111, 131101 abstract,
    with the 1.27 fiducial recorded as an unreconciled "Gottlieb-vs-Bauswein
    numerical gap" needing a science change to close.  It does not need one.
    That band came from a survey of 12 equations of state.  Bauswein et al.
    (2020, arXiv:2004.00846) Table I extends the same measurement to 40, and
    its 23-EOS base sample spans k in [1.259, 1.514]: five of those 23 sit
    below 1.30 (eosAU 1.259, WFF1 1.263, eosUU 1.291, APR 1.292, WFF2 1.292).

    ``K_THRESH_DEFAULT = 1.27`` therefore sits inside the modern surveyed
    range, between WFF1 and eosUU, at the soft end where Gottlieb (2023)
    intends it: the fiducial is chosen so that ``K * M_TOV ~ 2.8 Msun =
    M_CRIT_BNS``.  The constant is unchanged; only the band it is tested
    against was stale.

    The upper edge is the base-sample maximum rounded out to 1.60, which also
    admits the excluded-hadronic models (up to 1.581).  Same band as
    ``test_eos_models_M_crit_over_M_TOV_in_bauswein_band``, and for the same
    reason: it comes from the table the values come from.
    """
    from grb_physics import K_THRESH_DEFAULT

    assert 1.25 <= K_THRESH_DEFAULT <= 1.60, (
        f"K_THRESH_DEFAULT = {K_THRESH_DEFAULT} is outside the k range "
        f"Bauswein et al. (2020) Table I surveys, [1.259, 1.514] for the "
        f"base sample, rounded to [1.25, 1.60]."
    )


@pytest.mark.parametrize(
    "eos_name, k_min, k_max",
    [
        # k = M_thres(q = 1) / M_max per EOS, Bauswein et al. (2020) Table I:
        # APR 1.292, SFHO 1.398, LS220 1.458, DD2 1.375.  Computed on the
        # project's own M_TOV these become 1.284, 1.396, 1.458, 1.374.
        ("APR4", 1.25, 1.60),
        ("SFHo", 1.25, 1.60),
        ("LS220", 1.25, 1.60),
        ("DD2", 1.25, 1.60),
    ],
)
def test_eos_models_M_crit_over_M_TOV_in_bauswein_band(eos_name, k_min, k_max):
    """`EOS_MODELS[eos]['M_crit'] / M_TOV` lies in the Bauswein (2020) k range.

    The band is the range k spans across the 23 EOS of the Bauswein et al.
    (2020) Table I base sample, [1.259, 1.514], rounded outward to
    [1.25, 1.60] so that the excluded-hadronic models (up to 1.581) also fit.
    It is taken from the same table the ``M_crit`` values are, which is the
    point: a band and the values it brackets should come from one source.

    An earlier version of this test quoted the Bauswein (2013) Table I k
    values as 1.26 for SFHo and 1.33 for LS220.  Table I gives 1.432 and
    1.495; the project constants had been transcribed with M_crit low by
    0.35 Msun, and the acceptance band had then been widened from the
    canonical [1.30, 1.70] to [1.20, 1.70] to admit them, attributed to a
    Bauswein (2020) re-analysis that does not support it (its M_thres range
    starts near 2.7 Msun).  Both the constants and the band now come from
    Bauswein (2020) Table I directly.
    """
    from grb_physics import EOS_MODELS

    eos = EOS_MODELS[eos_name]
    k = eos["M_crit"] / eos["M_TOV"]
    assert k_min <= k <= k_max, (
        f"EOS_MODELS[{eos_name!r}] gives M_crit/M_TOV = {k:.3f} = "
        f"{eos['M_crit']}/{eos['M_TOV']}; Bauswein (2013) Table I "
        f"survey requires k in [{k_min}, {k_max}]."
    )


# ─────────────────────────────────────────────────────────────────────
# Read, Lackey, Owen, Friedman (2009) PRD 79, 124032 [papers/Read_2009.pdf]
# EOS table III: NS radius at 1.4 Msun and maximum mass.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "eos_name, R14_central, R14_tol_km, MTOV_central, MTOV_tol_Msun",
    [
        # Per-EOS published values.  Only APR4 comes from Read (2009); the
        # other three postdate it or were never in it, which is why the test
        # is named for the quantity rather than for one paper.
        # APR4 (Akmal, Pandharipande and Ravenhall): Read (2009) Table III,
        # R_1.4 ~ 11.4 km, M_TOV ~ 2.20 Msun.  Project value 11.1 km.
        ("APR4", 11.30, 0.50, 2.20, 0.10),
        # SFHo (Steiner, Hempel and Fischer 2013): R_1.4 ~ 11.9 km,
        # M_TOV ~ 2.06 Msun.  Bauswein (2020) Table I gives M_max = 2.056 and
        # R_1.6 = 11.751 km, consistent with a slightly larger R_1.4.
        ("SFHo", 11.90, 0.50, 2.06, 0.10),
        # LS220 (Lattimer and Swesty 1991): widely tabulated R_1.4 ~ 12.7 km,
        # M_TOV ~ 2.04.  Bauswein (2020) Table I: M_max = 2.041, R_1.6 = 12.478.
        ("LS220", 12.70, 0.60, 2.04, 0.10),
        # DD2 (Typel et al. 2010): R_1.4 ~ 13.2 km, M_TOV ~ 2.42 Msun.
        # Bauswein (2020) Table I: M_max = 2.419, R_1.6 = 13.247.
        ("DD2", 13.20, 0.50, 2.42, 0.10),
    ],
)
def test_eos_models_R14_and_MTOV_match_published_values(
    eos_name, R14_central, R14_tol_km, MTOV_central, MTOV_tol_Msun
):
    """Pin of ``EOS_MODELS[eos]`` ``R_1.4`` and ``M_TOV`` to published values.

    One EOS per parametrisation, each from its own source, because no
    single paper covers all four: APR4 from Read, Lackey, Owen and
    Friedman (2009) PRD 79, 124032 Table III; SFHo from Steiner, Hempel
    and Fischer (2013); LS220 from Lattimer and Swesty (1991); DD2 from
    Typel et al. (2010).  Only APR4 is in Read (2009) at all, which is why
    this test is no longer named for that paper.

    Bauswein et al. (2020) Table I is a useful cross-check on ``M_TOV``
    since it reports ``M_max`` for all four from one set of simulations
    (2.187, 2.056, 2.041, 2.419); the project values agree to under one
    percent.  It is not used for ``R_1.4``, because it tabulates R_1.6.

    The test ranges are the published central value plus its
    nuclear-physics 1-sigma uncertainty (~0.5 km for ``R_1.4``, ~0.1 Msun
    for ``M_TOV``).  Drift outside them signals either a transcription
    error in ``EOS_MODELS`` or a change of canonical reference.
    """
    from grb_physics import EOS_MODELS

    eos = EOS_MODELS[eos_name]
    assert abs(eos["R_1p4"] - R14_central) <= R14_tol_km, (
        f"EOS_MODELS[{eos_name!r}]['R_1p4'] = {eos['R_1p4']} km is "
        f"more than {R14_tol_km} km from Read (2009)/Bauswein (2013) "
        f"central R_1.4 = {R14_central} km."
    )
    assert abs(eos["M_TOV"] - MTOV_central) <= MTOV_tol_Msun, (
        f"EOS_MODELS[{eos_name!r}]['M_TOV'] = {eos['M_TOV']} Msun is "
        f"more than {MTOV_tol_Msun} Msun from Read (2009)/Bauswein (2013) "
        f"central M_TOV = {MTOV_central} Msun."
    )


# ─────────────────────────────────────────────────────────────────────
# Read et al. (2009) PRD 79, 124032 [papers/Read_2009.pdf]
# `mcrit_to_r14`: linear interpolation between APR4 and DD2 anchors.
# ─────────────────────────────────────────────────────────────────────
def test_mcrit_to_r14_recovers_apr4_and_dd2_read_2009_anchors():
    """`mcrit_to_r14(M_crit)` must reproduce R_1.4 at the APR4 and DD2 anchors.

    The helper is a two-point linear interpolation between
    Read et al. (2009) PRD 79, 124032 Table III entries:

      APR4: (M_crit, R_1.4) = (2.46, 11.1) km
      DD2:  (M_crit, R_1.4) = (3.35, 13.2) km

    (The `EOS_MODELS["APR4"]["M_crit"]` constant in the project is
    2.88, not the Bauswein 2.46; the helper interpolates on whichever
    M_crit values populate `EOS_MODELS`, so this test pins the
    interpolation against the values *currently* in that dict rather
    than against the Bauswein Table I numbers directly.  The latter
    are pinned separately by `test_bauswein_2013_prompt_collapse_k_ratio_in_published_band`.)
    """
    from grb_physics import EOS_MODELS, mcrit_to_r14

    apr4 = EOS_MODELS["APR4"]
    dd2 = EOS_MODELS["DD2"]
    assert float(mcrit_to_r14(apr4["M_crit"])) == pytest.approx(apr4["R_1p4"], rel=1e-12)
    assert float(mcrit_to_r14(dd2["M_crit"])) == pytest.approx(dd2["R_1p4"], rel=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Raaijmakers et al. (2021) ApJL 918, L29 [papers/Raaijmakers_2021.pdf]
# Combined NICER + GW + KN posterior on M_TOV.
# ─────────────────────────────────────────────────────────────────────
def test_raaijmakers_2021_M_TOV_inside_combined_posterior():
    """`M_TOV = 2.2` Msun must lie inside the Raaijmakers (2021) posterior band.

    Raaijmakers et al. (2021, arXiv:2105.06981) combined NICER X-ray
    timing of PSR J0740+6620 with GW170817 and AT2017gfo kilonova
    constraints to obtain the maximum non-rotating NS mass:

      PP (piecewise polytrope): M_TOV = 2.23 +0.14 / -0.23 Msun
      CS (speed of sound):      M_TOV = 2.11 +0.29 / -0.16 Msun

    The project's fiducial `M_TOV = 2.2` lies inside both posteriors.
    Allowed band conservative wrt the looser CS posterior: [1.95, 2.40].
    """
    from grb_physics import M_TOV

    assert 1.95 <= M_TOV <= 2.40, (
        f"M_TOV = {M_TOV} Msun is outside the Raaijmakers (2021) "
        f"NICER + GW + KN combined posterior band [1.95, 2.40] Msun "
        f"(PP 2.23 +0.14/-0.23, CS 2.11 +0.29/-0.16)."
    )


def test_ns_radius_heuristic_drops_by_15_percent_from_1p4_to_M_TOV():
    """`ns_radius(M_TOV) / ns_radius(1.4)` must equal 0.85.

    The phenomenological R(M) sequence in `ns_radius` applies a cubic
    suppression `R = R_1.4 * (1 - 0.15 * x^3)` with
    `x = (M - 1.4) / (M_TOV - 1.4)` saturated at `x = 1`, so the
    drop from the 1.4 Msun anchor to M_TOV is exactly 15 percent.
    The 0.15 amplitude is a CODE HEURISTIC consistent with the EOS
    spread inside `EOS_MODELS` (APR4 -> DD2 R_1.4 varies by ~ 20
    percent, and the cubic suppression sits inside that envelope)
    and with the Raaijmakers et al. (2021) NICER + GW + KN posterior
    on R(M) at the high-mass end.
    """
    from grb_physics import M_TOV, ns_radius

    R_anchor = float(ns_radius(1.4, R_1p4_km=12.0))
    R_top = float(ns_radius(M_TOV, R_1p4_km=12.0))
    ratio = R_top / R_anchor
    assert ratio == pytest.approx(0.85, rel=1e-12), (
        f"ns_radius R(M_TOV) / R(1.4) = {ratio} deviates from the "
        f"documented 0.15 cubic-suppression coefficient (-> 0.85)."
    )


# ─────────────────────────────────────────────────────────────────────
# Lattimer and Prakash (2001) ApJ 550, 426 [papers/Lattimer_2000.pdf]
# Gao, Hu, Lu, Tian, Lu (2020) [papers/Gao_2020.pdf]
# NS baryon mass: M_b = M_g + 0.080 * M_g^2 (L&P 2001 Eq. 56).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("M_g", [1.20, 1.30, 1.35, 1.40, 1.60, 1.80, 2.00])
def test_ns_baryon_mass_matches_lattimer_prakash_eq56(M_g):
    """`ns_baryon_mass(M_g)` must equal M_g + 0.080 * M_g^2.

    Lattimer and Prakash (2001) ApJ 550, 426, Eq. (56) (also Lattimer
    2012 ARNPS 62, 485) give the mass-dependent NS baryon-to-
    gravitational mass relation:

        M_b ~ M_g + 0.080 * M_g^2  [Msun]

    valid to a few-percent accuracy across the 1.0 to 2.0 Msun NS-mass
    range.  The 0.080 coefficient is the L&P 2001 fit value; Gao et al.
    (2020) confirm it sits within the multi-EOS scatter.  This test
    pins `ns_baryon_mass` to the formula.
    """
    from grb_physics import ns_baryon_mass

    expected = M_g + 0.080 * M_g**2
    got = float(ns_baryon_mass(M_g))
    assert got == pytest.approx(expected, rel=1e-12), (
        f"ns_baryon_mass({M_g}) = {got} != Lattimer-Prakash (2001) "
        f"Eq. (56) value {M_g} + 0.080 * {M_g}^2 = {expected}.  "
        f"The 0.080 coefficient is paper-pinned; do not change without "
        f"updating the citation."
    )


# ─────────────────────────────────────────────────────────────────────
# Alsing, Silva and Berti (2018) MNRAS 478, 1377 [arXiv:1810.03548]
# Galactic NS mass distribution: two-Gaussian fit, Table 3.
# ─────────────────────────────────────────────────────────────────────
def test_alsing_2018_double_gaussian_constants_match_table3():
    """`NS_REMAP_*` constants must match Alsing (2018) Table 3 two-Gaussian fit.

    Alsing, Silva and Berti (2018) MNRAS 478, 1377 (arXiv:1810.03548)
    Table 3 fit a two-Gaussian model to the Galactic NS mass
    distribution:

      Component 1 (recycled + slow pulsars):
        weight w_1 ~ 0.66, mu_1 = 1.34 +/- 0.02 Msun, sigma_1 = 0.07
      Component 2 (high-mass tail, J0740+6620, J1614-2230):
        weight w_2 ~ 0.34, mu_2 = 1.80 +/- 0.04 Msun, sigma_2 = 0.21

    `grb_physics.remap_ns_masses_double_gaussian` uses these values
    directly; the test pins them to the paper to within 0.02 Msun for
    centers, 0.05 Msun for widths, and 0.05 for weights.
    """
    from grb_physics import (
        NS_REMAP_M_MIN,
        NS_REMAP_MU1,
        NS_REMAP_MU2,
        NS_REMAP_SIG1,
        NS_REMAP_SIG2,
        NS_REMAP_W1,
        NS_REMAP_W2,
    )

    assert abs(NS_REMAP_W1 - 0.66) <= 0.05, NS_REMAP_W1
    assert abs(NS_REMAP_MU1 - 1.34) <= 0.02, NS_REMAP_MU1
    assert abs(NS_REMAP_SIG1 - 0.07) <= 0.05, NS_REMAP_SIG1
    assert abs(NS_REMAP_W2 - 0.34) <= 0.05, NS_REMAP_W2
    assert abs(NS_REMAP_MU2 - 1.80) <= 0.04, NS_REMAP_MU2
    assert abs(NS_REMAP_SIG2 - 0.21) <= 0.05, NS_REMAP_SIG2
    # Weights should sum to 1.
    assert NS_REMAP_W1 + NS_REMAP_W2 == pytest.approx(1.0, rel=1e-6)
    # Lower truncation must be below the lower-component peak.
    assert NS_REMAP_M_MIN < NS_REMAP_MU1


# ─────────────────────────────────────────────────────────────────────
# Foucart, Hinderer and Nissanke (2018) PRD 98, 081501
# [papers/Foucart_2018.pdf]
# Eq. (4) coefficients and validity ranges.
# ─────────────────────────────────────────────────────────────────────
def test_foucart_2018_eq4_coefficients_match_paper():
    """`foucart_remnant_mass` must use Eq. (6) coefficients (0.406, 0.139, 0.255, 1.761).

    Foucart, Hinderer and Nissanke (2018) PRD 98, 081501, Eq. (6) give

        (alpha, beta, gamma, delta) = (0.406, 0.139, 0.255, 1.761)

    as the rms-minimizing fit to 75 NR simulations spanning Q in [1, 7],
    chi_BH in [-0.5, 0.97], C_NS in [0.13, 0.182].  The
    `test_foucart_remnant_matches_eq4_by_hand` in `tests/unit/test_physics.py`
    already verifies the Foucart formula for one canonical input; this
    test pins the coefficients themselves by reading them from the
    function via two well-chosen probes that constrain the four
    parameters jointly (Q=2 and Q=5 at the same C_NS, chi_BH).
    """
    from grb_physics import _compactness, foucart_remnant_mass, ns_baryon_mass, r_isco

    def expected(Q, chi, M_NS, R_km, alpha=0.406, beta=0.139, gamma=0.255, delta=1.761):
        M_BH = Q * M_NS
        C_NS = float(_compactness(M_NS, R_km))
        eta = M_NS * M_BH / (M_NS + M_BH) ** 2
        R_hat = float(r_isco(chi))
        bracket = alpha * (1 - 2 * C_NS) / eta ** (1 / 3) - beta * R_hat * C_NS / eta + gamma
        M_b = float(ns_baryon_mass(M_NS))
        return max(0.0, bracket) ** delta * M_b

    M_NS = 1.35
    R_km = 12.0
    chi = 0.5
    for Q in (2.0, 5.0):
        got = float(foucart_remnant_mass(M_BH=Q * M_NS, M_NS=M_NS, a_BH=chi, R_NS_km=R_km))
        ref = expected(Q, chi, M_NS, R_km)
        assert got == pytest.approx(ref, rel=1e-10), (
            f"foucart_remnant_mass(Q={Q}, chi={chi}) drifted from "
            f"Foucart (2018) Eq. (6) coefficients (0.406, 0.139, "
            f"0.255, 1.761): got {got}, expected {ref}."
        )


def test_foucart_2018_validity_ranges_documented_warning_thresholds():
    """`foucart_remnant_mass` must warn at Q > 7 and |chi| > 0.9.

    Foucart (2018) Discussion: the average relative error in the
    remnant-mass prediction is ~15 percent for Q in [1, 7], chi_BH in
    [-0.5, 0.9], and ``M_rem <= 0.3 M_b^NS``.  The module emits a
    bulk-aggregated warning when either the Q or chi bound is
    exceeded.  The test pins the threshold values to the paper.
    """
    from grb_physics import foucart_remnant_mass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        foucart_remnant_mass(M_BH=8.0 * 1.35, M_NS=1.35, a_BH=0.5, R_NS_km=12.0)
    assert any("Q > 7" in str(item.message) for item in w), [str(i.message) for i in w]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        foucart_remnant_mass(M_BH=5.0 * 1.35, M_NS=1.35, a_BH=0.95, R_NS_km=12.0)
    assert any("|chi_BH| > 0.9" in str(item.message) for item in w), [str(i.message) for i in w]


def test_foucart_2018_qualitative_low_Q_below_FF12():
    """Low-Q (Q ~ 1.2) Foucart 2018 prediction must be below the FF12 prediction.

    Foucart (2018) Sec. III: the new model predicts a substantially
    smaller `M_rem` for nearly equal-mass NSBH mergers compared to
    Foucart and Faber (2012, FF12).  The ``eta = Q/(1+Q)^2``
    substitution in Eq. (4) is what produces this; the test verifies
    the qualitative trend by comparing the Foucart 2018 model output at
    Q = 1.2 against a hand-coded FF12-style prediction (without the eta
    substitution).
    """
    from grb_physics import _compactness, foucart_remnant_mass, ns_baryon_mass, r_isco

    M_NS = 1.35
    R_km = 11.5
    chi = 0.0
    Q = 1.2
    M_BH = Q * M_NS

    new_model = float(foucart_remnant_mass(M_BH=M_BH, M_NS=M_NS, a_BH=chi, R_NS_km=R_km))

    alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761
    C_NS = float(_compactness(M_NS, R_km))
    R_hat = float(r_isco(chi))
    bracket_ff12 = alpha * (1 - 2 * C_NS) * Q ** (1 / 3) - beta * R_hat * C_NS * Q + gamma
    M_b = float(ns_baryon_mass(M_NS))
    ff12_style = max(0.0, bracket_ff12) ** delta * M_b

    assert new_model < ff12_style, (
        f"Foucart (2018) prediction at Q={Q}, chi={chi} ({new_model:.4f}) "
        f"is not below FF12-style prediction ({ff12_style:.4f}); the eta "
        f"substitution in Eq. (4) is supposed to suppress M_rem at low Q."
    )


# ─────────────────────────────────────────────────────────────────────
# Kruger and Foucart (2020) PRD 101, 103002 [papers/Kruger_2020.pdf]
# Eqs. (4), (6), (9): BNS disk mass, BNS dyn ejecta, BHNS dyn ejecta.
# ─────────────────────────────────────────────────────────────────────
def test_kruger_foucart_2020_bhns_dyn_ejecta_eq9_coefficients_by_hand():
    """`bhns_dynamical_ejecta` must reproduce KF2020 Eq. (9) elementwise.

    Kruger and Foucart (2020) PRD 101, 103002 Table I (BHNS fit):

        a1 = 0.007116, a2 = 0.001436, a4 = -0.02762,
        n1 = 0.8636, n2 = 1.6840.

    The disk-disrupting bracket is

        a1 * Q**n1 * (1 - 2*C_NS) / C_NS - a2 * Q**n2 * R_hat + a4

    and the unbound ejecta is `max(0, bracket) * M_b^NS`.
    """
    from grb_physics import _compactness, bhns_dynamical_ejecta, ns_baryon_mass, r_isco

    M_NS = 1.35
    M_BH = 6.75
    a_BH = 0.5
    R_km = 12.0
    Q = M_BH / M_NS
    C_NS = float(_compactness(M_NS, R_km))
    R_hat = float(r_isco(a_BH))
    M_b = float(ns_baryon_mass(M_NS))

    a1, a2, a4 = 0.007116, 0.001436, -0.02762
    n1, n2 = 0.8636, 1.6840
    bracket = a1 * Q**n1 * (1 - 2 * C_NS) / C_NS - a2 * Q**n2 * R_hat + a4
    expected = max(0.0, bracket) * M_b

    got = float(bhns_dynamical_ejecta(M_BH=M_BH, M_NS=M_NS, a_BH=a_BH, R_NS_km=R_km))
    assert got == pytest.approx(expected, rel=1e-10), (
        f"bhns_dynamical_ejecta drifted from KF2020 Eq. (9) Table I "
        f"coefficients: got {got}, expected {expected}."
    )


def test_kruger_foucart_2020_bns_disk_eq4_coefficients_by_hand():
    """`bns_disk_mass` must reproduce KF2020 Eq. (4) elementwise.

    Kruger and Foucart (2020) Table I (BNS disk fit):

        a = -8.1580, c = 1.2695,
        M_disk / M_b^tot = max(5e-4, a * C_1 + c)

    where C_1 is the compactness of the lighter NS and M_b^tot is the
    sum of component baryon masses.
    """
    from grb_physics import _compactness, bns_disk_mass, ns_baryon_mass, ns_radius

    M1, M2 = 1.46, 1.27
    R_1p4 = 13.0
    R1 = float(ns_radius(M1, R_1p4_km=R_1p4))  # noqa: F841 (literature setup)
    R2 = float(ns_radius(M2, R_1p4_km=R_1p4))
    C1 = float(_compactness(M2, R2))  # lighter NS, Kruger-Foucart convention
    M_b_tot = float(ns_baryon_mass(M1) + ns_baryon_mass(M2))

    a, c = -8.1580, 1.2695
    expected = M_b_tot * max(5e-4, a * C1 + c)

    got = float(bns_disk_mass(M1=M1, M2=M2, R_1p4_km=R_1p4))
    assert got == pytest.approx(expected, rel=1e-10), (
        f"bns_disk_mass drifted from KF2020 Eq. (4) coefficients "
        f"(a=-8.1580, c=1.2695): got {got}, expected {expected}."
    )


def test_kruger_foucart_2020_bns_dyn_ejecta_eq6_coefficients_by_hand():
    """`bns_dynamical_ejecta` must reproduce KF2020 Eq. (6) elementwise.

    Kruger and Foucart (2020) Eq. (6) (BNS dynamical ejecta fit):

        M_ej_dyn = max(0, [(a/C1 + b*(M2/M1)^n + c*C1)*M1
                          + (a/C2 + b*(M1/M2)^n + c*C2)*M2]) * 1e-3

    with a = -9.3335, b = 114.17, c = -337.56, n = 1.5465.
    """
    from grb_physics import _compactness, bns_dynamical_ejecta, ns_radius

    M1, M2 = 1.46, 1.27
    R_1p4 = 12.0
    R1 = float(ns_radius(M1, R_1p4_km=R_1p4))
    R2 = float(ns_radius(M2, R_1p4_km=R_1p4))
    C1 = float(_compactness(M1, R1))
    C2 = float(_compactness(M2, R2))

    a, b, c, n = -9.3335, 114.17, -337.56, 1.5465
    term1 = (a / C1 + b * (M2 / M1) ** n + c * C1) * M1
    term2 = (a / C2 + b * (M1 / M2) ** n + c * C2) * M2
    expected = max(0.0, term1 + term2) * 1e-3

    got = float(bns_dynamical_ejecta(M1=M1, M2=M2, R_1p4_km=R_1p4))
    assert got == pytest.approx(expected, rel=1e-10), (
        f"bns_dynamical_ejecta drifted from KF2020 Eq. (6) coefficients: "
        f"got {got}, expected {expected}."
    )


def test_kruger_foucart_bns_dyn_ejecta_gw170817_band():
    """GW170817-like (1.46, 1.27, R=12 km) BNS dyn ejecta must be in 1e-3 to 1e-2 Msun.

    Sanity check anchored to Kruger and Foucart (2020) Sec. III
    discussion: a GW170817-like binary at R_1.4 = 12 km gives
    M_ej_dyn ~ a few times 1e-3 Msun, well below the AT2017gfo total
    ejecta ~0.05 to 0.08 Msun (Rastinejad et al. 2025, since the
    dynamical component is a subset and disk-wind ejecta dominate the
    rest).
    """
    from grb_physics import bns_dynamical_ejecta

    M_ej = float(bns_dynamical_ejecta(M1=1.46, M2=1.27, R_1p4_km=12.0))
    assert 1e-4 <= M_ej <= 5e-2, (
        f"GW170817-like KF2020 dyn-ejecta = {M_ej:.4e} Msun is outside "
        f"the [1e-4, 5e-2] band quoted in the paper Sec. III."
    )


# ─────────────────────────────────────────────────────────────────────
# Abbott et al. (2019) PRX 9, 011001  -- GW170817; not in papers/, the
# GW170817 PDF on disk is Abbott_2018.pdf (PRL 121, 161101, radii and EOS)
# Abbott et al. (2020) ApJL 892, L3   -- GW190425 [papers/Abbott_2020.pdf]
# Pin `OBSERVED_GW_EVENTS` to the published low-spin 90% CL ranges and
# source-frame chirp masses so the plot annotations cannot drift from the
# discovery papers.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name, M1_range, M2_range, Mtot_range, Mc_central, Mc_tol",
    [
        # GW170817 (Abbott+ 2019 PRX 9, 011001 Table I, low-spin |chi| < 0.05):
        # m1 in [1.36, 1.60], m2 in [1.17, 1.36], Mtot = 2.73 +0.04/-0.01,
        # Mc (source) = 1.186 +0.001/-0.001 Msun.
        ("GW170817", (1.36, 1.60), (1.17, 1.36), (2.69, 2.77), 1.186, 0.05),
        # GW190425 (Abbott+ 2020 ApJL 892, L3 Table 1, low-spin |chi| < 0.05):
        # m1 in [1.60, 1.87], m2 in [1.46, 1.69], Mtot = 3.3 +0.1/-0.1,
        # Mc (source) = 1.44 +0.02/-0.02 Msun.
        ("GW190425", (1.60, 1.87), (1.46, 1.69), (3.20, 3.40), 1.44, 0.05),
    ],
)
def test_observed_gw_events_in_published_bands(
    name, M1_range, M2_range, Mtot_range, Mc_central, Mc_tol
):
    """`OBSERVED_GW_EVENTS[name]` must lie inside the discovery-paper 90% CL bands.

    Each pinned (M1, M2) pair must satisfy three conditions taken from
    the discovery paper Table I / Table 1 under the low-spin prior:

      (i)   M1 inside the published primary-mass 90% range,
      (ii)  M2 inside the published secondary-mass 90% range,
      (iii) M1 + M2 inside the published total-mass 90% range,
      (iv)  source-frame chirp mass M_c = (M1 M2)^(3/5) / (M1+M2)^(1/5)
            within ``Mc_tol`` of the published source-frame value.

    The (i)-(iii) bands come straight from the tables; (iv) catches the
    case where M1 and M2 are individually inside their bands but their
    combination violates the more tightly measured chirp-mass anchor.
    """
    from grb_io import OBSERVED_GW_EVENTS

    ev = OBSERVED_GW_EVENTS[name]
    M1, M2 = float(ev["M1"]), float(ev["M2"])

    assert M1_range[0] <= M1 <= M1_range[1], (
        f"OBSERVED_GW_EVENTS[{name!r}]['M1'] = {M1} Msun is outside the "
        f"published low-spin 90% range {M1_range} Msun."
    )
    assert M2_range[0] <= M2 <= M2_range[1], (
        f"OBSERVED_GW_EVENTS[{name!r}]['M2'] = {M2} Msun is outside the "
        f"published low-spin 90% range {M2_range} Msun."
    )

    Mtot = M1 + M2
    assert Mtot_range[0] <= Mtot <= Mtot_range[1], (
        f"OBSERVED_GW_EVENTS[{name!r}] total mass = {Mtot:.3f} Msun is "
        f"outside the published low-spin 90% range {Mtot_range} Msun."
    )

    Mc = (M1 * M2) ** 0.6 / (M1 + M2) ** 0.2
    assert abs(Mc - Mc_central) <= Mc_tol, (
        f"OBSERVED_GW_EVENTS[{name!r}] chirp mass = {Mc:.3f} Msun differs "
        f"from the published source-frame value {Mc_central} Msun by more "
        f"than {Mc_tol} Msun."
    )


# ─────────────────────────────────────────────────────────────────────
# Hernquist (1990) ApJ 356, 359 [papers/Hernquist_1990.pdf]
# Closed-form anchors for `grb_offsets`.
# ─────────────────────────────────────────────────────────────────────
def test_hernquist_scale_radius_uses_projected_half_light_ratio():
    """`hernquist_scale_radius(R_e) = R_e / 1.8153` (Hernquist 1990 Table 1).

    Hernquist (1990) Table 1 quotes the projected half-light radius
    R_e in units of the scale radius a as R_e / a ~ 1.8153 (numerical
    Abel-transform result; differs from the 3-D half-mass relation
    r_half = a * (1 + sqrt(2)) ~ 2.414 a in Eq. 38).  Observational
    R_e values from Sersic fits (e.g. Fong and Berger 2013 HST data)
    are projected, so the projected ratio is the right one to use.
    """
    from grb_offsets import hernquist_scale_radius

    for R_e in [1.0, 5.0, 8.0, 13.7]:
        assert hernquist_scale_radius(R_e) == pytest.approx(R_e / 1.8153, rel=1e-9), (
            f"hernquist_scale_radius({R_e}) does not match Hernquist "
            f"(1990) Table 1 projected half-light ratio 1.8153."
        )


def test_hernquist_birth_radius_inverse_cdf_anchor():
    """Median birth radius from `hernquist_birth_radius` must equal `a * (1 + sqrt(2))`.

    Hernquist (1990) Eq. 38: enclosed mass fraction M(<r) / M = (r /
    (r + a))^2.  Setting this to 0.5 gives r_50 = a / (sqrt(2) - 1) =
    a * (1 + sqrt(2)) ~ 2.414 a.  The inverse-CDF sampler in the
    module is r = a * sqrt(u) / (1 - sqrt(u)) for u in (0, 1); for
    u = 0.5, this evaluates to a * sqrt(0.5) / (1 - sqrt(0.5)) ~
    2.414 a, identical to the Hernquist 50-percent enclosed-mass
    radius.  We test the empirical median over a large sample.
    """
    from grb_offsets import hernquist_birth_radius

    a = 1.0
    rng = np.random.default_rng(0)
    sample = hernquist_birth_radius(a=a, rng=rng, size=50_000)
    expected = a * (1.0 + np.sqrt(2.0))
    median_rel = abs(float(np.median(sample)) - expected) / expected
    assert median_rel < 0.02, (
        f"Hernquist birth-radius sample median = {np.median(sample):.4f} a, "
        f"expected {expected:.4f} a per Hernquist (1990) Eq. 38; "
        f"|delta|/expected = {median_rel:.4f}."
    )


@pytest.mark.parametrize("r0_over_a", [0.1, 1.0, 10.0])
def test_hernquist_escape_velocity_matches_2GM_over_r_plus_a(r0_over_a):
    """`escape_velocity(r0, M, a)` must equal `sqrt(2 G M / (r0 + a))`.

    Hernquist (1990) Eq. 5 gives Phi(r) = -GM/(r + a); the escape
    velocity is the kick at which kinetic energy equals the
    gravitational binding, ``v_esc^2 = 2 |Phi| = 2 G M / (r + a)``.
    """
    from grb_offsets import G_CGS, KPC_CM, MSUN_G, escape_velocity

    M_gal = 1e10 * MSUN_G
    a = 5.0 * KPC_CM
    r0 = r0_over_a * a
    expected = float(np.sqrt(2.0 * G_CGS * M_gal / (r0 + a)))
    got = float(escape_velocity(r=r0, M_gal=M_gal, a=a))
    assert got == pytest.approx(expected, rel=1e-12), (
        f"escape_velocity at r0/a = {r0_over_a} drifted from Hernquist "
        f"(1990) Phi(r) = -GM/(r+a): got {got}, expected {expected}."
    )


def test_hernquist_potential_and_acceleration_signs():
    """`hernquist_potential` < 0 and `hernquist_acceleration` < 0 (inward)."""
    from grb_offsets import KPC_CM, MSUN_G, hernquist_acceleration, hernquist_potential

    M_gal = 1e10 * MSUN_G
    a = 5.0 * KPC_CM
    r = 2.0 * a
    assert hernquist_potential(r, M_gal, a) < 0.0
    assert hernquist_acceleration(r, M_gal, a) < 0.0


def test_grb_offsets_cgs_constants_match_codata_values():
    """`grb_offsets` CGS unit-conversion constants match CODATA / IAU values.

    CODATA 2018 / IAU 2015 nominal values, converted to CGS:

      G        = 6.67430e-8  cm^3 g^-1 s^-2  (CODATA 2018)
      M_sun    = 1.98892e33  g               (IAU 2015 nominal)
      1 kpc    = 3.0857e21   cm              (IAU)
      1 Myr    = 3.15576e13  s               (Julian year x 1e6)
      1 km     = 1.0e5       cm              (exact)

    The module truncates these to 4-5 significant figures; the
    pinned relative tolerance of 5e-4 catches a typo or a unit-system
    swap without locking the values to spurious precision.
    """
    from grb_offsets import G_CGS, KM_CM, KPC_CM, MSUN_G, MYR_S

    assert G_CGS == pytest.approx(6.67430e-8, rel=5e-4)
    assert MSUN_G == pytest.approx(1.98892e33, rel=5e-4)
    assert KPC_CM == pytest.approx(3.0857e21, rel=5e-4)
    assert MYR_S == pytest.approx(3.15576e13, rel=5e-4)
    assert KM_CM == pytest.approx(1.0e5, rel=1e-12)


def test_R_CAP_FACTOR_pileup_matches_hernquist_eq38_closed_form():
    """`_R_CAP_FACTOR = 1000` gives a pileup fraction of (1/1001)^2 ~ 1e-6.

    Hernquist (1990) ApJ 356, 359, Eq. 38 gives the enclosed mass
    fraction ``M(<r)/M = (r / (r+a))^2`` for the cumulative inverse
    CDF used in ``hernquist_birth_radius``.  The fraction of draws
    that fall above ``r > f * a`` is then ``1 - (f / (f+1))^2 =
    (1 / (f+1))^2``.  Pinning the closed-form pileup at ``f = 1000``
    catches an accidental cap-factor change that would bias the
    offset CDF tail.
    """
    from grb_offsets import _R_CAP_FACTOR

    assert _R_CAP_FACTOR == 1000.0
    pileup = (1.0 / (1.0 + _R_CAP_FACTOR)) ** 2
    assert pileup == pytest.approx(9.98e-7, rel=1e-3), (
        f"Hernquist Eq. 38 pileup fraction at f = {_R_CAP_FACTOR} = "
        f"{pileup:.3e}; the documented 'negligible pileup ~ 1e-6' "
        f"claim is broken."
    )


# ─────────────────────────────────────────────────────────────────────
# Behroozi, Wechsler & Conroy (2013) ApJ 770, 57 (arXiv:1207.6105)
# Dutton & Maccio (2014) MNRAS 441, 3359 (arXiv:1402.7073)
# Dark-matter halo added to the Section 8 offset host potential.
# ─────────────────────────────────────────────────────────────────────
def test_behroozi13_intrinsic_params_match_published_values():
    """`BEHROOZI13_PARAMS` reproduces the Behroozi+2013 Eq. 3-4 intrinsic fit."""
    from grb_offsets import BEHROOZI13_PARAMS as p

    assert p["log10_M1_0"] == pytest.approx(11.514, abs=1e-6)
    assert p["log10_M1_a"] == pytest.approx(-1.793, abs=1e-6)
    assert p["log10_M1_z"] == pytest.approx(-0.251, abs=1e-6)
    assert p["log10_eps_0"] == pytest.approx(-1.777, abs=1e-6)
    assert p["log10_eps_a2"] == pytest.approx(-0.119, abs=1e-6)
    assert p["alpha_0"] == pytest.approx(-1.412, abs=1e-6)
    assert p["delta_0"] == pytest.approx(3.508, abs=1e-6)
    assert p["gamma_0"] == pytest.approx(0.316, abs=1e-6)


def test_dutton_maccio14_c200_zeropoint_and_slope_at_z0():
    """`DUTTON14_C200` reproduces the Planck NFW c200-M200 fit (their Eq. 8/10/11).

    At z = 0 the relation is log10 c200 = 0.905 - 0.101 log10(M200/[1e12 h^-1 Msun]).
    """
    from grb_offsets import DUTTON14_C200 as d
    from grb_offsets import H0_PLANCK15, MSUN_G, concentration_dutton_maccio

    assert d["b_0"] == pytest.approx(-0.101, abs=1e-6)
    assert d["b_z"] == pytest.approx(0.026, abs=1e-6)
    assert (d["a_0"] + d["a_inf_offset"]) == pytest.approx(0.905, abs=1e-6)
    assert d["z_scale"] == pytest.approx(0.617, abs=1e-6)
    assert d["z_index"] == pytest.approx(1.21, abs=1e-6)

    # A 1e12 h^-1 Msun halo at z = 0 has log10 c200 = 0.905 (c ~ 8.04).
    h = H0_PLANCK15 / 100.0
    M200_g = (1e12 / h) * MSUN_G
    c = concentration_dutton_maccio(M200_g, 0.0)
    assert c == pytest.approx(10.0**0.905, rel=1e-6)


def test_behroozi_smhm_inversion_round_trips_against_forward_relation():
    """`halo_mass_from_stellar_mass` inverts the published forward relation.

    Feeding a halo mass through the Behroozi+2013 forward M_*(M_h) and then
    back through the numerical inverse must recover the original halo mass.
    """
    from grb_offsets import (
        MSUN_G,
        _behroozi_log10_mstar,
        halo_mass_from_stellar_mass,
    )

    z = 0.5
    for log10_Mh in (11.0, 12.0, 13.0):
        log10_Mstar = _behroozi_log10_mstar(log10_Mh, z)
        M_star_g = 10.0**log10_Mstar * MSUN_G
        recovered = halo_mass_from_stellar_mass(M_star_g, z)
        assert np.log10(recovered / MSUN_G) == pytest.approx(log10_Mh, abs=1e-4)


def test_halo_to_stellar_ratio_is_tens_for_sgrb_hosts():
    """The recovered M200 / M_* lands in the tens for the sGRB host masses.

    Behroozi+2013 puts peak star-formation efficiency near
    log M_*/Msun ~ 10.5 with M_h/M_* ~ 30, rising on either side.  The
    HOST_MODELS templates should therefore all sit in the tens, never near
    unity (which would signal a unit slip in the inversion).
    """
    from grb_offsets import HALO_FIDUCIAL_Z, HOST_MODELS, halo_mass_from_stellar_mass

    for name, hp in HOST_MODELS.items():
        M200 = halo_mass_from_stellar_mass(hp["M_gal"], HALO_FIDUCIAL_Z)
        ratio = M200 / hp["M_gal"]
        assert 10.0 < ratio < 300.0, f"{name}: M_halo/M_* = {ratio:.1f} outside [10, 300]"


def test_halo_fiducial_z_is_median_measured_bright_redshift():
    """`HALO_FIDUCIAL_Z` is the median of the measured BRIGHT redshifts.

    Derived from the sample rather than chosen: the bursts Fong et al. 2022
    place at an assumed z = 1 are excluded so the fiducial reflects the
    measured-redshift subset only.
    """
    from grb_offsets import (
        FONG22_SGRB_ASSUMED_Z1,
        FONG22_SGRB_NAMES,
        FONG22_SGRB_REDSHIFTS,
        HALO_FIDUCIAL_Z,
    )

    measured = ~np.isin(np.asarray(FONG22_SGRB_NAMES), np.asarray(FONG22_SGRB_ASSUMED_Z1))
    expected = float(np.median(FONG22_SGRB_REDSHIFTS[measured]))
    assert HALO_FIDUCIAL_Z == pytest.approx(expected, abs=1e-9)
    assert 0.2 < HALO_FIDUCIAL_Z < 1.0


def test_total_escape_velocity_reduces_to_hernquist_when_halo_off():
    """With `Mc = 0` the composite escape velocity equals the Hernquist form."""
    from grb_offsets import (
        DEFAULT_M_GAL,
        DEFAULT_R_E,
        HostPotential,
        escape_velocity,
        hernquist_scale_radius,
        total_escape_velocity,
    )

    a = hernquist_scale_radius(DEFAULT_R_E)
    host0 = HostPotential(M_star=DEFAULT_M_GAL, a=a, Mc=0.0, r_s=1.0)
    for r_over_a in (0.1, 1.0, 10.0):
        r = r_over_a * a
        got = float(total_escape_velocity(r, host0))
        expected = float(escape_velocity(r, DEFAULT_M_GAL, a))
        assert got == pytest.approx(expected, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Fong and Berger (2013) ApJ 776, 18 [papers/Fong_2013.pdf]
# Fong et al. (2022) hosts I/II [papers/Fong_2022.pdf, papers/Nugent_2022.pdf]
# Default host R_e and host-type weights.
# ─────────────────────────────────────────────────────────────────────
def test_fong_berger_2013_default_R_e_within_observed_sgrb_band():
    """`DEFAULT_R_E = 5.0 kpc` must be inside the Fong sGRB host R_e median band.

    Fong and Berger (2013) ApJ 776, 18 (Sec. 3.2) report a median
    sGRB host effective radius R_e ~ 4 to 6 kpc; the Fong et al.
    (2022) follow-up (ApJ 940, 56 / 936, 16) confirms a median of
    5.0 +/- 1.0 kpc when star-forming and elliptical hosts are
    co-fit.
    """
    from grb_offsets import DEFAULT_R_E, KPC_CM

    R_e_kpc = DEFAULT_R_E / KPC_CM
    assert 3.0 <= R_e_kpc <= 7.0, (
        f"DEFAULT_R_E = {R_e_kpc:.2f} kpc lies outside the Fong "
        f"and Berger (2013) sGRB host R_e median band [3, 7] kpc."
    )


def test_fong_berger_2013_host_model_weights_sum_to_one():
    """`HOST_MODELS` weights must sum to 1 (75/25 SF/elliptical mix).

    Fong and Berger (2013) Sec. 4 quote a sGRB host-type breakdown
    of ~75 percent star-forming and ~25 percent elliptical; the
    project sub-divides star-forming into ``SF_disk`` (50 percent)
    and ``SF_massive`` (25 percent), giving the 50/25/25 mix that is
    consistent with their Table 4.
    """
    from grb_offsets import HOST_MODELS

    weights = sum(host["weight"] for host in HOST_MODELS.values())
    assert weights == pytest.approx(1.0, rel=1e-12), (
        f"HOST_MODELS weights sum to {weights}, not 1.0 as required "
        f"by Fong and Berger (2013) Sec. 4 host-type fractions."
    )
    sf_weight = HOST_MODELS["SF_disk"]["weight"] + HOST_MODELS["SF_massive"]["weight"]
    el_weight = HOST_MODELS["Elliptical"]["weight"]
    # Fong+ 2013 reports ~75/25 SF/elliptical; allow +/- 0.10 absolute.
    assert abs(sf_weight - 0.75) <= 0.10, sf_weight
    assert abs(el_weight - 0.25) <= 0.10, el_weight


# ─────────────────────────────────────────────────────────────────────
# Leibler and Berger (2010) ApJ 725, 1202 [arXiv:1009.1147]
# Nugent et al. (2022) ApJ 940, 57 [papers/Nugent_2022.pdf]
# Fong et al. (2022) ApJ 940, 56 [papers/Fong_2022.pdf]
# sGRB host stellar masses and effective radii per host type.
# ─────────────────────────────────────────────────────────────────────
def test_host_models_stellar_masses_within_leibler_berger_2010_bands():
    """`HOST_MODELS` stellar masses must match the sGRB host populations.

    Leibler and Berger (2010) ApJ 725, 1202 fit stellar masses for
    the sGRB host sample with two clusters: a low-to-intermediate-mass
    SF-host group (log M*/Msun ~ 9.5 to 10.7) and a high-mass
    elliptical group (log M*/Msun ~ 10.7 to 11.2).  Nugent et al.
    (2022), ApJ 940, 57 (Fong+ 2022 hosts II) confirms the SF-host
    median at log M*/Msun = 9.69 +0.75/-0.65 from 69 events.  The
    project's three host templates must sit inside these bands.
    """
    from grb_offsets import HOST_MODELS, MSUN_G

    log_m_sf_disk = np.log10(HOST_MODELS["SF_disk"]["M_gal"] / MSUN_G)
    log_m_sf_massive = np.log10(HOST_MODELS["SF_massive"]["M_gal"] / MSUN_G)
    log_m_elliptical = np.log10(HOST_MODELS["Elliptical"]["M_gal"] / MSUN_G)

    assert 9.5 <= log_m_sf_disk <= 10.0, (
        f"SF_disk log M*/Msun = {log_m_sf_disk:.2f} outside the Leibler "
        f"and Berger (2010) low-mass SF-host band [9.5, 10.0]."
    )
    assert 10.3 <= log_m_sf_massive <= 10.7, (
        f"SF_massive log M*/Msun = {log_m_sf_massive:.2f} outside the "
        f"Leibler and Berger (2010) massive SF-host band [10.3, 10.7]."
    )
    assert 10.7 <= log_m_elliptical <= 11.2, (
        f"Elliptical log M*/Msun = {log_m_elliptical:.2f} outside the "
        f"Leibler and Berger (2010) elliptical-host band [10.7, 11.2]."
    )


def test_host_models_R_e_within_fong_2022_hosts_bands():
    """`HOST_MODELS` R_e values must match Fong+ 2022 hosts I / 2013 fits.

    Fong and Berger (2013) ApJ 776, 18, Sec. 3.2 report a median sGRB
    host R_e ~ 3.6 kpc for the SF subset.  Fong et al. (2022) ApJ 940,
    56 (hosts I) refit the larger HST sample and find R_e ~ 5 +/- 1
    kpc for the co-fit population.  Ellipticals run somewhat larger,
    R_e ~ 6 to 10 kpc.
    """
    from grb_offsets import HOST_MODELS, KPC_CM

    re_sf_disk_kpc = HOST_MODELS["SF_disk"]["R_e"] / KPC_CM
    re_sf_massive_kpc = HOST_MODELS["SF_massive"]["R_e"] / KPC_CM
    re_elliptical_kpc = HOST_MODELS["Elliptical"]["R_e"] / KPC_CM

    assert 3.0 <= re_sf_disk_kpc <= 5.0, re_sf_disk_kpc
    assert 4.0 <= re_sf_massive_kpc <= 6.0, re_sf_massive_kpc
    assert 6.0 <= re_elliptical_kpc <= 10.0, re_elliptical_kpc


def test_sf_host_transition_t_myr_within_leibler_berger_2010_age_band():
    """`SF_HOST_TRANSITION_T_MYR` must sit inside the Leibler-Berger band.

    CODE HEURISTIC.  Leibler and Berger (2010) ApJ 725, 1202 report
    sGRB host stellar-population ages with late-type-host medians of
    a few tenths of a Gyr and early-type-host medians up to several
    Gyr; the transition between the two populations sits in the 1 to
    5 Gyr range.  The 3 Gyr fiducial in
    ``SF_HOST_TRANSITION_T_MYR`` must stay inside that band so that
    ``assign_host_by_delay`` keeps routing short-delay systems to SF
    hosts and long-delay systems to ellipticals consistently with the
    observed-host demographics.
    """
    from grb_offsets import SF_HOST_TRANSITION_T_MYR

    assert 1000.0 <= SF_HOST_TRANSITION_T_MYR <= 5000.0, (
        f"SF_HOST_TRANSITION_T_MYR = {SF_HOST_TRANSITION_T_MYR} Myr "
        f"outside the Leibler-Berger 2010 sGRB host age band [1, 5] Gyr."
    )


def test_sf_disk_frac_consistent_with_host_models_weights():
    """`SF_DISK_FRAC_OF_SF_HOSTS` must match the SF-host weight ratio.

    Internal consistency check: the per-event SF_disk routing
    probability has to equal the within-SF weight ratio in
    ``HOST_MODELS``, otherwise a sample built via ``assign_host_by_delay``
    drifts away from the population-level host mixture used by
    ``compute_offsets_mixed_hosts``.
    """
    from grb_offsets import HOST_MODELS, SF_DISK_FRAC_OF_SF_HOSTS

    w_disk = HOST_MODELS["SF_disk"]["weight"]
    w_massive = HOST_MODELS["SF_massive"]["weight"]
    expected = w_disk / (w_disk + w_massive)
    assert abs(SF_DISK_FRAC_OF_SF_HOSTS - expected) <= 0.05, (
        f"SF_DISK_FRAC_OF_SF_HOSTS = {SF_DISK_FRAC_OF_SF_HOSTS} drifted "
        f"from HOST_MODELS-derived SF-disk fraction {expected:.3f}."
    )


# ─────────────────────────────────────────────────────────────────────
# Fong and Berger (2013) ApJ 776, 18 + Fong et al. (2010) ApJ 708, 9
# Short-GRB projected physical offset distribution (Figure 5; N = 22).
# ─────────────────────────────────────────────────────────────────────
def test_fong_berger_2013_sgrb_offset_array_matches_published_summary():
    """`OBSERVED_SGRB_OFFSETS_KPC` must reproduce the published summary.

    Fong and Berger (2013) ApJ 776, 18, Sec. 3.3 (and Figure 5)
    summarises the 22-event sub-arcsecond sGRB host-offset sample as
    range ~ 0.5 - 75 kpc, median ~ 4.5 kpc, and ~ 25 percent of events
    at projected offset >= 10 kpc.  The per-burst values in
    ``OBSERVED_SGRB_OFFSETS_KPC`` come from Fong and Berger (2013)
    Table 2 (13 events), Fong, Berger and Fox (2010) ApJ 708, 9
    Table 3 (6 events with sub-arcsec positions), and three
    ground-based supplements quoted in Fong and Berger (2013)
    Sec. 3.3 (111020A, 111117A, 120804A).
    """
    from grb_offsets import OBSERVED_SGRB_OFFSETS_KPC

    arr = np.asarray(OBSERVED_SGRB_OFFSETS_KPC, dtype=float)
    assert arr.size == 22, (
        f"OBSERVED_SGRB_OFFSETS_KPC has {arr.size} entries; "
        f"Fong and Berger (2013) Figure 5 sample is 22."
    )

    median = float(np.median(arr))
    assert 4.0 <= median <= 5.0, (
        f"OBSERVED_SGRB_OFFSETS_KPC median = {median:.2f} kpc outside "
        f"the Fong and Berger (2013) Sec. 3.3 band [4.0, 5.0] (4.5 quoted)."
    )

    assert arr.min() < 0.5, (
        f"OBSERVED_SGRB_OFFSETS_KPC min = {arr.min():.2f} kpc is above "
        f"the F&B 2013 smallest projected offset (GRB 090426, 0.45 kpc)."
    )
    assert arr.max() >= 70.0, (
        f"OBSERVED_SGRB_OFFSETS_KPC max = {arr.max():.2f} kpc is below "
        f"the F&B 2013 largest offset (GRB 090515, 75.03 kpc)."
    )

    frac_ge_10 = float((arr >= 10.0).sum()) / arr.size
    assert 0.20 <= frac_ge_10 <= 0.40, (
        f"Fraction with offset >= 10 kpc = {frac_ge_10:.2f} outside "
        f"the F&B 2013 Sec. 3.3 quote (~25 percent); the 22-event "
        f"sample value is ~ 0.32."
    )


# ─────────────────────────────────────────────────────────────────────
# Long-GRB-with-kilonova projected offsets
# Della Valle et al. (2006) Nature 444, 1050 [060614];
# Rastinejad et al. (2022) Nature 612, 223 [papers/Rastinejad_2022.pdf] [211211A];
# Levan et al. (2024) Nature 626, 737 [papers/Levan_2024.pdf]
# [230307A].
# ─────────────────────────────────────────────────────────────────────
def test_lgrb_kn_offsets_match_skobe_2026_table_2():
    """`OBSERVED_LGRB_KN_OFFSETS_KPC` must match Skobe et al. (2026) Table 2.

    Skobe et al. (2026) arXiv:2604.06586 Sec. 2 classify four of their
    nine kilonova-bearing GRBs as LGRB-KNe: 050709, 060614, 211211A and
    230307A.  GRBs 050709 and 060614 are short bursts with extended
    emission, which they count as merger-driven long events.  Physical
    offsets are their Table 2, sourced from Fong et al. (2022) Table 4.
    """
    from grb_offsets import (
        OBSERVED_LGRB_KN_NAMES,
        OBSERVED_LGRB_KN_OFFSET_ERR_KPC_WMAP9,
        OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9,
    )

    arr = np.asarray(OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9, dtype=float)
    err = np.asarray(OBSERVED_LGRB_KN_OFFSET_ERR_KPC_WMAP9, dtype=float)
    assert arr.size == 4, (
        f"OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9 has {arr.size} entries; Skobe "
        f"et al. (2026) classify 4 events as LGRB-KN."
    )
    assert err.shape == arr.shape
    assert len(OBSERVED_LGRB_KN_NAMES) == arr.size

    expected = {"060614": 0.70, "050709": 3.76, "211211A": 7.92, "230307A": 38.90}
    assert tuple(OBSERVED_LGRB_KN_NAMES) == tuple(expected)
    for name, got in zip(OBSERVED_LGRB_KN_NAMES, arr):
        assert abs(got - expected[name]) <= 0.01, (
            f"LGRB-KN offset for {name} = {got} kpc drifted from the "
            f"Skobe et al. (2026) Table 2 value {expected[name]} kpc."
        )
    assert np.all(np.diff(arr) > 0), "LGRB-KN offsets must stay in ascending order."

    # Table 2 quotes 060614 at 0.31 +/- 0.35 arcsec, so its uncertainty
    # exceeds the central offset; the event is consistent with nuclear.
    assert err[0] > arr[0], (
        "The 060614 offset uncertainty should exceed its central value "
        "per Skobe et al. (2026) Table 2."
    )


def test_sgrb_kn_offsets_match_skobe_2026_table_2():
    """The published kpc column must match Skobe et al. (2026) Table 2.

    The five events they classify as sGRB-KNe are 130603B, 150101B,
    160821B, 170817A and 200522A, with physical offsets from their
    Table 2 (sourced from Fong et al. 2022 Table 4).  These are the
    physical offsets, not the host-normalised R/r50 column.
    """
    from grb_offsets import (
        OBSERVED_SGRB_KN_NAMES,
        OBSERVED_SGRB_KN_OFFSET_ERR_KPC_WMAP9,
        OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9,
    )

    arr = np.asarray(OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9, dtype=float)
    err = np.asarray(OBSERVED_SGRB_KN_OFFSET_ERR_KPC_WMAP9, dtype=float)
    assert arr.size == 5, (
        f"OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9 has {arr.size} entries; Skobe "
        f"et al. (2026) classify 5 events as sGRB-KN."
    )
    assert err.shape == arr.shape
    expected = {
        "200522A": 0.93,
        "170817A": 2.125,
        "130603B": 5.40,
        "150101B": 7.35,
        "160821B": 15.74,
    }
    assert tuple(OBSERVED_SGRB_KN_NAMES) == tuple(expected)
    for name, got in zip(OBSERVED_SGRB_KN_NAMES, arr):
        assert abs(got - expected[name]) <= 0.01, (
            f"sGRB-KN offset for {name} = {got} kpc drifted from the "
            f"Skobe et al. (2026) Table 2 value {expected[name]} kpc."
        )
    assert np.all(np.diff(arr) > 0), "sGRB-KN offsets must stay in ascending order."
    assert np.all(err > 0)


def test_kn_offsets_are_physical_not_host_normalised():
    """The KN arrays must hold Table 2 kpc, never the R/r50 column.

    Skobe et al. (2026) Table 2 carries both a physical offset and a
    host-normalised R/r50.  Only the physical column is comparable with the
    orbit-integrated model, and R/r50 is filter-dependent besides (r50 comes
    from the per-band GALFIT fit).  060614 is the trap worth pinning: its
    F606W R/r50 is 0.73, numerically identical to the discovery-paper kpc
    offset this module used to carry, so picking the wrong column there
    reproduces a stale value rather than raising.
    """
    from grb_offsets import (
        OBSERVED_LGRB_KN_NAMES,
        OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9,
        OBSERVED_SGRB_KN_NAMES,
        OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9,
    )

    r_over_r50 = {
        "200522A": 0.25,
        "170817A": 0.374,
        "130603B": 1.62,
        "150101B": 0.58,
        "160821B": 2.14,
        "060614": 0.94,
        "050709": 1.91,
        "211211A": 3.21,
        "230307A": 12.12,
    }
    stored = dict(
        zip(OBSERVED_SGRB_KN_NAMES, np.asarray(OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9, float))
    ) | dict(zip(OBSERVED_LGRB_KN_NAMES, np.asarray(OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9, float)))

    for name, normalised in r_over_r50.items():
        assert abs(stored[name] - normalised) > 0.05, (
            f"{name} is stored as {stored[name]}, which matches the Skobe "
            f"et al. (2026) Table 2 host-normalised R/r50 = {normalised} "
            f"rather than the physical kpc offset."
        )

    # The specific historical collision.
    assert abs(stored["060614"] - 0.73) > 0.02, (
        "060614 must not carry 0.73: that is its F606W R/r50 and also the "
        "superseded discovery-paper kpc offset."
    )


def test_kn_offsets_rescaled_to_project_cosmology():
    """The comparison arrays put Skobe's WMAP9 kpc on the project cosmology.

    Planck 2015 (H0 = 67.74) gives larger distances than the WMAP9 values
    (H0 = 69.6) Skobe et al. adopt, so every offset moves up by 1 to 3
    percent over the z = 0.01 to 0.55 span of the sample.
    """
    from grb_offsets import (
        OBSERVED_LGRB_KN_OFFSET_ERR_KPC,
        OBSERVED_LGRB_KN_OFFSET_ERR_KPC_WMAP9,
        OBSERVED_LGRB_KN_OFFSETS_KPC,
        OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9,
        OBSERVED_SGRB_KN_OFFSET_ERR_KPC,
        OBSERVED_SGRB_KN_OFFSET_ERR_KPC_WMAP9,
        OBSERVED_SGRB_KN_OFFSETS_KPC,
        OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9,
    )

    for derived, published in [
        (OBSERVED_SGRB_KN_OFFSETS_KPC, OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9),
        (OBSERVED_LGRB_KN_OFFSETS_KPC, OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9),
        (OBSERVED_SGRB_KN_OFFSET_ERR_KPC, OBSERVED_SGRB_KN_OFFSET_ERR_KPC_WMAP9),
        (OBSERVED_LGRB_KN_OFFSET_ERR_KPC, OBSERVED_LGRB_KN_OFFSET_ERR_KPC_WMAP9),
    ]:
        ratio = np.asarray(derived, float) / np.asarray(published, float)
        assert np.all(ratio > 1.0), (
            "Planck 2015 distances exceed WMAP9 over this redshift range, so "
            f"the rescaled offsets must all grow; got ratios {ratio}."
        )
        assert np.all(ratio < 1.03), f"Rescaling should be a percent-level shift, got {ratio}."

    # Rescaling is a pure distance factor, so it cannot reorder the sample.
    assert np.all(np.diff(OBSERVED_SGRB_KN_OFFSETS_KPC) > 0)
    assert np.all(np.diff(OBSERVED_LGRB_KN_OFFSETS_KPC) > 0)


def test_planck15_cosmology_constants_match_the_compas_pin():
    """`grb_physics` cosmology must stay on the COMPAS FastCosmicIntegration pin.

    Planck 2015 as hard-coded in COMPAS post-processing.  Neijssel et al.
    (2019) quote the rounded H0 = 67.8, Omega_m = 0.308; the pin here is
    deliberately the unrounded code constant so rates match COMPAS exactly.
    """
    from grb_physics import H0_PLANCK15, OMEGA_L_PLANCK15, OMEGA_M_PLANCK15

    assert H0_PLANCK15 == pytest.approx(67.74)
    assert OMEGA_M_PLANCK15 == pytest.approx(0.3089)
    assert OMEGA_L_PLANCK15 == pytest.approx(0.6911)
    assert OMEGA_M_PLANCK15 + OMEGA_L_PLANCK15 == pytest.approx(1.0), (
        "The cosmology is flat; the rate integration and the offset "
        "conversion both assume Omega_k = 0."
    )


def test_fong_2022_bright_offsets_match_published_summary():
    """`FONG22_SGRB_OFFSETS_KPC_WMAP9` must reproduce Fong et al. (2022) Table 4.

    Their Sec. 6.2 quotes an observed median projected offset of 7.92 kpc with
    a [-6.13, +20.71] kpc interval from the 16th and 84th percentiles of the
    full 83-burst distribution.  Reproducing all three from the transcribed
    array is the check that no row was dropped or mistyped; the median alone
    would not catch a swapped tail.
    """
    from grb_offsets import (
        FONG22_SGRB_ASSUMED_Z1,
        FONG22_SGRB_NAMES,
        FONG22_SGRB_OFFSET_ERR_KPC_WMAP9,
        FONG22_SGRB_OFFSETS_KPC_WMAP9,
        FONG22_SGRB_REDSHIFTS,
    )

    arr = np.asarray(FONG22_SGRB_OFFSETS_KPC_WMAP9, dtype=float)
    assert arr.size == 83, (
        f"Fong et al. (2022) Table 4 lists 83 measured offsets (plus one upper "
        f"limit for 081226A, excluded); got {arr.size}."
    )
    for parallel in (FONG22_SGRB_REDSHIFTS, FONG22_SGRB_OFFSET_ERR_KPC_WMAP9):
        assert np.asarray(parallel).size == arr.size
    assert len(FONG22_SGRB_NAMES) == arr.size
    assert len(set(FONG22_SGRB_NAMES)) == arr.size, "Duplicate burst in the catalog."
    assert "081226A" not in FONG22_SGRB_NAMES, (
        "081226A is an upper limit (< 4.06 kpc), not a measurement."
    )

    assert np.median(arr) == pytest.approx(7.92, abs=0.01), (
        f"Median offset {np.median(arr):.2f} kpc drifted from the published 7.92 kpc."
    )
    p16, p84 = np.percentile(arr, [16, 84])
    assert p16 == pytest.approx(7.92 - 6.13, abs=0.05), f"16th percentile {p16:.2f} kpc"
    assert p84 == pytest.approx(7.92 + 20.71, abs=0.05), f"84th percentile {p84:.2f} kpc"

    # Table 4 note a: bursts without a redshift get their physical offset at an
    # assumed z = 1, which the stored redshifts have to reproduce for the
    # cosmology rescaling to be defined.
    assert len(FONG22_SGRB_ASSUMED_Z1) == 11
    assert set(FONG22_SGRB_ASSUMED_Z1) <= set(FONG22_SGRB_NAMES)
    at_unity = {n for n, z in zip(FONG22_SGRB_NAMES, FONG22_SGRB_REDSHIFTS) if z == 1.0}
    assert at_unity == set(FONG22_SGRB_ASSUMED_Z1)


def test_bright_portal_revisions_are_recorded_but_stay_off_the_science_path():
    """The live BRIGHT portal disagrees with Table 4 for seven of the 83 bursts.

    ``BRIGHT_PORTAL_REVISIONS`` records what the portal reads while the
    comparison arrays keep the published values, so both halves need pinning:
    an edit must not quietly adopt the portal numbers, and the recorded deltas
    must not go stale.  Which revisions are safe to adopt is re-derived here
    rather than stored, since that is the whole content of the distinction: a
    revision driven only by a new redshift has to reproduce under the angular
    scaling, and one that does not means the host association moved.
    """
    from grb_offsets import (
        BRIGHT_PORTAL_REVISIONS,
        FONG22_SGRB_NAMES,
        FONG22_SGRB_OFFSETS_KPC_WMAP9,
        FONG22_SGRB_REDSHIFTS,
        SGRB_RECLASSIFIED_AS_LGRB_KN,
        kpc_per_arcsec,
    )

    published = {
        "120804": (2.30, 0.74),
        "150831A": (12.21, 1.09),
        "160525B": (8.61, 1.00),
        "170127B": (10.37, 2.28),
        "170728B": (6.76, 0.62),
        "200411A": (41.98, 1.93),
        "210726A": (12.26, 0.2244),
    }
    assert set(BRIGHT_PORTAL_REVISIONS) == set(published)

    names = list(FONG22_SGRB_NAMES)
    for burst, (offset, redshift) in published.items():
        i = names.index(burst)
        assert FONG22_SGRB_OFFSETS_KPC_WMAP9[i] == pytest.approx(offset, abs=0.005), (
            f"{burst} drifted off Fong et al. (2022) Table 4. The comparison "
            f"arrays are the published census, not the live portal."
        )
        assert FONG22_SGRB_REDSHIFTS[i] == pytest.approx(redshift, abs=5e-5)

    reproduced = {
        burst
        for burst, (offset, redshift) in BRIGHT_PORTAL_REVISIONS.items()
        if abs(
            published[burst][0] * kpc_per_arcsec(redshift) / kpc_per_arcsec(published[burst][1])
            - offset
        )
        <= 0.03 * offset
    }
    assert reproduced == {"150831A", "170127B", "170728B", "200411A"}

    # 210726A is the one the literature has not settled: Schroeder et al. (2024)
    # confirm the host but favour z ~ 2.38 over the portal's 0.37, which is why
    # its factor-50 revision is not adopted on the strength of the portal alone.
    assert BRIGHT_PORTAL_REVISIONS["210726A"][0] < 1.0 < published["210726A"][0]

    # Neither way of adopting the portal escapes the band Section 8 already
    # draws (Gold 5.1 kpc to astrometrically broadened 10.9 kpc), so the choice
    # reaches no conclusion in the section.
    offsets = np.asarray(FONG22_SGRB_OFFSETS_KPC_WMAP9, dtype=float)
    classical = np.array([n not in SGRB_RECLASSIFIED_AS_LGRB_KN for n in names])
    revised = offsets.copy()
    for burst, (offset, _) in BRIGHT_PORTAL_REVISIONS.items():
        revised[names.index(burst)] = offset
    assert np.median(offsets[classical]) == pytest.approx(8.23, abs=0.01)
    assert np.median(revised[classical]) == pytest.approx(7.94, abs=0.01)


def test_classical_sgrb_comparison_is_fong_2022_less_the_lgrb_kn_reclassifications():
    """The classical comparison must not double-count Skobe's LGRB-KNe.

    Skobe et al. (2026) count short bursts with extended emission as
    merger-driven LGRB-KNe.  Three of those sit in the Fong et al. (2022)
    short-GRB catalog (050709, 060614, 211211A) and also in
    ``OBSERVED_LGRB_KN_OFFSETS_KPC``, so the classical array drops them to keep
    the two Section 8 panels disjoint in taxonomy.  The kilonova-selected
    *short* bursts stay, because KN selection is not a partition of the sGRB
    population.
    """
    from grb_offsets import (
        FONG22_SGRB_NAMES,
        OBSERVED_SGRB_CLASSICAL_NAMES,
        OBSERVED_SGRB_OFFSET_ERR_CLASSICAL_KPC,
        OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC,
        SGRB_RECLASSIFIED_AS_LGRB_KN,
    )

    assert SGRB_RECLASSIFIED_AS_LGRB_KN == ("050709", "060614", "211211A")
    assert OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC.size == len(FONG22_SGRB_NAMES) - 3 == 80
    assert OBSERVED_SGRB_OFFSET_ERR_CLASSICAL_KPC.size == 80
    assert len(OBSERVED_SGRB_CLASSICAL_NAMES) == 80

    dropped = set(FONG22_SGRB_NAMES) - set(OBSERVED_SGRB_CLASSICAL_NAMES)
    assert dropped == set(SGRB_RECLASSIFIED_AS_LGRB_KN)

    # The sGRB-KNe are a sub-selection of the sGRB population, not a partition.
    for retained in ("130603B", "150101B", "160821B", "200522A"):
        assert retained in OBSERVED_SGRB_CLASSICAL_NAMES, (
            f"{retained} is an sGRB-KN in Skobe et al. and must stay in the "
            f"classical sGRB population."
        )


def test_fong_berger_2013_array_is_no_longer_the_comparison_sample():
    """F&B 2013 stays a pinned anchor but must not be the comparison array.

    Section 8 moved onto the Fong et al. (2022) BRIGHT catalog, whose median is
    ~3.5 kpc larger over four times the sample.  Keeping the older array wired
    into a comparison would silently understate the observed offsets.
    """
    from grb_offsets import (
        OBSERVED_SGRB_OFFSET_NAMES,
        OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC,
        OBSERVED_SGRB_OFFSETS_KPC,
    )

    assert len(OBSERVED_SGRB_OFFSET_NAMES) == OBSERVED_SGRB_OFFSETS_KPC.size, (
        "OBSERVED_SGRB_OFFSET_NAMES must stay parallel to OBSERVED_SGRB_OFFSETS_KPC."
    )
    assert OBSERVED_SGRB_OFFSETS_KPC.size == 22
    assert OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC.size != OBSERVED_SGRB_OFFSETS_KPC.size - 1
    assert np.median(OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC) > np.median(OBSERVED_SGRB_OFFSETS_KPC), (
        "The BRIGHT catalog median should exceed the F&B 2013 median."
    )


def test_fong_2022_sample_grades_match_published_counts_and_gold_median():
    """Gold / Silver / Bronze counts and the 4.92 kpc Gold median, Table 2 and Sec. 6.2.

    Fong et al. grade the 84 host associations by minimum probability of chance
    coincidence: 49 Gold, 22 Silver, 13 Bronze, with 6 further bursts
    Inconclusive.  Our arrays carry the 83 with measured offsets, so Gold drops
    to 48 (081226A is Gold but censored) and the other two grades are complete.

    The Gold median is the anchor that matters, because Sec. 6.2 reads it as a
    minimum on the short-GRB offset distribution and it is the conservative edge
    of the Section 8 band.
    """
    from grb_offsets import (
        FONG22_SGRB_BRONZE,
        FONG22_SGRB_NAMES,
        FONG22_SGRB_OFFSETS_KPC_WMAP9,
        FONG22_SGRB_SAMPLE_GRADE,
        FONG22_SGRB_SILVER,
    )

    grades = np.asarray(FONG22_SGRB_SAMPLE_GRADE)
    assert grades.size == len(FONG22_SGRB_NAMES) == 83
    counts = {g: int((grades == g).sum()) for g in ("Gold", "Silver", "Bronze")}
    assert counts == {"Gold": 48, "Silver": 22, "Bronze": 13}, (
        f"Grade counts {counts} drifted from Fong et al. (2022) Table 2 "
        f"(49 / 22 / 13 including the censored 081226A)."
    )

    # Every named Silver / Bronze burst has to exist in the catalog, or a typo
    # would silently promote it to Gold.
    for name in FONG22_SGRB_SILVER + FONG22_SGRB_BRONZE:
        assert name in FONG22_SGRB_NAMES, f"{name} is graded but not in the catalog."

    gold = np.asarray(FONG22_SGRB_OFFSETS_KPC_WMAP9)[grades == "Gold"]
    assert np.median(gold) == pytest.approx(4.92, abs=0.02), (
        f"Gold median {np.median(gold):.2f} kpc drifted from the published 4.92 kpc."
    )

    # The grade is not a neutral quality flag: robustness of association
    # correlates with offset, which is what forces the band in Section 8.
    bronze = np.asarray(FONG22_SGRB_OFFSETS_KPC_WMAP9)[grades == "Bronze"]
    assert np.median(bronze) > 4 * np.median(gold), (
        "Bronze associations should sit far outside the Gold ones (Sec. 7.1)."
    )


def test_rice_resampling_reproduces_the_fong_2022_sampled_distribution():
    """The Rice draws must land on Fong et al. (2022) Sec. 6.2's sampled quantiles.

    They quote a sampled median of 9.82 kpc with a [-7.50, +22.86] kpc interval
    against an observed 7.92 [-6.13, +20.71], the shift coming from the 15
    bursts whose uncertainty exceeds their offset.  Agreement here is at the
    1 to 2 percent level rather than exact, because the realizations come from a
    different random stream than theirs.
    """
    from grb_offsets import (
        FONG22_SGRB_OFFSET_ERR_KPC_WMAP9,
        FONG22_SGRB_OFFSETS_KPC_WMAP9,
        rice_resample_offsets,
    )

    off = np.asarray(FONG22_SGRB_OFFSETS_KPC_WMAP9)
    err = np.asarray(FONG22_SGRB_OFFSET_ERR_KPC_WMAP9)
    assert int((err > off).sum()) == 15, (
        "Sec. 6.2: 15 bursts have a measurement uncertainty larger than their offset."
    )

    draws = rice_resample_offsets(off, err)
    assert np.median(draws) == pytest.approx(9.82, abs=0.35), (
        f"Sampled median {np.median(draws):.2f} kpc against the published 9.82 kpc."
    )
    p16, p84 = np.percentile(draws, [16, 84])
    assert p16 == pytest.approx(9.82 - 7.50, abs=0.25), f"16th percentile {p16:.2f} kpc"
    assert p84 == pytest.approx(9.82 + 22.86, abs=1.0), f"84th percentile {p84:.2f} kpc"
    assert np.median(draws) > np.median(off), (
        "Folding in the uncertainties must raise the distribution, not lower it."
    )


# ─────────────────────────────────────────────────────────────────────
# Madau and Dickinson (2014) ARA&A 52, 415 [papers/Madau_2014.pdf]
# Neijssel et al. (2019) MNRAS 490, 3740, Eq. 6 [papers/Neijssel_2019.pdf]
# SFR(z) Madau-Dickinson functional form with Neijssel COMPAS-default fits.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_compas
def test_neijssel_2019_compas_default_sfr_peak_at_z_2p13():
    """COMPAS `find_sfr` defaults peak at z ~ 2.13 +/- 0.10 (Neijssel 2019 Eq. 6 fit).

    Sanity-checks the COMPAS `find_sfr` defaults; the project fiducial
    is the Levina+ 2026 TNG100-1 best-fit (see
    ``test_levina_2026_tng100_sfr_parameters_match_table_1``).

    COMPAS `FastCosmicIntegration.find_sfr` uses the Madau and
    Dickinson (2014) ARA&A 52, 415 functional form

        psi(z) = a * (1+z)^b / (1 + ((1+z)/c)^d)  [Msun yr^-1 Mpc^-3]

    with the Neijssel et al. (2019) MNRAS 490, 3740, Eq. 6 fit
    parameters ``(a, b, c, d) = (0.01, 2.77, 2.9, 4.7)`` instead of the
    Madau-Dickinson Table 1 values ``(0.015, 2.7, 2.9, 5.6)``.  The
    closed-form peak of psi(z) is

        1 + z_peak = c * (b / (d - b))^(1/d)

    which evaluates to 1+z = 2.9 * (2.77/1.93)^(1/4.7) ~ 3.13, i.e.
    z_peak ~ 2.13.  This test pins the COMPAS upstream SFR-slice peak
    to that value and so guards against an upstream coefficient drift
    that would silently re-shape every cosmic-integration rate curve.
    """
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )
    z = np.linspace(0.0, 6.0, 601)
    sfr = np.asarray(fci.find_sfr(z), dtype=float)
    z_peak = float(z[np.argmax(sfr)])
    # Closed-form peak from the differentiation above.
    a, b, c, d = 0.01, 2.77, 2.9, 4.7  # noqa: F841 (Neijssel 2019 Eq. 6 coefficients)
    z_peak_analytic = c * (b / (d - b)) ** (1.0 / d) - 1.0
    assert abs(z_peak - z_peak_analytic) <= 0.10, (
        f"COMPAS SFR peak z = {z_peak:.3f} drifted from the Neijssel "
        f"(2019) Eq. 6 closed-form peak {z_peak_analytic:.3f}; "
        f"upstream coefficients (a, b, c, d) = (0.01, 2.77, 2.9, 4.7) "
        f"may have changed."
    )
    # Sanity bound matches the COMPAS-default location to within the
    # SFR-shape tolerance Neijssel (2019) Sec. 3.2 quotes (~0.1 in z).
    assert 1.95 <= z_peak <= 2.30, z_peak


# ─────────────────────────────────────────────────────────────────────
# Kroupa (2001) MNRAS 322, 231 [papers/Kroupa_2001.pdf]
# IMF slopes (0.3, 1.3, 2.3); mean stellar mass.
# ─────────────────────────────────────────────────────────────────────
def test_kroupa_2001_imf_slopes_and_breakpoints():
    """`kroupa_imf` must use Kroupa (2001) Eq. 2 slopes (0.3, 1.3, 2.3).

    Kroupa (2001) MNRAS 322, 231, Eq. (2):

        m < 0.08 Msun:            xi(m) ~ m^-0.3 (alpha_1 = 0.3)
        0.08 <= m < 0.5 Msun:     xi(m) ~ m^-1.3 (alpha_2 = 1.3)
        m >= 0.5 Msun:            xi(m) ~ m^-2.3 (alpha_3 = 2.3, Salpeter)

    Continuity coefficients (1.0, 0.08, 0.04) follow from imposing
    xi(0.08-) = xi(0.08+) and xi(0.5-) = xi(0.5+).  Test pins the
    slopes by sampling the IMF at three points spanning each segment
    and recovering the local power-law slope by finite differences.
    """
    from grb_rates import kroupa_imf

    masses = [0.04, 0.06]
    f0 = float(kroupa_imf(masses[0]))
    f1 = float(kroupa_imf(masses[1]))
    slope_1 = -np.log(f1 / f0) / np.log(masses[1] / masses[0])
    assert slope_1 == pytest.approx(0.3, abs=1e-3), slope_1

    masses = [0.10, 0.30]
    f0 = float(kroupa_imf(masses[0]))
    f1 = float(kroupa_imf(masses[1]))
    slope_2 = -np.log(f1 / f0) / np.log(masses[1] / masses[0])
    assert slope_2 == pytest.approx(1.3, abs=1e-3), slope_2

    masses = [1.0, 10.0]
    f0 = float(kroupa_imf(masses[0]))
    f1 = float(kroupa_imf(masses[1]))
    slope_3 = -np.log(f1 / f0) / np.log(masses[1] / masses[0])
    assert slope_3 == pytest.approx(2.3, abs=1e-3), slope_3


def test_kroupa_2001_imf_continuity_at_breakpoints():
    """`kroupa_imf` must be continuous at the m = 0.08 and m = 0.5 breakpoints."""
    from grb_rates import kroupa_imf

    # Kroupa Eq. 2 piecewise definition is C^0; the segment coefficients
    # 1.0, 0.08, 0.04 enforce xi(0.08-) = xi(0.08+) etc.
    eps = 1e-9
    f_left_low = float(kroupa_imf(0.08 - eps))
    f_right_low = float(kroupa_imf(0.08))
    assert f_left_low == pytest.approx(f_right_low, rel=1e-3), (
        f"kroupa_imf discontinuous at m = 0.08: {f_left_low} vs {f_right_low}."
    )

    f_left_high = float(kroupa_imf(0.5 - eps))
    f_right_high = float(kroupa_imf(0.5))
    assert f_left_high == pytest.approx(f_right_high, rel=1e-3), (
        f"kroupa_imf discontinuous at m = 0.5: {f_left_high} vs {f_right_high}."
    )


def test_kroupa_2001_mean_stellar_mass_in_published_band():
    """Mean stellar mass from `verify_mean_mass_evolved` must match Kroupa convention.

    The published Kroupa <m> depends on the integration limits.  The
    project's `verify_mean_mass_evolved` defaults to [0.01, 200] Msun
    (brown-dwarf-inclusive), so the expected mean lies in the Kroupa
    (2002) Sci. 295, 82 Table 1 brown-dwarf-inclusive band ~0.36 to
    0.42 Msun.  Restricting to [0.08, 100] Msun (no brown dwarfs)
    gives the more commonly quoted ~0.55 to 0.65 Msun.  The test
    pins the brown-dwarf-inclusive value at the project default
    integration range and the no-brown-dwarf value at the standard
    Kroupa 2001 range.
    """
    from grb_rates import verify_mean_mass_evolved

    bd_inclusive = verify_mean_mass_evolved(
        m_lo_full=0.01,
        m_hi_full=200.0,
        m_lo_prim=5.0,
        m_hi_prim=150.0,
        mean_mass_evolved=1.0,
    )
    mean_bd = bd_inclusive["mean_star_mass"]
    assert 0.30 <= mean_bd <= 0.50, (
        f"Kroupa brown-dwarf-inclusive [0.01, 200] mean = {mean_bd:.3f} "
        f"Msun outside Kroupa (2002) Sci. Table 1 band [0.30, 0.50]."
    )

    no_bd = verify_mean_mass_evolved(
        m_lo_full=0.08,
        m_hi_full=100.0,
        m_lo_prim=5.0,
        m_hi_prim=150.0,
        mean_mass_evolved=1.0,
    )
    mean_nobd = no_bd["mean_star_mass"]
    assert 0.45 <= mean_nobd <= 0.75, (
        f"Kroupa no-brown-dwarf [0.08, 100] mean = {mean_nobd:.3f} "
        f"Msun outside Kroupa (2001) Eq. 2 band [0.45, 0.75]."
    )


def test_broekgaarden_2021_star_forming_mass_per_binary():
    """One simulated binary stands for 79.11 Msun of star formation.

    Broekgaarden+ 2021 (arXiv:2103.02608) Sec. 2.2 and Table 1 draw primaries
    from a Kroupa (2001) IMF over [5, 150] Msun with a flat mass ratio,
    m2 >= 0.1 Msun, and correct to f_bin = 1.  Under the COMPAS
    ``totalMassEvolvedPerZ.analytical_star_forming_mass_per_binary_using_kroupa_imf``
    convention that is 79.110 Msun of Universe star formation per simulated
    binary, and 53 x 1e6 binaries put the run's total at 4.193e9 Msun.

    Pinned because it is the denominator of the Section 5 formation yield.
    Digitising all 53 model-A points of Broekgaarden+ 2022 Fig. 1 implies
    77.2 Msun per binary for NSNS and 78.4 for BHNS, so the band below is
    the published figure's own readback tolerance.
    """
    from grb_rates import simulation_star_forming_mass, star_forming_mass_per_binary

    per_binary = star_forming_mass_per_binary()
    assert per_binary == pytest.approx(79.110, rel=1e-3), (
        f"star_forming_mass_per_binary = {per_binary:.3f} Msun, expected the "
        f"COMPAS analytic 79.110 for the Broekgaarden+ 2021 sampling."
    )
    assert 75.0 <= per_binary <= 82.0, (
        f"{per_binary:.2f} Msun per binary is outside the 75 to 82 band read "
        f"back from Broekgaarden+ 2022 Fig. 1."
    )

    total = simulation_star_forming_mass()
    assert total == pytest.approx(53 * 1.0e6 * per_binary, rel=1e-12)
    assert total == pytest.approx(4.193e9, rel=1e-3)


def test_compas_classmssfr_outer_metallicity_bin_edges():
    """The outer Riemann-cell edges are the COMPAS ClassMSSFR defaults.

    ``ClassMSSFR.__init__`` defaults ``metallicityLowerLimit = 1e-90`` and
    ``metallicityUpperLimit = 1``, and ``calculateMetallicityBinEdges`` places
    them at the two ends of the metallicity sum, so the outermost simulated
    metallicities carry the star formation off the ends of the grid.  Pinned
    because ``dPdlogZ_binned_to_grid`` is what puts the Model A BNS local rate
    on the published ``w_000`` anchor.
    """
    from grb_rates import MSSFR_Z_LOWER_LIMIT, MSSFR_Z_UPPER_LIMIT

    assert MSSFR_Z_LOWER_LIMIT == 1e-90
    assert MSSFR_Z_UPPER_LIMIT == 1.0


def test_dPdlogZ_binned_to_grid_conserves_the_metallicity_probability():
    """Re-binning must move the out-of-grid tails in, not create or lose mass.

    Summed over the 53 grid columns with FCI's own ``1 / (N p_draw)`` weight,
    the re-binned array has to integrate to 1 at every redshift: the bins tile
    ``[MSSFR_Z_LOWER_LIMIT, MSSFR_Z_UPPER_LIMIT]`` without overlap or gap.  The
    pointwise array does not, and the shortfall is exactly the star formation
    outside the simulated range.
    """
    from grb_rates import dPdlogZ_binned_to_grid

    n_grid = 53
    grid = np.geomspace(1e-4, 0.03, n_grid)
    ln_fine = np.arange(-12.0, 0.0, 0.01)
    fine = np.exp(ln_fine)
    p_draw = 1.0 / (np.log(grid[-1]) - np.log(grid[0]))

    # Two log-normals in ln Z, one centred inside the grid and one above its
    # ceiling, so the tail term is large in the second.
    dP = np.stack(
        [
            np.exp(-0.5 * ((ln_fine - mu) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
            for mu in (np.log(1e-3), np.log(0.05))
        ]
    )
    binned = dPdlogZ_binned_to_grid(dP, fine, grid, p_draw)

    cols = np.digitize(grid, fine)
    total = binned[:, cols].sum(axis=1) / (n_grid * p_draw)
    assert np.allclose(total, 1.0, rtol=2e-3), (
        f"re-binned probability {total} should be 1 at every redshift."
    )

    pointwise = dP[:, cols].sum(axis=1) / (n_grid * p_draw)
    assert pointwise[1] < 0.5 * total[1], (
        "the pointwise convention should lose most of a distribution centred "
        "above the grid ceiling; if it does not, the test no longer exercises "
        "the term dPdlogZ_binned_to_grid restores."
    )


def test_verify_mean_mass_evolved_recovers_the_table_1_binary_count():
    """A run-configuration mass must read back as 1e6 binaries per metallicity.

    ``verify_mean_mass_evolved`` divides by the star-forming mass per binary,
    so feeding it ``simulation_star_forming_mass()`` has to return the 53e6
    binaries of Broekgaarden+ 2021 Table 1.  The earlier version divided by
    ``mean_star_mass / f_primary`` = 52.1 Msun, which counts the primary but
    not its companion and so passed on a mass a factor 1.5 too small.
    """
    from grb_rates import simulation_star_forming_mass, verify_mean_mass_evolved

    out = verify_mean_mass_evolved(mean_mass_evolved=simulation_star_forming_mass())
    assert out["N_sim_implied"] / 53.0 == pytest.approx(1.0e6, rel=1e-3), (
        f"{out['N_sim_implied'] / 53.0:.3e} binaries per metallicity, expected "
        f"the 1e6 of Broekgaarden+ 2021 Table 1."
    )


# ─────────────────────────────────────────────────────────────────────
# Wanderman and Piran (2015) MNRAS 448, 3026 [papers/Wanderman_2015.pdf]
# Eq. (9): piecewise-exponential R(z), peak at z = 0.9.
# ─────────────────────────────────────────────────────────────────────
def test_wanderman_piran_2015_piecewise_exponential_continuity_and_slopes():
    """`wanderman_piran_2015_Rz` must be C^0 at z = 0.9 with rising/falling slopes.

    Wanderman and Piran (2015) Eq. (9) is the piecewise-exponential

        R(z) = R0 * exp(+(z - 0.9) / 0.39)   for z <= 0.9
        R(z) = R0 * exp(-(z - 0.9) / 0.26)   for z >  0.9

    so the function value is C^0 at z = 0.9 and the left-of-peak slope
    is positive, right-of-peak slope is negative.  Pins the
    Wanderman-Piran 2015 fit parameters (R0=45 at z=z_peak=0.9,
    sigma_lo=0.39, sigma_hi=0.26).
    """
    from grb_rates import wanderman_piran_2015_Rz

    eps = 1e-6
    z_grid = np.array([0.9 - eps, 0.9 + eps])
    out = wanderman_piran_2015_Rz(z_grid)
    R = out["R_best"]
    assert abs(R[0] - R[1]) / R[0] < 1e-4, (
        f"wanderman_piran_2015_Rz is not C^0 at z=0.9: left {R[0]}, right {R[1]}."
    )

    z_lo = np.array([0.5, 0.7, 0.9])
    R_lo = wanderman_piran_2015_Rz(z_lo)["R_best"]
    assert (np.diff(R_lo) > 0).all(), (
        f"wanderman_piran_2015_Rz must be rising for z <= 0.9; got diffs {np.diff(R_lo)}."
    )
    z_hi = np.array([0.9, 1.5, 3.0])
    R_hi = wanderman_piran_2015_Rz(z_hi)["R_best"]
    assert (np.diff(R_hi) < 0).all(), (
        f"wanderman_piran_2015_Rz must be falling for z >= 0.9; got diffs {np.diff(R_hi)}."
    )


def test_wanderman_piran_2015_R0_normalization_within_band():
    """Peak `R0 = 45 Gpc^-3 yr^-1` (WP15 Eq. 9) and local rate matches paper rho_0.

    WP15 Eq. 9 sets the peak-of-curve normalization at R0 = 45 Gpc^-3
    yr^-1 (z_peak = 0.9); the local rate R(z=0) = 45 * exp(-0.9/0.39)
    ~ 4.47 Gpc^-3 yr^-1 sits inside the paper's Eq. 10 canonical
    rho_0 = 4.1 (+1.8/-1.4) +/- 0.5 Gpc^-3 yr^-1 (internal 1sigma
    plus SFR1<->SFR2 systematic).  The R_rate_lo / R_rate_hi arrays
    propagate rho_0's *internal* 1sigma (+1.8/-1.4) onto the curve.
    """
    from grb_rates import wanderman_piran_2015_Rz

    R_peak = float(wanderman_piran_2015_Rz(np.array([0.9]))["R_best"][0])
    assert R_peak == pytest.approx(45.0, rel=1e-9), R_peak

    R_z0 = float(wanderman_piran_2015_Rz(np.array([0.0]))["R_best"][0])
    assert 2.7 <= R_z0 <= 5.9, (
        f"R(z=0) = {R_z0:.3f} outside WP15 Eq. 10 rho_0 internal 1sigma [2.7, 5.9]."
    )

    R_peak_lo = float(wanderman_piran_2015_Rz(np.array([0.9]))["R_rate_lo"][0])
    R_peak_hi = float(wanderman_piran_2015_Rz(np.array([0.9]))["R_rate_hi"][0])
    assert R_peak_lo == pytest.approx(45.0 * 2.7 / 4.1, rel=1e-3)
    assert R_peak_hi == pytest.approx(45.0 * 5.9 / 4.1, rel=1e-3)


def test_wanderman_piran_2015_sfr2_shape_broadens_peak():
    """SFR2 shape systematic broadens the peak by 1.5x (WP15 Sec. 5).

    WP15 Sec. 5 quotes peak FWHM Delta_z ~ 0.4 for SFR1 and ~ 0.6 for
    SFR2, so the SFR-model shape systematic broadens sigma by
    0.6/0.4 = 1.5x.  The broadened curve equals R_best at the peak
    (same R0 = 45) and exceeds R_best in the wings.
    """
    from grb_rates import wanderman_piran_2015_Rz

    out = wanderman_piran_2015_Rz(np.array([0.9, 0.2, 2.5]))
    # At the peak both shapes give R0.
    assert out["R_shape_sfr2"][0] == pytest.approx(out["R_best"][0], rel=1e-9)
    # In the rising wing (z = 0.2 << z_peak) the broadened shape is higher.
    assert out["R_shape_sfr2"][1] > out["R_best"][1]
    # Same for the falling wing (z = 2.5 >> z_peak).
    assert out["R_shape_sfr2"][2] > out["R_best"][2]


# ─────────────────────────────────────────────────────────────────────
# Pracchia & Salafia (2026) A&A, arXiv:2601.03861
#   [papers/Pracchia_2026.pdf]
# Eq. 9 numerical convolution of Madau & Fragos 2017 SFR (Eq. 10) with
# a DTD kernel (Eq. 7 power-law or Eq. 8 log-normal).  Table 2 posteriors.
# ─────────────────────────────────────────────────────────────────────
def test_pracchia_salafia_2026_table2_posteriors_within_paper_band():
    """Pin all four Table 2 rows to the paper's 90 percent CR posteriors.

    Pracchia and Salafia (2026) Table 2 reports log10(R_0 / Gpc^-3 yr^-1)
    plus DTD parameters for (ELF, QUSJ) x (power-law, log-normal DTD).
    R_0 is intrinsic source-frame, per merger, all-orientation.
    """
    from grb_rates import (
        MADAU_FRAGOS_2017_SFR,
        PRACCHIA_SALAFIA_2026_ELF_LOGNORMAL,
        PRACCHIA_SALAFIA_2026_ELF_POWERLAW,
        PRACCHIA_SALAFIA_2026_QUSJ_LOGNORMAL,
        PRACCHIA_SALAFIA_2026_QUSJ_POWERLAW,
        PRACCHIA_SALAFIA_2026_ROWS,
    )

    # ELF power-law: log10 R_0 = 1.92 +1.50/-0.92, alpha_tau = 3.20,
    # tau_min = 0.03 Gyr, <tau_d> = 77 Myr.
    p = PRACCHIA_SALAFIA_2026_ELF_POWERLAW
    assert p["log10_R0"] == 1.92
    assert p["log10_R0_plus90"] == 1.50
    assert p["log10_R0_minus90"] == 0.92
    assert p["alpha_tau"] == 3.20
    assert p["tau_min_Gyr"] == 0.03
    assert p["mean_tau_d_Myr"] == 77

    # ELF log-normal: log10 R_0 = 2.60 +1.23/-1.49, mu_tau = 0.08 Gyr,
    # sigma = 0.11, <tau_d> = 92 Myr.
    p = PRACCHIA_SALAFIA_2026_ELF_LOGNORMAL
    assert p["log10_R0"] == 2.60
    assert p["log10_R0_plus90"] == 1.23
    assert p["log10_R0_minus90"] == 1.49
    assert p["mu_tau_Gyr"] == 0.08
    assert p["sigma_tau"] == 0.11
    assert p["mean_tau_d_Myr"] == 92

    # QUSJ power-law: log10 R_0 = 2.57 +0.67/-0.63, alpha_tau = 3.33,
    # tau_min = 0.04 Gyr, <tau_d> = 105 Myr.
    p = PRACCHIA_SALAFIA_2026_QUSJ_POWERLAW
    assert p["log10_R0"] == 2.57
    assert p["log10_R0_plus90"] == 0.67
    assert p["log10_R0_minus90"] == 0.63
    assert p["alpha_tau"] == 3.33
    assert p["tau_min_Gyr"] == 0.04
    assert p["mean_tau_d_Myr"] == 105

    # QUSJ log-normal: log10 R_0 = 2.25 +0.53/-0.50, mu = 0.09 Gyr,
    # sigma = 0.14, <tau_d> = 116 Myr.
    p = PRACCHIA_SALAFIA_2026_QUSJ_LOGNORMAL
    assert p["log10_R0"] == 2.25
    assert p["log10_R0_plus90"] == 0.53
    assert p["log10_R0_minus90"] == 0.50
    assert p["mu_tau_Gyr"] == 0.09
    assert p["sigma_tau"] == 0.14
    assert p["mean_tau_d_Myr"] == 116

    # Registry keys match the four labelled rows.
    assert set(PRACCHIA_SALAFIA_2026_ROWS.keys()) == {
        "ELF_powerlaw",
        "ELF_lognormal",
        "QUSJ_powerlaw",
        "QUSJ_lognormal",
    }

    # Madau and Fragos (2017); Pracchia and Salafia Eq. 10:
    # (a_psi, b_psi, z_psi) = (2.6, 6.2, 2.2).
    assert MADAU_FRAGOS_2017_SFR["a_psi"] == 2.6
    assert MADAU_FRAGOS_2017_SFR["b_psi"] == 6.2
    assert MADAU_FRAGOS_2017_SFR["z_psi"] == 2.2


def test_pracchia_salafia_2026_Rz_local_rate_matches_R0():
    """R(z=0) equals 10^log10_R0 (Table 2) for every row at rel 1e-6.

    The numerical convolution is normalised so R(z=0) hits the fitted
    local rate exactly; this pins the calibration.
    """
    from grb_rates import PRACCHIA_SALAFIA_2026_ROWS, pracchia_salafia_2026_Rz

    for row, params in PRACCHIA_SALAFIA_2026_ROWS.items():
        R_0 = 10.0 ** params["log10_R0"]
        out = pracchia_salafia_2026_Rz(np.array([0.0]), row=row)
        assert out["R_best"][0] == pytest.approx(R_0, rel=1e-6), (
            f"{row}: R(z=0) = {out['R_best'][0]:.4f} != R_0 = {R_0:.4f}"
        )
        # 90 percent CR bounds scale R_0 by 10^plus90 / 10^minus90.
        assert out["R_rate_hi"][0] == pytest.approx(
            10.0 ** (params["log10_R0"] + params["log10_R0_plus90"]), rel=1e-6
        )
        assert out["R_rate_lo"][0] == pytest.approx(
            10.0 ** (params["log10_R0"] - params["log10_R0_minus90"]), rel=1e-6
        )


def test_pracchia_salafia_2026_Rz_shape_tracks_short_DTD_sfr():
    """Short-DTD R(z) tracks SFR(z) within 15 percent for z <= 3.

    Table 2 rows all have <tau_d> = 77 to 116 Myr (short compared to the
    Madau and Fragos SFR variation timescale), so R(z) ~ SFR(z) up to a
    normalisation, and the peak of R(z) sits near the SFR peak at z ~ 1.9.
    Both are load-bearing shape features: a bug in the DTD kernel or the
    tau-substitution integrator would push the peak off or under-resolve
    the low-z rise.
    """
    from grb_rates import _madau_fragos_2017_sfr, pracchia_salafia_2026_Rz

    z_grid = np.linspace(0.0, 5.0, 200)
    out = pracchia_salafia_2026_Rz(z_grid, row="ELF_powerlaw")
    sfr = (
        _madau_fragos_2017_sfr(z_grid)
        * out["R_best"][0]
        / _madau_fragos_2017_sfr(np.array([0.0]))[0]
    )
    zpeak = z_grid[np.argmax(out["R_best"])]
    assert 1.7 <= zpeak <= 2.3, f"R(z) peak = {zpeak:.2f}; expected ~1.9 (SFR peak)"
    ratio = out["R_best"][z_grid <= 3.0] / sfr[z_grid <= 3.0]
    assert np.all(np.abs(ratio - 1.0) < 0.15), (
        f"R/SFR for z<=3 deviates by more than 15 percent; got range "
        f"[{ratio.min():.3f}, {ratio.max():.3f}]."
    )


def test_pracchia_salafia_2026_Rz_cosmology_drift_under_015_percent():
    """P&S curve is < 0.15 percent-sensitive to cosmology choice.

    Pracchia and Salafia (2026) do not quote H0/Omega_m explicitly and
    inherit their cosmology from Madau and Fragos (2017), who use the
    Madau and Dickinson (2014) "concordance" (H0=70, Omega_m=0.3).  The
    project pins Planck 2015 (H0=67.74, Omega_m=0.3089).  Because
    R(z=0) is normalised to Table 2's R_0 exactly, the absolute
    normalisation is insensitive to H0; only the look-back-time detail
    inside the convolution kernel changes.  With <tau_d> < 120 Myr the
    drift across all three tested cosmologies is bounded well under
    0.15 percent and the peak location is invariant at z=2.007.
    """
    from grb_rates import (
        COSMO_MADAU14,
        COSMO_PLANCK14,
        COSMO_PLANCK15,
        pracchia_salafia_2026_Rz,
    )

    z_grid = np.linspace(0.0, 3.0, 50)
    R_p15 = pracchia_salafia_2026_Rz(z_grid, row="ELF_powerlaw", cosmology=COSMO_PLANCK15)["R_best"]
    R_p14 = pracchia_salafia_2026_Rz(z_grid, row="ELF_powerlaw", cosmology=COSMO_PLANCK14)["R_best"]
    R_md14 = pracchia_salafia_2026_Rz(z_grid, row="ELF_powerlaw", cosmology=COSMO_MADAU14)["R_best"]
    for label, R_other in (("Planck2014", R_p14), ("Madau2014", R_md14)):
        drift = np.max(np.abs((R_other - R_p15) / R_p15))
        assert drift < 0.0015, (
            f"P&S R(z) drift from Planck2015 to {label} = {drift * 100:.3f}% "
            f"exceeds 0.15 percent bound."
        )
    # Peak invariance under cosmology swap.
    zpk_p15 = z_grid[np.argmax(R_p15)]
    zpk_md14 = z_grid[np.argmax(R_md14)]
    assert abs(zpk_p15 - zpk_md14) < 0.1, (
        f"P&S peak z drifts from {zpk_p15:.3f} (Planck 2015) to {zpk_md14:.3f} "
        f"(Madau 2014); expected invariant to <0.1."
    )


# ─────────────────────────────────────────────────────────────────────
# Du et al. (2025) ApJL 987, L13, DOI 10.3847/2041-8213/ade1d0
#   [papers/Du_2025.pdf]
# Table 5 observer-frame local rates per detector across five SFR + DTD
# combinations; Sec. 4.4 beaming.  Eq. 16 coefficients not published.
# ─────────────────────────────────────────────────────────────────────
def test_du25_swift_bat_table5_rates_match_paper():
    """Pin every Swift/BAT rate in Table 5 (Gpc^-3 yr^-1, +/- 1sigma).

    Du et al. (2025) Table 5 Swift/BAT row: five SFR + DTD combinations
    (Yuksel+ 2008 undelayed, Gauss-Yuksel, Lognormal-Yuksel, PowerLaw-Yuksel,
    Madau+ 2014 undelayed) with L_min = 2.57e48 erg/s and theta_j = 9.8 deg
    (Sec. 4.4; citing Soderberg+ 2006 and Zhang+ 2018).
    """
    from grb_rates import (
        DU25_L_MIN_SWIFT_BAT_ERG_S,
        DU25_LOCAL_RATES_SWIFT_BAT,
        DU25_THETA_J_DEG,
    )

    expected = {
        "Yuksel2008_undelayed": (0.60, 0.18),
        "Gauss_Yuksel": (0.56, 0.16),
        "Lognormal_Yuksel": (0.57, 0.15),
        "PowerLaw_Yuksel": (0.97, 0.26),
        "Madau2014_undelayed": (0.73, 0.18),
    }
    assert set(DU25_LOCAL_RATES_SWIFT_BAT.keys()) == set(expected.keys())
    for k, (r, err) in expected.items():
        assert DU25_LOCAL_RATES_SWIFT_BAT[k] == pytest.approx((r, err), rel=1e-9), (
            f"{k}: got {DU25_LOCAL_RATES_SWIFT_BAT[k]}, expected ({r}, {err})"
        )
    assert DU25_L_MIN_SWIFT_BAT_ERG_S == pytest.approx(2.57e48, rel=1e-9)
    assert DU25_THETA_J_DEG == pytest.approx(9.8, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────
# Chong-Yu Gao (2025) MNRAS 542, 1841, DOI 10.1093/mnras/staf1368
#   [papers/Gao_2025.pdf]
# Table 2 Gaussian-DTD row (DIC-preferred); Fermi/GBM local rate.
# ─────────────────────────────────────────────────────────────────────
def test_gao25_fermi_gbm_gaussian_dtd_local_rate_matches_table2():
    """Pin the Gaussian-DTD local rate R(z=0) = 1.37 +0.30/-0.27.

    Chong-Yu Gao (2025) Table 2 Gaussian-DTD row: DIC-preferred (DIC =
    94996 vs 95053/95058 for log-normal/power-law).  Fermi/GBM sample,
    N = 478 sGRBs.  tau_G = 1.14 Gyr central (posterior +0.02/-0.17);
    sigma_G < 0.26 Gyr is the Table 2 caption upper bound.
    """
    from grb_rates import GAO25_LOCAL_RATE_FERMI_GBM

    p = GAO25_LOCAL_RATE_FERMI_GBM
    assert p["R0"] == pytest.approx(1.37, rel=1e-9)
    assert p["R0_plus"] == pytest.approx(0.30, rel=1e-9)
    assert p["R0_minus"] == pytest.approx(0.27, rel=1e-9)
    assert p["tau_G_Gyr"] == pytest.approx(1.14, rel=1e-9)
    assert p["sigma_G_Gyr"] == pytest.approx(0.26, rel=1e-9)
    assert p["detector"] == "Fermi/GBM"


# ─────────────────────────────────────────────────────────────────────
# Fong et al. (2015) ApJ 815, 102 [papers/Fong_2015.pdf]
# Jet half-opening angle bands per class.
# ─────────────────────────────────────────────────────────────────────
def test_class_theta_j_sbgrb_band_is_fong_2015_median_and_1sigma():
    """`CLASS_THETA_J['sbGRB']` is the Fong et al. (2015) median, 16 +/- 10 deg.

    Fong et al. (2015) Sec. 5.2 and Fig. 8: 11 short GRBs, four with
    jet-break measurements and seven with lower limits, the limits spread
    uniformly up to an assumed theta_j,max = 30 deg, give a median of
    "16 +/- 10 deg (1 sigma)".  The band is that median and its 1 sigma
    edges, unmodified.
    """
    from grb_rates import CLASS_THETA_J

    sb = CLASS_THETA_J["sbGRB"]
    assert sb["fid"] == pytest.approx(16.0, abs=1e-12), sb["fid"]
    assert sb["lo"] == pytest.approx(6.0, abs=1e-12), sb["lo"]
    assert sb["hi"] == pytest.approx(26.0, abs=1e-12), sb["hi"]


def test_class_theta_j_lbgrb_band_is_narrower_than_sbgrb():
    """`CLASS_THETA_J['lbGRB']` is the project's 5 to 8 deg band.

    CODE HEURISTIC; not from any single paper.  Gottlieb et al. (2023)
    Sec. 3 and Gottlieb et al. (2024) Eq. 10 both adopt the Fong et al.
    (2015) beaming range 0.01 <= f_b <= 0.11 for every class, so they do
    not supply a narrower lbGRB angle.  The band encodes the expectation
    that BH-engine jets are more collimated than HMNS-engine jets; this
    test pins the numbers and that ordering, nothing more.
    """
    from grb_rates import CLASS_THETA_J

    lb = CLASS_THETA_J["lbGRB"]
    assert lb["lo"] == pytest.approx(5.0, abs=1e-12), lb["lo"]
    assert lb["fid"] == pytest.approx(6.5, abs=1e-12), lb["fid"]
    assert lb["hi"] == pytest.approx(8.0, abs=1e-12), lb["hi"]
    assert lb["fid"] < CLASS_THETA_J["sbGRB"]["fid"]


def test_fong_2015_beaming_factor_at_fiducial_sbgrb_angle():
    """f_beam at the sbGRB fiducial reproduces Fong et al. (2015) f_b ~ 0.04.

    Fong Sec. 5.2 quote the median beaming factor as f_b = 0.04 (+0.07 /
    -0.03) and its inverse as f_b^-1 = 27 (+158 / -18).  1 - cos(16 deg) =
    0.0387, which rounds to their 0.04 and inverts to 25.8, inside the
    rounding of their 27.  The 1 sigma edges are checked loosely because
    Fong et al. round the angle and the beaming factor independently: the
    6 deg edge gives 0.0055 against their 1/185 = 0.0054, and the 26 deg edge
    gives 0.101 against their 0.11.
    """
    from grb_rates import CLASS_THETA_J, beamed_rate

    sb = CLASS_THETA_J["sbGRB"]
    f_fid = float(beamed_rate(1.0, sb["fid"]))
    f_lo = float(beamed_rate(1.0, sb["lo"]))
    f_hi = float(beamed_rate(1.0, sb["hi"]))
    assert f_fid == pytest.approx(0.04, abs=0.005), f_fid
    assert 1.0 / f_fid == pytest.approx(27.0, abs=1.5), 1.0 / f_fid
    assert f_lo == pytest.approx(1.0 / (27.0 + 158.0), rel=0.05), f_lo
    assert f_hi == pytest.approx(0.04 + 0.07, abs=0.01), f_hi


# ─────────────────────────────────────────────────────────────────────
# Fong et al. (2022) ApJ 940, 56 [arXiv:2206.01763]
# Ghirlanda et al. (2016) A&A 594, A84
# Swift-BAT observing envelope + effective sky fraction, used for the
# Section 6c BRIGHT counts-to-rate-density conversion.
# ─────────────────────────────────────────────────────────────────────
def test_swift_bright_window_matches_fong_2022_sample_span():
    """`T_SWIFT_BRIGHT_YR` must span the Fong+ 2022 BRIGHT catalogue window.

    Fong et al. (2022, ApJ 940, 56, arXiv:2206.01763) Sec. 3 defines
    the BRIGHT sample as "107 events discovered over 2004-2021 which
    meet the criteria for our starting sample" (p. 4), i.e. from Swift's
    launch in November 2004 through the sample cutoff at December 2021.
    That envelope is ~17.1 detector-frame years; allowance kept for the
    exact launch-to-cutoff arithmetic and any downstream re-cut.
    """
    from grb_rates import T_SWIFT_BRIGHT_YR

    assert 16.5 <= T_SWIFT_BRIGHT_YR <= 17.5, (
        f"T_SWIFT_BRIGHT_YR = {T_SWIFT_BRIGHT_YR} yr is outside the "
        f"Fong+ 2022 BRIGHT window [16.5, 17.5] yr."
    )


def test_bat_effective_sky_fraction_matches_ghirlanda_2016():
    """`F_SKY_BAT` must match the Ghirlanda+ 2016 effective BAT sky fraction.

    Ghirlanda et al. (2016, A&A 594, A84) Sec. 3.2 adopt an effective
    Swift-BAT sky coverage of Omega_BAT / 4pi ~ 1/6 for their
    observed-sGRB rate integral (partial coding weighted).  The
    Lien et al. (2016, ApJ 829, 7) Swift-BAT triggered-GRB catalog
    gives a compatible number.  The plausible published range spans
    roughly [0.11, 0.20]; project fiducial is 1/6.
    """
    from grb_rates import F_SKY_BAT

    assert 0.11 <= F_SKY_BAT <= 0.20, (
        f"F_SKY_BAT = {F_SKY_BAT} is outside the published BAT effective "
        f"sky-fraction band [0.11, 0.20]."
    )
    assert F_SKY_BAT == pytest.approx(1.0 / 6.0, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────
# Kawaguchi et al. (2015) ApJ 825, 52 [papers/Kawaguchi_2015.pdf]
# Fragos et al. (2010) misalignment population.
# ─────────────────────────────────────────────────────────────────────
def test_kawaguchi_2015_misalignment_factor_in_physical_band():
    """`MISALIGNMENT_SYSTEMATIC_FACTOR = 0.5` must be in [0.4, 0.7].

    Kawaguchi et al. (2015) ApJ 825, 52 Fig. 4 show BHNS disk mass
    drops to near zero for misalignment angles > 50-60 deg.  Fragos
    et al. (2010) and Gerosa et al. (2018) population-synthesis tilt
    distributions imply roughly half of BHNS systems exceed 45 deg
    misalignment, motivating a population-averaged suppression
    factor near 0.5.  The plausible range admitted by these inputs
    is [0.4, 0.7] (e.g. Gerosa et al. 2018 Fig. 7 spread).
    """
    from grb_physics import MISALIGNMENT_SYSTEMATIC_FACTOR

    assert 0.4 <= MISALIGNMENT_SYSTEMATIC_FACTOR <= 0.7, (
        f"MISALIGNMENT_SYSTEMATIC_FACTOR = {MISALIGNMENT_SYSTEMATIC_FACTOR} "
        f"is outside the Kawaguchi+ 2015 / Fragos+ 2010 plausible "
        f"population band [0.4, 0.7]."
    )


def test_kawaguchi_2015_aligned_spin_projection_matches_paper_definition():
    """`effective_aligned_spin(a, theta) = max(0, a * cos(theta))` (Kawaguchi 2015)."""
    from grb_physics import effective_aligned_spin

    chi = 0.7
    for theta in [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3]:
        expected = max(0.0, chi * np.cos(theta))
        got = float(effective_aligned_spin(chi, theta))
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-12), (
            f"effective_aligned_spin({chi}, {theta}) = {got} != "
            f"max(0, a cos theta) = {expected} (Kawaguchi 2015 Sec. 4)."
        )


# ─────────────────────────────────────────────────────────────────────
# Margalit and Metzger (2017) ApJL 850, L19 [papers/Margalit_2017.pdf]
# HMNS supramassive remnant heuristic.
# ─────────────────────────────────────────────────────────────────────
def test_margalit_metzger_2017_hmns_factor_in_supramassive_band():
    """`HMNS_FACTOR_DEFAULT = 1.2` must lie in the [1.0, 1.3] Margalit-Metzger band.

    Margalit and Metzger (2017) ApJL 850, L19 Sec. 2 argue
    supramassive remnants with M significantly above M_TOV but below
    the prompt-collapse threshold collapse on viscous timescales (not
    long-lived).  The "significantly above" cut translates to a
    multiplier in [1.0, 1.3] times M_TOV; the project pins 1.2 as the
    Gottlieb (2024) sbGRB / lbGRB+HMNS fiducial.
    """
    from grb_physics import HMNS_FACTOR_DEFAULT

    assert 1.0 <= HMNS_FACTOR_DEFAULT <= 1.3, (
        f"HMNS_FACTOR_DEFAULT = {HMNS_FACTOR_DEFAULT} is outside the "
        f"Margalit-Metzger (2017) supramassive-remnant band [1.0, 1.3]."
    )


# ─────────────────────────────────────────────────────────────────────
# Levina et al. (2026) arXiv:2601.20202 [papers/Levina_2026.pdf]
# Table 1: TNG100-1 best-fit S(Z, z) parameters (project fiducial).
# Eq. (2) Madau and Dickinson (2014) S(z), Eq. (3-6) Azzalini skew-log-normal
# dP/dlnZ; the COMPAS ``find_metallicity_distribution`` parametrisation maps
# Levina's omega_0 / omega_z onto COMPAS sigma_0 / sigma_z with alpha as the
# skewness parameter.
# ─────────────────────────────────────────────────────────────────────
def test_levina_2026_tng100_mssfr_parameters_match_table_1():
    """MSSFR_PARAMS_LEVINA26_TNG100 must match Levina+ 2026 Table 1, TNG100-1 column."""
    from grb_rates import MSSFR_PARAMS_LEVINA26_TNG100 as P

    assert P["mu0"] == pytest.approx(0.0247, rel=1e-9), P["mu0"]
    assert P["muz"] == pytest.approx(-0.0521, rel=1e-9), P["muz"]
    assert P["sigma_0"] == pytest.approx(1.1509, rel=1e-9), P["sigma_0"]
    assert P["sigma_z"] == pytest.approx(0.0477, rel=1e-9), P["sigma_z"]
    assert P["alpha"] == pytest.approx(-1.8801, rel=1e-9), P["alpha"]


def test_levina_2026_tng100_sfr_parameters_match_table_1():
    """SFR_PARAMS_LEVINA26_TNG100 must match Levina+ 2026 Table 1, TNG100-1 column."""
    from grb_rates import SFR_PARAMS_LEVINA26_TNG100 as S

    assert S["a"] == pytest.approx(0.0172, rel=1e-9), S["a"]
    assert S["b"] == pytest.approx(1.4425, rel=1e-9), S["b"]
    assert S["c"] == pytest.approx(4.5299, rel=1e-9), S["c"]
    assert S["d"] == pytest.approx(6.2261, rel=1e-9), S["d"]


@pytest.mark.requires_compas
def test_levina_2026_tng100_sfr_peak_around_z_2p7():
    """Levina+ 2026 TNG100-1 SFR peak sits at z ~ 2.74.

    The TNG-fit d = 6.2261 (vs Neijssel d = 4.7) and c = 4.53 (vs 2.9)
    together with the shallower low-z slope b = 1.44 push the peak of
    ``a*(1+z)^b / (1 + ((1+z)/c)^d)`` to z ~ 2.7, later than Madau-Fragos
    or the Neijssel COMPAS default.  Anchored to the closed-form peak
    ``z_peak = c * (b/(d-b))**(1/d) - 1`` so a drift in any of the four
    parameters surfaces here, not deep inside a rate calculation.
    """
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import find_sfr

    from grb_rates import SFR_PARAMS_LEVINA26_TNG100

    z = np.linspace(0.0, 8.0, 4001)
    sfr = find_sfr(z, **SFR_PARAMS_LEVINA26_TNG100)
    z_peak = float(z[np.argmax(sfr)])
    a, b, c, d = (  # noqa: F841
        SFR_PARAMS_LEVINA26_TNG100["a"],
        SFR_PARAMS_LEVINA26_TNG100["b"],
        SFR_PARAMS_LEVINA26_TNG100["c"],
        SFR_PARAMS_LEVINA26_TNG100["d"],
    )
    z_peak_analytic = c * (b / (d - b)) ** (1.0 / d) - 1.0
    assert abs(z_peak - z_peak_analytic) <= 0.10, (z_peak, z_peak_analytic)
    assert 2.5 <= z_peak <= 3.0, z_peak


# ─────────────────────────────────────────────────────────────────────
# Levina+ 2026 Table 1: TNG50-1 and TNG300-1 columns.
# Levina+ 2026 Table 2: published BBH local merger rates.
# ─────────────────────────────────────────────────────────────────────
def test_levina_2026_tng50_parameters_match_table_1():
    """SFR_PARAMS_LEVINA26_TNG50 / MSSFR_PARAMS_LEVINA26_TNG50 match Levina+ 2026 Table 1."""
    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG50,
        SFR_PARAMS_LEVINA26_TNG50,
    )

    assert SFR_PARAMS_LEVINA26_TNG50["a"] == pytest.approx(0.0329, rel=1e-9)
    assert SFR_PARAMS_LEVINA26_TNG50["b"] == pytest.approx(1.4668, rel=1e-9)
    assert SFR_PARAMS_LEVINA26_TNG50["c"] == pytest.approx(3.8412, rel=1e-9)
    assert SFR_PARAMS_LEVINA26_TNG50["d"] == pytest.approx(5.0994, rel=1e-9)

    assert MSSFR_PARAMS_LEVINA26_TNG50["mu0"] == pytest.approx(0.0282, rel=1e-9)
    assert MSSFR_PARAMS_LEVINA26_TNG50["muz"] == pytest.approx(-0.0314, rel=1e-9)
    assert MSSFR_PARAMS_LEVINA26_TNG50["sigma_0"] == pytest.approx(1.1136, rel=1e-9)
    assert MSSFR_PARAMS_LEVINA26_TNG50["sigma_z"] == pytest.approx(0.0592, rel=1e-9)
    assert MSSFR_PARAMS_LEVINA26_TNG50["alpha"] == pytest.approx(-1.7353, rel=1e-9)


def test_levina_2026_tng300_parameters_match_table_1():
    """SFR_PARAMS_LEVINA26_TNG300 / MSSFR_PARAMS_LEVINA26_TNG300 match Levina+ 2026 Table 1."""
    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG300,
        SFR_PARAMS_LEVINA26_TNG300,
    )

    assert SFR_PARAMS_LEVINA26_TNG300["a"] == pytest.approx(0.0097, rel=1e-9)
    assert SFR_PARAMS_LEVINA26_TNG300["b"] == pytest.approx(1.5747, rel=1e-9)
    assert SFR_PARAMS_LEVINA26_TNG300["c"] == pytest.approx(4.5428, rel=1e-9)
    assert SFR_PARAMS_LEVINA26_TNG300["d"] == pytest.approx(6.8266, rel=1e-9)

    assert MSSFR_PARAMS_LEVINA26_TNG300["mu0"] == pytest.approx(0.0237, rel=1e-9)
    assert MSSFR_PARAMS_LEVINA26_TNG300["muz"] == pytest.approx(-0.0687, rel=1e-9)
    assert MSSFR_PARAMS_LEVINA26_TNG300["sigma_0"] == pytest.approx(1.1196, rel=1e-9)
    assert MSSFR_PARAMS_LEVINA26_TNG300["sigma_z"] == pytest.approx(0.0481, rel=1e-9)
    assert MSSFR_PARAMS_LEVINA26_TNG300["alpha"] == pytest.approx(-2.2726, rel=1e-9)


def test_levina_2026_bbh_local_rates_match_table_2():
    """LEVINA26_BBH_LOCAL_RATES match the six numbers in Levina+ 2026 Table 2."""
    from grb_rates import LEVINA26_BBH_LOCAL_RATES

    expected = {
        "TNG50-1": {"R_sim": 58.92, "R_fit": 73.72},
        "TNG100-1": {"R_sim": 42.91, "R_fit": 45.53},
        "TNG300-1": {"R_sim": 29.34, "R_fit": 27.81},
    }
    for tng, vals in expected.items():
        assert tng in LEVINA26_BBH_LOCAL_RATES
        for key, val in vals.items():
            assert LEVINA26_BBH_LOCAL_RATES[tng][key] == pytest.approx(val, rel=1e-9), (tng, key)


def test_LEVINA26_TNG_VARIANTS_dict_structure_and_pairings():
    """`LEVINA26_TNG_VARIANTS` must pair each TNG run with its own
    (SFR_PARAMS, MSSFR_PARAMS) constants.

    Levina et al. (2026), arXiv:2601.20202, Table 1 fits the
    Madau-Dickinson SFR and the Azzalini skew-log-normal metallicity
    PDF separately per TNG resolution (TNG50-1, TNG100-1, TNG300-1).
    The convenience dict `LEVINA26_TNG_VARIANTS` groups them as
    ``{tng: (SFR_PARAMS_LEVINA26_<tng>, MSSFR_PARAMS_LEVINA26_<tng>)}``
    so the Section 4b notebook iterates one MSSFR per simulation
    consistently.  A future swap of the SFR / MSSFR identity (e.g.
    pairing TNG100 SFR with TNG50 MSSFR) would silently shift every
    rate in the resolution sweep; pinning the object identity here
    catches that.
    """
    from grb_rates import (
        LEVINA26_TNG_VARIANTS,
        MSSFR_PARAMS_LEVINA26_TNG50,
        MSSFR_PARAMS_LEVINA26_TNG100,
        MSSFR_PARAMS_LEVINA26_TNG300,
        SFR_PARAMS_LEVINA26_TNG50,
        SFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG300,
    )

    assert set(LEVINA26_TNG_VARIANTS.keys()) == {"TNG50-1", "TNG100-1", "TNG300-1"}

    expected = {
        "TNG50-1": (SFR_PARAMS_LEVINA26_TNG50, MSSFR_PARAMS_LEVINA26_TNG50),
        "TNG100-1": (SFR_PARAMS_LEVINA26_TNG100, MSSFR_PARAMS_LEVINA26_TNG100),
        "TNG300-1": (SFR_PARAMS_LEVINA26_TNG300, MSSFR_PARAMS_LEVINA26_TNG300),
    }
    for tng, (sfr_expected, mssfr_expected) in expected.items():
        sfr_got, mssfr_got = LEVINA26_TNG_VARIANTS[tng]
        assert sfr_got is sfr_expected, (
            f"LEVINA26_TNG_VARIANTS[{tng!r}][0] is not the same dict as "
            f"SFR_PARAMS_LEVINA26_{tng.replace('-1', '').replace('TNG', 'TNG')}"
        )
        assert mssfr_got is mssfr_expected, (
            f"LEVINA26_TNG_VARIANTS[{tng!r}][1] is not the same dict as "
            f"MSSFR_PARAMS_LEVINA26_{tng.replace('-1', '').replace('TNG', 'TNG')}"
        )


def test_levina_tng_resolution_monotonic_R_local_under_analytical_fit():
    """Levina+ 2026 Sec. 3.2: BBH local rate decreases with simulation
    box size (TNG50 highest resolution and rate, TNG300 largest box and
    lowest rate).  Anchored on the constants so the test runs without
    BBH data; the BNS forward-pass version is in
    ``tests/sections/test_section_04_mssfr.py``.
    """
    from grb_rates import LEVINA26_BBH_LOCAL_RATES

    R_50 = LEVINA26_BBH_LOCAL_RATES["TNG50-1"]["R_fit"]
    R_100 = LEVINA26_BBH_LOCAL_RATES["TNG100-1"]["R_fit"]
    R_300 = LEVINA26_BBH_LOCAL_RATES["TNG300-1"]["R_fit"]
    assert R_50 > R_100 > R_300, (R_50, R_100, R_300)

    R_50_sim = LEVINA26_BBH_LOCAL_RATES["TNG50-1"]["R_sim"]
    R_100_sim = LEVINA26_BBH_LOCAL_RATES["TNG100-1"]["R_sim"]
    R_300_sim = LEVINA26_BBH_LOCAL_RATES["TNG300-1"]["R_sim"]
    assert R_50_sim > R_100_sim > R_300_sim, (R_50_sim, R_100_sim, R_300_sim)


# ─────────────────────────────────────────────────────────────────────
# LVK GWTC-5.0 population: LVK 2026, arXiv:2605.27226.
# ─────────────────────────────────────────────────────────────────────
def test_lvk_gwtc5_local_rates_match_lvk_2026():
    """LVK_GWTC5_LOCAL_RATES pin LVK 2026 (GWTC-5.0) 90 percent CRs.

    LVK 2026, "GWTC-5.0: Population Properties of Merging Compact
    Binaries", arXiv:2605.27226.  The paper reports the intrinsic
    merger rate density as the "Joint" row of Table 2, the union of the
    PixelPop (weakly-modelled) and FullPop (strongly-modelled) 90
    percent credible intervals, marginalising over modelling-systematic
    uncertainty between the two population approaches:

        BNS  : 5.1  - 154.7 Gpc^-3 yr^-1  (z = 0)
        NSBH : 6.7  -  32.8 Gpc^-3 yr^-1  (z = 0)
        BBH  : 27.5 -  49.4 Gpc^-3 yr^-1  (z = 0.2)

    These supersede the GWTC-4 values (Abac+ 2025, arXiv:2508.18083;
    BNS 7.6-250, NSBH 9.1-84, BBH 14-26); GWTC-5.0 added no new BNS or
    NSBH candidates, so the NS-side bands tightened.  This anchor fails
    loudly if the constants drift from the GWTC-5.0 paper numbers.
    """
    from grb_rates import LVK_GWTC5_LOCAL_RATES

    expected = {
        "BNS": (5.1, 154.7),
        "NSBH": (6.7, 32.8),
        "BBH": (27.5, 49.4),
    }
    for pop, (lo, hi) in expected.items():
        assert pop in LVK_GWTC5_LOCAL_RATES, f"LVK_GWTC5_LOCAL_RATES missing {pop!r}"
        assert LVK_GWTC5_LOCAL_RATES[pop]["R_lo"] == pytest.approx(lo, rel=1e-9), (
            f"LVK_GWTC5_LOCAL_RATES[{pop!r}]['R_lo'] = "
            f"{LVK_GWTC5_LOCAL_RATES[pop]['R_lo']}, expected {lo} (LVK 2026)."
        )
        assert LVK_GWTC5_LOCAL_RATES[pop]["R_hi"] == pytest.approx(hi, rel=1e-9), (
            f"LVK_GWTC5_LOCAL_RATES[{pop!r}]['R_hi'] = "
            f"{LVK_GWTC5_LOCAL_RATES[pop]['R_hi']}, expected {hi} (LVK 2026)."
        )
        assert "arXiv:2605.27226" in LVK_GWTC5_LOCAL_RATES[pop]["reference"], (
            f"LVK_GWTC5_LOCAL_RATES[{pop!r}] missing arXiv:2605.27226 reference; "
            f"got {LVK_GWTC5_LOCAL_RATES[pop]['reference']!r}."
        )


def test_lvk_gwtc4_local_rates_match_lvk_2025():
    """LVK_GWTC4_LOCAL_RATES pin the GWTC-4.0 90 percent CRs.

    Abac+ 2025, "GWTC-4.0: Population properties of merging compact
    binaries", arXiv:2508.18083.  Kept as the conservative fallback band
    behind the tightened GWTC-5.0 numbers: BNS 7.6-250, NSBH 9.1-84
    (Gpc^-3 yr^-1, z = 0).
    """
    from grb_rates import LVK_GWTC4_LOCAL_RATES

    expected = {"BNS": (7.6, 250.0), "NSBH": (9.1, 84.0)}
    for pop, (lo, hi) in expected.items():
        assert pop in LVK_GWTC4_LOCAL_RATES, f"LVK_GWTC4_LOCAL_RATES missing {pop!r}"
        assert LVK_GWTC4_LOCAL_RATES[pop]["R_lo"] == pytest.approx(lo, rel=1e-9), (
            f"LVK_GWTC4_LOCAL_RATES[{pop!r}]['R_lo'] = "
            f"{LVK_GWTC4_LOCAL_RATES[pop]['R_lo']}, expected {lo} (GWTC-4.0)."
        )
        assert LVK_GWTC4_LOCAL_RATES[pop]["R_hi"] == pytest.approx(hi, rel=1e-9), (
            f"LVK_GWTC4_LOCAL_RATES[{pop!r}]['R_hi'] = "
            f"{LVK_GWTC4_LOCAL_RATES[pop]['R_hi']}, expected {hi} (GWTC-4.0)."
        )
        assert "arXiv:2508.18083" in LVK_GWTC4_LOCAL_RATES[pop]["reference"], (
            f"LVK_GWTC4_LOCAL_RATES[{pop!r}] missing arXiv:2508.18083 reference; "
            f"got {LVK_GWTC4_LOCAL_RATES[pop]['reference']!r}."
        )


def test_lvk_gwtc5_per_model_rates_match_lvk_2026_table2():
    """LVK_GWTC5_PER_MODEL_RATES pins Table 2 rows 1-2 (PixelPop / FullPop).

    LVK 2026, GWTC-5.0 (arXiv:2605.27226), Table 2 (Gpc^-3 yr^-1):

        PixelPop  BNS  23.4 +54.7 / -18.2   (z = 0)
                  NSBH 15.9 +16.9 / -8.9    (z = 0)
                  BBH  37.5 +11.9 / -9.1    (z = 0.2)
        FullPop   BNS  59.3 +95.4 / -43.9   (z = 0)
                  NSBH 14.2 +12.0 / -7.5    (z = 0)
                  BBH  36.0 +11.1 / -8.5    (z = 0.2)

    The "Joint" union row (encoded in ``LVK_GWTC5_LOCAL_RATES``) is the
    union of the PixelPop and FullPop 90 percent CRs; the per-model rows
    pinned here are the inputs to that union.
    """
    from grb_rates import LVK_GWTC5_PER_MODEL_RATES

    expected = {
        "BNS": {
            "PixelPop": (23.4, 18.2, 54.7),
            "FullPop": (59.3, 43.9, 95.4),
        },
        "NSBH": {
            "PixelPop": (15.9, 8.9, 16.9),
            "FullPop": (14.2, 7.5, 12.0),
        },
        "BBH": {
            "PixelPop": (37.5, 9.1, 11.9),
            "FullPop": (36.0, 8.5, 11.1),
        },
    }
    for pop, per_model in expected.items():
        assert pop in LVK_GWTC5_PER_MODEL_RATES, (
            f"LVK_GWTC5_PER_MODEL_RATES missing population {pop!r}"
        )
        for model, (med, lo_err, hi_err) in per_model.items():
            entry = LVK_GWTC5_PER_MODEL_RATES[pop][model]
            assert entry["R_med"] == pytest.approx(med, rel=1e-9), (
                f"LVK_GWTC5_PER_MODEL_RATES[{pop!r}][{model!r}]['R_med'] = "
                f"{entry['R_med']}, expected {med} (LVK 2026 Table 2)."
            )
            assert entry["R_lo_err"] == pytest.approx(lo_err, rel=1e-9), (
                f"LVK_GWTC5_PER_MODEL_RATES[{pop!r}][{model!r}]['R_lo_err'] = "
                f"{entry['R_lo_err']}, expected {lo_err} (LVK 2026 Table 2)."
            )
            assert entry["R_hi_err"] == pytest.approx(hi_err, rel=1e-9), (
                f"LVK_GWTC5_PER_MODEL_RATES[{pop!r}][{model!r}]['R_hi_err'] = "
                f"{entry['R_hi_err']}, expected {hi_err} (LVK 2026 Table 2)."
            )
    assert "arXiv:2605.27226" in LVK_GWTC5_PER_MODEL_RATES["reference"], (
        f"LVK_GWTC5_PER_MODEL_RATES missing arXiv:2605.27226 reference; "
        f"got {LVK_GWTC5_PER_MODEL_RATES['reference']!r}."
    )


def test_lvk_gwtc5_nsbh_min_bh_mass_matches_lvk_2026():
    """LVK_GWTC5_NSBH_MIN_BH_MASS pins the value inherited from Abac+ 2024.

    GWTC-5.0 (LVK 2026, arXiv:2605.27226) added no new NS-bearing
    candidates and defers its NS-population analysis to GWTC-4 and to
    the GW230529 follow-up (Abac+ 2024, arXiv:2404.04248), which infers
    a minimum BH mass in NSBH systems of 3.4 +1.0 / -1.2 Msun.
    """
    from grb_rates import LVK_GWTC5_NSBH_MIN_BH_MASS

    assert LVK_GWTC5_NSBH_MIN_BH_MASS["M_BH_min"] == pytest.approx(3.4, rel=1e-9), (
        f"LVK_GWTC5_NSBH_MIN_BH_MASS['M_BH_min'] = "
        f"{LVK_GWTC5_NSBH_MIN_BH_MASS['M_BH_min']}, expected 3.4 (Abac+ 2024)."
    )
    assert LVK_GWTC5_NSBH_MIN_BH_MASS["M_BH_min_lo_err"] == pytest.approx(1.2, rel=1e-9), (
        f"LVK_GWTC5_NSBH_MIN_BH_MASS['M_BH_min_lo_err'] = "
        f"{LVK_GWTC5_NSBH_MIN_BH_MASS['M_BH_min_lo_err']}, expected 1.2."
    )
    assert LVK_GWTC5_NSBH_MIN_BH_MASS["M_BH_min_hi_err"] == pytest.approx(1.0, rel=1e-9), (
        f"LVK_GWTC5_NSBH_MIN_BH_MASS['M_BH_min_hi_err'] = "
        f"{LVK_GWTC5_NSBH_MIN_BH_MASS['M_BH_min_hi_err']}, expected 1.0."
    )
    assert "arXiv:2605.27226" in LVK_GWTC5_NSBH_MIN_BH_MASS["reference"], (
        f"LVK_GWTC5_NSBH_MIN_BH_MASS missing arXiv:2605.27226 reference; "
        f"got {LVK_GWTC5_NSBH_MIN_BH_MASS['reference']!r}."
    )


def test_logz_max_physical_anchor():
    """``LOGZ_MAX_PHYSICAL = 0`` (Z <= Zsun) for the Section 5 binning range.

    The COMPAS STROOPWAFEL prior covers Z in [1e-4, 0.03] ~ [0.007, 2.1]
    Zsun (Broekgaarden et al. 2021, arXiv:2103.02608 Sec. 3.2), and solar
    is where the yield curves are binned to, not where the star formation
    or the simulated grid ends.  Two facts fix what the constant may and
    may not be justified by:

    - Under the Levina et al. (2026, arXiv:2601.20202) Table 1 TNG100-1
      fit (Azzalini skew-log-normal, alpha = -1.8801, mu0 = 0.0247),
      64.8 percent of z = 0 cosmic star formation sits above Zsun
      (39.8 percent SFR-weighted over 0 < z < 10), and 28.5 percent sits
      above the Z = 0.03 grid ceiling.  The truncated range is the
      majority of local star formation, so this is a display convention
      and must never be written up as a relevance or noise cut.
    - Broekgaarden et al. (2021) Sec. 3.3 report the BHNS yield peaking
      at Z_i ~ Zsun/5 (Z_i ~ 0.003) and being lowest at Z_i >~ Zsun;
      Broekgaarden et al. (2022) Sec. 3.1 report the NSNS yield staying
      roughly constant in Z.  Model A here reproduces both: the BHNS
      native-grid peak lands at Z = 0.00303 and falls to 28 percent of
      peak at Zsun, while BNS is still at 76 percent of peak there.

    A future change that extends the axis above solar must update both
    the constant and this anchor.
    """
    from grb_rates import LOGZ_MAX_PHYSICAL

    assert LOGZ_MAX_PHYSICAL == pytest.approx(0.0, abs=1e-12), (
        f"LOGZ_MAX_PHYSICAL = {LOGZ_MAX_PHYSICAL} drifted from the Zsun "
        f"binning range; see Section 5 caption in grb_main.ipynb."
    )


def test_broekgaarden_2021_formation_efficiency_band():
    """Intensive eta(Z) must land on the published formation-yield axis.

    The ordinate is Broekgaarden et al. (2021, arXiv:2103.02608) Fig. 5,
    "number of BHNS mergers with t_delay <= t_H that form per unit star
    forming mass", and its BNS counterpart is the NSNS panel of
    Broekgaarden et al. (2022, arXiv:2112.05763) Fig. 1.  Both run at
    order 10^-6 to 10^-5 mergers per Msun across the sub-solar grid, with
    an integrated (mass-weighted) value of a few x 10^-6 / Msun.
    ``formation_efficiency`` reproduces that band only when it divides
    each bin by the per-metallicity star-forming mass
    ``mean_mass_evolved * f_i`` (equal mass at every simulated grid
    point).  Dividing by the bare scalar total instead would suppress
    every bin by 1/f_i = 53, landing nearly two orders of magnitude too
    low.

    Anchored on representative Model A numbers (total STROOPWAFEL DCO
    weight ~ 1.0e4, published-rate-calibrated mean_mass_evolved ~ 2.8e9
    Msun, Section 4) so the anchors job needs no data/.  The integrated
    value is an exact identity (sum(w) / mean_mass_evolved); the per-Z
    band is the physically meaningful target.
    """
    from grb_rates import formation_efficiency, metallicity_prior_mass_fraction

    grid = np.geomspace(1e-4, 0.03, 53)
    rng = np.random.default_rng(0)
    Z_all = rng.choice(grid, size=20000)
    w_all = rng.uniform(0.3, 0.7, size=Z_all.size)
    w_all *= 1.0e4 / w_all.sum()  # total STROOPWAFEL weight ~ Model A
    mean_mass_evolved = 2.804e9  # BNS, published-rate-calibrated (Section 4)

    eff = formation_efficiency(grid, Z_all, w_all, mean_mass_evolved=mean_mass_evolved)
    f = metallicity_prior_mass_fraction(grid)
    total = eff["total"]

    # Integrated (mass-fraction-weighted) efficiency is sum(w) / mean_mass.
    integrated = float(np.sum(total * f))
    assert integrated == pytest.approx(w_all.sum() / mean_mass_evolved, rel=1e-9)
    assert 1e-6 < integrated < 1e-5, (
        f"integrated BNS formation efficiency {integrated:.2e} /Msun outside "
        f"the published few x 10^-6 band."
    )

    sub = (grid <= 0.0142) & (total > 0)
    assert np.all((total[sub] > 1e-7) & (total[sub] < 5e-5)), (
        f"per-Z eta out of the published band [1e-7, 5e-5]: "
        f"min {total[sub].min():.2e}, max {total[sub].max():.2e} /Msun."
    )


# ─────────────────────────────────────────────────────────────────────
# Cosmology pin: Planck Collaboration 2016, A&A 594, A13.
# ─────────────────────────────────────────────────────────────────────
def test_planck15_cosmology_constants_match_compas_pin():
    """Planck 2015 H0 / Omega_m / Omega_Lambda must match COMPAS pin.

    Project cosmology (Ade et al. 2016, A&A 594, A13, TT+lowP+lensing+ext;
    matches COMPAS FastCosmicIntegration TNG-consistent default):

        H0 = 67.74 km/s/Mpc, Omega_m = 0.3089, Omega_Lambda = 0.6911.

    Mixing with Planck 2018 introduces ~2 percent inconsistencies at
    high z; that drift is enough to shift class fractions
    substantively.

    Note: ``astropy.cosmology.Planck15.Om0 = 0.3075`` is from the
    Planck 2015 baseline TT+lowP column (Ade et al. 2016, Table 4);
    `grb_physics.py` quotes Om0 = 0.3089 from the TT+lowP+lensing+ext
    column.  The project's stated value sits 1.4 sigma from the astropy
    default; the strict_paper test surfaces that discrepancy with a
    0.005 absolute tolerance band that admits either Planck 2015 column.
    A future change should reconcile by either updating the docstrings
    to 0.3075 or constructing an explicit
    `FlatLambdaCDM(H0=67.74, Om0=0.3089)` cosmology.
    """
    from astropy.cosmology import Planck15

    assert abs(Planck15.H0.value - 67.74) < 0.01, (
        f"astropy.Planck15.H0 = {Planck15.H0.value} drifted from "
        f"the project cosmology pin 67.74 km/s/Mpc (Ade et al. 2016)."
    )
    assert abs(Planck15.Om0 - 0.3089) <= 0.005, (
        f"astropy.Planck15.Om0 = {Planck15.Om0} more than 0.005 from "
        f"project cosmology pin 0.3089 (Ade et al. 2016, TT+lowP+lensing+ext)."
    )
    assert abs(Planck15.Ode0 - 0.6911) <= 0.005, (
        f"astropy.Planck15.Ode0 = {Planck15.Ode0} more than 0.005 from "
        f"project cosmology pin 0.6911 (Ade et al. 2016)."
    )


# ─────────────────────────────────────────────────────────────────────
# Bardeen, Press and Teukolsky (1972) ApJ 178, 347 [papers/Bardeen_1972.pdf]
# ISCO closed-form anchor values (extends test_isco.py).
# ─────────────────────────────────────────────────────────────────────
def test_bardeen_1972_isco_textbook_anchors():
    """`r_isco(0) = 6`, `r_isco(+1) ~ 1`, `r_isco(-1) ~ 9` per Bardeen Eq. 2.21.

    Three textbook anchor values for the ISCO in units of GM_BH/c^2:
    Schwarzschild (chi = 0) at r = 6, prograde extremal Kerr at r = 1,
    retrograde extremal at r = 9.  Anchored against Bardeen, Press
    and Teukolsky (1972) ApJ 178, 347, Eq. 2.21.
    """
    from grb_physics import r_isco

    assert r_isco(0.0) == pytest.approx(6.0, rel=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # |a|=1 triggers clip warning
        assert r_isco(+1.0) == pytest.approx(1.0, abs=5e-3)
        assert r_isco(-1.0) == pytest.approx(9.0, abs=5e-3)


# ─────────────────────────────────────────────────────────────────────
# Gottlieb (2023, 2024) classification thresholds [papers/Gottlieb_2023.pdf,
# Gottlieb_2024.pdf]
# ─────────────────────────────────────────────────────────────────────
def test_gottlieb_2023_disk_mass_thresholds_match_paper():
    """BHNS disk-mass thresholds must equal Gottlieb (2023) Sec. 4 / Fig. 6."""
    from grb_physics import MDISK_LONG, MDISK_SHORT

    assert MDISK_SHORT == pytest.approx(0.01, rel=1e-9), MDISK_SHORT
    assert MDISK_LONG == pytest.approx(0.10, rel=1e-9), MDISK_LONG


def test_gottlieb_2023_bns_M_crit_and_q_thresh_match_paper():
    """BNS prompt-collapse `M_CRIT_BNS = 2.8` and mass-ratio `Q_THRESH_BNS = 1.2`."""
    from grb_physics import M_CRIT_BNS, Q_THRESH_BNS

    assert M_CRIT_BNS == pytest.approx(2.8, rel=1e-9), M_CRIT_BNS
    assert Q_THRESH_BNS == pytest.approx(1.2, rel=1e-9), Q_THRESH_BNS


def test_grid_class_labels_match_gottlieb_2024_hybrid_taxonomy():
    """`GRID_CLASS_LABELS` must encode the Gottlieb 2024 6-class hybrid.

    The label map drives the Section 1 mass-plane figure: integers
    1 to 4 are the BNS classes (Faint lbGRB, lbGRB + red KN (HMNS),
    sbGRB + blue KN, lbGRB + red KN (BNS disk)) and integers 5 to 6
    are the BHNS classes (Faint lbGRB (BHNS), lbGRB + red KN (BHNS
    disk)).  Integer 0 is reserved for background (BBH or
    out-of-range) and is intentionally absent from the map.

    A future refactor that reshuffles the integers would silently
    mistint every cell on the Section 1 colourbar; pinning the exact
    label-to-integer mapping here catches that drift.
    """
    from grb_classify import GRID_CLASS_LABELS

    expected = {
        1: "Faint lbGRB (BNS)",
        2: "lbGRB + red KN (HMNS)",
        3: "sbGRB + blue KN (HMNS)",
        4: "lbGRB + red KN (BNS disk)",
        5: "Faint lbGRB (BHNS)",
        6: "lbGRB + red KN (BHNS disk)",
    }
    assert GRID_CLASS_LABELS == expected, GRID_CLASS_LABELS


def test_kn_m_ej_faint_max_aligned_with_mdisk_short():
    """`KN_M_EJ_FAINT_MAX` must equal `MDISK_SHORT = 0.01 Msun`.

    Gottlieb (2023, arXiv:2309.00038) Sec. 4 / Fig. 6 fixes the
    engine-fuel floor at 0.01 Msun: below that disk mass the central
    engine has no fuel for a bright GRB and below that ejecta mass
    the kilonova is not detectable.  The two thresholds therefore
    sit at the same value across the BNS / BHNS branches and the
    observed-merger classifier.  A future drift between the two
    would mean the model and observed-sample "Faint" classes no
    longer share a definition.
    """
    from grb_classify import KN_M_EJ_FAINT_MAX
    from grb_physics import MDISK_SHORT

    assert KN_M_EJ_FAINT_MAX == MDISK_SHORT == 0.01, (
        f"KN_M_EJ_FAINT_MAX = {KN_M_EJ_FAINT_MAX}, MDISK_SHORT = "
        f"{MDISK_SHORT}; both must equal 0.01 Msun per Gottlieb (2023) "
        f"Sec. 4 / Fig. 6."
    )


def test_classify_bns_2024_default_hmns_factor_matches_HMNS_FACTOR_DEFAULT():
    """The default `hmns_factor` keyword of `classify_bns_2024` must
    equal `grb_physics.HMNS_FACTOR_DEFAULT`.

    The Margalit-Metzger (2017) supramassive-remnant band sets the
    1.2 fiducial via `HMNS_FACTOR_DEFAULT`, and the existing
    `test_margalit_metzger_2017_hmns_factor_in_supramassive_band`
    pins that constant to the published band.  This test pins the
    pipeline-side consumer so that a refactor changing the
    classifier default without updating the constant is caught.
    """
    import inspect

    from grb_classify import classify_bns_2024
    from grb_physics import HMNS_FACTOR_DEFAULT

    sig = inspect.signature(classify_bns_2024)
    default = sig.parameters["hmns_factor"].default
    assert default == HMNS_FACTOR_DEFAULT, (
        f"classify_bns_2024 default hmns_factor = {default} drifted "
        f"from HMNS_FACTOR_DEFAULT = {HMNS_FACTOR_DEFAULT}; the "
        f"classifier and the Margalit-Metzger anchor are now out of "
        f"sync."
    )


# ─────────────────────────────────────────────────────────────────────
# Gottlieb et al. (2025) arXiv:2411.13657 [papers/Gottlieb_2024.pdf]
# Eq. (11) BH-engine disk-wind kilonova ejecta normalisation.
# ─────────────────────────────────────────────────────────────────────
def test_gottlieb_2025_eq11_canonical_normalisation_matches_paper():
    """`gottlieb25_eq11` normalisation must reproduce Gottlieb+25 Eq. 11.

    Gottlieb et al. (2025) arXiv:2411.13657, Eq. (11):

        M_ej = 1e-3 * f_{-1}^{-1} * (E_iso / 2e51 erg) *
               (T50 / 1 s)^{alpha - 1}   [Msun]

    At the reference point ``(T50 = 1 s, E_iso = 2e51 erg,
    alpha = 2, f_inv = 1)`` the right-hand side reduces to the
    prefactor ``1e-3 Msun``.  Paper-pinned literal; do not edit
    without updating the citation.
    """
    from grb_physics import gottlieb25_eq11

    assert gottlieb25_eq11(T50=1.0, E_iso=2e51, alpha=2.0, f_inv=1.0) == pytest.approx(
        1e-3, rel=1e-12
    ), "Gottlieb+25 Eq. 11 normalisation drifted from 1e-3 Msun"


# ─────────────────────────────────────────────────────────────────────
# Broekgaarden et al. (2021) [papers/Broekgaarden_2021.pdf]
# Project-level NS_MAX_BNS pin.
# ─────────────────────────────────────────────────────────────────────
def test_metallicity_grid_endpoints_and_size_match_broekgaarden_2021_prior():
    """`METALLICITY_GRID` must encode the Broekgaarden+ 2021 prior schema.

    Broekgaarden et al. (2021) arXiv:2103.02608 Sec. 3.2 describes the
    COMPAS metallicity prior as approximately log-uniform on
    ``Z in [1e-4, 3e-2]`` discretised to 53 bins.  A future drift
    (a duplicated end value, an inserted bin, or a copy-edit that
    shifts an endpoint) would silently re-bin every downstream
    rate computation that uses this grid.  Pinning the size and
    both endpoints here turns that drift into an immediate failure.
    """
    from grb_io import METALLICITY_GRID

    assert len(METALLICITY_GRID) == 53, len(METALLICITY_GRID)
    assert METALLICITY_GRID[0] == pytest.approx(1e-4, rel=1e-12), METALLICITY_GRID[0]
    assert METALLICITY_GRID[-1] == pytest.approx(3e-2, rel=1e-12), METALLICITY_GRID[-1]


def test_metallicity_grid_is_approximately_log_uniform():
    """`METALLICITY_GRID` log-spacing variation stays under 1.0.

    Broekgaarden+ 2021 Sec. 3.2 describes the prior as
    "approximately log-uniform"; the actual COMPAS grid has a
    measurable but bounded variation in log spacing (the relative
    range ``(max - min) / mean`` evaluates to roughly 0.85 on the
    current 53-entry grid).  A relative range of 1.0 is a
    conservative regression sentinel: an accidental linear-spaced
    insertion would land an outlier spacing well above this bound,
    while the documented log-uniform curvature stays comfortably
    inside.
    """
    from grb_io import METALLICITY_GRID

    log_z = np.log10(METALLICITY_GRID)
    dlog = np.diff(log_z)
    rel_range = (dlog.max() - dlog.min()) / dlog.mean()
    assert rel_range < 1.0, (
        f"METALLICITY_GRID log-spacing relative range = {rel_range:.3f} "
        f"exceeds 1.0; the grid is no longer approximately log-uniform "
        f"per Broekgaarden+ 2021 Sec. 3.2."
    )
    # Bonus: all spacings should be strictly positive (already
    # implied by the strict-monotonicity unit test).
    assert (dlog > 0).all()


def test_broekgaarden_2021_NS_MAX_FIDUCIAL_models_J_A_K():
    """Models J / A / K NS_MAX values must equal 2.0 / 2.5 / 3.0 Msun."""
    from grb_classify import NS_MAX_FIDUCIAL

    assert NS_MAX_FIDUCIAL == (2.0, 2.5, 3.0), NS_MAX_FIDUCIAL


def test_alpha_CE_per_model_matches_broekgaarden21_sec_5_2():
    """alpha_CE per model matches Broekgaarden+ 2021 Sec. 5.2 Table 2.

    Pins the CE_PRESCRIPTION_BROEKGAARDEN21 dict so a careless edit to
    the constants block fails this test rather than silently corrupting
    Sections 12.1, 12.6, 12.7, 12.8.  Also pins the prose names of the
    energy formalism, lambda_CE recipe, and stability criterion so a
    future move to the Hirai+ updated prescription would require an
    explicit edit to the test, not just to the data.
    """
    from grb_rates import CE_PRESCRIPTION_BROEKGAARDEN21

    assert CE_PRESCRIPTION_BROEKGAARDEN21["alpha_CE"] == {
        "A": 1.0,
        "F": 0.5,
        "G": 2.0,
        "J": 1.0,
        "K": 1.0,
    }
    assert CE_PRESCRIPTION_BROEKGAARDEN21["energy_formalism"] == "Webbink 1984"
    assert CE_PRESCRIPTION_BROEKGAARDEN21["lambda_CE"] == "Xu and Li 2010"
    assert "Hurley 2002" in CE_PRESCRIPTION_BROEKGAARDEN21["stability"]


def test_CE_PRESCRIPTION_BROEKGAARDEN21_full_string_schema():
    """`CE_PRESCRIPTION_BROEKGAARDEN21` must carry the full literature strings.

    Complements `test_alpha_CE_per_model_matches_broekgaarden21_sec_5_2`
    (which uses substring matches) by pinning the exact stability and
    double-CE strings as well.  The references are:

      energy_formalism : Webbink (1984) ApJ 277, 355 (alpha-lambda
                         energy formalism)
      lambda_CE        : Xu and Li (2010) ApJ 716, 114
                         (envelope binding energy fits)
      stability        : Hurley et al. (2002) MNRAS 329, 897 (zeta_ad)
                         with Vinciguerra et al. (2020) MNRAS 498, 4705
                         COMPAS modifications
      double_CE_flag   : "both stars in core-He burning at CE onset"
                         (Broekgaarden et al. 2021 Sec. 3.3)

    A future swap to the Hirai+ updated stability prescription would
    trip this exact-string check before the regression manifests in
    the formation-channel rates pinned in
    `tests/sections/test_section_06b_formation_channels.py`.
    """
    from grb_rates import CE_PRESCRIPTION_BROEKGAARDEN21 as CE

    assert CE["energy_formalism"] == "Webbink 1984"
    assert CE["lambda_CE"] == "Xu and Li 2010"
    assert CE["stability"] == "Hurley 2002 (Vinciguerra+ 2020 modified)"
    assert CE["double_CE_flag"] == "both stars in core-He burning at CE onset"
    # The five-model alpha_CE schema is independently pinned in the
    # neighbouring test; here we just lock the dict-shape contract.
    assert set(CE["alpha_CE"].keys()) == {"A", "F", "G", "J", "K"}


# ─────────────────────────────────────────────────────────────────────
# Section 1 mass-plane anchors.
#
# These pin the literature numbers that the two Section 1 figures of
# grb_main.ipynb display directly: the observed BNS GW event coordinates
# (Abbott+ 2019, 2020), the COMPAS Model A NS-mass cap shown in the
# panel extent, and the three boundary-line constants drawn in the
# legend (M_thresh = 2.79 Msun, 1.2 M_TOV = 2.64 Msun, q = 1.2).
# A drift in any of these is a drift between the legend and the data,
# which is the visible form of "honest plot" the audit guards against.
# ─────────────────────────────────────────────────────────────────────
def test_observed_gw_events_match_abbott_2019_2020():
    """`OBSERVED_GW_EVENTS` coordinates must sit inside the Abbott+ 2019 / 2020 low-spin 90% CR.

    Abbott et al. (2019) PRX 9, 011001 Table I (low-spin prior) reports
    GW170817 component-mass 90% CR ``M1 in [1.36, 1.60]``,
    ``M2 in [1.17, 1.36] Msun``; the representative point pinned here
    (``M1 = 1.46, M2 = 1.27``) is consistent with the GWTC-1 catalogue
    (Abbott et al. 2019 PRX 9, 031040) and lies inside those ranges.
    Abbott et al. (2020) ApJL 892, L3 Table 1 (low-spin prior) reports
    GW190425 component-mass 90% CR ``M1 in [1.60, 1.87]``,
    ``M2 in [1.46, 1.69] Msun``; the pinned point ``M1 = 1.73,
    M2 = 1.58 Msun`` is the CR midpoint (not a directly tabulated
    median).  These are the coordinates Section 1 of grb_main.ipynb
    annotates as star markers on the BNS panel; a drift means the
    labelled stars no longer sit inside the published low-spin CRs.
    """
    from grb_io import OBSERVED_GW_EVENTS

    assert set(OBSERVED_GW_EVENTS) >= {"GW170817", "GW190425"}, OBSERVED_GW_EVENTS

    gw170817 = OBSERVED_GW_EVENTS["GW170817"]
    assert gw170817["M1"] == pytest.approx(1.46, abs=0.02), gw170817
    assert gw170817["M2"] == pytest.approx(1.27, abs=0.02), gw170817
    assert "Abbott" in gw170817["reference"], gw170817["reference"]
    assert "PRX 9, 011001" in gw170817["reference"], gw170817["reference"]
    # m1 >= m2 invariant (Section 1 plots M2 on x-axis, M1 on y-axis).
    assert gw170817["M1"] >= gw170817["M2"], gw170817

    gw190425 = OBSERVED_GW_EVENTS["GW190425"]
    assert gw190425["M1"] == pytest.approx(1.73, abs=0.02), gw190425
    assert gw190425["M2"] == pytest.approx(1.58, abs=0.02), gw190425
    assert "Abbott" in gw190425["reference"], gw190425["reference"]
    assert "ApJL 892, L3" in gw190425["reference"], gw190425["reference"]
    assert gw190425["M1"] >= gw190425["M2"], gw190425


def test_ns_max_modelA_matches_broekgaarden_2021():
    """COMPAS Model A NS-mass cap = 2.5 Msun (Broekgaarden+ 2021, Sec. 3.4).

    Section 1 of ``grb_main.ipynb`` truncates the BNS panel at the
    fiducial ``NS_MAX_BNS = 2.5 Msun`` and passes the same value to
    ``classify_grid(ns_max=2.5)`` for the BNS / BHNS region split.
    The literal must (a) equal 2.5, the published Model A value from
    Broekgaarden+ 2021, arXiv:2103.02608, Sec. 3.4 / Table 2; and (b)
    sit in ``NS_MAX_FIDUCIAL`` so ``classify_grid`` accepts it without
    requiring ``strict_ns_max=False``.
    """
    from grb_classify import NS_MAX_FIDUCIAL

    # The notebook constant is duplicated as a literal (no module export
    # to avoid coupling ``grb_main.ipynb`` constants into a python
    # module that downstream sections do not need).  Pin the value here
    # so a future drift in the notebook breaks this test.
    NS_MAX_BNS_modelA = 2.5

    assert NS_MAX_BNS_modelA in NS_MAX_FIDUCIAL, (
        f"NS_MAX_BNS = {NS_MAX_BNS_modelA} not in NS_MAX_FIDUCIAL = "
        f"{NS_MAX_FIDUCIAL}; classify_grid would reject it under "
        f"strict_ns_max=True."
    )
    # NS_MAX_FIDUCIAL is sorted (J, A, K) -> (2.0, 2.5, 3.0); position 1
    # is Model A.
    assert NS_MAX_FIDUCIAL[1] == NS_MAX_BNS_modelA, NS_MAX_FIDUCIAL


def test_bns_boundary_lines_match_legend_values():
    """`bns_boundary_lines` must trace the three Gottlieb (2024) constants the legend displays.

    Section 1 of ``grb_main.ipynb`` draws the three Gottlieb (2024)
    BNS boundary curves with legend labels ``M_tot = 2.79 Msun``,
    ``1.2 * M_TOV = 2.64 Msun``, and ``q = 1.2``.  This test reads
    ``bns_boundary_lines`` at fiducial inputs and verifies that the
    returned (M2, M1) curves actually trace those algebraic constraints
    to machine precision -- a regression that, say, swapped
    ``hmns_factor`` for a different value would silently move the cyan
    dash-dot line but leave the legend untouched.
    """
    from grb_classify import bns_boundary_lines
    from grb_physics import M_THRESH, M_TOV, Q_THRESH_BNS

    # The three numbers the figure's legend prints.
    assert M_THRESH == pytest.approx(2.794, abs=1e-3), M_THRESH
    assert 1.2 * M_TOV == pytest.approx(2.64, abs=1e-9), M_TOV
    assert Q_THRESH_BNS == pytest.approx(1.2, abs=1e-9), Q_THRESH_BNS

    m2_axis = np.linspace(1.25, 2.2, 500)
    bdy = bns_boundary_lines(m2_axis)

    m2_mt, m1_mt = bdy["M_tot"]
    assert m1_mt.size > 0, "M_tot boundary returned empty curve"
    assert np.allclose(m2_mt + m1_mt, M_THRESH, atol=1e-12), (
        f"M_tot boundary does not trace m1 + m2 = M_THRESH = {M_THRESH} to machine precision."
    )

    m2_hmns, m1_hmns = bdy["HMNS"]
    assert m1_hmns.size > 0, "HMNS boundary returned empty curve"
    assert np.allclose(m2_hmns + m1_hmns, 1.2 * M_TOV, atol=1e-12), (
        f"HMNS boundary does not trace m1 + m2 = 1.2 * M_TOV = {1.2 * M_TOV} to machine precision."
    )

    m2_q, m1_q = bdy["q"]
    assert m1_q.size > 0, "q boundary returned empty curve"
    assert np.allclose(m1_q, Q_THRESH_BNS * m2_q, atol=1e-12), (
        f"q boundary does not trace m1 = {Q_THRESH_BNS} * m2 to machine precision."
    )
    # m1 >= m2 clip must hold (Gottlieb 2024 convention).
    assert (m1_mt >= m2_mt).all(), "M_tot boundary leaked into m1 < m2 region"
    assert (m1_hmns >= m2_hmns).all(), "HMNS boundary leaked into m1 < m2 region"
    assert (m1_q >= m2_q).all(), "q boundary leaked into m1 < m2 region"


# ─────────────────────────────────────────────────────────────────────
# Deprecated attribute regression sentinels (no published anchor)
# ─────────────────────────────────────────────────────────────────────
def test_grb_physics_mean_mass_evolved_deprecated_attribute():
    """`grb_physics.MEAN_MASS_EVOLVED` must emit a DeprecationWarning
    and return the cached legacy sentinel `_MEAN_MASS_EVOLVED_VALUE`.

    CODE HEURISTIC, not a published number.  The live science path
    re-derives the calibration constant on demand via
    `grb_rates.calibrate_mean_mass_evolved`; the deprecated module
    attribute is retained only for back-compat and emits a warning
    on access.  Pinning the integer value here catches a refactor
    that drops the legacy constant or changes its magnitude
    silently; pinning the warning catches a refactor that quietly
    re-exposes the attribute as live.
    """
    import warnings as _warnings

    import grb_physics as gp
    from grb_physics import _MEAN_MASS_EVOLVED_VALUE

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        value = gp.MEAN_MASS_EVOLVED
    deprecation = [item for item in w if issubclass(item.category, DeprecationWarning)]
    assert deprecation, "grb_physics.MEAN_MASS_EVOLVED did not emit DeprecationWarning"
    assert value == _MEAN_MASS_EVOLVED_VALUE
    assert _MEAN_MASS_EVOLVED_VALUE == 77708655, (
        f"_MEAN_MASS_EVOLVED_VALUE = {_MEAN_MASS_EVOLVED_VALUE} drifted "
        f"from the legacy regression sentinel 77708655."
    )


# ─────────────────────────────────────────────────────────────────────
# External long-merger-GRB E_iso anchor (engine discriminator, comparison.ipynb)
# ─────────────────────────────────────────────────────────────────────
def test_external_lbgrb_eiso_anchor_matches_literature():
    """Pin the external lbGRB E_iso anchor to the cited E_iso values.

    The BH-engine null in comparison.ipynb is anchored on five long-duration
    merger GRBs with kilonovae, with E_iso [erg] from the primary analyses:
    060614 (Mangano+ 2007), 211227A (Lu+ 2022), 191019A (Stratta+ 2025),
    211211A (Troja+ 2022), 230307A (Svinkin+ 2023 / Dichiara+ 2025).  Each
    value must stay inside the long-GRB E_iso band (~1e50 to 1e53 erg), and the
    log-mean / log-std are derived (not free) so they cannot drift from the set.
    """
    import numpy as np

    import grb_physics as gp

    expected = (8.4e50, 1.14e51, 1.70e51, 1.30e52, 6.97e52)
    assert gp.LBGRB_EISO_EXTERNAL_ERG == expected, (
        f"LBGRB_EISO_EXTERNAL_ERG = {gp.LBGRB_EISO_EXTERNAL_ERG} drifted from "
        f"the cited literature values {expected}."
    )
    eiso = np.array(gp.LBGRB_EISO_EXTERNAL_ERG)
    assert np.all((eiso > 1e50) & (eiso < 1e53)), (
        "external lbGRB E_iso values fall outside the long-GRB band [1e50, 1e53] erg"
    )
    assert np.isclose(gp.LBGRB_EISO_EXTERNAL_LOGMEAN, float(np.log10(eiso).mean()))
    assert np.isclose(gp.LBGRB_EISO_EXTERNAL_LOGSTD, float(np.log10(eiso).std()))
    assert 51.0 < gp.LBGRB_EISO_EXTERNAL_LOGMEAN < 52.5


# ─────────────────────────────────────────────────────────────────────
# Broekgaarden et al. (2021) 20-model variation grid
# arXiv:2103.02608 (Table 2; physics knobs Sec. 3-5) + arXiv:2112.05763
# ─────────────────────────────────────────────────────────────────────
def test_broekgaarden21_grid_has_twenty_variations():
    """The model registry carries all 20 Broekgaarden+ 2021 variations.

    Broekgaarden et al. (2021), arXiv:2103.02608, vary one binary-physics
    assumption at a time around the fiducial Model A to produce a 20-model
    grid.  The project ships all 20 (paper-I letters A-O plus the five
    paper-II-only tokens EH, alpha0p1, alpha10, fWR0p1, fWR5).
    """
    from grb_io import BROEKGAARDEN21_MODELS

    assert len(BROEKGAARDEN21_MODELS) == 20, (
        f"registry has {len(BROEKGAARDEN21_MODELS)} models, expected the "
        f"20-variation Broekgaarden+ 2021 grid."
    )
    expected = {
        "A", "B", "C", "D", "E", "EH", "alpha0p1", "F", "G", "alpha10",
        "H", "I", "J", "K", "L", "M", "N", "O", "fWR0p1", "fWR5",
    }  # fmt: skip
    assert set(BROEKGAARDEN21_MODELS) == expected


def test_broekgaarden21_alpha_CE_grid_values():
    """alpha_CE spans {0.1, 0.5, 1.0, 2.0, 10.0} across the grid.

    Broekgaarden et al. (2021) Sec. 5.2 vary the common-envelope efficiency
    over half a dozen values bracketing the fiducial alpha_CE = 1.0:
    0.1 (alpha0p1), 0.5 (F), 2.0 (G), 10 (alpha10).  Every other variation
    holds the fiducial 1.0.
    """
    from grb_io import BROEKGAARDEN21_MODELS

    alphas = {m["alpha_ce"] for m in BROEKGAARDEN21_MODELS.values()}
    assert alphas == {0.1, 0.5, 1.0, 2.0, 10.0}
    assert BROEKGAARDEN21_MODELS["A"]["alpha_ce"] == 1.0
    assert BROEKGAARDEN21_MODELS["alpha0p1"]["alpha_ce"] == 0.1
    assert BROEKGAARDEN21_MODELS["F"]["alpha_ce"] == 0.5
    assert BROEKGAARDEN21_MODELS["G"]["alpha_ce"] == 2.0
    assert BROEKGAARDEN21_MODELS["alpha10"]["alpha_ce"] == 10.0


def test_broekgaarden21_mass_transfer_efficiency_grid_values():
    """Fixed mass-transfer efficiency beta in {0.25, 0.5, 0.75} for B/C/D.

    Broekgaarden et al. (2021) Sec. 5.1 replace the fiducial adaptive
    (response-based) mass-transfer efficiency with three fixed values.
    The fiducial run leaves ``beta`` unset (``None``).
    """
    from grb_io import BROEKGAARDEN21_MODELS

    betas = {m["beta"] for m in BROEKGAARDEN21_MODELS.values() if m["beta"] is not None}
    assert betas == {0.25, 0.5, 0.75}
    assert BROEKGAARDEN21_MODELS["B"]["beta"] == 0.25
    assert BROEKGAARDEN21_MODELS["C"]["beta"] == 0.5
    assert BROEKGAARDEN21_MODELS["D"]["beta"] == 0.75
    assert BROEKGAARDEN21_MODELS["A"]["beta"] is None


def test_broekgaarden21_ns_max_grid_values():
    """Maximum NS mass spans {2.0, 2.5, 3.0} Msun across the grid.

    Broekgaarden et al. (2021) Sec. 3.4: fiducial M_NS,max = 2.5 Msun,
    lowered to 2.0 (Model J) and raised to 3.0 (Model K).
    """
    from grb_io import BROEKGAARDEN21_MODELS

    ns_maxes = {m["ns_max"] for m in BROEKGAARDEN21_MODELS.values()}
    assert ns_maxes == {2.0, 2.5, 3.0}
    assert BROEKGAARDEN21_MODELS["J"]["ns_max"] == 2.0
    assert BROEKGAARDEN21_MODELS["K"]["ns_max"] == 3.0
    assert BROEKGAARDEN21_MODELS["A"]["ns_max"] == 2.5


def test_broekgaarden21_sn_kick_and_engine_grid_values():
    """Core-collapse kick sigma in {30, 100} km/s; rapid engine only for I.

    Broekgaarden et al. (2021): the ccSN Maxwellian kick dispersion is
    lowered from the fiducial Hobbs+ 2005 value to 100 km/s (Model M) and
    30 km/s (Model N); the rapid Fryer+ 2012 remnant prescription replaces
    the fiducial delayed engine in exactly one variation (Model I).
    """
    from grb_io import BROEKGAARDEN21_MODELS

    sigmas = {
        m["sigma_kick_kms"]
        for m in BROEKGAARDEN21_MODELS.values()
        if m["sigma_kick_kms"] is not None
    }
    assert sigmas == {30.0, 100.0}
    rapid = {s for s, m in BROEKGAARDEN21_MODELS.items() if m["engine"] == "rapid"}
    assert rapid == {"I"}
    assert BROEKGAARDEN21_MODELS["A"]["engine"] == "delayed"


def test_broekgaarden21_wolf_rayet_wind_grid_values():
    """Wolf-Rayet wind multiplier f_WR in {0.1, 5.0} for the WR variations.

    Broekgaarden et al. (2021) Sec. 5.3 scale the Wolf-Rayet mass-loss rate
    by 0.1 (fWR0p1) and 5 (fWR5) around the fiducial multiplier of 1.
    """
    from grb_io import BROEKGAARDEN21_MODELS

    fwr = {m["f_WR"] for m in BROEKGAARDEN21_MODELS.values() if m["f_WR"] is not None}
    assert fwr == {0.1, 5.0}
    assert BROEKGAARDEN21_MODELS["fWR0p1"]["f_WR"] == 0.1
    assert BROEKGAARDEN21_MODELS["fWR5"]["f_WR"] == 5.0


def test_broekgaarden22_paper_ii_letter_map():
    """`BROEKGAARDEN22_PAPER_II_LETTERS` maps paper I onto the paper II release.

    The Broekgaarden et al. (2022) Zenodo records relabel the grid; the two
    conventions agree through model E and diverge afterwards (paper II K is the
    optimistic-CE variation, not M_NS,max = 3.0 Msun).
    """
    from grb_io import BROEKGAARDEN21_MODELS, BROEKGAARDEN22_PAPER_II_LETTERS

    # Bijection covering exactly the 20 registry models.
    assert set(BROEKGAARDEN22_PAPER_II_LETTERS) == set(BROEKGAARDEN21_MODELS)
    letters = list(BROEKGAARDEN22_PAPER_II_LETTERS.values())
    assert sorted(letters) == list("ABCDEFGHIJKLMNOPQRST")
    # Agreement through E, then the documented divergence.
    for s in ("A", "B", "C", "D", "E"):
        assert BROEKGAARDEN22_PAPER_II_LETTERS[s] == s
    assert BROEKGAARDEN22_PAPER_II_LETTERS["EH"] == "F"
    assert BROEKGAARDEN22_PAPER_II_LETTERS["F"] == "H"
    assert BROEKGAARDEN22_PAPER_II_LETTERS["I"] == "L"
    assert BROEKGAARDEN22_PAPER_II_LETTERS["K"] == "N"


# ─────────────────────────────────────────────────────────────────────
# Pais, Piran, Kiuchi and Shibata (2025) [papers/Pais_2025.pdf]
#   BNS jet breakout, Eq. (3)/(4)/(5), Gottlieb and Nakar (2022) adapted
# ─────────────────────────────────────────────────────────────────────
def test_pais25_breakout_coefficient():
    """`PAIS_BREAKOUT_COEFF` is the 150 prefactor of Pais (2025) Eq. (4).

    Pais et al. (2025) Eq. (4) write the breakout criterion as
    ``E_j > 150 [(t_d/t_e)^2 + 2] E_ej(<theta_j) theta_j^2``.
    """
    from grb_physics import PAIS_BREAKOUT_COEFF

    assert PAIS_BREAKOUT_COEFF == 150.0


def test_pais25_collimation_normalization():
    """`PAIS_COLL_NORM` is the 3.6e49 erg prefactor of Pais (2025) Eq. (3).

    Pais et al. (2025) Eq. (3):
    ``E_coll = 3.6e49 (theta_j / 0.1 rad)^4 (t_e / 1 s) erg``.  The
    ``pais_collimation_energy`` helper must reproduce this at theta_j =
    0.1 rad and t_e = 1 s.
    """
    from grb_physics import PAIS_COLL_NORM, pais_collimation_energy

    assert PAIS_COLL_NORM == 3.6e49
    assert np.isclose(pais_collimation_energy(0.1, 1.0), 3.6e49)


def test_pais25_ejecta_velocity():
    """`V_EJ_BNS` matches the ~0.15 c post-merger ejecta of Pais (2025) Sec. 2.

    Pais et al. (2025) Sec. 2 report a typical post-merger ejecta velocity
    of ~0.15 c at t ~ 1 s (slower than the ~0.2 c literature average).
    """
    from grb_physics import V_EJ_BNS

    assert V_EJ_BNS == 0.15


def test_pais25_engine_and_delay_time_windows():
    """`T_E_RANGE` / `T_D_PROMPT_RANGE` match Pais (2025) Sec. 3.3 / Sec. 2.

    Pais et al. (2025) Fig. 3 / Sec. 3.3 require engine times t_e ~
    0.05-0.3 s for theta_j = 5-10 deg jets to escape; the prompt-collapse
    funnel forms within ~0.01-0.1 s (Sec. 2, BH formation and funnel
    onset).
    """
    from grb_physics import T_D_PROMPT_RANGE, T_E_RANGE

    assert T_E_RANGE == (0.05, 0.3)
    assert T_D_PROMPT_RANGE == (0.01, 0.1)
    assert T_E_RANGE[0] < T_E_RANGE[1]
    assert T_D_PROMPT_RANGE[0] < T_D_PROMPT_RANGE[1]


def test_pais25_eps_jet_reproduces_gw170817_energy():
    """`EPS_JET` maps a GW170817-like disk to the ~3e49 erg Pais jet anchor.

    Pais et al. (2025) find a 3e49 erg jet launched late with theta_j ~
    5-7 deg matches the GW170817 radio afterglow.  With a GW170817-like
    disk mass (M_disk ~ 1.5e-3 Msun from ``bns_disk_mass`` at R_1.4 =
    12 km), ``jet_energy_from_disk`` must land within a factor of a few of
    3e49 erg.
    """
    from grb_physics import jet_energy_from_disk

    E_j = jet_energy_from_disk(1.5e-3)
    assert 1e49 < E_j < 1e50


def test_z_sun_matches_asplund_2009_photospheric():
    """``Z_SUN`` is the Asplund et al. (2009) photospheric metallicity."""
    from grb_physics import Z_SUN

    assert Z_SUN == pytest.approx(0.0142, abs=1e-5)


def test_metallicity_grid_is_the_broekgaarden_2021_table1_design_grid():
    """The 53-point grid is uniform in log Z, as Broekgaarden+ 2021 Table 1 states.

    Table 1 gives the metallicity range [0.0001, 0.03] "distributed using a
    uniform grid in log(Z_i) with 53 metallicities", and samples about 10^6
    binaries at each grid point.  Equal star-forming mass per point follows
    from that, and is what ``metallicity_prior_mass_fraction`` returns.

    The published values are rounded, most coarsely at the low-Z end, so the
    check is against the design grid rather than against uniform spacing:
    consecutive ln Z gaps in the stored values differ by a factor 2.54, which
    is rounding and not a real non-uniformity.
    """
    from grb_io import METALLICITY_GRID
    from grb_rates import GRID_LOG_UNIFORM_RTOL, metallicity_prior_mass_fraction

    grid = np.unique(METALLICITY_GRID)
    assert grid.size == 53
    assert grid.min() == pytest.approx(1e-4)
    assert grid.max() == pytest.approx(0.03)

    design = np.geomspace(grid.min(), grid.max(), grid.size)
    assert np.abs(grid / design - 1.0).max() == pytest.approx(0.036, abs=0.005)
    np.testing.assert_allclose(grid, design, rtol=GRID_LOG_UNIFORM_RTOL)

    # Rounding, not physics: the stored spacings are far from uniform even
    # though the design grid is.
    gaps = np.diff(np.log(grid))
    assert gaps.max() / gaps.min() == pytest.approx(2.54, rel=0.02)

    f = metallicity_prior_mass_fraction(grid)
    np.testing.assert_allclose(f, 1.0 / grid.size, rtol=1e-12)


def test_ce_lambda_bifurcation_matches_broekgaarden_2022_fig1():
    """``Z_CE_LAMBDA_BIFURCATION`` is the Broekgaarden+ 2022 Fig. 1 artefact Z.

    Their Fig. 1 caption puts the artificial yield jump, from a bifurcation
    in the COMPAS common-envelope lambda values, at Z ~ 0.0105.  It has to
    sit strictly inside the simulated metallicity grid and below solar, or
    the Section 5 cut would either remove everything or nothing.
    """
    from grb_io import METALLICITY_GRID
    from grb_physics import Z_SUN
    from grb_rates import Z_CE_LAMBDA_BIFURCATION

    assert Z_CE_LAMBDA_BIFURCATION == pytest.approx(0.0105, abs=1e-5)
    assert METALLICITY_GRID.min() < Z_CE_LAMBDA_BIFURCATION < Z_SUN
    # The cut has to separate the two grid points the step falls between,
    # otherwise it lands in the middle of a native bin.
    grid = np.unique(METALLICITY_GRID)
    below = grid[grid < Z_CE_LAMBDA_BIFURCATION].max()
    above = grid[grid > Z_CE_LAMBDA_BIFURCATION].min()
    assert below == pytest.approx(0.01018, rel=1e-3)
    assert above == pytest.approx(0.01137, rel=1e-3)


def test_gehrels86_poisson_interval_reproduces_the_published_tables():
    """``gehrels86_poisson_interval`` matches Gehrels (1986) at ``S = 1``.

    Their Table 1 (upper) and Table 2 (lower) one-sigma limits for small n.
    The n = 0 upper limit is the value already pinned as
    ``GEHRELS86_UL_ZERO_COUNTS``, so the two treatments of an empty bin and
    of a nearly-empty one meet at n = 0 instead of jumping.
    """
    from grb_rates import (
        GEHRELS86_ONE_SIGMA_TAIL,
        GEHRELS86_UL_ZERO_COUNTS,
        gehrels86_poisson_interval,
    )

    # The S = 1 column is the one-sided normal tail, which Gehrels rounds to
    # 84.13 per cent in the table headers.
    assert GEHRELS86_ONE_SIGMA_TAIL == pytest.approx(1.0 - 0.8413, abs=1e-4)

    n = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0])
    lo, hi = gehrels86_poisson_interval(n)
    np.testing.assert_allclose(hi, [1.841, 3.300, 4.638, 5.918, 8.382, 14.267], rtol=1e-3)
    np.testing.assert_allclose(lo[1:], [0.173, 0.708, 1.367, 2.840, 6.891], rtol=2e-3)
    assert hi[0] == pytest.approx(GEHRELS86_UL_ZERO_COUNTS, rel=1e-3)

    # Gaussian limit: both bounds go to sqrt(n) for large n, which is what
    # makes the asymmetric bar a generalisation of sqrt(sum w^2) and not a
    # different estimator.
    big = np.array([1.0e4])
    lo_b, hi_b = gehrels86_poisson_interval(big)
    assert (hi_b[0] - big[0]) / np.sqrt(big[0]) == pytest.approx(1.0, rel=0.02)
    assert (big[0] - lo_b[0]) / np.sqrt(big[0]) == pytest.approx(1.0, rel=0.02)


@pytest.mark.requires_data
def test_ce_lambda_bifurcation_is_visible_in_the_compas_lambda_column(bns_a_path):
    """The CE-lambda artefact is in the data, not only in a caption.

    Broekgaarden+ 2022 Fig. 1 attribute the yield step near Z = 0.0105 to
    "an artificial bifurcation in our lambda values for the common-envelope
    treatment in COMPAS at this Z", without saying what the lambda values do.
    Every Section 5 / 5b caveat and the shaded band rest on that claim, so it
    is checked against the stored lambda column instead of taken on trust.

    Model A adopts the Nanjing prescription.  Read the step off ``lambda1``
    only after dropping the events where star 1 has no envelope to unbind:
    there ``lambda1`` is exactly 1.0 as a default, and the share of such events
    jumps from 0.3 to 63 per cent across the step, so a median over the raw
    column tracks which star is the donor rather than the prescription.
    Restricted to real donors the median lambda falls by a factor 3 and the
    median envelope binding energy rises by a factor 4, which is the direction
    Broekgaarden+ 2022 Sec. 3.1 describe: more orbital energy is needed to
    unbind the envelope above Z = 0.0105.
    """
    import h5py

    from grb_rates import Z_CE_LAMBDA_BIFURCATION

    with h5py.File(bns_a_path, "r") as f:
        ce = f["commonEnvelopes"]
        lam = ce["lambda1"][...].squeeze()
        lam_nanjing = ce["lambdaNanjing1"][...].squeeze()
        m_env = ce["massEnvelope1"][...].squeeze()
        binding = ce["bindingEnergy1"][...].squeeze()
        seed = ce["randomSeed"][...].squeeze()
        sys_seed = f["systems"]["SEED"][...].squeeze()
        sys_Z = f["systems"]["Metallicity1"][...].squeeze()

    live = np.isfinite(lam) & (lam > 0)
    assert np.isclose(lam[live], lam_nanjing[live]).all(), (
        "Model A is expected to adopt lambda_Nanjing; a different prescription "
        "would change what the artefact is"
    )
    # lambda1 == 1.0 is the no-envelope default, not a fitted lambda.
    clamped = np.isclose(lam, 1.0)
    assert (m_env[clamped] == 0).all()
    assert (binding[clamped] == 0).all()
    assert (m_env[~clamped] > 0).all()

    uniq, first = np.unique(sys_seed, return_index=True)
    Z = sys_Z[first[np.clip(np.searchsorted(uniq, seed), 0, uniq.size - 1)]]

    grid = np.unique(Z)
    below = grid[grid < Z_CE_LAMBDA_BIFURCATION].max()
    above = grid[grid > Z_CE_LAMBDA_BIFURCATION].min()
    assert below == pytest.approx(0.01018, rel=1e-3)
    assert above == pytest.approx(0.01137, rel=1e-3)

    donor = live & (m_env > 0)
    at_below = donor & np.isclose(Z, below, rtol=1e-4)
    at_above = donor & np.isclose(Z, above, rtol=1e-4)

    # Donor composition itself steps at the branch, which is why the cut matters.
    frac_below = float((~clamped)[np.isclose(Z, below, rtol=1e-4)].mean())
    frac_above = float((~clamped)[np.isclose(Z, above, rtol=1e-4)].mean())
    assert frac_below == pytest.approx(0.997, abs=0.01)
    assert frac_above == pytest.approx(0.369, abs=0.01)

    med_below = float(np.median(lam[at_below]))
    med_above = float(np.median(lam[at_above]))
    assert med_below == pytest.approx(0.387, rel=0.02)
    assert med_above == pytest.approx(0.125, rel=0.02)
    assert med_below / med_above > 2.0, "the bifurcation has to be a step, not a drift"

    # A lower lambda is a more tightly bound envelope, so the binding energy
    # has to move the other way for the Broekgaarden+ 2022 mechanism to hold.
    be_below = float(np.median(binding[at_below]))
    be_above = float(np.median(binding[at_above]))
    assert be_below == pytest.approx(1.10e48, rel=0.02)
    assert be_above == pytest.approx(4.31e48, rel=0.02)
    assert be_above / be_below > 2.0


# ─────────────────────────────────────────────────────────────────────
# Observed local sGRB rates: Wanderman and Piran (2015) MNRAS 448, 3026
# [papers/Wanderman_2015.pdf]; Ghirlanda et al. (2016) A&A 594, A84
# [papers/Ghirlanda_2016.pdf]; Colombo et al. (2022) ApJ 937, 79
# [papers/Colombo_2022.pdf].
# ─────────────────────────────────────────────────────────────────────
def test_observed_sgrb_rates_match_the_published_local_rates():
    """Every ``OBSERVED_SGRB_RATES`` entry is the rate density its paper reports.

    The dict is documented as observed, beaming-limited local rates in
    Gpc^-3 yr^-1, so an entry has to be that quantity, in those units, with
    the paper's own interval.

    - Wanderman and Piran (2015) Sec. 5: R_0 = 4.1 +2.3/-1.9, so [2.2, 6.4].
    - Ghirlanda et al. (2016) Sec. 7: rho_0,a = 0.20 +0.04/-0.07 and
      rho_0,c = 0.8 +0.3/-0.15 for their two models, carried as two entries
      because the paper does not prefer one.

    This test exists because the dict previously carried a Ghirlanda entry of
    1.3 [0.5, 3.0], a value that appears nowhere in that paper, and a Colombo
    et al. (2022) entry of 3.6 [1.8, 6.5] taken from their Table 1 row of
    predicted *detections per year* per instrument.  Nothing pinned either
    against its source, so both propagated into the Section 6c comparison
    envelope unchallenged.
    """
    from grb_rates import COLOMBO22_NSNS_LOCAL_RATE, OBSERVED_SGRB_RATES

    wp = OBSERVED_SGRB_RATES["Wanderman & Piran 2015"]
    assert (wp["R_obs"], wp["R_obs_lo"], wp["R_obs_hi"]) == pytest.approx(
        (4.1, 2.2, 6.4), abs=0.05
    ), "WP15 local rate drifted from R_0 = 4.1 +2.3/-1.9 Gpc^-3 yr^-1."

    ga = OBSERVED_SGRB_RATES["Ghirlanda+ 2016 (model a)"]
    assert (ga["R_obs"], ga["R_obs_lo"], ga["R_obs_hi"]) == pytest.approx(
        (0.20, 0.13, 0.24), abs=0.01
    ), "Ghirlanda model (a) drifted from rho_0 = 0.20 +0.04/-0.07 Gpc^-3 yr^-1."

    gc = OBSERVED_SGRB_RATES["Ghirlanda+ 2016 (model c)"]
    assert (gc["R_obs"], gc["R_obs_lo"], gc["R_obs_hi"]) == pytest.approx(
        (0.8, 0.65, 1.1), abs=0.01
    ), "Ghirlanda model (c) drifted from rho_0 = 0.8 +0.3/-0.15 Gpc^-3 yr^-1."

    # Colombo et al. report an intrinsic NSNS merger rate, not a beamed sGRB
    # rate, so it must stay out of this dict however tempting the number is.
    assert not any("Colombo" in k for k in OBSERVED_SGRB_RATES), (
        "Colombo et al. (2022) is an intrinsic NSNS local merger rate "
        "(347 +536/-256 Gpc^-3 yr^-1), not a beaming-limited observed sGRB "
        "rate; it belongs in COLOMBO22_NSNS_LOCAL_RATE."
    )
    assert COLOMBO22_NSNS_LOCAL_RATE["R"] == pytest.approx(347.0, abs=1.0)

    # Every surviving entry is a beamed rate, so all must sit far below the
    # LVK intrinsic BNS band: beaming suppresses by 1 - cos(theta_j) ~ 0.03.
    from grb_rates import LVK_GWTC5_LOCAL_RATES

    for name, entry in OBSERVED_SGRB_RATES.items():
        assert entry["R_obs"] < LVK_GWTC5_LOCAL_RATES["BNS"]["R_lo"], (
            f"{name} R_obs = {entry['R_obs']} is not below the LVK intrinsic "
            f"BNS lower edge; a beamed rate must be, or it is not beamed."
        )


def test_a_bh_fid_is_a_declared_fiducial_inside_the_foucart_calibration_range():
    """``A_BH_FID`` is a project choice, and every consumer must share it.

    No paper prefers chi_BH = 0.5.  What can be checked is that it sits inside
    the Foucart et al. (2018) calibration interval chi_BH in [-0.5, 0.9]
    (arXiv:1807.00011 Eq. 4), that it is the centre of the standard sweep, and
    that it lives in ``grb_physics`` rather than being retyped per notebook
    cell, which is how it drifted before: ``classify_bhns`` carried a literal
    0.5 default while the notebook defined its own copy twice.
    """
    import inspect

    from grb_classify import classify_bhns
    from grb_physics import A_BH_FID

    assert A_BH_FID == 0.5
    assert -0.5 <= A_BH_FID <= 0.9, "outside the Foucart 2018 calibrated chi range"
    assert A_BH_FID in (0.0, 0.3, 0.5, 0.7, 0.9)

    default = inspect.signature(classify_bhns).parameters["a_BH"].default
    assert default is A_BH_FID, (
        "classify_bhns no longer defaults to grb_physics.A_BH_FID; the fiducial "
        "spin has forked again."
    )


# ─────────────────────────────────────────────────────────────────────
# Gottlieb et al. (2025) arXiv:2411.13657 [papers/Gottlieb_2024.pdf]
# Eq. (10) and (11) inputs used by comparison.ipynb.
# ─────────────────────────────────────────────────────────────────────
def test_gottlieb25_f_range_is_eq10_fb_over_epsilon_bounds():
    """``GOTTLIEB25_F_RANGE`` spans f_b / eps_gamma for Eq. (10)'s bounds.

    Gottlieb+25 Eq. (10): ``0.01 <= f_b <= 0.11`` and
    ``0.15 <= eps_gamma <= 0.5``, so f_b / eps_gamma runs from 0.01 / 0.5 to
    0.11 / 0.15 exactly, and f_{-1} = 0.1 eps_gamma / f_b spans [0.136, 5].
    """
    from grb_physics import GOTTLIEB25_F_RANGE

    assert GOTTLIEB25_F_RANGE[0] == pytest.approx(0.01 / 0.5, rel=1e-12)
    assert GOTTLIEB25_F_RANGE[1] == pytest.approx(0.11 / 0.15, rel=1e-12)
    f_m1 = 0.1 / np.array(GOTTLIEB25_F_RANGE)
    assert f_m1.min() == pytest.approx(0.1 * 0.15 / 0.11, rel=1e-12)
    assert f_m1.max() == pytest.approx(5.0, rel=1e-12)


def test_gottlieb25_eq11_scales_with_inverse_f_minus_1():
    """Eq. (11) carries f_{-1}^{-1}: raising f_{-1} lowers the predicted M_ej.

    From Eqs. (9)-(10), E_j = E_iso / (10 f_{-1}) and M_ej = 0.3 M_d, giving
    M_ej ~ 1.2e-3 f_{-1}^{-1} (E_iso / 2e51) (T50 / s)^(alpha - 1) Msun, which
    Gottlieb+25 round to 1e-3.  ``gottlieb25_eq11`` multiplies by its
    ``f_inv`` argument, so callers must pass 1 / f_{-1}.  The notebook once
    passed f_{-1} itself, biasing every prediction low by ~0.17 dex.
    """
    from grb_physics import gottlieb25_eq11

    f_m1 = 2.0
    m = gottlieb25_eq11(1.0, 2e51, alpha=2.0, f_inv=1.0 / f_m1)
    assert float(m) == pytest.approx(1e-3 / f_m1, rel=1e-12)


def test_gottlieb25_wind_frac_range_is_weak_and_strong_field_values():
    """``GOTTLIEB25_WIND_FRAC_RANGE`` is Gottlieb+25 Sec. 3.1's 0.3 to 0.4.

    Sec. 3.1: disks with strong (weak) magnetic fields eject
    ``M_ej ~ 0.4 (0.3) M_d``, and the paper adopts 0.3.  The earlier
    [0.2, 0.4] range cited Radice et al. (2018, arXiv:1809.11163), which
    reports no wind fraction.
    """
    from grb_physics import GOTTLIEB25_WIND_FRAC, GOTTLIEB25_WIND_FRAC_RANGE

    assert GOTTLIEB25_WIND_FRAC_RANGE == (0.3, 0.4)
    assert GOTTLIEB25_WIND_FRAC == pytest.approx(0.3, abs=1e-12)
