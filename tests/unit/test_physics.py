"""Validation tests for Foucart 2018 / Kruger-Foucart 2020 / Neijssel 2019.

One test per cited paper, anchoring the corresponding module function
to a published equation or check point.  All tests are pure-function
and data-free except where explicitly marked ``requires_data``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from grb_physics import (
    EOS_MODELS,
    K_THRESH_DEFAULT,
    M_THRESH,
    M_TOV,
    _compactness,
    bhns_dynamical_ejecta,
    chirp_mass,
    foucart_disk_mass,
    foucart_remnant_mass,
    ns_baryon_mass,
    r_isco,
)


# ─────────────────────────────────────────────────────────────────────
# Foucart, Hinderer & Nissanke (2018) PRD 98, 081501 -- Eq. (4)
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def foucart_canonical_inputs():
    """Representative BHNS configuration well inside Foucart 2018 calibration.

    Q = 5, chi_BH = 0.5, M_NS = 1.35 Msun, R_NS = 12 km.  Foucart (2018)
    calibrated for Q in [1, 7], chi_BH in [-0.5, 0.9], C_NS in [0.13,
    0.182]; this point sits in the middle of all three intervals.
    """
    return dict(M_BH=6.75, M_NS=1.35, a_BH=0.5, R_NS_km=12.0)


def test_foucart_remnant_matches_eq4_by_hand(foucart_canonical_inputs):
    """Recompute Eq. (4) elementwise and compare to the module function.

    Anchors against Foucart, Hinderer and Nissanke (2018) PRD 98, 081501,
    Eq. (4) with their Table I best-fit (alpha, beta, gamma, delta) =
    (0.406, 0.139, 0.255, 1.761).  This is the exact paper formula, not
    a regression value, so a future code edit that changes coefficients
    or sign conventions will fail loudly.
    """
    args = foucart_canonical_inputs
    M_BH = args["M_BH"]
    M_NS = args["M_NS"]
    a_BH = args["a_BH"]
    R = args["R_NS_km"]

    Q = M_BH / M_NS  # noqa: F841 (Foucart 2018 Eq. 4 setup; Q kept as literature anchor)
    C_NS = float(_compactness(M_NS, R))
    eta = M_NS * M_BH / (M_NS + M_BH) ** 2
    R_hat = float(r_isco(a_BH))
    alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761
    bracket = alpha * (1 - 2 * C_NS) / eta ** (1 / 3) - beta * R_hat * C_NS / eta + gamma
    M_b = float(ns_baryon_mass(M_NS))
    expected = max(0.0, bracket) ** delta * M_b

    got = float(foucart_remnant_mass(**args))
    assert got == pytest.approx(expected, rel=1e-10), (
        f"foucart_remnant_mass deviates from Eq. (4) by-hand: "
        f"got {got:.6f}, expected {expected:.6f}"
    )


def test_foucart_remnant_is_finite_and_nonnegative(foucart_canonical_inputs):
    M_rem = float(foucart_remnant_mass(**foucart_canonical_inputs))
    assert np.isfinite(M_rem)
    assert M_rem >= 0.0
    # A canonical Q=5 BHNS produces a disk-tier remnant in the
    # ~0.05 - 0.5 Msun range; this is a wide sanity bound.
    assert 0.01 < M_rem < 1.0, f"M_rem = {M_rem:.4f} Msun outside sane range"


def test_foucart_remnant_zero_for_no_disruption():
    """Q >> 1 with chi = 0 swallows the NS without disruption.

    Foucart (2018) Eq. (4) fit value approaches zero (clipped at 0 by
    np.maximum) when the bracket goes negative, which happens for high
    Q at low spin.
    """
    with warnings.catch_warnings():
        # Q > 7 is outside the calibration range; the module emits a
        # bulk-warning we don't care about here.
        warnings.simplefilter("ignore")
        M_rem = float(foucart_remnant_mass(M_BH=20.0, M_NS=1.35, a_BH=0.0, R_NS_km=11.0))
    assert M_rem == 0.0


def test_foucart_disk_mass_subtracts_dynamical_ejecta(foucart_canonical_inputs):
    """foucart_disk_mass = M_rem - M_dyn (default behavior, f_disk=None)."""
    M_rem = float(foucart_remnant_mass(**foucart_canonical_inputs))
    M_dyn = float(
        bhns_dynamical_ejecta(
            foucart_canonical_inputs["M_BH"],
            foucart_canonical_inputs["M_NS"],
            foucart_canonical_inputs["a_BH"],
            R_NS_km=foucart_canonical_inputs["R_NS_km"],
        )
    )
    M_disk = float(foucart_disk_mass(**foucart_canonical_inputs))
    assert M_disk == pytest.approx(max(0.0, M_rem - M_dyn), rel=1e-10)
    assert M_disk >= 0.0


def test_foucart_disk_mass_legacy_f_disk_path(foucart_canonical_inputs):
    """Legacy f_disk path returns f_disk * M_rem unchanged."""
    M_rem = float(foucart_remnant_mass(**foucart_canonical_inputs))
    M_disk = float(foucart_disk_mass(f_disk=0.4, **foucart_canonical_inputs))
    assert M_disk == pytest.approx(0.4 * M_rem, rel=1e-12)


def test_foucart_warns_outside_calibration():
    """|chi| > 0.9 must emit the Foucart (2018) calibration warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        foucart_remnant_mass(M_BH=7.0, M_NS=1.4, a_BH=0.95, R_NS_km=12.0)
    chi_warn = any("|chi_BH| > 0.9" in str(item.message) for item in w)
    assert chi_warn, [str(item.message) for item in w]


# ─────────────────────────────────────────────────────────────────────
# Kruger & Foucart (2020) Eq. (9) BHNS dynamical ejecta
# ─────────────────────────────────────────────────────────────────────
def test_kruger_foucart_dyn_ejecta_finite():
    """KF2020 Eq. (9) returns a finite, non-negative ejecta mass."""
    M_dyn = float(bhns_dynamical_ejecta(M_BH=6.75, M_NS=1.35, a_BH=0.5, R_NS_km=12.0))
    assert np.isfinite(M_dyn)
    assert M_dyn >= 0.0
    # Dynamical ejecta in the disk-disrupting BHNS regime is typically
    # a few times 1e-3 to a few 1e-2 Msun (KF2020 Fig. 4); broad bound.
    assert M_dyn < 0.1


# Bin-averaged dPdlogZ tests removed when the rate path migrated to
# FCI's ``find_metallicity_distribution`` (see
# ``tests/unit/test_rates.py::test_check_dPdlogZ_normalization_runs_on_compas_output``
# and the literature anchor ``test_levina_2026_tng100_mssfr_parameters_match_table_1``).


# ─────────────────────────────────────────────────────────────────────
# Project-level constants (catches accidental refactors)
# ─────────────────────────────────────────────────────────────────────
def test_M_THRESH_equals_K_times_M_TOV():
    """EOS-sweep coherence: M_THRESH must be exactly K_THRESH_DEFAULT * M_TOV."""
    assert M_THRESH == pytest.approx(K_THRESH_DEFAULT * M_TOV, rel=1e-12)


def test_eos_models_carry_M_crit_R_1p4_M_TOV():
    """EOS_MODELS must populate the three sweep parameters used downstream."""
    for name, eos in EOS_MODELS.items():
        for key in ("M_crit", "R_1p4", "M_TOV"):
            assert key in eos, f"EOS_MODELS[{name!r}] missing {key!r}"
            assert eos[key] > 0


# ─────────────────────────────────────────────────────────────────────
# Data-bound tests (skipped if Data/ empty)
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_load_bns_mass_ordering(bns_a_path):
    """m1 >= m2 invariant after load_bns(sort_masses=True)."""
    from grb_io import load_bns

    bns = load_bns(path=bns_a_path)
    assert (bns["m1"] >= bns["m2"]).all(), "load_bns violated the m1 >= m2 invariant on Model A"
    # Sanity bounds on the COMPAS NS-mass distribution.
    assert bns["m1"].min() > 0.5
    assert bns["m1"].max() < 3.5


@pytest.mark.requires_data
def test_stroopwafel_weights_positive(bns_a_path):
    """STROOPWAFEL weights are strictly positive (importance weights)."""
    from grb_io import load_bns

    w = load_bns(path=bns_a_path)["weights"]
    assert (w > 0).all()
    assert np.isfinite(w).all()


@pytest.mark.requires_data
@pytest.mark.slow
def test_stroopwafel_local_rate_vs_broekgaarden_table3(bns_a_path):
    """w_000 sum recovers a Broekgaarden+ (2021) Table 3 BNS Model A rate.

    The HDF5's ``weights_intrinsic/w_000`` column stores per-system
    contributions to R(z=0); their sum is the BNS Model A intrinsic
    local merger rate density [Gpc^-3 yr^-1].  The locally checked
    archive returns ~33 Gpc^-3 yr^-1, well inside the Broekgaarden+
    2021 (arXiv:2103.02608) BNS rate band of ~10 - 300 across MSSFR /
    metallicity-prior variations (their Table 3 fiducial line plus
    surrounding discussion).

    A weight-extraction bug in ``compas_python_utils`` would manifest
    as a value of zero, NaN, or many orders of magnitude off; this test
    catches that regime without baking in a single reference number.
    """
    from grb_io import read_expected_local_rate

    R_local = read_expected_local_rate(bns_a_path)
    assert np.isfinite(R_local), (
        f"Local intrinsic rate is not finite ({R_local}); "
        "suspect compas_python_utils weight extraction."
    )
    assert 5.0 < R_local < 500.0, (
        f"Local intrinsic rate {R_local:.1f} Gpc^-3 yr^-1 is outside "
        f"the Broekgaarden+ (2021) BNS rate band of ~10 - 300; "
        f"suspect compas_python_utils weight extraction or HDF5 "
        f"column drift."
    )


@pytest.mark.requires_data
@pytest.mark.slow
def test_stroopwafel_per_class_local_rates_within_band(bns_a_path):
    """Per-class round-trip vs Broekgaarden+ (2021) Table 3.

    Splits the ``weights_intrinsic/w_000`` column by Gottlieb (2024)
    four-class mask and asserts that

      (a) each class rate is finite and lies in a literature-consistent
          band wide enough to absorb prior shifts but narrow enough to
          catch order-of-magnitude regressions;
      (b) the sum of the four class rates equals
          ``read_expected_local_rate(...)`` exactly (sum-of-parts identity).

    The remap to the Alsing-Silva-Berti (2018) Galactic-NS double
    Gaussian is applied to mirror the figure-pipeline state in
    ``grb_main.ipynb``; the remap is rank-preserving and weight-conserving
    so the total rate is invariant under it (the per-class split is not).
    """
    import h5py as h5

    from grb_classify import classify_bns_2024
    from grb_io import load_bns, read_expected_local_rate
    from grb_physics import remap_ns_masses_double_gaussian

    bns = load_bns(path=bns_a_path)
    m1_remap, m2_remap = remap_ns_masses_double_gaussian(
        bns["m1"],
        bns["m2"],
        weights=bns["weights"],
        rng=np.random.default_rng(42),
    )

    cls = classify_bns_2024(m1_remap, m2_remap)
    class_keys = [
        "sbGRB + blue KN",
        "lbGRB + red KN (HMNS)",
        "lbGRB + red KN (disk)",
        "Faint lbGRB",
    ]
    masks = {k: np.asarray(cls[k], dtype=bool) for k in class_keys}

    with h5.File(bns_a_path, "r") as f:
        w_000 = f["weights_intrinsic"]["w_000"][...].squeeze()

    assert w_000.shape[0] == m1_remap.shape[0], (
        f"weights_intrinsic/w_000 length {w_000.shape[0]} does not "
        f"match merging-system count {m1_remap.shape[0]}; HDF5 layout "
        f"changed and the alignment assumption is broken."
    )

    R_total = float(w_000.sum())
    R_per_class = {k: float(w_000[m].sum()) for k, m in masks.items()}

    bands = {
        "sbGRB + blue KN": (0.0, 100.0),
        "lbGRB + red KN (HMNS)": (1.0, 300.0),
        "lbGRB + red KN (disk)": (0.0, 100.0),
        "Faint lbGRB": (0.0, 100.0),
    }
    for k, R in R_per_class.items():
        lo, hi = bands[k]
        assert np.isfinite(R), f"{k!r} rate is not finite: {R}"
        assert lo <= R < hi, (
            f"{k!r} per-class rate {R:.2f} Gpc^-3 yr^-1 outside the "
            f"literature-consistent band [{lo}, {hi})"
        )

    R_sum = sum(R_per_class.values())
    R_ref = read_expected_local_rate(bns_a_path)
    assert R_sum == pytest.approx(R_ref, rel=1e-9), (
        f"Sum-of-parts {R_sum:.6f} != total {R_ref:.6f} from "
        f"read_expected_local_rate; the four Gottlieb 2024 class "
        f"masks do not partition the BNS sample."
    )
    assert R_total == pytest.approx(R_ref, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────
# Chirp mass: Peters (1964) PRD 136, B1224 -- definition
# ─────────────────────────────────────────────────────────────────────
def test_chirp_mass_equal_mass_identity():
    """For m1 = m2 = m, M_chirp = m * (1/2)**(1/5).

    Peters (1964) defines M_chirp = (m1 m2)**(3/5) / (m1 + m2)**(1/5);
    setting m1 = m2 = m gives (m^2)**(3/5) / (2m)**(1/5)
    = m / 2**(1/5) = m * (1/2)**(1/5) ~ 0.8706 m.  Anchors the helper
    to the closed-form algebra.
    """
    m = 1.4
    expected = m * (0.5) ** 0.2
    assert chirp_mass(m, m) == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(
        chirp_mass(np.array([m, 2 * m]), np.array([m, 2 * m])),
        np.array([expected, 2 * expected]),
        rtol=1e-12,
        atol=0,
    )


def test_chirp_mass_ordering_invariance():
    """``chirp_mass`` is symmetric in (m1, m2): swap-invariant.

    The ``m1 >= m2`` invariant is enforced at load time in
    ``grb_io.load_bns`` / ``load_bhns``, but chirp mass is a derived
    quantity that must be order-agnostic so the helper does not
    silently shift if a future caller violates the ordering.
    """
    rng = np.random.default_rng(0)
    m1 = rng.uniform(0.8, 2.5, size=64)
    m2 = rng.uniform(0.8, 2.5, size=64)
    np.testing.assert_allclose(chirp_mass(m1, m2), chirp_mass(m2, m1), rtol=1e-15, atol=0)


def test_chirp_mass_scalar_array_parity():
    """Scalar and array-of-one inputs must agree to machine precision.

    Vectorisation parity guard for the ``np.asarray`` cast inside
    ``chirp_mass``.
    """
    m1, m2 = 1.35, 1.30
    s = float(chirp_mass(m1, m2))
    a = chirp_mass(np.array([m1]), np.array([m2]))
    assert a.shape == (1,)
    assert s == pytest.approx(float(a[0]), rel=1e-15)
    expected = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    assert s == pytest.approx(expected, rel=1e-15)
