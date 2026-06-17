"""Validation tests for ``grb_rates.py``.

Anchors the cosmic-rate convolution and the per-class beaming helpers
to (a) the cited literature (Fong+ 2015, Beniamini and Nakar 2019,
Neijssel+ 2019, Madau and Dickinson 2014, Wanderman and Piran 2015) and
(b) the COMPAS reference implementation in
``compas_python_utils.cosmic_integration.FastCosmicIntegration``.

The COMPAS-vs-grb_rates agreement test is marked ``requires_data`` plus
``requires_compas`` because it loads ``data/COMPASCompactOutput_BNS_A.h5``
and runs the upstream find_formation_and_merger_rates side by side with
``compute_merger_rate``; it is skipped cleanly on machines that do not
have either piece available.
"""

from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────
# Beaming round-trip: f_beam = 1 - cos(theta_j); Fong+ 2015 ApJ 815, 102
# ─────────────────────────────────────────────────────────────────────
def test_beaming_round_trip_sbgrb():
    """sbGRB fiducial: theta_j = 13 deg => f_beam ~ 0.0256."""
    from grb_rates import CLASS_THETA_J, beamed_rate

    theta_fid = CLASS_THETA_J["sbGRB"]["fid"]
    expected = 100.0 * (1.0 - np.cos(np.radians(theta_fid)))
    got = beamed_rate(100.0, theta_fid)
    assert np.isclose(got, expected, rtol=1e-12)
    assert 2.55 < got < 2.58, got


def test_beaming_round_trip_lbgrb():
    """lbGRB fiducial: theta_j = 6.5 deg => f_beam ~ 0.00643."""
    from grb_rates import CLASS_THETA_J, beamed_rate

    theta_fid = CLASS_THETA_J["lbGRB"]["fid"]
    got = beamed_rate(100.0, theta_fid)
    assert 0.64 < got < 0.65, got


def test_beamed_rate_mixed_matches_per_class_sum():
    """Mixed-population beaming must equal the per-class sum.

    Calling ``beamed_rate(R_total, <theta_j>)`` instead would silently
    overestimate observed rates because ``1 - cos`` is non-linear in
    theta_j.
    """
    from grb_rates import CLASS_THETA_J, beamed_rate, beamed_rate_mixed

    rates = {"sbGRB": 50.0, "lbGRB": 50.0}
    angles = {k: CLASS_THETA_J[k]["fid"] for k in rates}

    expected = beamed_rate(50.0, CLASS_THETA_J["sbGRB"]["fid"]) + beamed_rate(
        50.0, CLASS_THETA_J["lbGRB"]["fid"]
    )
    got = beamed_rate_mixed(rates, angles)
    assert np.isclose(got, expected, rtol=1e-12)

    # Sanity: this is NOT what scalar averaging would give
    naive = beamed_rate(
        100.0, 0.5 * (CLASS_THETA_J["sbGRB"]["fid"] + CLASS_THETA_J["lbGRB"]["fid"])
    )
    assert not np.isclose(got, naive, rtol=1e-3), (
        "beamed_rate_mixed agrees with the naive scalar average; the "
        "non-linearity of (1 - cos) should make these differ."
    )


def test_beamed_rate_mixed_missing_class_raises():
    """A rate class without a corresponding theta_j is a programmer
    error and must raise loudly rather than silently dropping the
    class from the sum."""
    from grb_rates import beamed_rate_mixed

    with pytest.raises(KeyError):
        beamed_rate_mixed({"sbGRB": 1.0, "lbGRB": 1.0}, {"sbGRB": 13.0})


# ─────────────────────────────────────────────────────────────────────
# Cosmology pin: Planck 2015 (Ade et al. 2016, A&A 594, A13).
# ─────────────────────────────────────────────────────────────────────
def test_cosmology_planck15_h0():
    """Planck 2015 H0 = 67.74 km/s/Mpc (Planck Collaboration 2016
    A&A 594, A13).  This is the value the manuscript prose claims and
    the value grb_main.ipynb passes to calculate_redshift_related_params
    after the 2026-05-06 cosmology fix.  If a future change reverts the
    notebook to the COMPAS Planck18 default (H0 = 67.4) this test
    catches it before the rates drift by ~2 percent at high z.
    """
    from astropy.cosmology import Planck15

    assert abs(Planck15.H0.value - 67.74) < 0.01


# Voronoi-bin metallicity test removed: ``_bin_averaged_dPdlogZ`` was
# deleted when the rate path migrated to FCI's ``find_metallicity_distribution``
# (see ``test_check_dPdlogZ_normalization_runs_on_compas_output`` for the
# FCI-side normalisation check).


@pytest.mark.requires_compas
def test_check_dPdlogZ_normalization_runs_on_compas_output():
    """``check_dPdlogZ_normalization`` should accept the COMPAS
    ``find_metallicity_distribution`` output without raising.
    Skips cleanly if the upstream package is unavailable."""
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )
    find_metallicity_distribution = fci.find_metallicity_distribution
    from grb_rates import check_dPdlogZ_normalization

    redshifts = np.linspace(0.0, 5.0, 51)
    dPdlogZ, metallicities, _ = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(1e-4),
        max_logZ_COMPAS=np.log(0.03),
    )
    norm = check_dPdlogZ_normalization(dPdlogZ, metallicities, rtol=0.10)
    assert np.all(np.isfinite(norm))


# ─────────────────────────────────────────────────────────────────────
# Per-class partition invariant for Section 7 BNS rate plot
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_compas
def test_per_class_rates_partition_total_bns_2024():
    """The four ``classify_bns_2024`` classes partition the BNS sample,
    so ``sum_c R_c(z) == R_total(z)`` to floating-point precision when
    ``compute_merger_rate`` is called with the same ``n_formed``,
    ``p_draw``, and shared FCI ``dPdlogZ``/``metallicities`` for each
    subset.

    Locks the invariant the Section 7 BNS rate plot relies on:
    ``compute_merger_rate`` is a pure additive accumulator (FCI's
    ``find_formation_and_merger_rates`` summed axis-0 across chunks),
    so disjoint masks that union to the full sample must produce rates
    that sum to the all-sample rate.
    """
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
        find_metallicity_distribution,
        find_sfr,
    )

    from grb_classify import classify_bns_2024
    from grb_rates import compute_merger_rate

    rng = np.random.default_rng(0)

    m1 = np.concatenate(
        [
            np.full(15, 1.20),
            np.full(15, 1.40),
            np.full(10, 2.10),
            np.full(10, 1.55),
        ]
    )
    m2 = np.concatenate(
        [
            np.full(15, 1.20),
            np.full(15, 1.30),
            np.full(10, 1.40),
            np.full(10, 1.45),
        ]
    )
    cls = classify_bns_2024(m1, m2)
    for label, c in cls.items():
        assert int(c.sum()) > 0, f"no test systems land in class {label!r}"

    Z_full = np.array([1e-4, 1e-3, 1e-2, 3e-2])
    Z_compas = rng.choice(Z_full, size=len(m1))
    delays = rng.uniform(50.0, 5000.0, size=len(m1))
    weights = rng.uniform(0.5, 2.0, size=len(m1))

    # Use FCI's coherent (z, t, time_first_SF) grid so the times_to_z
    # interpolation does not run out of bounds for any (binary, z) pair.
    redshifts, _, times, time_first_SF, _, _ = calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.1, cosmology=Planck15
    )
    sfr = find_sfr(redshifts)

    dPdlogZ, metallicities, p_draw = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(Z_full[0]),
        max_logZ_COMPAS=np.log(Z_full[-1]),
    )

    common = dict(
        redshifts=redshifts,
        times=times,
        time_first_SF=time_first_SF,
        n_formed=sfr,
        p_draw=p_draw,
        dPdlogZ=dPdlogZ,
        metallicities=metallicities,
        smooth_sigma=0,
    )

    R_total = compute_merger_rate(
        COMPAS_Z=Z_compas,
        COMPAS_delay_times=delays,
        COMPAS_weights=weights,
        **common,
    )

    R_per_class = {}
    for label, mask in cls.items():
        R_per_class[label] = compute_merger_rate(
            COMPAS_Z=Z_compas[mask],
            COMPAS_delay_times=delays[mask],
            COMPAS_weights=weights[mask],
            **common,
        )
    R_sum = sum(R_per_class.values())

    np.testing.assert_allclose(R_total, R_sum, rtol=1e-12, atol=0)


# ─────────────────────────────────────────────────────────────────────
# Calibration anchor: MEAN_MASS_EVOLVED must be redshift_step-invariant
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_compas
def test_calibrate_mean_mass_evolved_redshift_step_invariant():
    """``MEAN_MASS_EVOLVED`` must be ~stable across ``redshift_step``.

    The function back-derives MEAN_MASS_EVOLVED from the Neijssel
    ``w_000`` anchor at the sharp pointwise R(z=0); ``smooth_sigma=0``
    is hard-coded inside the helper so the calibration does not pick
    up the ``gaussian_filter1d`` ``mode='reflect'`` boundary bias that
    would otherwise let the result drift with ``redshift_step``.

    With FCI's per-binary ceiling-snap (``np.ceil(z_form / dz)``)
    instead of the project's old linear interpolation, the residual
    drift is dominated by single-bin discretisation of ``z_form``;
    ~0.3 percent at dz between 0.005 and 0.01 (vs ~30 percent before
    the smooth_sigma=0 fix).  The 1 percent threshold below leaves
    headroom while still catching a regression of the boundary bias.
    """
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
    )

    from grb_rates import calibrate_mean_mass_evolved

    rng = np.random.default_rng(0)
    N = 500
    Z_grid = 10.0 ** np.linspace(-4, np.log10(0.03), 8)
    Z = rng.choice(Z_grid, size=N)
    delays = 10.0 ** rng.uniform(np.log10(10.0), np.log10(13000.0), size=N)
    w = rng.uniform(0.1, 1.0, size=N)
    expected = 33.0

    def _calibrate(dz):
        redshifts, _, times, time_first_SF, _, _ = calculate_redshift_related_params(
            max_redshift=10.0, redshift_step=dz, cosmology=Planck15
        )
        return calibrate_mean_mass_evolved(
            redshifts,
            times,
            time_first_SF=time_first_SF,
            COMPAS_Z=Z,
            COMPAS_delay_times=delays,
            COMPAS_weights=w,
            expected_local_rate=expected,
            Z_min_COMPAS=Z_grid[0],
            Z_max_COMPAS=Z_grid[-1],
        )

    m_coarse = _calibrate(0.01)
    m_fine = _calibrate(0.005)
    rel = abs(m_fine / m_coarse - 1.0)
    assert rel < 1e-2, (m_coarse, m_fine, rel)


# ─────────────────────────────────────────────────────────────────────
# Wanderman and Piran (2015) sGRB R(z), MNRAS 448, 3026 -- Eq. (9)
# ─────────────────────────────────────────────────────────────────────
def test_wanderman_piran_2015_peak_at_z_peak():
    """``wanderman_piran_2015_Rz`` returns R0 at z = z_peak (the
    piecewise-exponential is C^0 there, so both sides give the same
    value)."""
    from grb_rates import wanderman_piran_2015_Rz

    out = wanderman_piran_2015_Rz(np.array([0.9]))
    assert np.isclose(out["R_best"][0], 4.1)
    assert np.isclose(out["R_lo"][0], 2.2)
    assert np.isclose(out["R_hi"][0], 6.4)


# ─────────────────────────────────────────────────────────────────────
# kroupa_imf return-type contract (NumPy 1.25 deprecation guard)
# ─────────────────────────────────────────────────────────────────────
def test_kroupa_imf_size_one_array_returns_python_float_no_deprecation():
    """Length-1 ndarray input must return a Python ``float`` and must not
    raise a NumPy DeprecationWarning.

    Pre-fix the function returned ``float(result)`` directly on a length-1
    ndarray with ``ndim > 0``, which trips the NumPy 1.25 deprecation
    "Conversion of an array with ndim > 0 to a scalar is deprecated"
    and will become a hard error in a future NumPy release.  The fix
    routes through ``result.item()`` which is the documented scalar
    extractor for length-1 arrays.  This test pins both the return type
    and the warning-free contract so the regression cannot reappear.
    """
    import warnings

    from grb_rates import kroupa_imf

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        out = kroupa_imf(np.array([0.5]))

    assert isinstance(out, float), f"expected Python float, got {type(out).__name__}"


def test_kroupa_imf_scalar_input_returns_python_float():
    """Scalar input goes through the same ``np.atleast_1d`` -> size-1
    branch and must also return a Python ``float`` (not a 0-d ndarray
    or a NumPy scalar)."""
    from grb_rates import kroupa_imf

    out = kroupa_imf(0.5)
    assert isinstance(out, float), f"expected Python float, got {type(out).__name__}"


def test_kroupa_imf_array_input_returns_ndarray():
    """Multi-element input keeps the ndarray return path so downstream
    vectorised callers (Kroupa IMF integration, IMF-weighted means)
    are not silently demoted to per-element Python floats."""
    from grb_rates import kroupa_imf

    out = kroupa_imf(np.array([0.1, 1.0]))
    assert isinstance(out, np.ndarray), f"expected ndarray, got {type(out).__name__}"
    assert out.shape == (2,)


# ─────────────────────────────────────────────────────────────────────
# Chunking idempotency: compute_merger_rate is a chunked sum over FCI
# ─────────────────────────────────────────────────────────────────────
def _make_synth_population_for_rates(N, n_z, rng):
    redshifts = np.linspace(0.0, 5.0, n_z)
    times = np.linspace(13700.0, 100.0, n_z)
    sfr = np.full(n_z, 1e7)
    n_formed = sfr / 1.5e8
    Z_grid = 10.0 ** np.linspace(-4, np.log10(0.03), 8)
    COMPAS_Z = rng.choice(Z_grid, size=N)
    COMPAS_delays = 10.0 ** rng.uniform(np.log10(10.0), np.log10(13000.0), size=N)
    COMPAS_w = rng.uniform(0.1, 1.0, size=N)
    return (redshifts, times, sfr, n_formed, Z_grid, COMPAS_Z, COMPAS_delays, COMPAS_w)


@pytest.mark.requires_compas
def test_compute_merger_rate_chunking_is_idempotent():
    """``compute_merger_rate(..., n_chunk=N)`` is independent of ``N``.

    The wrapper sums FCI's per-chunk merger_rate.sum(axis=0) into a
    running ``(n_z,)`` accumulator, so changing ``n_chunk`` must only
    reshape the floating-point summation order (numpy float64 is not
    associative under permutation, so we tolerate ~1e-12).
    """
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        find_metallicity_distribution,
    )

    from grb_rates import compute_merger_rate

    rng = np.random.default_rng(2024)
    (redshifts, times, _, n_formed, Z_grid, Z, delays, w) = _make_synth_population_for_rates(
        N=2000, n_z=200, rng=rng
    )
    dPdlogZ, mets, p_draw = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(Z_grid[0]),
        max_logZ_COMPAS=np.log(Z_grid[-1]),
    )
    common = dict(
        redshifts=redshifts,
        times=times,
        time_first_SF=times.min() + 10.0,
        n_formed=n_formed,
        p_draw=p_draw,
        dPdlogZ=dPdlogZ,
        metallicities=mets,
        COMPAS_Z=Z,
        COMPAS_delay_times=delays,
        COMPAS_weights=w,
        smooth_sigma=0,
    )
    R_500 = compute_merger_rate(n_chunk=500, **common)
    R_5000 = compute_merger_rate(n_chunk=5_000, **common)
    R_50000 = compute_merger_rate(n_chunk=50_000, **common)
    np.testing.assert_allclose(R_500, R_5000, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(R_5000, R_50000, rtol=1e-12, atol=1e-12)


@pytest.mark.requires_compas
def test_compute_merger_rate_empty_population_returns_zeros():
    """The empty-population guard returns a zero (n_z,) array."""
    from grb_rates import compute_merger_rate

    redshifts = np.linspace(0.0, 5.0, 100)
    times = np.linspace(13700.0, 100.0, 100)
    n_formed = np.full(100, 1e7 / 1.5e8)
    # Empty sample: dPdlogZ / metallicities are unused, but compute_merger_rate
    # short-circuits on len(COMPAS_delay_times)==0 before consuming them.
    R = compute_merger_rate(
        redshifts,
        times,
        time_first_SF=times.min() + 10.0,
        n_formed=n_formed,
        p_draw=0.1,
        dPdlogZ=np.zeros((100, 1)),
        metallicities=np.array([1e-2]),
        COMPAS_Z=np.array([]),
        COMPAS_delay_times=np.array([]),
        COMPAS_weights=np.array([]),
        smooth_sigma=30,
    )
    assert R.shape == redshifts.shape
    assert np.all(R == 0)


@pytest.mark.requires_compas
def test_compute_merger_rate_smoothing_intact():
    """smooth_sigma applies a Gaussian kernel to the unsmoothed rate."""
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        find_metallicity_distribution,
    )
    from scipy.ndimage import gaussian_filter1d

    from grb_rates import compute_merger_rate

    rng = np.random.default_rng(2025)
    (redshifts, times, _, n_formed, Z_grid, Z, delays, w) = _make_synth_population_for_rates(
        N=300, n_z=200, rng=rng
    )
    dPdlogZ, mets, p_draw = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(Z_grid[0]),
        max_logZ_COMPAS=np.log(Z_grid[-1]),
    )

    common = dict(
        redshifts=redshifts,
        times=times,
        time_first_SF=times.min() + 10.0,
        n_formed=n_formed,
        p_draw=p_draw,
        dPdlogZ=dPdlogZ,
        metallicities=mets,
        COMPAS_Z=Z,
        COMPAS_delay_times=delays,
        COMPAS_weights=w,
    )

    R_unsmoothed = compute_merger_rate(smooth_sigma=0, **common)
    R_smoothed = compute_merger_rate(smooth_sigma=30, **common)
    np.testing.assert_allclose(
        R_smoothed,
        gaussian_filter1d(R_unsmoothed, sigma=30),
        rtol=1e-12,
        atol=1e-12,
    )


# ─────────────────────────────────────────────────────────────────────
# COMPAS-vs-grb_rates: numerical agreement on the same HDF5 at z = 0.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_compute_merger_rate_matches_compas_exactly(bns_a_path):
    """``compute_merger_rate`` is now a chunked accumulator over FCI's
    own ``find_formation_and_merger_rates``, so passing the full sample
    in one chunk must reproduce ``merger_rate.sum(axis=0)`` from FCI to
    floating-point precision.
    """
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )
    calculate_redshift_related_params = fci.calculate_redshift_related_params
    find_sfr = fci.find_sfr
    find_metallicity_distribution = fci.find_metallicity_distribution
    find_formation_and_merger_rates = fci.find_formation_and_merger_rates

    from astropy.cosmology import Planck15

    from grb_io import load_bns
    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG100,
        compute_merger_rate,
    )

    data = load_bns(path=bns_a_path)
    Z = data["metallicity"]
    delays = data["delay_time"]
    w = data["weights"]

    redshifts, _, times, time_first_SF, _, _ = calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.05, cosmology=Planck15
    )
    sfr = find_sfr(redshifts, **SFR_PARAMS_LEVINA26_TNG100)

    dPdlogZ, metallicities, p_draw = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(np.min(Z)),
        max_logZ_COMPAS=np.log(np.max(Z)),
        **MSSFR_PARAMS_LEVINA26_TNG100,
    )

    # Use a unit MEAN_MASS_EVOLVED (n_formed = sfr) so this is a pure
    # apples-to-apples comparison of the FCI loop body vs the chunked
    # accumulator; the calibration step is exercised separately.
    n_formed = sfr

    R_ours = compute_merger_rate(
        redshifts,
        times,
        time_first_SF,
        n_formed,
        p_draw,
        dPdlogZ,
        metallicities,
        Z,
        delays,
        w,
        smooth_sigma=0,
        n_chunk=len(Z),
    )

    _, merger_rate_compas = find_formation_and_merger_rates(
        n_binaries=len(Z),
        redshifts=redshifts,
        times=times,
        time_first_SF=time_first_SF,
        n_formed=n_formed,
        dPdlogZ=dPdlogZ,
        metallicities=metallicities,
        p_draw_metallicity=p_draw,
        COMPAS_metallicites=Z,
        COMPAS_delay_times=delays,
        COMPAS_weights=w,
    )
    R_compas = merger_rate_compas.sum(axis=0)

    np.testing.assert_allclose(
        R_ours,
        R_compas,
        rtol=1e-12,
        atol=1e-12,
        err_msg="compute_merger_rate must equal sum-axis-0 of FCI exactly.",
    )


@pytest.mark.requires_compas
def test_per_system_rate_weights_matches_fci_formation_column_at_grid_aligned_z_target():
    """``per_system_rate_weights[i]`` equals FCI ``formation_rate[i, j_target]``
    when ``z_form`` lands on the grid (delays = 0, ``z_target`` on grid).

    Pins the algorithm-equivalence claim at ``grb_rates.per_system_rate_weights``
    (the ``np.digitize(Z, metallicities)`` column lookup mirrors
    ``find_formation_and_merger_rates``) to floating-point precision.
    With every binary's delay set to zero and ``z_target`` chosen on the
    redshift grid, ``z_form_i = z_target`` lies exactly on the grid for
    every i, so the linear-z interpolation in ``per_system_rate_weights``
    degenerates to the same expression FCI evaluates per row:

        n_formed[j_target] * dPdlogZ[j_target, digitize(Z[i])] / p_draw * w[i]
    """
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )
    from astropy.cosmology import Planck15

    from grb_rates import per_system_rate_weights

    redshifts, _, times, time_first_SF, _, _ = fci.calculate_redshift_related_params(
        max_redshift=2.0,
        redshift_step=0.01,
        cosmology=Planck15,
    )
    sfr = fci.find_sfr(redshifts)

    rng = np.random.default_rng(0)
    Z_lo, Z_hi = 1e-4, 0.03
    Z = np.exp(rng.uniform(np.log(Z_lo), np.log(Z_hi), size=24))
    delays = np.zeros_like(Z)
    w = np.full_like(Z, 1.0)

    dPdlogZ, mets, p_draw = fci.find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(Z_lo),
        max_logZ_COMPAS=np.log(Z_hi),
    )
    # n_formed scale cancels: identical inputs feed both pipelines.
    n_formed = sfr / 1.0e7

    j_target = 50
    z_target = float(redshifts[j_target])

    proj = per_system_rate_weights(
        z_target,
        redshifts,
        times,
        time_first_SF,
        n_formed,
        p_draw,
        dPdlogZ,
        mets,
        Z,
        delays,
        w,
    )
    formation_rate_fci, _ = fci.find_formation_and_merger_rates(
        n_binaries=len(Z),
        redshifts=redshifts,
        times=times,
        time_first_SF=time_first_SF,
        n_formed=n_formed,
        dPdlogZ=dPdlogZ,
        metallicities=mets,
        p_draw_metallicity=p_draw,
        COMPAS_metallicites=Z,
        COMPAS_delay_times=delays,
        COMPAS_weights=w,
    )
    np.testing.assert_allclose(
        proj,
        formation_rate_fci[:, j_target],
        rtol=1e-12,
        atol=0.0,
        err_msg=(
            "per_system_rate_weights drifted from FCI's formation_rate "
            "column at a grid-aligned z_target; the np.digitize column "
            "lookup in grb_rates.per_system_rate_weights is no longer "
            "in sync with FastCosmicIntegration."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# detected_rate: GW selection effects via FCI
# ─────────────────────────────────────────────────────────────────────
def _make_synth_population_for_detection(N, rng):
    """Build a small BHNS-like sample (heavier chirp mass than BNS so
    O3 detection probability is non-trivial across the FCI default
    detection horizon z_det = 1.0)."""
    m1 = rng.uniform(5.0, 12.0, N)
    m2 = rng.uniform(1.0, 1.6, N)
    Z = rng.choice(10.0 ** np.linspace(-4, np.log10(0.03), 8), N)
    delays = 10.0 ** rng.uniform(np.log10(10.0), np.log10(13000.0), N)
    weights = rng.uniform(0.5, 1.5, N)
    return m1, m2, Z, delays, weights


@pytest.mark.requires_compas
def test_detected_rate_non_negative_and_empty_guard():
    """detected_rate returns a non-negative array of the right shape and
    a zero array on an empty sample."""
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
        find_metallicity_distribution,
        find_sfr,
    )

    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG100,
        detected_rate,
    )

    redshifts, n_z_det, times, time_first_SF, distances, _ = calculate_redshift_related_params(
        max_redshift=10.0,
        max_redshift_detection=2.0,
        redshift_step=0.05,
        cosmology=Planck15,
    )
    sfr = find_sfr(redshifts, **SFR_PARAMS_LEVINA26_TNG100)
    dPdlogZ, mets, p_draw = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(1e-4),
        max_logZ_COMPAS=np.log(0.03),
        **MSSFR_PARAMS_LEVINA26_TNG100,
    )
    n_formed = sfr / 1e8

    rng = np.random.default_rng(2026)
    m1, m2, Z, delays, w = _make_synth_population_for_detection(N=200, rng=rng)

    R_det = detected_rate(
        redshifts,
        times,
        time_first_SF,
        n_formed,
        p_draw,
        dPdlogZ,
        mets,
        m1,
        m2,
        Z,
        delays,
        w,
        distances,
        n_z_det,
        sensitivity="O3",
        snr_threshold=8.0,
    )
    assert R_det.shape == (n_z_det,)
    assert np.all(R_det >= 0.0), R_det.min()
    # At least one redshift bin must carry positive detected rate for a
    # 200-binary BHNS-like sample at O3 (otherwise the noise PSD or
    # SNR-grid lookup is broken).
    assert R_det.max() > 0.0, R_det

    R_empty = detected_rate(
        redshifts,
        times,
        time_first_SF,
        n_formed,
        p_draw,
        dPdlogZ,
        mets,
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        distances,
        n_z_det,
    )
    assert R_empty.shape == (n_z_det,)
    assert np.all(R_empty == 0.0)


@pytest.mark.requires_compas
def test_detected_rate_decreases_with_higher_snr_threshold():
    """At fixed sensitivity, raising the SNR threshold cannot increase the
    detected rate at any redshift (FCI ``find_detection_probability`` is
    monotonically non-increasing in ``snr_threshold``)."""
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
        find_metallicity_distribution,
        find_sfr,
    )

    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG100,
        detected_rate,
    )

    redshifts, n_z_det, times, time_first_SF, distances, _ = calculate_redshift_related_params(
        max_redshift=10.0,
        max_redshift_detection=2.0,
        redshift_step=0.05,
        cosmology=Planck15,
    )
    sfr = find_sfr(redshifts, **SFR_PARAMS_LEVINA26_TNG100)
    dPdlogZ, mets, p_draw = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(1e-4),
        max_logZ_COMPAS=np.log(0.03),
        **MSSFR_PARAMS_LEVINA26_TNG100,
    )
    n_formed = sfr / 1e8
    rng = np.random.default_rng(2026)
    m1, m2, Z, delays, w = _make_synth_population_for_detection(N=200, rng=rng)

    R_8 = detected_rate(
        redshifts,
        times,
        time_first_SF,
        n_formed,
        p_draw,
        dPdlogZ,
        mets,
        m1,
        m2,
        Z,
        delays,
        w,
        distances,
        n_z_det,
        sensitivity="O3",
        snr_threshold=8.0,
    )
    R_12 = detected_rate(
        redshifts,
        times,
        time_first_SF,
        n_formed,
        p_draw,
        dPdlogZ,
        mets,
        m1,
        m2,
        Z,
        delays,
        w,
        distances,
        n_z_det,
        sensitivity="O3",
        snr_threshold=12.0,
    )
    # Elementwise non-increasing within float tolerance.
    assert np.all(R_12 <= R_8 + 1e-12), (R_8, R_12)
    # Total detected rate must strictly decrease for a sample with
    # sub-threshold systems present.
    assert R_12.sum() < R_8.sum(), (R_8.sum(), R_12.sum())


# ─────────────────────────────────────────────────────────────────────
# Kish (1965) effective sample size; Section 5 small-N diagnostic.
# ─────────────────────────────────────────────────────────────────────
def test_kish_effective_sample_size_uniform_weights_equals_N():
    """Uniform weights yield n_eff = N exactly (Kish 1965 Sec. 1.6)."""
    from grb_rates import kish_effective_sample_size

    for N in (1, 10, 100, 10_000):
        w = np.ones(N)
        assert kish_effective_sample_size(w) == pytest.approx(float(N), rel=1e-12)


def test_kish_effective_sample_size_single_dominant_weight_approaches_one():
    """One weight much larger than the rest collapses n_eff toward 1."""
    from grb_rates import kish_effective_sample_size

    w = np.concatenate(([1e9], np.ones(999)))
    n_eff = kish_effective_sample_size(w)
    assert n_eff == pytest.approx(1.0, abs=1e-3), n_eff


def test_kish_effective_sample_size_bounded_by_N():
    """n_eff <= N for any non-negative weight distribution."""
    from grb_rates import kish_effective_sample_size

    rng = np.random.default_rng(0)
    for _ in range(20):
        N = int(rng.integers(2, 5000))
        w = rng.uniform(0.0, 10.0, N)
        n_eff = kish_effective_sample_size(w)
        assert 0.0 <= n_eff <= N + 1e-9, (N, n_eff)


def test_kish_effective_sample_size_two_equal_weights_returns_two():
    """A textbook case: two equal weights give n_eff = 2."""
    from grb_rates import kish_effective_sample_size

    assert kish_effective_sample_size([3.0, 3.0]) == pytest.approx(2.0)


def test_kish_effective_sample_size_zero_array_returns_zero():
    """All-zero weight array degenerates to n_eff = 0 (handled without ZeroDiv)."""
    from grb_rates import kish_effective_sample_size

    assert kish_effective_sample_size(np.zeros(10)) == 0.0
    assert kish_effective_sample_size([]) == 0.0


def test_kish_effective_sample_size_rejects_negative_weights():
    """Negative weights are not a valid importance-sample input."""
    from grb_rates import kish_effective_sample_size

    with pytest.raises(ValueError, match="non-negative"):
        kish_effective_sample_size([1.0, -0.1, 0.5])


def test_weighted_frac_err_uniform_weights_reduces_to_binomial_count():
    """With unit weights, f = N_class/N and sigma = sqrt(N_class)/N."""
    from grb_rates import weighted_frac_err

    w = np.ones(100)
    mask = np.zeros(100, dtype=bool)
    mask[:25] = True
    f, sigma = weighted_frac_err(w, mask)
    assert f == pytest.approx(0.25)
    assert sigma == pytest.approx(np.sqrt(25.0) / 100.0)


def test_weighted_frac_err_matches_sqrt_sum_w2_over_total():
    from grb_rates import weighted_frac_err

    rng = np.random.default_rng(7)
    w = rng.uniform(0.1, 10.0, 500)
    mask = rng.random(500) < 0.3
    f, sigma = weighted_frac_err(w, mask)
    assert f == pytest.approx(w[mask].sum() / w.sum())
    assert sigma == pytest.approx(np.sqrt(np.sum(w[mask] ** 2)) / w.sum())


def test_weighted_frac_err_explicit_total_and_empty_population():
    from grb_rates import weighted_frac_err

    w = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, True])
    f, sigma = weighted_frac_err(w, mask, w_total=10.0)
    assert f == pytest.approx(4.0 / 10.0)
    assert sigma == pytest.approx(np.sqrt(1.0 + 9.0) / 10.0)

    f0, s0 = weighted_frac_err(np.zeros(5), np.ones(5, dtype=bool))
    assert np.isnan(f0) and np.isnan(s0)


# ─────────────────────────────────────────────────────────────────────
# BNS jet breakout efficiency (Pais et al. 2025)
# ─────────────────────────────────────────────────────────────────────
def test_apply_bns_jet_breakout_multiplier():
    """apply_bns_jet_breakout scales the rate by f_breakout."""
    from grb_rates import apply_bns_jet_breakout

    assert apply_bns_jet_breakout(100.0, 0.6) == pytest.approx(60.0)
    np.testing.assert_allclose(
        apply_bns_jet_breakout(np.array([10.0, 20.0]), 0.5), np.array([5.0, 10.0])
    )


def test_breakout_fraction_bns_band_ordering_and_bounds():
    """Each class returns lo <= med <= hi within [0, 1]."""
    from grb_rates import breakout_fraction_bns

    rng = np.random.default_rng(1)
    n = 400
    class_masks = {
        "sbGRB + blue KN": np.zeros(n, dtype=bool),
        "lbGRB + red KN (disk)": np.zeros(n, dtype=bool),
    }
    class_masks["sbGRB + blue KN"][: n // 2] = True
    class_masks["lbGRB + red KN (disk)"][n // 2 :] = True
    M_disk = rng.uniform(1e-3, 0.1, n)
    M_ej = rng.uniform(1e-4, 5e-3, n)
    weights = rng.uniform(0.1, 1.0, n)

    out = breakout_fraction_bns(class_masks, M_disk, M_ej, weights, n_draws=64)
    for label, band in out.items():
        assert band["lo"] <= band["med"] <= band["hi"], label
        assert 0.0 <= band["lo"] <= 1.0 and 0.0 <= band["hi"] <= 1.0, label


def test_breakout_fraction_bns_empty_class_is_nan():
    """A class with zero weighted population yields NaN band entries."""
    from grb_rates import breakout_fraction_bns

    n = 50
    class_masks = {
        "Faint lbGRB": np.zeros(n, dtype=bool),  # empty
        "lbGRB + red KN (disk)": np.ones(n, dtype=bool),
    }
    M_disk = np.full(n, 1e-2)
    M_ej = np.full(n, 1e-3)
    weights = np.ones(n)

    out = breakout_fraction_bns(class_masks, M_disk, M_ej, weights, n_draws=32)
    assert np.isnan(out["Faint lbGRB"]["med"])
    assert np.isnan(out["Faint lbGRB"]["lo"])
    assert np.isnan(out["Faint lbGRB"]["hi"])


def test_breakout_fraction_bns_remnant_conditioning_suppresses_hmns():
    """HMNS-class breakout is suppressed relative to a prompt-collapse class.

    The long HMNS launch delay enters the (t_d/t_e)^2 term of Pais Eq. (4)
    and lowers the breakout fraction relative to a prompt-collapse class
    with identical per-system disk and ejecta masses.
    """
    from grb_rates import breakout_fraction_bns

    n = 600
    class_masks = {
        "sbGRB + blue KN": np.zeros(n, dtype=bool),
        "lbGRB + red KN (disk)": np.zeros(n, dtype=bool),
    }
    class_masks["sbGRB + blue KN"][: n // 2] = True
    class_masks["lbGRB + red KN (disk)"][n // 2 :] = True
    # Identical physical inputs across the two classes.
    M_disk = np.full(n, 1e-2)
    M_ej = np.full(n, 1e-3)
    weights = np.ones(n)

    out = breakout_fraction_bns(class_masks, M_disk, M_ej, weights, n_draws=128)
    assert out["sbGRB + blue KN"]["med"] <= out["lbGRB + red KN (disk)"]["med"]


def test_breakout_fraction_bns_mixed_population_in_literature_band():
    """A realistic prompt+HMNS mix gives an aggregate f_breakout ~ 0.5-0.7.

    Loose sanity check against the De Santis (2026) / Kunnumkai (2026)
    population jet-fraction band.  Prompt-collapse remnants (clean funnel)
    break out efficiently while long-delay HMNS remnants are strongly
    suppressed, so a population split between the two lands near the
    observationally favoured fraction.  The absolute value is only
    accurate to a factor of a few given the Pais single-simulation
    calibration, so the accepted window is deliberately wide.
    """
    from grb_rates import breakout_fraction_bns

    rng = np.random.default_rng(3)
    n = 2000
    half = n // 2
    class_masks = {
        "lbGRB + red KN (HMNS)": np.zeros(n, dtype=bool),  # HMNS engine, suppressed
        "lbGRB + red KN (disk)": np.zeros(n, dtype=bool),  # prompt collapse
    }
    class_masks["lbGRB + red KN (HMNS)"][:half] = True
    class_masks["lbGRB + red KN (disk)"][half:] = True
    # Disk masses bracketing the GW170817-like ~1.5e-3 Msun anchor and
    # total ejecta up to the AT2017gfo ~0.05 Msun obstruction.
    M_disk = rng.uniform(5e-4, 8e-3, n)
    M_ej = rng.uniform(1e-3, 5e-2, n)
    weights = rng.uniform(0.1, 1.0, n)

    out = breakout_fraction_bns(class_masks, M_disk, M_ej, weights, n_draws=128)

    # Prompt collapse breaks out far more readily than the HMNS engine.
    assert out["lbGRB + red KN (disk)"]["med"] > out["lbGRB + red KN (HMNS)"]["med"]

    # Population-weighted aggregate over the two classes.
    agg = 0.0
    for label, mask in class_masks.items():
        w_frac = float(weights[mask].sum()) / float(weights.sum())
        agg += w_frac * out[label]["med"]
    assert 0.3 < agg < 0.8
