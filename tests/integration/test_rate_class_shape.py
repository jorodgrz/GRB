"""Validation tests for the per-class BNS rate curve shape.

Section 7 of ``grb_main.ipynb`` produces ``Plots/rate_bns_by_class.{pdf,png}``,
where the cyan ``sbGRB + blue KN`` channel exhibits a peak near z ~ 1.5, a
~10x dip near z ~ 4-5, and a recovery near z ~ 7.  The author's caption
attributes this to low-Z formation-channel dominance for the lightest BNS
systems (M_tot < 1.2 * M_TOV); the existing additivity test
``test_per_class_rates_partition_total_bns_2024`` is silent about whether
the shape is a numerical artifact.

The two tests in this file close that gap on the actual production curve:

(1) ``test_sbGRB_rate_dip_redshift_binning_invariant`` recomputes
    ``R_sbGRB(z)`` at production binning (dz = 0.01) and at a 2x refined
    grid (dz = 0.005), with a matched physical Gaussian kernel
    (``smooth_sigma * dz = 0.30``).  Asserts the dip and the two peaks
    land at the same redshifts (within ~5 production bins) and that the
    contrast ratios ``R_peak1 / R_dip`` and ``R_peak2 / R_dip`` agree
    across binnings to within 15%.  Tested as a contrast invariant
    rather than absolute amplitude because
    ``calibrate_mean_mass_evolved`` uses ``smooth_sigma=30`` in BIN
    units, so its boundary-reflective Gaussian at z=0 mixes in a
    different slice of R(z) at different ``redshift_step``, drifting
    the calibration constant and every downstream amplitude with it.
    Shape (locations, contrast ratios) is what the "is the dip a
    binning artifact" question actually asks about, and shape is
    invariant under that calibration drift.

(2) ``test_sbGRB_rate_dip_n_eff_above_threshold`` evaluates
    ``per_system_rate_weights`` at the dip redshift and asserts the
    weighted-sample ``N_eff = (sum w_i)^2 / sum w_i^2`` is comfortably
    above the few-binary regime that would make the curve sensitive to
    individual STROOPWAFEL outliers.

Both tests are gated on ``Data/COMPASCompactOutput_BNS_A.h5`` (via the
``bns_a_path`` fixture in ``conftest.py``) and on the upstream
``compas_python_utils`` package (via ``pytest.importorskip``); they skip
cleanly on machines lacking either.  Each test runs at least one full
Madau and Dickinson SFR convolution at production resolution and is
therefore marked ``slow``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _build_setup(bns_a_path):
    """Mirror Section 7 of grb_main.ipynb up to the ``compute_merger_rate``
    call: load BNS A, apply the Alsing+ 2018 NS-mass remap, build the
    sbGRB + blue KN mask, and prep the cosmic-integration plumbing
    (``compas_python_utils.FastCosmicIntegration`` is required here)."""
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )

    from astropy.cosmology import Planck15

    # Project cosmology pin: Planck 2015 (Ade et al. 2016, A&A 594, A13;
    # matches COMPAS FastCosmicIntegration TNG-consistent default).
    assert abs(Planck15.H0.value - 67.74) < 0.01

    from grb_classify import classify_bns_2024
    from grb_io import load_bns, read_expected_local_rate
    from grb_physics import remap_ns_masses_double_gaussian
    from grb_rates import calibrate_mean_mass_evolved

    bns = load_bns(path=bns_a_path)
    # Alsing, Silva and Berti (2018) double-Gaussian remap; rng seed 42
    # matches grb_main.ipynb so the sbGRB mask reproduces the cyan curve
    # the production plot actually shows.
    m1, m2 = remap_ns_masses_double_gaussian(
        bns["m1"].copy(), bns["m2"].copy(), weights=bns["weights"], rng=np.random.default_rng(42)
    )

    Z = bns["metallicity"]
    delays = bns["delay_time"]
    w = bns["weights"]
    Z_grid = np.unique(Z)

    cls = classify_bns_2024(m1, m2)
    mask = cls["sbGRB + blue KN"]
    assert mask.any(), "sbGRB + blue KN class is empty on BNS A; setup is broken"

    return SimpleNamespace(
        fci=fci,
        Planck15=Planck15,
        Z=Z,
        delays=delays,
        w=w,
        Z_grid=Z_grid,
        mask=mask,
        expected_local_rate=read_expected_local_rate(bns_a_path),
        calibrate_mean_mass_evolved=calibrate_mean_mass_evolved,
    )


def _cosmic_grid(setup, redshift_step, max_redshift=10.0):
    """Run the FastCosmicIntegration plumbing at a chosen redshift step
    using Levina+ 2026 TNG100-1 parameters.
    """
    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG100,
    )

    redshifts, _, times, time_first_SF, _, _ = setup.fci.calculate_redshift_related_params(
        max_redshift=max_redshift, redshift_step=redshift_step, cosmology=setup.Planck15
    )
    sfr = setup.fci.find_sfr(redshifts, **SFR_PARAMS_LEVINA26_TNG100)
    dPdlogZ, mets, p_draw = setup.fci.find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(setup.Z.min()),
        max_logZ_COMPAS=np.log(setup.Z.max()),
        **MSSFR_PARAMS_LEVINA26_TNG100,
    )
    mean_mass = setup.calibrate_mean_mass_evolved(
        redshifts,
        times,
        time_first_SF,
        setup.Z,
        setup.delays,
        setup.w,
        setup.expected_local_rate,
        Z_min_COMPAS=setup.Z.min(),
        Z_max_COMPAS=setup.Z.max(),
    )
    n_formed = sfr / mean_mass
    return SimpleNamespace(
        redshifts=redshifts,
        times=times,
        time_first_SF=time_first_SF,
        n_formed=n_formed,
        p_draw=p_draw,
        dPdlogZ=dPdlogZ,
        metallicities=mets,
    )


def _R_sbgrb(setup, grid, smooth_sigma):
    """Per-class compute_merger_rate for the sbGRB + blue KN subset."""
    from grb_rates import compute_merger_rate

    return compute_merger_rate(
        grid.redshifts,
        grid.times,
        grid.time_first_SF,
        grid.n_formed,
        grid.p_draw,
        grid.dPdlogZ,
        grid.metallicities,
        setup.Z[setup.mask],
        setup.delays[setup.mask],
        setup.w[setup.mask],
        smooth_sigma=smooth_sigma,
    )


def _find_extremum(redshifts, R, z_lo, z_hi, kind):
    """Return (z, R, idx) of the extremum within [z_lo, z_hi]."""
    sl = np.where((redshifts >= z_lo) & (redshifts <= z_hi))[0]
    assert sl.size > 0, f"no redshifts in window ({z_lo}, {z_hi})"
    if kind == "min":
        idx_in_sl = int(np.argmin(R[sl]))
    elif kind == "max":
        idx_in_sl = int(np.argmax(R[sl]))
    else:
        raise ValueError(f"kind must be 'min' or 'max'; got {kind!r}")
    j = int(sl[idx_in_sl])
    return float(redshifts[j]), float(R[j]), j


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_calibrated_rate_matches_expected_local_rate_at_z0(bns_a_path):
    """Round-trip: running the calibrated forward pass with the SAME
    Neijssel MSSFR that the upstream Broekgaarden+ 2021 pipeline used
    to compute ``weights_intrinsic/w_000`` reproduces
    ``expected_local_rate`` to within numerical precision at every
    ``redshift_step``.  This is the contract of
    ``calibrate_mean_mass_evolved`` (back-derives MEAN_MASS_EVOLVED
    from the Neijssel z=0 anchor).

    The science-path forward pass uses Levina TNG100-1 and so produces
    a different absolute R(z=0); that is a physics outcome, not a
    calibration drift, and is exercised in
    ``test_section_07_*`` integration tests instead.
    """
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
        find_metallicity_distribution,
        find_sfr,
    )

    from grb_rates import calibrate_mean_mass_evolved, compute_merger_rate

    setup = _build_setup(bns_a_path)
    for dz in (0.01, 0.005):
        redshifts, _, times, time_first_SF, _, _ = calculate_redshift_related_params(
            max_redshift=10.0, redshift_step=dz, cosmology=Planck15
        )
        # Calibrate with the Neijssel-anchored helper.
        mean_mass = calibrate_mean_mass_evolved(
            redshifts,
            times,
            time_first_SF,
            setup.Z,
            setup.delays,
            setup.w,
            setup.expected_local_rate,
            Z_min_COMPAS=setup.Z.min(),
            Z_max_COMPAS=setup.Z.max(),
        )
        # Run forward with the same Neijssel MSSFR (round-trip).
        sfr = find_sfr(redshifts)
        dPdlogZ, mets, p_draw = find_metallicity_distribution(
            redshifts,
            min_logZ_COMPAS=np.log(setup.Z.min()),
            max_logZ_COMPAS=np.log(setup.Z.max()),
        )
        R_full = compute_merger_rate(
            redshifts,
            times,
            time_first_SF,
            sfr / mean_mass,
            p_draw,
            dPdlogZ,
            mets,
            setup.Z,
            setup.delays,
            setup.w,
            smooth_sigma=0,
        )
        rel = abs(R_full[0] / setup.expected_local_rate - 1.0)
        assert rel < 1e-3, (dz, R_full[0], setup.expected_local_rate, rel)


@pytest.mark.xfail(
    reason="sbGRB+blueKN bimodality is a Neijssel-fiducial feature; Levina TNG100-1 "
    "(omega_0 = 1.15 vs Neijssel 0.39) smears the metallicity weighting and "
    "the dip-and-recovery flattens.  Test is preserved as documentation; "
    "rerun on the Neijssel sweep variant when one is added.",
    strict=False,
)
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_sbGRB_rate_dip_redshift_binning_invariant(bns_a_path):
    """The cyan sbGRB + blue KN dip-and-recovery structure must persist
    under a 2x redshift-grid refinement.

    Tested in two complementary ways:

    1. CONTRAST-RATIO invariance (``R_peak / R_dip``).  The shape-vs-
       scale separation is what "is the dip a binning artifact"
       actually asks about: a real feature has a redshift-stable depth
       *relative* to the surrounding peaks, regardless of any overall
       scaling.  Empirically these ratios are stable to ~0.01% across
       dz = 0.01, 0.005, 0.0025 with the matched physical kernel below.

    2. ABSOLUTE-AMPLITUDE invariance (``R_dip``, ``R_peak2``).  After
       the ``smooth_sigma=0`` calibration fix in
       ``calibrate_mean_mass_evolved`` (sharp z=0 anchor matching
       Broekgaarden+ 2021 w_000), MEAN_MASS_EVOLVED no longer drifts
       with redshift_step, so the per-class rate amplitudes are also
       binning-stable when smoothing is applied with a matched physical
       kernel.  Pre-fix this assertion failed at +38.9%; the 10%
       threshold below leaves headroom for finite-dz residuals.  See
       ``test_calibrated_rate_matches_expected_local_rate_at_z0`` for
       the calibration-anchor regression that complements this one.

    Tolerance choices:
      - At each grid: ``R_peak1 / R_dip > 5x`` and ``R_peak2 / R_dip > 2x``
        (the user observes ~10x at the first peak; 5x is a conservative
        flattening floor).
      - z_dip stable to within 5 production bins (Delta z < 0.05);
        z_peak1 / z_peak2 stable within 10 bins (broader features).
      - p1/d and p2/d contrast ratios stable to within 15%, matching the
        shape tolerance used in
        ``test_compute_merger_rate_matches_compas_shape``.
      - ``R_dip`` and ``R_peak2`` absolute amplitudes stable to within
        10% (post-calibration-fix only).

    Both grids use a matched physical Gaussian kernel
    (``smooth_sigma * redshift_step = 0.30``): ``smooth_sigma=30`` at
    ``dz=0.01`` (the production setting) and ``smooth_sigma=60`` at
    ``dz=0.005``.  This is the same low-pass the cyan plot in
    ``Plots/rate_bns_by_class`` actually applies.
    """
    setup = _build_setup(bns_a_path)

    grid_coarse = _cosmic_grid(setup, redshift_step=0.01)
    R_coarse = _R_sbgrb(setup, grid_coarse, smooth_sigma=30)

    z_p1_c, R_p1_c, _ = _find_extremum(grid_coarse.redshifts, R_coarse, 0.5, 2.5, "max")
    z_d_c, R_d_c, _ = _find_extremum(grid_coarse.redshifts, R_coarse, 3.0, 5.5, "min")
    z_p2_c, R_p2_c, _ = _find_extremum(grid_coarse.redshifts, R_coarse, 5.5, 9.0, "max")

    contrast_p1_c = R_p1_c / R_d_c
    contrast_p2_c = R_p2_c / R_d_c

    # Smoke check: cyan curve at production binning actually has a dip
    # with R_peak1 / R_dip well above ~10x; 5x is a conservative
    # flattening floor that would still trip on a serious regression.
    assert contrast_p1_c > 5.0, (
        f"production cyan curve at dz=0.01 no longer has the expected "
        f"first-peak / dip contrast; got {contrast_p1_c:.2f}x at "
        f"z_p1={z_p1_c:.2f}, z_d={z_d_c:.2f}."
    )
    assert contrast_p2_c > 2.0, (
        f"production cyan curve at dz=0.01 no longer has the expected "
        f"second-peak / dip contrast; got {contrast_p2_c:.2f}x at "
        f"z_p2={z_p2_c:.2f}, z_d={z_d_c:.2f}."
    )

    grid_fine = _cosmic_grid(setup, redshift_step=0.005)
    # smooth_sigma=60 at dz=0.005 keeps the physical Gaussian kernel
    # width identical to smooth_sigma=30 at dz=0.01 (Delta_z = 0.30).
    R_fine = _R_sbgrb(setup, grid_fine, smooth_sigma=60)

    z_p1_f, R_p1_f, _ = _find_extremum(grid_fine.redshifts, R_fine, 0.5, 2.5, "max")
    z_d_f, R_d_f, _ = _find_extremum(grid_fine.redshifts, R_fine, 3.0, 5.5, "min")
    z_p2_f, R_p2_f, _ = _find_extremum(grid_fine.redshifts, R_fine, 5.5, 9.0, "max")

    contrast_p1_f = R_p1_f / R_d_f
    contrast_p2_f = R_p2_f / R_d_f

    print(
        f"\n[sbGRB binning]"
        f"\n  coarse dz=0.01:  z_p1={z_p1_c:.3f} z_d={z_d_c:.3f} "
        f"z_p2={z_p2_c:.3f}  R(p1)/R(d)={contrast_p1_c:.3f}x  "
        f"R(p2)/R(d)={contrast_p2_c:.3f}x  R_d={R_d_c:.3e}"
        f"\n  fine   dz=0.005: z_p1={z_p1_f:.3f} z_d={z_d_f:.3f} "
        f"z_p2={z_p2_f:.3f}  R(p1)/R(d)={contrast_p1_f:.3f}x  "
        f"R(p2)/R(d)={contrast_p2_f:.3f}x  R_d={R_d_f:.3e}"
    )

    # Persistence at the fine grid: structure must survive refinement.
    assert contrast_p1_f > 5.0, (
        f"sbGRB dip dissolved at dz=0.005: R_p1/R_d = {contrast_p1_f:.2f}x "
        f"(production threshold > 5x).  The cyan double-peak is binning-"
        f"sensitive in a way the production curve does not capture."
    )
    assert contrast_p2_f > 2.0, (
        f"sbGRB recovery dissolved at dz=0.005: R_p2/R_d = "
        f"{contrast_p2_f:.2f}x (production threshold > 2x)."
    )

    # Location stability across binnings.
    assert abs(z_p1_f - z_p1_c) < 0.10, (
        f"sbGRB first-peak shifted by {z_p1_f - z_p1_c:+.4f} between "
        f"dz=0.01 (z_p1={z_p1_c:.3f}) and dz=0.005 (z_p1={z_p1_f:.3f})."
    )
    assert abs(z_d_f - z_d_c) < 0.05, (
        f"sbGRB dip shifted by {z_d_f - z_d_c:+.4f} between dz=0.01 "
        f"(z_d={z_d_c:.3f}) and dz=0.005 (z_d={z_d_f:.3f}); a stable "
        f"physical feature should land within 5 production bins."
    )
    assert abs(z_p2_f - z_p2_c) < 0.10, (
        f"sbGRB second-peak shifted by {z_p2_f - z_p2_c:+.4f} between "
        f"dz=0.01 (z_p2={z_p2_c:.3f}) and dz=0.005 (z_p2={z_p2_f:.3f}); "
        f"> 0.10 hints at an interpolation artifact in compute_merger_rate."
    )

    # Headline: contrast-ratio invariance under refinement.  If the dip
    # were a binning artifact, its depth relative to the surrounding
    # peaks would change when redshift_step halves.  Empirically these
    # ratios are stable to ~0.01% across the dz sweep; the 15% threshold
    # below leaves headroom for downstream changes that legitimately
    # nudge the convolution shape.
    delta_p1 = abs(contrast_p1_f - contrast_p1_c) / contrast_p1_c
    delta_p2 = abs(contrast_p2_f - contrast_p2_c) / contrast_p2_c
    assert delta_p1 < 0.15, (
        f"sbGRB R_peak1 / R_dip changed by {delta_p1 * 100:+.1f}% "
        f"({contrast_p1_c:.3f} -> {contrast_p1_f:.3f}) between dz=0.01 "
        f"and dz=0.005; > 15% would flag the dip's depth as a binning "
        f"artifact relative to the surrounding peaks."
    )
    assert delta_p2 < 0.15, (
        f"sbGRB R_peak2 / R_dip changed by {delta_p2 * 100:+.1f}% "
        f"({contrast_p2_c:.3f} -> {contrast_p2_f:.3f}) between dz=0.01 "
        f"and dz=0.005."
    )

    # Absolute-amplitude invariance.  Only meaningful after the
    # smooth_sigma=0 calibration fix in calibrate_mean_mass_evolved;
    # pre-fix this drifted +38.9% via the gaussian_filter1d reflective
    # boundary at z=0 in the calibration helper.  With matched physical
    # smoothing (Delta_z = 0.30 on both grids) the production-side
    # boundary bias is also matched, so the subset rate at z near the
    # dip / second peak should be invariant in the continuum limit.
    # 10% leaves headroom for finite-dz residuals.
    amp_dip = abs(R_d_f / R_d_c - 1.0)
    amp_p2 = abs(R_p2_f / R_p2_c - 1.0)
    assert amp_dip < 0.10, (
        f"sbGRB R_dip drifted by {amp_dip * 100:+.1f}% between dz=0.01 "
        f"(R_dip={R_d_c:.3e}) and dz=0.005 (R_dip={R_d_f:.3e}); pre-fix "
        f"this was 38.9% via the calibrate_mean_mass_evolved boundary-"
        f"smoothing bias.  Verify smooth_sigma=0 is still being passed "
        f"to its internal compute_merger_rate call."
    )
    assert amp_p2 < 0.10, (
        f"sbGRB R_peak2 drifted by {amp_p2 * 100:+.1f}% between dz=0.01 "
        f"(R_peak2={R_p2_c:.3e}) and dz=0.005 (R_peak2={R_p2_f:.3e})."
    )


@pytest.mark.xfail(
    reason="sbGRB+blueKN dip is Neijssel-specific; under Levina TNG100-1 there is "
    "no dip, so 'N_eff at the dip' is meaningless.  Same provenance as "
    "test_sbGRB_rate_dip_redshift_binning_invariant.",
    strict=False,
)
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_sbGRB_rate_dip_n_eff_above_threshold(bns_a_path):
    """Effective sample size at the cyan dip must exceed the few-binary
    regime that would make the curve sensitive to individual STROOPWAFEL
    outliers.

    Computed via ``per_system_rate_weights`` (the same function the
    notebook uses for the Section 7 1-sigma envelope):

        N_eff(z) = (sum_i w_i(z))^2 / sum_i w_i(z)^2

    A dip driven by 1-3 outlier binaries would show N_eff(z_dip) of
    order unity; a real low-Z-channel feature should still be supported
    by tens of effective binaries.  Threshold of 50 is set to 2x
    headroom over the measured N_eff(z_dip) ~ 99 on BNS A: high enough
    to flag a regression that drops effective sample size by half, low
    enough to leave headroom for legitimate downstream changes.  The
    first peak (~ 12700 effective binaries) and the second peak
    (~ 1300) are both far above the dip, as expected for a metallicity-
    transition feature at z ~ 4-5.
    """
    from grb_rates import per_system_rate_weights

    setup = _build_setup(bns_a_path)
    grid = _cosmic_grid(setup, redshift_step=0.01)
    # Find the dip on the smoothed production curve (same low-pass the
    # plot uses); per_system_rate_weights itself ignores smoothing, so
    # this only affects WHERE we evaluate N_eff, not N_eff itself.
    R = _R_sbgrb(setup, grid, smooth_sigma=30)

    z_peak1, _, _ = _find_extremum(grid.redshifts, R, 0.5, 2.5, "max")
    z_dip, _, _ = _find_extremum(grid.redshifts, R, 3.0, 5.5, "min")
    z_peak2, _, _ = _find_extremum(grid.redshifts, R, 5.5, 9.0, "max")

    def _n_eff(z_target):
        wi = per_system_rate_weights(
            z_target,
            grid.redshifts,
            grid.times,
            grid.time_first_SF,
            grid.n_formed,
            grid.p_draw,
            grid.dPdlogZ,
            grid.metallicities,
            setup.Z[setup.mask],
            setup.delays[setup.mask],
            setup.w[setup.mask],
        )
        wsum = float(wi.sum())
        wsum2 = float((wi**2).sum())
        if wsum2 <= 0.0:
            return 0.0
        return (wsum**2) / wsum2

    n_peak1 = _n_eff(z_peak1)
    n_dip = _n_eff(z_dip)
    n_peak2 = _n_eff(z_peak2)

    # Diagnostic context (visible under ``pytest -s``).
    print(
        f"\n[sbGRB + blue KN] N_eff(z={z_peak1:.2f}) = {n_peak1:.1f}, "
        f"N_eff(z={z_dip:.2f}) = {n_dip:.1f}, "
        f"N_eff(z={z_peak2:.2f}) = {n_peak2:.1f}"
    )

    assert n_dip >= 50.0, (
        f"sbGRB N_eff at z_dip = {z_dip:.2f} is {n_dip:.1f}; below the "
        f"50-binary floor that distinguishes a physical convolution "
        f"feature from STROOPWAFEL sparse-sample bumpiness (BNS A "
        f"baseline ~ 99).  Treat the cyan dip as a sample-starvation "
        f"artifact and either widen the sbGRB class or down-weight the "
        f"high-z portion of the curve."
    )
    assert n_peak1 > n_dip, (
        f"sbGRB N_eff at z_peak1={z_peak1:.2f} ({n_peak1:.1f}) does not "
        f"exceed N_eff at z_dip={z_dip:.2f} ({n_dip:.1f}); the lightest-"
        f"BNS class should have its largest effective contribution near "
        f"the SFR peak (z ~ 1.5), not at z > 3."
    )
    assert n_peak2 > 0.5 * n_dip, (
        f"sbGRB N_eff at z_peak2={z_peak2:.2f} ({n_peak2:.1f}) is below "
        f"half the dip's N_eff ({n_dip:.1f}); high-z bins should not "
        f"crash relative to the dip if the recovery is a physical "
        f"feature rather than weight-induced noise."
    )
