"""End-to-end notebook validation against the cited literature.

Part C of the physical-validation plan.  Reproduces the numerical
reductions printed by ``grb_main.ipynb`` Sections 4 to 11 (cosmic
integration, classification, rates, beaming, EOS sweep, offsets) by
calling the same ``grb_*.py`` helpers the notebook calls, then anchors
the resulting numbers to published values from:

- Broekgaarden et al. (2021) arXiv:2103.02608 (calibration anchor;
  ``weights_intrinsic/w_000``);
- Abbott et al. (2023) ApJ 949, 76 (LIGO O4 BNS / BHNS local-rate
  posteriors);
- Wanderman and Piran (2015) MNRAS 448, 3026 (observed sGRB R(z));
- Ghirlanda et al. (2016) A&A 594, A84 (observed sGRB local rate);
- Fong and Berger (2013) ApJ 776, 18 (observed sGRB host offsets);
- Gottlieb et al. (2023) arXiv:2309.00038, (2024) arXiv:2411.13657
  (BHNS disk-mass thresholds, BNS classification fractions);
- Bauswein et al. (2013) PRL 111, 131101 (EOS-dependent prompt-collapse
  threshold);
- Fong et al. (2015) ApJ 815, 102; Beniamini and Nakar (2019) MNRAS
  482, 5430 (jet half-opening angles for beaming).

A single module-scoped fixture builds the BNS A + BHNS A pipeline once
without spinning up a Jupyter kernel; tests are individually
``@pytest.mark.requires_data`` (and ``@pytest.mark.requires_compas``
for those that need the upstream cosmic-integration helpers).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def notebook_pipeline(bns_a_path, bhns_a_path):
    """Reproduce ``grb_main.ipynb`` Sections 4-11 numerical pipeline.

    Loads BNS A + BHNS A on the raw COMPAS masses, applies the Section 0
    cold-stability cut, runs the Gottlieb 2024 BNS classifier and the
    Foucart 2018 BHNS disk-mass classifier at fiducial ``a_BH = 0.5``,
    normalises by the shared run-configuration star-forming mass, and
    computes ``compute_merger_rate`` at ``smooth_sigma=0``.

    Unsmoothed is not an optimisation, it is the only correct setting for
    a test that reads R(z = 0) off the array.  The production
    ``smooth_sigma=30`` kernel reflects at the z = 0 boundary and folds the
    rising side of R(z) back onto the first bin, inflating the local rate by
    about 30 percent; the notebook smooths a display copy only and quotes
    every number from the unsmoothed array.  Comparing a smoothed R(z = 0)
    against a literature band tests the smoothing kernel, not the model.

    Returns a dict mirroring the variables the notebook prints; tests
    consume the exact same numbers.
    """
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed",
    )
    from astropy.cosmology import Planck15

    from grb_classify import classify_bhns, classify_bns_2024
    from grb_io import (
        DEFAULT_BHNS_PATH,
        DEFAULT_BNS_PATH,
        METALLICITY_GRID,
        filter_systems,
        load_bhns_with_kicks,
        load_bns_with_channels,
        load_bns_with_kicks,
        read_expected_local_rate,
        verify_shared_metallicity_prior,
    )
    from grb_physics import (
        M_TOV,
        MISALIGNMENT_SYSTEMATIC_FACTOR,
        cold_stable_ns_mask,
    )
    from grb_rates import (
        apply_bhns_misalignment,
        compute_merger_rate,
        dPdlogZ_binned_to_grid,
        simulation_star_forming_mass,
    )

    # ------------------------------------------------------------------
    # BNS load and cold-stability cut (notebook Section 0).  Raw COMPAS
    # gravitational masses, no quantile remap, and every component above
    # M_TOV leaves the sample before any classifier sees it.
    # ------------------------------------------------------------------
    bns_ch = load_bns_with_channels(path=bns_a_path)
    bns_k = load_bns_with_kicks(path=bns_a_path)
    _keep_bns = cold_stable_ns_mask(bns_ch["m1"], bns_ch["m2"], m_tov=M_TOV)
    bns_ch = filter_systems(bns_ch, _keep_bns)
    bns_k = filter_systems(bns_k, _keep_bns)

    m1_bns, m2_bns = bns_ch["m1"], bns_ch["m2"]
    w_bns = bns_ch["weights"]
    Z_bns = bns_ch["metallicity"]
    delay_bns = bns_ch["delay_time"]

    cls24 = classify_bns_2024(m1_bns, m2_bns)
    sbGRB_blue = cls24["sbGRB + blue KN"]
    lbGRB_hmns = cls24["lbGRB + red KN (HMNS)"]
    lbGRB_disk = cls24["lbGRB + red KN (disk)"]
    bns_faint_lb = cls24["Faint lbGRB"]

    # ------------------------------------------------------------------
    # BHNS load and Foucart 2018 disk-mass classification at a_BH=0.5.
    # ------------------------------------------------------------------
    bhns = load_bhns_with_kicks(path=bhns_a_path)
    bhns = filter_systems(bhns, cold_stable_ns_mask(bhns["M_NS"], m_tov=M_TOV))
    BH, NS_bh = bhns["M_BH"], bhns["M_NS"]
    w_bhns = bhns["weights"]
    Z_bhns = bhns["metallicity"]
    delay_bhns = bhns["delay_time"]

    cbhns = classify_bhns(BH, NS_bh, a_BH=0.5)
    bhns_no_grb = cbhns["No GRB"]
    bhns_short = cbhns["Faint lbGRB (BHNS)"]
    bhns_long = cbhns["lbGRB + red KN (BHNS disk)"]

    # ------------------------------------------------------------------
    # Cosmic integration (notebook Section 4).
    # ------------------------------------------------------------------
    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG100,
    )

    redshifts, _, times, time_first_SF, _, _ = fci.calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.01, cosmology=Planck15
    )
    sfr = fci.find_sfr(redshifts, **SFR_PARAMS_LEVINA26_TNG100)

    if bns_a_path == DEFAULT_BNS_PATH and bhns_a_path == DEFAULT_BHNS_PATH:
        Z_range = verify_shared_metallicity_prior(DEFAULT_BNS_PATH, DEFAULT_BHNS_PATH)
    else:
        Z_range = (float(min(Z_bns.min(), Z_bhns.min())), float(max(Z_bns.max(), Z_bhns.max())))

    dPdlogZ, metallicities, p_draw = fci.find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(Z_range[0]),
        max_logZ_COMPAS=np.log(Z_range[1]),
        **MSSFR_PARAMS_LEVINA26_TNG100,
    )

    expected_rate_bns = read_expected_local_rate(bns_a_path)
    expected_rate_bhns = read_expected_local_rate(bhns_a_path)

    # COMPAS ClassMSSFR bin convention (notebook Section 4): the top grid
    # point absorbs the 28.5 percent of z = 0 star formation above the
    # Z = 0.03 ceiling, which the pointwise lookup assigns to nothing.
    Z_grid_53 = METALLICITY_GRID[np.concatenate(([True], np.diff(METALLICITY_GRID) != 0))]
    dPdlogZ = dPdlogZ_binned_to_grid(dPdlogZ, metallicities, Z_grid_53, p_draw)

    # Run-configuration star-forming mass, shared by both populations.  The
    # published w_000 anchors above stay reference values: with this
    # normalisation and the binning above the rate path has no free
    # parameter, so reproducing them is a result rather than an input.
    mean_mass_bns = mean_mass_bhns = simulation_star_forming_mass(n_metallicities=Z_grid_53.size)
    n_formed_BNS = sfr / mean_mass_bns
    n_formed_BHNS = sfr / mean_mass_bhns

    # ------------------------------------------------------------------
    # Per-class BNS rates (notebook Section 7).
    # ------------------------------------------------------------------
    bns_class_masks = {
        "sbGRB + blue KN": sbGRB_blue,
        "lbGRB (HMNS)": lbGRB_hmns,
        "lbGRB (disk)": lbGRB_disk,
        "Faint lbGRB": bns_faint_lb,
        "All BNS": np.ones(len(delay_bns), dtype=bool),
    }
    merger_rates_BNS = {
        label: compute_merger_rate(
            redshifts,
            times,
            time_first_SF,
            n_formed_BNS,
            p_draw,
            dPdlogZ,
            metallicities,
            Z_bns[mask],
            delay_bns[mask],
            w_bns[mask],
            smooth_sigma=0,
        )
        for label, mask in bns_class_masks.items()
    }

    # ------------------------------------------------------------------
    # All-BHNS rate at the fiducial a_BH = 0.5 (notebook Section 8).
    # ------------------------------------------------------------------
    rate_bhns_all = compute_merger_rate(
        redshifts,
        times,
        time_first_SF,
        n_formed_BHNS,
        p_draw,
        dPdlogZ,
        metallicities,
        Z_bhns,
        delay_bhns,
        w_bhns,
        smooth_sigma=0,
    )
    rate_bhns_all_misaligned = apply_bhns_misalignment(rate_bhns_all)

    rate_bhns_long = compute_merger_rate(
        redshifts,
        times,
        time_first_SF,
        n_formed_BHNS,
        p_draw,
        dPdlogZ,
        metallicities,
        Z_bhns[bhns_long],
        delay_bhns[bhns_long],
        w_bhns[bhns_long],
        smooth_sigma=0,
    )

    iz0 = int(np.argmin(np.abs(redshifts)))

    return {
        "redshifts": redshifts,
        "iz0": iz0,
        "merger_rates_BNS": merger_rates_BNS,
        "rate_bhns_all": rate_bhns_all,
        "rate_bhns_all_misaligned": rate_bhns_all_misaligned,
        "rate_bhns_long": rate_bhns_long,
        "misalignment_factor": MISALIGNMENT_SYSTEMATIC_FACTOR,
        "cls24_masks": {
            k: v for k, v in cls24.items() if isinstance(v, np.ndarray) and v.dtype == bool
        },
        "bhns_class_masks": {
            "No GRB": bhns_no_grb,
            "Faint lbGRB (BHNS)": bhns_short,
            "lbGRB + red KN (BHNS disk)": bhns_long,
        },
        "w_bns": w_bns,
        "w_bhns": w_bhns,
        "delay_bns": delay_bns,
        "delay_bhns": delay_bhns,
        "v_sys_bns": bns_k["v_sys"],
        "v_sys_bhns": bhns["v_sys"],
        "v_sys_1_bns": bns_k["v_sys_1"],
        "v_sys_1_bhns": bhns["v_sys_1"],
        "t_leg1_bns": bns_k["t_sn2"] - bns_k["t_sn1"],
        "t_leg1_bhns": bhns["t_sn2"] - bhns["t_sn1"],
        "t_insp_bns": bns_k["t_inspiral"],
        "t_insp_bhns": bhns["t_inspiral"],
        "m1_bns": m1_bns,
        "m2_bns": m2_bns,
        "BH": BH,
        "NS_bh": NS_bh,
        "sbGRB_blue": sbGRB_blue,
        "lbGRB_hmns": lbGRB_hmns,
        "expected_rate_bns": expected_rate_bns,
        "expected_rate_bhns": expected_rate_bhns,
    }


# ─────────────────────────────────────────────────────────────────────
# LIGO O4 envelope: Abbott et al. (2023) ApJ 949, 76 (GWTC-3 BNS / BHNS).
# [no papers/ entry; the published rate posteriors are cited below.]
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
def test_bns_modelA_R0_within_LIGO_O4_BNS_band(notebook_pipeline):
    """All-BNS intrinsic R(z=0) sits inside the LVK GWTC-5.0 BNS 90 percent CR.

    ``LVK_GWTC5_LOCAL_RATES["BNS"]`` carries the intrinsic source-frame
    merger-rate density at z = 0 from the GWTC-5.0 population analysis
    (LVK 2026, arXiv:2605.27226, Table 2 "Joint" row): 5.1 to 154.7
    Gpc^-3 yr^-1.  This supersedes the GWTC-3 envelope of 10 to 1700 the
    test used to quote, which is roughly ten times wider and could not
    fail for any drift this pipeline is capable of.

    Both sides are intrinsic rates at z = 0, so the comparison is direct:
    no beaming factor and no misalignment factor, per the GW-rate
    convention in CLAUDE.md.  Model A lands at about 47 Gpc^-3 yr^-1,
    inside the band with a factor of roughly 3 of headroom either way.
    """
    from grb_rates import LVK_GWTC5_LOCAL_RATES

    pipe = notebook_pipeline
    R0 = float(pipe["merger_rates_BNS"]["All BNS"][pipe["iz0"]])
    assert R0 > 0
    band = LVK_GWTC5_LOCAL_RATES["BNS"]
    lo, hi = band["R_lo"], band["R_hi"]
    assert lo <= R0 <= hi, (
        f"All-BNS R(z=0) = {R0:.2f} Gpc^-3 yr^-1 is outside the LVK "
        f"GWTC-5.0 BNS 90 percent CR [{lo}, {hi}].  Either the "
        f"star-forming-mass normalisation broke, the metallicity binning "
        f"drifted, or the sample is no longer Model A."
    )


@pytest.mark.requires_data
@pytest.mark.requires_compas
def test_bhns_modelA_R0_within_LIGO_O4_BHNS_band(notebook_pipeline):
    """All-BHNS intrinsic R(z=0) overshoots the GWTC-5.0 NSBH band, by the documented factor.

    Two separate claims, both of which the earlier form of this test got
    wrong by comparing a misalignment-corrected rate against a GW band
    inside an envelope wide enough to accept anything.

    First, the comparison has to be made on the *intrinsic* rate.
    ``MISALIGNMENT_SYSTEMATIC_FACTOR`` is a jet-orientation suppression;
    the GW measurement already folds in its own selection effects through
    the population analysis, so applying the jet factor as well would
    double-suppress the model by 2x against a band it is being tested on
    (the GW-rate convention in CLAUDE.md).

    Second, Model A does not sit inside the band and the notebook does not
    claim it does.  GWTC-5.0 added no new NSBH, so the NSBH band tightened
    to 6.7 to 32.8 Gpc^-3 yr^-1 while Model A predicts about 73, a factor
    of roughly 2.2 above the upper edge.  That overshoot is the finding
    Sections 7 and 11.1b report, so the test asserts it rather than
    hiding it inside a loose envelope.  The band on the ratio is a
    regression tripwire: it would catch a factor-of-2 drift in either
    direction while tolerating the sampling and MSSFR spread.

    The rate-level-only nature of the misalignment factor is checked here
    too, since a sample-level double-count is the classic way to halve
    this rate twice without any test noticing.
    """
    from grb_rates import LVK_GWTC5_LOCAL_RATES

    pipe = notebook_pipeline
    iz0 = pipe["iz0"]
    R0_intrinsic = float(pipe["rate_bhns_all"][iz0])
    R0_misaligned = float(pipe["rate_bhns_all_misaligned"][iz0])
    assert R0_intrinsic > 0

    # Misalignment is a scalar applied once, at the rate level.
    assert R0_misaligned == pytest.approx(R0_intrinsic * pipe["misalignment_factor"], rel=1e-12), (
        f"misalignment-corrected R(z=0) = {R0_misaligned:.4f} is not "
        f"{pipe['misalignment_factor']} x the intrinsic {R0_intrinsic:.4f}; "
        f"apply_bhns_misalignment must be rate-level only, never also a "
        f"sample-level cut."
    )

    band = LVK_GWTC5_LOCAL_RATES["NSBH"]
    ratio = R0_intrinsic / band["R_hi"]
    lo, hi = 1.5, 4.0
    assert lo <= ratio <= hi, (
        f"All-BHNS intrinsic R(z=0) = {R0_intrinsic:.2f} Gpc^-3 yr^-1 is "
        f"{ratio:.2f}x the GWTC-5.0 NSBH upper edge {band['R_hi']}, outside "
        f"the documented overshoot band [{lo}, {hi}].  Below {lo} the "
        f"headline tension in Sections 7 and 11.1b has gone away and the "
        f"manuscript needs rewording; above {hi} something broke in the "
        f"normalisation or the Foucart disk-mass classification."
    )


# ─────────────────────────────────────────────────────────────────────
# Beamed-rate observed comparison (Fong 2015, Beniamini-Nakar 2019,
# Wanderman-Piran 2015, Ghirlanda 2016).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
def test_bns_beamed_sbgrb_in_observed_sgrb_band(notebook_pipeline):
    """Beamed sbGRB + blue KN local rate sits in the observed sGRB band.

    The observational handles on the local, beaming-limited sGRB rate,
    as the papers actually report them:

      Wanderman and Piran (2015) MNRAS 448, 3026: R0 = 4.1 +2.3/-1.9,
        so [2.2, 6.4] Gpc^-3 yr^-1.
      Ghirlanda et al. (2016) A&A 594, A84 Sec. 7: rho_0 = 0.20 +0.04/-0.07
        for their model (a) and 0.8 +0.3/-0.15 for model (c), so
        [0.13, 1.1] across both.

    Union: [0.13, 6.4] Gpc^-3 yr^-1.  That is wide because the two papers
    disagree by more than an order of magnitude on the same quantity, not
    because the band has been padded; Model A's 0.83 lands in the gap
    between them.

    Colombo et al. (2022) used to be a third handle here at 3.6 (1.8-6.5).
    Those numbers came from their Table 1, "Assumed Detection Limits and
    Predicted Detection Rates in Our Observing Scenarios", whose "Rate" row
    is predicted detections per year for an instrument, not a rate density.
    Their actual local rate density is 347 +536/-256 Gpc^-3 yr^-1 and is an
    intrinsic NSNS merger rate, so it cannot be compared against a beamed
    sGRB rate at all.  It is now ``grb_rates.COLOMBO22_NSNS_LOCAL_RATE``.

    The intrinsic BNS sbGRB rate from Section 6 multiplied by f_beam =
    1 - cos(theta_j_fid) at theta_j_fid = 13 deg (Fong 2015 / Beniamini
    and Nakar 2019 fiducial) should sit in this band.  Section 6c prints
    R_sbGRB beamed ~ 0.8 Gpc^-3 yr^-1 for Model A, well inside the
    envelope.

    Cited assumption: 100 percent jet launching above the disk-mass
    threshold (Gottlieb 2023); all disk-mass GRB rates are therefore
    upper bounds, and real beamed rates are at most this value.
    """
    from grb_rates import CLASS_THETA_J, beamed_rate

    pipe = notebook_pipeline
    R_sb_int = float(pipe["merger_rates_BNS"]["sbGRB + blue KN"][pipe["iz0"]])
    theta_fid = CLASS_THETA_J["sbGRB"]["fid"]
    R_sb_beamed = float(beamed_rate(R_sb_int, theta_fid))

    # The union of the two papers' published intervals, used as published.
    # Two earlier forms of this were wrong in opposite directions: the band
    # had been widened to [0.10, 10.0] "to account for the upper-bound
    # caveat", and its lower edge rested on a Ghirlanda value of 0.5 that
    # the paper does not report.
    lo, hi = 0.13, 6.4
    assert lo <= R_sb_beamed <= hi, (
        f"Beamed sbGRB R(z=0) = {R_sb_beamed:.4f} Gpc^-3 yr^-1 (theta_j "
        f"= {theta_fid} deg) is outside the observed sGRB envelope "
        f"[{lo}, {hi}] from WP15 / Ghirlanda 2016.  "
        f"Either the intrinsic sbGRB rate broke or the beaming "
        f"convention drifted from f_beam = 1 - cos(theta_j)."
    )


# ─────────────────────────────────────────────────────────────────────
# Class-fraction sanity tests (Gottlieb 2023, 2024).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_class_fractions_sum_to_unity(notebook_pipeline):
    """Gottlieb (2024) four-class fractions sum to 1 and are non-negative.

    Gottlieb et al. (2024) arXiv:2411.13657 Sec. 2.3 partitions the BNS
    population into four mutually-exclusive engine classes (sbGRB +
    blue KN, lbGRB + red KN HMNS, lbGRB + red KN disk, Faint lbGRB).
    By construction every BNS belongs to exactly one class, so the
    weighted fractions must sum to 1 to floating-point precision.

    Surfaces double-counting bugs in classify_bns_2024 (e.g. an
    overlapping mask boundary) and missing-class bugs (e.g. the prior
    Q_NO_DISK = 1.05 cut that was removed; see classify_bns_2024
    docstring).
    """
    pipe = notebook_pipeline
    w = pipe["w_bns"]
    masks = pipe["cls24_masks"]
    fractions = {label: float((w[m] / w.sum()).sum()) for label, m in masks.items()}
    total = sum(fractions.values())
    assert total == pytest.approx(1.0, abs=1e-9), (
        f"Gottlieb (2024) class fractions sum to {total}, not 1.  Per-class breakdown: {fractions}"
    )
    assert all(f >= 0.0 for f in fractions.values()), (
        f"Negative class fraction encountered: {fractions}"
    )


@pytest.mark.requires_data
def test_class_fractions_dominated_by_HMNS_for_modelA(notebook_pipeline):
    """For Model A + Alsing remap, sbGRB + lbGRB(HMNS) > 50 percent of BNS.

    Gottlieb et al. (2024) Fig. 3 shows that for ``M_TOV ~ 2.2 Msun``
    (the project fiducial) the BNS population near the Galactic peak
    (~1.30-1.35 Msun NSs) sits below the prompt-collapse threshold
    ``M_thresh ~ 1.27 * M_TOV ~ 2.79 Msun``, so the HMNS-engine
    classes (sbGRB + blue KN and lbGRB + red KN HMNS) carry the
    majority of the population.  After the Alsing 2018 NS remap closes
    the Fryer rapid-engine gap and pushes mass into the 1.7-1.8 Msun
    tail, the lbGRB (HMNS) class grows but the combined HMNS-engine
    fraction stays above 50 percent.

    Test enforces this qualitative prediction.  A drop below 50
    percent would either indicate (a) the remap is not being applied
    (notebook line 112-115), (b) ``hmns_factor`` was raised, or (c)
    the COMPAS sample shifted from Model A.
    """
    pipe = notebook_pipeline
    w = pipe["w_bns"]
    sb = pipe["sbGRB_blue"]
    hmns = pipe["lbGRB_hmns"]
    f_hmns_engine = float((w[sb | hmns] / w.sum()).sum())
    assert f_hmns_engine > 0.50, (
        f"HMNS-engine fraction (sbGRB + lbGRB HMNS) = "
        f"{f_hmns_engine:.3f} is below the 0.50 threshold expected "
        f"for Model A + Alsing remap from Gottlieb (2024) Fig. 3.  "
        f"Verify the Alsing NS remap is applied and hmns_factor = 1.2."
    )


@pytest.mark.requires_data
def test_bhns_lbgrb_disk_fraction_matches_gottlieb_2023_band(notebook_pipeline):
    """BHNS lbGRB + red KN (BHNS disk) fraction at a_BH=0.5 sits in the documented Gottlieb 2023 band.

    Gottlieb et al. (2023) arXiv:2309.00038 Sec. 4 / Fig. 6 set the
    BHNS disk-mass thresholds.  Under the Gottlieb 2024 hybrid relabel,
    the ``M_disk >= 0.1 Msun`` class is ``lbGRB + red KN (BHNS disk)``
    (Foucart 2018 disk mass at fiducial spin); it carries a small
    fraction of the BHNS population, with the remainder mostly the
    No-GRB class (``M_disk < 0.01 Msun``, NS swallowed without
    disrupting).

    The notebook reports ~few percent of this class at ``a_BH = 0.5``
    for Model A (Broekgaarden+ 2021); the actual fraction depends on
    the BH-spin distribution and the NS-radius EOS.  Test enforces the
    fraction sits in [0.001, 0.30], a deliberately wide band: a value
    above 0.30 would indicate the disk-mass threshold collapsed (e.g.
    ``MDISK_LONG`` drift), a value below 0.001 would indicate the
    Foucart formula clipped everything.
    """
    pipe = notebook_pipeline
    w = pipe["w_bhns"]
    long_mask = pipe["bhns_class_masks"]["lbGRB + red KN (BHNS disk)"]
    f_long = float((w[long_mask] / w.sum()).sum())
    lo, hi = 0.001, 0.30
    assert lo <= f_long <= hi, (
        f"BHNS lbGRB + red KN (BHNS disk) fraction at a_BH=0.5 = "
        f"{f_long:.4f} sits outside the Gottlieb (2023) Sec. 4 band "
        f"[{lo}, {hi}].  "
        f"Check MDISK_LONG, the Foucart 2018 disk-mass formula, and "
        f"the BH-spin assumption."
    )


# ─────────────────────────────────────────────────────────────────────
# Wanderman-Piran (2015): the model R(z) is broader than the observed
# short-GRB rate history.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
def test_model_rate_history_is_broader_than_wanderman_piran_2015(notebook_pipeline):
    """The intrinsic model R(z) outruns the WP15 observed sGRB rate above the peak.

    This replaces a reduced-chi-square regression band.  That test
    normalised the beamed all-BNS curve to the Wanderman and Piran (2015)
    R(z = 0), computed a chi^2 against their 1-sigma envelope, and
    asserted the result landed in [1e4, 1e6] around a value of 1.7e5.  A
    reduced chi-square of 170,000 is not a fit statistic anyone should
    record as a baseline, and the quantity is not a goodness-of-fit in the
    first place: the two curves are different observables, so the number
    was never going to approach unity and the band around it could only
    ever detect a gross structural break.  The notebook no longer draws a
    WP15 panel at all; Section 6c compares against Pracchia and Salafia
    (2026) and Du et al. (2025) instead.

    What the old docstring correctly identified is worth keeping, so it is
    asserted directly here.  WP15 Eq. (9) is a luminosity-function- and
    beaming-folded *observed* short-GRB rate that falls off sharply above
    its z = 0.9 peak.  The model curve is an intrinsic merger-rate
    density carrying only a flat top-hat beaming factor, and it is fed by
    a star-formation history that peaks near z = 2.4.  It must therefore
    grow relative to WP15 with increasing redshift once past the WP15
    peak.  Both curves are normalised at z = 0 so the comparison is one of
    shape alone.

    A failure here means the delay-time convolution or the MSSFR shape
    changed enough to alter the redshift evolution of the population,
    which is a real result and should not pass silently.

    Reference: Wanderman and Piran (2015) MNRAS 448, 3026, Eq. (9);
    ``grb_rates.wanderman_piran_2015_Rz``.  The functional form itself is
    anchored in ``tests/anchors/test_literature_anchors.py``.
    """
    from grb_rates import CLASS_THETA_J, beamed_rate, wanderman_piran_2015_Rz

    pipe = notebook_pipeline
    redshifts = pipe["redshifts"]
    z_mask = redshifts <= 5.0
    z_plot = redshifts[z_mask]
    wp15 = wanderman_piran_2015_Rz(z_plot)["R_best"]

    theta_fid = CLASS_THETA_J["sbGRB"]["fid"]
    model = beamed_rate(pipe["merger_rates_BNS"]["All BNS"], theta_fid)[z_mask]

    # Shape-only comparison: both curves scaled to 1 at z = 0.
    model_norm = model / model[0]
    wp15_norm = wp15 / wp15[0]
    ratio = model_norm / wp15_norm

    assert np.all(np.isfinite(ratio)), "model / WP15 shape ratio is not finite"

    def _at(z):
        return float(np.interp(z, z_plot, ratio))

    r_peak, r_1p5, r_3 = _at(0.9), _at(1.5), _at(3.0)

    assert r_peak < r_1p5 < r_3, (
        f"model / WP15 shape ratio is not increasing past the WP15 peak: "
        f"{r_peak:.3g} at z = 0.9, {r_1p5:.3g} at z = 1.5, {r_3:.3g} at "
        f"z = 3.  The intrinsic model history should outrun the observed "
        f"short-GRB history above the WP15 turnover; check the delay-time "
        f"convolution and the MSSFR shape."
    )
    assert r_3 > 10.0, (
        f"model / WP15 shape ratio at z = 3 is only {r_3:.3g}.  WP15 falls "
        f"off exponentially above its peak while the model tracks the star "
        f"formation history, so the separation by z = 3 should be at least "
        f"an order of magnitude."
    )


# ─────────────────────────────────────────────────────────────────────
# Offset CDFs against the Fong et al. 2022 and Skobe et al. 2026 samples.
# ─────────────────────────────────────────────────────────────────────
def _class_offsets(pipe, mask, population="bns"):
    """Orbit-integrate one class through the Section 8 host mixture.

    Uses the two-leg kinematics: the binary drifts at the post-SN1 velocity
    for the SN1-to-SN2 baseline, then at the post-SN2 velocity for the
    post-DCO inspiral time, so ``t_delay_myr`` here is the inspiral time.
    """
    from grb_offsets import compute_offsets_mixed_hosts

    if mask.sum() < 100:
        pytest.skip("not enough systems in this class to AD-test")

    res = compute_offsets_mixed_hosts(
        pipe[f"v_sys_{population}"][mask],
        pipe[f"t_insp_{population}"][mask],
        weights=pipe[f"w_{population}"][mask],
        v_sys_1_km=pipe[f"v_sys_1_{population}"][mask],
        t_leg1_myr=pipe[f"t_leg1_{population}"][mask],
        max_systems=20000,
    )
    finite = np.isfinite(res["mixed_offsets"])
    if finite.sum() < 10:
        pytest.skip("offset integration produced too few finite samples")
    return res["mixed_offsets"][finite], res["mixed_weights"][finite]


@pytest.mark.requires_data
@pytest.mark.slow
def test_offset_cdf_ad_pvalue_sgrb_engine_above_threshold(notebook_pipeline):
    """The short-GRB engine is consistent with the classical short-GRB offsets.

    Notebook Section 8 orbit-integrates BNS systems through a Hernquist
    (1990) potential with a 75/25 SF/elliptical host mix and compares each
    class against the observed sample its predicted signature belongs to.
    ``sbGRB + blue KN`` is the only class that predicts a short burst, so it
    is the only one entitled to face the classical short-GRB sample: the Fong
    et al. (2022) BRIGHT catalogue (ApJ 940, 56) less the bursts Skobe et al.
    (2026) reclassify as merger-driven long.  Comparing an lbGRB class here
    would test the model against observations it says those systems cannot
    produce.

    The test mirrors the notebook: a Monte-Carlo-calibrated one-sample
    Anderson-Darling goodness-of-fit test (``ad_gof_weighted``), which treats
    the weighted STROOPWAFEL model as the reference distribution and the
    ~80-burst catalogue as the observed sample, since the model outnumbers the
    data by a factor of order 1000.  A p-value above 0.01 means the observed
    offsets are not distinguishable from the model at the 1 percent level.
    """
    from grb_offsets import OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC, ad_gof_weighted

    offsets, weights = _class_offsets(notebook_pipeline, notebook_pipeline["sbGRB_blue"])
    statistic, pvalue = ad_gof_weighted(
        offsets, OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC, weights=weights
    )
    assert pvalue > 0.01, (
        f"One-sample AD p-value of the classical sGRB sample against the "
        f"sbGRB + blue KN model offset CDF = {pvalue:.4f} (A2 = {statistic:.3f}) "
        f"is below the 0.01 consistency threshold.  Check the host mixture "
        f"(HOST_MODELS), the Hernquist scale radius, or the v_sys "
        f"distribution for drift."
    )


@pytest.mark.requires_data
@pytest.mark.slow
def test_lgrb_engines_are_consistent_with_the_kilonova_selected_long_grbs(
    notebook_pipeline,
):
    """The lbGRB classes match the observed long-GRB kilonova offsets.

    Section 8 puts the five lbGRB engines against ``OBSERVED_LGRB_KN_OFFSETS_KPC``
    (Skobe et al. 2026 Table 2) rather than against the short-GRB samples.  At
    N = 4 the test has essentially no power, so this pins the weaker claim the
    notebook actually makes: none of the lbGRB classes is grossly inconsistent
    with that sample.  A failure here means the predicted offsets have moved far
    enough that even four events can see it.
    """
    from grb_offsets import OBSERVED_LGRB_KN_OFFSETS_KPC, ad_gof_weighted

    pipe = notebook_pipeline
    lgrb_engines = [
        (label, pipe["cls24_masks"][label], "bns")
        for label in ("lbGRB + red KN (HMNS)", "lbGRB + red KN (disk)", "Faint lbGRB")
    ] + [
        (label, pipe["bhns_class_masks"][label], "bhns")
        for label in ("Faint lbGRB (BHNS)", "lbGRB + red KN (BHNS disk)")
    ]

    for label, mask, population in lgrb_engines:
        offsets, weights = _class_offsets(pipe, mask, population)
        _, pvalue = ad_gof_weighted(offsets, OBSERVED_LGRB_KN_OFFSETS_KPC, weights=weights)
        assert pvalue > 0.05, (
            f"One-sample AD p-value of the lGRB-KN sample against {label} = "
            f"{pvalue:.4f}; four events should not be able to reject a class."
        )


@pytest.mark.requires_data
@pytest.mark.slow
def test_offset_consistency_depends_on_the_host_association_cut(notebook_pipeline):
    """The Section 8 verdict depends on which edge of the classical band you read.

    With the two-leg kinematics the sbGRB engine predicts compact offsets
    (median ~5-6 kpc), so the consistency ordering runs the other way from a
    wide-tailed model: the model is most consistent with the tight Gold
    secure-host sample (median 5.1 kpc), marginal against the all-associations
    catalogue (8.4 kpc), and rejected by the uncertainty-broadened edge (10.9
    kpc).  Anderson-Darling is tail-weighted, so broadening the observed sample
    upward is what breaks the comparison.

    The class under test is ``sbGRB + blue KN``, the only one whose predicted
    signature belongs in the classical short-GRB sample.

    Pinning the monotone ordering (Gold most consistent, broadened least) means
    a change to the kick or host prescription that alters the offset scale
    cannot quietly flip the reported consistency.  The Gold curve is a floor,
    not a preferred comparison: Fong et al. (2022) Sec. 6.2 read its median as a
    minimum, and the Inconclusive bursts missing from every curve are plausibly
    wider still (their Sec. 7.1).
    """
    from grb_offsets import classical_sgrb_ad_pvalues

    offsets, weights = _class_offsets(notebook_pipeline, notebook_pipeline["sbGRB_blue"])
    p = classical_sgrb_ad_pvalues(offsets, weights)
    assert p["p_sampled"] < p["p_observed"] < p["p_gold"], (
        f"The compact two-leg model should be most consistent with the tight "
        f"Gold sample and least with the broadened edge, got Gold "
        f"{p['p_gold']:.4f}, all-associations {p['p_observed']:.4f}, broadened "
        f"{p['p_sampled']:.4f}."
    )
    assert p["p_gold"] > 0.05, (
        f"The Gold secure-host comparison is expected to be consistent with this "
        f"class (p = {p['p_gold']:.4f}); if it now rejects, the model's offset "
        f"scale has shifted and the Section 8 caption needs rewriting."
    )
    assert p["p_sampled"] < 0.05, (
        f"The uncertainty-broadened edge is expected to reject the compact model "
        f"(p = {p['p_sampled']:.4f})."
    )
    assert p["p_sampled_lo"] <= p["p_sampled"] <= p["p_sampled_hi"]


# ─────────────────────────────────────────────────────────────────────
# EOS sweep coherence (Bauswein et al. 2013, PRL 111, 131101).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_eos_sensitivity_M_thresh_moves_with_M_TOV(notebook_pipeline):
    """``compute_eos_sensitivity`` shifts M_thresh and class fractions monotonically.

    Bauswein et al. (2013) PRL 111, 131101 Table I: the prompt-collapse
    total-mass threshold scales with ``M_TOV`` via a roughly EOS-
    independent ratio ``k = M_thresh / M_TOV in [1.30, 1.70]``.  The
    project pins ``K_THRESH_DEFAULT = 1.27``; with that fiducial,
    sweeping EOSs from SFHo (``M_TOV = 2.06``) to DD2 (``M_TOV = 2.42``)
    must move ``M_thresh`` by

        Delta M_thresh = K_THRESH_DEFAULT * Delta M_TOV
                       = 1.27 * (2.42 - 2.06) = 0.46 Msun.

    More importantly, the prompt-collapse class fraction must move
    monotonically: stiffer EOSs (higher M_TOV) raise M_thresh, fewer
    BNS systems exceed it, fewer prompt collapses, and the
    HMNS-engine class fractions grow.

    This test checks both: (a) M_thresh in the table differs by >=
    0.40 Msun between SFHo and DD2 (validates the K-coupling); (b) the
    sum of the two prompt-collapse class fractions ('lbGRB + red KN
    (disk)' + 'Faint lbGRB') is strictly larger for SFHo than for DD2
    (validates the monotonicity of the sweep).

    Reference: Bauswein et al. (2013) PRL 111, 131101 Table I;
    ``grb_rates.compute_eos_sensitivity`` and
    ``grb_classify.classify_bns_2024`` jointly enforce the EOS-sweep
    coherence (M_thresh = k * M_TOV with k held fixed at K_THRESH_DEFAULT).
    """
    from grb_physics import EOS_MODELS
    from grb_rates import compute_eos_sensitivity

    pipe = notebook_pipeline
    eos_table = compute_eos_sensitivity(pipe["m1_bns"], pipe["m2_bns"], pipe["w_bns"])

    # M_thresh coupling: Delta M_thresh = K * Delta M_TOV.
    m_thresh_sfho = float(eos_table.loc["SFHo", "M_thresh"])
    m_thresh_dd2 = float(eos_table.loc["DD2", "M_thresh"])
    delta = m_thresh_dd2 - m_thresh_sfho
    delta_expected = 1.27 * (EOS_MODELS["DD2"]["M_TOV"] - EOS_MODELS["SFHo"]["M_TOV"])
    assert delta == pytest.approx(delta_expected, rel=1e-6), (
        f"M_thresh shift across EOSs = {delta:.4f} Msun does not match "
        f"k * Delta M_TOV = 1.27 * "
        f"({EOS_MODELS['DD2']['M_TOV']} - {EOS_MODELS['SFHo']['M_TOV']}) "
        f"= {delta_expected:.4f}.  EOS sweep coherence invariant broken."
    )
    assert delta >= 0.40, (
        f"M_thresh shift across SFHo -> DD2 = {delta:.3f} Msun is "
        f"below the expected 0.4 Msun band; either the EOS table or "
        f"the K_THRESH_DEFAULT coupling regressed."
    )

    # Monotonicity of prompt-collapse fraction.
    f_prompt_sfho = float(
        eos_table.loc["SFHo", "lbGRB + red KN (disk)"] + eos_table.loc["SFHo", "Faint lbGRB"]
    )
    f_prompt_dd2 = float(
        eos_table.loc["DD2", "lbGRB + red KN (disk)"] + eos_table.loc["DD2", "Faint lbGRB"]
    )
    assert f_prompt_sfho > f_prompt_dd2, (
        f"Prompt-collapse fraction at SFHo ({f_prompt_sfho:.4f}) is "
        f"not greater than at DD2 ({f_prompt_dd2:.4f}), violating the "
        f"Bauswein (2013) expectation that softer EOSs (lower M_TOV "
        f"-> lower M_thresh) prompt-collapse more often."
    )
