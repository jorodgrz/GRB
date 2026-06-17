"""
Cosmic merger rate computation and related utilities.

compute_merger_rate is a thin chunked accumulator around
``compas_python_utils.cosmic_integration.FastCosmicIntegration.find_formation_and_merger_rates``;
the MSSFR fiducial is the Levina+ 2026 (arXiv:2601.20202) skewed log-normal
fit to IllustrisTNG TNG100-1 (Table 1).  Also includes Kroupa (2001) IMF
verification, per-system rate weights, EOS / spin / beaming sweeps, and
LVK observed-rate constants (``OBSERVED_SGRB_RATES`` for beaming-limited
sGRB rates, ``LVK_GWTC5_LOCAL_RATES`` for intrinsic GW merger rates from
the GWTC-5.0 population analysis (arXiv:2605.27226)).
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d

from grb_physics import (
    EPS_JET,
    GOTTLIEB25_T_HMNS_RANGE,
    MISALIGNMENT_SYSTEMATIC_FACTOR,
    T_D_PROMPT_RANGE,
    T_E_RANGE,
    V_EJ_BNS,
    ejecta_energy_in_cone,
    jet_breaks_out_pais,
    jet_energy_from_disk,
)

# ═══════════════════════════════════════════════════════════════════════════
# MSSFR / SFR parameters: Levina+ 2026 IllustrisTNG TNG100-1 fiducial
# ═══════════════════════════════════════════════════════════════════════════
# Madau and Dickinson (2014) functional form S(z) = a (1+z)^b / (1 + ((1+z)/c)^d),
# best-fit to TNG100-1 in Levina+ 2026 Table 1.
SFR_PARAMS_LEVINA26_TNG100 = {
    "a": 0.0172,
    "b": 1.4425,
    "c": 4.5299,
    "d": 6.2261,
}

# Skewed log-normal metallicity PDF (Azzalini), best-fit to TNG100-1 in
# Levina+ 2026 Table 1.  Eq. (3)-(6) of Levina+ 2026 use the parametrisation
#   <Z>(z) = mu0 * 10^(muz * z),   omega(z) = omega0 * 10^(omegaz * z)
# which matches COMPAS ``find_metallicity_distribution`` exactly: Levina's
# (omega0, omegaz) maps onto COMPAS (sigma_0, sigma_z), and Levina's alpha
# is the COMPAS skewness parameter.
MSSFR_PARAMS_LEVINA26_TNG100 = {
    "mu0": 0.0247,
    "muz": -0.0521,
    "sigma_0": 1.1509,
    "sigma_z": 0.0477,
    "alpha": -1.8801,
}

# Levina+ 2026 Table 1 TNG50-1 column (highest resolution, smallest box).
SFR_PARAMS_LEVINA26_TNG50 = {
    "a": 0.0329,
    "b": 1.4668,
    "c": 3.8412,
    "d": 5.0994,
}
MSSFR_PARAMS_LEVINA26_TNG50 = {
    "mu0": 0.0282,
    "muz": -0.0314,
    "sigma_0": 1.1136,
    "sigma_z": 0.0592,
    "alpha": -1.7353,
}

# Levina+ 2026 Table 1 TNG300-1 column (largest box, lowest resolution).
SFR_PARAMS_LEVINA26_TNG300 = {
    "a": 0.0097,
    "b": 1.5747,
    "c": 4.5428,
    "d": 6.8266,
}
MSSFR_PARAMS_LEVINA26_TNG300 = {
    "mu0": 0.0237,
    "muz": -0.0687,
    "sigma_0": 1.1196,
    "sigma_z": 0.0481,
    "alpha": -2.2726,
}

# Convenience grouping used by the TNG-resolution sweep notebook section.
LEVINA26_TNG_VARIANTS = {
    "TNG50-1": (SFR_PARAMS_LEVINA26_TNG50, MSSFR_PARAMS_LEVINA26_TNG50),
    "TNG100-1": (SFR_PARAMS_LEVINA26_TNG100, MSSFR_PARAMS_LEVINA26_TNG100),
    "TNG300-1": (SFR_PARAMS_LEVINA26_TNG300, MSSFR_PARAMS_LEVINA26_TNG300),
}

# Levina+ 2026 Table 2 published BBH local merger rates [Gpc^-3 yr^-1].
# ``R_sim`` integrates the simulation S(Z, z) directly; ``R_fit`` integrates
# the analytical skewed log-normal fit (the parameter sets above).  These
# are reference values; the project does not load BBH samples and cannot
# reproduce them end-to-end on BNS / BHNS data.  Used as a numerical anchor
# (constants pinned in tests/anchors/test_literature_anchors.py) and as the
# context for the TNG-resolution sweep in Section 4b.
LEVINA26_BBH_LOCAL_RATES = {
    "TNG50-1": {"R_sim": 58.92, "R_fit": 73.72},
    "TNG100-1": {"R_sim": 42.91, "R_fit": 45.53},
    "TNG300-1": {"R_sim": 29.34, "R_fit": 27.81},
}


# ═══════════════════════════════════════════════════════════════════════════
# Common envelope prescription pin
# ═══════════════════════════════════════════════════════════════════════════
# Common envelope prescription, frozen at the COMPAS commit pinned in
# environment.yml.  Energy formalism (Webbink 1984, ApJ 277, 355); envelope
# binding energies from Xu and Li (2010, ApJ 716, 114) lambda_CE fits as
# used by Broekgaarden+ 2021 (arXiv:2103.02608) Sec. 3.3.  Stability
# criterion is the Hurley+ 2002 (MNRAS 329, 897) zeta_ad approach with
# COMPAS modifications from Vinciguerra+ 2020 (MNRAS 498, 4705).  We do
# NOT adopt the Hirai+ updated stability prescription; switching to that
# would require a fresh COMPAS run and would trip the regression tests
# in tests/sections/test_section_06_formation_channels.py.
# alpha_CE per Broekgaarden+ 2021 Sec. 5.2 Table 2.
CE_PRESCRIPTION_BROEKGAARDEN21 = {
    "energy_formalism": "Webbink 1984",
    "lambda_CE": "Xu and Li 2010",
    "stability": "Hurley 2002 (Vinciguerra+ 2020 modified)",
    "double_CE_flag": "both stars in core-He burning at CE onset",
    "alpha_CE": {"A": 1.0, "F": 0.5, "G": 2.0, "J": 1.0, "K": 1.0},
}


# ═══════════════════════════════════════════════════════════════════════════
# Cosmic integration: chunked wrapper around FastCosmicIntegration
# ═══════════════════════════════════════════════════════════════════════════
def compute_merger_rate(
    redshifts,
    times,
    time_first_SF,
    n_formed,
    p_draw,
    dPdlogZ,
    metallicities,
    COMPAS_Z,
    COMPAS_delay_times,
    COMPAS_weights,
    smooth_sigma=30,
    n_chunk=10_000,
):
    """Cosmic merger rate density [Gpc^-3 yr^-1] vs redshift, via FCI.

    Thin chunked accumulator around
    ``compas_python_utils.cosmic_integration.FastCosmicIntegration.find_formation_and_merger_rates``.
    FCI returns an ``(n_binaries, n_z)`` merger-rate matrix; at BHNS sample
    size (~1.5e6 systems) with dz = 0.01 (~1e3 redshift bins) that is
    ~24 GB and OOMs on most laptops.  We loop over chunks of ``n_chunk``
    binaries, sum each chunk's merger_rate axis-0 into a running
    ``(n_z,)`` accumulator, and keep peak allocation at
    ``n_chunk * n_z * 8 bytes ~= 80 MB`` per chunk.

    The caller computes ``dPdlogZ`` / ``metallicities`` / ``p_draw`` once
    via ``find_metallicity_distribution`` and passes them in, so the MSSFR
    convolution is not redone per spin / EOS / class subset.

    ``n_formed`` must already include the 1/MEAN_MASS_EVOLVED factor.
    ``find_sfr`` returns raw SFR in Msun/yr/Gpc^3; divide by the
    IMF-analytical value from
    ``calibrate_mean_mass_evolved(n_systems_drawn=...)`` first.

    A light Gaussian kernel (``smooth_sigma`` redshift bins, default 30 =
    dz 0.3 at step 0.01) suppresses residual Monte-Carlo wobble from the
    discrete COMPAS metallicity grid.  Pass ``smooth_sigma=0`` to disable.

    Never quote a local rate from the smoothed array: the kernel's
    reflective boundary folds the rising side of R(z) back onto z = 0
    and inflates the smoothed R(z=0) by ~30 percent at production sample
    sizes.  Compute with ``smooth_sigma=0`` wherever R(z=0) is read off,
    and smooth a display copy via ``smooth_rate_curve`` for plotting
    (the same artifact is guarded against in
    ``calibrate_mean_mass_evolved``).

    Parameters
    ----------
    p_draw : float
        COMPAS metallicity sampling density
        (= 1 / (max_logZ_COMPAS - min_logZ_COMPAS)).
    dPdlogZ : 2-D array, shape (n_z, n_metallicities)
        Pointwise metallicity PDF from
        ``FastCosmicIntegration.find_metallicity_distribution``.
    metallicities : 1-D array
        Metallicity grid on which ``dPdlogZ`` is evaluated.
    n_chunk : int
        Number of binaries per chunk; default 10_000 keeps the per-chunk
        allocation at ~80 MB for n_z ~ 1000.  Result is independent of
        ``n_chunk`` (regression-tested).
    """
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        find_formation_and_merger_rates,
    )

    n_z = len(redshifts)
    if len(COMPAS_delay_times) == 0:
        return np.zeros(n_z)

    Z_arr = np.asarray(COMPAS_Z, dtype=float)
    delays = np.asarray(COMPAS_delay_times, dtype=float)
    w = np.asarray(COMPAS_weights, dtype=float)
    n_total = len(Z_arr)

    total = np.zeros(n_z)
    for s in range(0, n_total, n_chunk):
        e = min(s + n_chunk, n_total)
        _, mr = find_formation_and_merger_rates(
            n_binaries=e - s,
            redshifts=redshifts,
            times=times,
            time_first_SF=time_first_SF,
            n_formed=n_formed,
            dPdlogZ=dPdlogZ,
            metallicities=metallicities,
            p_draw_metallicity=p_draw,
            COMPAS_metallicites=Z_arr[s:e],
            COMPAS_delay_times=delays[s:e],
            COMPAS_weights=w[s:e],
        )
        total += mr.sum(axis=0)

    if smooth_sigma > 0:
        total = _gaussian_filter1d(total, sigma=smooth_sigma)
    return total


def smooth_rate_curve(R, sigma=30):
    """Display-only Gaussian smoothing of an R(z) curve.

    Applies the same kernel ``compute_merger_rate`` uses internally.
    The reflective boundary inflates the smoothed value at index 0
    (z = 0) on any rising R(z), so use this only on plotted copies and
    quote R(z=0) from the unsmoothed array.
    """
    return _gaussian_filter1d(np.asarray(R, dtype=float), sigma=sigma)


# ═══════════════════════════════════════════════════════════════════════════
# GW detection-rate via FCI selection effects
# ═══════════════════════════════════════════════════════════════════════════
def _build_snr_detection_grids(sensitivity="O3", snr_threshold=8.0, Mc_max=300.0, eta_step=0.01):
    """Wrapper around FCI ``compute_snr_and_detection_grids``.

    Returns the SNR-vs-(M_c, eta) grid evaluated at 1 Mpc plus a 1-D
    detection-probability table indexed by SNR.  Builds them once per
    sensitivity choice; caching is the caller's responsibility (these
    arrays are < 1 MB and cheap to recompute).
    """
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        compute_snr_and_detection_grids,
    )

    # dco_type is a no-op for non-WD types in current FCI: the (M_c, eta)
    # grid is set by Mc_max=300 Msun, Mc_step=0.1, eta_max=0.25, eta_step=0.01
    # and is independent of dco_type.  The argument only fires a warning for
    # {WDWD, NSWD, WDBH} (no LVK sensitivity defined for those).  Our BNS
    # (M_c ~ 0.9-2.2, eta ~ 0.24-0.25) and BHNS (M_c ~ 1.5-30, eta ~ 0.05-0.22)
    # samples sit well inside the grid, so passing "BHBH" gives identical
    # output to "NSNS" or "BHNS".  Revisit if the FCI pin in environment.yml
    # changes the function's contract, or if a future sweep pushes M_c above
    # 300 Msun (IMBH territory).
    return compute_snr_and_detection_grids(
        dco_type="BHBH",
        sensitivity=sensitivity,
        snr_threshold=snr_threshold,
        Mc_max=Mc_max,
        eta_step=eta_step,
    )


def detected_rate(
    redshifts,
    times,
    time_first_SF,
    n_formed,
    p_draw,
    dPdlogZ,
    metallicities,
    m1,
    m2,
    COMPAS_Z,
    COMPAS_delay_times,
    COMPAS_weights,
    distances,
    n_redshifts_detection,
    sensitivity="O3",
    snr_threshold=8.0,
    n_chunk=10_000,
):
    """Detector-frame detected merger rate density [Gpc^-3 yr^-1] vs redshift.

    Wraps FCI's ``compute_snr_and_detection_grids`` and
    ``find_detection_probability`` and folds them into the same chunked
    accumulator that ``compute_merger_rate`` uses.  At each chunk:

        merger_rate[i, j] = find_formation_and_merger_rates(...)
        det_prob[i, j]    = find_detection_probability(...)
        contrib[i, j]     = merger_rate[i, j] * det_prob[i, j]

    summed over the binary axis, returning a ``(n_redshifts_detection,)``
    array.  This is the LVK-detectable subset of the intrinsic merger
    rate as a function of redshift, given the chosen sensitivity curve.

    The shape ``(n_binaries, n_redshifts_detection)`` of FCI's
    ``find_detection_probability`` is shorter than the full
    ``(n_binaries, n_z)`` merger-rate matrix (FCI defaults
    ``max_redshift_detection = 1.0``, well below ``max_redshift = 10.0``);
    detection probabilities at z > z_detection_max are not defined and
    we cap the output at ``n_redshifts_detection`` accordingly.

    Parameters
    ----------
    m1, m2 : 1-D array
        Component masses [Msun] for chirp-mass / symmetric-mass-ratio.
    distances : 1-D array, shape (n_z,)
        Luminosity distance at each redshift [Mpc] (from
        ``calculate_redshift_related_params``).
    n_redshifts_detection : int
        Index in ``redshifts`` up to which detection probabilities are
        evaluated (``int(max_redshift_detection / redshift_step)``).
    sensitivity : {'O1', 'O3', 'design'}
        LIGO sensitivity curve.  Default ``'O3'`` matches the post-O3
        observed rate references in ``OBSERVED_SGRB_RATES``.  ``'O1'``
        and ``'design'`` are also accepted by FCI.
    snr_threshold : float
        Network-SNR detection threshold; default 8.0 (Finn-Chernoff).
    n_chunk : int
        Number of binaries per chunk.

    Returns
    -------
    R_det : 1-D array, shape (n_redshifts_detection,)
        Detected merger rate density at the source-frame redshift
        slice; multiply by ``shell_volumes / (1 + z)`` and sum to
        recover the absolute detected count per detector-frame year.
    """
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        find_detection_probability,
        find_formation_and_merger_rates,
    )

    from grb_physics import chirp_mass

    n_binaries = len(m1)
    if n_binaries == 0:
        return np.zeros(n_redshifts_detection)

    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    Z_arr = np.asarray(COMPAS_Z, dtype=float)
    delays = np.asarray(COMPAS_delay_times, dtype=float)
    w = np.asarray(COMPAS_weights, dtype=float)

    Mc = chirp_mass(m1, m2)
    eta = (m1 * m2) / (m1 + m2) ** 2

    snr_grid_at_1Mpc, det_prob_from_snr = _build_snr_detection_grids(
        sensitivity=sensitivity,
        snr_threshold=snr_threshold,
    )

    total = np.zeros(n_redshifts_detection)
    for s in range(0, n_binaries, n_chunk):
        e = min(s + n_chunk, n_binaries)
        _, mr_chunk = find_formation_and_merger_rates(
            n_binaries=e - s,
            redshifts=redshifts,
            times=times,
            time_first_SF=time_first_SF,
            n_formed=n_formed,
            dPdlogZ=dPdlogZ,
            metallicities=metallicities,
            p_draw_metallicity=p_draw,
            COMPAS_metallicites=Z_arr[s:e],
            COMPAS_delay_times=delays[s:e],
            COMPAS_weights=w[s:e],
        )
        det_prob_chunk = find_detection_probability(
            Mc=Mc[s:e],
            eta=eta[s:e],
            redshifts=redshifts,
            distances=distances,
            n_redshifts_detection=n_redshifts_detection,
            n_binaries=e - s,
            snr_grid_at_1Mpc=snr_grid_at_1Mpc,
            detection_probability_from_snr=det_prob_from_snr,
        )
        total += (mr_chunk[:, :n_redshifts_detection] * det_prob_chunk).sum(axis=0)

    return total


def per_system_rate_weights(
    z_target,
    redshifts,
    times,
    time_first_SF,
    n_formed,
    p_draw,
    dPdlogZ,
    metallicities,
    COMPAS_Z,
    COMPAS_delay_times,
    COMPAS_weights,
):
    """Per-system rate weights at z_target.

    Returns ``w_i(z_target) = n_formed(z_form) * dPdlogZ(z_form, Z_i) /
    p_draw * w_i^STROOPWAFEL`` per binary, where ``z_form`` is the
    formation redshift required to merge at ``z_target`` given the
    binary's delay time.  Used for weighted-Poisson 1-sigma envelopes
    (sigma_R = sqrt(sum_i w_i^2)) and for the per-class rate-weighted Z
    and chirp-mass histograms in Section 13.

    Mirrors the formation-rate column of FCI's
    ``find_formation_and_merger_rates`` (``np.digitize`` lookup against
    ``metallicities``), evaluated at the single formation time
    corresponding to ``z_target``.

    Parameters
    ----------
    dPdlogZ : 2-D array, shape (n_z, n_metallicities)
        Pointwise metallicity PDF from FCI's
        ``find_metallicity_distribution``.
    metallicities : 1-D array
        Metallicity grid that ``dPdlogZ`` is evaluated on.
    """
    n_z = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]

    if len(COMPAS_weights) == 0:
        return np.zeros(0)

    times_to_z = interp1d(times, redshifts)

    j_target = np.argmin(np.abs(redshifts - z_target))
    t_merge = times[j_target]
    t_min = max(time_first_SF, times.min())

    delays = np.asarray(COMPAS_delay_times, dtype=float)
    Z_arr = np.asarray(COMPAS_Z, dtype=float)
    w_all = np.asarray(COMPAS_weights, dtype=float)

    out = np.zeros(len(w_all))
    t_form = t_merge - delays
    valid = t_form >= t_min
    if not valid.any():
        return out

    idx = np.where(valid)[0]
    z_form = times_to_z(t_form[idx])
    z_idx_float = z_form / redshift_step
    z_lo = np.clip(np.floor(z_idx_float).astype(np.int64), 0, n_z - 1)
    z_hi = np.clip(z_lo + 1, 0, n_z - 1)
    frac = z_idx_float - np.floor(z_idx_float)

    # FCI uses np.digitize(Z, metallicities) for the column index; mirror that
    # here so subset rate weights match the per-binary formation-rate row of
    # find_formation_and_merger_rates exactly.
    cols = np.digitize(Z_arr[idx], metallicities)
    cols = np.clip(cols, 0, dPdlogZ.shape[1] - 1)

    dP_lo = dPdlogZ[z_lo, cols]
    dP_hi = dPdlogZ[z_hi, cols]
    w = w_all[idx]

    inv_p_draw = 1.0 / p_draw
    f_lo = n_formed[z_lo] * dP_lo * inv_p_draw * w
    f_hi = n_formed[z_hi] * dP_hi * inv_p_draw * w
    out[idx] = f_lo * (1.0 - frac) + f_hi * frac
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Per-population normalization (back-derived from upstream Neijssel anchor)
# ═══════════════════════════════════════════════════════════════════════════
def calibrate_mean_mass_evolved(
    redshifts,
    times,
    time_first_SF,
    COMPAS_Z,
    COMPAS_delay_times,
    COMPAS_weights,
    expected_local_rate,
    Z_min_COMPAS=None,
    Z_max_COMPAS=None,
):
    """MSSFR-independent MEAN_MASS_EVOLVED [Msun] back-derived at z = 0.

    The COMPAS Broekgaarden+ 2021 files store ``weights_intrinsic/w_000``,
    which is the cosmic-integration weight at the z = 0 slice computed by the
    upstream pipeline with Neijssel+ 2019 MSSFR parameters.  Summed across
    binaries, ``w_000`` gives the published intrinsic local merger rate
    density, ``expected_local_rate``.

    The simulation's total stellar mass evolved is a property of the
    sampler (IMF, mass cuts, binary fraction, STROOPWAFEL phases) and is
    therefore MSSFR-independent.  We recover it by running our own
    ``compute_merger_rate`` with Neijssel+ 2019 MSSFR (the same parameters
    the upstream pipeline used) and dividing the unsmoothed R(z = 0) by
    ``expected_local_rate``::

        MEAN_MASS_EVOLVED = R_unnorm(z = 0; Neijssel) / expected_local_rate

    The factor of MEAN_MASS_EVOLVED that comes out is the actual simulation
    mass and can be plugged into any subsequent ``compute_merger_rate`` call
    with a different MSSFR (Levina+ 2026 TNG100-1, ...).  The Neijssel
    parameters are used here only as a calibration tool, not as the
    science-path MSSFR.

    Note: STROOPWAFEL files do not store the total number of binaries that
    COMPAS evolved upstream (the file holds only the surviving DCOs, not the
    full draw count).  An IMF-analytical estimate
    ``N_drawn * star_forming_mass_per_binary`` therefore cannot be computed
    from the file alone, which is why this routine back-derives from the
    Neijssel anchor instead.

    Parameters
    ----------
    expected_local_rate : float
        Sum of ``weights_intrinsic/w_000`` from the Broekgaarden+ 2021
        HDF5 file (the published intrinsic local rate, Gpc^-3 yr^-1).
        Use ``grb_io.read_expected_local_rate``.
    Z_min_COMPAS, Z_max_COMPAS : float, optional
        COMPAS metallicity sampling range [linear, not log].  Defaults
        to ``(min(COMPAS_Z), max(COMPAS_Z))``.  Pass the simulation's
        full range (``METALLICITY_GRID[0]``, ``METALLICITY_GRID[-1]``)
        if a class subset has narrower coverage.

    Returns
    -------
    mean_mass_evolved : float
        Total stellar mass [Msun] formed in the simulation.
    """
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        find_metallicity_distribution,
        find_sfr,
    )

    if Z_min_COMPAS is None:
        Z_min_COMPAS = float(np.min(COMPAS_Z))
    if Z_max_COMPAS is None:
        Z_max_COMPAS = float(np.max(COMPAS_Z))

    sfr_nei = find_sfr(redshifts)  # Madau and Dickinson 2014, Neijssel+19 fit
    dPdlogZ_nei, mets_nei, p_draw_nei = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(Z_min_COMPAS),
        max_logZ_COMPAS=np.log(Z_max_COMPAS),
    )  # Neijssel+19 default mu0/muz/sigma_0/sigma_z/alpha; matches w_000 anchor.

    rate_unnorm = compute_merger_rate(
        redshifts,
        times,
        time_first_SF,
        sfr_nei,
        p_draw_nei,
        dPdlogZ_nei,
        mets_nei,
        COMPAS_Z,
        COMPAS_delay_times,
        COMPAS_weights,
        smooth_sigma=0,
    )
    return float(rate_unnorm[0] / expected_local_rate)


# ═══════════════════════════════════════════════════════════════════════════
# Formation efficiency (per metallicity bin)
# ═══════════════════════════════════════════════════════════════════════════
# Right edge of the physically meaningful metallicity axis, log10(Z / Zsun).
# The COMPAS STROOPWAFEL prior extends to Z = 0.03 ~ 2.1 Zsun, but the
# Levina+ 2026 (arXiv:2601.20202) Azzalini skew-log-normal fit to TNG100-1
# (Table 1) places the bulk of IMF-weighted cosmic star formation below Zsun;
# the COMPAS Model A merging sample retains < 1 percent of its STROOPWAFEL
# weight above Zsun in the per-bin tallies used by ``formation_efficiency``.
# Plotting that high-Z tail without truncation gives the per-class efficiency
# curves a slope that is dominated by importance-sampling noise rather than
# the underlying binary physics (Broekgaarden+ 2021 arXiv:2103.02608 Fig. 5
# rolls over the same way for the same reason).
LOGZ_MAX_PHYSICAL: float = 0.0


def kish_effective_sample_size(weights):
    """Kish (1965) effective sample size for an importance-sampled set.

    n_eff = (sum w)^2 / sum(w^2); equals N for uniform weights and
    collapses to ~1 when a single entry dominates the sum.  Use as the
    small-N diagnostic in any per-Z reduction over the STROOPWAFEL
    sample: a bin with N_raw >= 50 but n_eff ~ 2 is statistically
    indistinguishable from a single-system bin and should be masked.

    Reference: Kish (1965), Survey Sampling, Wiley, Sec. 1.6.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    if np.any(w < 0):
        raise ValueError("Kish ESS requires non-negative weights")
    s1 = w.sum()
    s2 = np.sum(w * w)
    if s2 == 0.0:
        return 0.0
    return float(s1 * s1 / s2)


def metallicity_prior_mass_fraction(metallicityGrid, z_min=None, z_max=None):
    """Prior star-forming-mass fraction per metallicity grid point.

    COMPAS draws birth metallicity uniformly in ln Z over the sampling
    range [z_min, z_max]; FastCosmicIntegration encodes this as
    ``p_draw = 1 / (ln z_max - ln z_min)``.  The star-forming mass
    evolved at grid point Z_i is therefore proportional to the width of
    its log-Z cell, and the fraction of the simulation's total mass at
    Z_i is::

        f_i = Delta(ln Z)_i / (ln z_max - ln z_min),   sum_i f_i = 1

    Cell edges sit at the midpoints between adjacent grid points in
    ln Z, with the outer edges clamped to ln z_min / ln z_max.  Multiply
    by a population's total ``mean_mass_evolved`` to get the
    per-metallicity star-forming mass M_SF(Z_i), the denominator of the
    intensive formation efficiency in Broekgaarden+ 2021
    (arXiv:2103.02608, Fig. 5).

    Parameters
    ----------
    metallicityGrid : 1-D array
        Metallicity grid (e.g. the 53-point COMPAS grid).  Need not be
        sorted; the result matches the input ordering.
    z_min, z_max : float, optional
        COMPAS metallicity sampling bounds [linear].  Default to the
        grid min / max.

    Returns
    -------
    f : 1-D array, same shape / ordering as ``metallicityGrid``, summing to 1.
    """
    Z = np.asarray(metallicityGrid, dtype=float)
    if Z.ndim != 1 or Z.size == 0:
        raise ValueError("metallicityGrid must be a non-empty 1-D array")
    order = np.argsort(Z)
    lnZ = np.log(Z[order])
    lo = float(np.log(z_min)) if z_min is not None else lnZ[0]
    hi = float(np.log(z_max)) if z_max is not None else lnZ[-1]
    if hi <= lo:
        raise ValueError("require z_max > z_min")
    edges = np.empty(lnZ.size + 1)
    edges[1:-1] = 0.5 * (lnZ[:-1] + lnZ[1:])
    edges[0] = lo
    edges[-1] = hi
    f_sorted = np.diff(edges) / (hi - lo)
    f = np.empty_like(f_sorted)
    f[order] = f_sorted
    return f


def formation_efficiency(
    metallicityGrid, Z_all, w_all, masks=None, mean_mass_evolved=None, z_min=None, z_max=None
):
    """
    Intensive formation efficiency eta(Z) per metallicity bin.

    Returns the number of merging systems formed per solar mass of star
    formation *at each metallicity*, the quantity Broekgaarden+ 2021
    (arXiv:2103.02608, Fig. 5) plots::

        eta(Z_i) = (sum of STROOPWAFEL weights at Z_i) / M_SF(Z_i)

    The per-metallicity star-forming mass is the population total split
    by the COMPAS uniform-in-ln Z sampling prior,
    ``M_SF(Z_i) = mean_mass_evolved * f_i`` with ``f_i`` from
    ``metallicity_prior_mass_fraction``.  Dividing by the per-Z mass
    rather than the scalar total makes eta(Z) intensive (independent of
    how the grid is binned) and lands it on the published
    Gpc-scale formation-efficiency axis; using the bare total instead
    suppresses every bin by ~1/f_i (the number of sampled metallicities).

    Parameters
    ----------
    metallicityGrid : 1-D array
        Grid of metallicity values (e.g. 53-point COMPAS grid).
    Z_all : 1-D array
        Metallicity of each merging system.
    w_all : 1-D array
        STROOPWAFEL weight of each merging system.
    masks : dict of boolean arrays, optional
        Sub-population masks (e.g. {'Short': mask_s, 'Long': mask_l}).
        If None, only total efficiency is returned.  Per-mask curves are
        additive (they share one M_SF(Z_i) denominator) and sum to
        'total' bin by bin.
    mean_mass_evolved : float
        Total stellar mass evolved in the simulation [Msun].
        Required; use ``calibrate_mean_mass_evolved()`` to derive it
        for each population (BNS, BHNS) separately.
    z_min, z_max : float, optional
        COMPAS metallicity sampling bounds passed through to
        ``metallicity_prior_mass_fraction``.

    Returns
    -------
    dict of 1-D arrays keyed by mask name, plus 'total'.
    """
    if mean_mass_evolved is None:
        raise ValueError(
            "mean_mass_evolved must be provided; use calibrate_mean_mass_evolved() to derive it"
        )
    f = metallicity_prior_mass_fraction(metallicityGrid, z_min=z_min, z_max=z_max)
    M_SF = mean_mass_evolved * f  # per-metallicity star-forming mass [Msun]
    unique_Z = np.unique(Z_all)
    result = {"total": np.zeros(len(metallicityGrid))}
    if masks is not None:
        for name in masks:
            result[name] = np.zeros(len(metallicityGrid))

    for i, Z in enumerate(metallicityGrid):
        if Z in unique_Z:
            maskZ = Z_all == Z
            result["total"][i] = np.sum(w_all[maskZ]) / M_SF[i]
            if masks is not None:
                for name, m in masks.items():
                    result[name][i] = np.sum(w_all[maskZ & m]) / M_SF[i]

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BH spin marginalization
# ═══════════════════════════════════════════════════════════════════════════
def marginalize(rate_dict, weights):
    """Weighted average over spin values: sum(w_a * rate[a]).

    Prefer ``marginalize_bh_spin`` for continuous or callable PDFs.
    """
    return sum(weights[a] * rate_dict[a] for a in rate_dict)


def marginalize_bh_spin(rate_per_spin, p_chi, spin_grid=None):
    """BH-spin-marginalised rate: integral over the BH spin PDF.

    Two input modes are accepted:

    - ``rate_per_spin`` is a dict ``{chi: rate_array_or_scalar}`` and
      ``p_chi`` is a dict with the same keys carrying weights that sum
      to 1.  This matches the existing notebook ``marginalize(...)``
      pattern and is preserved for back-compatibility.
    - ``rate_per_spin`` is a 1-D array indexed by ``spin_grid`` and
      ``p_chi`` is either a 1-D array of weights (same length as
      ``spin_grid``) or a callable ``p_chi(chi)`` that returns the
      probability density at each grid point.  Useful when
      integrating against an explicit prior PDF.

    Parameters
    ----------
    rate_per_spin : dict or array
        Per-spin rates.  See above for input modes.
    p_chi : dict, array, or callable
        Spin-prior weights (sum to 1) or PDF.
    spin_grid : 1-D array, optional
        Required when ``rate_per_spin`` is an array.

    Returns
    -------
    rate_marginalised : float or ndarray
        Same shape as the elements of ``rate_per_spin``.

    Caveat: this routine does not model channel-dependent spin
    evolution (Fuller and Ma 2019 efficient AM transport, post-SN
    tidal spin-up, Bavera+ 2020 accretion).  The caller supplies the
    appropriate ``p_chi`` for the channel of interest.
    """
    if isinstance(rate_per_spin, dict):
        if not isinstance(p_chi, dict):
            raise TypeError(
                "When rate_per_spin is a dict, p_chi must also be a dict with matching keys."
            )
        # Sum weighted by p_chi[chi]; np.asarray handles both scalar
        # and array elements uniformly.
        return sum(p_chi[chi] * np.asarray(rate_per_spin[chi]) for chi in rate_per_spin)

    rate_arr = np.asarray(rate_per_spin)
    if spin_grid is None:
        raise ValueError("spin_grid is required when rate_per_spin is an array.")
    spin_grid = np.asarray(spin_grid, dtype=float)
    if callable(p_chi):
        weights = np.asarray([p_chi(chi) for chi in spin_grid], dtype=float)
        # Trapezoidal integration over the spin grid for a continuous PDF.
        # Falls back to discrete sum if the grid is single-point.
        if spin_grid.size > 1:
            integrand = weights[:, None] * rate_arr if rate_arr.ndim > 1 else weights * rate_arr
            # numpy>=2 prefers ``trapezoid``; fall back to ``trapz`` for 1.x.
            trap = getattr(np, "trapezoid", None) or np.trapz
            return trap(integrand, spin_grid, axis=0)
        return weights[0] * rate_arr[0]
    weights = np.asarray(p_chi, dtype=float)
    if weights.shape != spin_grid.shape:
        raise ValueError(
            f"p_chi shape {weights.shape} does not match spin_grid shape {spin_grid.shape}."
        )
    if rate_arr.ndim == 1:
        return float(np.sum(weights * rate_arr))
    return np.sum(weights[:, None] * rate_arr, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# BHNS spin-orbit misalignment population correction
# ═══════════════════════════════════════════════════════════════════════════
def apply_bhns_misalignment(rate_bhns, factor=MISALIGNMENT_SYSTEMATIC_FACTOR):
    """BHNS GRB rate * ``MISALIGNMENT_SYSTEMATIC_FACTOR``.

    Population-level half-suppression motivated by Fragos+ 2010 and
    Kawaguchi+ 2015.  Do not also use ``effective_aligned_spin`` on the
    same sample or the suppression double-counts.
    """
    return np.asarray(rate_bhns) * factor


# ═══════════════════════════════════════════════════════════════════════════
# BNS jet breakout efficiency (Pais et al. 2025)
# ═══════════════════════════════════════════════════════════════════════════
# Remnant -> delay-time window mapping.  Prompt-collapse remnants form a
# clean polar funnel within ~0.1 s, so the jet launches with a short t_d;
# long-lived / short-lived HMNS remnants pollute the funnel until they
# collapse, so the jet launches only after the HMNS lifetime.  The
# (t_d/t_e)^2 term in Pais Eq. (4) then suppresses HMNS-class breakout.
_HMNS_CLASS_LABELS = frozenset({"sbGRB + blue KN", "lbGRB + red KN (HMNS)"})


def breakout_fraction_bns(
    class_masks,
    M_disk,
    M_ej,
    weights,
    theta_j_by_class=None,
    n_draws=512,
    rng=None,
):
    """Remnant-conditioned BNS jet-breakout fraction with a band, per class.

    For each of ``n_draws`` Monte-Carlo realisations, draws the uncertain
    jet/ejecta parameters (engine time ``t_e``, jet efficiency
    ``eps_jet``, ejecta velocity ``v_ej``) log-uniformly from their cited
    ranges and the launch delay ``t_d`` from the class-appropriate window
    (prompt-collapse vs HMNS), evaluates the Pais et al. (2025) Eq. (4)
    breakout test (with the Eq. 3 collimation gate) per system, and
    reduces to a STROOPWAFEL-weighted breakout fraction for each Gottlieb
    (2024) class.  The returned band is the 16/50/84 percentile over the
    draw distribution.

    Parameters
    ----------
    class_masks : dict[str, 1-D bool array]
        Gottlieb (2024) class membership, e.g. from
        ``grb_classify.classify_bns_2024``.  Non-mask entries are ignored.
    M_disk : 1-D array
        Per-system BNS accretion-disk mass [Msun]
        (``grb_physics.bns_disk_mass``).
    M_ej : 1-D array
        Per-system ejecta mass obstructing the jet [Msun]
        (``grb_physics.bns_dynamical_ejecta``, optionally plus disk wind).
    weights : 1-D array
        STROOPWAFEL weights, aligned with ``M_disk`` / ``M_ej``.
    theta_j_by_class : dict[str, float], optional
        Per-class jet half-opening angle [deg].  When None, the sbGRB
        class uses ``CLASS_THETA_J['sbGRB']['fid']`` and the three lbGRB
        classes use ``CLASS_THETA_J['lbGRB']['fid']``.
    n_draws : int, optional
        Number of Monte-Carlo draws over the uncertain inputs.
    rng : np.random.Generator, optional
        Defaults to ``np.random.default_rng(0)`` for reproducibility.

    Returns
    -------
    dict[str, dict]
        ``{class_label: {'med': float, 'lo': float, 'hi': float}}``.  A
        class with zero weighted population yields NaN entries.

    Notes
    -----
    The Pais prefactors are calibrated to a single GW170817-like Kiuchi
    et al. (2023) run; ``M_ej`` scales per system but the absolute
    fractions are accurate to a factor of a few.  ``bns_disk_mass`` is
    knife-edge sensitive near R_1.4 ~ 12 km (see its docstring), so
    average over an EOS prior or pass explicit radii when computing
    ``M_disk`` for a science result.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    M_disk = np.asarray(M_disk, dtype=float)
    M_ej = np.asarray(M_ej, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if theta_j_by_class is None:
        theta_j_by_class = {
            "sbGRB + blue KN": CLASS_THETA_J["sbGRB"]["fid"],
            "lbGRB + red KN (HMNS)": CLASS_THETA_J["lbGRB"]["fid"],
            "lbGRB + red KN (disk)": CLASS_THETA_J["lbGRB"]["fid"],
            "Faint lbGRB": CLASS_THETA_J["lbGRB"]["fid"],
        }

    class_keys = [
        k for k, v in class_masks.items() if isinstance(v, np.ndarray) and v.dtype == bool
    ]

    def _loguniform(lo, hi, size):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))

    # Per-draw scalars shared across systems within a draw.
    t_e = _loguniform(T_E_RANGE[0], T_E_RANGE[1], n_draws)
    eps = _loguniform(EPS_JET * 0.5, EPS_JET * 2.0, n_draws)
    v_ej = rng.uniform(V_EJ_BNS * 0.8, V_EJ_BNS * 1.2, size=n_draws)
    t_d_prompt = _loguniform(T_D_PROMPT_RANGE[0], T_D_PROMPT_RANGE[1], n_draws)
    t_d_hmns = _loguniform(GOTTLIEB25_T_HMNS_RANGE[0], GOTTLIEB25_T_HMNS_RANGE[1], n_draws)

    out = {}
    for label in class_keys:
        mask = np.asarray(class_masks[label], dtype=bool)
        w = weights[mask]
        w_tot = float(w.sum())
        if w_tot <= 0.0:
            out[label] = {"med": float("nan"), "lo": float("nan"), "hi": float("nan")}
            continue

        md = M_disk[mask]
        mej = M_ej[mask]
        theta = np.radians(theta_j_by_class.get(label, CLASS_THETA_J["lbGRB"]["fid"]))
        t_d_draws = t_d_hmns if label in _HMNS_CLASS_LABELS else t_d_prompt

        fracs = np.empty(n_draws, dtype=float)
        for d in range(n_draws):
            E_j = jet_energy_from_disk(md, eps_jet=eps[d])
            E_ej_cone = ejecta_energy_in_cone(mej, theta, v_ej=v_ej[d])
            broke = jet_breaks_out_pais(
                E_j, E_ej_cone, theta, t_d_draws[d], t_e[d], require_collimation=True
            )
            fracs[d] = float(w[broke].sum() / w_tot)

        lo, med, hi = np.percentile(fracs, [16, 50, 84])
        out[label] = {"med": float(med), "lo": float(lo), "hi": float(hi)}
    return out


def apply_bns_jet_breakout(rate, f_breakout):
    """BNS GRB rate * jet-breakout efficiency ``f_breakout``.

    Population-level multiplier; the intended chain is
    ``R_intrinsic -> apply_bns_jet_breakout -> beamed_rate``.  Breakout is
    independent of beaming and of the BHNS spin-orbit misalignment factor,
    so do not fold those into ``f_breakout`` or the suppression
    double-counts.
    """
    return np.asarray(rate) * np.asarray(f_breakout)


# ═══════════════════════════════════════════════════════════════════════════
# Fraction helpers
# ═══════════════════════════════════════════════════════════════════════════
def frac4(r_s, r_l, r_bs, r_bl):
    """Four-component percentage fractions, NaN where total is zero."""
    tot = np.where((r_s + r_l + r_bs + r_bl) > 0, r_s + r_l + r_bs + r_bl, np.nan)
    return r_s / tot * 100, r_l / tot * 100, r_bs / tot * 100, r_bl / tot * 100


def rate_label(val):
    """Format a rate value: integer for >= 1, two decimals otherwise."""
    return f"{val:,.0f}" if val >= 1 else f"{val:.2f}"


def weighted_frac_err(w, mask, w_total=None):
    """Class fraction and its weighted-Poisson 1-sigma error.

    For a class fraction estimated from a STROOPWAFEL-weighted sample,
    ``f = sum(w[mask]) / w_total``, the weighted-Poisson uncertainty on
    the numerator is the weighted analogue of sqrt(N),
    ``sigma_count = sqrt(sum w[mask]^2)`` (the per-bin rule in the project
    figure standard).  Holding the denominator fixed at the full-population
    weight sum,

        sigma_f = sqrt(sum w[mask]^2) / w_total.

    Parameters
    ----------
    w : 1-D array
        STROOPWAFEL weights of the full population.
    mask : 1-D bool array
        Class membership over the same population.
    w_total : float, optional
        Denominator weight sum.  Defaults to ``sum(w)``, appropriate when
        ``mask`` selects a subclass of the population carried in ``w``.

    Returns
    -------
    (f, sigma_f) : tuple of float
        Class fraction and its 1-sigma weighted-Poisson error.  Both NaN
        when ``w_total`` is zero or non-positive.
    """
    w_arr = np.asarray(w, dtype=float)
    m = np.asarray(mask, dtype=bool)
    denom = float(w_arr.sum()) if w_total is None else float(w_total)
    if denom <= 0.0:
        return float("nan"), float("nan")
    wc = w_arr[m]
    f = float(wc.sum() / denom)
    sigma = float(np.sqrt(np.sum(wc * wc)) / denom)
    return f, sigma


# ═══════════════════════════════════════════════════════════════════════════
# Kroupa IMF verification
# ═══════════════════════════════════════════════════════════════════════════
def kroupa_imf(m):
    """Un-normalized Kroupa (2001) three-segment IMF (scalar or array).

    Power-law exponents and breakpoints from Kroupa (2001), MNRAS 322,
    231, Eq. 2: alpha_1 = 0.3 below 0.08 Msun, alpha_2 = 1.3 between 0.08
    and 0.5 Msun, alpha_3 = 2.3 above 0.5 Msun (= Salpeter 1955).
    Continuity coefficients (1.0, 0.08, 0.04) match the piecewise
    definition in Eq. 2 so the segments join without a step.
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    result = np.piecewise(
        m,
        [m < 0.08, (m >= 0.08) & (m < 0.5), m >= 0.5],
        [
            lambda m: m ** (-0.3),  # Kroupa 2001 Eq. 2: alpha_1 = 0.3
            lambda m: 0.08 * m ** (-1.3),  # Kroupa 2001 Eq. 2: alpha_2 = 1.3
            lambda m: 0.04 * m ** (-2.3),
        ],
    )  # Kroupa 2001 Eq. 2: alpha_3 = 2.3
    return result.item() if result.size == 1 else result


def verify_mean_mass_evolved(
    m_lo_full=0.01, m_hi_full=200.0, m_lo_prim=5.0, m_hi_prim=150.0, mean_mass_evolved=None
):
    """Analytically verify a MEAN_MASS_EVOLVED value via Kroupa IMF.

    Parameters
    ----------
    mean_mass_evolved : float
        Value to check.  Use ``calibrate_mean_mass_evolved()`` to derive
        per-population values.
    """
    if mean_mass_evolved is None:
        raise ValueError(
            "mean_mass_evolved must be provided; use "
            "calibrate_mean_mass_evolved() to derive per-population values"
        )

    total_mass, _ = quad(lambda m: m * kroupa_imf(m), m_lo_full, m_hi_full)
    total_number, _ = quad(kroupa_imf, m_lo_full, m_hi_full)
    mean_star_mass = total_mass / total_number

    n_primary, _ = quad(kroupa_imf, m_lo_prim, m_hi_prim)
    f_primary = n_primary / total_number

    mass_per_drawn = mean_star_mass / f_primary

    return {
        "mean_star_mass": mean_star_mass,
        "f_primary": f_primary,
        "mass_per_drawn_primary": mass_per_drawn,
        "N_sim_implied": mean_mass_evolved / mass_per_drawn,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EOS sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════════
def compute_eos_sensitivity(
    m1, m2, weights, eos_models=None, k_thresh=None, q_thresh=None, hmns_factor=None
):
    """Per-EOS Gottlieb (2024) class fractions for a single BNS sample.

    Sweeps each EOS in ``eos_models`` by passing its ``M_TOV`` to
    ``classify_bns_2024`` together with the coupling rule
    ``M_thresh = k_thresh * M_TOV`` (the EOS sweep coherence invariant).
    Returns the STROOPWAFEL-weighted fraction of systems in each of the
    four Gottlieb (2024) classes.

    Parameters
    ----------
    m1, m2 : 1-D array
        Component masses [Msun] (any ordering).
    weights : 1-D array
        STROOPWAFEL weights aligned with ``m1`` / ``m2``.
    eos_models : dict, optional
        Sub-dict of ``grb_physics.EOS_MODELS`` (defaults to all four:
        APR4, SFHo, LS220, DD2).  Each value must carry ``M_TOV`` and
        optionally ``R_1p4``.
    k_thresh : float, optional
        Ratio ``M_thresh / M_TOV`` to apply to every EOS.  Defaults to
        ``grb_physics.K_THRESH_DEFAULT`` (1.27).
    q_thresh : float, optional
        Mass-ratio threshold; defaults to ``grb_physics.Q_THRESH_BNS``.
    hmns_factor : float, optional
        Long-/short-lived HMNS split multiplier; defaults to
        ``grb_physics.HMNS_FACTOR_DEFAULT``.

    Returns
    -------
    pandas.DataFrame
        Indexed by EOS name with columns
        ``[M_TOV, R_1p4, M_thresh, hmns_split, total_weight]`` plus
        one column per Gottlieb class carrying the weighted fraction
        (in [0, 1]).  The class fractions in each row sum to 1.
    """
    import pandas as pd

    from grb_classify import classify_bns_2024
    from grb_physics import (
        EOS_MODELS as _EOS,
    )
    from grb_physics import (
        HMNS_FACTOR_DEFAULT,
        K_THRESH_DEFAULT,
        Q_THRESH_BNS,
    )

    if eos_models is None:
        eos_models = _EOS
    if k_thresh is None:
        k_thresh = K_THRESH_DEFAULT
    if q_thresh is None:
        q_thresh = Q_THRESH_BNS
    if hmns_factor is None:
        hmns_factor = HMNS_FACTOR_DEFAULT

    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    w = np.asarray(weights, dtype=float)
    w_total = float(w.sum())
    if w_total <= 0:
        raise ValueError("weights sum to zero; cannot compute fractions.")

    rows = []
    for name, eos in eos_models.items():
        m_tov = float(eos["M_TOV"])
        cls = classify_bns_2024(
            m1,
            m2,
            m_tov=m_tov,
            m_thresh=None,
            k_thresh=k_thresh,
            q_thresh=q_thresh,
            hmns_factor=hmns_factor,
        )
        row = {
            "EOS": name,
            "M_TOV": m_tov,
            "R_1p4": float(eos.get("R_1p4", np.nan)),
            "M_thresh": k_thresh * m_tov,
            "hmns_split": hmns_factor * m_tov,
            "total_weight": w_total,
        }
        for label, mask in cls.items():
            row[label] = float(w[mask].sum() / w_total)
        rows.append(row)
    return pd.DataFrame(rows).set_index("EOS")


# ═══════════════════════════════════════════════════════════════════════════
# M_crit sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════════
def mcrit_sweep(M_tot, q, w_all, M_crit_range=None, q_thresh=1.2):
    """Weighted fractions of Short-I, Short-II, Long vs M_crit.

    Returns arrays (frac_short_I, frac_short_II, frac_long) each of
    shape (len(M_crit_range),).
    """
    if M_crit_range is None:
        # Bauswein+ 2013 (PRD 88, 043009, Eq. 6) gives EOS-dependent
        # M_thresh in the range ~2.6-3.0 Msun for soft-to-stiff EOSs
        # (APR4 -> DD2).  We extend the lower edge to 2.3 to bracket the
        # Raaijmakers+ 2021 (arXiv:2105.06981) M_TOV posterior tail and
        # the upper edge to 3.5 to cover stiff Read+ 2009 EOSs.  50 points
        # is dense enough to resolve the steep frac transitions near the
        # fiducial M_thresh ~ 2.8 Msun.
        M_crit_range = np.linspace(2.3, 3.5, 50)

    w_tot = np.sum(w_all)
    frac_I, frac_II, frac_L = [], [], []
    for Mc in M_crit_range:
        s_I = M_tot < Mc
        s_II = (M_tot >= Mc) & (q < q_thresh)
        lon = (M_tot >= Mc) & (q >= q_thresh)
        frac_I.append(np.sum(w_all[s_I]) / w_tot)
        frac_II.append(np.sum(w_all[s_II]) / w_tot)
        frac_L.append(np.sum(w_all[lon]) / w_tot)

    return (np.array(frac_I), np.array(frac_II), np.array(frac_L), M_crit_range)


# ═══════════════════════════════════════════════════════════════════════════
# Beaming correction
# ═══════════════════════════════════════════════════════════════════════════
CLASS_THETA_J = {
    "sbGRB": {"lo": 10.0, "fid": 13.0, "hi": 16.0},
    "lbGRB": {"lo": 5.0, "fid": 6.5, "hi": 8.0},
}
"""Class-dependent jet half-opening angles [deg].

sbGRB: fiducial prior range, NOT directly the Fong+ 2015 (ApJ 815, 102)
measurement.  Fong+ 2015 report a median theta_j = 16 +/- 10 deg over
11 sGRBs (so a symmetric 1-sigma band would extend ~6 to 26 deg).  The
narrower 10 to 16 deg band used here is informed by the Beniamini and
Nakar (2019, MNRAS 482, 5430, arXiv:1812.02194) structured-jet
analysis of GW170817 and the population reanalysis they recommend
(typical core opening angles closer to 10 deg with structured wings).
lbGRB: Gottlieb (2023) argues MAD-powered BH jets from lbGRB sources
have narrower collimation than HMNS-powered sbGRB jets.
"""


def check_dPdlogZ_normalization(dPdlogZ, metallicities, rtol=0.05):
    """Verify dP/dlogZ integrates to ~1 at every redshift slice.

    Convention: ``dPdlogZ`` is dP/d(ln Z) (natural log), consistent with
    COMPAS ``find_metallicity_distribution``.  The integration uses
    Delta(ln Z) = np.diff(np.log(Z)), NOT Delta(log10 Z).
    If your dPdlogZ is per d(log10 Z), multiply by ln(10) first.

    Returns the per-slice integral array.  Raises ValueError if any
    slice deviates by more than *rtol* from unity.
    """
    dlogZ = np.diff(np.log(metallicities))
    dlogZ = np.append(dlogZ, dlogZ[-1])
    norm = (dPdlogZ * dlogZ[None, :]).sum(axis=1)
    if np.any(np.abs(norm - 1.0) > rtol):
        bad = np.where(np.abs(norm - 1.0) > rtol)[0]
        raise ValueError(
            f"dPdlogZ integral deviates from 1 at {len(bad)} z-slices "
            f"(range {norm[bad].min():.4f}-{norm[bad].max():.4f})"
        )
    return norm


def observed_frame_rate(rate_intrinsic_z, redshifts):
    """Detector-frame rate density: R_det(z) = R_int(z) / (1 + z).

    ``compute_merger_rate`` returns intrinsic source-frame rate density
    [Gpc^-3 yr^-1].  The (1 + z)^-1 factor is cosmological time
    dilation of the source-frame year to the detector frame; the
    comoving Gpc^-3 volume element is unchanged.
    """
    z = np.asarray(redshifts, dtype=float)
    R = np.asarray(rate_intrinsic_z, dtype=float)
    return R / (1.0 + z)


def beamed_class_comparison(
    rate_intrinsic_by_class, theta_j_deg_by_class=None, observed_by_class=None
):
    """Per-class table of intrinsic, beamed, and observed sGRB rates.

    Wraps ``beamed_rate`` and joins against ``OBSERVED_RATES_BY_CLASS``
    so each Gottlieb (2024) class is shown alongside the closest
    observational handle in a single DataFrame.

    Parameters
    ----------
    rate_intrinsic_by_class : dict[str, float]
        Intrinsic per-class local merger rate density at z = 0
        [Gpc^-3 yr^-1].  Keys are Gottlieb (2024) class labels
        ('sbGRB + blue KN', etc.).
    theta_j_deg_by_class : dict[str, float], optional
        Per-class jet half-opening angle [deg].  When None, derive from
        ``CLASS_THETA_J``: 'sbGRB' fid for the sbGRB class, 'lbGRB' fid
        for the three lbGRB classes.
    observed_by_class : dict, optional
        Per-class observed rate references.  Defaults to
        ``OBSERVED_RATES_BY_CLASS``.

    Returns
    -------
    pandas.DataFrame
        Indexed by Gottlieb class label with columns
        ``[R_intrinsic, theta_j_deg, f_beam, R_beamed, R_obs,
           R_obs_lo, R_obs_hi, reference]``.
    """
    import pandas as pd

    if observed_by_class is None:
        observed_by_class = OBSERVED_RATES_BY_CLASS

    if theta_j_deg_by_class is None:
        theta_j_deg_by_class = {
            "sbGRB + blue KN": CLASS_THETA_J["sbGRB"]["fid"],
            "lbGRB + red KN (HMNS)": CLASS_THETA_J["lbGRB"]["fid"],
            "lbGRB + red KN (disk)": CLASS_THETA_J["lbGRB"]["fid"],
            "Faint lbGRB": CLASS_THETA_J["lbGRB"]["fid"],
        }

    rows = []
    for label, R_int in rate_intrinsic_by_class.items():
        theta = theta_j_deg_by_class.get(label, np.nan)
        f_beam = 1.0 - np.cos(np.radians(theta)) if np.isfinite(theta) else np.nan
        R_beamed = R_int * f_beam if np.isfinite(f_beam) else np.nan
        obs = observed_by_class.get(label, {})
        rows.append(
            {
                "class": label,
                "R_intrinsic": float(R_int),
                "theta_j_deg": float(theta),
                "f_beam": float(f_beam) if np.isfinite(f_beam) else np.nan,
                "R_beamed": float(R_beamed) if np.isfinite(R_beamed) else np.nan,
                "R_obs": float(obs.get("R_obs", np.nan)),
                "R_obs_lo": float(obs.get("R_obs_lo", np.nan)),
                "R_obs_hi": float(obs.get("R_obs_hi", np.nan)),
                "reference": obs.get("reference", ""),
            }
        )
    return pd.DataFrame(rows).set_index("class")


def beamed_rate(rate_intrinsic, theta_j_deg):
    """``R_obs = R_intrinsic * (1 - cos(theta_j))``.

    Fong+ 2015 sGRB median is theta_j = 16 +/- 10 deg; the 10-16 deg
    fiducial band used in ``CLASS_THETA_J`` (Beniamini and Nakar 2019)
    gives ``f_beam ~ 0.015-0.04``.
    """
    theta_j = np.radians(theta_j_deg)
    f_beam = 1.0 - np.cos(theta_j)
    return rate_intrinsic * f_beam


def beamed_rate_mixed(rates_by_class, theta_j_deg_by_class):
    """Class-weighted observed-frame rate for a mixed-class population.

    Each class has its own ``theta_j``, so ``beamed_rate(R_total,
    <theta_j>)`` is the wrong scalar approximation.  This helper sums
    ``R_c * (1 - cos(theta_j_c))`` across classes.  Both dict keys must
    match exactly; missing entries raise ``KeyError``.
    """
    missing = set(rates_by_class) - set(theta_j_deg_by_class)
    if missing:
        raise KeyError(
            f"theta_j_deg_by_class is missing entries for {sorted(missing)}; "
            f"every rate class must have a corresponding jet angle."
        )
    total = 0.0
    for name, R in rates_by_class.items():
        total = total + beamed_rate(R, theta_j_deg_by_class[name])
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Observed sGRB rate reference data
# ═══════════════════════════════════════════════════════════════════════════
# All rates are *observed* (beaming-limited) local values [Gpc^-3 yr^-1].
OBSERVED_SGRB_RATES = {
    "Wanderman & Piran 2015": {
        "R_obs": 4.1,
        "R_obs_lo": 2.2,
        "R_obs_hi": 6.4,
        "note": "MNRAS 448, 3026; intrinsic ~270 at theta_j ~10 deg",
    },
    "Ghirlanda+ 2016": {
        "R_obs": 1.3,
        "R_obs_lo": 0.5,
        "R_obs_hi": 3.0,
        "note": "A&A 594, A84; Fermi/GBM, intrinsic ~200-700 after beaming",
    },
    "Colombo+ 2022": {
        "R_obs": 3.6,
        "R_obs_lo": 1.8,
        "R_obs_hi": 6.5,
        "note": "ApJ 937, 79; Fermi/GBM update",
    },
}


# Per-class observed-rate references for the beaming comparator.  No
# paper publishes a class-by-class rate; each entry is the closest
# observed (beaming-limited, local) handle for that class, and the
# 'caveat' field documents the mapping choice.
OBSERVED_RATES_BY_CLASS = {
    "sbGRB + blue KN": {
        "R_obs": OBSERVED_SGRB_RATES["Colombo+ 2022"]["R_obs"],
        "R_obs_lo": OBSERVED_SGRB_RATES["Colombo+ 2022"]["R_obs_lo"],
        "R_obs_hi": OBSERVED_SGRB_RATES["Colombo+ 2022"]["R_obs_hi"],
        "reference": "Colombo+ 2022, ApJ 937, 79",
        "caveat": "Observed sGRB rate; sbGRBs dominate the local-volume short-GRB population.",
    },
    "lbGRB + red KN (HMNS)": {
        "R_obs": OBSERVED_SGRB_RATES["Ghirlanda+ 2016"]["R_obs"],
        "R_obs_lo": OBSERVED_SGRB_RATES["Ghirlanda+ 2016"]["R_obs_lo"],
        "R_obs_hi": OBSERVED_SGRB_RATES["Ghirlanda+ 2016"]["R_obs_hi"],
        "reference": "Ghirlanda+ 2016, A&A 594, A84",
        "caveat": "No dedicated observed lbGRB+HMNS rate in the literature; "
        "mapped to the lower-luminosity end of the observed sGRB "
        "distribution (Ghirlanda+ 2016 GBM sample).",
    },
    "lbGRB + red KN (disk)": {
        "R_obs": np.nan,
        "R_obs_lo": np.nan,
        "R_obs_hi": np.nan,
        "reference": "No published observed rate",
        "caveat": "No measured local rate for disk-driven long mergers; "
        "observational identification is still emerging "
        "(Rastinejad+ 2022, GRB 211211A class).",
    },
    "Faint lbGRB": {
        "R_obs": np.nan,
        "R_obs_lo": np.nan,
        "R_obs_hi": np.nan,
        "reference": "No published observed rate",
        "caveat": "Faint lbGRBs (small disk, prompt collapse, q < q_thresh) "
        "are below current GBM sensitivity for almost all "
        "plausible viewing angles. Treat as upper limit < 1 "
        "Gpc^-3 yr^-1.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# LVK GWTC-5.0 intrinsic local merger rate references
# ═══════════════════════════════════════════════════════════════════════════
# Intrinsic merger rate densities from the GWTC-5.0 population analysis: the
# LIGO-Virgo-KAGRA Collaboration (2026), "GWTC-5.0: Population Properties of
# Merging Compact Binaries" (arXiv:2605.27226).  The published 90 percent
# credible intervals are the "Joint" row of their Table 2, the union of the
# PixelPop (weakly-modelled) and FullPop (strongly-modelled) 90 percent CRs,
# which marginalises over modelling-systematic uncertainty between the two
# population approaches.  267 merger candidates accumulated through the
# second part of the fourth LVK observing run (O4b); GWTC-5.0 added no new
# BNS or NSBH candidates, so the NS-side rates tightened relative to GWTC-4
# (which gave BNS 7.6-250, NSBH 9.1-84).  BNS and NSBH rates are quoted at
# z = 0; BBH at z = 0.2 (the redshift where the BBH rate is best measured).
# All values are intrinsic source-frame densities in Gpc^-3 yr^-1, NOT
# beaming-limited GRB rates; compare against ``compute_merger_rate(...)[iz0]``
# directly, NOT against ``beamed_rate(...)``.
LVK_GWTC5_LOCAL_RATES = {
    "BNS": {
        "R_lo": 5.1,
        "R_hi": 154.7,
        "reference": "LVK 2026, GWTC-5.0 (arXiv:2605.27226)",
        "note": "90% CR at z=0; joint union of PixelPop and FullPop model "
        "CRs (Table 2); no new BNS in GWTC-5.0",
    },
    "NSBH": {
        "R_lo": 6.7,
        "R_hi": 32.8,
        "reference": "LVK 2026, GWTC-5.0 (arXiv:2605.27226)",
        "note": "90% CR at z=0; joint union of PixelPop and FullPop model "
        "CRs (Table 2); called NSBH in GWTC-5.0 (= BHNS in this project); "
        "no new NSBH in GWTC-5.0",
    },
    "BBH": {
        "R_lo": 27.5,
        "R_hi": 49.4,
        "reference": "LVK 2026, GWTC-5.0 (arXiv:2605.27226)",
        "note": "90% CR at z=0.2; joint union of PixelPop and FullPop model "
        "CRs (Table 2); reference only, this project does not load BBH samples",
    },
}


# Per-model decomposition of the GWTC-5.0 local intrinsic merger rate density
# (LVK 2026 Table 2 rows 1-2): the PixelPop weakly-modelled posterior and the
# FullPop strongly-modelled posterior, each with a median and asymmetric 90
# percent credible interval.  The "Joint" row of Table 2 (encoded above in
# ``LVK_GWTC5_LOCAL_RATES``) is the union of these two intervals; showing them
# individually exposes which side of the modelling-systematic split a model
# rate is in tension with, rather than just the wider union.  BNS and NSBH are
# quoted at z = 0, BBH at z = 0.2, all in Gpc^-3 yr^-1.
LVK_GWTC5_PER_MODEL_RATES = {
    "BNS": {
        "PixelPop": {"R_med": 23.4, "R_lo_err": 18.2, "R_hi_err": 54.7},
        "FullPop": {"R_med": 59.3, "R_lo_err": 43.9, "R_hi_err": 95.4},
    },
    "NSBH": {
        "PixelPop": {"R_med": 15.9, "R_lo_err": 8.9, "R_hi_err": 16.9},
        "FullPop": {"R_med": 14.2, "R_lo_err": 7.5, "R_hi_err": 12.0},
    },
    "BBH": {
        "PixelPop": {"R_med": 37.5, "R_lo_err": 9.1, "R_hi_err": 11.9},
        "FullPop": {"R_med": 36.0, "R_lo_err": 8.5, "R_hi_err": 11.1},
    },
    "reference": "LVK 2026, GWTC-5.0 (arXiv:2605.27226), Table 2 rows 1-2",
}


# Inferred minimum BH mass in NSBH systems.  GWTC-5.0 added no new NS-bearing
# candidates and defers its NS-population analysis to GWTC-4 (Abac+ 2025
# Section 4.3) and to Abac+ 2024 (arXiv:2404.04248, GW230529 follow-up using
# the NSBHPop model of Biscoveanu+ 2022): 3.4 +1.0 / -1.2 Msun at 90 percent
# credibility.  Overlay this as a horizontal line + shaded band on a BHNS
# mass-plane panel to compare against the COMPAS BH-mass distribution.
LVK_GWTC5_NSBH_MIN_BH_MASS = {
    "M_BH_min": 3.4,
    "M_BH_min_lo_err": 1.2,
    "M_BH_min_hi_err": 1.0,
    "reference": "LVK 2026, GWTC-5.0 (arXiv:2605.27226); NS-population "
    "analysis inherited from Abac+ 2024 (arXiv:2404.04248)",
    "note": "Inferred minimum BH mass in NSBH systems; no new NSBH in GWTC-5.0",
}


# ═══════════════════════════════════════════════════════════════════════════
# Observed sGRB rate vs redshift: Wanderman & Piran (2015)
# ═══════════════════════════════════════════════════════════════════════════
def wanderman_piran_2015_Rz(
    z, R0=4.1, z_peak=0.9, sigma_lo=0.39, sigma_hi=0.26, R0_lo=2.2, R0_hi=6.4
):
    """Observed sGRB comoving rate density vs redshift from
    Wanderman & Piran (2015), MNRAS 448, 3026.

    Implements the piecewise-exponential form from WP15 Eq. (9):

        R(z) = R0 * exp(+(z - z_peak) / sigma_lo)   for z <= z_peak
        R(z) = R0 * exp(-(z - z_peak) / sigma_hi)   for z >  z_peak

    with z_peak = 0.9, sigma_lo = 0.39 (rising), sigma_hi = 0.26
    (falling).  R0 = 4.1 +2.3/-1.9 Gpc^-3 yr^-1 is the observed,
    beaming-limited normalization at the peak.  This is the comoving
    rate density of *detectable* sGRBs (luminosity-function- and
    beaming-folded), not the intrinsic merger rate, so it is the
    appropriate target for ``beamed_rate(...)`` model curves.

    The uncertainty band ``R_lo`` / ``R_hi`` varies only the
    normalization R0; the shape parameters' uncertainties are
    correlated with the luminosity function in the original fit and
    are not propagated here.

    Returns dict with 'R_best', 'R_lo', 'R_hi' arrays (same shape as z).
    """
    z = np.asarray(z, dtype=float)
    dz = z - z_peak
    shape = np.where(dz <= 0.0, np.exp(+dz / sigma_lo), np.exp(-dz / sigma_hi))
    return {
        "R_best": R0 * shape,
        "R_lo": R0_lo * shape,
        "R_hi": R0_hi * shape,
    }
