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
from scipy.stats import chi2 as _chi2

from grb_physics import (
    EOS_MODELS,
    EPS_JET,
    GOTTLIEB25_T_HMNS_RANGE,
    GOTTLIEB25_WIND_FRAC,
    H0_PLANCK15,
    MISALIGNMENT_SYSTEMATIC_FACTOR,
    OMEGA_L_PLANCK15,
    OMEGA_M_PLANCK15,
    T_D_PROMPT_RANGE,
    T_E_RANGE,
    V_EJ_BNS,
    bns_disk_mass,
    bns_dynamical_ejecta,
    ejecta_energy_in_cone,
    jet_breaks_out_pais,
    jet_energy_from_disk,
    ns_radius_from_eos,
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
# in tests/sections/test_section_06b_formation_channels.py.
# alpha_CE per Broekgaarden+ 2021 Sec. 5.2 Table 2.
CE_PRESCRIPTION_BROEKGAARDEN21 = {
    "energy_formalism": "Webbink 1984",
    "lambda_CE": "Xu and Li 2010",
    "stability": "Hurley 2002 (Vinciguerra+ 2020 modified)",
    "double_CE_flag": "both stars in core-He burning at CE onset",
    "alpha_CE": {"A": 1.0, "F": 0.5, "G": 2.0, "J": 1.0, "K": 1.0},
}


MSSFR_Z_LOWER_LIMIT: float = 1e-90
MSSFR_Z_UPPER_LIMIT: float = 1.0
"""Outer metallicity bin edges of the upstream cosmic integration.  COMPAS
``ClassMSSFR.__init__`` defaults ``metallicityLowerLimit = 1e-90`` and
``metallicityUpperLimit = 1.0``, and ``calculateMetallicityBinEdges`` puts them
at the two ends of the Riemann sum, so the lowest and highest simulated
metallicities carry every star-forming galaxy off the ends of the grid."""


def dPdlogZ_binned_to_grid(dPdlogZ, metallicities, metallicityGrid, p_draw):
    """Re-bin a pointwise dP/dlnZ onto the COMPAS grid's own Riemann cells.

    ``FastCosmicIntegration.find_formation_and_merger_rates`` takes the
    metallicity PDF *pointwise*: it looks up ``dPdlogZ`` at the fine-grid
    column nearest each system's birth Z (via ``np.digitize``) and multiplies
    by ``1 / p_draw``.  That implicitly hands every one of the ``N`` simulated
    metallicities the same cell width ``(ln Z_max - ln Z_min) / N`` and, more
    importantly, assigns nothing at all to star formation outside
    ``[Z_min, Z_max]``.

    The cosmic integration that produced the Broekgaarden+ 2021 ``w_000``
    anchors does neither.  COMPAS ``ClassMSSFR.calculateMetallicityBinEdges``
    integrates dP/dZ over bins whose interior edges are the log-midpoints
    between adjacent grid points and whose outer edges are
    ``MSSFR_Z_LOWER_LIMIT`` and ``MSSFR_Z_UPPER_LIMIT``, so the top and bottom
    grid points absorb the out-of-range tails.  This function returns a
    ``dPdlogZ`` array that reproduces that convention when passed through the
    FCI call: column ``i`` of the grid is set to ``P_i / dcell`` with ``P_i``
    the bin integral and ``dcell = (ln Z_max - ln Z_min) / N`` the width FCI
    will implicitly re-multiply by.  Every other column is left alone; no
    system maps to one, since the grid spacing (0.1097 in ln Z) is ten times
    the fine-grid step.

    The difference is not cosmetic on a metallicity-loving population.  Under
    the Neijssel+ 2019 MSSFR the top bin carries 58 per cent of z = 0 star
    formation, and dropping it puts the Model A BNS R(z = 0) a factor 1.48
    below the file's own anchor while leaving BHNS, whose yield has collapsed
    by Z_max, correct to 2 per cent.

    Parameters
    ----------
    dPdlogZ : 2-D array, shape (n_z, n_fine)
        Pointwise PDF from ``find_metallicity_distribution``, per d(ln Z).
    metallicities : 1-D array
        The fine grid ``dPdlogZ`` is evaluated on (log-uniform).
    metallicityGrid : 1-D array
        The simulated COMPAS metallicities, sorted ascending.
    p_draw : float
        ``1 / (ln Z_max - ln Z_min)`` from ``find_metallicity_distribution``.

    Returns
    -------
    2-D array, same shape as ``dPdlogZ``.

    Notes
    -----
    Absorbing the tails extrapolates the outermost simulated yield to
    metallicities the run never sampled.  For BNS the top grid point sits
    inside the common-envelope lambda artefact range, so the extrapolated
    yield is the artefact-inflated one; the size of that systematic is
    reported in Section 6 rather than hidden in a normalisation.
    """
    dP = np.asarray(dPdlogZ, dtype=float)
    fine = np.asarray(metallicities, dtype=float)
    grid = np.asarray(metallicityGrid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("metallicityGrid must hold at least two metallicities")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("metallicityGrid must be sorted and strictly increasing")

    ln_fine = np.log(fine)
    step = float(ln_fine[1] - ln_fine[0])

    ln_mid = 0.5 * (np.log(grid[:-1]) + np.log(grid[1:]))
    edges = np.concatenate(([np.log(MSSFR_Z_LOWER_LIMIT)], ln_mid, [np.log(MSSFR_Z_UPPER_LIMIT)]))

    # FCI evaluates n_formed * dPdlogZ[col] / p_draw * w, and n_formed already
    # carries 1 / M_SF over the whole run rather than per metallicity, so the
    # column that reproduces a bin probability P_i is N * P_i * p_draw.
    out = dP.copy()
    cols = np.digitize(grid, fine)
    for i, col in enumerate(cols):
        in_bin = (ln_fine >= edges[i]) & (ln_fine < edges[i + 1])
        out[:, col] = grid.size * p_draw * dP[:, in_bin].sum(axis=1) * step
    return out


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

    Carries a ~1 percent numerical bias from two FCI discretisations, on
    top of whatever the astrophysics contributes.  FCI takes the
    metallicity column with ``np.digitize``, which selects the grid bin
    *above* each system's Z rather than the nearest (on the Model A BNS
    sample the two agree for only 22 percent of systems, mean offset
    +0.78 bins of dlnZ = 0.01), and it indexes formation times on its own
    redshift grid instead of interpolating.  Evaluating the same equation
    with nearest-bin metallicity and an interpolated age-redshift
    relation gives 0.988 of this function's R(z=0), rising to 1.000 by
    z ~ 4; adopting both FCI conventions reproduces it to 1.2e-4.  The
    bias is far below the modelling systematics (the common-envelope
    lambda artefact alone moves the BNS normalisation by a factor ~3.7),
    but it is a floor on how precisely any rate here can be quoted.

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
    and chirp-mass histograms.

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
    metallicityGrid=None,
):
    """Back-derive the simulation's star-forming mass [Msun] from the z = 0 anchor.

    The COMPAS Broekgaarden+ 2021 files store ``weights_intrinsic/w_000``,
    which is the cosmic-integration weight at the z = 0 slice computed by the
    upstream pipeline with Neijssel+ 2019 MSSFR parameters.  Summed across
    binaries, ``w_000`` gives the published intrinsic local merger rate
    density, ``expected_local_rate``.  Re-running that integration here and
    dividing gives back the mass it was normalised by::

        M_SF = R_unnorm(z = 0; Neijssel) / expected_local_rate

    **This is a cross-check, not the science-path normalisation.** The
    simulation's star-forming mass is known from the run configuration and is
    what ``simulation_star_forming_mass`` returns (53 grid points, about a
    million binaries each, 79.11 Msun of star formation per binary;
    Broekgaarden+ 2021 Table 1 and Sec. 2.2).  Use that.  What this function
    adds is an independent test that the whole cosmic-integration chain
    reproduces the upstream anchor at that mass: pass ``metallicityGrid`` and
    it returns 4.19e9 Msun for both Model A populations, within 0.4 per cent
    of the configuration value.

    ``metallicityGrid`` is what makes it agree.  Without it the internal
    integration uses FCI's pointwise metallicity convention, which assigns
    nothing to star formation outside the simulated ``[Z_min, Z_max]``, where
    the upstream pipeline folds those tails into the outermost grid points
    (see ``dPdlogZ_binned_to_grid``).  On a metallicity-loving population that
    is not a detail: the pointwise convention returns 2.80e9 Msun for Model A
    BNS, a factor 1.50 below the configuration mass, against 4.05e9 (3 per
    cent) for BHNS, whose yield has already collapsed by ``Z_max``.  Adopting
    the returned number as a normalisation therefore hides a missing term in
    the integration inside the mass, and propagates it to anything else that
    mass divides, the Section 5 formation yield included.

    Parameters
    ----------
    COMPAS_Z, COMPAS_delay_times, COMPAS_weights : 1-D arrays
        The **full** merging population of the file, before any physics
        cut.  ``expected_local_rate`` sums ``w_000`` over every row in the
        file, so calibrating against a subset silently rescales the
        recovered mass by the subset's share of R(z = 0): feeding the
        cold-stability-cut sample makes the cut vanish from the rate
        normalisation instead of lowering it (7.6 per cent on Model A
        BHNS, which loses 18.5 per cent of its weight to that cut).
        Apply cuts to the sample passed to ``compute_merger_rate``, never
        to the sample passed here.
    expected_local_rate : float
        Sum of ``weights_intrinsic/w_000`` from the Broekgaarden+ 2021
        HDF5 file (the published intrinsic local rate, Gpc^-3 yr^-1).
        Use ``grb_io.read_expected_local_rate``.
    Z_min_COMPAS, Z_max_COMPAS : float, optional
        COMPAS metallicity sampling range [linear, not log].  Defaults
        to ``(min(COMPAS_Z), max(COMPAS_Z))``.  Pass the simulation's
        full range (``METALLICITY_GRID[0]``, ``METALLICITY_GRID[-1]``)
        if a class subset has narrower coverage.
    metallicityGrid : 1-D array, optional
        The simulated metallicities, for the upstream bin convention.  Pass
        the design grid (``grb_io.METALLICITY_GRID``, deduplicated), never
        ``np.unique(COMPAS_Z)``: models E, F and alpha0p1 leave 6 to 12 of
        the 53 points with no merging system, and inferring the grid from the
        sample would move the bin edges with the sample.  Omitting it keeps
        the FCI pointwise convention and is only correct for a population
        that produces nothing at the ends of the grid.

    Returns
    -------
    mean_mass_evolved : float
        Star-forming mass [Msun] the anchor implies.
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

    if metallicityGrid is not None:
        dPdlogZ_nei = dPdlogZ_binned_to_grid(dPdlogZ_nei, mets_nei, metallicityGrid, p_draw_nei)

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
# Right edge of the metallicity axis used for display binning, log10(Z / Zsun).
# The COMPAS STROOPWAFEL prior extends to Z = 0.03 ~ 2.1 Zsun, where the
# simulated grid runs out while cosmic star formation does not: under the
# Levina+ 2026 (arXiv:2601.20202) Azzalini skew-log-normal fit to TNG100-1
# (Table 1) the mean birth metallicity is <Z>(z=0) = 0.0247 = 1.74 Zsun, and
# 28.5 percent of z = 0 star formation (16.2 percent SFR-weighted over
# 0 < z < 10) falls above the grid ceiling entirely.  Solar is a round place to
# stop a yield plot, not a place where the star formation stops.
#
# This is therefore a display convention, not a relevance cut and not a noise
# cut: 64.8 percent of z = 0 cosmic star formation (39.8 percent SFR-weighted)
# sits above Zsun, and the Model A merging samples hold 13.4 percent (BNS) and
# 1.7 percent (BHNS) of their STROOPWAFEL weight there, at Kish effective
# sample sizes of 4.7e4 and 4.3e3.  Any figure that applies the truncation
# should say what fraction it removed rather than imply the tail was
# unimportant or too noisy to plot.
LOGZ_MAX_PHYSICAL: float = 0.0

# Birth metallicity above which the Broekgaarden+ 2021 COMPAS grid carries an
# artificial jump in formation yield.  Their paper II (Broekgaarden+ 2022,
# arXiv:2112.05763) states in the Fig. 1 caption that the sharp yield increase
# around Z = 0.0105, "particularly visible in the NSNS panel", comes from a
# bifurcation in the common-envelope lambda values at this Z, not from stellar
# physics.  The value is not read off their figure: COMPAS hardcodes it as
# LAMBDA_NANJING_ZLIMIT = 0.0105 in src/constants.h, the branch that picks the
# Xu & Li (2010) Population I lambda coefficients above it and Population II
# below, with no interpolation on the path this grid was run with.  Yield curves plotted against Z should stop here, or the step reads
# as a prediction.  It is a step and not a spike: eta jumps between the
# adjacent grid points at Z = 0.01018 and 0.01137 and stays at the new level
# for the rest of the grid.  Both populations are affected once they are split
# by class, so neither is safe to plot through it.  Model A jump factors are
# 2.9 (BNS sbGRB + blue KN), 3.4 (BNS lbGRB + red KN HMNS) and 4.5 (BHNS
# lbGRB + red KN disk).  A population total can hide this: the BHNS total
# falls by 0.75 across the same step because the 82-percent-weight No GRB
# class declines there, which is why the check has to be done per class.
Z_CE_LAMBDA_BIFURCATION: float = 0.0105


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


# Fractional tolerance on the check that a metallicity grid really is the
# uniform-in-log design grid the equal-mass-per-point convention assumes.  The
# published Broekgaarden+ 2021 grid is stored rounded (two significant figures
# at the low-Z end, more at the high-Z end) and departs from the uniform design
# grid by at most 3.6 per cent, so the tolerance sits above the rounding and
# well below any genuinely non-uniform grid.
GRID_LOG_UNIFORM_RTOL: float = 0.05


def metallicity_prior_mass_fraction(metallicityGrid):
    """Star-forming-mass fraction simulated at each metallicity grid point.

    Broekgaarden+ 2021 (arXiv:2103.02608) Table 1 samples "about a million
    binaries per Z_i grid point" on a grid "distributed using a uniform grid
    in log(Z_i) with 53 metallicities".  Every grid point therefore draws the
    same number of binaries from the same Z-independent initial-condition
    distributions, so each carries the same star-forming mass::

        f_i = 1 / N,   sum_i f_i = 1

    Multiply by a population's total ``mean_mass_evolved`` to get the
    per-metallicity star-forming mass M_SF(Z_i), the denominator of the
    intensive formation efficiency their Fig. 5 plots.

    Do not derive f_i from differences of the stored grid values.  They are
    rounded (0.0001, 0.00011, 0.00012 at the low-Z end against a design
    spacing of Delta ln Z = 0.10969), which makes the apparent cell widths
    jitter by a factor 2.54 across the grid and would swing eta(Z) by up to
    25 per cent at interior points on rounding noise alone, and by a factor
    2.3 at the two ends, where a midpoint-edge construction gives the first
    and last points half-width cells.  Equal mass per
    point is also what the rate path already assumes: FastCosmicIntegration
    weights every system by a constant ``dlogZ / p_draw``, independent of
    where the system's Z sits in the grid.

    Raises
    ------
    ValueError
        If the grid is not uniform in log Z to within
        ``GRID_LOG_UNIFORM_RTOL``, since equal mass per point is then not
        the right convention.
    """
    Z = np.asarray(metallicityGrid, dtype=float)
    if Z.ndim != 1 or Z.size == 0:
        raise ValueError("metallicityGrid must be a non-empty 1-D array")
    if np.any(Z <= 0):
        raise ValueError("metallicity grid values must be positive")
    if Z.size > 2:
        design = np.geomspace(Z.min(), Z.max(), Z.size)
        if not np.allclose(np.sort(Z), design, rtol=GRID_LOG_UNIFORM_RTOL):
            raise ValueError(
                "metallicity grid is not uniform in log Z to within "
                f"{GRID_LOG_UNIFORM_RTOL:.0%}; equal star-forming mass per "
                "grid point does not apply"
            )
    return np.full(Z.size, 1.0 / Z.size)


def _grid_index(metallicityGrid, Z_all):
    """Index of each system's birth metallicity in a sorted metallicity grid.

    Raises if any system falls off the grid, rather than dropping it: the
    system would vanish from the eta(Z) numerator while its star-forming
    mass stayed in the denominator, biasing the yield low with no warning.
    """
    grid = np.asarray(metallicityGrid, dtype=float)
    Z = np.asarray(Z_all, dtype=float)
    idx = np.searchsorted(grid, Z)
    idx = np.clip(idx, 0, grid.size - 1)
    on_grid = np.isclose(Z, grid[idx], rtol=1e-6, atol=0.0)
    if not np.all(on_grid):
        n_off = int((~on_grid).sum())
        raise ValueError(
            f"{n_off} of {Z.size} systems carry a birth metallicity that is "
            "not on the supplied grid; eta(Z) would silently lose them"
        )
    return idx


def _grid_weight_sums(idx, w_all, n_grid, mask=None):
    """Per-grid-point (sum w, sum w^2, raw count) for an optional subset."""
    w = np.asarray(w_all, dtype=float)
    if mask is not None:
        idx, w = idx[mask], w[mask]
    Sw = np.bincount(idx, weights=w, minlength=n_grid)
    Sw2 = np.bincount(idx, weights=w * w, minlength=n_grid)
    N = np.bincount(idx, minlength=n_grid)
    return Sw, Sw2, N


def formation_efficiency(metallicityGrid, Z_all, w_all, masks=None, mean_mass_evolved=None):
    """
    Intensive formation efficiency eta(Z) per metallicity bin.

    Returns the number of merging systems formed per solar mass of star
    formation *at each metallicity*, the quantity Broekgaarden+ 2021
    (arXiv:2103.02608, Fig. 5) plots::

        eta(Z_i) = (sum of STROOPWAFEL weights at Z_i) / M_SF(Z_i)

    The per-metallicity star-forming mass is the population total split
    equally over the simulated grid points,
    ``M_SF(Z_i) = mean_mass_evolved * f_i`` with ``f_i`` from
    ``metallicity_prior_mass_fraction``.  Dividing by the per-Z mass
    rather than the scalar total makes eta(Z) intensive (independent of
    how the grid is binned) and lands it on the published
    formation-efficiency axis; using the bare total instead
    suppresses every bin by ~1/f_i (the number of sampled metallicities).

    Parameters
    ----------
    metallicityGrid : 1-D array
        Sorted grid of metallicity values (e.g. 53-point COMPAS grid).
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
        Total stellar mass evolved in the simulation [Msun].  Required;
        use ``calibrate_mean_mass_evolved()`` on the population *before*
        any physics cut, so the cut shows up in the yield instead of
        being absorbed into the normalisation.

    Returns
    -------
    dict of 1-D arrays keyed by mask name, plus 'total'.
    """
    if mean_mass_evolved is None:
        raise ValueError(
            "mean_mass_evolved must be provided; use calibrate_mean_mass_evolved() to derive it"
        )
    grid = np.asarray(metallicityGrid, dtype=float)
    M_SF = mean_mass_evolved * metallicity_prior_mass_fraction(grid)
    idx = _grid_index(grid, Z_all)

    result = {"total": _grid_weight_sums(idx, w_all, grid.size)[0] / M_SF}
    if masks is not None:
        for name, m in masks.items():
            result[name] = _grid_weight_sums(idx, w_all, grid.size, m)[0] / M_SF
    return result


GEHRELS86_UL_ZERO_COUNTS: float = 1.841
"""One-sigma (84.13 per cent) Poisson upper limit on the true rate given zero
observed events.  Gehrels (1986), ApJ 303, 336, Table 1, ``n = 0`` column at
``S = 1``.  Used to place an upper limit on eta in a coarse bin that holds no
system of a given subset, so an empty bin is drawn as a bound rather than as
a gap the reader cannot distinguish from an unmeasured one."""

ETA_GATE_N_MIN: int = 50
"""Raw-count floor below which ``formation_efficiency_binned`` stops calling a
coarse bin well sampled.  CODE HEURISTIC; not from any single paper.  Neither
Broekgaarden+ 2021 nor Broekgaarden+ 2022 gates their yield-versus-metallicity
figures at all (their Fig. 5 and Fig. 1 plot every simulated grid point and name
sampling noise in the caption), so no published threshold exists to inherit.
The value is returned as ``ok`` for callers who want it; the Section 5 and 5b
figures do not draw it, because the cut is arbitrary at the margin: on Model A
the two bins it rejects hold 45 and 40 systems against a neighbour it accepts
at 56."""

ETA_GATE_ESS_MIN: float = 10.0
"""Kish (1965) effective-size floor paired with ``ETA_GATE_N_MIN``.  CODE
HEURISTIC; not from any single paper.  It exists because STROOPWAFEL importance
sampling lets a bin hold many raw systems whose weight is carried by one of
them, so a raw count alone can overstate the support."""

GEHRELS86_ONE_SIGMA_TAIL: float = 0.158655
"""Lower-tail probability of the one-sigma (84.13 per cent) Poisson bound,
the ``S = 1`` column Gehrels (1986) tabulates."""


def gehrels86_poisson_interval(n):
    """One-sigma Poisson bounds on a true count given ``n`` counts observed.

    Exact chi-square form of the limits Gehrels (1986), ApJ 303, 336
    tabulates at ``S = 1`` (upper in their Table 1, lower in their Table 2),
    rather than his series approximations: ``n = 0`` returns
    ``GEHRELS86_UL_ZERO_COUNTS`` (1.8410 against the tabulated 1.841) and
    ``n = 1`` a lower bound of 0.1728 against the tabulated 0.173.

    ``n`` need not be an integer.  A weighted sample enters through its Kish
    effective count, which is continuous, and the bounds are smooth in n.
    """
    n = np.asarray(n, dtype=float)
    if np.any(n < 0):
        raise ValueError("counts must be non-negative")
    lo = np.where(n > 0, 0.5 * _chi2.ppf(GEHRELS86_ONE_SIGMA_TAIL, 2.0 * n), 0.0)
    hi = 0.5 * _chi2.ppf(1.0 - GEHRELS86_ONE_SIGMA_TAIL, 2.0 * n + 2.0)
    return lo, hi


def formation_efficiency_binned(
    metallicityGrid,
    Z_all,
    w_all,
    masks=None,
    mean_mass_evolved=None,
    n_bins=15,
    Z_max=None,
    Z_cut=None,
    n_min=ETA_GATE_N_MIN,
    ess_min=ETA_GATE_ESS_MIN,
):
    """``formation_efficiency`` pooled into equal-count coarse log-Z bins.

    Pooling adjacent grid points into ``n_bins`` equal-count groups trades
    metallicity resolution for per-bin sample size.  eta stays intensive
    under the pooling because numerator and denominator are summed over the
    same group, and stays additive over disjoint masks because every mask
    shares one denominator, so ``n_bins`` is a display choice and not a
    change of estimator.  Passing ``n_bins = metallicityGrid.size`` leaves
    the native grid untouched, which is what Broekgaarden+ 2021 Fig. 5 and
    Broekgaarden+ 2022 Fig. 1 plot.

    How far the pooling has to go depends on how finely the sample is cut,
    not on the grid.  A split by GRB class alone holds a few hundred systems
    per native point and needs none; a split by class and formation channel
    puts most of the weight in one channel and leaves the others empty over
    much of the grid, and there the pooling is what keeps a curve joined.

    ``ok`` marks the bins that clear raw count ``n_min`` and Kish (1965)
    effective size ``ess_min``.  It is a precision statement and nothing
    else: a bin that fails it still carries a measured eta and its
    ``eta_err``, and the caller decides how to draw it.  Deleting the bin
    instead throws away coverage the sample actually has.

    Parameters
    ----------
    Z_max : float, optional
        Drop native grid points above this metallicity before pooling.
    Z_cut : float, optional
        Flag (do not drop) any coarse bin holding a native point at or above
        this metallicity, for the Broekgaarden+ 2022 common-envelope lambda
        artefact at ``Z_CE_LAMBDA_BIFURCATION``.  Returned as ``artifact``.

    Returns
    -------
    dict
        ``Z`` (geometric-mean centre of each coarse bin), ``Z_hi`` (its
        highest native member) and ``artifact`` (the ``Z_cut`` flag) are
        per-bin arrays.  ``eta``, ``eta_err``, ``eta_err_lo``, ``eta_err_hi``,
        ``eta_ul``, ``ok``, ``n_raw`` and ``ess`` are dicts keyed as ``masks``
        plus ``'total'``.  ``eta_err`` is the symmetric sqrt(sum w^2)/M_SF;
        ``eta_err_lo`` / ``eta_err_hi`` are the asymmetric Poisson bars that
        converge to it and stay above zero at small ``ess``.

        Three states are distinguishable and a log ordinate renders them
        identically, so a caller that draws this has to separate them:
        ``ok`` (draw the point), ``~ok & n_raw > 0`` (measured but
        imprecise, draw with ``eta_err_lo`` / ``eta_err_hi``), and ``n_raw == 0`` (the subset
        genuinely produces nothing here, draw ``eta_ul`` as an upper limit).
    """
    if mean_mass_evolved is None:
        raise ValueError(
            "mean_mass_evolved must be provided; use calibrate_mean_mass_evolved() to derive it"
        )
    grid = np.asarray(metallicityGrid, dtype=float)
    f_native = metallicity_prior_mass_fraction(grid)
    idx = _grid_index(grid, Z_all)

    keep_grid = np.arange(grid.size) if Z_max is None else np.where(grid <= Z_max)[0]
    if keep_grid.size < n_bins:
        raise ValueError(
            f"{keep_grid.size} grid points below Z_max cannot fill {n_bins} coarse bins"
        )
    grp = np.arange(keep_grid.size) * n_bins // keep_grid.size
    count = np.bincount(grp, minlength=n_bins)

    Z_c = np.exp(np.bincount(grp, weights=np.log(grid[keep_grid]), minlength=n_bins) / count)
    Z_hi = np.zeros(n_bins)
    np.maximum.at(Z_hi, grp, grid[keep_grid])
    f_c = np.bincount(grp, weights=f_native[keep_grid], minlength=n_bins)
    M_SF = mean_mass_evolved * f_c

    artifact = np.zeros(n_bins, dtype=bool) if Z_cut is None else (Z_hi >= Z_cut)

    # Mean weight of every system in each coarse bin, whatever its class.  It
    # is the local estimate of what one more system would have contributed,
    # and so sets the scale of the zero-count upper limit below.
    Sw_all, _, N_all = _grid_weight_sums(idx, w_all, grid.size)
    Sw_all_c = np.bincount(grp, weights=Sw_all[keep_grid], minlength=n_bins)
    N_all_c = np.bincount(grp, weights=N_all[keep_grid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        w_bar = np.where(N_all_c > 0, Sw_all_c / N_all_c, np.nan)
    eta_ul_bin = GEHRELS86_UL_ZERO_COUNTS * w_bar / M_SF

    def _one(mask):
        Sw, Sw2, N = _grid_weight_sums(idx, w_all, grid.size, mask)
        Sw_c = np.bincount(grp, weights=Sw[keep_grid], minlength=n_bins)
        Sw2_c = np.bincount(grp, weights=Sw2[keep_grid], minlength=n_bins)
        N_c = np.bincount(grp, weights=N[keep_grid], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            ess = np.where(Sw2_c > 0, Sw_c * Sw_c / Sw2_c, 0.0)
        ok = (N_c >= n_min) & (ess >= ess_min)
        ul = np.where(N_c == 0, eta_ul_bin, np.nan)

        # Asymmetric bars for the bins the gate rejects.  sqrt(sum w^2) is the
        # Gaussian limit of the same statistic and runs below zero once a bin
        # holds a handful of systems, which a log ordinate cannot draw; the
        # last coarse bin of the Model A BHNS sample holds five.  Putting the
        # exact Poisson bounds on the Kish effective count and scaling by the
        # bin's effective weight sum w^2 / sum w recovers sqrt(sum w^2)
        # exactly in the large-ess limit, so this generalises eta_err rather
        # than replacing it with a different estimator.
        with np.errstate(invalid="ignore", divide="ignore"):
            w_eff = np.where(Sw_c > 0, Sw2_c / Sw_c, 0.0)
        lo_n, hi_n = gehrels86_poisson_interval(ess)
        err_lo = (ess - lo_n) * w_eff / M_SF
        err_hi = (hi_n - ess) * w_eff / M_SF
        return Sw_c / M_SF, np.sqrt(Sw2_c) / M_SF, err_lo, err_hi, ul, ok, N_c, ess

    keys = ("eta", "eta_err", "eta_err_lo", "eta_err_hi", "eta_ul", "ok", "n_raw", "ess")
    out = {"Z": Z_c, "Z_hi": Z_hi, "artifact": artifact}
    out.update({k: {} for k in keys})
    for name, m in [("total", None)] + list((masks or {}).items()):
        for k, v in zip(keys, _one(m), strict=True):
            out[k][name] = v
    return out


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


def breakout_fraction_bns_eos(
    class_masks,
    m1,
    m2,
    weights,
    eos_models=None,
    hmns_wind_frac=GOTTLIEB25_WIND_FRAC,
    hmns_labels=_HMNS_CLASS_LABELS,
    theta_j_by_class=None,
    n_draws=512,
    rng=None,
):
    """EOS-marginalized, remnant-conditioned BNS jet-breakout fraction per class.

    Wraps :func:`breakout_fraction_bns` to remove its dependence on a single
    NS radius.  ``bns_disk_mass`` is knife-edge sensitive near R_1.4 ~ 12 km
    (its docstring warns of ~100x swings for a 1 km radius change), so a
    science-grade breakout fraction must average over an EOS prior rather
    than fix one radius.  For each EOS in ``eos_models`` this routine derives
    self-consistent component radii from ``ns_radius_from_eos`` and rebuilds
    the per-system disk mass (``bns_disk_mass``) and obstructing ejecta
    (``bns_dynamical_ejecta``), then runs the full Pais et al. (2025) Monte
    Carlo breakout test.

    Ejecta composition follows the launch geometry.  Prompt-collapse classes
    launch their jet at ``t_d ~ 0.01-0.1 s`` (``T_D_PROMPT_RANGE``), before
    the secular disk wind is blown, so their obstruction is the dynamical
    ejecta alone.  The two HMNS classes launch only after the remnant
    collapses, by which point a wind of ``hmns_wind_frac * M_disk`` has
    polluted the funnel; that wind is added to their obstructing ejecta.

    The returned band brackets two independent sources of spread: the Pais
    MC over the uncertain jet/ejecta parameters (inside each
    ``breakout_fraction_bns`` call) and the EOS systematic across
    ``eos_models``.  Per class the envelope is the min ``lo`` and max ``hi``
    over the EOS set, with ``med`` the median of the per-EOS medians.

    Parameters
    ----------
    class_masks : dict[str, 1-D bool array]
        Gottlieb (2024) class membership keyed by the canonical labels
        (so the HMNS conditioning in ``hmns_labels`` matches).
    m1, m2 : 1-D array
        Component gravitational masses [Msun], aligned with ``weights``.
    weights : 1-D array
        STROOPWAFEL weights.
    eos_models : dict, optional
        Mapping of EOS name to a dict carrying ``R_1p4`` / ``M_TOV``
        (as in ``grb_physics.EOS_MODELS``, the default).
    hmns_wind_frac : float, optional
        Disk-wind ejected fraction added to the HMNS-class obstruction.
        Defaults to ``GOTTLIEB25_WIND_FRAC``.  Set to 0.0 to disable the
        wind contribution (used in tests).
    hmns_labels : container of str, optional
        Class labels treated as HMNS remnants for both the launch-delay
        window and the disk-wind addition.
    theta_j_by_class, n_draws, rng :
        Forwarded to ``breakout_fraction_bns``.

    Returns
    -------
    dict[str, dict]
        ``{class_label: {'med': float, 'lo': float, 'hi': float}}``; a class
        with zero weighted population yields NaN entries.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if eos_models is None:
        eos_models = EOS_MODELS

    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    weights = np.asarray(weights, dtype=float)

    prompt_masks = {k: v for k, v in class_masks.items() if k not in hmns_labels}
    hmns_masks = {k: v for k, v in class_masks.items() if k in hmns_labels}

    # Per-EOS percentile triples, collected per class then enveloped.
    per_eos = {label: {"med": [], "lo": [], "hi": []} for label in class_masks}

    for eos_name in eos_models:
        R1 = ns_radius_from_eos(m1, eos_name)
        R2 = ns_radius_from_eos(m2, eos_name)
        M_disk = bns_disk_mass(m1, m2, R1_km=R1, R2_km=R2)
        M_dyn = bns_dynamical_ejecta(m1, m2, R1_km=R1, R2_km=R2)
        M_ej_hmns = M_dyn + hmns_wind_frac * M_disk

        results = {}
        if prompt_masks:
            results.update(
                breakout_fraction_bns(
                    prompt_masks,
                    M_disk,
                    M_dyn,
                    weights,
                    theta_j_by_class=theta_j_by_class,
                    n_draws=n_draws,
                    rng=rng,
                )
            )
        if hmns_masks:
            results.update(
                breakout_fraction_bns(
                    hmns_masks,
                    M_disk,
                    M_ej_hmns,
                    weights,
                    theta_j_by_class=theta_j_by_class,
                    n_draws=n_draws,
                    rng=rng,
                )
            )

        for label, band in results.items():
            per_eos[label]["med"].append(band["med"])
            per_eos[label]["lo"].append(band["lo"])
            per_eos[label]["hi"].append(band["hi"])

    out = {}
    for label, triples in per_eos.items():
        meds = np.asarray(triples["med"], dtype=float)
        los = np.asarray(triples["lo"], dtype=float)
        his = np.asarray(triples["hi"], dtype=float)
        if meds.size == 0 or np.all(np.isnan(meds)):
            out[label] = {"med": float("nan"), "lo": float("nan"), "hi": float("nan")}
            continue
        out[label] = {
            "med": float(np.nanmedian(meds)),
            "lo": float(np.nanmin(los)),
            "hi": float(np.nanmax(his)),
        }
    return out


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


def bootstrap_class_fractions(
    class_masks, weights, n_boot=1000, rng=None, percentiles=(16.0, 50.0, 84.0)
):
    """Resampling-bootstrap credible interval on weighted class fractions.

    The class fraction ``f_c = sum_{i in c} w_i / sum_i w_i`` is a ratio of
    weighted sums; its sampling uncertainty under the finite STROOPWAFEL
    draw is obtained here by a nonparametric bootstrap rather than the
    closed-form weighted-Poisson 1-sigma of ``weighted_frac_err``.  Each of
    ``n_boot`` replicates resamples the ``N`` systems with replacement and
    recomputes every class fraction on the resampled weights; the requested
    percentiles of the replicate distribution give the interval.  To leading
    order the bootstrap SE matches ``weighted_frac_err`` (verified in
    ``tests/unit/test_rates.py``); the bootstrap is the version a referee
    asks for because it does not assume the Poisson/large-N limit and
    captures the (1 - f) ratio structure exactly.

    Replicates are looped (not stacked into one ``(n_boot, N)`` array) so
    peak memory stays at ``O(N)`` even for the BHNS sample (``N ~ 1.5e6``).

    Parameters
    ----------
    class_masks : 2-D bool array, shape (n_class, N)
        Row ``c`` is the membership of class ``c`` over the ``N`` systems.
        Rows need not be mutually exclusive, but for disjoint classes the
        per-replicate fractions sum to one.
    weights : 1-D array, shape (N,)
        STROOPWAFEL weights of the full population.
    n_boot : int, optional
        Number of bootstrap replicates (default 1000).
    rng : numpy.random.Generator, optional
        Defaults to ``np.random.default_rng(0)`` for reproducibility.
    percentiles : sequence of float, optional
        Percentiles returned per class (default the 16/50/84 central
        68 percent interval).

    Returns
    -------
    ndarray, shape (n_class, len(percentiles))
        Per-class percentiles of the bootstrap fraction distribution.
        All-NaN when the population is empty or its weight sum is
        non-positive.
    """
    masks = np.atleast_2d(np.asarray(class_masks, dtype=bool))
    w = np.asarray(weights, dtype=float)
    n_class, n_sys = masks.shape
    pct = np.asarray(percentiles, dtype=float)
    if masks.shape[1] != w.shape[0]:
        raise ValueError("class_masks and weights disagree on the system count")
    if n_sys == 0 or not np.isfinite(w.sum()) or w.sum() <= 0.0:
        return np.full((n_class, pct.size), np.nan)

    if rng is None:
        rng = np.random.default_rng(0)

    reps = np.empty((n_boot, n_class), dtype=float)
    masks_f = masks.astype(float)
    for b in range(n_boot):
        idx = rng.integers(0, n_sys, size=n_sys)
        wb = w[idx]
        denom = wb.sum()
        # masks_f[:, idx] @ wb is the per-class resampled weight sum.
        reps[b] = (masks_f[:, idx] * wb).sum(axis=1) / denom
    return np.percentile(reps, pct, axis=0).T


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


BROEKGAARDEN21_SAMPLING = {
    "m1_min": 5.0,
    "m1_max": 150.0,
    "m2_min": 0.1,
    "f_bin": 1.0,
    "n_binaries_per_Z": 1.0e6,
    "n_metallicities": 53,
}
"""COMPAS initial-condition sampling of the Broekgaarden+ 2021 grid, their
Table 1: primary mass drawn from Kroupa (2001) over [5, 150] Msun, flat mass
ratio with m2 >= 0.1 Msun, binary fraction corrected to f_bin = 1
(arXiv:2103.02608 Sec. 2.2), about a million binaries at each of the 53
log-uniform metallicity grid points."""


def star_forming_mass_per_binary(
    m1_min=None, m1_max=None, m2_min=None, f_bin=None, m_lo_full=0.01, m_hi_full=200.0
):
    """Universe star-forming mass represented by one COMPAS binary [Msun].

    COMPAS evolves only the sliver of the IMF that makes DCO progenitors, so
    every simulated binary stands in for a much larger amount of star
    formation.  Following the COMPAS
    ``totalMassEvolvedPerZ.analytical_star_forming_mass_per_binary_using_kroupa_imf``
    convention (a flat mass ratio, so a fraction ``1 - m2_min / m1`` of drawn
    primaries has an admissible companion)::

        M_SF per binary = (1 + f_bin / 2) <m>  /  [f_bin * P(m1 in range, m2 >= m2_min)]

    The ``1 + f_bin / 2`` is the companion: at f_bin = 1 every system is a
    binary whose secondary averages half the primary's IMF mass, so the
    Universe carries 1.5 mean stellar masses per system.  Dropping it, and
    dividing by the bare primary fraction, is what gives the too-small
    52.1 Msun instead of 79.1 Msun for the Broekgaarden+ 2021 sampling.

    Defaults are the Broekgaarden+ 2021 values in ``BROEKGAARDEN21_SAMPLING``
    and return 79.11 Msun.
    """
    s = BROEKGAARDEN21_SAMPLING
    m1_min = s["m1_min"] if m1_min is None else m1_min
    m1_max = s["m1_max"] if m1_max is None else m1_max
    m2_min = s["m2_min"] if m2_min is None else m2_min
    f_bin = s["f_bin"] if f_bin is None else f_bin

    total_mass, _ = quad(lambda m: m * kroupa_imf(m), m_lo_full, m_hi_full)
    total_number, _ = quad(kroupa_imf, m_lo_full, m_hi_full)
    mean_star_mass = total_mass / total_number

    n_bin, _ = quad(lambda m: kroupa_imf(m) * (1.0 - m2_min / m), m1_min, m1_max)
    f_bin_in_compas = f_bin * n_bin / total_number

    return (1.0 + 0.5 * f_bin) * mean_star_mass / f_bin_in_compas


def simulation_star_forming_mass(n_binaries_per_Z=None, n_metallicities=None, **kwargs):
    """Total star-forming mass a Broekgaarden+ 2021 run represents [Msun].

    ``n_binaries_per_Z * n_metallicities * star_forming_mass_per_binary()``,
    i.e. 53 * 1e6 * 79.11 = 4.193e9 Msun for the fiducial grid.  This is the
    denominator of the formation yield ``R_form = dN_form / dM_SFR`` that
    Broekgaarden+ 2021 Fig. 5 and Broekgaarden+ 2022 Fig. 1 plot: digitising
    all 53 model-A points of their Fig. 1 recovers 77.2 Msun per binary for
    NSNS and 78.4 for BHNS against the 79.11 here.

    It is a property of the sampler, so it is the same for every model of the
    grid and the same for the BNS and BHNS runs.  Prefer it over
    ``calibrate_mean_mass_evolved`` wherever the quantity wanted is a yield
    rather than a rate; the calibration back-derives a mass from the file's
    ``w_000`` local-rate anchor and so absorbs any mismatch between this
    project's cosmic integration and the upstream 2021 pipeline into the
    normalisation (on Model A that is 3 per cent for BHNS but a factor 1.50
    for BNS).
    """
    s = BROEKGAARDEN21_SAMPLING
    n_bin = s["n_binaries_per_Z"] if n_binaries_per_Z is None else n_binaries_per_Z
    n_Z = s["n_metallicities"] if n_metallicities is None else n_metallicities
    return n_bin * n_Z * star_forming_mass_per_binary(**kwargs)


def verify_mean_mass_evolved(
    m_lo_full=0.01, m_hi_full=200.0, m_lo_prim=5.0, m_hi_prim=150.0, mean_mass_evolved=None
):
    """Analytically verify a MEAN_MASS_EVOLVED value via Kroupa IMF.

    ``N_sim_implied`` divides by ``star_forming_mass_per_binary`` (79.11 Msun
    for the Broekgaarden+ 2021 sampling), so a calibrated mass that matches
    the run configuration returns ``53e6`` binaries.  The earlier version
    divided by ``mean_star_mass / f_primary`` = 52.1 Msun, which counts the
    primary but not its companion; that made the BNS calibration read as
    1.01x the Table 1 value when the mass behind it is a factor 1.50 short.

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

    mass_per_binary = star_forming_mass_per_binary(
        m1_min=m_lo_prim, m1_max=m_hi_prim, m_lo_full=m_lo_full, m_hi_full=m_hi_full
    )

    return {
        "mean_star_mass": mean_star_mass,
        "f_primary": f_primary,
        "mass_per_drawn_primary": mean_star_mass / f_primary,
        "star_forming_mass_per_binary": mass_per_binary,
        "N_sim_implied": mean_mass_evolved / mass_per_binary,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EOS sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════════
def compute_eos_sensitivity(
    m1,
    m2,
    weights,
    eos_models=None,
    k_thresh=None,
    q_thresh=None,
    hmns_factor=None,
    require_cold_stable=True,
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
    require_cold_stable : bool, optional
        When True (default), each EOS row first drops systems with a
        component above that EOS's ``M_TOV`` via
        ``grb_physics.cold_stable_ns_mask`` (COMPAS types NSs up to its
        own ``M_NS_max``, which exceeds ``M_TOV``), then normalises the
        class fractions over the retained weight.  The cut is
        EOS-specific, so it lives inside the sweep rather than before the
        call.  Set False to classify every system regardless of
        cold-stability (the legacy behaviour).

    Returns
    -------
    pandas.DataFrame
        Indexed by EOS name with columns
        ``[M_TOV, R_1p4, M_thresh, hmns_split, total_weight,
        dropped_weight_frac]`` plus one column per Gottlieb class
        carrying the weighted fraction (in [0, 1]).  The class fractions
        in each row sum to 1 (normalised over the cold-stable systems
        when ``require_cold_stable``).
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
        cold_stable_ns_mask,
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
        if require_cold_stable:
            keep = cold_stable_ns_mask(m1, m2, m_tov=m_tov)
        else:
            keep = np.ones(m1.shape, dtype=bool)
        w_keep = float(w[keep].sum())
        if w_keep <= 0:
            raise ValueError(
                f"EOS {name}: no cold-stable weight retained at M_TOV={m_tov}; "
                f"cannot normalise class fractions."
            )
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
            "dropped_weight_frac": 1.0 - w_keep / w_total,
        }
        for label, mask in cls.items():
            row[label] = float(w[mask & keep].sum() / w_keep)
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


T_SWIFT_BRIGHT_YR = 17.1
"""Swift-BAT observing envelope for the Fong et al. (2022) BRIGHT short-GRB
sample, ApJ 940, 56 (arXiv:2206.01763) Sec. 3: BAT operated from Swift's
launch in 2004-11 through the BRIGHT cutoff at 2021-12 ("107 events
discovered over 2004-2021", p. 4 of the paper), roughly 17.1 detector-frame
years.  Used to convert observed BRIGHT burst counts to an all-sky rate
density (see the Section 6c comparison in ``grb_main.ipynb``).
"""


F_SKY_BAT = 1.0 / 6.0
"""Effective Swift-BAT sky fraction Omega_BAT / 4pi.  Ghirlanda et al.
(2016, A&A 594, A84) Sec. 3.2 adopt this value for the observed-sGRB rate
integral; consistent with the Lien et al. (2016, ApJ 829, 7) Swift-BAT
triggered-GRB catalog effective coverage.  Combined with T_SWIFT_BRIGHT_YR
in the Section 6c counts-to-rate-density conversion.
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
    "Ghirlanda+ 2016 (model a)": {
        "R_obs": 0.20,
        "R_obs_lo": 0.13,
        "R_obs_hi": 0.24,
        "note": (
            "A&A 594, A84 Sec. 7: rho_0,a = 0.20 +0.04/-0.07 Gpc^-3 yr^-1 for "
            "their model (a).  Their alternative model (c) gives "
            "rho_0,c = 0.8 +0.3/-0.15, so the two together span 0.13 to 1.1; "
            "model (a) is recorded here and (c) is the spread to quote with it."
        ),
    },
    "Ghirlanda+ 2016 (model c)": {
        "R_obs": 0.8,
        "R_obs_lo": 0.65,
        "R_obs_hi": 1.1,
        "note": "A&A 594, A84 Sec. 7: rho_0,c = 0.8 +0.3/-0.15 Gpc^-3 yr^-1.",
    },
}

# Colombo et al. (2022) ApJ 937, 79 is deliberately NOT in the dict above.
# It was previously entered as an observed sGRB rate of 3.6 [1.8, 6.5]
# Gpc^-3 yr^-1.  Those numbers are from their Table 1, "Assumed Detection
# Limits and Predicted Detection Rates in Our Observing Scenarios", whose
# "Rate" row is predicted *detections per year* for a given instrument and
# observing scenario, not a local rate density: wrong quantity and wrong
# units.  The local rate density Colombo et al. actually normalise to is
# R_0 = 347 +536/-256 Gpc^-3 yr^-1 (their Appendix A.3), and that is an
# *intrinsic NSNS merger* rate, so it does not belong in a dict of
# beaming-limited observed sGRB rates either.  It is kept here as a
# reference constant instead.
COLOMBO22_NSNS_LOCAL_RATE = {
    "R": 347.0,
    "R_lo": 91.0,
    "R_hi": 883.0,
    "note": (
        "ApJ 937, 79 Appendix A.3: intrinsic NSNS local merger-rate density "
        "R_0 = 347 +536/-256 Gpc^-3 yr^-1, normalised to reproduce the "
        "observed number of significant NSNS detections.  Intrinsic, not "
        "beaming-limited: compare against compute_merger_rate directly, the "
        "same convention as LVK_GWTC5_LOCAL_RATES."
    ),
}


# Per-class observed-rate references for the beaming comparator.  No
# paper publishes a class-by-class rate; each entry is the closest
# observed (beaming-limited, local) handle for that class, and the
# 'caveat' field documents the mapping choice.
OBSERVED_RATES_BY_CLASS = {
    "sbGRB + blue KN": {
        "R_obs": OBSERVED_SGRB_RATES["Wanderman & Piran 2015"]["R_obs"],
        "R_obs_lo": OBSERVED_SGRB_RATES["Wanderman & Piran 2015"]["R_obs_lo"],
        "R_obs_hi": OBSERVED_SGRB_RATES["Wanderman & Piran 2015"]["R_obs_hi"],
        "reference": "Wanderman and Piran 2015, MNRAS 448, 3026",
        "caveat": "Observed, beaming-limited sGRB rate; sbGRBs dominate the "
        "local-volume short-GRB population.  Previously mapped to Colombo+ "
        "2022, whose entry was a predicted detection count rather than a "
        "rate density.",
    },
    "lbGRB + red KN (HMNS)": {
        "R_obs": OBSERVED_SGRB_RATES["Ghirlanda+ 2016 (model a)"]["R_obs"],
        "R_obs_lo": OBSERVED_SGRB_RATES["Ghirlanda+ 2016 (model a)"]["R_obs_lo"],
        "R_obs_hi": OBSERVED_SGRB_RATES["Ghirlanda+ 2016 (model a)"]["R_obs_hi"],
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


# Previous-catalog (GWTC-4.0) intrinsic local merger-rate bands, kept as a
# conservative fallback so the LVK comparison degrades gracefully if the
# GWTC-5.0 population analysis (a 2026 preprint) shifts.  GWTC-4.0 added the NS
# detections that GWTC-5.0 did not extend, so its NS-side 90 percent CRs are
# looser than GWTC-5.0's; BNS and NSBH at z = 0 in Gpc^-3 yr^-1 (Abac+ 2025,
# GWTC-4.0 Population Properties, arXiv:2508.18083).  Compare against
# ``compute_merger_rate(...)[iz0]`` directly, as for the GWTC-5.0 bands.
LVK_GWTC4_LOCAL_RATES = {
    "BNS": {
        "R_lo": 7.6,
        "R_hi": 250.0,
        "reference": "LVK 2025, GWTC-4.0 (arXiv:2508.18083)",
        "note": "90% CR at z=0; looser than GWTC-5.0 on the NS side",
    },
    "NSBH": {
        "R_lo": 9.1,
        "R_hi": 84.0,
        "reference": "LVK 2025, GWTC-4.0 (arXiv:2508.18083)",
        "note": "90% CR at z=0; called NSBH (= BHNS in this project); looser "
        "than the tightened GWTC-5.0 band",
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
    z,
    R0=45.0,
    z_peak=0.9,
    sigma_lo=0.39,
    sigma_hi=0.26,
    rho_0=4.1,
    rho_0_minus=1.4,
    rho_0_plus=1.8,
    sfr2_sigma_broadening=1.5,
):
    """Observed sGRB comoving rate density vs redshift from
    Wanderman & Piran (2015), MNRAS 448, 3026.

    Implements Eq. (9), the piecewise-exponential fit to the SFR1
    log-normal-delay maximum-likelihood R(z):

        R(z) = R0 * exp(+(z - z_peak) / sigma_lo)   for z <= z_peak
        R(z) = R0 * exp(-(z - z_peak) / sigma_hi)   for z >  z_peak

    with the paper's coefficients R0 = 45 Gpc^-3 yr^-1, z_peak = 0.9,
    sigma_lo = 0.39, sigma_hi = 0.26.  Evaluated at z = 0 this returns
    R(0) = 45 * exp(-0.9/0.39) ~ 4.47 Gpc^-3 yr^-1, which sits inside
    the paper's Eq. (10) canonical local rate

        rho_0 = 4.1 (+1.8 / -1.4) +/- 0.5 Gpc^-3 yr^-1

    (internal 1sigma +/- SFR1<->SFR2 systematic; at L_min = 5e49 erg/s).

    Output is the comoving rate density of *detectable* (beaming- and
    luminosity-function-folded) sGRBs, not the intrinsic merger rate,
    so it is the direct target for ``beamed_rate(...)`` model curves.

    Returns two paper-anchored uncertainty bands separately, so a plot
    can render them distinctly and the reader can tell rate-anchor
    from shape systematics apart:

      R_rate_lo, R_rate_hi:
        Rate normalization 1sigma from WP15 Eq. (10) *internal* only
        (rho_0 = 4.1 +1.8/-1.4).  Same shape as R_best, scaled by
        (rho_0 +/- delta_rho_0) / rho_0.  Does NOT fold in SFR1<->SFR2:
        the paper's abstract quote rho_0 = 4.1 +2.3/-1.9 adds the
        SFR-model spread linearly, which would double-count with the
        SFR-shape band below.  L_min is fixed at 5e49 erg/s (Eq. 10);
        allowing L_min to vary would move the anchor by ~1 dex
        (Table 4), which is a separate systematic no plotted band
        should cover.

      R_shape_sfr2:
        SFR-shape systematic.  The fiducial shape (sigma_lo, sigma_hi)
        is the SFR1 log-normal-delay peak (FWHM Delta_z ~ 0.4, WP15
        Sec. 5).  SFR2 (Planck 2013 halo model) broadens the peak to
        FWHM ~ 0.6; the ratio 0.6/0.4 = 1.5 sets the default
        ``sfr2_sigma_broadening``.  Both slopes are broadened by the
        same factor.  R0 is held at 45 so the SFR2 systematic isolates
        the shape change, not the normalization change.  This is a
        one-sided band: WP15 report no fit narrower than SFR1.

    R_best keeps the exact WP15 Eq. (9) form (R0 = 45, fiducial shape).

    Not folded into any plotted band, but relevant to citations:
      - t_d = 2.9 +/- 0.4 Gyr (SFR1) and 3.9 +0.4/-0.5 Gyr (SFR2) from
        WP15 Table 2 shift z_peak by ~+/-0.1 to 0.15; z_peak is left
        fixed here because Eq. 9 does not resolve t_d separately.
      - The z > 1.2 sample truncation (WP15 Sec. 4) can only broaden
        the peak, not sharpen it; captured qualitatively by the
        SFR2-broadened band.

    Returns dict with 'R_best', 'R_rate_lo', 'R_rate_hi',
    'R_shape_sfr2' arrays (same shape as z).
    """
    z = np.asarray(z, dtype=float)
    dz = z - z_peak

    def _shape(sl, sh):
        return np.where(dz <= 0.0, np.exp(+dz / sl), np.exp(-dz / sh))

    shape_fid = _shape(sigma_lo, sigma_hi)
    shape_sfr2 = _shape(sigma_lo * sfr2_sigma_broadening, sigma_hi * sfr2_sigma_broadening)

    R_best = R0 * shape_fid
    R_rate_lo = R_best * (rho_0 - rho_0_minus) / rho_0
    R_rate_hi = R_best * (rho_0 + rho_0_plus) / rho_0

    return {
        "R_best": R_best,
        "R_rate_lo": R_rate_lo,
        "R_rate_hi": R_rate_hi,
        "R_shape_sfr2": R0 * shape_sfr2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Modern short-GRB rate literature (replaces WP15 as the Section 6c overlay).
#
# Three papers, three roles:
#   - Pracchia and Salafia (2026), A&A, arXiv:2601.03861: intrinsic rho_dot(z)
#     as a numerical convolution of Madau and Fragos (2017) SFR with a DTD
#     kernel; four Table 2 posteriors (ELF/QUSJ x power-law/log-normal DTD).
#     Their preferred short delays (⟨tau_d⟩ = 77-116 Myr) invert the WP15
#     ~Gyr scale.  Provided here as a computable curve.
#   - Du et al. (2025), ApJL 987, L13: observer-frame Swift/BAT R_SGRB(z)
#     fit (their Eq. 16 power-law + Gaussian in 1+z), but the numerical
#     coefficients A1, A2, B, C, D are only rendered as curves in Figure 4
#     and are not published in Table 5 or the article text.  Table 5 gives
#     local rates across five SFR + DTD combinations per detector; only
#     those tabulated values are exposed here.
#   - Chong-Yu Gao (2025), MNRAS 542, 1841: Fermi/GBM Bayesian fit; R(z=0)
#     with three DTD family posteriors, but no published R(z > 0).  Local
#     rate is exposed as a Fermi/GBM anchor.
# ═══════════════════════════════════════════════════════════════════════════

# Madau and Fragos (2017), ApJ 840, 39; Pracchia & Salafia (2026) Eq. 10.
# Cosmic star-formation-rate density
#   psi(z) prop to (1+z)^a_psi / (1 + ((1+z)/(1+z_psi))^b_psi)
# with (a_psi, b_psi, z_psi) = (2.6, 6.2, 2.2).
MADAU_FRAGOS_2017_SFR = {
    "a_psi": 2.6,
    "b_psi": 6.2,
    "z_psi": 2.2,
}

# Pracchia and Salafia (2026) Table 2 posteriors on log10(R_0 / Gpc^-3 yr^-1)
# with 90 percent credible intervals.  R_0 is the SGRB local rate density
# defined by their Eq. 4 (intrinsic source-frame; Eq. 6 is the posterior
# PDF that infers it).  ELF = empirical luminosity function (broken
# power-law, their Eq. 12); QUSJ = quasi-universal structured jet
# (Salafia et al. 2023).  DTD parameters follow Eq. 7 (power-law) and
# Eq. 8 (log-normal).  Mean delay time is Table 2 col. "⟨tau_d⟩" in Myr.
PRACCHIA_SALAFIA_2026_ELF_POWERLAW = {
    "log10_R0": 1.92,
    "log10_R0_plus90": 1.50,
    "log10_R0_minus90": 0.92,
    "alpha_tau": 3.20,
    "tau_min_Gyr": 0.03,
    "mean_tau_d_Myr": 77,
}
PRACCHIA_SALAFIA_2026_ELF_LOGNORMAL = {
    "log10_R0": 2.60,
    "log10_R0_plus90": 1.23,
    "log10_R0_minus90": 1.49,
    "mu_tau_Gyr": 0.08,
    "sigma_tau": 0.11,
    "mean_tau_d_Myr": 92,
}
PRACCHIA_SALAFIA_2026_QUSJ_POWERLAW = {
    "log10_R0": 2.57,
    "log10_R0_plus90": 0.67,
    "log10_R0_minus90": 0.63,
    "alpha_tau": 3.33,
    "tau_min_Gyr": 0.04,
    "mean_tau_d_Myr": 105,
}
PRACCHIA_SALAFIA_2026_QUSJ_LOGNORMAL = {
    "log10_R0": 2.25,
    "log10_R0_plus90": 0.53,
    "log10_R0_minus90": 0.50,
    "mu_tau_Gyr": 0.09,
    "sigma_tau": 0.14,
    "mean_tau_d_Myr": 116,
}
PRACCHIA_SALAFIA_2026_ROWS = {
    "ELF_powerlaw": PRACCHIA_SALAFIA_2026_ELF_POWERLAW,
    "ELF_lognormal": PRACCHIA_SALAFIA_2026_ELF_LOGNORMAL,
    "QUSJ_powerlaw": PRACCHIA_SALAFIA_2026_QUSJ_POWERLAW,
    "QUSJ_lognormal": PRACCHIA_SALAFIA_2026_QUSJ_LOGNORMAL,
}

# Du et al. (2025), ApJL 987, L13, Table 5.  Observer-frame local rates
# rho_0 (Gpc^-3 yr^-1) per detector across five SFR + delay-time
# combinations (columns Yuksel+2008 / Gauss-Yuksel / Lognormal-Yuksel /
# Power-law-Yuksel / Madau+2014).  L_min per detector is the last column.
# Beaming: theta_j = 9.8 +/- 1.3 deg (Sec. 4.4, citing Soderberg et al.
# 2006 and Z. B. Zhang et al. 2018, PASP 130, 054202; not the more widely
# cited B. Zhang 2018 textbook) with f_B = 1 - cos(theta_j) applied only
# to local rates, not to R(z).
DU25_LOCAL_RATES_SWIFT_BAT = {
    "Yuksel2008_undelayed": (0.60, 0.18),
    "Gauss_Yuksel": (0.56, 0.16),
    "Lognormal_Yuksel": (0.57, 0.15),
    "PowerLaw_Yuksel": (0.97, 0.26),
    "Madau2014_undelayed": (0.73, 0.18),
}
DU25_LOCAL_RATES_FERMI_GBM = {
    "Yuksel2008_undelayed": (4.35e-3, 0.74e-3),
    "Gauss_Yuksel": (7.56e-3, 0.90e-3),
    "Lognormal_Yuksel": (1.31e-2, 0.14e-2),
    "PowerLaw_Yuksel": (1.43e-2, 0.19e-2),
    "Madau2014_undelayed": (6.18e-3, 0.98e-3),
}
DU25_LOCAL_RATES_KONUS_WIND = {
    "Yuksel2008_undelayed": (2.41e-2, 1.04e-2),
    "Gauss_Yuksel": (2.30e-2, 0.94e-2),
    "Lognormal_Yuksel": (2.45e-2, 0.88e-2),
    "PowerLaw_Yuksel": (4.02e-2, 1.51e-2),
    "Madau2014_undelayed": (2.92e-2, 0.78e-2),
}
DU25_L_MIN_SWIFT_BAT_ERG_S = 2.57e48
DU25_L_MIN_FERMI_GBM_ERG_S = 1.97e50
DU25_L_MIN_KONUS_WIND_ERG_S = 2.91e50
DU25_THETA_J_DEG = 9.8

# Chong-Yu Gao (2025), MNRAS 542, 1841, Table 2 Gaussian-DTD row (their
# preferred DIC-lowest model).  R(z=0) is beamed / detector-selected via
# their fitted eta = f_b * C.  Fermi/GBM-only sample (N = 478); not
# plotted in Section 6c because the axis there is Swift/BAT-normalised.
GAO25_LOCAL_RATE_FERMI_GBM = {
    "R0": 1.37,
    "R0_plus": 0.30,
    "R0_minus": 0.27,
    "tau_G_Gyr": 1.14,
    "sigma_G_Gyr": 0.26,  # upper bound (Table 2 caption)
    "detector": "Fermi/GBM",
    "note": "Not overlaid on the Swift/BAT-normalised axis; text-only anchor.",
}


def _madau_fragos_2017_sfr(z, params=MADAU_FRAGOS_2017_SFR):
    """Madau and Fragos (2017) cosmic SFR density, arbitrary normalisation."""
    a = params["a_psi"]
    b = params["b_psi"]
    zp = params["z_psi"]
    x = 1.0 + np.asarray(z, dtype=float)
    return x**a / (1.0 + (x / (1.0 + zp)) ** b)


def _hubble_time_Gyr(H0):
    """Hubble time 1/H0 in Gyr for arbitrary H0 in km/s/Mpc.

    1 km/s/Mpc = 1.0227e-3 / Gyr, so tH_Gyr = 1 / (H0 * 1.0227e-3).
    """
    return 1.0 / (H0 * 1.0227121650e-3)


# Cosmology presets available to callers who want to reproduce a paper's
# convention exactly.  Planck 2015 is the project pin and the default for
# every cosmology-integration path.
COSMO_PLANCK15 = {"H0": H0_PLANCK15, "Omega_m": OMEGA_M_PLANCK15, "Omega_L": OMEGA_L_PLANCK15}
COSMO_PLANCK14 = {"H0": 67.30, "Omega_m": 0.315, "Omega_L": 0.685}  # Planck 2014, A&A 571, A16
COSMO_MADAU14 = {
    "H0": 70.00,
    "Omega_m": 0.3,
    "Omega_L": 0.7,
}  # Madau & Dickinson (2014) "cosmic concordance"


def _lookback_time_Gyr(z, cosmology=None, n_sub=2000):
    """Look-back time in Gyr for flat LCDM, arbitrary cosmology.

    t_LB(z) = (1/H0) integral_0^z dz' / [(1+z') E(z')]
    with E(z) = sqrt(Omega_m (1+z)^3 + Omega_L).  Default cosmology is
    the module Planck 2015 pin (matches COMPAS FastCosmicIntegration).
    Pass a dict with keys 'H0' (km/s/Mpc), 'Omega_m', 'Omega_L' to
    override, or one of the COSMO_PLANCK14 / COSMO_MADAU14 presets.
    """
    if cosmology is None:
        cosmology = COSMO_PLANCK15
    H0 = cosmology["H0"]
    Om = cosmology["Omega_m"]
    OL = cosmology["Omega_L"]
    z = np.atleast_1d(np.asarray(z, dtype=float))
    zmax = float(z.max())
    if zmax <= 0.0:
        return np.zeros_like(z)
    z_fine = np.linspace(0.0, zmax, n_sub)
    E = np.sqrt(Om * (1.0 + z_fine) ** 3 + OL)
    integrand = 1.0 / ((1.0 + z_fine) * E)
    t_of_z_fine = _hubble_time_Gyr(H0) * np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z_fine))]
    )
    return np.interp(z, z_fine, t_of_z_fine)


# Backwards-compatible aliases (Planck 2015 defaults).
def _hubble_time_Gyr_planck15():
    """Hubble time in Gyr on the module Planck 2015 pin."""
    return _hubble_time_Gyr(H0_PLANCK15)


_lookback_time_Gyr_planck15 = _lookback_time_Gyr


def _dtd_pracchia_salafia_2026(tau_Gyr, row_params):
    """Pracchia & Salafia (2026) DTD kernel P(tau) in Gyr^-1.

    Eq. 7 (power-law) if row_params carries ``alpha_tau``; Eq. 8
    (log-normal) if it carries ``mu_tau_Gyr``.  Return is unnormalised
    (the outer convolution rescales to R_0), so the log-normal 1/sqrt(2pi)
    factor is skipped for numerical stability at small tau.
    """
    tau = np.asarray(tau_Gyr, dtype=float)
    if "alpha_tau" in row_params:
        alpha = row_params["alpha_tau"]
        tau_min = row_params["tau_min_Gyr"]
        out = np.zeros_like(tau)
        mask = tau >= tau_min
        out[mask] = tau[mask] ** (-alpha)
        return out
    # log-normal DTD (Eq. 8)
    mu = row_params["mu_tau_Gyr"]
    sig = row_params["sigma_tau"]
    tau_safe = np.clip(tau, 1e-6, None)
    valid = tau > 0.0
    out = np.zeros_like(tau)
    out[valid] = (
        np.exp(-0.5 * ((np.log(tau_safe[valid]) - np.log(mu)) / sig) ** 2) / tau_safe[valid]
    )
    return out


def pracchia_salafia_2026_Rz(
    z,
    row="ELF_powerlaw",
    cosmology=None,
    integration_z_max=15.0,
    n_zprime=4000,
    n_tau=400,
    tau_min_floor_Gyr=1e-3,
    tau_max_Gyr=14.0,
):
    """Intrinsic short-GRB rate density, Pracchia & Salafia (2026) Eq. 9.

    Numerical convolution of the Madau and Fragos (2017) SFR (their Eq. 10)
    with the delay-time distribution family selected by ``row``, normalised
    so that R(z=0) equals 10**log10_R0 from Table 2.  R(z) is intrinsic
    (per merger, all orientations); the caller applies the beaming factor
    ``(1 - cos theta_j)`` before comparing against a Swift-like observed
    rate density.

    Integration is done as a change of variables from z' to tau (delay
    time), because the DTDs in Table 2 are extremely short-tailed
    (⟨tau_d⟩ = 77 to 116 Myr) and a uniform z' grid under-resolves the
    kernel at low z_out.  The tau grid is log-spaced from
    ``max(tau_min_floor_Gyr, DTD tau_min)`` to ``tau_max_Gyr``; tau values
    that map to z' > ``integration_z_max`` are dropped from the integral.

    ``row`` is one of ``ELF_powerlaw``, ``ELF_lognormal``, ``QUSJ_powerlaw``,
    ``QUSJ_lognormal`` (keys of ``PRACCHIA_SALAFIA_2026_ROWS``).  ELF uses
    an empirical broken-power-law luminosity function (their Eq. 12); QUSJ
    imposes a quasi-universal structured jet (Salafia et al. 2023); the
    top-hat beaming multiplier is strictly consistent with ELF but not with
    QUSJ (structured-jet inclination is folded into the luminosity function
    rather than a single opening angle).

    ``cosmology`` is optional; default is the module Planck 2015 pin
    (``COSMO_PLANCK15``), matching the rest of the project.  Pracchia and
    Salafia do not explicitly quote a cosmology and inherit it from Madau
    and Fragos (2017), who use the Madau and Dickinson (2014) "concordance"
    (``COSMO_MADAU14``: H0=70, Om=0.3).  The R(z=0) normalisation cancels
    the H0 dependence, and the short DTDs (< 120 Myr mean) make the drift
    from Planck 2015 to Madau 2014 numerically < 0.15 percent across
    z <= 3 with the peak invariant at z = 2.007 (verified by direct
    computation; see the anchor tests).

    Returns a dict with:
      - ``R_best``: central curve at log10_R0 (intrinsic Gpc^-3 yr^-1).
      - ``R_rate_lo`` / ``R_rate_hi``: 90 percent CR on log10_R0, shape
        held fixed.  Shape is much better constrained than normalisation,
        so this is the dominant plotted band.
    """
    if row not in PRACCHIA_SALAFIA_2026_ROWS:
        raise ValueError(f"unknown row {row!r}; choose from {list(PRACCHIA_SALAFIA_2026_ROWS)}")
    params = PRACCHIA_SALAFIA_2026_ROWS[row]

    # Precompute look-back time on a dense z' grid; interpolate for the
    # tau -> z' inversion inside the convolution.
    z_p_grid = np.linspace(0.0, integration_z_max, n_zprime)
    t_LB_grid = _lookback_time_Gyr(z_p_grid, cosmology=cosmology)

    # DTD support: log-spaced tau grid from the DTD's characteristic
    # short-time cutoff to ~ age of universe.
    tau_min = max(tau_min_floor_Gyr, params.get("tau_min_Gyr", 0.0))
    tau_grid = np.logspace(np.log10(tau_min), np.log10(tau_max_Gyr), n_tau)
    P_tau = _dtd_pracchia_salafia_2026(tau_grid, params)

    def _rate_at(z_out):
        # rho_dot(z_out) prop to int P(tau) SFR(z'(tau)) dtau, with
        # z'(tau) the redshift whose look-back time is t_LB(z_out) + tau.
        t_zout = float(np.interp(z_out, z_p_grid, t_LB_grid))
        t_zprime = t_zout + tau_grid
        # tau values that push t_zprime past the grid edge (~ age of
        # universe) fall outside the integration range and get zero SFR.
        valid = t_zprime <= t_LB_grid[-1]
        if valid.sum() < 2:
            return 0.0
        z_prime = np.interp(t_zprime[valid], t_LB_grid, z_p_grid)
        sfr = _madau_fragos_2017_sfr(z_prime)
        integrand = P_tau[valid] * sfr
        return float(np.trapezoid(integrand, tau_grid[valid]))

    # Compute R at z = 0 for normalisation, then at every input z.
    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    rate_z0 = _rate_at(0.0)
    if rate_z0 <= 0.0:
        raise RuntimeError(
            f"P&S 2026 convolution at z=0 returned {rate_z0}; check DTD parameters for row {row!r}."
        )
    R_0 = 10.0 ** params["log10_R0"]

    R_unnorm = np.array([_rate_at(zi) for zi in z_in])
    R_best = R_unnorm * (R_0 / rate_z0)

    # Scale to the 90 percent CR bounds on log10(R_0) (shape held fixed).
    R_0_hi = 10.0 ** (params["log10_R0"] + params["log10_R0_plus90"])
    R_0_lo = 10.0 ** (params["log10_R0"] - params["log10_R0_minus90"])
    R_rate_hi = R_unnorm * (R_0_hi / rate_z0)
    R_rate_lo = R_unnorm * (R_0_lo / rate_z0)

    out_shape = np.asarray(z).shape
    return {
        "R_best": R_best.reshape(out_shape) if out_shape else R_best[0],
        "R_rate_lo": R_rate_lo.reshape(out_shape) if out_shape else R_rate_lo[0],
        "R_rate_hi": R_rate_hi.reshape(out_shape) if out_shape else R_rate_hi[0],
    }
