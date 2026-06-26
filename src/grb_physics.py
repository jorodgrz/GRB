"""
Shared physics functions for GRB classification from compact binary mergers.

Foucart et al. (2018) remnant mass formula, Kruger & Foucart (2020)
dynamical ejecta fits, NS equation-of-state helpers, and
Gottlieb et al. (2023, 2024) classification thresholds.

Cosmology
---------
Throughout this pipeline I adopt the cosmological parameters pinned by
COMPAS ``FastCosmicIntegration`` (Planck 2015, TNG-consistent):
  H0 = 67.74 km/s/Mpc,  Omega_m = 0.3089,  Omega_Lambda = 0.6911
All lookback-time to redshift conversions, comoving volumes, and SFR
integrals use these values.  Neijssel et al. (2019) report the slightly
rounded Planck 2015 values (H0 = 67.8, Omega_m = 0.308); the small
offset is an intentional pin to the COMPAS code constants so that
rates computed here match the COMPAS post-processing exactly.  Mixing
with Planck 2018 parameters would introduce ~2 percent inconsistencies
at high z.

Supernova engine
----------------
The COMPAS simulations (Broekgaarden et al. 2021, all five Tier-1 models
A, F, G, J, K) use the Fryer et al. (2012) *delayed* supernova
explosion mechanism (Broekgaarden+ 2021 Table 1; the rapid prescription
is Model I, a variation).  Both delayed and rapid produce a ~1.7 Msun
NS gravitational-mass deficit inherited from the Fryer 2012 Eq. 12-13
baryonic-to-gravitational mass conversion (Broekgaarden+ 2021 footnote
3); :func:`remap_ns_masses_double_gaussian` and
:func:`remap_ns_marginal` close this gap by quantile-remapping to the
Alsing+ 2018 Galactic-NS PDF, following Mandel & Muller (2020).

Jet efficiency caveat
---------------------
The disk-mass-based GRB classifications assume 100% jet launching
efficiency above the disk-mass threshold.  In reality, the jet must
break out of the merger ejecta and the disk must reach the MAD
(magnetically arrested disk) state on a timescale shorter than the
accretion time (Gottlieb 2023).  The Pais et al. (2025) breakout
helpers (:func:`jet_breaks_out_pais` and the ``PAIS_*`` / ``EPS_JET`` /
``V_EJ_BNS`` constants, consumed by
:func:`grb_rates.breakout_fraction_bns`) quantify the breakout step, so
a rate corrected via :func:`grb_rates.apply_bns_jet_breakout` is no
longer a pure upper bound; an uncorrected disk-mass rate still is.
"""

import warnings

import numpy as np
from scipy.special import erf

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
# Disk-mass cuts (MDISK_SHORT, MDISK_LONG) come from Gottlieb et al. (2023,
# arXiv:2309.00038) Sec. 4 and Fig. 6; the live taxonomy that consumes them
# is the Gottlieb et al. (2024) hybrid (see grb_classify.classify_bhns).
# The BNS mass-ratio threshold and the M_CRIT_BNS = 2.8 prompt-collapse
# total-mass anchor come from the same paper's Sec. 3 plus the Bauswein et
# al. (2013) prompt-collapse line.  M_CRIT_BNS is retained as a literature
# anchor (pinned by tests/anchors); the live 2024 hybrid uses M_THRESH.
M_CRIT_BNS = 2.8  # BNS prompt-collapse total mass [Msun]; literature anchor (Gottlieb 2023, Sec. 3)
Q_THRESH_BNS = 1.2  # BNS mass ratio q = M_max / M_min >= 1 (Gottlieb 2023, Sec. 3)
MDISK_SHORT = 0.01  # BHNS Faint-lbGRB / lbGRB-disk split [Msun] (Gottlieb 2023 Fig. 6)
MDISK_LONG = 0.1  # BHNS lbGRB + red KN (BHNS disk) [Msun] (Gottlieb 2023 Fig. 6)

# ---------------------------------------------------------------------------
# Gottlieb et al. (2024) additions
# ---------------------------------------------------------------------------
M_TOV = 2.2
"""Maximum non-rotating NS mass [Msun].  Raaijmakers et al. (2021,
arXiv:2105.06981) combined NICER + GW + KN: 2.23 +0.14/-0.23 (PP),
2.11 +0.29/-0.16 (CS); 2.2 is a central estimate."""

K_THRESH_DEFAULT = 1.27
"""Default ratio M_THRESH / M_TOV.  Chosen as a Gottlieb (2023, Eq. 1
discussion) fiducial so that ``K * M_TOV = 1.27 * 2.2 ~ 2.8 Msun``,
matching the literature ``M_CRIT_BNS = 2.8`` anchor.  This is NOT a
value taken from the Bauswein Table I.  The prompt-collapse threshold
concept itself is from Bauswein et al. (2013, arXiv:1307.5191) and
Bauswein et al. (2020, arXiv:2004.00846), who find EOS-dependent
``k = M_thresh / M_TOV`` in the range ~1.3-1.7 (stiffer EOS -> larger
k); see also Koppel et al. (2019).  Override per-EOS for sensitivity
studies via the ``k_thresh`` kwarg of
``grb_classify.classify_bns_2024`` / ``classify_grid`` (see
``EOS_MODELS`` for tabulated M_crit values per EOS)."""

M_THRESH = K_THRESH_DEFAULT * M_TOV  # 2.794 ~ 2.8 by default
"""Prompt-collapse total-mass threshold [Msun], expressed as
``K_THRESH_DEFAULT * M_TOV`` so EOS sweeps that change ``M_TOV`` also
move ``M_THRESH``.  This is the threshold used by every live
classifier (``classify_bns_2024``, ``classify_grid``).  ``M_CRIT_BNS
= 2.8`` is retained in the module as a literature anchor only."""

HMNS_FACTOR_DEFAULT = 1.2
"""Multiplier on ``M_TOV`` for the long-lived / short-lived HMNS split,
1.2 * M_TOV ~ 2.64 Msun.  Code heuristic, not a Gottlieb (2024) number:
Gottlieb sets the split by HMNS lifetime, not a fixed factor on
``M_TOV``.  The 1.2 fiducial follows the supramassive-remnant argument
of Margalit and Metzger (2017, ApJL 850, L19), and is overrideable
through the ``hmns_factor`` kwarg in ``grb_classify``."""

# ---------------------------------------------------------------------------
# Legacy remnant-to-disk fraction (deprecated; kept for back-compat)
# ---------------------------------------------------------------------------
# The proper decomposition is now M_disk = M_rem - M_dyn, using the
# Kruger & Foucart (2020) dynamical ejecta fit.  F_DISK is retained only
# as a fallback when the legacy f_disk keyword is explicitly passed.
F_DISK = 0.4

# ---------------------------------------------------------------------------
# COMPAS simulation constants
# ---------------------------------------------------------------------------
_MEAN_MASS_EVOLVED_VALUE = 77708655

# ---------------------------------------------------------------------------
# EOS reference models
# ---------------------------------------------------------------------------
# M_crit : BNS prompt-collapse total-mass threshold [Msun]
#          Bauswein et al. (2013, arXiv:1307.5191) Table I
# R_1p4  : NS radius at 1.4 Msun [km]  (Read et al. 2009, Table III)
# M_TOV  : maximum non-rotating NS mass [Msun]  (Read et al. 2009)
EOS_MODELS = {
    "APR4": {"M_crit": 2.88, "R_1p4": 11.1, "M_TOV": 2.20},
    "SFHo": {"M_crit": 2.60, "R_1p4": 11.9, "M_TOV": 2.06},
    "LS220": {"M_crit": 2.72, "R_1p4": 12.7, "M_TOV": 2.04},
    "DD2": {"M_crit": 3.35, "R_1p4": 13.2, "M_TOV": 2.42},
}


# ---------------------------------------------------------------------------
# Chirp mass
# ---------------------------------------------------------------------------
def chirp_mass(m1, m2):
    """Chirp mass (m1 m2)^(3/5) / (m1 + m2)^(1/5) [Msun].

    Symmetric in (m1, m2); vectorises over arrays.
    """
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


# ---------------------------------------------------------------------------
# ISCO
# ---------------------------------------------------------------------------
def r_isco(a_BH):
    """ISCO radius in units of G*M_BH/c^2 (Bardeen et al. 1972).

    Uses np.where so that a_BH = 0 maps to the prograde ISCO (sign = +1)
    without silently mis-handling small negative (retrograde) spins.

    Inputs with ``|a_BH| >= 1`` are unphysical (Kerr spins satisfy
    ``|a| < 1``) and would make ``(1 - a**2)**(1/3)`` complex,
    silently propagating ``complex128`` values through the entire
    Foucart chain.  Such inputs are clipped to +/- (1 - 1e-9) and a
    single aggregated warning is emitted.
    """
    a = np.asarray(a_BH, dtype=float)
    if np.any(np.abs(a) >= 1.0):
        n_bad = int(np.sum(np.abs(a) >= 1.0))
        warnings.warn(
            f"r_isco received {n_bad} spin values with |a| >= 1 "
            f"(max |a|={float(np.max(np.abs(a))):.6f}); clipping to "
            f"+/- (1 - 1e-9). Physical Kerr spins satisfy |a| < 1.",
            stacklevel=2,
        )
    a = np.clip(a, -1.0 + 1e-9, 1.0 - 1e-9)
    Z1 = 1 + (1 - a**2) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    sign = np.where(a >= 0, 1.0, -1.0)
    return 3 + Z2 - sign * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))


# ---------------------------------------------------------------------------
# Neutron star helpers
# ---------------------------------------------------------------------------
def ns_baryon_mass(M_NS):
    """Approximate NS baryon mass from gravitational mass [M_sun].

    M^b ~ M_g + 0.080 * M_g^2 (Gao et al. 2020; Lattimer & Prakash 2001).
    """
    return M_NS + 0.080 * M_NS**2


def ns_radius(M_NS, R_1p4_km=12.0, M_TOV_local=None):
    """Mass-dependent NS radius [km] (heuristic model).

    Phenomenological interpolation NOT derived from a specific EOS table.
    The cubic suppression R = R_1p4*(1 - 0.15*x^3) with x = (M - 1.4) /
    (M_TOV - 1.4) is anchored at M = 1.4 Msun so that R(1.4) = R_1p4_km
    exactly, matching the observational anchor (Raaijmakers et al. 2021,
    arXiv:2105.06981: R_1.4 = 12.18 +0.56/-0.79 km, CS model).  Above
    1.4 Msun the cubic suppression drops R by ~15% at M_TOV.

    Below 1.4 Msun this function returns ``R_1p4_km`` exactly (flat
    plateau).  This is a *modeling choice*, not a derivation: real NS
    R(M) sequences are nearly flat or weakly varying across 1.0 - 1.4
    Msun for most EOSs (small *increase* in R for stiff EOSs, small
    *decrease* for soft), but the qualitative direction is EOS-
    dependent and small in magnitude.  The plateau biases compactness
    by at most a few percent for low-mass NSs but is consequential for
    populations dominated by sub-1.4 Msun NSs (Galactic BNS, the
    sub-Chandrasekhar end of the COMPAS NS distribution).  Use
    ``ns_radius_from_eos()``
    or pass ``R_NS_km`` directly to ``foucart_disk_mass()`` /
    ``bns_disk_mass()`` for EOS-realistic low-mass behaviour.

    Qualitative behaviour:
      - Flat plateau at R_1p4_km for M < 1.4 Msun (modeling choice)
      - Steepening drop from 1.4 Msun to M_TOV (~15% below R_1.4)
      - No neutron star above M_TOV (returns NaN)

    Quantitative comparison at M_NS = 1.4 Msun with default R_1p4_km:
      - This heuristic: 12.0 km (by construction)
      - APR4: 11.1 km  |  SFHo: 11.9 km  |  DD2: 13.2 km
    The ~10% spread in R propagates as ~10% in compactness C_NS, which
    can shift disk masses near the MDISK_SHORT/MDISK_LONG thresholds.
    """
    if M_TOV_local is None:
        M_TOV_local = M_TOV
    M_NS = np.asarray(M_NS, dtype=float)
    # x in (-inf, 1] -- negative below 1.4 Msun, clipped on the upper
    # end so the cubic suppression saturates at M_TOV.  The flat
    # plateau below 1.4 Msun is made explicit via np.where rather than
    # buried in a clip; see the docstring for the modeling rationale.
    x = (M_NS - 1.4) / (M_TOV_local - 1.4)
    R = np.where(M_NS >= 1.4, R_1p4_km * (1.0 - 0.15 * np.clip(x, 0.0, 1.0) ** 3), R_1p4_km)
    R = np.where(M_TOV_local < M_NS, np.nan, R)
    return R


def ns_radius_from_eos(M_NS, eos_name):
    """EOS-anchored NS radius [km] using ``EOS_MODELS`` parameters.

    Calls ``ns_radius`` with R_1p4 and M_TOV from the named EOS,
    ensuring self-consistent compactness for the Foucart formula.

    Parameters
    ----------
    M_NS : float or array
        Gravitational mass [Msun].
    eos_name : str
        Key in ``EOS_MODELS`` (e.g. 'APR4', 'SFHo', 'DD2').
    """
    eos = EOS_MODELS[eos_name]
    return ns_radius(M_NS, R_1p4_km=eos["R_1p4"], M_TOV_local=eos["M_TOV"])


def mcrit_to_r14(mc):
    """Linear interpolation from M_crit [Msun] to R_{1.4} [km].

    Anchored at APR4 (M_crit=2.46, R_{1.4}=11.1 km) and
    DD2 (M_crit=3.35, R_{1.4}=13.2 km) from Read et al. (2008).
    """
    apr4 = EOS_MODELS["APR4"]
    dd2 = EOS_MODELS["DD2"]
    return apr4["R_1p4"] + (
        (dd2["R_1p4"] - apr4["R_1p4"]) / (dd2["M_crit"] - apr4["M_crit"]) * (mc - apr4["M_crit"])
    )


# ---------------------------------------------------------------------------
# NS-mass quantile remap (Mandel & Muller 2020 style)
# ---------------------------------------------------------------------------
# Alsing, Silva and Berti (2018) MNRAS 478, 1377 (arXiv:1810.03548)
# double-Gaussian fit to the Galactic NS mass distribution.  Component 1
# captures the recycled + slow-pulsar peak near 1.34 Msun; component 2
# captures the high-mass tail (e.g. PSR J0740+6620, J1614-2230) near
# 1.80 Msun.  PDF not present in the project papers/ folder; values
# below are taken from arXiv:1810.03548 Table 3 (their two-Gaussian fit).
NS_REMAP_W1 = 0.66
NS_REMAP_MU1 = 1.34  # [Msun]
NS_REMAP_SIG1 = 0.07  # [Msun]
NS_REMAP_W2 = 0.34
NS_REMAP_MU2 = 1.80  # [Msun]
NS_REMAP_SIG2 = 0.21  # [Msun]
NS_REMAP_M_MIN = 1.10  # [Msun] lower truncation


def _truncated_double_gauss_cdf(x, m_min, m_max):
    """Truncated Alsing+ 2018 double-Gaussian CDF, evaluated at x [Msun].

    Computed analytically from the Gaussian erf, then renormalised so
    F(m_min) = 0 and F(m_max) = 1.
    """

    def _g_cdf(t, mu, sig):
        return 0.5 * (1.0 + erf((t - mu) / (sig * np.sqrt(2.0))))

    raw = NS_REMAP_W1 * _g_cdf(x, NS_REMAP_MU1, NS_REMAP_SIG1) + NS_REMAP_W2 * _g_cdf(
        x, NS_REMAP_MU2, NS_REMAP_SIG2
    )
    raw_lo = NS_REMAP_W1 * _g_cdf(m_min, NS_REMAP_MU1, NS_REMAP_SIG1) + NS_REMAP_W2 * _g_cdf(
        m_min, NS_REMAP_MU2, NS_REMAP_SIG2
    )
    raw_hi = NS_REMAP_W1 * _g_cdf(m_max, NS_REMAP_MU1, NS_REMAP_SIG1) + NS_REMAP_W2 * _g_cdf(
        m_max, NS_REMAP_MU2, NS_REMAP_SIG2
    )
    return (raw - raw_lo) / (raw_hi - raw_lo)


def uniform_mass_cdf(x, m_min, m_max):
    """Flat-prior CDF on [m_min, m_max], for the remap-dependence test.

    A deliberately uninformative alternative to the Alsing+ 2018 target:
    passing this to ``remap_ns_marginal(..., target_cdf=uniform_mass_cdf)``
    spreads the NS-mass marginal uniformly over the allowed range, which
    isolates how much of the BNS class demographics is set by the choice
    of target mass function versus by COMPAS.
    """
    x = np.asarray(x, dtype=float)
    return np.clip((x - m_min) / (m_max - m_min), 0.0, 1.0)


def remap_ns_marginal(
    m_ns, weights=None, m_tov=None, m_min=NS_REMAP_M_MIN, n_grid=10000, rng=None, target_cdf=None
):
    """Weighted quantile remap of a 1D NS-mass array to the Alsing+ 2018 PDF.

    Operates on a single NS-mass stream (e.g. the NS component of a BHNS
    sample) rather than a paired ``(m1, m2)`` BNS sample.  Used for both
    populations: ``remap_ns_masses_double_gaussian`` calls this on the
    stacked ``[m1, m2]`` array, while BHNS callers pass ``M_NS`` directly.

    The transform is weighted-rank preserving: each input is mapped to the
    quantile of the truncated double-Gaussian target with the same
    weighted CDF position.  Sub-microsolar deterministic jitter from
    ``rng`` breaks rank ties at the discrete COMPAS mass-bin level.

    ``target_cdf`` overrides the default Alsing+ 2018 target with an
    arbitrary ``cdf(x, m_min, m_max)`` callable (e.g. a uniform prior),
    used by the Section 2b remap-dependence test to quantify how much of
    the BNS demographics is set by the imposed mass function rather than
    by COMPAS.  When ``None`` the Alsing double Gaussian is used.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if m_tov is None:
        m_tov = M_TOV

    m_ns = np.asarray(m_ns, dtype=float)
    if weights is None:
        w = np.ones(m_ns.size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != m_ns.shape:
            raise ValueError("weights must have the same shape as m_ns")

    jitter = rng.uniform(-1e-6, 1e-6, size=m_ns.size)
    m_jitt = m_ns + jitter

    order = np.argsort(m_jitt)
    cum_w = np.cumsum(w[order])
    cum_w_total = cum_w[-1]
    u_ranked = (cum_w - 0.5 * w[order]) / cum_w_total

    grid = np.linspace(m_min, m_tov, n_grid)
    if target_cdf is None:
        cdf_grid = _truncated_double_gauss_cdf(grid, m_min, m_tov)
    else:
        cdf_grid = np.asarray(target_cdf(grid, m_min, m_tov), dtype=float)
    m_target_ranked = np.interp(u_ranked, cdf_grid, grid)

    m_remap = np.empty_like(m_ns)
    m_remap[order] = m_target_ranked
    return m_remap


def remap_ns_masses_double_gaussian(
    m1, m2, weights=None, m_tov=None, m_min=NS_REMAP_M_MIN, n_grid=10000, rng=None, target_cdf=None
):
    """Quantile-remap paired BNS NS gravitational masses to an empirical Galactic-NS PDF.

    The Broekgaarden et al. (2021) COMPAS catalogues use the Fryer
    (2012) *delayed* SN engine (their Table 1; Model A is the fiducial,
    Model I is the rapid variation).  Both delayed and rapid produce a
    NS gravitational-mass deficit near 1.7 Msun, inherited from the
    Fryer (2012) Eq. 12-13 baryonic-to-gravitational mass conversion
    (Broekgaarden+ 2021 footnote 3; see also Mandel & Muller 2020 MNRAS
    499, 3214; Patton & Sukhbold 2020 MNRAS 499, 2803).  This helper
    performs a weighted, rank-preserving quantile transform from the
    empirical COMPAS NS-mass distribution to the Alsing, Silva & Berti
    (2018) MNRAS 478, 1377 double-Gaussian fit to Galactic NSs:

        f(m) propto W1 * N(m; mu1, sig1) + W2 * N(m; mu2, sig2),
        truncated to [m_min, m_tov].

    The remap is *per-component marginal*: m1 and m2 are ranked in
    their own component streams (not stacked) against the same target
    PDF.  Both marginals are independently Alsing-conformal; the
    weighted rank order within each component is preserved exactly.
    Cross-component rank correlations across the pair are NOT
    preserved: a system whose original m1, m2 sat in the same input
    rank bin can land in different target quantiles.  This is the
    intended behaviour and matches Mandel & Muller (2020) Section 4.

    An earlier variant of this function stacked ``[m1, m2]`` and
    ranked the combined array.  That implementation forced m1 ranks
    into the upper half of the stacked CDF and m2 ranks into the
    lower half, which the inverse Alsing CDF then split at the target
    median (~1.34 Msun).  The result was a visible "median wall" in
    the BNS mass-plane figure: m2 squeezed into roughly [1.10, 1.34]
    and m1 into roughly [1.34, 2.20], with a deficit band at exactly
    M_NS = 1.34.  The per-component remap implemented below removes
    that artefact because each component samples the full target PDF
    independently.

    Length, indexing, STROOPWAFEL weights, and the m1 >= m2 invariant
    are preserved.  Per-system chirp mass and mass ratio q reflect
    the Alsing target's joint structure (after re-sort), not the
    COMPAS binary-evolution joint structure.

    For BHNS samples use :func:`remap_ns_marginal` directly on the
    single NS-mass column.

    Parameters
    ----------
    m1, m2 : array_like
        Component gravitational masses [Msun], with m1 >= m2 (as
        returned by ``grb_io.load_bns(sort_masses=True)``).
    weights : array_like, optional
        Per-system STROOPWAFEL weights.  If None, unweighted ranks.
    m_tov : float, optional
        Maximum NS gravitational mass [Msun].  Defaults to ``M_TOV``
        (2.2).  The target distribution is truncated at ``m_tov``.
    m_min : float, optional
        Lower truncation of the target distribution [Msun].
    n_grid : int, optional
        Grid resolution for the inverse target CDF.
    rng : np.random.Generator, optional
        Used to spawn two independent jitter streams, one per
        component, to break rank ties at the discrete COMPAS
        mass-bin level.  Default ``np.random.default_rng(0)``.
    target_cdf : callable, optional
        Override the Alsing+ 2018 target with an arbitrary
        ``cdf(x, m_min, m_max)`` (e.g. ``uniform_mass_cdf``).  Passed
        through to both per-component :func:`remap_ns_marginal` calls.
        Used by the Section 2b remap-dependence test.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    if m1.shape != m2.shape:
        raise ValueError("m1 and m2 must have the same shape")

    if weights is None:
        w_sys = np.ones(m1.size, dtype=float)
    else:
        w_sys = np.asarray(weights, dtype=float)
        if w_sys.shape != m1.shape:
            raise ValueError("weights must have the same shape as m1")

    # Two independent jitter streams: one per component.  Drawing both
    # seeds from the caller-supplied rng keeps the whole transform
    # deterministic under a fixed top-level seed.
    seed1 = int(rng.integers(0, 2**31 - 1))
    seed2 = int(rng.integers(0, 2**31 - 1))
    rng1 = np.random.default_rng(seed1)
    rng2 = np.random.default_rng(seed2)

    m1_remap = remap_ns_marginal(
        m1, weights=w_sys, m_tov=m_tov, m_min=m_min, n_grid=n_grid, rng=rng1, target_cdf=target_cdf
    )
    m2_remap = remap_ns_marginal(
        m2, weights=w_sys, m_tov=m_tov, m_min=m_min, n_grid=n_grid, rng=rng2, target_cdf=target_cdf
    )

    m1_new = np.maximum(m1_remap, m2_remap)
    m2_new = np.minimum(m1_remap, m2_remap)
    return m1_new, m2_new


# ---------------------------------------------------------------------------
# NS compactness helper
# ---------------------------------------------------------------------------
def _compactness(M_NS, R_km):
    """Dimensionless NS compactness C = G*M/(R*c^2)."""
    G = 6.674e-11
    c = 2.99792458e8
    Msun = 1.989e30
    return G * np.asarray(M_NS, dtype=float) * Msun / (np.asarray(R_km, dtype=float) * 1e3 * c**2)


# ---------------------------------------------------------------------------
# Foucart et al. (2018) total remnant baryon mass
# ---------------------------------------------------------------------------
def foucart_remnant_mass(
    M_BH, M_NS, a_BH=0.0, R_NS_km=None, R_1p4_km=12.0, clip_Q=None, clip_chi=None
):
    """Foucart et al. (2018) Eq. (4) & (6) [arXiv:1807.00011].

    Returns the *total* baryon mass outside the BH after merger (disk +
    tidal tail + dynamical ejecta), **not** the disk mass alone.

    Validity ranges (Foucart+ 2018):
      - *Calibration range* (abstract / Eq. 4 fit validity):
            Q in [1, 7],  chi_BH in [-0.5, 0.9],  C_NS in [0.13, 0.182]
        Stay within these bounds for the documented ~15% accuracy.
      - *Data range* (Table II coverage):
            individual NR simulations extend to chi_BH = 0.97, but with
            larger residuals (Q=3, chi=0.97 underestimates the NR disk
            mass by ~30%).

    A warning is emitted when Q > 7 OR |a_BH| > 0.9 (count and max
    value reported per call).  Use ``clip_Q`` and ``clip_chi`` to zero
    out extrapolated systems instead of returning fit values.

    Note on bulk usage: the warnings are aggregated to a single
    ``warnings.warn`` per call (with a count), so they do NOT flood
    when called once on a large array.  If you call this in a Python
    loop over many systems and want to suppress repeated messages,
    wrap the loop in ``warnings.catch_warnings()`` with
    ``warnings.simplefilter('once')`` (see the standard library
    ``warnings`` documentation).

    Parameters
    ----------
    clip_Q : float, optional
        If given, systems with Q > clip_Q are assigned zero remnant mass
        instead of extrapolating the fit.  Set to 7.0 for conservative
        results within the calibration range.
    clip_chi : float, optional
        If given, systems with |a_BH| > clip_chi are assigned zero
        remnant mass instead of extrapolating the fit.  Set to 0.9 to
        stay strictly within the Foucart+ 2018 calibration range.
    """
    M_BH_a = np.asarray(M_BH, dtype=float)
    M_NS_a = np.asarray(M_NS, dtype=float)
    a_BH_a = np.asarray(a_BH, dtype=float)
    Q = M_BH_a / M_NS_a
    if np.any(Q > 7.0):
        n_extrap = int(np.sum(Q > 7.0))
        warnings.warn(
            f"Foucart (2018) formula applied to {n_extrap} systems with "
            f"Q > 7 (max Q={float(np.max(Q)):.1f}); "
            f"calibrated for Q in [1, 7]",
            stacklevel=2,
        )
    if np.any(np.abs(a_BH_a) > 0.9):
        n_chi = int(np.sum(np.abs(a_BH_a) > 0.9))
        warnings.warn(
            f"Foucart (2018) formula applied to {n_chi} systems with "
            f"|chi_BH| > 0.9 (max |chi|={float(np.max(np.abs(a_BH_a))):.3f}); "
            f"calibrated for chi_BH in [-0.5, 0.9]",
            stacklevel=2,
        )

    R_km = R_NS_km if R_NS_km is not None else ns_radius(M_NS, R_1p4_km=R_1p4_km)
    C_NS = _compactness(M_NS, R_km)
    eta = M_NS_a * M_BH_a / (M_NS_a + M_BH_a) ** 2
    R_hat = r_isco(a_BH)

    alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761  # Foucart+ 2018 Eq. 6 fit
    bracket = alpha * (1 - 2 * C_NS) / eta ** (1 / 3) - beta * R_hat * C_NS / eta + gamma

    M_b = ns_baryon_mass(M_NS)
    result = np.maximum(0.0, bracket) ** delta * M_b

    if clip_Q is not None:
        result = np.where(clip_Q < Q, 0.0, result)
    if clip_chi is not None:
        result = np.where(np.abs(a_BH_a) > clip_chi, 0.0, result)

    return result


# ---------------------------------------------------------------------------
# Kruger & Foucart (2020) dynamical ejecta  [arXiv:2002.07728]
# ---------------------------------------------------------------------------
def bhns_dynamical_ejecta(M_BH, M_NS, a_BH, R_NS_km=None, R_1p4_km=12.0):
    """BHNS dynamical ejecta mass [Msun] -- Kruger & Foucart (2020) Eq. (9).

    Fitted to 45+ NR simulations (Kawaguchi+ 2015, Foucart+ 2019).
    Valid for Q ~ 3-7, chi_BH ~ 0-0.9, C_NS ~ 0.14-0.18.
    """
    R_km = R_NS_km if R_NS_km is not None else ns_radius(M_NS, R_1p4_km=R_1p4_km)
    C_NS = _compactness(M_NS, R_km)
    M_b = ns_baryon_mass(M_NS)
    Q = np.asarray(M_BH, dtype=float) / np.asarray(M_NS, dtype=float)
    R_hat = r_isco(a_BH)

    a1, a2, a4 = 0.007116, 0.001436, -0.02762  # Kruger & Foucart 2020 Eq. 9, Table I
    n1, n2 = 0.8636, 1.6840  # Kruger & Foucart 2020 Eq. 9, Table I
    bracket = a1 * Q**n1 * (1 - 2 * C_NS) / C_NS - a2 * Q**n2 * R_hat + a4
    return np.maximum(0.0, bracket) * M_b


def bns_disk_mass(M1, M2, R1_km=None, R2_km=None, R_1p4_km=12.0, M_TOV_local=None):
    """BNS post-merger accretion disk mass [Msun] -- linearized Kruger & Foucart (2020) fit.

    .. warning::
        This linearized BNS disk-mass fit has a sharp transition at
        C_1 ~ 0.156 (R_NS ~ 12 km for a 1.27 Msun NS), where the
        linear ``a * C_1 + c`` term hits the 5e-4 floor.  Empirically,
        the disk mass can swing by ~100x for a 1 km change in NS
        radius near this threshold: a GW170817-like system at
        R_1.4 = 12 km gives M_disk ~ 1.5e-3 Msun, while R_1.4 = 13 km
        gives ~0.28 Msun.  For studies near R_1.4 ~ 12 km, average
        over an EOS prior or use the smoother Radice et al. (2018)
        Eq. (1) instead.  Single-EOS results in this regime are not
        physically meaningful.

    .. warning::
        Interaction with the ``ns_radius`` plateau.  When component
        radii are not supplied explicitly, ``R_light`` is computed
        via the heuristic ``ns_radius``, which is *flat* at
        ``R_1p4_km`` for ``m_light < 1.4 Msun`` (see
        ``ns_radius`` docstring).  Combined with the KF2020
        discontinuity above, this means: (i) the disk mass is
        completely insensitive to ``m_light`` below 1.4 Msun and
        depends only on ``R_1p4_km``; (ii) the discontinuity becomes
        one-sided.  For ``M1 = 1.46, M2 = 1.27`` the result is exactly
        the KF2020 floor (~1.5e-3 Msun) for any
        ``R_1p4_km <= ~12.05`` km, then jumps two orders of magnitude
        as ``R_1p4_km`` crosses ~12.1 km.  For sub-1.4 Msun light
        components, supply explicit per-component ``R1_km`` /
        ``R2_km`` from an EOS table (or use ``ns_radius_from_eos``)
        to recover the proper R(M) margin and mitigate this
        knife-edge sensitivity.

    Distinct from ``foucart_disk_mass`` (which is BHNS-only).  KF2020
    fit the disk mass of BNS NR simulations to the compactness of the
    *lighter* NS.  We use a linearized form of their Eq. (4):

        M_disk / M_b^tot = max(5e-4, a * C_1 + c),  (a, c) = (-8.1580, 1.2695)

    where C_1 is the compactness of the less-massive component and
    M_b^tot is the total baryon mass of the binary.  This is NOT the
    verbatim published Eq. (4), which carries an exponent:

        M_disk = M_1 * max{5e-4, (a * C_1 + c)^d},
        (a, c, d) = (-8.1324, 1.4820, 1.7784).

    We drop the exponent and refit the linear coefficients; the
    published fit zeros the disk at C_1 ~ 0.182, this linear form at
    C_1 ~ 0.156.

    The BNS classifiers in ``grb_classify`` use mass-ratio and
    total-mass thresholds (Gottlieb 2023, 2024) rather than disk mass,
    so this helper is provided for completeness rather than direct use.

    Parameters
    ----------
    M1, M2 : array-like
        Component gravitational masses [Msun] (any ordering).
    R1_km, R2_km : array-like, optional
        Component radii [km].  If None, computed via ``ns_radius``.
    R_1p4_km : float, optional
        Reference R_{1.4} for the heuristic ``ns_radius`` (only used if
        the per-component radii are not supplied).
    M_TOV_local : float, optional
        TOV mass forwarded to ``ns_radius``; defaults to ``M_TOV``.

    Returns
    -------
    M_disk : float or array
        BNS disk mass [Msun].
    """
    M1 = np.asarray(M1, dtype=float)
    M2 = np.asarray(M2, dtype=float)
    m_heavy = np.maximum(M1, M2)
    m_light = np.minimum(M1, M2)

    if R1_km is not None and R2_km is not None:
        R1 = np.asarray(R1_km, dtype=float)
        R2 = np.asarray(R2_km, dtype=float)
        R_heavy = np.where(M1 >= M2, R1, R2)  # noqa: F841 (kept for symmetry with R_light)
        R_light = np.where(M1 >= M2, R2, R1)
    else:
        R_heavy = ns_radius(m_heavy, R_1p4_km=R_1p4_km, M_TOV_local=M_TOV_local)  # noqa: F841
        R_light = ns_radius(m_light, R_1p4_km=R_1p4_km, M_TOV_local=M_TOV_local)

    C_1 = _compactness(m_light, R_light)
    M_b_tot = ns_baryon_mass(m_heavy) + ns_baryon_mass(m_light)

    a, c = -8.1580, 1.2695  # linearized KF2020 Eq. 4 BNS disk fit (refit; see docstring)
    return M_b_tot * np.maximum(5e-4, a * C_1 + c)


def bns_dynamical_ejecta(M1, M2, R1_km=None, R2_km=None, R_1p4_km=12.0):
    """BNS dynamical ejecta mass [Msun] -- Kruger & Foucart (2020) Eq. (6).

    Fitted to 200 NR simulations (Dietrich & Ujevic 2017, Kiuchi+ 2019).
    Result is in solar masses (the formula output is in units of 1e-3 Msun).

    Sanity check (GW170817-like, M1=1.46, M2=1.27, R_1p4=12.0 km):
        expected M_ej_dyn ~ 0.003-0.010 Msun
    The fit residual quoted by Kruger & Foucart (2020) is sigma ~
    0.004 Msun, so a single-system value within this band is consistent
    with the fit even though the central value is closer to ~0.003-0.006.
    AT2017gfo total ejecta ~ 0.05-0.08 Msun (Rastinejad+ 2025); the
    dynamical component is a subset, with disk wind adding ~0.01-0.05 Msun.
    """
    M1 = np.asarray(M1, dtype=float)
    M2 = np.asarray(M2, dtype=float)
    R1 = R1_km if R1_km is not None else ns_radius(M1, R_1p4_km=R_1p4_km)
    R2 = R2_km if R2_km is not None else ns_radius(M2, R_1p4_km=R_1p4_km)
    C1 = _compactness(M1, R1)
    C2 = _compactness(M2, R2)

    a, b, c, n = -9.3335, 114.17, -337.56, 1.5465  # Kruger & Foucart 2020 Eq. 6, Table I
    term1 = (a / C1 + b * (M2 / M1) ** n + c * C1) * M1
    term2 = (a / C2 + b * (M1 / M2) ** n + c * C2) * M2
    return np.maximum(0.0, term1 + term2) * 1e-3


# ---------------------------------------------------------------------------
# Disk mass = remnant mass - dynamical ejecta
# ---------------------------------------------------------------------------
def foucart_disk_mass(
    M_BH, M_NS, a_BH=0.0, R_NS_km=None, R_1p4_km=12.0, f_disk=None, clip_Q=None, clip_chi=None
):
    """BHNS *late-time effective* accretion disk mass [Msun].

    Default behaviour (f_disk=None): computes
        M_disk = M_rem(Foucart 2018) - M_dyn(Kruger & Foucart 2020)
    which properly separates the disk from dynamical ejecta.

    Time-scope note.  Foucart (2018) defines ``M_rem`` as the bound
    baryon mass outside the BH at t ~ 10 ms after merger.  At that
    snapshot ``M_rem`` includes (i) the accretion disk, (ii) the bound
    tidal tail that falls back over ~0.1 - 1 s, and (iii) the unbound
    dynamical ejecta.  Subtracting ``M_dyn`` (component iii) therefore
    gives the *late-time effective disk mass* after tidal-tail
    fallback has accreted onto the disk -- not the instantaneous
    t = 10 ms accretion disk.  This is the right quantity for
    powering the GRB jet engine (which operates on the viscous /
    accretion timescale, well after the 10 ms snapshot) and for
    comparison with downstream classification thresholds
    (``MDISK_SHORT``, ``MDISK_LONG``).

    Legacy behaviour (f_disk=<float>): returns f_disk * M_rem as before.

    Parameters
    ----------
    clip_Q : float, optional
        Forwarded to ``foucart_remnant_mass``.  Systems with Q > clip_Q
        get zero remnant (and therefore zero disk) mass.
    clip_chi : float, optional
        Forwarded to ``foucart_remnant_mass``.  Systems with
        |a_BH| > clip_chi get zero remnant (and therefore zero disk)
        mass; set to 0.9 to stay strictly within the Foucart+ 2018
        calibration range.
    """
    M_rem = foucart_remnant_mass(
        M_BH, M_NS, a_BH=a_BH, R_NS_km=R_NS_km, R_1p4_km=R_1p4_km, clip_Q=clip_Q, clip_chi=clip_chi
    )
    if f_disk is not None:
        return f_disk * M_rem

    M_dyn = bhns_dynamical_ejecta(M_BH, M_NS, a_BH, R_NS_km=R_NS_km, R_1p4_km=R_1p4_km)
    return np.maximum(0.0, M_rem - M_dyn)


# ---------------------------------------------------------------------------
# BH spin misalignment (Issue 5)
# ---------------------------------------------------------------------------
def effective_aligned_spin(a_BH, theta_tilt):
    """``a_BH * cos(theta_tilt)``, clipped at zero.

    The aligned-spin Foucart 2018 fit overestimates BHNS disk mass for
    misaligned BHs; Kawaguchi et al. (2015) NR shows disk mass dropping
    to near zero past ~50-60 deg of tilt.
    """
    return np.maximum(0.0, np.asarray(a_BH) * np.cos(np.asarray(theta_tilt)))


MISALIGNMENT_SYSTEMATIC_FACTOR = 0.5
"""Order-unity heuristic for the population-averaged reduction in BHNS
GRB fractions from BH spin-orbit misalignment.  Roughly half of BHNS
systems have tilt > 45 deg in the Fragos et al. (2010) and Gerosa et al.
(2018) distributions, and Kawaguchi et al. (2015) NR simulations show
the disk mass collapses to near zero past tilt ~50-60 deg.  Applied at
the *population* level by ``grb_rates.apply_bhns_misalignment``; do
*not* combine it with the per-system ``effective_aligned_spin`` route
or the suppression double-counts."""


# ---------------------------------------------------------------------------
# Gottlieb et al. (2025) Eq. (11) disk-wind ejecta relation
# ---------------------------------------------------------------------------
# Constants and helpers used by ``comparison.ipynb`` to compare BH-engine
# and HMNS-engine kilonova ejecta predictions against the Rastinejad+24
# observational sample.  Single source of truth; previously duplicated
# in the notebook.
#
# Reference: Gottlieb et al. 2025, arXiv:2411.13657.

GOTTLIEB25_F_RANGE = (0.02, 0.71)
"""Range of the dimensionless ratio ``f_code = f_b / epsilon_gamma``,
which the notebook samples log-uniformly to marginalise the BH-engine
prediction in ``predict_bh_mc``.

Gottlieb+25 Eq. (10) defines

    E_iso,gamma = (epsilon_gamma / f_b) * E_j = 10 * f_{-1} * E_j,
    f_{-1} = 0.1 * epsilon_gamma / f_b,

with ``0.01 <= f_b <= 0.11`` (beaming fraction, Berger 2014,
ARA&A 52, 43) and ``0.15 <= epsilon_gamma <= 0.5`` (gamma-ray
radiative efficiency, Beniamini & Granot 2015).  Combining the
boundary values gives ``f_{-1} in [0.14, 5.0]``.

The notebook stores the inverse, ``f_code = f_b / epsilon_gamma``, so
that ``f_inv = 0.1 / f_code`` reproduces the paper's ``f_{-1}``.
Algebraically ``f_code in [0.01/0.5, 0.11/0.15] = [0.02, 0.733]``;
the upper bound is rounded down to 0.71 here, which compresses the
high-``f_code`` (low-``f_inv``) tail by about 3% relative to the strict
algebraic limit and is a deliberate cosmetic choice, not a derived
value."""

GOTTLIEB25_DISK_RANGE = (0.01, 0.10)
"""Post-merger accretion-disk-mass prior for the HMNS / magnetar engine
[Msun].  Anchored on the BNS disk-mass fit of Kruger & Foucart 2020
(arXiv:2002.07728, Eq. 4), which gives ``M_disk <~ 0.1 Msun`` for
``R_NS ~ 12 km``.  The upper edge is consistent with Gottlieb+25 §3.1
(sbGRB-producing HMNSs have "less massive disks" than the ~0.1 Msun
lbGRB benchmark) and with the disk masses sampled by the Radice+18 NR
simulations (arXiv:1809.11163)."""

GOTTLIEB25_WIND_FRAC = 0.3
"""Disk-wind ejected fraction ``f_wind = M_ej / M_disk``.  Gottlieb+25
§3.1, line 474; consistent with ``f_wind ~ 0.2-0.4`` in Radice+18
GRMHD simulations (arXiv:1809.11163).

Point value retained for the Fig. 1 visual band in
``comparison.ipynb``; the Fig. 2 / Fig. 3 Monte-Carlo path samples
``GOTTLIEB25_WIND_FRAC_RANGE`` log-uniformly instead, so that the
HMNS-engine prediction reflects the GRMHD scatter."""

GOTTLIEB25_WIND_FRAC_RANGE = (0.2, 0.4)
"""Disk-wind ejected fraction range, ``f_wind in [0.2, 0.4]``.  Radice
et al. 2018 GRMHD simulations (arXiv:1809.11163, Table 2) give
``f_wind ~ 0.2-0.4`` across the BNS post-merger disk-mass range; this
range is drawn log-uniformly in the HMNS-engine MC.  The point value
``GOTTLIEB25_WIND_FRAC = 0.3`` is the geomean and is retained for the
Fig. 1 visual band only."""

GOTTLIEB25_T_HMNS_RANGE = (0.1, 10.0)
"""Long-lived-HMNS lifetime window [s].  Set by magnetar spin-down and
viscous timescales (Lippuner+17, Fujibayashi+18, Metzger 2019).  Used
as a *guide for the eye* in Fig. 1 of ``comparison.ipynb``; Gottlieb+25
§3.1 does not claim the HMNS engine predicts T_50 directly."""

# External long-duration merger-GRB E_iso anchor [erg].  Literature isotropic
# gamma-ray energies of long-duration GRBs with kilonova / compact-merger
# evidence (the Gottlieb 2024 "lbGRB" engine class).  Used to de-circularize
# the BH-engine null in comparison.ipynb: the BH track was previously anchored
# on the in-sample Rastinejad+ 2024 lbGRBs it is then compared against, so the
# lbGRB residual was partly internal to its own anchor.  Anchoring on this
# external set makes the short-vs-long separation anchor-independent.  Sources
# are tabulated in data/external_lgrb_eiso.csv (one primary analysis per event).
LBGRB_EISO_EXTERNAL_ERG = (8.4e50, 1.14e51, 1.70e51, 1.30e52, 6.97e52)
LBGRB_EISO_EXTERNAL_LOGMEAN = float(np.mean(np.log10(LBGRB_EISO_EXTERNAL_ERG)))
LBGRB_EISO_EXTERNAL_LOGSTD = float(np.std(np.log10(LBGRB_EISO_EXTERNAL_ERG)))


def gottlieb25_eq11(T50, E_iso, alpha=2.0, f_inv=1.0):
    """Predicted disk-wind kilonova ejecta mass [Msun], BH engine.

    Gottlieb et al. 2025 (arXiv:2411.13657) Eq. (11):

        M_ej = 1e-3 * f_{-1}^{-1} * (E_iso / 2e51 erg) *
               (T50 / 1 s) ** (alpha - 1)   Msun

    Parameters
    ----------
    T50 : array-like
        50% gamma-ray fluence duration [s].
    E_iso : array-like
        Isotropic-equivalent gamma-ray energy [erg].
    alpha : float, default 2.0
        Power-law index of the prompt luminosity function.
    f_inv : float, default 1.0
        Inverse GRB radiative efficiency normalised to f = 0.1:
        ``f_inv = (f / 0.1) ** -1``.
    """
    T50 = np.asarray(T50, dtype=float)
    E_iso = np.asarray(E_iso, dtype=float)
    return 1e-3 * f_inv * (E_iso / 2e51) * (T50 / 1.0) ** (alpha - 1.0)


def hmns_wind_ejecta(M_d, wind_frac=GOTTLIEB25_WIND_FRAC):
    """Predicted disk-wind kilonova ejecta mass [Msun], HMNS engine.

    ``M_ej = wind_frac * M_d``.  Independent of T_50 to first order.
    Default ``wind_frac = 0.3`` from Gottlieb+25 §3.1.
    """
    return wind_frac * np.asarray(M_d, dtype=float)


# ---------------------------------------------------------------------------
# Pais et al. (2025) BNS jet breakout  [arXiv:2407.19002]
# ---------------------------------------------------------------------------
# Pais, Piran, Kiuchi & Shibata (2025) inject an unmagnetized relativistic
# jet into the neutrino-radiation-GRMHD BNS post-merger environment of
# Kiuchi et al. (2023) and confirm the Gottlieb & Nakar (2022,
# arXiv:2106.03860) semi-analytic breakout estimate adapted to the
# anisotropic, funnel-bearing ejecta of a BNS merger.  The constants below
# back the per-system breakout test used by ``grb_rates.breakout_fraction_bns``.
#
# Provenance caveat: the prefactors PAIS_BREAKOUT_COEFF and PAIS_COLL_NORM
# are calibrated to a single GW170817-like run (a 1.2 + 1.5 Msun SFHo
# merger, M_ej ~ 1.5e-2 Msun, ~10 deg funnel).  Pais et al. note the escape
# energy "might differ by a factor of a few"; absolute breakout fractions
# inherit that systematic even though E_ej scales per system here.

_C_CM_S = 2.99792458e10  # speed of light [cm/s]
_MSUN_G = 1.989e33  # solar mass [g]

PAIS_BREAKOUT_COEFF = 150.0
"""Numerical prefactor in the Pais et al. (2025) Eq. (4) breakout
criterion, ``E_j > coeff * [(t_d/t_e)^2 + 2] * E_ej(<theta_j) * theta_j^2``.
This is the Gottlieb & Nakar (2022) spherical estimate adapted to the
anisotropic BNS ejecta of Kiuchi et al. (2023); the value absorbs the
funnel geometry of that simulation."""

PAIS_COLL_NORM = 3.6e49
"""Collimation-energy normalisation [erg] in Pais et al. (2025) Eq. (3),
``E_coll(theta_j) = norm * (theta_j / 0.1 rad)^4 * (t_e / 1 s)``.  Fit to
the Kiuchi et al. (2023) background; the prefactor is tailored to that
ejecta energy distribution."""

V_EJ_BNS = 0.15
"""Characteristic BNS post-merger ejecta velocity [units of c].  Pais et
al. (2025) §2 report ~0.15 c for the secular post-merger ejecta at
t ~ 1 s, slower than the ~0.2 c literature average (Sekiguchi et al.
2016) because it is launched at the outskirt of the massive torus."""

EPS_JET = 0.01
"""Disk-to-jet energy conversion efficiency, ``E_j = EPS_JET * M_disk *
c^2``.  CALIBRATION, not from a single paper: chosen so a GW170817-like
disk (``M_disk ~ 1.5e-3 Msun`` from ``bns_disk_mass`` at R_1.4 = 12 km)
yields ``E_j ~ 3e49 erg``, the successful-jet energy of the late-launch
Pais et al. (2025) simulation that matches the GW170817 radio afterglow."""

T_E_RANGE = (0.05, 0.3)
"""Engine working-time window [s] for a sGRB jet (Pais et al. 2025 §3.3,
Fig. 3): jets with theta_j = 5-10 deg need t_e ~ 0.05-0.3 s to break out,
all compatible with observed sGRB durations."""

T_D_PROMPT_RANGE = (0.01, 0.1)
"""Merger-to-jet-launch delay window [s] for prompt-collapse remnants,
where the BH forms and the low-density polar funnel clears quickly (Pais
et al. 2025 §2: BH formation and funnel onset within ~0.03-0.1 s).
Long-lived / short-lived HMNS remnants instead launch only after the NS
collapses, so their delay is the HMNS lifetime ``GOTTLIEB25_T_HMNS_RANGE``;
the ``(t_d/t_e)^2`` term in Eq. (4) then suppresses their breakout."""


def jet_energy_from_disk(M_disk, eps_jet=EPS_JET):
    """Two-sided jet energy [erg] from the accretion disk mass.

    ``E_j = eps_jet * M_disk * c^2`` with ``M_disk`` in Msun.  See
    ``EPS_JET`` for the efficiency calibration.
    """
    M_disk = np.asarray(M_disk, dtype=float)
    return eps_jet * M_disk * _MSUN_G * _C_CM_S**2


def ejecta_energy_in_cone(M_ej, theta_j_rad, v_ej=V_EJ_BNS):
    """Kinetic energy of ejecta inside the jet cone [erg].

    ``E_ej(<theta_j) = (1 - cos theta_j) * 0.5 * M_ej * (v_ej c)^2`` with
    ``M_ej`` in Msun and ``v_ej`` in units of c.  The ``(1 - cos theta_j)``
    factor is the one-sided cone solid-angle fraction; for small angles it
    reproduces the ``E_ej(<theta) ~ theta^2`` scaling Pais et al. (2025)
    fit to the BNS background (their Eq. 4 discussion).
    """
    M_ej = np.asarray(M_ej, dtype=float)
    theta = np.asarray(theta_j_rad, dtype=float)
    v_cm = v_ej * _C_CM_S
    return (1.0 - np.cos(theta)) * 0.5 * M_ej * _MSUN_G * v_cm**2


def pais_collimation_energy(theta_j_rad, t_e, norm=PAIS_COLL_NORM):
    """Energy required to collimate the jet to ``theta_j`` [erg].

    Pais et al. (2025) Eq. (3):
    ``E_coll = norm * (theta_j / 0.1 rad)^4 * (t_e / 1 s)``.
    """
    theta = np.asarray(theta_j_rad, dtype=float)
    t_e = np.asarray(t_e, dtype=float)
    return norm * (theta / 0.1) ** 4 * (t_e / 1.0)


def jet_breaks_out_pais(
    E_j, E_ej_cone, theta_j_rad, t_d, t_e, coeff=PAIS_BREAKOUT_COEFF, require_collimation=True
):
    """Per-system jet breakout boolean (Pais et al. 2025).

    Escape criterion (Eq. 4):
        ``E_j > coeff * [(t_d / t_e)^2 + 2] * E_ej(<theta_j) * theta_j^2``.
    When ``require_collimation`` is True the jet must also satisfy the
    collimation condition (Eq. 3), ``E_j > pais_collimation_energy(...)``.

    All inputs broadcast against each other; energies in erg, angles in
    radians, times in seconds.  Returns a boolean array.
    """
    E_j = np.asarray(E_j, dtype=float)
    E_ej_cone = np.asarray(E_ej_cone, dtype=float)
    theta = np.asarray(theta_j_rad, dtype=float)
    t_d = np.asarray(t_d, dtype=float)
    t_e = np.asarray(t_e, dtype=float)

    E_thresh = coeff * ((t_d / t_e) ** 2 + 2.0) * E_ej_cone * theta**2
    escape = E_j > E_thresh
    if require_collimation:
        escape = escape & (E_j > pais_collimation_energy(theta, t_e, norm=PAIS_COLL_NORM))
    return escape


def _selftest_gottlieb25():
    """Run on import; verifies that Eq. (11) normalisation and the f_inv
    derivation from Eq. (10) are consistent with the cited values."""
    assert np.isclose(gottlieb25_eq11(1.0, 2e51, alpha=2.0, f_inv=1.0), 1e-3), (
        "Eq. 11 normalisation failed"
    )
    assert np.isclose(gottlieb25_eq11(4.0, 2e51, alpha=1.5, f_inv=1.0), 2e-3), (
        "Eq. 11 alpha=1.5 scaling failed"
    )
    f_lo, f_hi = GOTTLIEB25_F_RANGE
    finv_min, finv_max = 0.1 / f_hi, 0.1 / f_lo
    assert np.isclose(finv_min, 0.14, atol=0.01), f"f_inv_min should be ~0.14 (got {finv_min:.3f})"
    assert np.isclose(finv_max, 5.0, atol=0.05), f"f_inv_max should be ~5.0 (got {finv_max:.3f})"
    lo, hi = hmns_wind_ejecta(np.array(GOTTLIEB25_DISK_RANGE))
    assert 0 < lo < hi, "HMNS wind-ejecta range ordering failed"
    fw_lo, fw_hi = GOTTLIEB25_WIND_FRAC_RANGE
    assert 0 < fw_lo < fw_hi, "HMNS wind-fraction range ordering failed"
    assert fw_lo <= GOTTLIEB25_WIND_FRAC <= fw_hi, (
        "GOTTLIEB25_WIND_FRAC must lie inside GOTTLIEB25_WIND_FRAC_RANGE"
    )
    t_lo, t_hi = GOTTLIEB25_T_HMNS_RANGE
    assert 0 < t_lo < t_hi, "HMNS t_range ordering failed"


_selftest_gottlieb25()


# ---------------------------------------------------------------------------
# Module-level __getattr__ for deprecated attributes
# ---------------------------------------------------------------------------
def __getattr__(name):
    if name == "MEAN_MASS_EVOLVED":
        warnings.warn(
            "MEAN_MASS_EVOLVED is deprecated; use grb_rates.calibrate_mean_mass_evolved() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _MEAN_MASS_EVOLVED_VALUE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
