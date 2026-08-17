"""
Physical host-galaxy offset predictions for compact binary mergers.

Integrates post-SN systemic velocity through a Hernquist (1990) galaxy
potential to predict projected galactocentric offsets at merger time.
Replaces the ballistic upper-bound (v_sys * t_delay) with a
gravitationally-bound orbit model following the approach of
Bloom et al. (1999) and Fong & Berger (2013).

Typical usage
-------------
>>> from grb_offsets import compute_offsets_population
>>> offsets = compute_offsets_population(v_sys, delay_time, weights)
"""

from collections import namedtuple

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from grb_physics import H0_PLANCK15, OMEGA_M_PLANCK15

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
G_CGS = 6.674e-8  # cm^3 g^-1 s^-2
MSUN_G = 1.989e33  # g
KPC_CM = 3.0857e21  # cm
MYR_S = 3.1557e13  # s
KM_CM = 1e5  # cm
C_KMS = 299792.458  # km/s
ARCSEC_PER_RAD = 206264.806247  # arcsec

# Default host galaxy: typical SGRB host (Fong & Berger 2013)
DEFAULT_M_GAL = 10**10.5 * MSUN_G  # stellar mass [g]
DEFAULT_R_E = 5.0 * KPC_CM  # effective radius [cm]

# Representative host galaxy types.  Per-type stellar masses anchor on
# the sGRB host stellar-mass distribution of Leibler & Berger (2010),
# ApJ 725, 1202, Section 5 and Table 4: SF hosts cluster around log
# M*/Msun ~ 9.5 to 10.7 with a tail to massive disks, and ellipticals
# at ~10.7 to 11.2.  Nugent et al. (2022), ApJ 940, 57 (Fong+ 2022
# hosts II) confirms the SF-host median at log M*/Msun = 9.69 +0.75/-0.65
# from a uniformly modelled sample of 69 sGRB hosts.  Per-type effective
# radii anchor on the HST Sersic fits in Fong & Berger (2013) Sec. 3.2
# (median R_e ~ 3.6 kpc for SF disks) and Fong et al. (2022), ApJ 940,
# 56 (hosts I), which gives a 5 +/- 1 kpc co-fit median.  The 75/25
# SF/elliptical mixture is Fong & Berger (2013) Sec. 4; the 50/25 split
# between SF_disk and SF_massive is a CODE HEURISTIC consistent with
# the bimodal stellar-mass distribution in Leibler & Berger 2010.
HOST_MODELS = {
    "SF_disk": {
        "M_gal": 10**9.8 * MSUN_G,
        "R_e": 3.6 * KPC_CM,
        "weight": 0.50,
    },
    "SF_massive": {
        "M_gal": 10**10.5 * MSUN_G,
        "R_e": 5.0 * KPC_CM,
        "weight": 0.25,
    },
    "Elliptical": {
        "M_gal": 10**11.0 * MSUN_G,
        "R_e": 8.0 * KPC_CM,
        "weight": 0.25,
    },
}


SF_HOST_TRANSITION_T_MYR = 3000.0
"""Delay-time threshold [Myr] separating star-forming and elliptical hosts.

CODE HEURISTIC.  Systems with delay time below this value are routed to
``SF_disk`` / ``SF_massive`` hosts; longer-delay systems go to
``Elliptical``.  The 3 Gyr fiducial brackets the sGRB host stellar
population ages reported by Leibler & Berger (2010), ApJ 725, 1202, who
find late-type host stellar populations with mass-weighted ages ~ 0.1
to a few Gyr and early-type host stellar populations skewed older.
"""

SF_DISK_FRAC_OF_SF_HOSTS = 0.67
"""Probability of routing a SF-classified system to ``SF_disk``.

CODE HEURISTIC.  Within the SF-host subset, this fraction goes to the
low-mass disk template and the remainder to the more massive SF
template.  The 2:1 split (0.67 vs 0.33) lines up with the project-wide
``HOST_MODELS`` weights (``SF_disk`` 0.50, ``SF_massive`` 0.25), which
together cover the 0.75 SF fraction from Fong & Berger (2013) Sec. 4.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Dark-matter halo: stellar-to-halo mass, concentration, and NFW parameters
# ═══════════════════════════════════════════════════════════════════════════
# Behroozi, Wechsler & Conroy (2013), ApJ 770, 57 (arXiv:1207.6105), Eq. 3-4.
# The forward relation M_*(M_h) is a five-parameter fit whose coefficients
# evolve with scale factor a = 1/(1+z) through nu = exp(-4 a^2).  Values are
# the intrinsic-relation best fits quoted in their Section 5 (their published
# parameter listing).  We invert M_*(M_h) numerically for M_h(M_*).
BEHROOZI13_PARAMS = {
    "log10_M1_0": 11.514,
    "log10_M1_a": -1.793,
    "log10_M1_z": -0.251,
    "log10_eps_0": -1.777,
    "log10_eps_a": -0.006,
    "log10_eps_z": -0.000,
    "log10_eps_a2": -0.119,
    "alpha_0": -1.412,
    "alpha_a": 0.731,
    "delta_0": 3.508,
    "delta_a": 2.608,
    "delta_z": -0.043,
    "gamma_0": 0.316,
    "gamma_a": 1.319,
    "gamma_z": 0.279,
}

# Dutton & Maccio (2014), MNRAS 441, 3359 (arXiv:1402.7073), Eq. 8/10/11:
# the NFW c200-M200 relation in the Planck cosmology,
# log10 c200 = a + b log10(M200 / [1e12 h^-1 Msun]) with a, b evolving in z.
DUTTON14_C200 = {
    "b_0": -0.101,
    "b_z": 0.026,
    "a_0": 0.520,
    "a_inf_offset": 0.905 - 0.520,  # a(z=0) - a(z->inf)
    "z_scale": 0.617,
    "z_index": 1.21,
}

# Velocity-Verlet control.  The step count per system is set from the local
# orbital period so each revolution is resolved with a fixed sampling; the
# symplectic scheme then keeps the per-orbit energy error bounded rather than
# growing with orbit count (Verlet 1967; Leimkuhler & Reich 2004).  The phase
# cap applies only to bound orbits: a bound orbit randomises its radial phase
# within a few tens of radial periods, so a bound system whose delay time
# exceeds VERLET_MAX_ORBITS radial periods is integrated over that capped,
# phase-randomised baseline instead of its full (and unresolvably long)
# lifetime, which also removes any residual delay-time correlation in the
# integration error.  Unbound systems have no radial period and no phase to
# mix, so they are integrated for their full delay time; when the step count
# hits VERLET_MAX_STEPS the star is already far outside the host and nearly
# free, so the remaining time is completed ballistically.
VERLET_STEPS_PER_ORBIT = 120
VERLET_MAX_ORBITS = 40
VERLET_MIN_STEPS = 256
VERLET_MAX_STEPS = 32768


# ═══════════════════════════════════════════════════════════════════════════
# Hernquist potential
# ═══════════════════════════════════════════════════════════════════════════
def hernquist_scale_radius(R_e):
    """Convert projected half-light radius to Hernquist scale radius.

    For the Hernquist (1990) profile the *projected* half-light radius
    satisfies R_e / a ~ 1.8153 (numerical solution of the Abel-
    transformed surface-brightness integral; Hernquist 1990, Table 1).
    This differs from the 3-D half-mass relation r_half = a*(1+sqrt(2))
    (Eq. 38).  Observational effective radii from Sersic fits (e.g.
    Fong & Berger 2013 HST data) are projected half-light radii, so
    the projected relation is the correct one to use here.
    """
    return R_e / 1.8153


_R_CAP_FACTOR = 1000.0
"""Hard cap on radii expressed as a multiple of the Hernquist scale
radius ``a``.  Used by ``hernquist_birth_radius`` and ``integrate_orbit``
to bound the rare tail of the Hernquist profile (Hernquist 1990).

For the inverse-CDF sample r = a*sqrt(u)/(1 - sqrt(u)), the fraction
clipped at ``f * a`` is (1 - f/(1+f))^2.  Setting f = 1000 gives a
pile-up of (1/1001)^2 ~ 1e-6, which is negligible for the offset CDF
(Fong & Berger 2013 observed range tops out near 75 kpc, well below
1000*a even for compact SF hosts).  The previous 20*a cap clipped
~9.3% of draws into a delta function at 20*a, biasing the offset tail.
"""


def hernquist_birth_radius(a, rng=None, size=1, r_cap_factor=_R_CAP_FACTOR):
    """Draw birth radii from the Hernquist stellar density profile.

    The enclosed mass fraction is M(<r)/M_tot = (r/(r+a))^2, so the
    inverse CDF gives r = a * sqrt(u) / (1 - sqrt(u)) for uniform
    u in [0, 1).  Capped at ``r_cap_factor * a`` (default 1000*a)
    to bound numerical edge cases without significantly distorting
    the distribution.

    Parameters
    ----------
    a : float
        Hernquist scale radius [same units as output].
    rng : np.random.Generator, optional
    size : int
        Number of samples.
    r_cap_factor : float, optional
        Multiplier of ``a`` for the upper cap.  See ``_R_CAP_FACTOR``
        for the rationale; the default yields a pileup fraction
        ~1e-6.

    Returns
    -------
    r_birth : array of shape (size,)
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(0, 1, size=size)
    su = np.sqrt(u)
    r = a * su / (1.0 - su)
    return np.minimum(r, r_cap_factor * a)


def hernquist_potential(r, M_gal, a):
    """Hernquist (1990) gravitational potential Phi(r) = -GM/(r+a)."""
    return -G_CGS * M_gal / (r + a)


def hernquist_acceleration(r, M_gal, a):
    """Radial gravitational acceleration (inward, negative) for Hernquist profile."""
    return -G_CGS * M_gal / (r + a) ** 2


def escape_velocity(r, M_gal, a):
    """Escape velocity at radius r in Hernquist potential [cm/s]."""
    return np.sqrt(2.0 * G_CGS * M_gal / (r + a))


def hernquist_density(r, M_gal, a):
    """Hernquist (1990) Eq. 2 stellar mass density rho(r) [g/cm^3]."""
    return M_gal * a / (2.0 * np.pi * r * (r + a) ** 3)


# ═══════════════════════════════════════════════════════════════════════════
# Composite potential: Hernquist stars + NFW dark-matter halo
# ═══════════════════════════════════════════════════════════════════════════
# The stellar-only Hernquist potential understates the escape speed at large
# radius, so kicked systems that would stay bound in a realistic galaxy fly
# out and inflate the wide-offset tail.  A cold-dark-matter NFW halo (Navarro,
# Frenk & White 1997) sized from the host stellar mass closes that gap.  The
# Hernquist helpers above are left untouched as the stellar component; the
# ``total_*`` helpers below sum the two, and reduce to the pure Hernquist forms
# when the halo mass is zero.
HostPotential = namedtuple("HostPotential", ["M_star", "a", "Mc", "r_s"])
"""Composite host potential.

M_star, a
    Hernquist stellar mass [g] and scale radius [cm].
Mc, r_s
    NFW characteristic mass [g] and scale radius [cm].  ``Mc = 0`` marks a
    halo-free (stellar-only) potential, in which case every ``total_*`` helper
    collapses to its Hernquist counterpart.  ``Mc`` is the normalisation
    M200 / (ln(1+c) - c/(1+c)), so that M_NFW(<r) = Mc [ln(1+r/r_s) -
    (r/r_s)/(1+r/r_s)].
"""


def _hubble_cgs(z):
    """H(z) [1/s] for the project cosmology (Planck 2015; grb_physics)."""
    h0_cgs = H0_PLANCK15 * 1e5 / (1e3 * KPC_CM)  # (km/s/Mpc) -> 1/s
    return h0_cgs * np.sqrt(OMEGA_M_PLANCK15 * (1.0 + z) ** 3 + (1.0 - OMEGA_M_PLANCK15))


def rho_crit_z(z):
    """Critical density [g/cm^3] at redshift z for the project cosmology."""
    return 3.0 * _hubble_cgs(z) ** 2 / (8.0 * np.pi * G_CGS)


def _behroozi_log10_mstar(log10_Mh, z):
    """Behroozi+2013 Eq. 3: log10 M_*(M_h) [Msun in, Msun out], median relation."""
    p = BEHROOZI13_PARAMS
    a = 1.0 / (1.0 + z)
    nu = np.exp(-4.0 * a * a)
    log10_M1 = p["log10_M1_0"] + (p["log10_M1_a"] * (a - 1.0) + p["log10_M1_z"] * z) * nu
    log10_eps = (
        p["log10_eps_0"]
        + (p["log10_eps_a"] * (a - 1.0) + p["log10_eps_z"] * z) * nu
        + p["log10_eps_a2"] * (a - 1.0)
    )
    alpha = p["alpha_0"] + (p["alpha_a"] * (a - 1.0)) * nu
    delta = p["delta_0"] + (p["delta_a"] * (a - 1.0) + p["delta_z"] * z) * nu
    gamma = p["gamma_0"] + (p["gamma_a"] * (a - 1.0) + p["gamma_z"] * z) * nu

    def f(x):
        return -np.log10(10.0 ** (alpha * x) + 1.0) + delta * (
            np.log10(1.0 + np.exp(x)) ** gamma
        ) / (1.0 + np.exp(10.0 ** (-x)))

    x = log10_Mh - log10_M1
    return log10_eps + log10_M1 + f(x) - f(0.0)


def halo_mass_from_stellar_mass(M_star_g, z):
    """Invert Behroozi+2013 for the halo mass M200 [g] of a given M_* [g].

    The published relation gives the median M_* for a halo; we root-find the
    log10 halo mass that reproduces the target log10 M_* (Behroozi+2013 note
    the inverse is not the mean halo mass at fixed M_*, but for a fiducial
    template galaxy the median mapping is the appropriate one).
    """
    target = np.log10(M_star_g / MSUN_G)
    root = brentq(lambda lm: _behroozi_log10_mstar(lm, z) - target, 9.0, 15.5, xtol=1e-6)
    return 10.0**root * MSUN_G


def concentration_dutton_maccio(M200_g, z):
    """NFW c200 from Dutton & Maccio 2014 (Planck), M200 [g] in."""
    c = DUTTON14_C200
    h = H0_PLANCK15 / 100.0
    m200_hinv = (M200_g / MSUN_G) * h  # h^-1 Msun
    b = c["b_0"] + c["b_z"] * z
    a_dm = c["a_0"] + c["a_inf_offset"] * np.exp(-c["z_scale"] * z ** c["z_index"])
    return 10.0 ** (a_dm + b * np.log10(m200_hinv / 1e12))


def make_host_potential(M_gal, R_e, z=None, with_halo=True):
    """Build a ``HostPotential`` from stellar mass [g] and effective radius [cm].

    With ``with_halo=False`` (or a resolved halo mass of zero) the NFW piece is
    switched off and the container describes a pure Hernquist galaxy, so the
    ``total_*`` helpers reproduce the stellar-only model exactly.
    """
    a = hernquist_scale_radius(R_e)
    if not with_halo:
        return HostPotential(M_star=M_gal, a=a, Mc=0.0, r_s=1.0)
    if z is None:
        z = HALO_FIDUCIAL_Z
    M200 = halo_mass_from_stellar_mass(M_gal, z)
    c = concentration_dutton_maccio(M200, z)
    r200 = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit_z(z))) ** (1.0 / 3.0)
    r_s = r200 / c
    Mc = M200 / (np.log(1.0 + c) - c / (1.0 + c))
    return HostPotential(M_star=M_gal, a=a, Mc=Mc, r_s=r_s)


def nfw_enclosed_mass(r, Mc, r_s):
    """NFW mass within radius r [g].  Finite as r->0 (M ~ r^2)."""
    x = r / r_s
    return Mc * (np.log1p(x) - x / (1.0 + x))


def nfw_potential(r, Mc, r_s):
    """NFW gravitational potential [cm^2/s^2], Phi(r) = -G Mc ln(1+r/r_s)/r."""
    x = r / r_s
    return -G_CGS * Mc * np.log1p(x) / r


def nfw_acceleration(r, Mc, r_s):
    """Radial NFW acceleration (inward, negative) [cm/s^2]."""
    return -G_CGS * nfw_enclosed_mass(r, Mc, r_s) / r**2


def total_potential(r, host):
    """Composite Phi(r): Hernquist stars plus NFW halo when ``host.Mc > 0``."""
    phi = hernquist_potential(r, host.M_star, host.a)
    if host.Mc > 0.0:
        phi = phi + nfw_potential(r, host.Mc, host.r_s)
    return phi


def total_acceleration(r, host):
    """Composite inward radial acceleration [cm/s^2]."""
    acc = hernquist_acceleration(r, host.M_star, host.a)
    if host.Mc > 0.0:
        acc = acc + nfw_acceleration(r, host.Mc, host.r_s)
    return acc


def total_enclosed_mass(r, host):
    """Composite enclosed mass [g] (Hernquist stellar + NFW halo)."""
    # Hernquist M(<r) = M_star r^2 / (r + a)^2.
    m = host.M_star * r**2 / (r + host.a) ** 2
    if host.Mc > 0.0:
        m = m + nfw_enclosed_mass(r, host.Mc, host.r_s)
    return m


def total_escape_velocity(r, host):
    """Escape velocity in the composite potential [cm/s].

    Reduces to ``escape_velocity(r, host.M_star, host.a)`` when ``host.Mc = 0``
    because the NFW potential vanishes at infinity.
    """
    return np.sqrt(-2.0 * total_potential(r, host))


def jeans_sigma_r(r, host, r_max_factor=1e4):
    """Isotropic 1D velocity dispersion sigma_r(r) [cm/s] of the Hernquist
    tracer density in the composite potential.

    Solves the isotropic spherical Jeans equation

        sigma_r^2(r) = (1/rho_*) int_r^inf rho_*(r') dPhi_tot/dr' dr',

    with the Hernquist stellar density as the tracer and the composite
    (stars + halo) potential providing the force.  With the halo off this
    reduces to the closed-form Hernquist (1990) Eq. 10 dispersion; with a halo
    the tracers are hotter, so the population is launched in equilibrium in the
    total potential rather than in the stellar potential alone.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    r_upper = r_max_factor * host.a

    # Integrate in log radius: the integrand rho_* dPhi/dr spans many decades
    # and is sharply peaked near the lower limit, so a linear-range quad
    # under-resolves it.  With u = ln r' the measure becomes r' du and the
    # tail (~ r'^-4 du) is well behaved out to r_upper.
    def integrand_u(u):
        rp = np.exp(u)
        return hernquist_density(rp, host.M_star, host.a) * (-total_acceleration(rp, host)) * rp

    out = np.empty_like(r)
    for i, ri in enumerate(r):
        integral, _ = quad(integrand_u, np.log(ri), np.log(r_upper), limit=200)
        out[i] = integral / hernquist_density(ri, host.M_star, host.a)
    return np.sqrt(np.maximum(out, 0.0))


def _sigma_r_interpolator(host, n_grid=160):
    """Log-spaced sigma_r(r) interpolant for one host, built once and reused.

    Returns a callable r [cm] -> sigma_r [cm/s].  Sampling the Jeans integral
    on a grid and interpolating keeps the per-system cost negligible at
    population scale.
    """
    log_r = np.linspace(np.log10(1e-3 * host.a), np.log10(1e3 * host.a), n_grid)
    r_grid = 10.0**log_r
    sig = jeans_sigma_r(r_grid, host)
    log_sig = np.log10(np.maximum(sig, 1.0))

    def interp(r):
        r = np.asarray(r, dtype=float)
        lr = np.log10(np.clip(r, r_grid[0], r_grid[-1]))
        return 10.0 ** np.interp(lr, log_r, log_sig)

    return interp


# ═══════════════════════════════════════════════════════════════════════════
# Orbit integration
# ═══════════════════════════════════════════════════════════════════════════
def _orbit_rhs(t, y, M_gal, a):
    """Right-hand side for 2D orbit in Hernquist potential (polar: r, v_r, L)."""
    r, vr = y[0], y[1]
    L = y[2]
    r = max(r, 1e-3 * a)
    a_grav = hernquist_acceleration(r, M_gal, a)
    drdt = vr
    dvrdt = a_grav + L**2 / r**3
    dLdt = 0.0
    return [drdt, dvrdt, dLdt]


def integrate_orbit(v_sys_cm, t_delay_s, M_gal, a, r0=None, theta_launch=None, rng=None):
    """Integrate a single orbit in Hernquist potential.

    Parameters
    ----------
    v_sys_cm : float
        Systemic velocity [cm/s].
    t_delay_s : float
        Delay time (merger minus formation) [s].
    M_gal : float
        Galaxy mass [g].
    a : float
        Hernquist scale radius [cm].
    r0 : float or None
        Launch radius as fraction of scale radius.  If None (default),
        draws from the Hernquist stellar density profile via inverse-CDF
        sampling so that birth sites trace the galaxy light.
    theta_launch : float or None
        Angle between kick velocity and radial direction [rad].
        None (default) draws isotropically from arccos(U(-1,1)),
        following Bloom et al. (1999).
    rng : np.random.Generator, optional
        Random generator for isotropic sampling.

    Returns
    -------
    r_final : float
        Galactocentric distance at merger [cm].
    """
    if r0 is None:
        if rng is None:
            rng = np.random.default_rng()
        r_init = hernquist_birth_radius(a, rng=rng, size=1)[0]
    else:
        r_init = r0 * a

    if v_sys_cm <= 0 or t_delay_s <= 0:
        return r_init

    if theta_launch is None:
        if rng is None:
            rng = np.random.default_rng()
        theta_launch = np.arccos(rng.uniform(-1, 1))

    vr0 = v_sys_cm * np.cos(theta_launch)
    vt0 = v_sys_cm * np.sin(theta_launch)
    L0 = r_init * vt0

    y0 = [r_init, vr0, L0]
    t_span = (0.0, t_delay_s)

    try:
        sol = solve_ivp(
            _orbit_rhs,
            t_span,
            y0,
            args=(M_gal, a),
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            max_step=t_delay_s / 50.0,
            dense_output=False,
        )
        if sol.success:
            r_final = abs(sol.y[0, -1])
            return min(r_final, _R_CAP_FACTOR * a)
        return r_init
    except (RuntimeError, ValueError):
        return r_init


def compute_offset_single(
    v_sys_km, t_delay_myr, M_gal=DEFAULT_M_GAL, R_e=DEFAULT_R_E, n_angles=8, rng=None
):
    """Compute projected offset for a single binary [kpc].

    Launches the binary from a birthplace near the galaxy center and
    integrates through the Hernquist potential.  The projected offset
    is the median over ``n_angles`` random viewing angles.

    Parameters
    ----------
    v_sys_km : float
        Systemic velocity [km/s].
    t_delay_myr : float
        Delay time [Myr].
    M_gal : float
        Galaxy stellar mass [g].
    R_e : float
        Effective radius [cm].
    n_angles : int
        Number of random projection angles for median.
    rng : np.random.Generator, optional

    Returns
    -------
    offset_kpc : float
        Median projected galactocentric offset [kpc].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    a = hernquist_scale_radius(R_e)
    v_cm = v_sys_km * KM_CM
    t_s = t_delay_myr * MYR_S

    r_3d = integrate_orbit(v_cm, t_s, M_gal, a, rng=rng)

    cos_theta = rng.uniform(-1, 1, size=n_angles)
    projected = r_3d * np.sqrt(1.0 - cos_theta**2)
    offset_kpc = np.median(projected) / KPC_CM
    return offset_kpc


# ═══════════════════════════════════════════════════════════════════════════
# Analytic bound-orbit approximation (fast path)
# ═══════════════════════════════════════════════════════════════════════════
def _hernquist_apocenter(E, L, M_gal, a):
    """Find the apocenter of a bound orbit in a Hernquist potential.

    Solves E = L^2/(2*r^2) - GM/(r+a) for the outer turning point
    where the radial velocity vanishes.
    """
    GM = G_CGS * M_gal

    def eff_potential_minus_E(r):
        return 0.5 * L**2 / r**2 - GM / (r + a) - E

    try:
        return brentq(eff_potential_minus_E, 1e-3 * a, _R_CAP_FACTOR * a)
    except ValueError:
        return _R_CAP_FACTOR * a


def _analytic_offset(v_sys_km, t_delay_myr, M_gal, a, r0_frac=None, rng=None):
    """Fast analytic estimate of 3D galactocentric distance at merger.

    For bound orbits in a Hernquist potential, uses energy and angular
    momentum conservation to find the apocenter via root-finding in the
    true Hernquist effective potential, then estimates the time-averaged
    radius.  Falls back to full integration for unbound orbits and
    short-delay systems.

    The kick direction ``theta_launch`` is drawn isotropically once
    per call (Bloom+ 1999) and is threaded through to ``integrate_orbit``
    so the bound/unbound branching uses the same orbit that is then
    actually integrated.

    Returns r_3d [cm].
    """
    if rng is None:
        rng = np.random.default_rng()

    if r0_frac is None:
        r0 = hernquist_birth_radius(a, rng=rng, size=1)[0]
    else:
        r0 = r0_frac * a
    v_cm = v_sys_km * KM_CM
    t_s = t_delay_myr * MYR_S

    if v_cm <= 0 or t_s <= 0:
        return r0

    theta_launch = np.arccos(rng.uniform(-1, 1))

    E = 0.5 * v_cm**2 + hernquist_potential(r0, M_gal, a)
    v_esc = escape_velocity(r0, M_gal, a)

    if v_cm >= v_esc:
        r0_f = r0 / a
        return integrate_orbit(v_cm, t_s, M_gal, a, r0=r0_f, theta_launch=theta_launch, rng=rng)

    vt = v_cm * np.sin(theta_launch)
    L = r0 * vt

    GM = G_CGS * M_gal

    r_apo = _hernquist_apocenter(E, L, M_gal, a)
    r_mean = (r0 + r_apo) / 2.0

    # Period heuristic.  The exact radial period in a Hernquist
    # potential is T_r = 2 * integral_{r_peri}^{r_apo} dr / sqrt(2*(E - Phi_eff))
    # (Binney & Tremaine 2008, "Galactic Dynamics", Section 3.1).
    # The expression below recovers the Kepler limit for r_mean >> a
    # and is order-of-magnitude correct for r_mean ~ a; for r_mean << a
    # it overestimates the period.  Used here only to branch between
    # the time-averaged-radius shortcut (when t >> P_est) and full
    # numerical integration, so a coarse estimate is sufficient.
    P_est = 2.0 * np.pi * np.sqrt(r_mean**3 / GM) * (1 + a / r_mean)

    if t_s > 5.0 * P_est:
        return r_mean
    else:
        r0_f = r0 / a
        return integrate_orbit(v_cm, t_s, M_gal, a, r0=r0_f, theta_launch=theta_launch, rng=rng)


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized population computation
# ═══════════════════════════════════════════════════════════════════════════
def _vectorized_orbit_3d(v_cm, t_s, M_gal, a, rng, n_steps=400, newton_iter=30):
    """Batch orbit integrator: scalar in, array out for N systems.

    Reproduces ``_analytic_offset`` (line 275) for the whole population at
    once.  The control flow is identical: compute energy and angular
    momentum from a per-system Hernquist birth radius and isotropic kick;
    Newton-iterate the apocenter (replaces the brentq root find at line
    270); use the time-averaged radius for systems with ``t_s > 5 P_est``;
    otherwise integrate with fixed-step RK4 against ``_orbit_rhs`` (line
    136).  All operations broadcast over ``(N,)``.

    Parameters
    ----------
    v_cm, t_s : (N,) arrays
        Kick speeds [cm/s] and delay times [s].
    M_gal, a : float
        Hernquist galaxy mass [g] and scale radius [cm].
    rng : np.random.Generator
        For birth radius and kick angle draws.
    n_steps : int, optional
        Fixed RK4 step count for systems that need full integration.  At
        n_steps = 400 with typical t_s in [10 Myr, 14 Gyr] and Hernquist
        orbital periods ~10-1000 Myr, dt is ~2.5 to 25 percent of the
        local period, so RK4 energy drift stays << 1 percent (Hairer,
        Norsett, Wanner "Solving ODEs I", Chapter II.1).
    newton_iter : int, optional
        Iteration count for the vectorized apocenter root finder.
        Converges in <10 in practice; 30 leaves headroom.

    Returns
    -------
    r_3d : (N,) array
        Galactocentric distance at merger [cm], capped at
        ``_R_CAP_FACTOR * a``.
    """
    GM = G_CGS * M_gal
    a_eps = 1e-3 * a
    r_cap = _R_CAP_FACTOR * a

    v_cm = np.asarray(v_cm, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    N = v_cm.size

    if N == 0:
        return np.zeros(0)

    # Vectorized Hernquist birth radius via inverse CDF (line 85-115).
    u = rng.uniform(0, 1, size=N)
    su = np.sqrt(u)
    r0 = np.minimum(a * su / (1.0 - su), r_cap)

    # Isotropic kick angle (Bloom+ 1999); cos drawn flat in [-1, 1].
    cos_kick = rng.uniform(-1, 1, size=N)
    sin_kick = np.sqrt(np.maximum(0.0, 1.0 - cos_kick * cos_kick))

    # Initial conditions in (r, vr, L) with L conserved.
    vr0 = v_cm * cos_kick
    vt0 = v_cm * sin_kick
    L = r0 * vt0

    # Energy and bound/unbound branch.
    E = 0.5 * v_cm * v_cm - GM / (r0 + a)
    v_esc_sq = 2.0 * GM / (r0 + a)
    bound = (v_cm * v_cm) < v_esc_sq

    # Vectorized Newton iteration for the apocenter.
    # f(r)  = 0.5 L^2 / r^2 - GM / (r + a) - E
    # f'(r) = -L^2 / r^3 + GM / (r + a)^2
    # Initial guess: 10 a, well outside the typical Hernquist core; clipped
    # every step into [a_eps, r_cap].
    r_apo = np.full(N, r_cap)
    if bound.any():
        r_iter = np.where(bound, 10.0 * a, np.nan)
        for _ in range(newton_iter):
            r_safe = np.maximum(r_iter, a_eps)
            f = 0.5 * L * L / (r_safe * r_safe) - GM / (r_safe + a) - E
            fp = -L * L / (r_safe**3) + GM / (r_safe + a) ** 2
            # Avoid division by zero when fp ~ 0 at degenerate L.
            step = f / np.where(np.abs(fp) > 1e-300, fp, 1e-300)
            r_iter = np.clip(r_iter - step, a_eps, r_cap)
        r_apo = np.where(bound, r_iter, r_cap)

    # Period heuristic, matching _analytic_offset line 331.
    r_mean_est = 0.5 * (r0 + r_apo)
    P_est = (
        2.0
        * np.pi
        * np.sqrt(np.maximum(r_mean_est, a_eps) ** 3 / GM)
        * (1.0 + a / np.maximum(r_mean_est, a_eps))
    )

    # Trivial branches: zero kick or zero delay -> stay at r0.
    trivial = (v_cm <= 0) | (t_s <= 0)

    # Phase-mixed branch: bound and t >> period -> time-averaged radius.
    use_mean = bound & (~trivial) & (t_s > 5.0 * P_est)

    # Everything else needs RK4 (unbound or short delay).
    integrate = ~(trivial | use_mean)

    r_3d = np.where(trivial, r0, np.where(use_mean, r_mean_est, np.nan))

    if integrate.any():
        idx = np.where(integrate)[0]
        r = r0[idx].copy()
        vr = vr0[idx].copy()
        L_i = L[idx]
        t_i = t_s[idx]
        dt = t_i / n_steps  # (M,)

        for _ in range(n_steps):
            r_safe = np.maximum(r, a_eps)
            k1_r = vr
            k1_v = -GM / (r_safe + a) ** 2 + L_i * L_i / (r_safe**3)

            r2 = np.maximum(r + 0.5 * dt * k1_r, a_eps)
            v2 = vr + 0.5 * dt * k1_v
            k2_r = v2
            k2_v = -GM / (r2 + a) ** 2 + L_i * L_i / (r2**3)

            r3 = np.maximum(r + 0.5 * dt * k2_r, a_eps)
            v3 = vr + 0.5 * dt * k2_v
            k3_r = v3
            k3_v = -GM / (r3 + a) ** 2 + L_i * L_i / (r3**3)

            r4 = np.maximum(r + dt * k3_r, a_eps)
            v4 = vr + dt * k3_v
            k4_r = v4
            k4_v = -GM / (r4 + a) ** 2 + L_i * L_i / (r4**3)

            r = r + dt * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r) / 6.0
            vr = vr + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0

        r_3d[idx] = np.minimum(np.abs(r), r_cap)

    # Final safety net: replace any residual NaN (should not occur after
    # the explicit branching above, but pin it to the cap so downstream
    # CDF helpers do not propagate non-finite values).
    return np.where(np.isfinite(r_3d), r_3d, r_cap)


def _apocenter_composite(E, L, r0, host, r_cap, n_iter=64):
    """Vectorized outer turning point of a bound orbit in the composite potential.

    Solves ``0.5 L^2 / r^2 + Phi_tot(r) = E`` for the outer root by bisection on
    ``[r0, r_cap]``.  The radial kinetic term ``g(r) = E - 0.5 L^2/r^2 -
    Phi_tot(r)`` satisfies ``g(r0) = 0.5 v_r0^2 >= 0`` (whatever the sign of the
    radial velocity) and ``g(r_cap) = E < 0`` for a bound orbit, so the bracket
    always contains the apocenter.  Bisection rather than Newton because the
    bracket is guaranteed and the vectorized update stays branch-free.
    """
    lo = np.array(r0, dtype=float, copy=True)
    hi = np.full_like(lo, r_cap)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        g = E - 0.5 * L * L / (mid * mid) - total_potential(mid, host)
        # g >= 0 inside the orbit, g < 0 beyond the apocenter.
        inside = g >= 0.0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return 0.5 * (lo + hi)


def _vectorized_orbit_composite(v_cm, t_s, host, rng, sigma_interp=None):
    """Batch orbit integrator for the composite potential (velocity-Verlet).

    Symplectic replacement for the RK4 branch of ``_vectorized_orbit_3d`` used
    on the halo science path.  A bound orbit that survives for hundreds of
    radial periods would accumulate secular energy drift under a non-symplectic
    scheme, and the number of periods grows with delay time, exactly the axis
    the GRB-class offsets differ along; velocity-Verlet keeps the energy error
    bounded instead.

    Bound and unbound systems are treated differently, gated on the launch
    speed against the local escape speed in the *total* potential:

    - Bound orbits oscillate between a pericentre and an apocentre with a finite
      radial period ``P_r``.  Once the delay time exceeds ``VERLET_MAX_ORBITS``
      radial periods the merger phase is effectively uniform, so the orbit is
      integrated over a capped, phase-randomised baseline ``(VERLET_MAX_ORBITS
      + U[0,1]) * P_r`` rather than its full lifetime.  The cap uses the radial
      period (from a vectorised apocentre solve), not the circular period at the
      birth radius, so an eccentric orbit is phase-mixed over its true period.
    - Unbound systems have no radial period and escape monotonically, so they
      are integrated for their full delay time.  When the required step count
      would exceed ``VERLET_MAX_STEPS`` the star is already far outside the host
      and nearly free, so the remaining time is finished ballistically from the
      integrated state.

    Integration is in 2D Cartesian coordinates in the (conserved) orbital
    plane rather than the reduced 1D radial equation: the effective radial
    potential carries a centrifugal barrier L^2/2r^2 whose force diverges at
    small r, which a fixed-step scheme cannot resolve for the near-radial
    orbits isotropic kicks produce.  In Cartesian phase space the force is the
    ordinary gravitational acceleration, finite everywhere (both the Hernquist
    and NFW M(<r) ~ r^2 near the centre), so a near-radial orbit simply passes
    through small r without a coordinate singularity and the symplectic step
    stays stable.

    Parameters
    ----------
    v_cm, t_s : (N,) arrays
        Kick speeds [cm/s] and delay times [s].
    host : HostPotential
        Composite potential; ``host.Mc = 0`` gives the stellar-only limit.
    rng : np.random.Generator
        Birth radius, kick direction, and pre-kick dispersion draws.
    sigma_interp : callable or None
        ``r [cm] -> sigma_r [cm/s]``.  When given, each system also receives an
        isotropic progenitor velocity drawn from N(0, sigma_r(r0)) per axis
        (the isotropic-model equilibrium), added to the supernova kick.  None
        disables the pre-kick term.

    Returns
    -------
    r_3d : (N,) array
        Galactocentric distance at merger [cm], capped at ``_R_CAP_FACTOR*a``.
    """
    a = host.a
    r_cap = _R_CAP_FACTOR * a

    v_cm = np.asarray(v_cm, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    N = v_cm.size
    if N == 0:
        return np.zeros(0)

    # Birth radii trace the stellar light regardless of the halo.
    u = rng.uniform(0, 1, size=N)
    su = np.sqrt(u)
    r0 = np.minimum(a * su / (1.0 - su), r_cap)

    # Isotropic kick as a 3-vector; take r-hat = z-hat without loss of
    # generality for a spherical potential.
    kv = rng.standard_normal((N, 3))
    kv /= np.linalg.norm(kv, axis=1, keepdims=True)
    v_vec = kv * v_cm[:, None]

    if sigma_interp is not None:
        sig = np.asarray(sigma_interp(r0), dtype=float)
        v_vec = v_vec + rng.standard_normal((N, 3)) * sig[:, None]

    # Launch in the orbital plane: position along x, velocity split into the
    # radial (x) and tangential (y) components.  The plane is fixed because L
    # is conserved, so 2D Cartesian captures the full orbit.
    vr0 = v_vec[:, 2]
    vt0 = np.hypot(v_vec[:, 0], v_vec[:, 1])

    trivial = (v_cm <= 0) | (t_s <= 0)
    r_3d = np.where(trivial, r0, np.nan)

    integrate = ~trivial
    if integrate.any():
        idx = np.where(integrate)[0]
        r_3d[idx] = _advance_planar_composite(
            r0[idx], vr0[idx], vt0[idx], t_s[idx], host, rng, phase_cap=True
        )

    return np.where(np.isfinite(r_3d), r_3d, r_cap)


def _advance_planar_composite(r_init, vr_init, vt_init, t_s, host, rng, phase_cap=True):
    """Advance planar orbits in the composite potential and return final |r|.

    Given each system's initial galactocentric radius ``r_init``, radial speed
    ``vr_init`` and tangential speed ``vt_init``, integrate for ``t_s`` and
    return the final galactocentric distance [cm], capped at
    ``_R_CAP_FACTOR * host.a``.  This is the shared body of the single- and
    two-leg composite paths; the launch state is supplied by the caller so an
    arbitrary planar state can be handed in (a later leg rotates its Cartesian
    state onto the ``+x`` axis first, which is exact for a spherical potential
    because only ``|r|`` is returned).

    Bound orbits are phase-mixed over their radial period when ``phase_cap`` is
    True; the first (pre-second-supernova) leg passes ``phase_cap=False`` so its
    short baseline is integrated in full and it draws no random numbers.
    """
    a = host.a
    a_eps = 1e-3 * a
    r_cap = _R_CAP_FACTOR * a

    r0_i = np.asarray(r_init, dtype=float)
    vr0_i = np.asarray(vr_init, dtype=float)
    vt0_i = np.asarray(vt_init, dtype=float)
    t_i = np.asarray(t_s, dtype=float)
    if r0_i.size == 0:
        return np.zeros(0)

    r0_safe = np.maximum(r0_i, a_eps)

    # Bound/unbound split from the launch speed against the escape speed in
    # the TOTAL potential (halo included).  phi0 < 0, so v_esc^2 = -2 phi0.
    phi0 = total_potential(r0_safe, host)
    v_tot2 = vr0_i * vr0_i + vt0_i * vt0_i
    bound = v_tot2 < (-2.0 * phi0)

    # Circular dynamical time at r0, the step-size scale for unbound systems
    # (which have no radial period).
    m_enc0 = total_enclosed_mass(r0_safe, host)
    t_dyn = 2.0 * np.pi * np.sqrt(r0_safe**3 / (G_CGS * m_enc0))

    # Radial period of the bound orbits from the apocentre solve; unbound
    # entries fall back to t_dyn (unused for the cap, since only bound orbits
    # are phase-mixed).
    E = 0.5 * v_tot2 + phi0
    L = r0_i * vt0_i
    period = t_dyn.copy()
    if bound.any():
        bmask = bound
        r_apo = _apocenter_composite(E[bmask], L[bmask], r0_i[bmask], host, r_cap)
        r_mean = 0.5 * (r0_i[bmask] + r_apo)
        r_mean = np.maximum(r_mean, a_eps)
        m_mean = total_enclosed_mass(r_mean, host)
        period[bmask] = 2.0 * np.pi * np.sqrt(r_mean**3 / (G_CGS * m_mean))

    # Phase cap applies only to bound orbits, keyed to the radial period.
    if phase_cap:
        n_periods = t_i / period
        mixed = bound & (n_periods > VERLET_MAX_ORBITS)
        frac = rng.uniform(0.0, 1.0, size=t_i.size)
        t_eff = np.where(mixed, (VERLET_MAX_ORBITS + frac) * period, t_i)
    else:
        t_eff = t_i

    # Fine step keeps a fixed sampling of the local period for every system.
    # When the fine-resolution step count would exceed the ceiling (only the
    # long-lived unbound systems), integrate VERLET_MAX_STEPS fine steps then
    # coast ballistically for the remainder.
    dt_fine = period / VERLET_STEPS_PER_ORBIT
    n_full = np.ceil(t_eff / dt_fine).astype(np.int64)
    ceiling = n_full > VERLET_MAX_STEPS
    n_steps_arr = np.clip(n_full, VERLET_MIN_STEPS, VERLET_MAX_STEPS)
    t_step = np.where(ceiling, n_steps_arr * dt_fine, t_eff)
    dt_arr = t_step / n_steps_arr
    t_ball = np.maximum(t_eff - t_step, 0.0)

    def accel(xx, yy):
        rr = np.maximum(np.hypot(xx, yy), a_eps)
        a_rad = total_acceleration(rr, host)  # inward (negative)
        return a_rad * xx / rr, a_rad * yy / rr

    r_out = np.empty_like(r0_i)
    log2n = np.ceil(np.log2(np.maximum(n_steps_arr, 1))).astype(int)
    for k in np.unique(log2n):
        bucket = np.where(log2n == k)[0]
        n_steps_b = int(2**k)
        x = r0_i[bucket].copy()
        y = np.zeros_like(x)
        vx = vr0_i[bucket].copy()
        vy = vt0_i[bucket].copy()
        dt = dt_arr[bucket]  # per-system step [s]

        ax, ay = accel(x, y)
        for _ in range(n_steps_b):
            vx += 0.5 * dt * ax
            vy += 0.5 * dt * ay
            x += dt * vx
            y += dt * vy
            ax, ay = accel(x, y)
            vx += 0.5 * dt * ax
            vy += 0.5 * dt * ay

        # Ballistic coast for the ceiling-bound (unbound, far-out) systems.
        tb = t_ball[bucket]
        x = x + vx * tb
        y = y + vy * tb
        r_out[bucket] = np.hypot(x, y)

    return np.minimum(r_out, r_cap)


def _vectorized_orbit_composite_two_leg(v1_cm, v2_cm, t1_s, t2_s, host, rng, sigma_interp=None):
    """Two-leg composite-potential orbit integrator returning final |r| [cm].

    The systemic velocity COMPAS reports is the post-supernova center-of-mass
    speed, so a binary only travels at ``v_sys`` from the supernova that
    produced it, not from zero-age.  This splits the drift into the two legs
    the data actually support:

    - Leg 1 (SN1 -> SN2, ``t1_s``): the binary starts from its equilibrium
      birth radius with an isotropic kick of magnitude ``v1_cm`` plus, when
      ``sigma_interp`` is given, the progenitor's pre-kick Jeans dispersion.
      This leg is short (median ~1 Myr) and is integrated in full without a
      phase cap.  Only the resulting galactocentric radius is carried forward;
      the orientation is irrelevant for a spherical potential.
    - Leg 2 (SN2 -> merger, ``t2_s``): the kick contribution to the
      center-of-mass velocity is reassigned to a fresh isotropic vector of
      magnitude ``v2_cm`` (COMPAS reports the cumulative COM kick speed, so
      ``v2_cm`` replaces rather than adds to the leg-1 kick ``v1_cm``), and the
      binary's ambient orbital velocity is redrawn from the Jeans dispersion at
      the leg-1 end radius, exactly as the single-leg path builds its launch
      velocity.  The orbit is then integrated with the same bound/unbound,
      phase-cap, and ballistic-coast treatment.  Dropping the dispersion here
      would discard the galactic orbital velocity, which COMPAS's ``v_sys``
      does not include.

    The pre-SN1 interval is deliberately not integrated: before the first
    supernova the binary is an ordinary star sitting in equilibrium, so its
    position at SN1 is statistically a fresh draw from the birth-radius
    distribution, which is exactly ``r0``.
    """
    a = host.a
    r_cap = _R_CAP_FACTOR * a

    v1 = np.asarray(v1_cm, dtype=float)
    v2 = np.asarray(v2_cm, dtype=float)
    t1 = np.asarray(t1_s, dtype=float)
    t2 = np.asarray(t2_s, dtype=float)
    N = v2.size
    if N == 0:
        return np.zeros(0)

    # Birth radii trace the stellar light regardless of the halo.
    u = rng.uniform(0, 1, size=N)
    su = np.sqrt(u)
    r0 = np.minimum(a * su / (1.0 - su), r_cap)

    # Leg-1 launch velocity: isotropic kick of magnitude v1 plus the pre-kick
    # Jeans dispersion.  r-hat = z-hat without loss of generality.
    kv = rng.standard_normal((N, 3))
    kv /= np.linalg.norm(kv, axis=1, keepdims=True)
    v_vec = kv * v1[:, None]
    if sigma_interp is not None:
        sig = np.asarray(sigma_interp(r0), dtype=float)
        v_vec = v_vec + rng.standard_normal((N, 3)) * sig[:, None]
    vr1 = v_vec[:, 2]
    vt1 = np.hypot(v_vec[:, 0], v_vec[:, 1])

    # Leg 1: integrate the short pre-SN2 drift in full (no phase cap).  Systems
    # with a non-positive leg-1 baseline stay at their birth radius.
    r1 = r0.copy()
    leg1 = t1 > 0
    if leg1.any():
        j = np.where(leg1)[0]
        r1[j] = _advance_planar_composite(r0[j], vr1[j], vt1[j], t1[j], host, rng, phase_cap=False)

    # Leg 2: rebuild the launch velocity at the leg-1 end radius exactly as the
    # single-leg path does at r0, a fresh isotropic kick of magnitude v2 plus
    # the Jeans dispersion.  r1 is the radius; the position is placed on the
    # +x axis of a rotated frame, which is exact for a spherical potential
    # because only |r| is returned (the "arbitrary planar state" entry).
    kv2 = rng.standard_normal((N, 3))
    kv2 /= np.linalg.norm(kv2, axis=1, keepdims=True)
    v_vec2 = kv2 * v2[:, None]
    if sigma_interp is not None:
        sig2 = np.asarray(sigma_interp(r1), dtype=float)
        v_vec2 = v_vec2 + rng.standard_normal((N, 3)) * sig2[:, None]
    vr2 = v_vec2[:, 2]
    vt2 = np.hypot(v_vec2[:, 0], v_vec2[:, 1])

    r_final = r1.copy()
    leg2 = t2 > 0
    if leg2.any():
        j = np.where(leg2)[0]
        r_final[j] = _advance_planar_composite(
            r1[j], vr2[j], vt2[j], t2[j], host, rng, phase_cap=True
        )

    return np.where(np.isfinite(r_final), np.minimum(r_final, r_cap), r_cap)


# Cache of (HostPotential, sigma interpolant) keyed by the physical inputs, so
# the Jeans integral and SMHM inversion run once per distinct host per session.
_HOST_POTENTIAL_CACHE: dict = {}


def _host_potential_and_sigma(M_gal, R_e, z, with_halo, with_dispersion):
    """Return a cached ``(HostPotential, sigma_interp)`` for one host."""
    key = (
        float(M_gal),
        float(R_e),
        None if z is None else float(z),
        bool(with_halo),
        bool(with_dispersion),
    )
    cached = _HOST_POTENTIAL_CACHE.get(key)
    if cached is not None:
        return cached
    host = make_host_potential(M_gal, R_e, z=z, with_halo=with_halo)
    sigma_interp = _sigma_r_interpolator(host) if with_dispersion else None
    _HOST_POTENTIAL_CACHE[key] = (host, sigma_interp)
    return host, sigma_interp


def compute_offsets_population_vectorized(
    v_sys_km,
    t_delay_myr,
    weights=None,
    M_gal=DEFAULT_M_GAL,
    R_e=DEFAULT_R_E,
    max_systems=50000,
    n_steps=400,
    rng=None,
    with_halo=False,
    with_dispersion=None,
    halo_z=None,
    integrator="auto",
    v_sys_1_km=None,
    t_leg1_myr=None,
):
    """Vectorized drop-in replacement for ``compute_offsets_population``.

    Same I/O contract: returns the dict ``{'offsets_kpc', 'indices',
    'weights_sub'}``.  Replaces the per-system Python loop in
    ``compute_offsets_population`` with a single batched integrator.

    Parameters
    ----------
    v_sys_km, t_delay_myr : array-like
        Systemic velocity [km/s] and delay time [Myr].  May contain NaN
        or non-positive values; these are filtered out via the same
        ``valid`` mask used by the legacy code.  In the two-leg mode
        (``v_sys_1_km`` and ``t_leg1_myr`` supplied) these are the post-SN2
        velocity and the post-DCO inspiral time.
    weights : array-like, optional
        STROOPWAFEL weights aligned with the inputs; defaults to ones.
    M_gal, R_e : float
        Hernquist galaxy mass [g] and effective radius [cm].
    max_systems : int
        Subsample cap.  The draw is a simple random subsample without
        replacement, which is exactly unbiased for any weighted functional;
        the retained ``weights_sub`` are the original STROOPWAFEL weights.
    n_steps : int, optional
        RK4 step count for the stellar-only orbit integrator.  See
        ``_vectorized_orbit_3d``.  Ignored on the composite (halo) path,
        which sets its own per-system step count from the local orbital
        period.
    rng : np.random.Generator, optional
        Reproducible random source.  Defaults to ``default_rng(42)``.
    with_halo : bool, optional
        Add an NFW dark-matter halo sized from ``M_gal`` (Behroozi+2013 SMHM,
        Dutton & Maccio 2014 concentration) and integrate with symplectic
        velocity-Verlet via ``_vectorized_orbit_composite``.  Default False
        keeps the byte-identical stellar-only Hernquist path so the legacy
        agreement test in ``tests/unit/test_phase4_helpers.py`` still holds.
    with_dispersion : bool or None, optional
        Add the progenitor's pre-kick velocity from the Jeans dispersion of
        the composite potential.  ``None`` (default) follows ``with_halo``.
    halo_z : float or None, optional
        Redshift at which the halo is built.  ``None`` uses
        ``HALO_FIDUCIAL_Z``.
    integrator : {'auto', 'composite', 'legacy'}, optional
        Which orbit integrator to use.  ``'auto'`` (default) picks the
        symplectic composite path when a halo or dispersion is requested and
        the legacy RK4 Hernquist path otherwise, preserving historical
        behaviour.  ``'composite'`` forces the velocity-Verlet path even with
        the halo off (``with_halo=False`` gives the stellar-only Verlet edge,
        the clean one-variable reference for the panel-1 model band).
        ``'legacy'`` forces the RK4 Hernquist path and ignores the halo.
    v_sys_1_km, t_leg1_myr : array-like or None, optional
        Post-SN1 systemic velocity [km/s] and the SN1-to-SN2 baseline [Myr].
        When both are given the orbit is integrated in two legs (drift at
        ``v_sys_1_km`` for ``t_leg1_myr``, then at ``v_sys_km`` for
        ``t_delay_myr``); this requires the composite integrator.  When either
        is None the single-leg path integrates ``t_delay_myr`` at
        ``v_sys_km``.

    Returns
    -------
    dict with keys ``'offsets_kpc' : (M,)``, ``'indices' : (M,)``,
    ``'weights_sub' : (M,)`` where M = min(N_valid, max_systems).
    """
    if integrator not in ("auto", "composite", "legacy"):
        raise ValueError(f"integrator must be 'auto', 'composite', or 'legacy'; got {integrator!r}")
    if rng is None:
        rng = np.random.default_rng(42)

    v_sys_km = np.asarray(v_sys_km, dtype=float)
    t_delay_myr = np.asarray(t_delay_myr, dtype=float)

    two_leg = v_sys_1_km is not None and t_leg1_myr is not None
    if two_leg:
        v_sys_1_km = np.asarray(v_sys_1_km, dtype=float)
        t_leg1_myr = np.asarray(t_leg1_myr, dtype=float)

    valid = np.isfinite(v_sys_km) & np.isfinite(t_delay_myr) & (v_sys_km > 0) & (t_delay_myr > 0)
    if two_leg:
        valid &= (
            np.isfinite(v_sys_1_km)
            & np.isfinite(t_leg1_myr)
            & (v_sys_1_km >= 0)
            & (t_leg1_myr >= 0)
        )
    valid_idx = np.where(valid)[0]

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        w_valid = weights[valid_idx]
    else:
        w_valid = np.ones(len(valid_idx))

    # Simple random subsample without replacement: unbiased for any weighted
    # functional, so the retained weights stay the original STROOPWAFEL values.
    # A weight-proportional draw that also kept the weights would apply them
    # twice (the estimator would go as w^2).
    if len(valid_idx) > max_systems:
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    if len(valid_idx) == 0:
        return {"offsets_kpc": np.zeros(0), "indices": valid_idx, "weights_sub": w_valid}

    a = hernquist_scale_radius(R_e)
    v_cm = v_sys_km[valid_idx] * KM_CM
    t_s = t_delay_myr[valid_idx] * MYR_S

    if with_dispersion is None:
        with_dispersion = with_halo
    use_composite = integrator == "composite" or (
        integrator == "auto" and (with_halo or with_dispersion)
    )
    if two_leg and not use_composite:
        raise ValueError(
            "two-leg kinematics (v_sys_1_km / t_leg1_myr) require the composite "
            "integrator; pass integrator='composite' or enable the halo/dispersion."
        )
    if use_composite:
        host, sigma_interp = _host_potential_and_sigma(
            M_gal, R_e, halo_z, with_halo, with_dispersion
        )
        if two_leg:
            r_3d = _vectorized_orbit_composite_two_leg(
                v_sys_1_km[valid_idx] * KM_CM,
                v_cm,
                t_leg1_myr[valid_idx] * MYR_S,
                t_s,
                host,
                rng,
                sigma_interp=sigma_interp,
            )
        else:
            r_3d = _vectorized_orbit_composite(v_cm, t_s, host, rng, sigma_interp=sigma_interp)
    else:
        r_3d = _vectorized_orbit_3d(v_cm, t_s, M_gal, a, rng, n_steps=n_steps)

    # One isotropic viewing angle per system, so each offset is a genuine
    # projection realization (Bloom+ 1999): mu ~ U(-1, 1), R_off = r sqrt(1-mu^2).
    # A per-system median over several draws would give the distribution of a
    # median rather than of a projected offset, which is not what the observed
    # single-sightline offsets are.
    mu = rng.uniform(-1, 1, size=len(valid_idx))
    offsets = r_3d * np.sqrt(np.maximum(0.0, 1.0 - mu * mu)) / KPC_CM

    return {"offsets_kpc": offsets, "indices": valid_idx, "weights_sub": w_valid}


def compute_offsets_population(
    v_sys_km,
    t_delay_myr,
    weights=None,
    M_gal=DEFAULT_M_GAL,
    R_e=DEFAULT_R_E,
    use_analytic=True,
    max_systems=50000,
    vectorized=True,
    rng=None,
    with_halo=False,
    with_dispersion=None,
    halo_z=None,
    integrator="auto",
    v_sys_1_km=None,
    t_leg1_myr=None,
):
    """Compute projected offsets for a population of merging binaries.

    Parameters
    ----------
    v_sys_km : array
        Systemic velocities [km/s].
    t_delay_myr : array
        Delay times [Myr].
    weights : array, optional
        STROOPWAFEL weights for subsetting.  If more than ``max_systems``
        have finite v_sys, a simple random (unbiased) subsample is drawn and
        the original weights are retained.
    M_gal : float
        Galaxy stellar mass [g].
    R_e : float
        Effective radius [cm].
    use_analytic : bool
        Legacy flag; used only when ``vectorized=False``.  If True, use
        the per-system analytic shortcut (``_analytic_offset``);
        otherwise call ``integrate_orbit`` per system.
    max_systems : int
        Maximum number of systems to integrate (subsampled if exceeded).
    vectorized : bool
        When True (default), dispatch to
        ``compute_offsets_population_vectorized`` and integrate the
        whole population in one batched pass.  Set False to retain the legacy
        scalar code path; tests in ``tests/unit/test_phase4_helpers.py``
        exercise both.
    rng : np.random.Generator, optional
    with_halo, with_dispersion, halo_z, integrator :
        Composite-potential controls forwarded to
        ``compute_offsets_population_vectorized`` (only on the ``vectorized``
        path).  Default ``with_halo=False`` keeps the stellar-only Hernquist
        model; ``integrator`` selects the orbit scheme (see that function).
    v_sys_1_km, t_leg1_myr : array-like or None, optional
        Two-leg kinematic inputs (post-SN1 velocity and SN1-to-SN2 baseline)
        forwarded to the vectorized composite path; unsupported on the legacy
        scalar path.  See ``compute_offsets_population_vectorized``.

    Returns
    -------
    dict with keys:
        'offsets_kpc'  : array of projected offsets [kpc]
        'indices'      : indices into the input arrays
        'weights_sub'  : corresponding weights
    """
    if vectorized:
        return compute_offsets_population_vectorized(
            v_sys_km,
            t_delay_myr,
            weights=weights,
            M_gal=M_gal,
            R_e=R_e,
            max_systems=max_systems,
            rng=rng,
            with_halo=with_halo,
            with_dispersion=with_dispersion,
            halo_z=halo_z,
            integrator=integrator,
            v_sys_1_km=v_sys_1_km,
            t_leg1_myr=t_leg1_myr,
        )

    if v_sys_1_km is not None or t_leg1_myr is not None:
        raise ValueError("two-leg kinematics require vectorized=True (the composite path).")

    if rng is None:
        rng = np.random.default_rng(42)

    v_sys_km = np.asarray(v_sys_km, dtype=float)
    t_delay_myr = np.asarray(t_delay_myr, dtype=float)

    valid = np.isfinite(v_sys_km) & np.isfinite(t_delay_myr) & (v_sys_km > 0) & (t_delay_myr > 0)
    valid_idx = np.where(valid)[0]

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        w_valid = weights[valid_idx]
    else:
        w_valid = np.ones(len(valid_idx))

    # Simple random subsample without replacement (unbiased); see the
    # vectorized path for the rationale.
    if len(valid_idx) > max_systems:
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    a = hernquist_scale_radius(R_e)
    offsets = np.zeros(len(valid_idx))

    for i, idx in enumerate(valid_idx):
        v = v_sys_km[idx]
        t = t_delay_myr[idx]

        if use_analytic:
            r_3d = _analytic_offset(v, t, M_gal, a, rng=rng)
        else:
            r_3d = integrate_orbit(v * KM_CM, t * MYR_S, M_gal, a, rng=rng)

        # One isotropic viewing angle per system (Bloom+ 1999).
        mu = rng.uniform(-1, 1)
        offsets[i] = r_3d * np.sqrt(max(0.0, 1.0 - mu * mu)) / KPC_CM

    return {
        "offsets_kpc": offsets,
        "indices": valid_idx,
        "weights_sub": w_valid,
    }


def compute_offsets_mixed_hosts(
    v_sys_km,
    t_delay_myr,
    weights=None,
    host_models=None,
    max_systems=50000,
    rng=None,
    with_halo=True,
    with_dispersion=None,
    halo_z=None,
    v_sys_1_km=None,
    t_leg1_myr=None,
    **kw,
):
    """Compute offsets for a population using a mixture of host galaxy types.

    A single shared subsample is drawn once and then each host potential
    is evaluated on the *same* set of binaries, avoiding inconsistent
    subsampling across galaxy types.

    The fiducial science path (``with_halo=True``) adds an NFW dark-matter
    halo to each host and integrates with symplectic velocity-Verlet; pass
    ``with_halo=False`` for the stellar-only Hernquist edge of the Section 8
    model band.

    Parameters
    ----------
    host_models : dict, optional
        Keys are host names, values are dicts with ``M_gal``, ``R_e``,
        and ``weight`` (mixture fraction, should sum to 1).
        Defaults to ``HOST_MODELS``.
    max_systems : int
        Maximum population size.  The shared subsample is a simple random
        (unbiased) draw without replacement; the original weights are kept.
    rng : np.random.Generator, optional
    with_halo, with_dispersion, halo_z :
        Composite-potential controls forwarded to
        ``compute_offsets_population``.
    v_sys_1_km, t_leg1_myr : array-like or None, optional
        Two-leg kinematic inputs (post-SN1 velocity [km/s] and SN1-to-SN2
        baseline [Myr]).  When both are given the offsets are computed with the
        two-leg composite integrator: ``v_sys_km`` / ``t_delay_myr`` are then
        the post-SN2 velocity and inspiral time.  Subsampled with the same
        shared indices as the other columns.
    **kw :
        Extra keywords forwarded to ``compute_offsets_population``; in
        particular ``integrator={'auto','composite','legacy'}`` selects the
        orbit scheme (pass ``with_halo=False, with_dispersion=False,
        integrator='composite'`` for the stellar-only Verlet band edge).

    Returns
    -------
    dict with keys:
        'per_host' : {name: compute_offsets_population result}
        'mixed_offsets' : combined offset array
        'mixed_weights' : combined weight array (including host mixture)
    """
    if host_models is None:
        host_models = HOST_MODELS
    if rng is None:
        rng = np.random.default_rng(42)

    v = np.asarray(v_sys_km, dtype=float)
    t = np.asarray(t_delay_myr, dtype=float)
    valid = np.isfinite(v) & np.isfinite(t) & (v > 0) & (t > 0)

    two_leg = v_sys_1_km is not None and t_leg1_myr is not None
    if two_leg:
        v1 = np.asarray(v_sys_1_km, dtype=float)
        tl1 = np.asarray(t_leg1_myr, dtype=float)
        valid &= np.isfinite(v1) & np.isfinite(tl1) & (v1 >= 0) & (tl1 >= 0)

    valid_idx = np.where(valid)[0]
    if weights is not None:
        w_valid = np.asarray(weights, dtype=float)[valid_idx]
    else:
        w_valid = np.ones(len(valid_idx))

    # Shared simple random subsample (unbiased); the same indices are applied
    # to every column so the two-leg arrays stay aligned with v_sys / delay.
    if len(valid_idx) > max_systems:
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    leg_kw = {}
    if two_leg:
        leg_kw = {"v_sys_1_km": v1[valid_idx], "t_leg1_myr": tl1[valid_idx]}

    per_host = {}
    all_off, all_w = [], []
    for name, hp in host_models.items():
        res = compute_offsets_population(
            v[valid_idx],
            t[valid_idx],
            weights=w_valid,
            M_gal=hp["M_gal"],
            R_e=hp["R_e"],
            max_systems=len(valid_idx),
            rng=rng,
            with_halo=with_halo,
            with_dispersion=with_dispersion,
            halo_z=halo_z,
            **leg_kw,
            **kw,
        )
        per_host[name] = res
        all_off.append(res["offsets_kpc"])
        all_w.append(res["weights_sub"] * hp["weight"])

    return {
        "per_host": per_host,
        "mixed_offsets": np.concatenate(all_off),
        "mixed_weights": np.concatenate(all_w),
    }


def assign_host_by_delay(t_delay_myr, t_sf_max=SF_HOST_TRANSITION_T_MYR, rng=None):
    """Assign each binary to a host galaxy type based on delay time.

    Short delay (< *t_sf_max* Myr) systems are placed in star-forming
    hosts (split between ``SF_disk`` and ``SF_massive`` via
    ``SF_DISK_FRAC_OF_SF_HOSTS``).  Long delay systems go to elliptical
    hosts.  This connects GRB class properties (delay time tracks
    M_tot) to host type, following the observation from Fong & Berger
    (2013) that ~25% of sGRBs are in elliptical hosts with older
    stellar populations.

    Parameters
    ----------
    t_sf_max : float, optional
        Delay-time threshold [Myr].  Defaults to
        ``SF_HOST_TRANSITION_T_MYR``.
    rng : np.random.Generator, optional
        Random generator for the SF_disk / SF_massive split.  If None,
        defaults to ``np.random.default_rng(42)`` for backward
        compatibility.  Pass an external ``rng`` to keep this function
        consistent with a global reproducible stream.

    Returns
    -------
    host_assignment : array of str
        Host model key per system ('SF_disk', 'SF_massive', 'Elliptical').
    """
    t = np.asarray(t_delay_myr)
    out = np.full(len(t), "Elliptical", dtype="<U12")
    sf_mask = t < t_sf_max
    n_sf = sf_mask.sum()
    if rng is None:
        rng = np.random.default_rng(42)
    disk_frac = rng.random(n_sf) < SF_DISK_FRAC_OF_SF_HOSTS
    out[sf_mask] = np.where(disk_frac, "SF_disk", "SF_massive")
    return out


def compute_offsets_delay_hosts(
    v_sys_km,
    t_delay_myr,
    weights=None,
    t_sf_max=SF_HOST_TRANSITION_T_MYR,
    host_models=None,
    **kw,
):
    """Compute offsets with delay-time-dependent host assignment.

    Each system is assigned a host type via ``assign_host_by_delay``,
    then its offset is computed in the corresponding galaxy potential.

    Vectorization: each unique host appears
    exactly once in ``host_models``, so we gather the indices of systems
    routed to each host and dispatch one batched
    ``_vectorized_orbit_3d`` call per host, then scatter back into the
    flat output array.  This turns the prior per-system Python loop into
    at most ``len(host_models)`` numpy ops (3 for the default mixture).

    Returns
    -------
    dict with 'offsets_kpc', 'weights_sub', 'host_assignments'.
    """
    if host_models is None:
        host_models = HOST_MODELS
    rng = kw.pop("rng", None)
    if rng is None:
        rng = np.random.default_rng(42)
    n_steps = kw.pop("n_steps", 400)
    with_halo = kw.pop("with_halo", False)
    with_dispersion = kw.pop("with_dispersion", None)
    halo_z = kw.pop("halo_z", None)
    integrator = kw.pop("integrator", "auto")
    if integrator not in ("auto", "composite", "legacy"):
        raise ValueError(f"integrator must be 'auto', 'composite', or 'legacy'; got {integrator!r}")
    if with_dispersion is None:
        with_dispersion = with_halo
    use_composite = integrator == "composite" or (
        integrator == "auto" and (with_halo or with_dispersion)
    )

    v = np.asarray(v_sys_km, dtype=float)
    t = np.asarray(t_delay_myr, dtype=float)
    hosts = assign_host_by_delay(t, t_sf_max=t_sf_max, rng=rng)

    valid = np.isfinite(v) & np.isfinite(t) & (v > 0) & (t > 0)
    valid_idx = np.where(valid)[0]
    if weights is not None:
        w_valid = np.asarray(weights, dtype=float)[valid_idx]
    else:
        w_valid = np.ones(len(valid_idx))

    max_systems = kw.pop("max_systems", 50000)
    # Simple random subsample without replacement (unbiased); the original
    # weights are retained.  See compute_offsets_population_vectorized.
    if len(valid_idx) > max_systems:
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    offsets = np.zeros(len(valid_idx))
    if len(valid_idx) > 0:
        host_sub = hosts[valid_idx]
        v_cm_all = v[valid_idx] * KM_CM
        t_s_all = t[valid_idx] * MYR_S
        for name, hp in host_models.items():
            host_mask = host_sub == name
            if not host_mask.any():
                continue
            a_host = hernquist_scale_radius(hp["R_e"])
            if use_composite:
                host, sigma_interp = _host_potential_and_sigma(
                    hp["M_gal"], hp["R_e"], halo_z, with_halo, with_dispersion
                )
                r_3d = _vectorized_orbit_composite(
                    v_cm_all[host_mask], t_s_all[host_mask], host, rng, sigma_interp=sigma_interp
                )
            else:
                r_3d = _vectorized_orbit_3d(
                    v_cm_all[host_mask],
                    t_s_all[host_mask],
                    hp["M_gal"],
                    a_host,
                    rng,
                    n_steps=n_steps,
                )
            # One isotropic viewing angle per system (Bloom+ 1999).
            mu = rng.uniform(-1, 1, size=host_mask.sum())
            offsets[host_mask] = r_3d * np.sqrt(np.maximum(0.0, 1.0 - mu * mu)) / KPC_CM

    return {
        "offsets_kpc": offsets,
        "weights_sub": w_valid,
        "host_assignments": hosts[valid_idx],
    }


# ═══════════════════════════════════════════════════════════════════════════
# CDF helper
# ═══════════════════════════════════════════════════════════════════════════
def weighted_offset_cdf(offsets, weights):
    """Build a weighted CDF from offset and weight arrays.

    Returns
    -------
    sorted_offsets : array
        Sorted offset values.
    cdf : array
        Cumulative probability (0 to 1).
    """
    ok = np.isfinite(offsets) & (offsets > 0) & np.isfinite(weights)
    if ok.sum() < 2:
        return np.array([0.0]), np.array([0.0])

    o = offsets[ok]
    w = weights[ok]
    order = np.argsort(o)
    o_sorted = o[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    cdf /= cdf[-1]
    return o_sorted, cdf


# Number of Rice realizations and the seed used for them.  500 is the Fong
# et al. (2022) Sec. 6.2 choice; the seed is fixed so a figure regenerates
# identically (42 and 43 are already taken by the Alsing remap in Section 2b).
RICE_N_SAMPLES = 500
RICE_SEED = 44


def rice_resample_offsets(offsets, errors, n_samples=RICE_N_SAMPLES, rng=None):
    """Resample projected offsets from their Rice measurement distributions.

    A projected offset is the length of a 2D vector whose two components each
    carry Gaussian astrometric error, so its measurement distribution is Rice
    rather than Gaussian: positive definite, and pushed away from zero when the
    uncertainty is comparable to the offset.  Fong et al. (2022) Sec. 6.2 draw
    500 realizations this way (their Eq. 2 reference is Blanchard et al. 2016)
    and report a sampled median of 9.82 kpc against an observed 7.92 kpc, the
    shift being driven by the 15 bursts whose uncertainty exceeds their offset.

    Sampling uses the defining construction, ``hypot`` of two normals with means
    ``(offset, 0)`` and common width ``error``, rather than ``scipy.stats.rice``,
    so no per-burst shape parameter has to be formed and ``error = 0`` degrades
    gracefully to the measured value.

    Returns
    -------
    array, shape (n_samples, len(offsets))
        One row per realization.
    """
    offsets = np.asarray(offsets, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if offsets.shape != errors.shape:
        raise ValueError(
            f"offsets and errors must have the same shape, got {offsets.shape} and {errors.shape}"
        )
    if np.any(errors < 0):
        raise ValueError("offset uncertainties must be non-negative")

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(RICE_SEED if rng is None else rng)

    shape = (int(n_samples), offsets.size)
    along = offsets + errors * rng.standard_normal(shape)
    across = errors * rng.standard_normal(shape)
    return np.hypot(along, across)


def weighted_ks_2samp(data1, data2, weights1=None, weights2=None):
    """Two-sample Kolmogorov-Smirnov test for weighted samples.

    The STROOPWAFEL model offsets carry importance weights spanning orders
    of magnitude, so an unweighted ``scipy.stats.ks_2samp`` against the
    observed sample compares the wrong empirical distribution.  This builds
    the weighted empirical CDF of each sample, takes the supremum distance
    ``D``, and converts it to an asymptotic p-value with the Kish (1965)
    effective sample sizes standing in for the raw counts in the usual
    Kolmogorov scaling (Press et al. 2007, Numerical Recipes, Sec. 14.3,
    Eq. 14.3.18, with the ``en + 0.12 + 0.11/en`` finite-size correction).

    Parameters
    ----------
    data1, data2 : 1-D array
        Samples to compare.  Non-finite entries are dropped.
    weights1, weights2 : 1-D array, optional
        Per-sample weights (default uniform).  ``data2`` is typically the
        observed sample, left unweighted.

    Returns
    -------
    (D, p) : tuple of float
        KS statistic and asymptotic p-value, both NaN if either sample has
        fewer than two valid points or non-positive total weight.
    """
    from scipy.special import kolmogorov

    x1 = np.asarray(data1, dtype=float)
    x2 = np.asarray(data2, dtype=float)
    w1 = np.ones_like(x1) if weights1 is None else np.asarray(weights1, dtype=float)
    w2 = np.ones_like(x2) if weights2 is None else np.asarray(weights2, dtype=float)

    ok1 = np.isfinite(x1) & np.isfinite(w1) & (w1 >= 0)
    ok2 = np.isfinite(x2) & np.isfinite(w2) & (w2 >= 0)
    x1, w1 = x1[ok1], w1[ok1]
    x2, w2 = x2[ok2], w2[ok2]
    if x1.size < 2 or x2.size < 2 or w1.sum() <= 0 or w2.sum() <= 0:
        return float("nan"), float("nan")

    o1 = np.argsort(x1)
    o2 = np.argsort(x2)
    x1, w1 = x1[o1], w1[o1]
    x2, w2 = x2[o2], w2[o2]
    cw1 = np.cumsum(w1) / w1.sum()
    cw2 = np.cumsum(w2) / w2.sum()

    xs = np.concatenate([x1, x2])
    i1 = np.searchsorted(x1, xs, side="right")
    i2 = np.searchsorted(x2, xs, side="right")
    cdf1 = np.where(i1 > 0, cw1[np.clip(i1 - 1, 0, None)], 0.0)
    cdf2 = np.where(i2 > 0, cw2[np.clip(i2 - 1, 0, None)], 0.0)
    D = float(np.max(np.abs(cdf1 - cdf2)))

    def _ess(w):
        s2 = float(np.sum(w * w))
        return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0

    ne1, ne2 = _ess(w1), _ess(w2)
    if ne1 <= 0 or ne2 <= 0:
        return D, float("nan")
    en = np.sqrt(ne1 * ne2 / (ne1 + ne2))
    p = float(kolmogorov((en + 0.12 + 0.11 / en) * D))
    return D, min(1.0, max(0.0, p))


# Scholz & Stephens (1987) JASA 82, 918, Table 1: the standardised k-sample
# Anderson-Darling statistic T has critical values b0 + b1/sqrt(m) + b2/m at
# these significance levels, with m = k - 1 degrees of freedom.
_AD_SIG_LEVELS = np.array([0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001])
_AD_B0 = np.array([0.675, 1.281, 1.645, 1.960, 2.326, 2.573, 3.085])
_AD_B1 = np.array([-0.245, 0.250, 0.678, 1.149, 1.822, 2.364, 3.615])
_AD_B2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])


def weighted_anderson_2samp(data1, data2, weights1=None, weights2=None):
    """Two-sample Anderson-Darling test for weighted samples.

    Anderson-Darling is preferable to Kolmogorov-Smirnov for host-offset
    distributions because it is sensitive in the tails rather than near the
    median, which is where the large-offset events sit (Skobe et al. 2026,
    arXiv:2604.06586, Sec. 4.1, who use it for exactly this comparison).
    ``scipy.stats.anderson_ksamp`` cannot weight its inputs, so this
    reimplements the Scholz & Stephens (1987) JASA 82, 918 statistic with
    weight sums replacing counts.

    The statistic is their Eq. (3) with the multiplicity ``l_j`` read as the
    pooled weight at each distinct value, ``B_j`` as the running pooled
    weight, and ``M_ij`` as the running weight of sample ``i``; it reduces
    exactly to ``anderson_ksamp(..., midrank=False)`` when all weights are
    unity.  Standardisation uses their Eq. (4) variance with Kish (1965)
    effective sample sizes in place of the raw ``n_i``, matching the
    convention already used by :func:`weighted_ks_2samp`.  Because the
    variance terms are harmonic sums over integer indices, the pooled
    effective size is rounded to the nearest integer, which is negligible
    above a handful of effective points.

    Returns
    -------
    (A2, p) : tuple of float
        Standardised AD statistic and its p-value.  Larger ``A2`` means the
        samples are less consistent.  The Scholz & Stephens table only spans
        ``p`` in [0.001, 0.25], so the p-value saturates at those bounds and
        should be quoted as an inequality there.  Both are NaN when either
        sample has fewer than two valid points, when a total weight is
        non-positive, or when the pooled effective size is 4 or fewer (the
        variance denominator ``(N-1)(N-2)(N-3)`` degenerates).
    """
    x1 = np.asarray(data1, dtype=float)
    x2 = np.asarray(data2, dtype=float)
    w1 = np.ones_like(x1) if weights1 is None else np.asarray(weights1, dtype=float)
    w2 = np.ones_like(x2) if weights2 is None else np.asarray(weights2, dtype=float)

    ok1 = np.isfinite(x1) & np.isfinite(w1) & (w1 >= 0)
    ok2 = np.isfinite(x2) & np.isfinite(w2) & (w2 >= 0)
    x1, w1 = x1[ok1], w1[ok1]
    x2, w2 = x2[ok2], w2[ok2]
    if x1.size < 2 or x2.size < 2 or w1.sum() <= 0 or w2.sum() <= 0:
        return float("nan"), float("nan")

    def _ess(w):
        s2 = float(np.sum(w * w))
        return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0

    ne1, ne2 = _ess(w1), _ess(w2)
    n_pooled = int(round(ne1 + ne2))
    if ne1 <= 0 or ne2 <= 0 or n_pooled <= 4:
        return float("nan"), float("nan")

    # A2 grows with the number of pooled observations and with the ratio of
    # the two sample sizes, so each sample's weights are put in units of its
    # own effective observations.  Rescaling both by one common factor would
    # leave the statistic sensitive to an arbitrary per-sample weight
    # normalisation.  Uniform weights make this a no-op, since the Kish size
    # is then the raw count.
    w1 = w1 * (ne1 / float(w1.sum()))
    w2 = w2 * (ne2 / float(w2.sum()))

    W1, W2 = float(w1.sum()), float(w2.sum())
    W = W1 + W2

    z_star = np.unique(np.concatenate([x1, x2]))
    if z_star.size < 2:
        return float("nan"), float("nan")

    def _weight_at(x, w):
        """Total weight of ``x`` at each pooled distinct value."""
        idx = np.searchsorted(z_star, x)
        return np.bincount(idx, weights=w, minlength=z_star.size)

    at1, at2 = _weight_at(x1, w1), _weight_at(x2, w2)

    # The largest pooled value is dropped: there B_j saturates at W and the
    # (W - B_j) denominator vanishes.
    l_j = (at1 + at2)[:-1]
    B_j = np.cumsum(at1 + at2)[:-1]
    M1 = np.cumsum(at1)[:-1]
    M2 = np.cumsum(at2)[:-1]

    A2 = 0.0
    for M_i, W_i in ((M1, W1), (M2, W2)):
        inner = l_j / W * (W * M_i - B_j * W_i) ** 2 / (B_j * (W - B_j))
        A2 += float(inner.sum()) / W_i

    k = 2
    H = 1.0 / ne1 + 1.0 / ne2
    hs_cs = np.cumsum(1.0 / np.arange(n_pooled - 1, 1, -1))
    h = float(hs_cs[-1]) + 1.0
    g = float((hs_cs / np.arange(2, n_pooled)).sum())

    a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
    b = (2 * g - 4) * k**2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
    c = (6 * h + 2 * g - 2) * k**2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
    d = (2 * h + 6) * k**2 - 4 * h * k
    N = float(n_pooled)
    sigmasq = (a * N**3 + b * N**2 + c * N + d) / ((N - 1.0) * (N - 2.0) * (N - 3.0))
    if not np.isfinite(sigmasq) or sigmasq <= 0:
        return float("nan"), float("nan")

    m = k - 1
    T_kN = (A2 - m) / np.sqrt(sigmasq)

    critical = _AD_B0 + _AD_B1 / np.sqrt(m) + _AD_B2 / m
    if T_kN < critical.min():
        return float(T_kN), float(_AD_SIG_LEVELS.max())
    if T_kN > critical.max():
        return float(T_kN), float(_AD_SIG_LEVELS.min())
    coeffs = np.polyfit(critical, np.log(_AD_SIG_LEVELS), 2)
    return float(T_kN), float(np.exp(np.polyval(coeffs, T_kN)))


# ═══════════════════════════════════════════════════════════════════════════
# One-sample Anderson-Darling goodness-of-fit against a weighted model CDF
# ═══════════════════════════════════════════════════════════════════════════
# The STROOPWAFEL model sample outnumbers the observed offset samples by a
# factor of order 1000, so a model-vs-data comparison is a one-sample
# goodness-of-fit problem, not a two-sample one.  The one-sample Anderson
# -Darling statistic (Anderson & Darling 1954) against a reference CDF has a
# null distribution that depends only on the observed sample size N, so it can
# be calibrated exactly by Monte Carlo: under the null the probability-integral
# transform u = F_model(x) is uniform on (0, 1), and the null statistic is the
# same functional of sorted uniforms.  This avoids the effective-sample-size
# substitution into the Scholz & Stephens variance that the two-sample weighted
# test needs, and it returns p-values across the whole range rather than only
# the tabulated [0.001, 0.25].
_AD_GOF_EPS = 1e-12
_AD_GOF_NULL_SEED = 20260812
_AD_GOF_N_NULL_DEFAULT = 20000
_AD_GOF_NULL_CACHE: dict = {}


def _ad_gof_statistic_from_u(u_sorted):
    """One-sample AD statistic from PIT values ``u`` sorted ascending.

    ``u`` may be 1-D (one sample) or 2-D ``(n_samples, N)`` (each row a
    sample); the statistic is computed along the last axis.
    """
    u = np.asarray(u_sorted, dtype=float)
    n = u.shape[-1]
    u = np.clip(u, _AD_GOF_EPS, 1.0 - _AD_GOF_EPS)
    i = np.arange(1, n + 1)
    terms = (2 * i - 1) * (np.log(u) + np.log(1.0 - u[..., ::-1]))
    return -n - terms.sum(axis=-1) / n


def _ad_gof_null(n, n_null=_AD_GOF_N_NULL_DEFAULT):
    """Sorted Monte Carlo null distribution of the one-sample AD statistic.

    Distribution-free in ``n``, so cached per ``(n, n_null)`` and reused across
    every model and every Rice realization at that sample size.
    """
    key = (int(n), int(n_null))
    cached = _AD_GOF_NULL_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(_AD_GOF_NULL_SEED)
    u = np.sort(rng.random((int(n_null), int(n))), axis=1)
    null = np.sort(_ad_gof_statistic_from_u(u))
    _AD_GOF_NULL_CACHE[key] = null
    return null


def ad_gof_weighted(model_offsets, obs, weights=None, n_null=_AD_GOF_N_NULL_DEFAULT):
    """Monte-Carlo-calibrated one-sample AD goodness-of-fit p-value.

    Tests whether the observed offsets ``obs`` are consistent with being drawn
    from the weighted model offset distribution ``model_offsets`` (weights
    ``weights``).  The model CDF is the weighted empirical CDF; ``obs`` is
    transformed through it and scored with the one-sample AD statistic, whose
    null is obtained by Monte Carlo from sorted uniforms of the same size.

    Returns
    -------
    (A2, p) : tuple of float
        AD statistic and p-value ``p = (1 + #{null >= A2}) / (1 + n_null)``.
        Larger ``A2`` (smaller ``p``) means the model is less consistent with
        the data.  Both NaN if fewer than two finite observed points, or if the
        model has no positive-weight finite support.
    """
    x_model = np.asarray(model_offsets, dtype=float)
    w_model = np.ones_like(x_model) if weights is None else np.asarray(weights, dtype=float)
    ok_m = np.isfinite(x_model) & np.isfinite(w_model) & (w_model >= 0) & (x_model > 0)
    x_model, w_model = x_model[ok_m], w_model[ok_m]

    obs = np.asarray(obs, dtype=float)
    obs = np.sort(obs[np.isfinite(obs) & (obs > 0)])
    n = obs.size
    if n < 2 or x_model.size < 2 or w_model.sum() <= 0:
        return float("nan"), float("nan")

    order = np.argsort(x_model)
    xs = x_model[order]
    cw = np.cumsum(w_model[order])
    cw /= cw[-1]
    idx = np.searchsorted(xs, obs, side="right")
    u = np.where(idx > 0, cw[np.clip(idx - 1, 0, None)], 0.0)

    A2 = float(_ad_gof_statistic_from_u(u))
    null = _ad_gof_null(n, n_null)
    p = (1 + int(np.count_nonzero(null >= A2))) / (1 + null.size)
    return A2, float(p)


def offset_cdf_by_class(offsets, weights, class_masks):
    """Per-class weighted offset CDFs.

    ``class_masks`` is the ``classify_bns_2024`` / ``classify_bhns``
    output dict with non-mask keys (like ``'M_disk'``) stripped.
    Classes with fewer than two valid systems return the sentinel
    ``(np.array([0.0]), np.array([0.0]))``.
    """
    out = {}
    for label, mask in class_masks.items():
        m = np.asarray(mask, dtype=bool)
        out[label] = weighted_offset_cdf(offsets[m], weights[m])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Angular to physical offset conversion
# ═══════════════════════════════════════════════════════════════════════════
# Published offset catalogues quote kpc under whatever cosmology their authors
# adopted, which is generally not the Planck 2015 pin this project shares with
# COMPAS FastCosmicIntegration.  Converting from the angular measurement under
# our own cosmology keeps the observed samples on the same distance scale as
# the model, at the cost of no longer matching the source table verbatim; the
# published columns are kept alongside as the literature anchors.
#
# Flat LambdaCDM, so the comoving distance is the elementary integral
# D_C = (c / H0) * int_0^z dz' / E(z') with E(z) = sqrt(Om (1+z)^3 + OL), and
# the proper transverse scale is D_A = D_C / (1 + z).  Written out rather than
# pulled from astropy.cosmology so the cosmology cannot silently drift to a
# Planck 2018 instance (see the grb_physics module docstring).

# WMAP9 (Bennett et al. 2014, ApJS 208, 20).  Both observed offset catalogs
# used below adopt it: Fong et al. (2022) Sec. 1 and, following them, Skobe
# et al. (2026) Sec. 4.1.
WMAP9_H0 = 69.6  # km/s/Mpc
WMAP9_OMEGA_M = 0.286


def kpc_per_arcsec(z, H0=H0_PLANCK15, omega_m=OMEGA_M_PLANCK15):
    """Proper transverse scale at redshift ``z`` for a flat LambdaCDM model.

    Defaults to the project cosmology.  Accepts a scalar or an array of
    redshifts and returns kpc per arcsecond.
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    omega_l = 1.0 - omega_m
    hubble_distance_mpc = C_KMS / H0

    def _inv_E(zp):
        return 1.0 / np.sqrt(omega_m * (1.0 + zp) ** 3 + omega_l)

    d_c = np.array([quad(_inv_E, 0.0, float(zi))[0] for zi in z_arr])
    d_a_mpc = hubble_distance_mpc * d_c / (1.0 + z_arr)
    scale = 1e3 * d_a_mpc / ARCSEC_PER_RAD
    return scale if np.ndim(z) else float(scale[0])


def _cosmology_correction(z):
    """Ratio of transverse scales, project cosmology over the catalogs' WMAP9.

    Applied multiplicatively to the published kpc offsets rather than
    reconverting from an angular column, because the published values do not
    all follow from a Hubble-law distance.  Seven of the nine kilonova events
    reproduce to better than 0.2 percent as ``theta * kpc_per_arcsec(z)``, but
    the two lowest-redshift ones do not (170817A 2.086 against 2.125 published,
    230307A 37.95 against 38.90): at z of 0.01 to 0.07 the host distance
    carries direct measurements and peculiar-velocity corrections that a
    redshift-only inference discards.  Rescaling by the ratio keeps that
    information and changes only the cosmological parameters.
    """
    return kpc_per_arcsec(z) / kpc_per_arcsec(z, H0=WMAP9_H0, omega_m=WMAP9_OMEGA_M)


# ═══════════════════════════════════════════════════════════════════════════
# Short-GRB host offsets (historical anchor)
# ═══════════════════════════════════════════════════════════════════════════
# Retained as a pinned literature anchor and as the source of the HOST_MODELS
# host-type mixture, not as a comparison sample: the classical-sGRB comparison
# moved to the Fong et al. (2022) BRIGHT catalog below, which supersedes this
# one and is what Skobe et al. (2026) compare against.  Do not edit the values
# or drop entries; tests/anchors pins them to the published summary.
#
# Per-burst projected physical offsets [kpc] for the 22-event sub-arcsecond
# sample assembled by Fong & Berger (2013), ApJ 776, 18, Figure 5.  The
# paper combines its own HST sample (Table 2) with six prior measurements
# from Fong, Berger & Fox (2010), ApJ 708, 9, Table 3, and three ground-
# based offsets quoted in the text (Section 3.3, p. 9).  Ambiguous-host
# events (GRBs 061201, 091109B) and XRT-only events (GRBs 050509B, 051210,
# 070429B) are excluded from this sample by the paper, matching its
# "sub-arcsecond positions" selection.
#
# Per-burst breakdown (kpc; sources keyed alongside each value):
#
#   Fong & Berger (2013) Table 2 [13 events]:
#     070707     3.22   (z = 1 assumed; F&B 2013)
#     070714B   12.21   (F160W)
#     070724A    5.46
#     070809    33.22   (host-less; most-probable-host; F160W)
#     071227    15.50   (F160W)
#     080503     7.24   (host-less; z = 1 assumed)
#     080905A   17.96   (G1 association per Rowlinson+ 2010; F160W)
#     090305A    3.46   (z = 1 assumed)
#     090426     0.45
#     090510    10.37
#     090515    75.03   (host-less; most-probable-host)
#     100117A    1.32
#     130603B    5.21   (F160W)
#
#   Fong, Berger & Fox (2010) Table 3 [6 events with sub-arcsec positions]:
#     050709     3.71   (HST/ACS direct detection; F&B 2013 cites 3.7)
#     050724     2.69   (Magellan/PANIC; consistent with PANIC Ks 2.76)
#     051221A    1.92   (WFPC2/F555W)
#     060121     0.96   (z = 1 assumed)
#     060313     2.86   (z = 1 assumed)
#     061006     1.30   (ACS/F814W)
#
#   Ground-based supplements quoted in F&B 2013 Sec. 3.3 [3 events]:
#     111020A    6.0    (Fong et al. 2012, ApJ 756, 189)
#     111117A   10.5    (Margutti et al. 2012, ApJ 756, 63;
#                        Sakamoto et al. 2013, ApJ 766, 41)
#     120804A    2.2    (Berger et al. 2013, ApJ 765, 121)
#
# Summary statistics (Fong & Berger 2013, Sec. 3.3): N = 22, range
# ~ 0.5 - 75 kpc, median ~ 4.5 kpc, ~ 25 percent of events at >= 10 kpc.
# The empirical array below reproduces these: N = 22, median = 4.46 kpc,
# max = 75.03, 7/22 = 31.8 percent at >= 10 kpc.
OBSERVED_SGRB_OFFSETS_KPC = np.array(
    [
        0.45,  # 090426
        0.96,  # 060121
        1.30,  # 061006
        1.32,  # 100117A
        1.92,  # 051221A
        2.20,  # 120804A
        2.69,  # 050724
        2.86,  # 060313
        3.22,  # 070707
        3.46,  # 090305A
        3.71,  # 050709
        5.21,  # 130603B
        5.46,  # 070724A
        6.0,  # 111020A
        7.24,  # 080503
        10.37,  # 090510
        10.5,  # 111117A
        12.21,  # 070714B
        15.50,  # 071227
        17.96,  # 080905A
        33.22,  # 070809
        75.03,  # 090515
    ]
)

# Burst names in the same order as OBSERVED_SGRB_OFFSETS_KPC, so the
# kilonova-bearing members can be selected by name rather than by matching
# floats.
OBSERVED_SGRB_OFFSET_NAMES = (
    "090426",
    "060121",
    "061006",
    "100117A",
    "051221A",
    "120804A",
    "050724",
    "060313",
    "070707",
    "090305A",
    "050709",
    "130603B",
    "070724A",
    "111020A",
    "080503",
    "090510",
    "111117A",
    "070714B",
    "071227",
    "080905A",
    "070809",
    "090515",
)

# ═══════════════════════════════════════════════════════════════════════════
# Fong et al. (2022) BRIGHT short-GRB offset catalog
# ═══════════════════════════════════════════════════════════════════════════
# Fong et al. (2022) ApJ 940, 56 (arXiv:2206.01763) Table 4: projected
# galactocentric offsets for the 90-burst BRIGHT census of Swift short GRBs.
# Section 8 compares against this rather than the Fong & Berger (2013) sample
# above, which it supersedes: 83 bursts against 22, XRT-localized as well as
# sub-arcsecond, and a median offset 2 to 3 kpc larger (Fong et al. 2022
# Sec. 6.2).  It is also the astrometric source for the kilonova-selected
# offsets further down (Skobe et al. 2026 Sec. 4.1 take their angular and
# physical columns from this table), so both panels of the Section 8 figure
# share one astrometric system and one cosmology by construction.
#
# Note that this is *not* the sample Skobe et al. compare against: their own
# classical-sGRB curve is Palmese et al. (2017) and their long-GRB curve is
# Lyman et al. (2017).  BRIGHT is the larger and more homogeneous catalog and
# it postdates both, which is why it is used here.
#
# The one Table 4 entry omitted is GRB 081226A, an upper limit (< 4.06 kpc)
# rather than a measurement; an Anderson-Darling test has no natural way to
# carry a censored point.  Redshifts are the Table 4 values; eleven bursts have
# no redshift and Fong et al. compute their physical offsets at an assumed
# z = 1, which is reproduced here so the cosmology rescaling below is defined
# for every burst.  (Their Sec. 6.2 text says 13 such bursts, but Table 4 lists
# 12 blank redshifts, one of which is the censored 081226A.)

FONG22_SGRB_NAMES = (
    "050509B",
    "050709",
    "050724",
    "050813",
    "051210",
    "051221A",
    "060121",
    "060313",
    "060614",
    "060801",
    "061006",
    "061210",
    "070429B",
    "070707",
    "070714B",
    "070724",
    "070729",
    "070809",
    "071227",
    "080123",
    "080503",
    "080905",
    "090305",
    "090426",
    "090510",
    "090515",
    "091109",
    "100117A",
    "100206",
    "100625A",
    "101219",
    "101224",
    "111117",
    "120305",
    "120804",
    "121226A",
    "130515A",
    "130603B",
    "130716A",
    "130822A",
    "130912A",
    "131004A",
    "140129B",
    "140622A",
    "140903A",
    "140930B",
    "141212A",
    "150101B",
    "150120A",
    "150424A",
    "150728A",
    "150831A",
    "151229A",
    "160303A",
    "160408A",
    "160411A",
    "160525B",
    "160601A",
    "160624A",
    "160821B",
    "161001A",
    "161104A",
    "170127B",
    "170428A",
    "170728A",
    "170728B",
    "180418A",
    "180618A",
    "180727A",
    "180805B",
    "181123B",
    "191031D",
    "200219A",
    "200411A",
    "200522A",
    "200907B",
    "201221D",
    "210323A",
    "210726A",
    "210919A",
    "211023B",
    "211106A",
    "211211A",
)

# Bursts without a measured redshift; Fong et al. (2022) Table 4 note a
# assumes z = 1 for their physical offsets.  These 11 events are every
# entry in the code's 83-event offset sample whose Table 4 redshift column
# is "...".  Table 4 has 12 such entries in total; the 12th is 081226A,
# which carries only an offset upper limit and is deliberately excluded
# from `FONG22_SGRB_NAMES` (see the loader docstring and CLAUDE.md).  Note
# that Fong+ 2022 Sec. 6.2 text says "13 events with an assumed z = 1",
# which is an internal off-by-one against Table 4; the constant here
# tracks the authoritative Table 4 data, not the text summary.
FONG22_SGRB_ASSUMED_Z1 = (
    "060121",
    "060313",
    "070707",
    "080503",
    "090305",
    "091109",
    "130912A",
    "150424A",
    "160525B",
    "160601A",
    "211106A",
)

FONG22_SGRB_REDSHIFTS = np.array(
    [
        0.2248,
        0.1607,
        0.254,
        0.719,
        2.58,
        0.5464,
        1.0,
        1.0,
        0.125,
        1.131,
        0.461,
        0.4095,
        0.902,
        1.0,
        0.923,
        0.457,
        0.52,
        0.473,
        0.381,
        0.495,
        1.0,
        0.1218,
        1.0,
        2.609,
        0.903,
        0.403,
        1.0,
        0.914,
        0.407,
        0.452,
        0.718,
        0.454,
        2.211,
        0.225,
        0.74,
        1.37,
        0.8,
        0.3568,
        2.2,
        0.154,
        1.0,
        0.717,
        0.43,
        0.959,
        0.351,
        1.465,
        0.596,
        0.134,
        0.46,
        1.0,
        0.461,
        1.09,
        0.63,
        1.01,
        1.9,
        0.82,
        1.0,
        1.0,
        0.4842,
        0.1619,
        0.67,
        0.793,
        2.28,
        0.453,
        1.493,
        0.62,
        1.56,
        0.52,
        1.95,
        0.6612,
        1.754,
        1.93,
        0.48,
        1.93,
        0.5536,
        0.56,
        1.055,
        0.733,
        0.2244,
        0.2415,
        0.862,
        1.0,
        0.0763,
    ]
)

# Redshift at which the fiducial dark-matter halo is built for the Section 8
# offset model: the median measured redshift of the classical comparison
# sample, dropping the bursts Fong et al. 2022 place at an assumed z = 1.
_FONG22_MEASURED_Z = ~np.isin(np.asarray(FONG22_SGRB_NAMES), np.asarray(FONG22_SGRB_ASSUMED_Z1))
HALO_FIDUCIAL_Z = float(np.median(FONG22_SGRB_REDSHIFTS[_FONG22_MEASURED_Z]))

FONG22_SGRB_OFFSETS_KPC_WMAP9 = np.array(
    [
        55.19,
        3.76,
        2.74,
        43.57,
        29.08,
        2.08,
        0.97,
        2.60,
        0.70,
        10.25,
        1.39,
        15.51,
        6.00,
        3.25,
        12.33,
        5.52,
        19.72,
        34.11,
        14.74,
        53.63,
        7.31,
        18.30,
        3.49,
        0.49,
        10.51,
        76.19,
        4.22,
        1.35,
        25.28,
        2.63,
        5.48,
        12.75,
        10.52,
        18.09,
        2.30,
        2.31,
        61.22,
        5.40,
        33.08,
        60.09,
        3.90,
        0.80,
        1.76,
        32.95,
        0.90,
        9.62,
        18.75,
        7.36,
        4.77,
        3.41,
        7.52,
        12.21,
        8.16,
        15.31,
        14.13,
        1.40,
        8.61,
        1.38,
        9.63,
        15.74,
        18.54,
        1.66,
        10.37,
        7.72,
        32.25,
        6.76,
        1.30,
        9.70,
        2.56,
        24.30,
        5.08,
        13.08,
        8.30,
        41.98,
        0.93,
        2.41,
        29.35,
        5.89,
        12.26,
        51.05,
        3.84,
        0.79,
        7.92,
    ]
)

# Symmetric 1-sigma from Table 4.  Fifteen bursts carry an uncertainty larger
# than the offset itself, almost all of them XRT-localized; Fong et al. handle
# that with Rice-distribution sampling (their Sec. 6.2), which is why the
# sampled median (9.82 kpc) sits above the observed one (7.92 kpc).  Section 8
# compares against the observed offsets, so it inherits the narrower value.
FONG22_SGRB_OFFSET_ERR_KPC_WMAP9 = np.array(
    [
        12.43,
        0.056,
        0.080,
        17.37,
        16.34,
        0.19,
        0.37,
        0.55,
        0.79,
        10.92,
        0.29,
        14.36,
        13.44,
        0.24,
        0.87,
        0.18,
        14.49,
        2.75,
        0.26,
        7.67,
        0.24,
        0.18,
        0.24,
        0.24,
        2.92,
        0.16,
        0.41,
        0.32,
        13.05,
        6.77,
        6.65,
        13.51,
        1.68,
        5.25,
        1.28,
        9.15,
        13.77,
        0.20,
        14.23,
        4.90,
        1.06,
        0.22,
        1.76,
        11.25,
        0.10,
        4.30,
        12.29,
        0.072,
        6.44,
        0.32,
        20.29,
        9.77,
        6.05,
        0.90,
        1.25,
        2.30,
        8.61,
        4.06,
        6.24,
        0.03,
        6.22,
        16.60,
        13.60,
        3.39,
        3.01,
        2.06,
        0.32,
        1.69,
        5.13,
        7.49,
        1.38,
        10.69,
        5.28,
        7.48,
        0.19,
        7.78,
        24.09,
        3.68,
        2.22,
        1.92,
        2.57,
        0.29,
        0.029,
    ]
)

# Host-association robustness, Fong et al. (2022) Table 2 and Sec. 4.3.  Every
# burst with a field galaxy at P_cc,min <= 0.2 is assigned a host, graded Gold
# (P_cc,min <= 0.02), Silver (0.02 to 0.10) or Bronze (0.10 to 0.20).  The grade
# matters for the offset distribution because it is not a neutral quality flag:
# Gold is biased toward sub-arcsecond localizations, brighter hosts and smaller
# offsets, and the Bronze median sits at 32 kpc against 4.9 kpc for Gold.  Fong
# et al. expect <= 2 / 10 / 20 percent of Gold / Silver / Bronze associations to
# be spurious, about 5.8 bursts overall.
#
# Listing the two less-robust grades rather than all 83 keeps the published
# groupings readable; GRBs 150424A and 161104A are Bronze by Fong et al.'s
# judgement rather than by threshold (both have P_cc,min = 0.06, which is
# nominally Silver), so the grades cannot be re-derived from P_cc alone.
FONG22_SGRB_SILVER = (
    "051210",
    "070729",
    "080503",
    "090515",
    "100625A",
    "101219",
    "111117",
    "120305",
    "130515A",
    "130822A",
    "131004A",
    "140930B",
    "150831A",
    "151229A",
    "160303A",
    "160624A",
    "160821B",
    "161001A",
    "170127B",
    "180805B",
    "191031D",
    "210726A",
)

FONG22_SGRB_BRONZE = (
    "050813",
    "080123",
    "091109",
    "130716A",
    "130912A",
    "140622A",
    "150424A",
    "160408A",
    "161104A",
    "170728A",
    "200411A",
    "201221D",
    "210919A",
)

FONG22_SGRB_SAMPLE_GRADE = tuple(
    "Silver" if n in FONG22_SGRB_SILVER else "Bronze" if n in FONG22_SGRB_BRONZE else "Gold"
    for n in FONG22_SGRB_NAMES
)

# The BRIGHT portal (bright.ciera.northwestern.edu) is a live catalog and has
# moved past Table 4 for seven of the 83 bursts, in the offset, the redshift or
# both.  Everything above stays on the published values, which are what Sec. 6.2
# summarises, what the anchor test reproduces, and what later compilations still
# carry (Gaspari et al. 2025, arXiv:2504.04825, Table 1 lists 210726A at the
# published 12.26 +/- 2.22 kpc).  The portal readings are recorded here, checked
# against the site on 2026-08-13, so the disagreement is documented rather than
# looking like a transcription slip.
#
# Four of the seven are pure redshift updates: rescaling the published offset by
# the angular-scale ratio between the two redshifts reproduces the portal number
# to better than 3 percent, so nothing but z moved.  For 120804, 160525B and
# 210726A the rescaling fails, meaning the host or the astrometry changed, and
# the portal publishes neither an uncertainty nor a revised association grade to
# go with it.  210726A is the one to be careful with: Schroeder et al. (2024),
# ApJ 970, 139 confirm the host at P_cc = 5e-4 but derive a double-peaked
# photometric redshift favouring z ~ 2.38, so the portal's 0.37 is not a settled
# replacement for the 0.2244 used here.
#
# Adopting all six revised offsets moves the 80-burst classical median from 8.38
# to 8.09 kpc on the project cosmology; adopting only the cleanly reproduced ones
# moves it the other way, to 8.51 kpc.  Both sit far inside the host-association
# and astrometric bracket of classical_sgrb_offset_band (5.1 to 10.9 kpc), so no
# Section 8 conclusion turns on the choice.
BRIGHT_PORTAL_REVISIONS = {
    # burst: (projected offset in kpc, redshift), both as the portal reads them
    "120804": (2.22, 1.05),
    "150831A": (12.43, 1.18),
    "160525B": (5.50, 0.64),
    "170127B": (10.37, 2.21),
    "170728B": (8.40, 1.272),
    "200411A": (37.66, 0.82),
    "210726A": (0.23, 0.37),
}

# Skobe et al. (2026) Sec. 2 count short bursts with extended emission as
# merger-driven LGRB-KNe rather than as classical sGRBs.  Three of them sit in
# the BRIGHT short-GRB catalog, so they are dropped here: they appear in
# OBSERVED_LGRB_KN_OFFSETS_KPC below, and comparing a model against both arrays
# would double-count them.  This is a deliberate departure from Skobe et al.,
# whose own classical curve is the general sGRB population and therefore
# retains them; at 3 of 83 mid-distribution events it shifts nothing
# measurable, and it keeps our two panels disjoint in taxonomy.  The
# kilonova-selected *short* bursts (130603B, 150101B, 160821B, 200522A) stay,
# because KN selection is a cut on kilonova detection rather than a partition
# of the sGRB population.
SGRB_RECLASSIFIED_AS_LGRB_KN = ("050709", "060614", "211211A")

_FONG22_IS_CLASSICAL = np.array([n not in SGRB_RECLASSIFIED_AS_LGRB_KN for n in FONG22_SGRB_NAMES])

# Fong et al. (2022) Sec. 1 adopt the same WMAP9 parameters Skobe et al. do, so
# the comparison array is rescaled onto the project cosmology the same way.
# Unlike the kilonova-selected samples, reconverting is not an option even in
# principle: Table 4 quotes no angular offset for the bursts whose position
# comes from the enhanced XRT localization.
_FONG22_CORR = _cosmology_correction(FONG22_SGRB_REDSHIFTS)

OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC = (FONG22_SGRB_OFFSETS_KPC_WMAP9 * _FONG22_CORR)[
    _FONG22_IS_CLASSICAL
]
OBSERVED_SGRB_OFFSET_ERR_CLASSICAL_KPC = (FONG22_SGRB_OFFSET_ERR_KPC_WMAP9 * _FONG22_CORR)[
    _FONG22_IS_CLASSICAL
]
OBSERVED_SGRB_CLASSICAL_NAMES = tuple(
    n for n in FONG22_SGRB_NAMES if n not in SGRB_RECLASSIFIED_AS_LGRB_KN
)

# Gold-only version of the same comparison, the conservative edge of the
# Section 8 band.  Fong et al. (2022) Sec. 6.2 read their Gold median of
# 4.92 kpc as "a minimum on the short GRB median offset", so a model that
# already sits at large offsets is hardest to reconcile with this curve.
_FONG22_IS_GOLD = np.array([g == "Gold" for g in FONG22_SGRB_SAMPLE_GRADE])
_FONG22_GOLD_CLASSICAL = _FONG22_IS_GOLD & _FONG22_IS_CLASSICAL

OBSERVED_SGRB_OFFSETS_GOLD_KPC = (FONG22_SGRB_OFFSETS_KPC_WMAP9 * _FONG22_CORR)[
    _FONG22_GOLD_CLASSICAL
]
OBSERVED_SGRB_OFFSET_ERR_GOLD_KPC = (FONG22_SGRB_OFFSET_ERR_KPC_WMAP9 * _FONG22_CORR)[
    _FONG22_GOLD_CLASSICAL
]
OBSERVED_SGRB_GOLD_NAMES = tuple(
    n for n, keep in zip(FONG22_SGRB_NAMES, _FONG22_GOLD_CLASSICAL, strict=True) if keep
)

# ═══════════════════════════════════════════════════════════════════════════
# Kilonova-selected GRB host offsets
# ═══════════════════════════════════════════════════════════════════════════
# Skobe et al. (2026) arXiv:2604.06586 analyse the nine GRBs with robust
# kilonova associations at z < 0.6 and split them at a prompt duration of
# 2 s into five sGRB-KNe and four LGRB-KNe (their Sec. 2).  Angular offsets,
# redshifts and the published kpc columns below are their Table 2; the
# astrometry is Table 4 of Fong et al. (2022) ApJ 940, 56.
#
# These are the *physical* offsets, not the host-normalised R/r50 column.
# R/r50 is filter-dependent (r50 comes from the per-band GALFIT fit, so
# 160821B spans 1.77 to 2.41 across filters), it has no counterpart in the
# orbit-integrated model prediction, and Skobe et al. find that single-Sersic
# fits overestimate it when the transient sits on a host subcomponent.
#
# Using Fong+ 2022 astrometry throughout keeps the KN samples internally
# consistent, at the cost of small offsets from the discovery-paper values
# this module used previously (050709 3.71 -> 3.76, 060614 0.73 -> 0.70,
# 130603B 5.21 -> 5.40, 211211A 7.91 -> 7.92).  The classical-sGRB anchor
# above is deliberately left on its own Fong & Berger (2013) astrometry.
#
# Two events sit in both a KN array and the classical sGRB anchor (050709 and
# 130603B).  That is intended and matches Skobe et al., whose classical
# comparison samples also contain them: the KN arrays are a selection on
# kilonova detection, not a disjoint partition of the sGRB population.
#
# GRB 191019A (Levan et al. 2023, Nat. Astron. 7, 976) is a candidate
# dynamical / AGN-disk lGRB-merger with a near-nuclear offset of
# 78 +/- 109 pc but no confirmed kilonova counterpart, so it is excluded.

OBSERVED_SGRB_KN_NAMES = ("200522A", "170817A", "130603B", "150101B", "160821B")
OBSERVED_SGRB_KN_REDSHIFTS = np.array([0.554, 0.0098, 0.356, 0.1341, 0.1613])
OBSERVED_SGRB_KN_OFFSETS_ARCSEC = np.array([0.143, 10.317, 1.07, 3.07, 5.61])
OBSERVED_SGRB_KN_OFFSET_ERR_ARCSEC = np.array([0.029, 0.005, 0.04, 0.03, 0.01])

OBSERVED_LGRB_KN_NAMES = ("060614", "050709", "211211A", "230307A")
OBSERVED_LGRB_KN_REDSHIFTS = np.array([0.125, 0.161, 0.0762, 0.065])
OBSERVED_LGRB_KN_OFFSETS_ARCSEC = np.array([0.31, 1.35, 5.44, 30.20])
# The 060614 angular uncertainty exceeds its central value, so that event is
# consistent with a nuclear position.
OBSERVED_LGRB_KN_OFFSET_ERR_ARCSEC = np.array([0.35, 0.02, 0.02, 0.01])

# Skobe et al. (2026) Table 2 kpc columns, verbatim under their WMAP9
# conversion.  Literature anchors: the comparison arrays below put these on the
# project cosmology instead, which raises every offset by ~1 to 3 percent.
OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9 = np.array([0.93, 2.125, 5.40, 7.35, 15.74])
OBSERVED_SGRB_KN_OFFSET_ERR_KPC_WMAP9 = np.array([0.19, 0.001, 0.20, 0.072, 0.03])
OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9 = np.array([0.70, 3.76, 7.92, 38.90])
OBSERVED_LGRB_KN_OFFSET_ERR_KPC_WMAP9 = np.array([0.79, 0.056, 0.029, 0.01])


_SGRB_KN_CORR = _cosmology_correction(OBSERVED_SGRB_KN_REDSHIFTS)
_LGRB_KN_CORR = _cosmology_correction(OBSERVED_LGRB_KN_REDSHIFTS)

OBSERVED_SGRB_KN_OFFSETS_KPC = OBSERVED_SGRB_KN_OFFSETS_KPC_WMAP9 * _SGRB_KN_CORR
OBSERVED_SGRB_KN_OFFSET_ERR_KPC = OBSERVED_SGRB_KN_OFFSET_ERR_KPC_WMAP9 * _SGRB_KN_CORR
OBSERVED_LGRB_KN_OFFSETS_KPC = OBSERVED_LGRB_KN_OFFSETS_KPC_WMAP9 * _LGRB_KN_CORR
OBSERVED_LGRB_KN_OFFSET_ERR_KPC = OBSERVED_LGRB_KN_OFFSET_ERR_KPC_WMAP9 * _LGRB_KN_CORR


# ═══════════════════════════════════════════════════════════════════════════
# Classical-sGRB comparison band
# ═══════════════════════════════════════════════════════════════════════════


def classical_sgrb_offset_band(n_samples=RICE_N_SAMPLES, rng=None):
    """Bracket the classical-sGRB offset comparison by its two systematics.

    The observed BRIGHT offset distribution is not a single curve.  Two choices
    that Fong et al. (2022) make explicitly move its median by more than the
    sampling error does, and they move it in opposite directions:

    - restricting to the most robust host associations (Gold, ``P_cc,min <=
      0.02``) lowers the median from 7.9 to 4.9 kpc, because the tenuous
      associations are preferentially the widely separated ones;
    - folding in the astrometric uncertainties raises it to 9.8 kpc, because
      15 of the 83 bursts have an uncertainty exceeding their offset.

    Quoting a comparison against the middle curve alone hides both.  The band
    below is what Section 8 shades and what the reported p-value range spans.

    Returns
    -------
    dict
        ``gold`` and ``observed`` are 1D offset arrays; ``sampled`` is the
        ``(n_samples, N)`` Rice resampling of ``observed``.
    """
    return {
        "gold": OBSERVED_SGRB_OFFSETS_GOLD_KPC,
        "observed": OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC,
        "sampled": rice_resample_offsets(
            OBSERVED_SGRB_OFFSETS_CLASSICAL_KPC,
            OBSERVED_SGRB_OFFSET_ERR_CLASSICAL_KPC,
            n_samples=n_samples,
            rng=rng,
        ),
    }


def classical_sgrb_ad_pvalues(model_offsets, model_weights=None, band=None):
    """Anderson-Darling p-values of a model offset sample against the band.

    Uses the Monte-Carlo-calibrated one-sample AD goodness-of-fit test
    (``ad_gof_weighted``): the model is the reference distribution and each
    classical-sGRB array is the observed sample tested against it.  The Rice
    entry is one test per realization rather than one test against the pooled
    draws (pooling would inflate the observed sample to ``n_samples x N`` and
    manufacture significance); ``p_sampled`` is the median over realizations and
    ``p_sampled_lo`` / ``p_sampled_hi`` its 16th and 84th percentiles.
    """
    if band is None:
        band = classical_sgrb_offset_band()

    _, p_gold = ad_gof_weighted(model_offsets, band["gold"], weights=model_weights)
    _, p_obs = ad_gof_weighted(model_offsets, band["observed"], weights=model_weights)
    p_draws = np.array(
        [ad_gof_weighted(model_offsets, draw, weights=model_weights)[1] for draw in band["sampled"]]
    )
    lo, mid, hi = np.percentile(p_draws, [16, 50, 84])
    return {
        "p_gold": float(p_gold),
        "p_observed": float(p_obs),
        "p_sampled": float(mid),
        "p_sampled_lo": float(lo),
        "p_sampled_hi": float(hi),
    }
