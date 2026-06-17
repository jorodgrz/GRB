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

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
G_CGS = 6.674e-8  # cm^3 g^-1 s^-2
MSUN_G = 1.989e33  # g
KPC_CM = 3.0857e21  # cm
MYR_S = 3.1557e13  # s
KM_CM = 1e5  # cm

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


def compute_offsets_population_vectorized(
    v_sys_km,
    t_delay_myr,
    weights=None,
    M_gal=DEFAULT_M_GAL,
    R_e=DEFAULT_R_E,
    max_systems=50000,
    n_proj=8,
    n_steps=400,
    rng=None,
):
    """Vectorized drop-in replacement for ``compute_offsets_population``.

    Same I/O contract: returns the dict ``{'offsets_kpc', 'indices',
    'weights_sub'}``.  Replaces the per-system Python loop in
    ``compute_offsets_population`` (line 401) with a single batched RK4
    integrator (``_vectorized_orbit_3d``).  Empirical CDFs match the
    legacy code to KS distance < 0.05 (one figure-line thickness) at
    N = 200; see ``tests/unit/test_phase4_helpers.py``.

    Parameters
    ----------
    v_sys_km, t_delay_myr : array-like
        Systemic velocity [km/s] and delay time [Myr].  May contain NaN
        or non-positive values; these are filtered out via the same
        ``valid`` mask used by the legacy code.
    weights : array-like, optional
        STROOPWAFEL weights aligned with the inputs; defaults to ones.
    M_gal, R_e : float
        Hernquist galaxy mass [g] and effective radius [cm].
    max_systems : int
        Weight-based subsample cap; subsampling stays STROOPWAFEL-aware.
    n_proj : int, optional
        Number of isotropic projection angles per system; the median of
        these gives the projected offset (matches legacy n_proj = 8).
    n_steps : int, optional
        RK4 step count for the orbit integrator.  See
        ``_vectorized_orbit_3d``.
    rng : np.random.Generator, optional
        Reproducible random source.  Defaults to ``default_rng(42)``.

    Returns
    -------
    dict with keys ``'offsets_kpc' : (M,)``, ``'indices' : (M,)``,
    ``'weights_sub' : (M,)`` where M = min(N_valid, max_systems).
    """
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

    if len(valid_idx) > max_systems:
        p = w_valid / w_valid.sum()
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False, p=p)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    if len(valid_idx) == 0:
        return {"offsets_kpc": np.zeros(0), "indices": valid_idx, "weights_sub": w_valid}

    a = hernquist_scale_radius(R_e)
    v_cm = v_sys_km[valid_idx] * KM_CM
    t_s = t_delay_myr[valid_idx] * MYR_S

    r_3d = _vectorized_orbit_3d(v_cm, t_s, M_gal, a, rng, n_steps=n_steps)

    # Vectorized 2D projection (Bloom+ 1999) over (N, n_proj).
    cos_theta = rng.uniform(-1, 1, size=(len(valid_idx), n_proj))
    projected = r_3d[:, None] * np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    offsets = np.median(projected, axis=1) / KPC_CM

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
):
    """Compute projected offsets for a population of merging binaries.

    Parameters
    ----------
    v_sys_km : array
        Systemic velocities [km/s].
    t_delay_myr : array
        Delay times [Myr].
    weights : array, optional
        STROOPWAFEL weights for subsetting. If more than ``max_systems``
        have finite v_sys, a weight-based subsample is drawn.
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
        whole population in one batched RK4 pass (~100x speedup over
        the legacy per-system loop).  Set False to retain the legacy
        scalar code path; tests in ``tests/unit/test_phase4_helpers.py``
        exercise both.
    rng : np.random.Generator, optional

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
        )

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

    if len(valid_idx) > max_systems:
        p = w_valid / w_valid.sum()
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False, p=p)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

    a = hernquist_scale_radius(R_e)
    n_proj = 8
    offsets = np.zeros(len(valid_idx))

    for i, idx in enumerate(valid_idx):
        v = v_sys_km[idx]
        t = t_delay_myr[idx]

        if use_analytic:
            r_3d = _analytic_offset(v, t, M_gal, a, rng=rng)
        else:
            r_3d = integrate_orbit(v * KM_CM, t * MYR_S, M_gal, a, rng=rng)

        cos_theta = rng.uniform(-1, 1, size=n_proj)
        projected = r_3d * np.sqrt(1.0 - cos_theta**2)
        offsets[i] = np.median(projected) / KPC_CM

    return {
        "offsets_kpc": offsets,
        "indices": valid_idx,
        "weights_sub": w_valid,
    }


def compute_offsets_mixed_hosts(
    v_sys_km, t_delay_myr, weights=None, host_models=None, max_systems=50000, rng=None, **kw
):
    """Compute offsets for a population using a mixture of host galaxy types.

    A single shared subsample is drawn once and then each host potential
    is evaluated on the *same* set of binaries, avoiding inconsistent
    subsampling across galaxy types.

    Parameters
    ----------
    host_models : dict, optional
        Keys are host names, values are dicts with ``M_gal``, ``R_e``,
        and ``weight`` (mixture fraction, should sum to 1).
        Defaults to ``HOST_MODELS``.
    max_systems : int
        Maximum population size (weight-based subsample if exceeded).
    rng : np.random.Generator, optional

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
    valid_idx = np.where(valid)[0]
    if weights is not None:
        w_valid = np.asarray(weights, dtype=float)[valid_idx]
    else:
        w_valid = np.ones(len(valid_idx))

    if len(valid_idx) > max_systems:
        p = w_valid / w_valid.sum()
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False, p=p)
        valid_idx = valid_idx[chosen]
        w_valid = w_valid[chosen]

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
    n_proj = kw.pop("n_proj", 8)

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
    if len(valid_idx) > max_systems:
        p = w_valid / w_valid.sum()
        chosen = rng.choice(len(valid_idx), size=max_systems, replace=False, p=p)
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
            r_3d = _vectorized_orbit_3d(
                v_cm_all[host_mask], t_s_all[host_mask], hp["M_gal"], a_host, rng, n_steps=n_steps
            )
            cos_th = rng.uniform(-1, 1, size=(host_mask.sum(), n_proj))
            projected = r_3d[:, None] * np.sqrt(np.maximum(0.0, 1.0 - cos_th * cos_th))
            offsets[host_mask] = np.median(projected, axis=1) / KPC_CM

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
# Short-GRB host offsets
# ═══════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════
# Long-GRB-with-kilonova host offsets
# ═══════════════════════════════════════════════════════════════════════════
# Per-event projected physical offsets [kpc] for the three long GRBs with
# spectroscopically or photometrically confirmed kilonova counterparts,
# which form the lGRB sub-sample of Rastinejad et al. (2024), ApJ 979, 190
# ("uniform modeling of observed kilonovae"; arXiv:2409.02158).  Offset
# measurements are from the discovery papers:
#
#   060614    0.73 kpc  (Della Valle et al. 2006, Nature 444, 1050;
#                        GCN 5276; afterglow centroid 0.32" S, 0.09" E of
#                        host centroid at z = 0.125; H0 = 71 km/s/Mpc)
#   211211A   7.91 kpc  (Rastinejad et al. 2022, Nature 612, 223;
#                        arXiv:2204.10864; "modest offset of 5.55"+/-0.03"
#                        = 7.91 +/- 0.03 kpc in projection")
#   230307A  38.9  kpc  (Levan et al. 2024, Nature 626, 737;
#                        arXiv:2307.02098; "30.2 arcseconds (38.9 kpc in
#                        projection) from the burst position")
#
# GRB 191019A (Levan et al. 2023, Nat. Astron. 7, 976) is a candidate
# dynamical / AGN-disk lGRB-merger with a near-nuclear offset of
# 78 +/- 109 pc but no confirmed kilonova counterpart, so it is excluded
# from this kilonova-anchored sample.
OBSERVED_LGRB_KN_OFFSETS_KPC = np.array(
    [
        0.73,  # 060614
        7.91,  # 211211A
        38.9,  # 230307A
    ]
)
