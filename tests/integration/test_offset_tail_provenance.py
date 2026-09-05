"""Where the wide tail of the Section 8 offset distribution comes from.

Section 8 reports that roughly 5 percent of each GRB class's weight lies
beyond the 100 kpc plot axis, and compares the fraction beyond 30 kpc
(15 percent on the composite fiducial) against the Fong et al. 2022 BRIGHT
sample.  Both numbers read naturally as kinematics: systems that received a
large systemic kick and travelled far before merging.  They are not.

On Model A the systems that end beyond 100 kpc have a median birth radius of
about 180 kpc and a median ``v_sys`` indistinguishable from the population
median.  They did not travel; they were born in the outskirts.  The source is
the birth-radius sampler: ``hernquist_birth_radius`` inverts the exact
Hernquist (1990) Eq. 38 mass profile, whose enclosed fraction
``M(<r)/M = r^2 / (r + a)^2`` converges slowly because the density falls only
as r^-4.  At the fiducial scale radius (a = 2.75 kpc for R_e = 5 kpc) that
profile puts 16 percent of the stellar mass beyond 30 kpc and 5 percent beyond
100 kpc before any supernova has gone off.

This matters for two claims.  The agreement between the model's 15 percent
beyond 30 kpc and the observed 15 percent is largely a property of the assumed
stellar profile rather than of the kick prescription: confine birth radii to a
few effective radii and the model fraction collapses.  And Mandel et al. (2026)
ApJL, arXiv:2512.12018, whose maximum-offset bound is the natural external
check on a wide tail, derive it under the explicit assumption (their Sec. 3)
that massive stars are born deep inside the halo so that the local velocity is
small compared with the systemic velocity.  That assumption does not hold for
about a sixth of this population, so their bound and these offsets are not
measuring the same thing.

The tests below pin the provenance so it is not rediscovered, and pin the
Mandel envelope that the population does satisfy.  They deliberately do not
assert that the tail is correct: whether to truncate the Hernquist profile at
a physical stellar extent is a modelling decision, and both choices are
defensible.  What is not defensible is quoting the tail as a kinematic result.

Birth radii are supplied explicitly and the orbits advanced through
``_advance_planar_composite`` so that the correlation between birth radius and
final offset is available without replaying the integrator's internal random
draws in call order.
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_io import filter_systems, load_bns_with_kicks
from grb_offsets import (
    DEFAULT_M_GAL,
    DEFAULT_R_E,
    HALO_FIDUCIAL_Z,
    KPC_CM,
    MYR_S,
    _advance_planar_composite,
    _sigma_r_interpolator,
    hernquist_birth_radius,
    hernquist_scale_radius,
    make_host_potential,
)
from grb_physics import M_TOV, cold_stable_ns_mask

# Mandel et al. (2026) Eq. 2: the ballistic travel distance of a merging DNS is
# bounded by v_sys * min(14 Gyr, tau_GW), and tau_GW ~ v^-8 through the
# pre-second-supernova orbital velocity, giving a v^-7 branch above the
# turnover.  Both branches meet at 5 Mpc.
MANDEL26_TURNOVER_KMS = 335.0
MANDEL26_SCALE_PC = 5.0e6
PC_PER_KMS_MYR = 1.02271

_N_SUB = 50_000
_SEED = 2024


def _weighted_fraction_above(x, w, threshold):
    return float(w[x > threshold].sum() / w.sum())


def _weighted_median(x, w):
    order = np.argsort(x)
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(0.5 * w.sum(), c, x))


@pytest.fixture(scope="module")
def model_a_kicks(bns_a_path):
    """Model A BNS systemic velocities and delay times, cold-stability cut applied."""
    d = load_bns_with_kicks(path=bns_a_path, expected_model="A")
    d = filter_systems(d, cold_stable_ns_mask(d["m1"], d["m2"], m_tov=M_TOV))
    v = np.asarray(d["v_sys"], dtype=float)
    t = np.asarray(d["delay_time"], dtype=float)
    w = np.asarray(d["weights"], dtype=float)
    ok = np.isfinite(v) & np.isfinite(t) & (v > 0) & (t > 0)
    v, t, w = v[ok], t[ok], w[ok]
    sub = np.random.default_rng(7).choice(v.size, size=min(_N_SUB, v.size), replace=False)
    return v[sub], t[sub], w[sub]


@pytest.fixture(scope="module")
def integrated_offsets(model_a_kicks):
    """Birth radii, offsets, and inputs for one deterministic Section 8 realisation.

    Returns ``(r0_kpc, offsets_kpc, v_sys, weights)`` on the composite
    fiducial potential.
    """
    v, t, w = model_a_kicks
    host = make_host_potential(DEFAULT_M_GAL, DEFAULT_R_E, z=HALO_FIDUCIAL_Z, with_halo=True)
    a = hernquist_scale_radius(DEFAULT_R_E)
    sigma = _sigma_r_interpolator(host)

    rng = np.random.default_rng(_SEED)
    r0 = hernquist_birth_radius(a, rng=rng, size=v.size)

    direction = rng.standard_normal((v.size, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    v_vec = direction * (v * 1e5)[:, None]
    v_vec = v_vec + rng.standard_normal((v.size, 3)) * np.asarray(sigma(r0), dtype=float)[:, None]

    r_merge = _advance_planar_composite(
        r0,
        v_vec[:, 2],
        np.hypot(v_vec[:, 0], v_vec[:, 1]),
        t * MYR_S,
        host,
        rng,
        phase_cap=True,
    )
    return r0 / KPC_CM, r_merge / KPC_CM, v, w


@pytest.mark.requires_data
@pytest.mark.slow
def test_wide_offset_tail_is_birth_radius_not_kick_velocity(integrated_offsets):
    """Systems beyond 100 kpc started there; their kicks are unremarkable."""
    r0, offsets, v_sys, w = integrated_offsets
    far = offsets > 100.0
    assert w[far].sum() > 0, "no weight beyond 100 kpc; fixture is not exercising the tail"

    born_far = w[far & (r0 > 50.0)].sum() / w[far].sum()
    assert born_far > 0.90, (
        f"only {born_far:.1%} of the >100 kpc weight was born beyond 50 kpc; "
        "the tail's provenance has changed"
    )

    # A kicked-out population would show an elevated systemic velocity.  It does not.
    v_far = _weighted_median(v_sys[far], w[far])
    v_all = _weighted_median(v_sys, w)
    assert v_far / v_all < 1.25, (
        f"median v_sys beyond 100 kpc is {v_far:.1f} km/s against {v_all:.1f} km/s "
        "for the population; the tail now looks kinematic"
    )


@pytest.mark.requires_data
@pytest.mark.slow
def test_offset_tail_collapses_when_birth_radii_are_confined(integrated_offsets):
    """Confining birth radii to 5 R_e removes the wide tail but keeps most weight."""
    r0, offsets, _, w = integrated_offsets
    confined = r0 <= 5.0 * DEFAULT_R_E / KPC_CM

    kept = w[confined].sum() / w.sum()
    assert 0.75 < kept < 0.90, f"birth-radius cut retains {kept:.1%} of weight, expected about 81%"

    full_tail = _weighted_fraction_above(offsets, w, 100.0)
    cut_tail = _weighted_fraction_above(offsets[confined], w[confined], 100.0)
    assert full_tail > 20.0 * cut_tail, (
        f"tail beyond 100 kpc is {full_tail:.3%} uncut against {cut_tail:.3%} confined; "
        "the birth-radius profile no longer dominates the tail"
    )

    # The median is comparatively insensitive: the profile drives the tail, not the centre.
    assert (
        abs(_weighted_median(offsets, w) - _weighted_median(offsets[confined], w[confined])) < 4.0
    )


@pytest.mark.requires_data
def test_population_satisfies_the_mandel26_ballistic_envelope(model_a_kicks):
    """Every system respects Mandel+ 2026 Eq. 2, and almost none reach where it bites."""
    v, t, w = model_a_kicks
    d_ballistic_pc = v * PC_PER_KMS_MYR * t
    ratio = v / MANDEL26_TURNOVER_KMS
    d_max_pc = MANDEL26_SCALE_PC * np.minimum(ratio, ratio**-7.0)

    violating = _weighted_fraction_above(d_ballistic_pc - d_max_pc, w, 0.0)
    assert violating < 0.005, f"{violating:.2%} of weight exceeds the Eq. 2 envelope"

    # The envelope only constrains systems above the turnover, and the Fryer 2012
    # delayed kicks in this grid essentially do not populate that regime, so the
    # bound passing is not evidence about the tail either way.
    above_turnover = _weighted_fraction_above(v, w, MANDEL26_TURNOVER_KMS)
    assert above_turnover < 0.01, (
        f"{above_turnover:.2%} of weight now exceeds {MANDEL26_TURNOVER_KMS:.0f} km/s; "
        "the Eq. 2 envelope has become informative and this test should be revisited"
    )


def test_hernquist_birth_radius_tail_matches_the_analytic_mass_profile():
    """The sampler reproduces M(<r)/M = r^2/(r+a)^2, tail included.

    Hernquist (1990) Eq. 38.  This needs no COMPAS data and is the root cause of
    the offset tail: at the fiducial scale radius the profile places about a
    sixth of the stellar mass beyond 30 kpc.
    """
    a = hernquist_scale_radius(DEFAULT_R_E)
    r = hernquist_birth_radius(a, rng=np.random.default_rng(11), size=400_000) / KPC_CM
    a_kpc = a / KPC_CM

    for r_kpc in (10.0, 30.0, 100.0):
        analytic = 1.0 - (r_kpc / (r_kpc + a_kpc)) ** 2
        sampled = float((r > r_kpc).mean())
        assert sampled == pytest.approx(analytic, rel=0.05), (
            f"beyond {r_kpc:.0f} kpc: sampled {sampled:.4f} against analytic {analytic:.4f}"
        )

    assert 0.14 < float((r > 30.0).mean()) < 0.18
    assert 0.045 < float((r > 100.0).mean()) < 0.060
