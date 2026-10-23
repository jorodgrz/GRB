"""Section 8a of grb_main.ipynb: the free-drift check that precedes the orbits.

The panel carries no potential, so what it has to get right is the arithmetic:
the km/s x Myr conversion, the two-leg sum, and the ordering it sets up for the
integrated figure, where the slow long-delay BHNS population drifts further
than the fast prompt BNS one.

Reference: Bloom, Sigurdsson and Pols (1999), MNRAS 305, 763.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from grb_offsets import ballistic_offset


def test_free_drift_matches_the_hand_conversion():
    """200 km/s for 100 Myr is 20.5 kpc."""
    d = ballistic_offset(np.array([200.0]), np.array([100.0]))
    assert np.isclose(d[0], 200.0 * 100.0 * 1.0227e-3, rtol=1e-3)


def test_slow_long_delay_population_drifts_further_than_fast_prompt_one():
    """The BHNS-versus-BNS ordering the panel shows, on the section's medians.

    BHNS systems are launched at a few tens of km/s but wait gigayears; BNS
    systems are launched at over a hundred but merge within a few hundred Myr.
    Free drift puts the slow population an order of magnitude further out,
    which the orbit integration then reverses.
    """
    d_bhns = ballistic_offset(np.array([50.0]), np.array([2500.0]))
    d_bns = ballistic_offset(np.array([120.0]), np.array([120.0]))
    assert d_bhns[0] > 100.0
    assert d_bns[0] < 20.0


def test_two_leg_sum_never_falls_below_the_inspiral_leg_alone():
    rng = np.random.default_rng(5)
    n = 200
    v2, t2 = rng.uniform(10.0, 400.0, n), 10.0 ** rng.uniform(0.0, 4.0, n)
    v1, t1 = rng.uniform(10.0, 300.0, n), rng.uniform(0.0, 30.0, n)
    both = ballistic_offset(v2, t2, v_sys_1_km=v1, t_leg1_myr=t1)
    assert (both >= ballistic_offset(v2, t2)).all()


def test_free_drift_is_not_an_upper_bound_on_the_integrated_offset():
    """Free drift omits the birth radius, so it does not bound the model offset.

    The Section 8a prose says so explicitly, because the opposite reading is
    the natural one and it is wrong: a binary born at 20 kpc and drifting 2 kpc
    ends at a larger offset than free drift assigns it.  ``compute_offsets_population``
    rather than the mixed-host wrapper, because only the single-host function
    returns indices into the arrays the caller passed; the wrapper subsamples
    first, so its per-host indices address the subsample and pairing a system
    with its own free drift through them silently compares different binaries.
    """
    from grb_offsets import DEFAULT_M_GAL, DEFAULT_R_E, compute_offsets_population

    rng = np.random.default_rng(19)
    n = 600
    # Slow and prompt: the drift is small next to a typical birth radius.
    v = rng.uniform(5.0, 25.0, n)
    t = rng.uniform(10.0, 100.0, n)
    res = compute_offsets_population(
        v,
        t,
        weights=np.ones(n),
        M_gal=DEFAULT_M_GAL,
        R_e=DEFAULT_R_E,
        max_systems=n,
        rng=np.random.default_rng(4),
        with_halo=True,
    )
    free = ballistic_offset(v, t)
    exceeds = res["offsets_kpc"] > free[res["indices"]]
    assert exceeds.mean() > 0.5, exceeds.mean()


# ---------------------------------------------------------------------------
# What the two panels of the figure are allowed to claim
# ---------------------------------------------------------------------------
def test_collinear_two_leg_sum_bounds_the_true_displacement():
    """The plotted quantity is a bound on the displacement, not the displacement.

    ``ballistic_offset`` adds the two leg lengths, but the legs point in
    uncorrelated directions, so the vector displacement is smaller.  Section 8a
    projects the sum, and the figure is only honest if that sum is an upper
    bound; on Model A the gap is 0.2 to 0.8 percent at the median because the
    first leg is sub-percent, but the inequality is what licenses the wording.
    """
    rng = np.random.default_rng(3)
    n = 20000
    v1, t1 = rng.uniform(0.0, 300.0, n), rng.uniform(0.0, 50.0, n)
    v2, t2 = rng.uniform(0.0, 300.0, n), rng.uniform(0.0, 3000.0, n)
    d1, d2 = ballistic_offset(v1, t1), ballistic_offset(v2, t2)
    summed = ballistic_offset(v2, t2, v1, t1)
    np.testing.assert_allclose(summed, d1 + d2, rtol=1e-12)
    cos_ang = rng.uniform(-1.0, 1.0, n)
    displacement = np.sqrt(d1**2 + d2**2 + 2.0 * d1 * d2 * cos_ang)
    assert np.all(displacement <= summed + 1e-9)


def test_isotropic_projection_costs_more_than_the_single_distance_factor():
    """Why the caption quotes 0.64 to 0.73 rather than the geometric 0.866.

    For one fixed distance the median of d sqrt(1 - mu^2) over an isotropic
    viewing angle is sqrt(3)/2 times d.  Spread the distance over decades and
    the ratio of medians falls to exp(ln 2 - 1) = 0.736, because the projection
    factor is left-skewed in the log.  The six classes sit at or below that
    asymptote, so reading 0.866 off the geometry would halve the apparent gap
    between the joint panel's diagonals and the projected axis beside them.
    """
    rng = np.random.default_rng(4)
    n = 400000

    def project(d):
        mu = rng.uniform(-1.0, 1.0, d.size)
        return d * np.sqrt(1.0 - mu * mu)

    fixed = np.full(n, 10.0)
    assert np.median(project(fixed)) / 10.0 == pytest.approx(np.sqrt(3.0) / 2.0, abs=0.005)

    broad = 10.0 * np.exp(rng.normal(0.0, 2.0, n))
    ratio = np.median(project(broad)) / np.median(broad)
    assert ratio == pytest.approx(np.exp(np.log(2.0) - 1.0), abs=0.01), ratio
    assert ratio < 0.80


def _half_weight_level(H):
    """The level enclosing half the histogram weight, as the notebook computes it."""
    flat = np.sort(H.ravel())[::-1]
    return flat[np.searchsorted(np.cumsum(flat) / flat.sum(), 0.50)]


def test_binning_on_the_visible_frame_would_undercount_the_contour():
    """Why the joint panel bins the whole sample instead of the axis limits.

    ``np.histogram2d`` discards out-of-range values without complaint, so a
    grid sized to the visible axes makes a "50 percent" contour enclose half
    the *visible* weight rather than half the class.  Model A puts up to 4.2
    percent of a class outside the frame; the fraction here is exaggerated so
    the mechanism cannot be mistaken for noise.
    """
    rng = np.random.default_rng(9)
    # A peaked density, because a uniform one makes every bin tie at the
    # contour level and the comparison degenerates.
    x = np.concatenate([rng.normal(5.0, 1.0, 8000), rng.normal(0.2, 0.1, 2000)])
    y = np.concatenate([rng.normal(5.0, 1.0, 8000), rng.normal(5.0, 1.0, 2000)])
    w = np.ones(x.size)

    enclosed = []
    for bins in (np.linspace(1.0, 10.0, 40), np.linspace(-1.0, 10.0, 48)):
        H = np.histogram2d(x, y, bins=[bins, bins], weights=w)[0]
        enclosed.append(H[H >= _half_weight_level(H)].sum() / w.sum())

    frame_grid, full_grid = enclosed
    # The frame grid sees 80 percent of the sample, so its "half" is 0.41 of
    # the class rather than 0.50.
    assert frame_grid == pytest.approx(0.41, abs=0.03), frame_grid
    assert full_grid == pytest.approx(0.51, abs=0.03), full_grid
    assert full_grid - frame_grid > 0.05


# ---------------------------------------------------------------------------
# Data provenance: the columns Section 8a reads, on the real COMPAS files
# ---------------------------------------------------------------------------
@pytest.mark.requires_data
@pytest.mark.parametrize("loader_pair", ["bns", "bhns"])
def test_kicks_and_channels_loaders_return_the_same_rows(loader_pair):
    """Section 8a pairs weights from one loader with velocities from another.

    ``grb_main`` Section 0 checks only that the two agree in length.  If a
    loader change ever reordered rows, every system's velocity would be paired
    with another system's weight and GRB class, and nothing in the shape check
    would notice.  The offsets are only meaningful while this holds.
    """
    from grb_io import (
        load_bhns_with_channels,
        load_bhns_with_kicks,
        load_bns_with_channels,
        load_bns_with_kicks,
    )

    if loader_pair == "bns":
        ch = load_bns_with_channels(expected_model="A", expected_ns_max=2.5)
        kk = load_bns_with_kicks(expected_model="A", expected_ns_max=2.5)
    else:
        ch = load_bhns_with_channels(expected_model="A", expected_ns_max=2.5)
        kk = load_bhns_with_kicks(expected_model="A", expected_ns_max=2.5)

    for col in ("weights", "delay_time", "metallicity"):
        np.testing.assert_array_equal(ch[col], kk[col], err_msg=col)


@pytest.mark.requires_data
@pytest.mark.parametrize("loader_pair", ["bns", "bhns"])
def test_drift_baseline_identity_holds_on_the_real_data(loader_pair):
    """The two-leg baseline is the delay time less the progenitor lifetime.

    COMPAS defines delay_time = tform + tc with tform the second supernova
    time, so t_leg1 + t_inspiral = delay_time - t_sn1 exactly.  Section 8a and
    Section 8 both drift systems for that baseline rather than the delay time,
    and the whole short-delay end of the distribution depends on the
    difference, so the identity is worth pinning rather than believing.
    """
    from grb_io import load_bhns_with_kicks, load_bns_with_kicks

    d = (load_bns_with_kicks if loader_pair == "bns" else load_bhns_with_kicks)(
        expected_model="A", expected_ns_max=2.5
    )
    t_leg1 = d["t_sn2"] - d["t_sn1"]
    assert np.all(t_leg1 >= 0.0), "supernovae out of time order"
    np.testing.assert_allclose(d["delay_time"], d["t_sn2"] + d["t_inspiral"], atol=1e-9)
    np.testing.assert_allclose(t_leg1 + d["t_inspiral"], d["delay_time"] - d["t_sn1"], atol=1e-9)
    assert np.array_equal(d["v_sys"], d["v_sys_2"]), "v_sys is not the post-SN2 speed"


@pytest.mark.requires_data
def test_kinematic_columns_carry_no_sentinel_values():
    """No NaN, zero or negative in the five columns the free-drift panel reads.

    ``_wcdf`` filters on ``isfinite`` and ``> 0``, so any of those would leave
    the panel silently without complaint.
    """
    from grb_io import load_bns_with_kicks

    d = load_bns_with_kicks(expected_model="A", expected_ns_max=2.5)
    for col in ("v_sys", "v_sys_1", "t_sn1", "t_sn2", "t_inspiral"):
        x = d[col]
        assert np.all(np.isfinite(x)), col
        assert np.all(x > 0.0), col


@pytest.mark.requires_data
def test_kinematic_column_units_are_the_assumed_ones():
    """The HDF5 carries no units attribute, so the convention needs a check.

    A silent factor of 1000 in either column would rescale every offset in
    Sections 8 and 8a.  Two ranges settle it: the inspiral time is cut at the
    Hubble time, which can only be ~1.4e4 in Myr, and a systemic speed of a few
    hundred can only be km/s.
    """
    from grb_io import load_bns_with_kicks

    d = load_bns_with_kicks(expected_model="A", expected_ns_max=2.5)
    assert 1e4 < np.nanmax(d["t_inspiral"]) < 1.5e4, "t_inspiral is not in Myr"
    assert 1e2 < np.nanmax(d["v_sys"]) < 1e4, "v_sys is not in km/s"
    assert np.nanmax(d["t_sn1"]) < 1e3, "t_sn1 is not a massive-star lifetime in Myr"


@pytest.mark.requires_data
def test_file_kick_directions_are_selected_and_not_isotropic():
    """The sky projection does not rest on the kicks being isotropic here.

    The supernovae group of a DCO output holds only systems that survived both
    supernovae and merged, so its ``theta`` / ``phi`` are a selected sample and
    are measurably anisotropic: ``sin(theta)`` is over-dispersed against the
    uniform an isotropic prescription would give, and splitting on whether the
    supernova tightened or widened the orbit puts the two subsets on either
    side of it, which is what selection looks like.

    This is recorded so it is not later mistaken for a bug in the loader or in
    COMPAS.  ``mu ~ U(-1, 1)`` is exact regardless, because theta and phi are
    measured in the pre-supernova orbital frame and that frame is randomly
    oriented with respect to the observer.
    """
    import h5py

    path = "data/COMPASCompactOutput_BNS_A.h5"
    if not os.path.exists(path):
        pytest.skip("Model A BNS file not present")
    with h5py.File(path, "r") as f:
        sn = f["supernovae"]
        theta = sn["theta"][...].squeeze()
        sep_after = sn["separationAfter"][...].squeeze()
        sep_before = sn["separationBefore"][...].squeeze()

    s = np.sin(theta)
    isotropic_std = 1.0 / np.sqrt(3.0)
    assert abs(s.mean()) < 0.01, "sin(theta) is not centred; the frame convention has changed"
    assert s.std() > isotropic_std, "sin(theta) is no longer over-dispersed; check the sample"

    tightened = sep_after < sep_before
    assert s[tightened].std() < isotropic_std < s[~tightened].std(), (
        "the tightening and widening subsets no longer bracket the isotropic spread, "
        "so the anisotropy is no longer explained by selection on orbital outcome"
    )
