"""Behaviour and edge-case tests for ``grb_classify``.

Complements ``tests/unit/test_classify_asserts.py`` (kwarg coherence,
fiducial-ns_max guard rails), ``tests/unit/test_phase4_helpers.py``
(``classify_observed_mergers`` mass-plane mapping and
``channel_class_crosstab`` normalisations), and
``tests/sections/test_section_01_mass_plane.py`` (per-system <->
``classify_grid`` consistency over a synthetic meshgrid).  This file
fills the gaps: partition completeness, exact boundary placement,
kwarg coherence for ``k_thresh`` / ``hmns_factor`` / ``R_1p4_km`` /
``a_bh``, ``foucart_kw`` forwarding, ``classify_formation_channels``
event-sequence priorities, and per-call thresholds on
``classify_observed_mergers``.

References cited inline per test: Gottlieb et al. (2023, 2024,
arXiv:2309.00038, 2411.13657) classification thresholds; Foucart et
al. (2018) BHNS remnant; Broekgaarden et al. (2021, arXiv:2103.02608)
formation-channel definitions; Margalit and Metzger (2017, ApJL 850,
L19) HMNS-lifetime factor; Rastinejad et al. (2024, ApJ 979, 190)
ejecta decomposition.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from grb_classify import (
    GRID_CLASS_LABELS,
    KN_M_EJ_FAINT_MAX,
    KN_RED_FRAC_BLUE_MAX,
    KN_RED_FRAC_RED_MIN,
    bns_boundary_lines,
    channel_class_crosstab,
    classify_bhns,
    classify_bhns_spins,
    classify_bns_2024,
    classify_formation_channels,
    classify_grid,
    classify_observed_mergers,
)
from grb_physics import (
    HMNS_FACTOR_DEFAULT,
    M_THRESH,
    M_TOV,
    MDISK_LONG,
    MDISK_SHORT,
    Q_THRESH_BNS,
)


# ---------------------------------------------------------------------------
# classify_bns_2024: partition, boundaries, ordering, kwargs, vectorisation
# ---------------------------------------------------------------------------
def _bns_synthetic_sample(n=2000, seed=2026):
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(1.10, 2.20, size=n)
    m2 = rng.uniform(1.10, 2.20, size=n)
    return m1, m2


def test_classify_bns_2024_partition_is_exclusive_and_complete():
    """The four Gottlieb 2024 class masks form a partition.

    Every (m1, m2) lands in exactly one class.  Catches a refactor
    that lets two masks overlap or leaves a system unclassified at
    the floating-point boundary.
    """
    m1, m2 = _bns_synthetic_sample()
    out = classify_bns_2024(m1, m2)
    stack = np.stack(
        [
            out["sbGRB + blue KN"],
            out["lbGRB + red KN (HMNS)"],
            out["lbGRB + red KN (disk)"],
            out["Faint lbGRB"],
        ],
        axis=0,
    )
    counts = stack.sum(axis=0)
    assert (counts == 1).all(), (
        f"{int((counts != 1).sum())} systems land in 0 or >1 classes; "
        f"unique counts present = {np.unique(counts).tolist()}"
    )


def test_classify_bns_2024_boundary_at_hmns_split_lands_in_hmns_class():
    """A system with ``M_tot = h * M_TOV`` falls in the lbGRB-HMNS class.

    Gottlieb 2024 hybrid convention: the HMNS-lifetime split uses
    ``M_tot < hmns_split`` for the long-lived branch, so the boundary
    itself belongs to the short-lived (``>=``) side.
    """
    m_tot = HMNS_FACTOR_DEFAULT * M_TOV
    out = classify_bns_2024(m_tot / 2.0, m_tot / 2.0)
    assert not bool(out["sbGRB + blue KN"])
    assert bool(out["lbGRB + red KN (HMNS)"])


def test_classify_bns_2024_boundary_at_m_thresh_lands_in_prompt_collapse():
    """A system with ``M_tot = m_thresh`` falls in a prompt-collapse class.

    Uses ``m_thresh = 2.8 Msun`` explicitly to avoid the
    floating-point rounding of the default ``K_THRESH_DEFAULT * M_TOV
    = 2.7940000...05``.  A 1.4 + 1.4 Msun binary at q = 1 < q_thresh
    therefore lands in 'Faint lbGRB'.
    """
    out = classify_bns_2024(1.4, 1.4, m_thresh=2.8)
    assert bool(out["Faint lbGRB"])
    assert not bool(out["lbGRB + red KN (HMNS)"])


def test_classify_bns_2024_boundary_at_q_thresh_lands_in_disk_class():
    """A prompt-collapse system at ``q = q_thresh`` lands in 'lbGRB + red KN (disk)'.

    The mass-ratio boundary uses ``q >= q_thresh`` for the disk side,
    so the boundary itself belongs there.  Constructed with
    ``m1 = 1.5, m2 = 1.25`` so ``q = 1.20`` and ``m1 + m2 = 2.75``
    are both exact floating-point representations; pinning ``m_thresh
    = 2.75`` then puts the system on both the q and M_tot boundaries
    simultaneously, with no floating-point slop.
    """
    m1, m2 = 1.5, 1.25
    assert m1 + m2 == 2.75
    assert m1 / m2 == Q_THRESH_BNS
    out = classify_bns_2024(m1, m2, m_thresh=2.75)
    assert bool(out["lbGRB + red KN (disk)"])
    assert not bool(out["Faint lbGRB"])


def test_classify_bns_2024_ordering_invariance_under_m1_m2_swap():
    """``classify_bns_2024`` is symmetric in ``(m1, m2)``.

    The classifier sorts internally via ``np.maximum`` / ``np.minimum``,
    so a future regression that drops the sort step would fail this
    invariance.
    """
    m1, m2 = _bns_synthetic_sample()
    out_ab = classify_bns_2024(m1, m2)
    out_ba = classify_bns_2024(m2, m1)
    for key in out_ab:
        np.testing.assert_array_equal(out_ab[key], out_ba[key])


def test_classify_bns_2024_k_thresh_overrides_m_thresh():
    """Passing ``k_thresh`` moves the prompt-collapse boundary to ``k_thresh * m_tov``.

    The classifier delegates to ``_resolve_m_thresh`` so EOS sweeps
    that change ``m_tov`` also move the prompt-collapse threshold;
    here we pin the override path with ``m_tov = 2.0, k_thresh =
    1.30 -> m_thresh_eff = 2.60``.  A 1.30 + 1.30 (M_tot = 2.60)
    system at q = 1 < q_thresh therefore lands in 'Faint lbGRB'.
    """
    out = classify_bns_2024(1.30, 1.30, m_tov=2.0, m_thresh=None, k_thresh=1.30)
    assert bool(out["Faint lbGRB"])
    assert not bool(out["lbGRB + red KN (HMNS)"])


def test_classify_bns_2024_hmns_factor_1_shrinks_sbgrb_to_below_m_tov():
    """``hmns_factor = 1.0`` shrinks the sbGRB class to ``M_tot < M_TOV``.

    The classifier defines ``hmns_split = hmns_factor * m_tov`` and
    routes ``M_tot < hmns_split`` to sbGRB.  Setting
    ``hmns_factor = 1.0`` collapses the long-lived HMNS branch to
    systems with ``M_tot < M_TOV`` only; everything with
    ``M_TOV <= M_tot < M_thresh`` shifts into the lbGRB + red KN
    (HMNS) class instead.  Sweep a synthetic sample of systems
    bracketing ``M_TOV`` and pin the threshold drop.
    """
    rng = np.random.default_rng(3)
    n = 1000
    m_tot = rng.uniform(M_TOV - 0.20, M_TOV + 0.20, size=n)
    m1 = 0.55 * m_tot
    m2 = 0.45 * m_tot
    out = classify_bns_2024(m1, m2, hmns_factor=1.0)
    # No sbGRB system can land above M_TOV under hmns_factor = 1.0.
    sb_mask = out["sbGRB + blue KN"]
    assert (m_tot[sb_mask] < M_TOV).all(), (
        f"{int((m_tot[sb_mask] >= M_TOV).sum())} sbGRB systems landed "
        f"at or above M_TOV under hmns_factor = 1.0; the sbGRB upper "
        f"boundary did not move."
    )
    # And every system with M_TOV <= M_tot < M_thresh is HMNS now.
    in_window = (m_tot >= M_TOV) & (m_tot < M_THRESH)
    assert out["lbGRB + red KN (HMNS)"][in_window].all(), (
        "Some systems with M_TOV <= M_tot < M_THRESH did not move "
        "into the HMNS class under hmns_factor = 1.0."
    )


def test_classify_bns_2024_vectorisation_parity():
    """Vectorised mask output matches an elementwise scalar loop."""
    m1, m2 = _bns_synthetic_sample(n=100, seed=7)
    out_vec = classify_bns_2024(m1, m2)
    out_loop = {k: [] for k in out_vec}
    for i in range(m1.size):
        cell = classify_bns_2024(float(m1[i]), float(m2[i]))
        for k in out_vec:
            out_loop[k].append(bool(cell[k]))
    for k in out_vec:
        np.testing.assert_array_equal(out_vec[k], np.asarray(out_loop[k], dtype=bool))


# ---------------------------------------------------------------------------
# bns_boundary_lines: m1_lim clip, empty-curve edge, k_thresh
# ---------------------------------------------------------------------------
def test_bns_boundary_lines_m1_lim_clipping():
    """``m1_lim = (lo, hi)`` clips every returned curve to ``[lo, hi]``.

    Pinning the optional vertical clip is what lets Section 1's plot
    code drop the per-curve clipping algebra.
    """
    m2 = np.linspace(1.10, 2.10, 200)
    lo, hi = 1.5, 2.0
    bdy = bns_boundary_lines(m2, m1_lim=(lo, hi))
    for key, (m2_arr, m1_arr) in bdy.items():
        assert (m1_arr >= lo - 1e-12).all(), (key, m1_arr.min())
        assert (m1_arr <= hi + 1e-12).all(), (key, m1_arr.max())
        assert (m1_arr >= m2_arr - 1e-12).all(), (key, "m1 >= m2 leak")


def test_bns_boundary_lines_m_tot_and_hmns_empty_when_m2_above_threshold():
    """When ``m2 > m_thresh`` everywhere, ``M_tot`` and ``HMNS`` curves are empty.

    The ``M_tot`` curve traces ``m1 = m_thresh - m2`` which goes
    negative once ``m2 > m_thresh``; the ``HMNS`` curve traces
    ``m1 = hmns_split - m2`` which is even smaller.  Both are clipped
    by the ``m1 >= m2`` guard inside ``_clip``.  The ``q`` curve is
    unaffected: ``m1 = q_thresh * m2`` always satisfies ``m1 >= m2``
    for ``q_thresh >= 1``.
    """
    m2 = np.linspace(2.85, 3.20, 10)  # all above default M_THRESH ~ 2.794
    bdy = bns_boundary_lines(m2)
    assert bdy["M_tot"][0].size == 0
    assert bdy["HMNS"][0].size == 0
    assert bdy["q"][0].size == m2.size


def test_bns_boundary_lines_k_thresh_override():
    """``k_thresh`` with ``m_tov`` shifts the ``M_tot`` curve to ``k_thresh * m_tov``."""
    m2 = np.linspace(0.5, 1.5, 50)
    bdy = bns_boundary_lines(m2, m_tov=2.0, m_thresh=None, k_thresh=1.30)
    m2_mt, m1_mt = bdy["M_tot"]
    assert m1_mt.size > 0
    np.testing.assert_allclose(m2_mt + m1_mt, 1.30 * 2.0, atol=1e-12)


# ---------------------------------------------------------------------------
# classify_bhns: partition, boundary, M_disk diagnostic, foucart_kw forwarding
# ---------------------------------------------------------------------------
def test_classify_bhns_three_class_partition_is_exclusive_and_complete():
    """The three BHNS class masks partition every system exactly once."""
    rng = np.random.default_rng(5)
    n = 500
    M_NS = rng.uniform(1.20, 2.20, size=n)
    Q = rng.uniform(2.0, 8.0, size=n)
    M_BH = Q * M_NS
    chi = rng.uniform(0.0, 0.85, size=n)
    out = classify_bhns(M_BH, M_NS, a_BH=chi, R_NS_km=12.0)
    masks = [
        out["No GRB"],
        out["Faint lbGRB (BHNS)"],
        out["lbGRB + red KN (BHNS disk)"],
    ]
    counts = np.stack(masks, axis=0).sum(axis=0)
    assert (counts == 1).all(), np.unique(counts).tolist()


def test_classify_bhns_boundary_at_md_short_lands_in_faint():
    """A system with ``M_disk = md_short`` lands in 'Faint lbGRB (BHNS)'.

    The classifier uses ``M_disk >= md_short`` for the lower edge of
    the faint class, so the boundary belongs there.  The test
    pre-computes the canonical ``M_disk`` at ``M_BH = 4.5, M_NS =
    1.35, a_BH = 0.5, R_NS = 12 km`` and pins ``md_short`` to that
    exact value so the ``>=`` boundary is hit to floating-point
    precision; ``md_long`` is held a factor of 1.5 higher so the
    system cannot leak into the upper class.
    """
    from grb_physics import foucart_disk_mass

    M_BH = np.array([4.5])
    M_NS = np.array([1.35])
    M_disk_known = float(foucart_disk_mass(M_BH[0], M_NS[0], a_BH=0.5, R_NS_km=12.0))
    out = classify_bhns(
        M_BH,
        M_NS,
        a_BH=0.5,
        R_NS_km=12.0,
        md_short=M_disk_known,
        md_long=1.5 * M_disk_known,
    )
    assert bool(out["Faint lbGRB (BHNS)"][0])
    assert not bool(out["No GRB"][0])
    assert not bool(out["lbGRB + red KN (BHNS disk)"][0])


def test_classify_bhns_returns_M_disk_diagnostic_key():
    """``classify_bhns`` returns a raw ``M_disk`` array alongside the masks.

    The 'M_disk' diagnostic key feeds the BHNS class-mass histograms
    in Section 8 of the notebook; pinning its shape catches a
    refactor that hides the diagnostic.
    """
    M_BH = np.linspace(3.0, 12.0, 8)
    M_NS = np.full_like(M_BH, 1.35)
    out = classify_bhns(M_BH, M_NS, a_BH=0.5, R_NS_km=12.0)
    assert "M_disk" in out
    assert isinstance(out["M_disk"], np.ndarray)
    assert out["M_disk"].shape == M_BH.shape


def test_classify_bhns_vectorisation_parity():
    """Vectorised classifier matches an elementwise scalar loop per class."""
    rng = np.random.default_rng(9)
    n = 64
    M_NS = rng.uniform(1.25, 2.0, size=n)
    Q = rng.uniform(2.5, 6.0, size=n)
    M_BH = Q * M_NS
    out_vec = classify_bhns(M_BH, M_NS, a_BH=0.5, R_NS_km=12.0)
    out_loop = {
        k: np.array(
            [
                bool(
                    classify_bhns(
                        float(M_BH[i]),
                        float(M_NS[i]),
                        a_BH=0.5,
                        R_NS_km=12.0,
                    )[k]
                )
                for i in range(n)
            ]
        )
        for k in ("No GRB", "Faint lbGRB (BHNS)", "lbGRB + red KN (BHNS disk)")
    }
    for k, expected in out_loop.items():
        np.testing.assert_array_equal(out_vec[k], expected)


def test_classify_bhns_foucart_kw_clip_Q_forwarding():
    """``**foucart_kw`` reaches ``foucart_disk_mass``; ``clip_Q`` zeros high-Q systems.

    Constructed at high BH spin (``a_BH = 0.9``) so the Foucart
    formula gives a non-zero disk for both Q = 4 and Q = 10 without
    clipping; ``clip_Q = 7`` then drops the Q = 10 system to zero
    disk mass and routes it to 'No GRB'.
    """
    M_NS = np.array([1.35, 1.35])
    Q = np.array([4.0, 10.0])
    M_BH = Q * M_NS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        no_clip = classify_bhns(M_BH, M_NS, a_BH=0.9, R_NS_km=12.0)
        with_clip = classify_bhns(M_BH, M_NS, a_BH=0.9, R_NS_km=12.0, clip_Q=7.0)
    assert no_clip["M_disk"][1] > 0.0
    assert with_clip["M_disk"][1] == 0.0
    assert bool(with_clip["No GRB"][1])


# ---------------------------------------------------------------------------
# classify_bhns_spins: dict-of-results
# ---------------------------------------------------------------------------
def test_classify_bhns_spins_returns_one_entry_per_spin():
    """``classify_bhns_spins`` keys match the requested ``spins`` tuple."""
    spins = (0.0, 0.3, 0.5, 0.7, 0.9)
    M_BH = np.array([6.0])
    M_NS = np.array([1.35])
    out = classify_bhns_spins(M_BH, M_NS, spins=spins, R_NS_km=12.0)
    assert set(out.keys()) == set(spins)


def test_classify_bhns_spins_shapes_match_input():
    """Every per-spin output array shares the input shape."""
    spins = (0.0, 0.5, 0.9)
    M_BH = np.linspace(3.0, 12.0, 6)
    M_NS = np.full_like(M_BH, 1.35)
    out = classify_bhns_spins(M_BH, M_NS, spins=spins, R_NS_km=12.0)
    for a, res in out.items():
        for key in ("No GRB", "Faint lbGRB (BHNS)", "lbGRB + red KN (BHNS disk)", "M_disk"):
            assert res[key].shape == M_BH.shape, (a, key, res[key].shape)


# ---------------------------------------------------------------------------
# classify_grid: BBH region, ns_min exclusion, default ns_max warning,
# R_1p4_km / a_bh effects
# ---------------------------------------------------------------------------
def test_classify_grid_bbh_region_labelled_zero():
    """Cells with both components > ``ns_max`` are labelled 0 (BBH)."""
    g_high = np.linspace(3.0, 8.0, 10)
    m1g, m2g = np.meshgrid(g_high, g_high)
    cls = classify_grid(m1g, m2g, ns_max=2.5, R_1p4_km=12.0)
    assert (cls == 0).all(), f"{int((cls != 0).sum())} BBH cells were not 0"


def test_classify_grid_excludes_m_light_below_ns_min_with_warning():
    """``0 < m_light < ns_min`` cells stay at 0 and a warning fires."""
    m1g = np.array([[6.0, 6.0]])  # BH
    m2g = np.array([[0.5, 1.0]])  # both below ns_min=1.1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cls = classify_grid(m1g, m2g, ns_max=2.5, ns_min=1.1, R_1p4_km=12.0)
    msgs = [str(item.message) for item in w]
    assert any("m_light <" in m for m in msgs), msgs
    assert (cls == 0).all()


def test_classify_grid_default_ns_max_emits_documented_warning():
    """Omitting ``ns_max`` falls back to ``m_tov + 0.15`` with a warning.

    The docstring promises this fallback and the warning text.  A
    refactor that removes the fallback (e.g. raising instead) would
    silently change every notebook that omits the kwarg.
    """
    g = np.linspace(1.0, 4.0, 8)
    m1g, m2g = np.meshgrid(g, g)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        classify_grid(m1g, m2g, R_1p4_km=12.0)
    msgs = [str(item.message) for item in w]
    assert any("ns_max not specified" in m for m in msgs), msgs


def test_classify_grid_R_1p4_override_shifts_bhns_partition():
    """Stiffer EOS (larger ``R_1.4``) puts more BHNS cells in class 6.

    Foucart 2018 disk mass scales with NS compactness via the
    ``(1 - 2 C_NS)`` factor: a stiffer EOS gives a larger NS, lower
    compactness, more tidal disruption, and a larger post-merger
    disk.  Compare the EOS_MODELS['APR4'] anchor (R_1.4 = 11.1 km)
    against EOS_MODELS['DD2'] (R_1.4 = 13.2 km) on the same grid.
    """
    M_BH_axis = np.linspace(3.0, 15.0, 40)
    M_NS_axis = np.linspace(1.2, 2.2, 30)
    BHg, NSg = np.meshgrid(M_BH_axis, M_NS_axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls_soft = classify_grid(BHg, NSg, ns_max=2.5, R_1p4_km=11.1, a_bh=0.5)
        cls_stiff = classify_grid(BHg, NSg, ns_max=2.5, R_1p4_km=13.2, a_bh=0.5)
    n6_soft = int((cls_soft == 6).sum())
    n6_stiff = int((cls_stiff == 6).sum())
    assert n6_stiff > n6_soft, (
        f"Stiff EOS (R_1.4 = 13.2) put {n6_stiff} cells in class 6 vs "
        f"{n6_soft} for soft EOS (R_1.4 = 11.1); expected the opposite "
        f"compactness ordering."
    )


def test_classify_grid_a_bh_shifts_bhns_partition():
    """Higher BH spin produces more massive-disk BHNS class 6 cells.

    Foucart 2018: smaller ISCO at higher spin -> less material falls
    in -> larger post-merger disk.
    """
    M_BH_axis = np.linspace(3.0, 15.0, 40)
    M_NS_axis = np.linspace(1.2, 2.2, 30)
    BHg, NSg = np.meshgrid(M_BH_axis, M_NS_axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls_low = classify_grid(BHg, NSg, ns_max=2.5, R_1p4_km=12.0, a_bh=0.0)
        cls_hi = classify_grid(BHg, NSg, ns_max=2.5, R_1p4_km=12.0, a_bh=0.9)
    n6_low = int((cls_low == 6).sum())
    n6_hi = int((cls_hi == 6).sum())
    assert n6_hi > n6_low, (
        f"High-spin grid landed {n6_hi} cells in class 6 vs {n6_low} "
        f"for a_bh = 0; Foucart 2018 monotonic spin dependence broken."
    )


# ---------------------------------------------------------------------------
# classify_formation_channels: partition, IV priority, V catch-all, II/III/I
# ---------------------------------------------------------------------------
def _channel_synthetic_table():
    """Six-row hand-built table covering all five Broekgaarden channels.

    Row 0: dblCE = 1 + CE + MT -> IV (priority).
    Row 1: stable MT first, CE later -> I.
    Row 2: stable MT only, no CE -> II.
    Row 3: CE first, stable MT later -> III.
    Row 4: dblCE = 1 only -> IV.
    Row 5: no MT, no CE, no dblCE -> V.
    """
    return dict(
        dblCE=np.array([1, 0, 0, 0, 1, 0]),
        fc_CEE=np.array([5.0, 5.0, 0.0, 2.0, 0.0, 0.0]),
        fc_mt_p1=np.array([3.0, 3.0, 4.0, 4.0, 0.0, 0.0]),
        fc_mt_s1=np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0]),
        fc_mt_p1_K1=np.array([0, 0, 0, 0, 0, 0]),
        fc_mt_s1_K2=np.array([0, 0, 0, 0, 0, 0]),
    )


def test_classify_formation_channels_partition_is_exclusive_and_complete():
    """Each system lands in exactly one of the five Broekgaarden channels.

    The implementation builds the masks in cascade with explicit
    ``~`` exclusions; this test catches a refactor that lets two
    channels overlap or leaves a system unclassified.
    """
    out = classify_formation_channels(**_channel_synthetic_table())
    stack = np.stack([out[k] for k in out], axis=0)
    counts = stack.sum(axis=0)
    assert (counts == 1).all(), f"Channel partition broken: per-row counts = {counts.tolist()}"


def test_dblCE_priority_routes_to_channel_IV():
    """A system with ``dblCE = 1`` always lands in Channel IV.

    Broekgaarden+ 2021 Sec. 5: double-core CE is a distinct channel
    that is checked first; the implementation enforces this priority
    by computing ``ch_IV = dblCE == 1`` before any other channel.
    """
    out = classify_formation_channels(**_channel_synthetic_table())
    for i in (0, 4):
        assert bool(out["IV  Double-core CE"][i]), i


def test_no_mt_and_no_ce_routes_to_channel_V():
    """A system with neither stable MT nor CE lands in Channel V (Other).

    Catch-all guard that prevents the cascade from leaving a row
    unclassified.
    """
    out = classify_formation_channels(**_channel_synthetic_table())
    assert bool(out["V   Other"][5])


def test_channel_II_when_stable_mt_and_no_ce():
    """Stable MT with no CE lands in Channel II (Stable MT only)."""
    out = classify_formation_channels(**_channel_synthetic_table())
    assert bool(out["II  Stable MT only"][2])


def test_channel_III_when_ce_precedes_stable_mt():
    """CE before stable MT lands in Channel III (Single-core CE).

    Row 3: ``fc_CEE = 2 < fc_mt_p1 = 4``, so the CE fires first.
    The cascade routes it to III rather than I.
    """
    out = classify_formation_channels(**_channel_synthetic_table())
    assert bool(out["III Single-core CE"][3])


def test_channel_I_when_stable_mt_precedes_ce():
    """Stable MT before CE lands in Channel I (Stable MT + CE).

    Row 1: ``fc_mt_p1 = 3 < fc_CEE = 5``, so the stable MT fires
    first.  The cascade routes it to I rather than III.
    """
    out = classify_formation_channels(**_channel_synthetic_table())
    assert bool(out["I  Stable MT + CE"][1])


# ---------------------------------------------------------------------------
# channel_class_crosstab: non-array key filter, zero weights
# ---------------------------------------------------------------------------
def test_channel_class_crosstab_filters_non_boolean_class_keys():
    """Diagnostic keys (e.g. ``'M_disk'``) are silently filtered out.

    The classifier dicts returned by ``classify_bhns`` and
    ``classify_observed_mergers`` carry float diagnostic arrays
    alongside the boolean class masks.  The crosstab must handle the
    mixed dict without raising or producing an extra column.
    """
    n = 4
    channels = {
        "I  Stable MT + CE": np.array([True, True, False, False]),
        "II  Stable MT only": np.array([False, False, True, True]),
        "III Single-core CE": np.zeros(n, dtype=bool),
        "IV  Double-core CE": np.zeros(n, dtype=bool),
        "V   Other": np.zeros(n, dtype=bool),
    }
    classes_with_diagnostic = {
        "sbGRB + blue KN": np.array([True, False, True, False]),
        "lbGRB + red KN (HMNS)": np.array([False, True, False, True]),
        "lbGRB + red KN (disk)": np.zeros(n, dtype=bool),
        "Faint lbGRB": np.zeros(n, dtype=bool),
        "M_disk": np.array([0.05, 0.001, 0.2, 0.005]),
    }
    weights = np.ones(n)
    df = channel_class_crosstab(channels, classes_with_diagnostic, weights)
    assert "M_disk" not in df.columns, (
        f"Diagnostic 'M_disk' key leaked into crosstab columns: {list(df.columns)}"
    )
    assert df.shape == (5, 4)


def test_channel_class_crosstab_zero_weights_returns_zero_matrix():
    """All-zero weights gives a zero crosstab; normalised paths return 0 not NaN."""
    n = 4
    channels = {
        "I": np.array([True, False, False, False]),
        "II": np.array([False, True, False, False]),
        "III": np.array([False, False, True, False]),
        "IV": np.array([False, False, False, True]),
        "V": np.zeros(n, dtype=bool),
    }
    classes = {
        "A": np.array([True, True, False, False]),
        "B": np.array([False, False, True, True]),
    }
    weights = np.zeros(n)
    raw = channel_class_crosstab(channels, classes, weights)
    assert (raw.values == 0.0).all()
    by_ch = channel_class_crosstab(channels, classes, weights, normalise="channel")
    assert not np.isnan(by_ch.values).any()
    assert (by_ch.values == 0.0).all()


# ---------------------------------------------------------------------------
# classify_observed_mergers: NaN, all-zero, threshold override, diagnostics
# ---------------------------------------------------------------------------
def test_classify_observed_mergers_nan_input_documented_quirk_lands_in_hmns():
    """A row with NaN ejecta lands in 'lbGRB + red KN (HMNS)' (documented quirk).

    The boolean cascade inside ``classify_observed_mergers`` does
    NOT propagate NaN as 'no class': ``M_ej = NaN`` causes both
    ``is_faint``, ``is_blue_dom`` and ``is_red_dom`` to evaluate False
    (any comparison with NaN is False under numpy), and the
    ``is_mixed = ~is_faint & ~is_blue_dom & ~is_red_dom`` catch-all
    then routes the row into the HMNS class.  The docstring claim
    about ``NaN class labels`` is misleading; the actual behaviour
    is locked in here as a regression sentinel rather than fixed
    inline.  Callers that need NaN-safe behaviour must guard
    upstream of ``classify_observed_mergers``.

    ``f_red`` itself is NaN as expected.
    """
    M_B = np.array([np.nan])
    M_P = np.array([np.nan])
    M_R = np.array([np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = classify_observed_mergers(M_B, M_P, M_R)
    assert np.isnan(out["M_ej_total"][0])
    assert np.isnan(out["f_red"][0])
    assert bool(out["lbGRB + red KN (HMNS)"][0]), (
        "NaN-input cascade behaviour has shifted; the catch-all "
        "is_mixed branch used to absorb NaN rows."
    )
    assert not bool(out["sbGRB + blue KN"][0])
    assert not bool(out["lbGRB + red KN (disk)"][0])
    assert not bool(out["Faint lbGRB"][0])


def test_classify_observed_mergers_all_zero_ejecta_lands_in_faint():
    """A source with ``M_B = M_P = M_R = 0`` is classified as Faint lbGRB.

    The docstring defines 'Faint lbGRB' by ``M_ej < KN_M_EJ_FAINT_MAX``;
    an exact-zero source sits comfortably below the 0.01 Msun floor.
    The 0/0 inside ``f_red`` raises a benign ``RuntimeWarning``
    which is suppressed locally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = classify_observed_mergers(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert bool(out["Faint lbGRB"][0])
    assert not bool(out["sbGRB + blue KN"][0])
    assert not bool(out["lbGRB + red KN (HMNS)"][0])


def test_classify_observed_mergers_threshold_override_changes_partition():
    """A per-call ``red_max_for_blue`` override actually reclassifies a row.

    Row with ``f_red = 0.4`` lands in the mixed class (HMNS) under
    the default ``red_max_for_blue = 0.30``; raising the threshold
    to 0.50 routes it to the blue-dominated class instead.  This
    pins the documented per-call sensitivity-knob behaviour.
    """
    M_B = np.array([0.06])  # f_red = 0.04 / 0.10 = 0.40
    M_P = np.array([0.0])
    M_R = np.array([0.04])
    out_default = classify_observed_mergers(M_B, M_P, M_R)
    out_relaxed = classify_observed_mergers(M_B, M_P, M_R, red_max_for_blue=0.50)
    assert bool(out_default["lbGRB + red KN (HMNS)"][0])
    assert not bool(out_default["sbGRB + blue KN"][0])
    assert bool(out_relaxed["sbGRB + blue KN"][0])
    assert not bool(out_relaxed["lbGRB + red KN (HMNS)"][0])


def test_classify_observed_mergers_diagnostic_keys_present():
    """``M_ej_total`` and ``f_red`` diagnostic keys are float arrays."""
    M_B = np.array([0.05, 0.01])
    M_P = np.array([0.01, 0.01])
    M_R = np.array([0.01, 0.06])
    out = classify_observed_mergers(M_B, M_P, M_R)
    for key in ("M_ej_total", "f_red"):
        assert key in out
        assert isinstance(out[key], np.ndarray)
        assert out[key].dtype == np.float64
        assert out[key].shape == M_B.shape


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
def test_GRID_CLASS_LABELS_covers_classify_grid_outputs():
    """``GRID_CLASS_LABELS`` has integer keys exactly equal to {1, ..., 6}.

    Class 0 is intentionally absent from the label map (it is
    reserved for background); classes 1 to 6 correspond to BNS (1-4)
    and BHNS (5-6) outputs of ``classify_grid``.  The Section 1
    colourbar uses this map directly; an orphan integer would
    silently mistint cells.
    """
    assert set(GRID_CLASS_LABELS.keys()) == {1, 2, 3, 4, 5, 6}
    bns_labels = {GRID_CLASS_LABELS[i] for i in (1, 2, 3, 4)}
    bhns_labels = {GRID_CLASS_LABELS[i] for i in (5, 6)}
    # BNS taxonomy line-up: every classify_bns_2024 key must appear in
    # the BNS label slice (modulo the "(BNS)" qualifier for class 1).
    expected_bns = {
        "Faint lbGRB (BNS)",
        "lbGRB + red KN (HMNS)",
        "sbGRB + blue KN",
        "lbGRB + red KN (BNS disk)",
    }
    assert bns_labels == expected_bns, (bns_labels, expected_bns)
    expected_bhns = {
        "Faint lbGRB (BHNS)",
        "lbGRB + red KN (BHNS disk)",
    }
    assert bhns_labels == expected_bhns, (bhns_labels, expected_bhns)


def test_KN_RED_FRAC_ordering_and_M_EJ_FAINT_alignment_with_MDISK_SHORT():
    """Kilonova red-fraction thresholds are ordered; faint floor matches MDISK_SHORT.

    ``classify_observed_mergers`` requires
    ``red_min_for_red > red_max_for_blue`` (raises otherwise); the
    defaults pin this ordering.  ``KN_M_EJ_FAINT_MAX = MDISK_SHORT
    = 0.01 Msun`` is the Gottlieb 2023 Sec. 4 alignment between the
    engine-fuel floor (BHNS disk-mass cut) and the kilonova-bright
    floor (observed-merger faint cut).
    """
    assert KN_RED_FRAC_BLUE_MAX < KN_RED_FRAC_RED_MIN
    assert KN_M_EJ_FAINT_MAX == MDISK_SHORT
    assert KN_M_EJ_FAINT_MAX == 0.01
    # And the upper red-fraction boundary should be symmetric about 0.5
    # given the project's documented choice (KN_RED_FRAC_RED_MIN ~ 0.70,
    # KN_RED_FRAC_BLUE_MAX ~ 0.30).  Pinning the symmetric centre keeps
    # the thresholds from drifting independently.
    assert (KN_RED_FRAC_BLUE_MAX + KN_RED_FRAC_RED_MIN) == pytest.approx(1.0, abs=0.05)
    # Unused-import guard for the pandas import that pandas-stub-y
    # frameworks would otherwise drop.
    _ = pd.DataFrame  # noqa: F841

    # Sanity guard: defaults of MDISK_SHORT / MDISK_LONG are within
    # the Gottlieb 2023 Sec. 4 quoted range.
    assert MDISK_SHORT < MDISK_LONG
    assert MDISK_LONG == 0.10

    # Tie back to M_THRESH > hmns_split sanity guard (cheap and pins
    # the BNS classifier's expected three-region structure even
    # under EOS sweeps).
    assert M_THRESH > HMNS_FACTOR_DEFAULT * M_TOV
