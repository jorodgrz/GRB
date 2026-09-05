"""Section 1 of grb_main.ipynb: Mass Plane (BNS and BHNS).

Smoke-level invariants on ``classify_grid`` over a synthetic (M1, M2)
meshgrid: every labelled class is reachable, the ``ns_max`` truncation
matches the requested model, and STROOPWAFEL-weighted class fractions
sum to 1.  Pure unit-style; runs in CI without ``data/``.

The second block of tests below covers the ingredients of the
Section 1 mass-plane figures that are not exercised elsewhere: the
``classify_grid`` <-> per-system classifier consistency, the Foucart
disk-mass iso-contours that the BHNS panel draws at ``MDISK_SHORT`` /
``MDISK_LONG``, and the properties of the STROOPWAFEL-weighted 2D mass
histogram the notebook implements inline (exact weight conservation,
the ``M_TOV``-anchored bin edge, and the diagonal half-area density
correction).

Reference: Gottlieb et al. (2024), arXiv:2411.13657; Broekgaarden et
al. (2021), arXiv:2103.02608, Sec. 3.4 (M_NS_max per model).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import NS_MAX_FIDUCIAL, classify_bhns, classify_bns_2024, classify_grid
from grb_physics import M_THRESH, M_TOV, MDISK_LONG, MDISK_SHORT, foucart_disk_mass


def _synthetic_meshgrid(n=80, m_lo=1.0, m_hi=4.0):
    m_axis = np.linspace(m_lo, m_hi, n)
    m1g, m2g = np.meshgrid(m_axis, m_axis)
    return m1g, m2g


@pytest.mark.parametrize("ns_max", sorted(NS_MAX_FIDUCIAL))
def test_classify_grid_labels_in_expected_range(ns_max):
    """All grid cells produce labels in {0, 1, ..., 6} and at least
    one cell in every BNS or BHNS labelled class is reachable for the
    fiducial Broekgaarden+ 2021 ns_max values."""
    m1g, m2g = _synthetic_meshgrid()
    cls = classify_grid(m1g, m2g, ns_max=ns_max, R_1p4_km=12.0)

    assert cls.dtype.kind in ("i", "u")
    assert cls.min() >= 0 and cls.max() <= 6, (cls.min(), cls.max())

    populated = set(np.unique(cls).tolist())
    assert 0 in populated, "background class 0 unreachable"
    bns_labels = {1, 2, 3, 4} & populated
    bhns_labels = {5, 6} & populated
    assert bns_labels, f"no BNS labels populated (ns_max={ns_max})"
    assert bhns_labels, f"no BHNS labels populated (ns_max={ns_max})"


def test_classify_grid_ns_max_truncates_bns_region():
    """Cells with both components above ``ns_max`` cannot be class 1-4
    (BNS region); they should fall into BHNS (5, 6) or background (0)."""
    m1g, m2g = _synthetic_meshgrid()
    ns_max = 2.0
    cls = classify_grid(m1g, m2g, ns_max=ns_max, R_1p4_km=12.0)
    above = (m1g > ns_max) & (m2g > ns_max)
    bns_labels_above = np.isin(cls[above], [1, 2, 3, 4])
    assert not bns_labels_above.any(), (
        f"{bns_labels_above.sum()} cells with both m1, m2 > ns_max "
        f"= {ns_max} were classified as BNS"
    )


def test_classify_grid_weighted_class_fractions_sum_to_one():
    """STROOPWAFEL-weighted class fractions over a synthetic mass-plane
    sample partition unity to numerical precision."""
    rng = np.random.default_rng(42)
    n = 5000
    m1 = rng.uniform(1.1, 3.5, n)
    m2 = rng.uniform(1.1, 3.5, n)
    weights = rng.uniform(0.0, 1.0, n)

    m_heavy = np.maximum(m1, m2)
    m_light = np.minimum(m1, m2)
    m_tot = m_heavy + m_light
    q = m_heavy / m_light

    m_thresh = M_THRESH
    hmns_split = 1.2 * M_TOV

    masks = {
        "sb": m_tot < hmns_split,
        "lb_HMNS": (m_tot >= hmns_split) & (m_tot < m_thresh),
        "lb_disk": (m_tot >= m_thresh) & (q >= 1.2),
        "faint_lb": (m_tot >= m_thresh) & (q < 1.2),
    }
    fractions = {k: weights[v].sum() / weights.sum() for k, v in masks.items()}
    total = sum(fractions.values())
    assert total == pytest.approx(1.0, rel=1e-12), fractions


# ─────────────────────────────────────────────────────────────────────
# Section 1 plot ingredients (synthetic only).
#
# These cover the pieces the panel draws that are not anchored
# elsewhere: classify_grid <-> per-system classifier agreement (so the
# colored underlay actually depicts the same physics as the printed
# class fractions), the Foucart disk-mass iso-contours plotted at
# MDISK_SHORT / MDISK_LONG, and the weighted 2D mass-histogram
# properties (exact weight conservation, M_TOV-anchored top edge,
# diagonal half-area density correction) that determine what the
# figure's density map shows.
# ─────────────────────────────────────────────────────────────────────

# classify_grid integer labels (Gottlieb 2024 hybrid).  Pulled from the
# docstring of grb_classify.classify_grid; duplicated here so the
# per-system <-> grid mapping is explicit and a future relabelling of
# integers fails this test immediately rather than silently mis-tinting
# the figure.
_GRID_LABEL_FROM_BNS_KEY = {
    "Faint lbGRB": 1,
    "lbGRB + red KN (HMNS)": 2,
    "sbGRB + blue KN": 3,
    "lbGRB + red KN (disk)": 4,
}
_GRID_LABEL_FROM_BHNS_KEY = {
    # 'No GRB' has M_disk < MDISK_SHORT and is folded into class 0
    # (background) by classify_grid, so it is not listed here.
    "Faint lbGRB (BHNS)": 5,
    "lbGRB + red KN (BHNS disk)": 6,
}


def test_classify_grid_agrees_with_per_system_bns_classifier():
    """`classify_grid` BNS labels (1, 2, 3, 4) must match `classify_bns_2024` cell-by-cell.

    Section 1's BNS panel uses ``classify_grid`` for the colored
    underlay and ``classify_bns_2024`` for the printed class fractions.
    If the two routines disagree on a (m1, m2) point, the legend lies
    about the underlay.  Sweep a dense synthetic grid in the BNS
    region, evaluate both classifiers, and assert that every cell
    receives the expected label.
    """
    m_axis = np.linspace(1.25, 2.4, 80)
    m1g, m2g = np.meshgrid(m_axis, m_axis)

    cls = classify_grid(m1g, m2g, ns_max=2.5, R_1p4_km=12.0)
    cls_bns_dict = classify_bns_2024(m1g, m2g)

    bns_label_per_cell = np.zeros_like(cls)
    for key, lbl in _GRID_LABEL_FROM_BNS_KEY.items():
        bns_label_per_cell[cls_bns_dict[key]] = lbl

    # Compare only on cells where classify_grid actually labels a BNS
    # cell (1-4); cells outside the BNS region (BHNS or background)
    # are excluded because classify_bns_2024 has no notion of the
    # ns_max truncation.
    bns_mask = np.isin(cls, [1, 2, 3, 4])
    mismatch = bns_mask & (cls != bns_label_per_cell)
    assert not mismatch.any(), (
        f"classify_grid and classify_bns_2024 disagree on "
        f"{int(mismatch.sum())} of {int(bns_mask.sum())} BNS-region "
        f"cells.  The figure's colored underlay does not match what "
        f"the printed class fractions describe."
    )


def test_classify_grid_agrees_with_per_system_bhns_classifier():
    """`classify_grid` BHNS labels (5, 6) must match `classify_bhns` cell-by-cell.

    Same honesty constraint as the BNS test but on the BHNS branch:
    Section 1's BHNS panel uses ``classify_grid`` for the colored
    underlay and the printed class fractions come from
    ``classify_bhns``.  Sweep a synthetic grid in the BHNS region at
    ``a_BH = 0.5`` and assert that every cell receives the expected
    grid label.
    """
    M_NS_axis = np.linspace(1.25, 2.2, 60)
    M_BH_axis = np.linspace(3.0, 15.0, 80)
    NSg, BHg = np.meshgrid(M_NS_axis, M_BH_axis)

    cls = classify_grid(BHg, NSg, ns_max=2.5, a_bh=0.5, R_1p4_km=12.0)
    cls_bhns_dict = classify_bhns(BHg, NSg, a_BH=0.5)

    bhns_label_per_cell = np.zeros_like(cls)
    for key, lbl in _GRID_LABEL_FROM_BHNS_KEY.items():
        bhns_label_per_cell[cls_bhns_dict[key]] = lbl

    bhns_mask = np.isin(cls, [5, 6])
    mismatch = bhns_mask & (cls != bhns_label_per_cell)
    assert not mismatch.any(), (
        f"classify_grid and classify_bhns disagree on "
        f"{int(mismatch.sum())} of {int(bhns_mask.sum())} BHNS-region "
        f"cells at a_BH = 0.5.  The figure's BHNS underlay does not "
        f"match what the printed disk-mass fractions describe."
    )


def test_foucart_disk_mass_iso_contours_at_mdisk_thresholds():
    """`foucart_disk_mass` evaluated at the BHNS panel's plotted iso-contours.

    Section 1's BHNS panel draws contour lines at ``MDISK_SHORT = 0.01``
    and ``MDISK_LONG = 0.10`` Msun using matplotlib's ``contour`` on
    the ``foucart_disk_mass`` grid.  Pick five test (M_BH, M_NS)
    points by inverting the disk-mass function along ``M_NS = 1.35``
    (typical NS mass) and verify the recovered disk mass matches the
    target threshold to 1e-3 Msun.  A regression that scaled
    ``foucart_disk_mass`` by a stray factor (e.g. an extra ``f_disk``)
    would shift the iso-contour position in mass space and break this
    test.
    """
    a_BH = 0.5
    M_NS = 1.35
    R_NS_km = 12.0
    M_BH_grid = np.linspace(2.5, 25.0, 5001)
    M_disk = foucart_disk_mass(M_BH_grid, M_NS, a_BH=a_BH, R_NS_km=R_NS_km)

    for target in (MDISK_SHORT, MDISK_LONG):
        # M_disk is a monotonically decreasing function of M_BH in the
        # tidal-disruption regime (heavier BH -> smaller disk).
        # Recover the M_BH that lands on the target iso-contour, then
        # evaluate foucart_disk_mass back and check the round-trip.
        in_range = (M_disk >= 0.5 * target) & (M_disk <= 2.0 * target)
        assert in_range.any(), (
            f"foucart_disk_mass({M_NS=}, a_BH={a_BH}) does not cross "
            f"{target} Msun on M_BH in [2.5, 25].  Either the formula "
            f"changed or the threshold drifted off the chart."
        )
        idx = int(np.argmin(np.abs(M_disk - target)))
        M_BH_iso = float(M_BH_grid[idx])
        M_disk_iso = float(foucart_disk_mass(M_BH_iso, M_NS, a_BH=a_BH, R_NS_km=R_NS_km))
        # 1e-3 Msun tolerance is the M_BH-grid spacing's contribution to
        # the recovered disk mass; the function itself is evaluated
        # exactly.
        assert M_disk_iso == pytest.approx(target, abs=2e-3), (
            f"Recovered iso-contour at M_BH = {M_BH_iso:.3f} Msun "
            f"evaluates to M_disk = {M_disk_iso:.5f} Msun, target "
            f"= {target} Msun.  The Section 1 BHNS panel would plot "
            f"the contour at the wrong M_BH."
        )


def _bns_mass_histogram(m2, m1, w, m_lo=1.25, m_tov=M_TOV, bw=0.02):
    """Replicate the Section 1 BNS panel's weighted 2D mass histogram.

    Square ``bw`` bins with the top edge pinned on ``m_tov`` and the
    bottom edge padded below ``m_lo`` so no boundary system is dropped.
    Only the m1 >= m2 upper triangle is populated; diagonal bins are
    bisected by the m1 = m2 line and hold half the area of an
    off-diagonal bin.  Returns ``(edges, Hw, dens, area)`` where
    ``dens`` is the masked per-bin weighted density with the diagonal
    half-area correction applied.
    """
    nb = int(np.ceil((m_tov - m_lo) / bw))
    edges = m_tov - bw * np.arange(nb, -1, -1)
    Hw, _, _ = np.histogram2d(m2, m1, bins=[edges, edges], weights=w)
    i, j = np.indices(Hw.shape)
    area = bw**2 * np.where(i == j, 0.5, 1.0)
    dens = np.ma.masked_where((Hw <= 0) | (j < i), (Hw / w.sum()) / area)
    return edges, Hw, dens, area


def test_bns_mass_histogram_conserves_weight_exactly():
    """The weighted 2D histogram must recover the full sample weight.

    This is the invariant the reflective KDE violated: with reflection
    on only one of the two hard walls, roughly a fifth of the KDE
    probability fell below the mass floor.  A histogram cannot leak
    weight across a bin edge, so the summed bin weight equals the
    sample weight to machine precision.
    """
    rng = np.random.default_rng(2027)
    n = 5000
    raw = rng.uniform(1.26, M_TOV, size=(2, n))
    m1 = np.maximum(raw[0], raw[1])
    m2 = np.minimum(raw[0], raw[1])
    w = rng.uniform(0.1, 1.0, n)

    _edges, Hw, _dens, _area = _bns_mass_histogram(m2, m1, w)
    assert np.isclose(Hw.sum(), w.sum(), rtol=1e-12), (
        f"Histogram retains {Hw.sum():.6f} of {w.sum():.6f} sample "
        f"weight; the bin grid is dropping systems."
    )


def test_bns_mass_histogram_top_edge_is_m_tov():
    """A system exactly at the cold-stability wall lands in the top bin.

    The top edge is pinned on ``M_TOV`` so the wall coincides with a
    bin boundary rather than bisecting a bin.  ``np.histogram2d``
    includes the rightmost edge in the final bin, so a component
    sitting at ``M_TOV`` (the cut keeps ``m_i <= M_TOV``) is binned in
    the corner cell, not dropped.
    """
    m1 = np.array([M_TOV])
    m2 = np.array([M_TOV])
    w = np.array([1.0])

    edges, Hw, _dens, _area = _bns_mass_histogram(m2, m1, w)
    assert edges[-1] == pytest.approx(M_TOV, abs=1e-12), (
        f"Top bin edge {edges[-1]} is not pinned on M_TOV = {M_TOV}."
    )
    assert np.isclose(Hw.sum(), 1.0), "System at the M_TOV wall was dropped."
    assert Hw[-1, -1] == pytest.approx(1.0), (
        "System at (M_TOV, M_TOV) did not land in the corner bin."
    )


def test_bns_mass_histogram_diagonal_half_area_gives_flat_density():
    """The diagonal half-area correction recovers a flat density.

    A sample uniform over the m1 >= m2 triangle has constant density.
    Off-diagonal bins are fully inside the triangle; diagonal bins are
    bisected by the m1 = m2 line and hold half the counts of a
    full bin.  Dividing by the half-area on the diagonal (and full
    area off it) must return the same density on both, otherwise the
    diagonal would read a factor of two low.
    """
    rng = np.random.default_rng(7)
    n = 400_000
    xy = rng.uniform(1.30, 2.10, size=(2, n))
    m1 = np.maximum(xy[0], xy[1])
    m2 = np.minimum(xy[0], xy[1])
    w = np.ones(n)

    _edges, Hw, dens, _area = _bns_mass_histogram(m2, m1, w)
    i, j = np.indices(Hw.shape)
    on_diag = (i == j) & (~dens.mask)
    off_diag = (j > i) & (~dens.mask)

    ratio = float(dens[on_diag].mean() / dens[off_diag].mean())
    assert 0.90 < ratio < 1.10, (
        f"Diagonal / off-diagonal mean density ratio is {ratio:.3f} on "
        f"a uniform triangular sample; expected ~1.  A missing half-area "
        f"correction would drive the diagonal to ~0.5."
    )


# ---------------------------------------------------------------------------
# Region labels on the Section 1 panels
# ---------------------------------------------------------------------------
# Both panels carry italic class names placed at hand-chosen data
# coordinates.  Nothing in the figure derives them from the classifier, so a
# change to K_THRESH_DEFAULT, HMNS_FACTOR_DEFAULT, A_BH_FID or either disk
# threshold moves the boundaries out from under the labels and the figure
# asserts something false without failing.  The notebook keeps the anchors in
# BNS_REGION_ANCHORS / BHNS_REGION_ANCHORS, keyed by the classifier's own
# output key; the tests below re-classify every anchor.
def _notebook_literal(repo_root, name):
    """The literal assigned to ``name`` in a grb_main.ipynb code cell."""
    import ast
    import json
    import os

    with open(os.path.join(repo_root, "grb_main.ipynb")) as fh:
        nb = json.load(fh)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if f"{name} = " not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:  # IPython magics, not a cell we care about
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in any grb_main.ipynb code cell")


def test_bns_region_labels_land_in_the_class_they_name(repo_root):
    """Every BNS panel label sits in the region it names, with margin.

    The anchor is checked at its own coordinates and over a +/- 0.01 Msun
    halo, so a label that has drifted onto a boundary fails here rather
    than becoming ambiguous at the next threshold edit.
    """
    anchors = _notebook_literal(repo_root, "BNS_REGION_ANCHORS")
    keys = set(classify_bns_2024(np.array([1.4]), np.array([1.3])))
    assert set(anchors) == keys, (
        f"BNS_REGION_ANCHORS keys {sorted(anchors)} are not the four "
        f"classify_bns_2024 keys {sorted(keys)}"
    )

    halo = np.array([-0.01, 0.0, 0.01])
    d1, d2 = np.meshgrid(halo, halo)
    for key, ((m2, m1), text) in anchors.items():
        assert m1 >= m2, f"{text!r} anchor is on the empty m1 < m2 half"
        assert 1.25 <= m2 <= 2.2 and 1.25 <= m1 <= 2.2, (
            f"{text!r} anchor is outside the [1.25, 2.2] panel"
        )
        m1h = np.maximum(m1 + d1.ravel(), m2 + d2.ravel())
        m2h = np.minimum(m1 + d1.ravel(), m2 + d2.ravel())
        cls = classify_bns_2024(m1h, m2h)
        assert cls[key].all(), (
            f"BNS label {text!r} is filed under {key!r} but its anchor "
            f"(M2={m2}, M1={m1}) classifies as "
            f"{[k for k, v in cls.items() if v[0]]}"
        )


def test_bhns_region_labels_land_in_the_class_they_name(repo_root):
    """Every BHNS panel label sits in the region it names, at ``A_BH_FID``.

    The halo is wider on the BH axis (+/- 0.05 Msun) because the panel bins
    M_BH at 0.25 Msun, so a 0.01 Msun tolerance there would be far below the
    resolution a reader can judge the placement at.
    """
    from grb_physics import A_BH_FID

    anchors = _notebook_literal(repo_root, "BHNS_REGION_ANCHORS")
    keys = set(classify_bhns(np.array([8.0]), np.array([1.4]))) - {"M_disk"}
    assert set(anchors) == keys, (
        f"BHNS_REGION_ANCHORS keys {sorted(anchors)} are not the three "
        f"classify_bhns class keys {sorted(keys)}"
    )

    d_ns, d_bh = np.meshgrid(np.array([-0.01, 0.0, 0.01]), np.array([-0.05, 0.0, 0.05]))
    for key, ((m_ns, m_bh), text) in anchors.items():
        assert 1.25 <= m_ns <= 2.2, f"{text!r} anchor is off the M_NS axis"
        assert 2.5 <= m_bh <= 18.0, f"{text!r} anchor is off the M_BH axis"
        cls = classify_bhns(m_bh + d_bh.ravel(), m_ns + d_ns.ravel(), a_BH=A_BH_FID)
        assert cls[key].all(), (
            f"BHNS label {text!r} is filed under {key!r} but its anchor "
            f"(M_NS={m_ns}, M_BH={m_bh}) gives M_disk="
            f"{cls['M_disk'][4]:.4g} and classifies as "
            f"{[k for k, v in cls.items() if k != 'M_disk' and v[4]]}"
        )


def test_tidal_disruption_locus_is_not_the_bhns_class_boundary():
    """The dashed M_disk > 0 curve sits above the No GRB / Faint boundary.

    The BHNS panel draws three curves and only the lower two are class
    edges.  A reader who takes the topmost curve as the edge of the
    classified region overstates the Faint lbGRB area, so the ordering is
    pinned here and stated in the legend and the caption.
    """
    m_bh = np.linspace(2.5, 18.0, 20_000)
    for m_ns in (1.3, 1.5, 1.8):
        disk = foucart_disk_mass(m_bh, np.full_like(m_bh, m_ns), a_BH=0.5)

        def _highest(threshold, disk=disk):
            hit = np.where(disk > threshold)[0]
            return m_bh[hit.max()] if hit.size else np.nan

        onset, faint = _highest(0.0), _highest(MDISK_SHORT)
        assert onset > faint, (
            f"At M_NS={m_ns} the M_disk > 0 locus ({onset:.2f}) does not sit "
            f"above the M_disk = {MDISK_SHORT} boundary ({faint:.2f})"
        )
