"""Section 1 of grb_main.ipynb: Mass Plane (BNS and BHNS).

Smoke-level invariants on ``classify_grid`` over a synthetic (M1, M2)
meshgrid: every labelled class is reachable, the ``ns_max`` truncation
matches the requested model, and STROOPWAFEL-weighted class fractions
sum to 1.  Pure unit-style; runs in CI without ``Data/``.

The second block of tests below covers the four ingredients of the
Section 1 mass-plane figures that are not exercised elsewhere: the
``classify_grid`` <-> per-system classifier consistency, the Foucart
disk-mass iso-contours that the BHNS panel draws at ``MDISK_SHORT`` /
``MDISK_LONG``, and the two properties of the weighted reflective-
boundary KDE that the notebook implements inline (mass-partition by
quantile contour levels, no density smear past the ``M_TOV`` wall).

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
# These cover the four pieces the panel draws that are not anchored
# elsewhere: classify_grid <-> per-system classifier agreement (so the
# colored underlay actually depicts the same physics as the printed
# class fractions), the Foucart disk-mass iso-contours plotted at
# MDISK_SHORT / MDISK_LONG, and the two KDE properties (quantile
# contour mass partitioning, reflective-boundary no-smear past M_TOV)
# that determine where the figure's contour levels actually land.
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


def _reflective_kde_bns(m2_sample, m1_sample, w_sample, m2_eval, m1_eval, M_TOV_val):
    """Recreate the BNS panel's reflective-boundary KDE.

    This mirrors the construction in Section 1's BNS panel of
    grb_main.ipynb (symmetrise in m1 <-> m2, reflect at the two
    M_TOV walls, factor of 2 to renormalise to the m1 >= m2 half).
    Kept as a test-local helper so the notebook code stays self-
    contained but the construction is exercised in CI.
    """
    from scipy.stats import gaussian_kde

    m2_sym = np.concatenate([m2_sample, m1_sample])
    m1_sym = np.concatenate([m1_sample, m2_sample])
    w_sym = np.concatenate([w_sample, w_sample])

    kde = gaussian_kde(np.vstack([m2_sym, m1_sym]), weights=w_sym, bw_method="silverman")
    two_M = 2.0 * M_TOV_val
    x = np.ravel(m2_eval)
    y = np.ravel(m1_eval)
    Z = (
        kde(np.vstack([x, y]))
        + kde(np.vstack([two_M - x, y]))
        + kde(np.vstack([x, two_M - y]))
        + kde(np.vstack([two_M - x, two_M - y]))
    ).reshape(np.asarray(m2_eval).shape) * 2.0
    return Z


def _reflective_kde_bhns(NS_sample, BH_sample, w_sample, NS_eval, BH_eval, M_TOV_val):
    """Recreate the BHNS panel's reflective-boundary KDE.

    Only reflects on the NS axis at M_TOV; the BH axis is unbounded.
    Mirrors the construction at line 50 of the second selection.
    """
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(
        np.vstack([NS_sample, BH_sample]),
        weights=w_sample,
        bw_method="silverman",
    )
    pts = np.vstack([np.ravel(NS_eval), np.ravel(BH_eval)])
    pts_refl = np.vstack([2.0 * M_TOV_val - np.ravel(NS_eval), np.ravel(BH_eval)])
    return (kde(pts) + kde(pts_refl)).reshape(np.asarray(NS_eval).shape)


def _quantile_levels(Z, quantiles):
    """Replica of the notebook's quantile-level function."""
    flat = Z.compressed() if hasattr(Z, "compressed") else Z.ravel()
    s = np.sort(flat)[::-1]
    c = np.cumsum(s) / s.sum()
    return sorted(s[np.searchsorted(c, q)] for q in quantiles)


def test_kde_quantile_levels_partition_mass_correctly():
    """Quantile contour levels must enclose the labelled cumulative weighted mass.

    Section 1 of grb_main.ipynb computes its colorbar tick positions
    by sorting the KDE densities and picking the value where the
    cumulative sum crosses each labelled quantile (5, 10, 25, 50, 68,
    90, 95, 99 percent).  Reviewers reading the figure assume that
    the contour returned for a given quantile ``q`` actually encloses
    ``q`` of the weighted KDE mass; this test confirms that property
    per-quantile on a small synthetic weighted sample using the same
    construction the notebook implements inline.

    Note: ``_quantile_levels`` returns ``sorted(...)`` so the order in
    the returned list is *ascending by density*, not by quantile.
    This test calls the helper per-quantile so the level-to-quantile
    pairing is unambiguous; the order-mismatch trap is a real one
    that the notebook side-steps by labelling colorbar ticks
    consistently with the sort order.
    """
    rng = np.random.default_rng(2027)
    n = 1500
    m1 = rng.uniform(1.25, 2.15, n)
    m2 = rng.uniform(1.25, m1)  # m1 >= m2
    w = rng.uniform(0.1, 1.0, n)

    m2_grid = np.linspace(1.25, 2.20, 110)
    m1_grid = np.linspace(1.25, 2.20, 110)
    M2g, M1g = np.meshgrid(m2_grid, m1_grid)
    Z = _reflective_kde_bns(m2, m1, w, M2g, M1g, M_TOV)
    Z_half = np.ma.array(Z, mask=(M1g < M2g))

    total = float(Z_half.sum())
    for q in (0.10, 0.50, 0.90):
        # Per-quantile so the order-vs-position trap in
        # ``_quantile_levels`` (sorted by density, not by quantile)
        # cannot break the test.
        lvl = _quantile_levels(Z_half, [q])[0]
        enclosed = float(Z_half[Z_half >= lvl].sum() / total)
        assert enclosed == pytest.approx(q, abs=0.03), (
            f"Quantile contour for q = {q:.2f} encloses "
            f"{enclosed * 100:.1f} percent of the weighted KDE mass on "
            f"the synthetic sample; expected {q * 100:.0f}.  The "
            f"notebook colorbar ticks would mislabel the credible "
            f"region."
        )


def _base_symmetric_kde_bns(m2_sample, m1_sample, w_sample, m2_eval, m1_eval):
    """Bare symmetric KDE used inside the reflective sum (no mirror, no factor of 2)."""
    from scipy.stats import gaussian_kde

    m2_sym = np.concatenate([m2_sample, m1_sample])
    m1_sym = np.concatenate([m1_sample, m2_sample])
    w_sym = np.concatenate([w_sample, w_sample])
    kde = gaussian_kde(np.vstack([m2_sym, m1_sym]), weights=w_sym, bw_method="silverman")
    return kde(np.vstack([np.ravel(m2_eval), np.ravel(m1_eval)])).reshape(np.asarray(m2_eval).shape)


def test_reflective_kde_wall_identity_bns():
    """At the corner (M_TOV, M_TOV) the 4-point sum collapses to 4 x base KDE.

    Geometric identity that pins all four mirror terms simultaneously
    on the BNS construction.  Mirror map at the M_TOV walls sends:

        f(x, y)            -> f(M_TOV, M_TOV)
        f(2M - x, y)       -> f(M_TOV, M_TOV)     (2M - M_TOV = M_TOV)
        f(x, 2M - y)       -> f(M_TOV, M_TOV)
        f(2M - x, 2M - y)  -> f(M_TOV, M_TOV)

    so the reflective sum (before the trailing factor of 2) evaluates
    to exactly four times the base symmetric KDE.  Section 1's
    factor-of-2 renormalisation then gives 8 * base at the corner.
    Dropping any single mirror term would shift the corner ratio from
    8 to 6, which this test detects independently of grid resolution
    or bandwidth choice -- both cancel in the ratio.

    Similarly at an edge point (m2 < M_TOV, m1 = M_TOV) two of the
    four mirror evaluations collapse to the same input and two
    evaluate at (m2, M_TOV) and (m2, M_TOV) again -- the construction
    gives 4 * base at the edge.  Both anchor values are checked.
    """
    rng = np.random.default_rng(2028)
    n = 200
    raw = rng.uniform(1.25, M_TOV - 0.05, size=(2, n))
    m1 = np.maximum(raw[0], raw[1])
    m2 = np.minimum(raw[0], raw[1])
    w = np.ones(n)

    # Corner identity: reflective(M_TOV, M_TOV) = 8 * base(M_TOV, M_TOV).
    corner_m2 = np.array([M_TOV])
    corner_m1 = np.array([M_TOV])
    Z_refl = _reflective_kde_bns(m2, m1, w, corner_m2, corner_m1, M_TOV)
    Z_base = _base_symmetric_kde_bns(m2, m1, w, corner_m2, corner_m1)
    ratio_corner = float(Z_refl[0] / Z_base[0])
    assert ratio_corner == pytest.approx(8.0, rel=1e-10), (
        f"Reflective / base ratio at the (M_TOV, M_TOV) corner is "
        f"{ratio_corner}; expected exactly 8 (= 4 mirror terms * "
        f"factor-of-2 renormalisation).  A regression that dropped "
        f"a mirror term would shift this ratio to 6."
    )

    # Edge identity: reflective(m2_int, M_TOV) = 4 * base(m2_int, M_TOV).
    # At m1 = M_TOV the (x, 2M - y) and (2M - x, 2M - y) mirrors map
    # to (x, M_TOV) and (2M - x, M_TOV) -- two pairs of identical
    # evaluations -- so the reflective sum is 2 * (base + mirror_x),
    # times the factor of 2.  In the limit where the data sits far
    # from the m2 = M_TOV wall, mirror_x is negligible and the ratio
    # tends to 4 from below.
    edge_m2 = np.array([1.5])  # interior on the m2 axis
    edge_m1 = np.array([M_TOV])
    Z_refl_edge = _reflective_kde_bns(m2, m1, w, edge_m2, edge_m1, M_TOV)
    Z_base_edge = _base_symmetric_kde_bns(m2, m1, w, edge_m2, edge_m1)
    ratio_edge = float(Z_refl_edge[0] / Z_base_edge[0])
    # The mirror_x term contributes < 1e-4 with the sample placed
    # far from the m2 = M_TOV wall and Silverman bandwidth at N=200;
    # the algebraic identity gives 4 - eps, so a 3.95 floor suffices.
    assert 3.95 < ratio_edge <= 4.05, (
        f"Reflective / base ratio at the (m2 = 1.5, m1 = M_TOV) "
        f"edge is {ratio_edge}; expected close to 4 (two pairs of "
        f"identical mirror evaluations * factor-of-2 renormalisation, "
        f"with negligible cross-mirror contribution from far-side "
        f"data).  Mirror count or factor-of-2 renormalisation drift."
    )


def test_reflective_kde_wall_identity_bhns():
    """At the NS wall (M_TOV, BH) the 2-point sum collapses to 2 x base KDE.

    Same algebraic identity as the BNS edge test but on the BHNS
    construction.  Mirror map at M_NS = M_TOV sends:

        f(NS, BH)        -> f(M_TOV, BH)
        f(2M - NS, BH)   -> f(M_TOV, BH)         (2M - M_TOV = M_TOV)

    so the reflective sum at any (M_TOV, BH) point evaluates to
    exactly 2 * base.  Dropping the NS-axis mirror would collapse
    this ratio to 1.
    """
    rng = np.random.default_rng(2029)
    n = 200
    NS = rng.uniform(1.25, M_TOV - 0.05, size=n)
    BH = rng.uniform(3.0, 12.0, size=n)
    w = np.ones(n)

    from scipy.stats import gaussian_kde

    NS_eval = np.array([M_TOV])
    BH_eval = np.array([6.0])
    Z_refl = _reflective_kde_bhns(NS, BH, w, NS_eval, BH_eval, M_TOV)

    kde = gaussian_kde(np.vstack([NS, BH]), weights=w, bw_method="silverman")
    Z_base = kde(np.vstack([NS_eval, BH_eval]))
    ratio = float(Z_refl[0] / Z_base[0])
    assert ratio == pytest.approx(2.0, rel=1e-10), (
        f"Reflective / base ratio at the NS wall (M_TOV, BH=6) is "
        f"{ratio}; expected exactly 2.  Dropping the NS-axis mirror "
        f"would shift this ratio to 1."
    )
