"""Real-data audit for Section 1 of ``grb_main.ipynb`` (BNS + BHNS mass-plane panels).

The two Section 1 panels are the project's headline figures: the
Gottlieb (2024) classifier underlay on the (M2, M1) BNS plane and on
the (M_NS, M_BH) BHNS plane, with a STROOPWAFEL-weighted reflective-
boundary Gaussian KDE on top.  The tests in this file lock the
end-to-end pipeline that feeds those panels against the COMPAS Model A
HDF5 files: file identity, NS-mass remap determinism + M_TOV cap +
``m1 >= m2`` invariant, KDE weighting honesty on the real sample, and
the printed class fractions the notebook's summary cell emits.

All tests are ``@pytest.mark.requires_data`` and skip on machines
without ``data/COMPASCompactOutput_{BNS,BHNS}_A.h5``.  The expected
class-fraction targets in
``test_printed_class_fractions_match_classifier_on_remapped_sample``
come from the latest committed run of ``grb_main.ipynb`` (cell at the
top of the notebook); update them only if a deliberate science change
moves the printed numbers.

Reference: Broekgaarden et al. (2021) arXiv:2103.02608 (Model A
configuration), Alsing, Silva and Berti (2018) MNRAS 478, 1377
(NS-mass remap target), Gottlieb et al. (2024) arXiv:2411.13657
(BNS classification), Foucart et al. (2018) PRD 98, 081501 (BHNS
disk-mass classification).
"""

from __future__ import annotations

import h5py as h5
import numpy as np
import pytest

from grb_classify import NS_MAX_FIDUCIAL, classify_bhns, classify_bns_2024
from grb_io import (
    load_bhns_with_channels,
    load_bns_with_channels,
)
from grb_physics import (
    M_TOV,
    remap_ns_marginal,
    remap_ns_masses_double_gaussian,
)

# ─────────────────────────────────────────────────────────────────────
# Section 1 panel extents (must match the literals in ``grb_main.ipynb``).
# Duplicated here so a regression in the notebook breaks this test
# rather than silently shifting the panel limits.
# ─────────────────────────────────────────────────────────────────────
_NS_MAX_BNS = 2.5  # Broekgaarden+ 2021 Model A M_NS_max
_M2_LO_BNS, _M2_HI_BNS = 1.25, 2.2
_M1_LO_BNS, _M1_HI_BNS = 1.25, 2.2

# Class-fraction targets from the latest grb_main.ipynb run.  These
# come from the printed cell output at the top of the notebook
# (see file header for the exact strings).
_EXPECTED_BNS_FRACTIONS_PCT = {
    "sbGRB + blue KN": 24.0,
    "lbGRB + red KN (HMNS)": 26.8,
    "lbGRB + red KN (disk)": 2.2,
    "Faint lbGRB": 47.0,
}
_EXPECTED_BHNS_FRACTIONS_PCT = {
    "lbGRB + red KN (BHNS disk)": 3.8,
    "Faint lbGRB (BHNS)": 20.6,
    "No GRB": 75.6,
}


@pytest.fixture(scope="module")
def bns_modelA_remapped(bns_a_path):
    """Section 0's BNS load + Alsing remap (RNG seed 42), keyed off the real HDF5."""
    bns = load_bns_with_channels(path=bns_a_path)
    m1, m2 = remap_ns_masses_double_gaussian(
        bns["m1"].copy(),
        bns["m2"].copy(),
        weights=bns["weights"],
        rng=np.random.default_rng(42),
    )
    return {
        "m1": m1,
        "m2": m2,
        "weights": bns["weights"],
        "model": bns.get("model"),
        "ns_max": bns.get("ns_max"),
    }


@pytest.fixture(scope="module")
def bhns_modelA_remapped(bhns_a_path):
    """Section 0's BHNS load + Alsing remap of M_NS (RNG seed 43)."""
    bhns = load_bhns_with_channels(path=bhns_a_path)
    M_NS = remap_ns_marginal(bhns["M_NS"], weights=bhns["weights"], rng=np.random.default_rng(43))
    return {
        "M_BH": bhns["M_BH"],
        "M_NS": M_NS,
        "weights": bhns["weights"],
        "model": bhns.get("model"),
        "ns_max": bhns.get("ns_max"),
    }


# ─────────────────────────────────────────────────────────────────────
# File identity: catch a future ``data/`` swap to a different model.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_bns_a_metadata_matches_modelA(bns_a_path):
    """``load_bns_with_channels`` with ``expected_model='A'`` accepts Model A HDF5.

    Section 0 of ``grb_main.ipynb`` does not currently pass
    ``expected_model='A'`` / ``expected_ns_max=2.5`` to its loader
    calls.  This test exercises the kwarg-validation path the loader
    already supports; together with the notebook patch in the same
    PR (``notebook_followup`` todo), Section 0 will fail loudly on a
    drift between the embedded HDF5 attribute and the figure caption.
    """
    out = load_bns_with_channels(path=bns_a_path, expected_model="A", expected_ns_max=_NS_MAX_BNS)
    # When the HDF5 carries the embedded ``model`` attribute (set by
    # ``tools/embed_model_metadata.py``), the loader propagates it.
    assert out.get("model") in (None, "A"), out.get("model")
    assert out.get("ns_max") in (None, _NS_MAX_BNS), out.get("ns_max")


@pytest.mark.requires_data
def test_bhns_a_metadata_matches_modelA(bhns_a_path):
    """``load_bhns_with_channels`` with ``expected_model='A'`` accepts Model A HDF5."""
    out = load_bhns_with_channels(path=bhns_a_path, expected_model="A", expected_ns_max=_NS_MAX_BNS)
    assert out.get("model") in (None, "A"), out.get("model")
    assert out.get("ns_max") in (None, _NS_MAX_BNS), out.get("ns_max")


@pytest.mark.requires_data
def test_bns_a_wrong_expected_model_raises(bns_a_path):
    """``load_bns_with_channels(expected_model='K')`` must fail on a Model A file.

    The validation guard is the whole point of the ``expected_model``
    kwarg; pin it so a future loader rewrite that downgrades the check
    to a warning breaks this test.
    """
    with h5.File(bns_a_path, "r") as f:
        embedded_model = f.attrs.get("model")
    if (
        embedded_model is None
        or isinstance(embedded_model, bytes)
        and embedded_model.decode() != "A"
    ):
        pytest.skip(
            "BNS A HDF5 does not carry an embedded 'model' attribute; "
            "run tools/embed_model_metadata.py to enable validation."
        )
    with pytest.raises(ValueError, match="model"):
        load_bns_with_channels(path=bns_a_path, expected_model="K")


# ─────────────────────────────────────────────────────────────────────
# Section 1 panel constants line up with what classify_grid expects.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_notebook_ns_max_matches_file_attribute(bns_a_path):
    """Section 1's literal ``NS_MAX_BNS = 2.5`` must match the HDF5 attribute.

    Canonical gotcha from CLAUDE.md: passing the wrong ``ns_max`` to
    ``classify_grid`` silently reassigns NS-NS systems to BHNS or BBH.
    This test pins the Section 0 literal against the embedded HDF5
    attribute set by ``tools/embed_model_metadata.py``.  Test skips
    if the attribute is absent so a fresh download without metadata
    annotation does not poison CI.
    """
    with h5.File(bns_a_path, "r") as f:
        ns_max_attr = f.attrs.get("ns_max")
    if ns_max_attr is None:
        pytest.skip(
            "BNS A HDF5 has no embedded 'ns_max' attribute; run "
            "tools/embed_model_metadata.py to enable validation."
        )
    assert float(ns_max_attr) == pytest.approx(_NS_MAX_BNS, rel=1e-9), (
        f"BNS A embedded ns_max = {float(ns_max_attr)} != notebook "
        f"literal NS_MAX_BNS = {_NS_MAX_BNS}.  Section 1's panel "
        f"extent and classify_grid call would silently mislabel the "
        f"BNS/BHNS boundary."
    )
    assert _NS_MAX_BNS in NS_MAX_FIDUCIAL, NS_MAX_FIDUCIAL


# ─────────────────────────────────────────────────────────────────────
# Alsing remap: determinism (same seed -> same output) + safety
# invariants (cap at M_TOV, preserve m1 >= m2, preserve total weight).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_bns_remap_seed42_is_deterministic(bns_a_path):
    """``remap_ns_masses_double_gaussian`` is reproducible at fixed seed.

    Section 0 uses ``np.random.default_rng(42)`` for the BNS remap;
    a future change to the function (e.g. an extra random call inside
    the tie-breaking jitter) would silently shift the entire BNS
    sample.  Run the remap twice with the same seed and assert bit-
    exact equality on both component arrays.
    """
    bns = load_bns_with_channels(path=bns_a_path)

    m1_a, m2_a = remap_ns_masses_double_gaussian(
        bns["m1"].copy(),
        bns["m2"].copy(),
        weights=bns["weights"],
        rng=np.random.default_rng(42),
    )
    m1_b, m2_b = remap_ns_masses_double_gaussian(
        bns["m1"].copy(),
        bns["m2"].copy(),
        weights=bns["weights"],
        rng=np.random.default_rng(42),
    )
    assert np.array_equal(m1_a, m1_b), (
        "BNS remap with seed 42 differs between runs; the remap or "
        "the underlying RNG path has acquired non-determinism."
    )
    assert np.array_equal(m2_a, m2_b), "Same as above for m2."


@pytest.mark.requires_data
def test_bhns_remap_seed43_is_deterministic(bhns_a_path):
    """``remap_ns_marginal`` (M_NS only) reproduces under seed 43."""
    bhns = load_bhns_with_channels(path=bhns_a_path)
    M_NS_a = remap_ns_marginal(bhns["M_NS"], weights=bhns["weights"], rng=np.random.default_rng(43))
    M_NS_b = remap_ns_marginal(bhns["M_NS"], weights=bhns["weights"], rng=np.random.default_rng(43))
    assert np.array_equal(M_NS_a, M_NS_b), (
        "BHNS remap with seed 43 differs between runs; ordering or jitter is non-deterministic."
    )


@pytest.mark.requires_data
def test_bns_remap_caps_at_m_tov_and_preserves_mass_ordering(bns_modelA_remapped):
    """Post-remap BNS masses obey the global invariants Section 1 relies on.

    The Section 1 BNS panel evaluates the reflective-boundary KDE on
    [m_min, M_TOV]^2; a single remapped mass above M_TOV would make
    the reflection trick incorrect (the mirror at 2*M_TOV - m would
    end up inside the support box).  Likewise m1 >= m2 is required
    by every downstream classifier; the remap must preserve it.
    """
    m1 = bns_modelA_remapped["m1"]
    m2 = bns_modelA_remapped["m2"]

    assert m1.max() <= M_TOV + 1e-6, (
        f"max(m1) = {m1.max()} > M_TOV + 1e-6; Alsing remap did not "
        f"cap NS mass at M_TOV.  Section 1 KDE reflection invalid."
    )
    assert m2.max() <= M_TOV + 1e-6, f"max(m2) = {m2.max()} > M_TOV + 1e-6; same as above for m2."
    assert (m1 >= m2).all(), (
        f"{int((m1 < m2).sum())} of {m1.size} systems have m1 < m2 "
        f"after the remap; the m1 >= m2 invariant is broken."
    )


@pytest.mark.requires_data
def test_bhns_remap_caps_M_NS_at_m_tov(bhns_modelA_remapped):
    """Post-remap BHNS NS masses are capped at M_TOV (no leak past wall)."""
    M_NS = bhns_modelA_remapped["M_NS"]
    assert M_NS.max() <= M_TOV + 1e-6, (
        f"max(M_NS) = {M_NS.max()} > M_TOV + 1e-6; Alsing marginal "
        f"remap did not cap NS mass at M_TOV.  Section 1 BHNS KDE "
        f"reflection invalid."
    )
    assert M_NS.min() > 0, "Remapped NS mass has non-positive entries."


@pytest.mark.requires_data
def test_remap_preserves_total_weight(bns_a_path, bhns_a_path):
    """STROOPWAFEL weights survive the remap unchanged.

    Both ``remap_ns_masses_double_gaussian`` and ``remap_ns_marginal``
    are rank-preserving quantile transforms applied to the mass
    arrays; the per-system weights are not touched.  This test
    guards against a future change that re-orders the weight array
    or accidentally mutates it.
    """
    bns = load_bns_with_channels(path=bns_a_path)
    w_before = bns["weights"].copy()
    _ = remap_ns_masses_double_gaussian(
        bns["m1"].copy(),
        bns["m2"].copy(),
        weights=bns["weights"],
        rng=np.random.default_rng(42),
    )
    assert np.array_equal(bns["weights"], w_before), (
        "remap_ns_masses_double_gaussian mutated the input weight "
        "array in place.  Section 1's weighted KDE and printed "
        "fractions would silently shift."
    )

    bhns = load_bhns_with_channels(path=bhns_a_path)
    w_before_bhns = bhns["weights"].copy()
    _ = remap_ns_marginal(bhns["M_NS"], weights=bhns["weights"], rng=np.random.default_rng(43))
    assert np.array_equal(bhns["weights"], w_before_bhns), (
        "remap_ns_marginal mutated the input weight array in place."
    )


# ─────────────────────────────────────────────────────────────────────
# Plot honesty: the BNS panel's KDE actually sees the STROOPWAFEL
# weights, not np.ones_like.  Easy diagnostic: the two contour fields
# must differ by a measurable fraction on the high-mass tail (where
# the STROOPWAFEL prior heavily up-weights low-Z systems).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_bns_kde_is_actually_weighted_on_real_sample(bns_modelA_remapped):
    """The Section 1 BNS KDE must change when weights are stripped.

    Section 1's BNS panel feeds STROOPWAFEL weights into
    ``gaussian_kde(..., weights=w_bns, bw_method='silverman')``.  A
    careless ``np.ones_like`` would compile and run but silently mis-
    represent the population.  Build the BNS panel's KDE twice on the
    real sample -- once with weights, once with unit weights -- and
    assert the two density fields differ by more than 5 percent in
    L1 norm on the (M2, M1) evaluation grid Section 1 uses.
    """
    from scipy.stats import gaussian_kde

    m1 = bns_modelA_remapped["m1"]
    m2 = bns_modelA_remapped["m2"]
    w = bns_modelA_remapped["weights"]

    m2_sym = np.concatenate([m2, m1])
    m1_sym = np.concatenate([m1, m2])
    w_sym = np.concatenate([w, w])

    kde_w = gaussian_kde(np.vstack([m2_sym, m1_sym]), weights=w_sym, bw_method="silverman")
    kde_u = gaussian_kde(np.vstack([m2_sym, m1_sym]), bw_method="silverman")

    # Sample on the actual Section 1 evaluation grid (180x180 over
    # [m_lo, min(M_TOV, m_hi)] on each axis).
    m2_grid = np.linspace(_M2_LO_BNS, min(M_TOV, _M2_HI_BNS), 60)
    m1_grid = np.linspace(_M1_LO_BNS, min(M_TOV, _M1_HI_BNS), 60)
    M2g, M1g = np.meshgrid(m2_grid, m1_grid)
    pts = np.vstack([M2g.ravel(), M1g.ravel()])

    Z_w = kde_w(pts).reshape(M2g.shape)
    Z_u = kde_u(pts).reshape(M2g.shape)

    # Normalise each so total mass on the grid is 1, then compare L1.
    Z_w_norm = Z_w / Z_w.sum()
    Z_u_norm = Z_u / Z_u.sum()
    l1 = float(np.abs(Z_w_norm - Z_u_norm).sum())
    assert l1 > 0.05, (
        f"Weighted and unweighted Section 1 BNS KDEs differ by L1 = "
        f"{l1:.4f}; expected > 0.05.  If this fails, either the "
        f"STROOPWAFEL weights are uniform (which they should not be "
        f"for a Broekgaarden+ 2021 sample) or the notebook is "
        f"plotting the unweighted distribution by accident."
    )


@pytest.mark.requires_data
def test_bhns_kde_is_actually_weighted_on_real_sample(bhns_modelA_remapped):
    """The Section 1 BHNS KDE must change when weights are stripped.

    Same honesty check as the BNS counterpart, on the (M_NS, M_BH)
    plane.  STROOPWAFEL up-weights low-Z BHNS systems hard, so the
    weighted and unweighted KDEs differ more in BHNS than in BNS.
    """
    from scipy.stats import gaussian_kde

    NS_bh = bhns_modelA_remapped["M_NS"]
    BH = bhns_modelA_remapped["M_BH"]
    w = bhns_modelA_remapped["weights"]

    kde_w = gaussian_kde(np.vstack([NS_bh, BH]), weights=w, bw_method="silverman")
    kde_u = gaussian_kde(np.vstack([NS_bh, BH]), bw_method="silverman")

    NS_grid = np.linspace(1.25, M_TOV, 60)
    BH_grid = np.linspace(2.5, 18.0, 60)
    NSg, BHg = np.meshgrid(NS_grid, BH_grid)
    pts = np.vstack([NSg.ravel(), BHg.ravel()])

    Z_w = kde_w(pts).reshape(NSg.shape)
    Z_u = kde_u(pts).reshape(NSg.shape)

    Z_w_norm = Z_w / Z_w.sum()
    Z_u_norm = Z_u / Z_u.sum()
    l1 = float(np.abs(Z_w_norm - Z_u_norm).sum())
    assert l1 > 0.05, (
        f"Weighted and unweighted Section 1 BHNS KDEs differ by L1 = {l1:.4f}; expected > 0.05."
    )


# ─────────────────────────────────────────────────────────────────────
# Self-consistency: the printed class fractions in Section 0's summary
# cell match the classifier replayed on the remapped sample.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_printed_class_fractions_match_classifier_on_remapped_sample(
    bns_modelA_remapped, bhns_modelA_remapped
):
    """Section 0's printed BNS / BHNS class fractions reproduce on the real sample.

    The summary cell at the top of ``grb_main.ipynb`` prints weighted
    class fractions like ``sbGRB + blue KN ... (24.0% weighted)``.
    Reproduce the exact reduction on the same remapped sample and
    assert the fractions match the values in the latest committed
    notebook to within 0.2 percentage points.  Drift here means one
    of: (i) the classifier changed, (ii) the remap RNG changed,
    (iii) the weight or load pipeline drifted, (iv) the notebook
    output is stale and needs a rerun.

    Update the ``_EXPECTED_*_FRACTIONS_PCT`` constants at the top of
    this file when a deliberate science change moves the fractions.
    """
    m1 = bns_modelA_remapped["m1"]
    m2 = bns_modelA_remapped["m2"]
    w = bns_modelA_remapped["weights"]
    cls = classify_bns_2024(m1, m2)
    for key, expected_pct in _EXPECTED_BNS_FRACTIONS_PCT.items():
        mask = cls[key]
        got_pct = 100.0 * float((mask * w).sum() / w.sum())
        assert got_pct == pytest.approx(expected_pct, abs=0.2), (
            f"BNS class '{key}' fraction = {got_pct:.2f} percent, "
            f"expected {expected_pct} percent (notebook header).  "
            f"Either the remap RNG, classifier, or weight pipeline "
            f"drifted, or the notebook printed output is stale."
        )

    BH = bhns_modelA_remapped["M_BH"]
    NS = bhns_modelA_remapped["M_NS"]
    w_bhns = bhns_modelA_remapped["weights"]
    cls_bhns = classify_bhns(BH, NS, a_BH=0.5)
    for key, expected_pct in _EXPECTED_BHNS_FRACTIONS_PCT.items():
        mask = cls_bhns[key]
        got_pct = 100.0 * float((mask * w_bhns).sum() / w_bhns.sum())
        assert got_pct == pytest.approx(expected_pct, abs=0.2), (
            f"BHNS class '{key}' fraction = {got_pct:.2f} percent, "
            f"expected {expected_pct} percent (notebook header).  "
            f"Either ``a_BH = 0.5``, the disk-mass formula, the "
            f"remap RNG, or the weight pipeline drifted."
        )


# ─────────────────────────────────────────────────────────────────────
# Observed GW events lie inside the Section 1 BNS panel.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_observed_gw_events_fall_inside_grid_panel():
    """GW170817 and GW190425 fit inside the BNS panel's plotted [M2, M1] box.

    Light sanity check.  Section 1 annotates star markers at the
    Abbott+ 2019/2020 low-spin medians; if the panel extent drifts
    below the M_TOV cap (or above the GW190425 secondary mass), the
    annotations would sit on the panel edge or off-panel entirely.
    The literature anchor is in
    ``tests/anchors/test_literature_anchors.py``; this test pins the
    *placement* inside the plotted extent.
    """
    from grb_io import OBSERVED_GW_EVENTS

    for name, event in OBSERVED_GW_EVENTS.items():
        m2 = event["M2"]
        m1 = event["M1"]
        assert _M2_LO_BNS <= m2 <= _M2_HI_BNS, (
            f"{name}: M2 = {m2} falls outside the Section 1 BNS panel "
            f"M2 extent [{_M2_LO_BNS}, {_M2_HI_BNS}]."
        )
        assert _M1_LO_BNS <= m1 <= _M1_HI_BNS, (
            f"{name}: M1 = {m1} falls outside the Section 1 BNS panel "
            f"M1 extent [{_M1_LO_BNS}, {_M1_HI_BNS}]."
        )
