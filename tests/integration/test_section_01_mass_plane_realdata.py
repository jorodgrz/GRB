"""Real-data audit for Section 1 of ``grb_main.ipynb`` (BNS + BHNS mass-plane panels).

The two Section 1 panels are the project's headline figures: the
Gottlieb (2024) classifier underlay on the (M2, M1) BNS plane and on
the (M_NS, M_BH) BHNS plane, with a STROOPWAFEL-weighted 2D mass
histogram on top.  The tests in this file lock the
end-to-end pipeline that feeds those panels against the COMPAS Model A
HDF5 files: file identity, the cold-stability cut (raw fiducial masses,
``m_i <= M_TOV``) plus the ``m1 >= m2`` invariant, the retained NS-mass
remap determinism (the Section 2b systematic), histogram weighting
honesty on the real sample, and the printed class fractions the summary
cell emits.

All tests are ``@pytest.mark.requires_data`` and skip on machines
without ``data/COMPASCompactOutput_{BNS,BHNS}_A.h5``.  The fiducial
sample is the raw COMPAS catalogue after the cold-stability cut; the
Alsing remap is exercised only as the reproducible sensitivity tool.
The printed class-fraction test asserts the structural invariants
(fractions in [0, 1] summing to 1, sbGRB + blue KN now a leading class)
rather than pinning exact percentages, which must be regenerated from a
notebook run under the raw fiducial.

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
    filter_systems,
    load_bhns_with_channels,
    load_bns_with_channels,
)
from grb_physics import (
    M_TOV,
    cold_stable_ns_mask,
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

_BNS_CLASS_KEYS = [
    "sbGRB + blue KN",
    "lbGRB + red KN (HMNS)",
    "lbGRB + red KN (disk)",
    "Faint lbGRB",
]
_BHNS_CLASS_KEYS = [
    "lbGRB + red KN (BHNS disk)",
    "Faint lbGRB (BHNS)",
    "No GRB",
]


@pytest.fixture(scope="module")
def bns_modelA_fiducial(bns_a_path):
    """Section 0's fiducial BNS sample: raw COMPAS masses + cold-stability cut."""
    bns = load_bns_with_channels(path=bns_a_path)
    bns = filter_systems(bns, cold_stable_ns_mask(bns["m1"], bns["m2"], m_tov=M_TOV))
    return {
        "m1": bns["m1"],
        "m2": bns["m2"],
        "weights": bns["weights"],
        "model": bns.get("model"),
        "ns_max": bns.get("ns_max"),
    }


@pytest.fixture(scope="module")
def bhns_modelA_fiducial(bhns_a_path):
    """Section 0's fiducial BHNS sample: raw M_NS + cold-stability cut on the NS."""
    bhns = load_bhns_with_channels(path=bhns_a_path)
    bhns = filter_systems(bhns, cold_stable_ns_mask(bhns["M_NS"], m_tov=M_TOV))
    return {
        "M_BH": bhns["M_BH"],
        "M_NS": bhns["M_NS"],
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
def test_bns_cut_enforces_m_tov_and_preserves_mass_ordering(bns_modelA_fiducial):
    """The cold-stability cut leaves the invariants Section 1 relies on.

    The Section 1 BNS panel bins the sample on [m_min, M_TOV]^2 with
    the top edge pinned on M_TOV; a component above M_TOV would fall
    off the grid and break the exact weight conservation.  The cut
    guarantees ``m_i <= M_TOV`` by construction, and the loader's
    ``sort_masses`` keeps ``m1 >= m2``.
    """
    m1 = bns_modelA_fiducial["m1"]
    m2 = bns_modelA_fiducial["m2"]

    assert m1.max() <= M_TOV + 1e-6, (
        f"max(m1) = {m1.max()} > M_TOV + 1e-6; the cold-stability cut "
        f"did not remove all above-M_TOV components."
    )
    assert m2.max() <= M_TOV + 1e-6, f"max(m2) = {m2.max()} > M_TOV + 1e-6; same as above for m2."
    assert (m1 >= m2).all(), (
        f"{int((m1 < m2).sum())} of {m1.size} systems have m1 < m2 "
        f"after the cut; the m1 >= m2 invariant is broken."
    )


@pytest.mark.requires_data
def test_bhns_cut_enforces_M_NS_at_m_tov(bhns_modelA_fiducial):
    """The cold-stability cut caps the BHNS NS component at M_TOV."""
    M_NS = bhns_modelA_fiducial["M_NS"]
    assert M_NS.max() <= M_TOV + 1e-6, (
        f"max(M_NS) = {M_NS.max()} > M_TOV + 1e-6; the cold-stability "
        f"cut did not remove all above-M_TOV NS components.  Section 1 "
        f"BHNS histogram would drop weight off the top NS bin edge."
    )
    assert M_NS.min() > 0, "Cut NS mass has non-positive entries."


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
# Plot honesty: the BNS panel's histogram actually sees the STROOPWAFEL
# weights, not np.ones_like.  Easy diagnostic: the weighted and
# unweighted bin fields must differ by a measurable fraction (the
# STROOPWAFEL prior heavily up-weights low-Z systems).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_bns_histogram_is_actually_weighted_on_real_sample(bns_modelA_fiducial):
    """The Section 1 BNS histogram must change when weights are stripped.

    Section 1's BNS panel feeds STROOPWAFEL weights into
    ``np.histogram2d(..., weights=w_bns)``.  A careless ``np.ones_like``
    would compile and run but silently misrepresent the population.
    Build the panel's histogram twice on the real sample -- once with
    weights, once with unit weights -- and assert the two normalised
    bin fields differ by more than 5 percent in L1 norm on the
    M_TOV-anchored 0.02 Msun grid Section 1 uses.
    """
    m1 = bns_modelA_fiducial["m1"]
    m2 = bns_modelA_fiducial["m2"]
    w = bns_modelA_fiducial["weights"]

    _bw = 0.02
    _nb = int(np.ceil((M_TOV - _M2_LO_BNS) / _bw))
    edges = M_TOV - _bw * np.arange(_nb, -1, -1)

    Hw, _, _ = np.histogram2d(m2, m1, bins=[edges, edges], weights=w)
    Hu, _, _ = np.histogram2d(m2, m1, bins=[edges, edges])

    Hw_norm = Hw / Hw.sum()
    Hu_norm = Hu / Hu.sum()
    l1 = float(np.abs(Hw_norm - Hu_norm).sum())
    assert l1 > 0.05, (
        f"Weighted and unweighted Section 1 BNS histograms differ by "
        f"L1 = {l1:.4f}; expected > 0.05.  If this fails, either the "
        f"STROOPWAFEL weights are uniform (which they should not be "
        f"for a Broekgaarden+ 2021 sample) or the notebook is "
        f"plotting the unweighted distribution by accident."
    )


@pytest.mark.requires_data
def test_bhns_histogram_is_actually_weighted_on_real_sample(bhns_modelA_fiducial):
    """The Section 1 BHNS histogram must change when weights are stripped.

    Same honesty check as the BNS counterpart, on the (M_NS, M_BH)
    plane.  STROOPWAFEL up-weights low-Z BHNS systems hard, so the
    weighted and unweighted histograms differ more in BHNS than in BNS.
    """
    NS_bh = bhns_modelA_fiducial["M_NS"]
    BH = bhns_modelA_fiducial["M_BH"]
    w = bhns_modelA_fiducial["weights"]

    _bw_ns = 0.02
    _nb_ns = int(np.ceil((M_TOV - 1.25) / _bw_ns))
    ns_edges = M_TOV - _bw_ns * np.arange(_nb_ns, -1, -1)
    bh_edges = 2.5 + 0.25 * np.arange(int(np.round((18.0 - 2.5) / 0.25)) + 1)

    Hw, _, _ = np.histogram2d(NS_bh, BH, bins=[ns_edges, bh_edges], weights=w)
    Hu, _, _ = np.histogram2d(NS_bh, BH, bins=[ns_edges, bh_edges])

    Hw_norm = Hw / Hw.sum()
    Hu_norm = Hu / Hu.sum()
    l1 = float(np.abs(Hw_norm - Hu_norm).sum())
    assert l1 > 0.05, (
        f"Weighted and unweighted Section 1 BHNS histograms differ by L1 = {l1:.4f}; expected > 0.05."
    )


@pytest.mark.requires_data
def test_bhns_offpanel_fraction_above_18_msun(bhns_modelA_fiducial):
    """The BHNS panel crops M_BH at 18 Msun; the off-panel weight is small.

    The caption states that ~1.2 percent of the weighted sample lies
    above the 18 Msun crop.  Pin that fraction so a future change to
    the axis limit (or a data swap) that moved a large fraction of the
    population off-panel would break here and force a caption update.
    """
    BH = bhns_modelA_fiducial["M_BH"]
    w = bhns_modelA_fiducial["weights"]
    off = float(w[BH > 18.0].sum() / w.sum())
    assert 0.010 <= off <= 0.015, (
        f"Off-panel weight fraction above M_BH = 18 Msun is {off:.4f}; "
        f"expected in [0.010, 0.015].  Update the panel crop and the "
        f"caption's stated off-panel fraction together."
    )


# ─────────────────────────────────────────────────────────────────────
# Structural invariants of the printed class fractions on the raw
# fiducial (cold-stability-cut) sample.  Exact per-class percentages are
# deliberately not pinned here: the switch to the raw fiducial is a
# science change, so any hardcoded target would have to come from a
# fresh notebook run.  The inversion the change predicts is that
# sbGRB + blue KN (low total mass, long-lived HMNS) becomes a leading
# class, because the raw COMPAS total-mass distribution peaks lower than
# the Alsing-remapped one that previously pushed systems into the HMNS
# and prompt-collapse regimes.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_class_fractions_on_fiducial_sample_are_valid_and_sbGRB_rises(
    bns_modelA_fiducial, bhns_modelA_fiducial
):
    """BNS / BHNS class fractions are valid probabilities; sbGRB rises on raw.

    Reproduce Section 0's weighted reduction on the raw fiducial sample.
    The four BNS class fractions must lie in [0, 1] and sum to 1, and
    the sbGRB + blue KN (long-lived HMNS) fraction must be substantial
    (> 0.35): the raw COMPAS total-mass distribution peaks lower than
    the old Alsing-remapped one, so more systems fall below the HMNS
    split and this class rises well above its former ~0.24.  A loose
    lower bound is used deliberately; the exact percentage must come
    from a notebook run under the raw fiducial.  The three BHNS
    fractions must likewise be valid and sum to 1.
    """
    m1 = bns_modelA_fiducial["m1"]
    m2 = bns_modelA_fiducial["m2"]
    w = bns_modelA_fiducial["weights"]
    cls = classify_bns_2024(m1, m2)
    fracs = {k: float((cls[k] * w).sum() / w.sum()) for k in _BNS_CLASS_KEYS}
    for k, f in fracs.items():
        assert 0.0 <= f <= 1.0, f"BNS class '{k}' fraction {f} outside [0, 1]"
    assert sum(fracs.values()) == pytest.approx(1.0, abs=1e-9)
    assert fracs["sbGRB + blue KN"] > 0.35, (
        f"sbGRB + blue KN fraction {fracs['sbGRB + blue KN']:.3f} did not "
        f"rise above 0.35 on the raw fiducial sample: {fracs}."
    )

    BH = bhns_modelA_fiducial["M_BH"]
    NS = bhns_modelA_fiducial["M_NS"]
    w_bhns = bhns_modelA_fiducial["weights"]
    cls_bhns = classify_bhns(BH, NS, a_BH=0.5)
    bfracs = {k: float((cls_bhns[k] * w_bhns).sum() / w_bhns.sum()) for k in _BHNS_CLASS_KEYS}
    for k, f in bfracs.items():
        assert 0.0 <= f <= 1.0, f"BHNS class '{k}' fraction {f} outside [0, 1]"
    assert sum(bfracs.values()) == pytest.approx(1.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────
# Observed GW events lie inside the Section 1 BNS panel.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_observed_gw_events_fall_inside_grid_panel():
    """GW170817 and GW190425 fit inside the BNS panel's plotted [M2, M1] box.

    Light sanity check.  Section 1 annotates star markers at the
    Abbott+ 2019/2020 low-spin representative points (which sit inside
    the published 90% CRs); if the panel extent drifts below the M_TOV
    cap (or above the GW190425 secondary mass), the annotations would
    sit on the panel edge or off-panel entirely.
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
