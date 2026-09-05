"""Section 5 of grb_main.ipynb: Metallicity Dependence of GRB Formation Efficiency.

Smoke-level invariants on the data plumbing the per-class plot consumes
and on how the figure represents the bins it cannot measure well.

The figure bins the whole 53-point COMPAS grid and draws every populated
bin as an ordinary point, ungated and with no error bar or limit, which is
what Broekgaarden+ 2021 Fig. 5 and Broekgaarden+ 2022 Fig. 1 do with the
same quantity; Broekgaarden+ 2021 Sec. 2.7 evaluate the STROOPWAFEL
sampling uncertainty on this yield and then decline to quote it anywhere.
Sparsity is reported as raw counts in the cell output and in the caption
instead.  One per-bin rule still changes how a bin is drawn:
``Z >= Z_CE_LAMBDA_BIFURCATION``, the Broekgaarden+ 2022 common-envelope
lambda artefact, flagged and drawn faded but never cut, since it holds
21 percent of the BNS weight and supplies most of the local rate the later
sections integrate.

``N_MIN`` / ``ESS_MIN`` below mirror the ``ETA_GATE_*`` heuristics that
``formation_efficiency_binned`` still exposes through ``ok``.  The figure
does not use them; the tests that reference them use them to select
well-sampled bins for shape checks, which is a different question from how
the figure renders.

These tests pin the per-bin reductions the figure consumes and verify the
logic on the real Model A sample, so a refactor that re-enables the
cross-gap stackplot interpolation, leaves a curve with an empty bin, or
goes back to deleting bins on metallicity grounds will fail loudly.

Reference: Broekgaarden et al. (2021), arXiv:2103.02608 Sec. 4
(formation efficiency definition); Gottlieb et al. (2024),
arXiv:2411.13657 (four-class scheme used here); Kish (1965), Survey
Sampling, Wiley, Sec. 1.6 (effective sample size).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import classify_bhns, classify_bns_2024
from grb_io import METALLICITY_GRID, filter_systems, load_bhns, load_bns
from grb_physics import (
    M_TOV,
    cold_stable_ns_mask,
    remap_ns_marginal,
    remap_ns_masses_double_gaussian,
)
from grb_rates import (
    LOGZ_MAX_PHYSICAL,
    Z_CE_LAMBDA_BIFURCATION,
    formation_efficiency,
    formation_efficiency_binned,
    simulation_star_forming_mass,
)

Z_SUN = 0.0142
N_MIN = 50
ESS_MIN = 10.0
from grb_physics import A_BH_FID  # noqa: E402  (single source of truth)


# ─────────────────────────────────────────────────────────────────────
# Existing invariants on the data plumbing.
# ─────────────────────────────────────────────────────────────────────
def test_per_class_efficiency_partitions_total_synthetic():
    """Per-class arrays sum to the total at every Z bin (synthetic)."""
    rng = np.random.default_rng(7)
    n = 2000
    Z_all = rng.choice(METALLICITY_GRID, size=n)
    w_all = rng.uniform(0.0, 1.0, n)
    m1 = rng.uniform(1.1, 3.5, n)
    m2 = rng.uniform(1.1, 3.5, n)
    masks = classify_bns_2024(m1, m2)

    eff = formation_efficiency(
        METALLICITY_GRID,
        Z_all,
        w_all,
        masks=masks,
        mean_mass_evolved=1.0,
    )

    assert "total" in eff
    summed = np.zeros_like(eff["total"])
    for label in masks:
        summed += eff[label]
    np.testing.assert_allclose(summed, eff["total"], rtol=1e-9, atol=1e-12)


@pytest.mark.requires_data
def test_per_class_efficiency_finite_on_modelA(bns_a_path):
    """Real-data smoke: every per-class efficiency entry is finite and
    non-negative.  Uses ``mean_mass_evolved=1.0`` because the partition
    invariant is independent of the calibration anchor."""
    bns = load_bns(bns_a_path, expected_model="A", expected_ns_max=2.5)
    masks = classify_bns_2024(bns["m1"], bns["m2"])
    eff = formation_efficiency(
        METALLICITY_GRID,
        bns["metallicity"],
        bns["weights"],
        masks=masks,
        mean_mass_evolved=1.0,
    )
    for label, arr in eff.items():
        assert np.all(np.isfinite(arr)), f"non-finite efficiency in {label}"
        assert np.all(arr >= 0.0), f"negative efficiency in {label}"


# ─────────────────────────────────────────────────────────────────────
# Helpers mirroring the notebook cell's masking pipeline.  These let
# the tests exercise the artifact-suppression rules without re-executing
# the notebook.
# ─────────────────────────────────────────────────────────────────────
def _bin_stats(Z_systems, w_systems, grid):
    """Per-bin (N_raw, Kish ESS) tuple.  Mirrors the helper in cell 16."""
    Z_s = np.asarray(Z_systems, dtype=float)
    w_s = np.asarray(w_systems, dtype=float)
    idx = np.searchsorted(grid, Z_s, side="left")
    in_grid = (idx < len(grid)) & np.isclose(Z_s, grid[np.clip(idx, 0, len(grid) - 1)])
    idx = idx[in_grid]
    w_s = w_s[in_grid]
    N = np.bincount(idx, minlength=len(grid))
    Sw = np.bincount(idx, weights=w_s, minlength=len(grid)).astype(float)
    Sw2 = np.bincount(idx, weights=w_s * w_s, minlength=len(grid)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ess = np.where(Sw2 > 0, Sw * Sw / Sw2, 0.0)
    return N, ess


def _contiguous_runs(mask):
    """Yield (start, stop) for each True run in a 1-D boolean mask.

    Mirrors the helper used by the notebook stackplot to ensure
    matplotlib never draws a band across a masked region.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    edges = np.diff(m.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _ok_bin(Z_systems, w_systems, grid):
    """Notebook precision gate: N >= 50 AND ESS >= 10.

    Metallicity plays no part.  Section 5 bins the whole 53-point grid and
    flags the Broekgaarden+ 2022 CE-lambda range instead of cutting it, so a
    Z ceiling folded in here would no longer mirror the notebook.
    """
    N, ess = _bin_stats(Z_systems, w_systems, grid)
    return (N >= N_MIN) & (ess >= ESS_MIN)


# ─────────────────────────────────────────────────────────────────────
# New tests.
# ─────────────────────────────────────────────────────────────────────
def test_section_5_bins_the_whole_grid_and_only_flags_the_artefact():
    """Every native metallicity is binned; the CE-lambda range is flagged, not cut.

    Section 5 runs ``Z_max=None`` so no grid point is dropped, and reads the
    artefact range off ``artifact`` rather than off ``ok``.  Cutting at Zsun
    instead would silently drop seven grid points carrying 13 percent of the
    BNS weight, and folding ``Z_cut`` into ``ok`` would drop a further fifth.
    ``LOGZ_MAX_PHYSICAL`` survives only as the solar guide line on the axis.
    """
    from grb_rates import formation_efficiency_binned

    grid = np.asarray(METALLICITY_GRID, dtype=float)
    grid = grid[np.concatenate(([True], np.diff(grid) != 0))]
    n_per_bin = 10 * N_MIN
    Z_all = np.repeat(grid, n_per_bin)
    w_all = np.ones_like(Z_all)

    res = formation_efficiency_binned(
        grid,
        Z_all,
        w_all,
        mean_mass_evolved=1e9,
        n_bins=18,
        Z_max=None,
        Z_cut=Z_CE_LAMBDA_BIFURCATION,
        n_min=N_MIN,
        ess_min=ESS_MIN,
    )
    assert int(res["n_raw"]["total"].sum()) == Z_all.size, (
        "binning lost systems: Z_max must stay None so the whole grid is covered"
    )
    assert res["Z_hi"].max() == pytest.approx(grid.max())
    assert res["ok"]["total"].all(), "a uniformly sampled grid must clear the precision gate"
    np.testing.assert_array_equal(res["artifact"], res["Z_hi"] >= Z_CE_LAMBDA_BIFURCATION)
    assert res["artifact"].any() and not res["artifact"].all()


@pytest.mark.requires_data
def test_section_5_draws_every_bin_ungated_on_modelA(bns_a_path, bhns_a_path):
    """Section 5's caption claims are true of the sample it plots.

    The figure is drawn on the native 53-point grid and ungated: every
    populated metallicity is an ordinary point and the sparsity is quoted
    as raw counts instead of a gate or a limit, following Broekgaarden+
    2021 Sec. 2.7, who evaluate the STROOPWAFEL sampling uncertainty on
    this yield and then decline to quote it, and their Fig. 5 /
    Broekgaarden+ 2022 Fig. 1, which plot every simulated grid point
    ungated.  An ungated curve renders an exact zero as a line gap, so the
    caption has to be able to say where the gaps are: none on the BNS
    panel, and on the BHNS panel only above solar, inside the CE-lambda
    band where the yield has already collapsed.

    The sample here is the notebook's: raw COMPAS masses with the
    cold-stability cut, no Alsing remap, which is the fiducial treatment.
    """
    from grb_io import filter_systems
    from grb_physics import M_TOV, cold_stable_ns_mask
    from grb_rates import formation_efficiency_binned

    grid = np.asarray(METALLICITY_GRID, dtype=float)
    grid = grid[np.concatenate(([True], np.diff(grid) != 0))]
    # The notebook's ETA_KW + ETA_N_BINS: whole grid at native resolution,
    # one point per simulated metallicity, gate switched off.
    eta_kw = dict(n_bins=grid.size, Z_max=None, Z_cut=Z_CE_LAMBDA_BIFURCATION, n_min=0, ess_min=0.0)

    bns = load_bns(bns_a_path, expected_model="A", expected_ns_max=2.5)
    bns = filter_systems(bns, cold_stable_ns_mask(bns["m1"], bns["m2"], m_tov=M_TOV))
    cls = classify_bns_2024(bns["m1"], bns["m2"])
    res_bns = formation_efficiency_binned(
        grid, bns["metallicity"], bns["weights"], dict(cls), mean_mass_evolved=1e9, **eta_kw
    )

    bhns = load_bhns(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    bhns = filter_systems(bhns, cold_stable_ns_mask(bhns["M_NS"], m_tov=M_TOV))
    cbh = classify_bhns(bhns["M_BH"], bhns["M_NS"], a_BH=A_BH_FID)
    res_bhns = formation_efficiency_binned(
        grid,
        bhns["metallicity"],
        bhns["weights"],
        {k: v for k, v in cbh.items() if k != "M_disk"},
        mean_mass_evolved=1e9,
        **eta_kw,
    )

    for label, res in (("BNS", res_bns), ("BHNS", res_bhns)):
        for key, n in res["n_raw"].items():
            assert n.size == grid.size
        assert res["ok"]["total"].all(), "gate must be off in the Section 5 config"

    # BNS is populated at every simulated metallicity, so that panel has no
    # gaps at all and the caption can say so without qualification.
    for key, n in res_bns["n_raw"].items():
        assert (n > 0).all(), (
            f"BNS {key}: {int((n == 0).sum())} empty metallicities.  The Section 5 "
            "caption claims the BNS panel is gap-free; pooling would be needed "
            "to keep that true."
        )

    # BHNS does have gaps.  They are allowed, but only where the caption says
    # they are: above solar, inside the CE-lambda band, on the collapsed tail.
    for key, n in res_bhns["n_raw"].items():
        empty = n == 0
        assert int(empty.sum()) <= 2, f"BHNS {key}: {int(empty.sum())} empty metallicities"
        assert res_bhns["artifact"][empty].all(), (
            f"BHNS {key}: an empty metallicity outside the CE-lambda band.  The "
            "caption confines every gap to Z > Zsun."
        )
        assert (res_bhns["Z"][empty] > Z_SUN).all()

    # Raw counts quoted in the Section 5 caption table.
    assert min(int(n.min()) for k, n in res_bns["n_raw"].items() if k != "total") == 5
    assert int(res_bns["n_raw"]["sbGRB + blue KN"].min()) == 29
    assert int(res_bns["n_raw"]["total"].min()) == 75
    assert int(res_bhns["n_raw"]["total"].min()) == 1


@pytest.mark.requires_data
def test_total_efficiency_single_hump_shape_on_modelA(bns_a_path):
    """BNS total ``eta(Z)`` has a single-hump shape on the plotted axis.

    Broekgaarden+ 2021 Fig. 3-4 / Fig. 5: COMPAS Model A formation
    efficiency rises with metallicity from very low Z, peaks somewhere
    in ``Z/Zsun ~ 0.05 - 0.3`` (driven by the interplay of wind-loss
    stripping, common-envelope efficiency, and SN engine), then falls
    again towards solar.  The 53-point COMPAS grid is too sparse to
    expect strict bin-to-bin monotonicity, so we check the qualitative
    single-hump structure: the lowest-Z bin and the highest-Z plotted
    bin both lie below the maximum, and the maximum sits in the
    interior of the well-sampled range.  Multiple genuine humps would
    indicate importance-sampling noise has leaked into the figure.
    """
    bns = load_bns(bns_a_path, expected_model="A", expected_ns_max=2.5)
    bns["m1"], bns["m2"] = remap_ns_masses_double_gaussian(
        bns["m1"],
        bns["m2"],
        weights=bns["weights"],
        rng=np.random.default_rng(42),
    )

    grid = np.asarray(METALLICITY_GRID, dtype=float)
    ok = _ok_bin(bns["metallicity"], bns["weights"], grid)

    eff = formation_efficiency(
        grid,
        bns["metallicity"],
        bns["weights"],
        mean_mass_evolved=1.0,
    )
    total = eff["total"][ok]
    assert total.size >= 10, "need >= 10 well-sampled bins for the shape test"

    i_peak = int(np.argmax(total))
    assert 0 < i_peak < total.size - 1, (
        f"BNS eta(Z) peaks at the edge of the well-sampled range "
        f"(index {i_peak} of {total.size}); expected the maximum to sit "
        f"in the interior per Broekgaarden+ 2021 Fig. 5."
    )
    assert total[0] < total[i_peak], (
        f"BNS eta(Z) at the lowest Z bin ({total[0]:.3e}) is not below "
        f"the peak ({total[i_peak]:.3e})."
    )
    assert total[-1] < total[i_peak], (
        f"BNS eta(Z) at the highest plotted Z bin ({total[-1]:.3e}) is "
        f"not below the peak ({total[i_peak]:.3e})."
    )

    z_peak_over_zsun = float(grid[ok][i_peak] / Z_SUN)
    assert 0.01 <= z_peak_over_zsun <= 1.0, (
        f"BNS eta(Z) peak Z/Zsun = {z_peak_over_zsun:.3f} is outside "
        f"the physically expected band [0.01, 1.0] "
        f"(Broekgaarden+ 2021 Fig. 5)."
    )


def test_no_stackplot_segment_crosses_masked_bin():
    """Per-run stackplot helper splits at every masked bin.

    Regression for the bug where ``stackplot([Z[sel]], [frac[:, sel]])``
    linearly interpolated across deleted bins and drew a fictitious
    slanted band at the right edge of the BNS / BHNS panels.  The
    refactored cell iterates contiguous runs of ``True`` in ``ok_bin``
    instead, so a single ``False`` always splits the output into
    distinct segments.
    """
    mask = np.array([True, True, False, True, True, True, False, True, True])
    runs = _contiguous_runs(mask)

    assert runs == [(0, 2), (3, 6), (7, 9)], (
        f"_contiguous_runs returned {runs}; expected three disjoint runs"
    )
    for start, stop in runs:
        assert np.all(mask[start:stop]), f"run [{start}, {stop}) crosses a False entry in the mask"

    edge_case_all = _contiguous_runs(np.array([True, True, True]))
    assert edge_case_all == [(0, 3)]

    edge_case_none = _contiguous_runs(np.array([False, False, False]))
    assert edge_case_none == []


def test_efficiency_scales_inversely_with_the_star_forming_mass():
    """``formation_efficiency`` is exactly linear in ``1 / mean_mass_evolved``.

    This is what makes the denominator separable from the shape: every
    additivity and partition test in this file can pass an arbitrary
    constant and still be checking something real, and the absolute scale
    can be pinned once, separately, by
    ``test_total_efficiency_absolute_scale_on_modelA`` below.

    Note what this test does *not* say.  An earlier version of it claimed
    Section 5 feeds a different calibrated mass to each population and
    that sharing one value would bias the amplitude.  That is backwards
    now: the denominator is the run-configuration star-forming mass, a
    property of the sampler rather than of either population, so one
    number serves BNS, BHNS and all 20 models of the grid.  A per-model or
    per-population mass would be the regression.
    """
    rng = np.random.default_rng(11)
    n = 1000
    Z_all = rng.choice(METALLICITY_GRID, size=n)
    w_all = rng.uniform(0.1, 1.0, n)
    mme_bns, mme_bhns = 6.6e7, 1.2e8

    eff_bns = formation_efficiency(
        METALLICITY_GRID,
        Z_all,
        w_all,
        mean_mass_evolved=mme_bns,
    )
    eff_bhns = formation_efficiency(
        METALLICITY_GRID,
        Z_all,
        w_all,
        mean_mass_evolved=mme_bhns,
    )

    ratio = mme_bhns / mme_bns
    nz = eff_bns["total"] > 0
    np.testing.assert_allclose(
        eff_bns["total"][nz] / eff_bhns["total"][nz],
        np.full(nz.sum(), ratio),
        rtol=1e-12,
    )

    with pytest.raises(ValueError, match="mean_mass_evolved"):
        formation_efficiency(METALLICITY_GRID, Z_all, w_all)


# ─────────────────────────────────────────────────────────────────────
# Integration-style check that everything composes on a real sample.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_full_masking_pipeline_on_modelA(bns_a_path, bhns_a_path):
    """End-to-end: load + remap + classify + ESS-mask + per-run runs.

    Exercises the whole Section 5 pipeline on the real Model A sample
    and asserts that (a) the well-sampled coverage is non-empty for
    both populations, (b) the right edge sits at the configured
    truncation, and (c) the per-run partitioning of ``ok_bin`` covers
    every plotted bin exactly once.  Honest-to-COMPAS smoke test against
    silent regressions of any single fix.
    """
    bns = load_bns(bns_a_path, expected_model="A", expected_ns_max=2.5)
    bns["m1"], bns["m2"] = remap_ns_masses_double_gaussian(
        bns["m1"],
        bns["m2"],
        weights=bns["weights"],
        rng=np.random.default_rng(42),
    )
    _ = classify_bns_2024(bns["m1"], bns["m2"])

    bhns = load_bhns(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    bhns["M_NS"] = remap_ns_marginal(
        bhns["M_NS"],
        weights=bhns["weights"],
        rng=np.random.default_rng(43),
    )
    _ = classify_bhns(bhns["M_BH"], bhns["M_NS"], a_BH=A_BH_FID)

    grid = np.asarray(METALLICITY_GRID, dtype=float)
    for label, Z, w in (
        ("BNS", bns["metallicity"], bns["weights"]),
        ("BHNS", bhns["metallicity"], bhns["weights"]),
    ):
        ok = _ok_bin(Z, w, grid)
        assert ok.any(), f"{label}: no well-sampled bins"
        # The gate is precision only, so super-solar bins are eligible and the
        # figure reaches the top of the simulated grid.
        assert grid[ok].max() > (10.0**LOGZ_MAX_PHYSICAL) * Z_SUN, (
            f"{label}: no bin above Zsun cleared the gate, so the Section 5 "
            f"figure would still stop at the old truncation."
        )

        runs = _contiguous_runs(ok)
        covered = np.zeros_like(ok)
        for start, stop in runs:
            assert ok[start:stop].all(), (
                f"{label}: contiguous run [{start}, {stop}) contains a masked bin"
            )
            covered[start:stop] = True
        np.testing.assert_array_equal(covered, ok)


@pytest.mark.requires_data
def test_cold_stability_cut_tilts_eta_rather_than_rescaling_it(bns_a_path, bhns_a_path):
    """The Section 5 caption quotes how far the cut reshapes the curve.

    ``eta`` counts systems surviving the cold-stability cut while
    ``M_evolved`` covers every row in the file, so the numerator and the
    denominator are different populations.  That is the right bookkeeping
    for a cold-stable-NS yield per unit star formation, but the cut is
    metallicity dependent, so it changes the shape of ``eta(Z)`` and not
    just its level.  It is also why this is not the quantity in the BHNS
    panel of Broekgaarden+ 2022 Fig. 1, which applies no M_TOV cut.
    """
    from grb_physics import M_TOV, cold_stable_ns_mask

    grid = np.asarray(METALLICITY_GRID, dtype=float)
    grid = grid[np.concatenate(([True], np.diff(grid) != 0))]

    bns = load_bns(bns_a_path, expected_model="A", expected_ns_max=2.5)
    bhns = load_bhns(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    cases = {
        "BNS": (
            bns["metallicity"],
            bns["weights"],
            cold_stable_ns_mask(bns["m1"], bns["m2"], m_tov=M_TOV),
            0.912,
            0.031,
        ),
        "BHNS": (
            bhns["metallicity"],
            bhns["weights"],
            cold_stable_ns_mask(bhns["M_NS"], m_tov=M_TOV),
            0.761,
            0.185,
        ),
    }
    for label, (Z, w, keep, surv_min, overall) in cases.items():
        surv = np.array([w[(Z == z) & keep].sum() / w[Z == z].sum() for z in grid])
        assert surv.max() == pytest.approx(1.0, abs=1e-12), (
            f"{label}: the caption says the cut removes nothing at some metallicity"
        )
        assert surv.min() == pytest.approx(surv_min, abs=0.005), (
            f"{label}: surviving fraction floor moved; the caption quotes {surv_min}"
        )
        assert 1 - w[keep].sum() / w.sum() == pytest.approx(overall, abs=0.005)
        # A pure normalisation shift would be flat.  It is not, and the
        # suppression is concentrated at low Z, against the reported trend.
        assert surv[0] < surv[-1], f"{label}: cut no longer bites hardest at low Z"


@pytest.mark.requires_data
def test_bhns_high_z_collapse_is_compas_not_the_cold_stability_cut(bhns_a_path):
    """The BHNS cliff above solar is in the simulation, not in our filter.

    Broekgaarden+ 2022 Fig. 1 shows every one of its twenty BHNS curves
    plunging near log10 Z = -1.7.  The caption attributes ours to the same
    cause, so pin that the cut is not doing it.
    """
    from grb_physics import M_TOV, cold_stable_ns_mask

    Z_SUN = 0.0142
    bhns = load_bhns(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    Z, w = bhns["metallicity"], bhns["weights"]
    keep = cold_stable_ns_mask(bhns["M_NS"], m_tov=M_TOV)

    grid = np.unique(Z)
    before = grid[np.argmin(np.abs(grid / Z_SUN - 1.39))]
    after = grid[np.argmin(np.abs(grid / Z_SUN - 1.55))]
    n_before, n_after = int((Z == before).sum()), int((Z == after).sum())

    assert n_before / n_after > 30, "the high-Z BHNS cliff is gone from the sample"
    assert w[Z == before].sum() / w[Z == after].sum() > 10
    removed = 1 - w[(Z == after) & keep].sum() / w[Z == after].sum()
    assert removed == pytest.approx(0.0, abs=1e-12), (
        "the cut now removes weight at the cliff, so the caption's attribution "
        "of the collapse to COMPAS alone no longer holds"
    )


# ─────────────────────────────────────────────────────────────────────
# Absolute scale of eta(Z), the quantity Section 5 compares to the
# published formation-efficiency figures.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
def test_total_efficiency_absolute_scale_on_modelA(bns_a_path, bhns_a_path):
    """Peak height and position of the total eta(Z) on the native 53-point grid.

    Every other test in this file is scale-free: they pass
    ``mean_mass_evolved=1.0`` or ``1e9`` and check partitions, additivity
    and shape, all of which are invariant under the denominator.  That
    left the one number the section is actually read for unpinned.  The
    ordinate of Section 5 is compared point by point against Broekgaarden
    et al. (2021) Fig. 5 and Broekgaarden et al. (2022) Fig. 1, and that
    comparison only means anything if the absolute normalisation is right.

    The denominator is ``simulation_star_forming_mass()``: 53 grid points x
    about 1e6 binaries x 79.11 Msun of Universe star formation per
    simulated binary, split equally over the grid because Broekgaarden et
    al. (2021) Table 1 draw the same number of binaries at every Z_i.
    Nothing is fitted, so the peak height below is a prediction rather
    than a calibration.

    Two claims are pinned:

    1. The BHNS yield peaks at Z = Zsun/4.7, which is the Z_i ~ Zsun/5
       peak Broekgaarden et al. (2021) quote in their Sec. 3.3.  The
       curve is drawn on the native grid, so the peak sits on a simulated
       metallicity and cannot be displaced by a binning choice.
    2. The peak heights, 9.5e-6 (BHNS) and 4.1e-6 (BNS) mergers per Msun,
       which are the values Section 5 states in prose.

    A drift here means the star-forming-mass normalisation, the
    equal-mass-per-grid-point split, or the cold-stability cut changed,
    any of which moves the whole panel off the published ordinate.
    """
    grid = METALLICITY_GRID[np.concatenate(([True], np.diff(METALLICITY_GRID) != 0))]
    m_sf = simulation_star_forming_mass(n_metallicities=grid.size)
    eta_kw = dict(
        Z_max=None,
        Z_cut=Z_CE_LAMBDA_BIFURCATION,
        n_min=0,
        ess_min=0.0,
        n_bins=grid.size,
    )

    def _total_eta(data, mask):
        data = filter_systems(data, mask)
        res = formation_efficiency_binned(
            grid,
            data["metallicity"],
            data["weights"],
            {"total": np.ones(data["weights"].size, dtype=bool)},
            mean_mass_evolved=m_sf,
            **eta_kw,
        )
        eta = res["eta"]["total"]
        j = int(np.nanargmax(eta))
        return float(eta[j]), float(res["Z"][j])

    bns = load_bns(path=bns_a_path)
    eta_bns, z_bns = _total_eta(bns, cold_stable_ns_mask(bns["m1"], bns["m2"], m_tov=M_TOV))
    bhns = load_bhns(path=bhns_a_path)
    eta_bhns, z_bhns = _total_eta(bhns, cold_stable_ns_mask(bhns["M_NS"], m_tov=M_TOV))

    # Broekgaarden+ 2021 Sec. 3.3: the BHNS formation efficiency peaks near
    # Z_i ~ Zsun/5.  Accept Zsun/4 to Zsun/6, which spans several native
    # grid points either side of the quoted value.
    assert Z_SUN / 6.0 <= z_bhns <= Z_SUN / 4.0, (
        f"BHNS eta(Z) peaks at Z = {z_bhns:.5g} = Zsun/{Z_SUN / z_bhns:.1f}, "
        f"outside the Zsun/5 peak Broekgaarden+ 2021 Sec. 3.3 report "
        f"(accepted Zsun/6 to Zsun/4)."
    )
    assert z_bns < z_bhns, (
        f"BNS eta(Z) should peak at lower metallicity than BHNS; got "
        f"Z_BNS = {z_bns:.5g} against Z_BHNS = {z_bhns:.5g}."
    )

    # Peak heights, 5 percent regression band on the values Section 5 quotes.
    assert eta_bhns == pytest.approx(9.5e-6, rel=0.05), (
        f"BHNS peak eta = {eta_bhns:.4g} /Msun drifted from the 9.5e-6 "
        f"Section 5 quotes.  Check simulation_star_forming_mass, the "
        f"equal-mass-per-grid-point split, and the cold-stability cut."
    )
    assert eta_bns == pytest.approx(4.1e-6, rel=0.05), (
        f"BNS peak eta = {eta_bns:.4g} /Msun drifted from the 4.1e-6 Section 5 quotes."
    )
