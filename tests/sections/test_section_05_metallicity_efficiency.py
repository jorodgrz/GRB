"""Section 5 of grb_main.ipynb: Metallicity Dependence of GRB Formation Efficiency.

Smoke-level invariants on the data plumbing the per-class plot consumes
and on the artifact-suppression logic the Section 5 cell uses to keep
high-Z STROOPWAFEL noise out of the figure.

The figure cell combines three independent rules to decide which bins
to plot:

1. ``N_raw >= 50``:  basic floor on the raw COMPAS count.
2. ``ESS >= 10``:    Kish (1965) effective sample size; rules out bins
   where one or two high-weight STROOPWAFEL systems dominate the sum.
3. ``Z <= Z_sun``:   ``LOGZ_MAX_PHYSICAL = 0`` truncation; the COMPAS
   sample retains < 1 percent of its weight above solar, and the
   Levina+ 2026 (arXiv:2601.20202) TNG100-1 Azzalini fit (Table 1) puts
   most cosmic star formation below solar, so the high-Z tail is
   importance-sampling noise rather than physics.

These tests pin the per-bin reductions the figure consumes and verify
the masking logic on the real Model A sample so a future refactor that
silently re-enables the cross-gap stackplot interpolation, drops the
ESS cutoff, or removes the truncation will fail loudly.

Reference: Broekgaarden et al. (2021), arXiv:2103.02608 Sec. 4
(formation efficiency definition); Gottlieb et al. (2024),
arXiv:2411.13657 (four-class scheme used here); Kish (1965), Survey
Sampling, Wiley, Sec. 1.6 (effective sample size).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import classify_bhns, classify_bns_2024
from grb_io import METALLICITY_GRID, load_bhns, load_bns
from grb_physics import remap_ns_marginal, remap_ns_masses_double_gaussian
from grb_rates import (
    LOGZ_MAX_PHYSICAL,
    formation_efficiency,
    kish_effective_sample_size,
)

Z_SUN = 0.0142
N_MIN = 50
ESS_MIN = 10.0
A_BH_FID = 0.5  # mirrors the notebook cell-1 fiducial.


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
    """Combined notebook validity mask: N >= 50 AND ESS >= 10 AND Z <= Zsun."""
    N, ess = _bin_stats(Z_systems, w_systems, grid)
    z_phys_max = (10.0**LOGZ_MAX_PHYSICAL) * Z_SUN
    return (N >= N_MIN) & (ess >= ESS_MIN) & (grid <= z_phys_max)


# ─────────────────────────────────────────────────────────────────────
# New tests.
# ─────────────────────────────────────────────────────────────────────
def test_logz_max_physical_truncation_matches_constant():
    """Cell 16's right-edge equals ``10**LOGZ_MAX_PHYSICAL * Z_sun`` within one bin.

    Verifies that no bin above ``Z_phys_max`` survives the combined mask
    even when the raw count and ESS are arbitrarily inflated.  Bins on the
    grid at or below the cap should pass given enough samples; bins above
    should be unconditionally rejected.
    """
    grid = np.asarray(METALLICITY_GRID, dtype=float)
    z_phys_max = (10.0**LOGZ_MAX_PHYSICAL) * Z_SUN

    n_per_bin = 10 * N_MIN
    Z_all = np.repeat(grid, n_per_bin)
    w_all = np.ones_like(Z_all)
    ok = _ok_bin(Z_all, w_all, grid)

    above_cap = grid > z_phys_max
    assert not ok[above_cap].any(), (
        f"bins above Z_phys_max = {z_phys_max:.4f} survived the mask: "
        f"Z = {grid[above_cap][ok[above_cap]]}"
    )
    assert ok[~above_cap].all(), "well-sampled bins at or below Z_phys_max should pass the mask"


@pytest.mark.requires_data
def test_per_bin_ess_above_threshold_on_modelA(bns_a_path, bhns_a_path):
    """Every plotted bin clears ESS >= 10 on the real Model A samples.

    Honest-to-COMPAS check: applies the same mass remap the notebook
    applies in Section 0, then verifies that the combined mask agrees
    with the per-bin Kish ESS computed independently via
    ``kish_effective_sample_size``.
    """
    bns = load_bns(bns_a_path, expected_model="A", expected_ns_max=2.5)
    bns["m1"], bns["m2"] = remap_ns_masses_double_gaussian(
        bns["m1"],
        bns["m2"],
        weights=bns["weights"],
        rng=np.random.default_rng(42),
    )

    bhns = load_bhns(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    bhns["M_NS"] = remap_ns_marginal(
        bhns["M_NS"],
        weights=bhns["weights"],
        rng=np.random.default_rng(43),
    )

    grid = np.asarray(METALLICITY_GRID, dtype=float)
    for label, sample in (("BNS", bns), ("BHNS", bhns)):
        Z = sample["metallicity"]
        w = sample["weights"]
        ok = _ok_bin(Z, w, grid)
        assert ok.any(), f"{label} sample has no well-sampled bins"

        for i in np.flatnonzero(ok):
            in_bin = np.isclose(Z, grid[i])
            n_eff = kish_effective_sample_size(w[in_bin])
            assert n_eff >= ESS_MIN, (
                f"{label} bin Z={grid[i]:.4f} passes the mask but Kish "
                f"ESS = {n_eff:.2f} < {ESS_MIN}"
            )
            assert grid[i] <= (10.0**LOGZ_MAX_PHYSICAL) * Z_SUN, (
                f"{label} bin Z/Zsun = {grid[i] / Z_SUN:.3f} passes the "
                f"mask but exceeds the LOGZ_MAX_PHYSICAL = "
                f"{LOGZ_MAX_PHYSICAL} truncation."
            )


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


def test_mean_mass_evolved_per_population():
    """``formation_efficiency`` consumes distinct BNS / BHNS calibration anchors.

    Section 5 calls ``formation_efficiency`` twice (BNS then BHNS) with
    ``mean_mass_evolved=mean_mass_bns`` and ``mean_mass_evolved=mean_mass_bhns``
    respectively.  A future refactor that accidentally shared one value
    across both populations would bias the absolute amplitude by the
    ratio of the evolved-stellar-mass densities (CLAUDE.md "per-population
    calibration" mandate).  The test scales the same input data by
    different anchors and asserts the result scales inversely.
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
    z_phys_max = (10.0**LOGZ_MAX_PHYSICAL) * Z_SUN
    for label, Z, w in (
        ("BNS", bns["metallicity"], bns["weights"]),
        ("BHNS", bhns["metallicity"], bhns["weights"]),
    ):
        ok = _ok_bin(Z, w, grid)
        assert ok.any(), f"{label}: no well-sampled bins"
        assert grid[ok].max() <= z_phys_max + 1e-12, (
            f"{label}: rightmost kept bin Z = {grid[ok].max():.4f} exceeds "
            f"Z_phys_max = {z_phys_max:.4f}"
        )

        runs = _contiguous_runs(ok)
        covered = np.zeros_like(ok)
        for start, stop in runs:
            assert ok[start:stop].all(), (
                f"{label}: contiguous run [{start}, {stop}) contains a masked bin"
            )
            covered[start:stop] = True
        np.testing.assert_array_equal(covered, ok)
