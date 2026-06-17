"""Unit invariants for the Section 7c / 8d per-class per-channel splits.

The two notebook cells run a Cartesian-product split of the Gottlieb
GRB class masks and the Broekgaarden formation-channel masks, with an
empty-cell short-circuit and a "drop all-zero channels from the legend"
filter on top.  This file tests those building blocks on synthetic
input so a regression trips on every PR via ``make smoke``, instead of
only when the notebook is executed end to end.

Five helpers under test:

1. additivity over channels per class (tripwire A);
2. additivity over classes per channel (tripwire B);
3. the ``if not m.any(): zeros`` short-circuit for empty cells;
4. the ``CH_PLOT`` legend filter, ``[ch for ch in keys if any(...)]``;
5. ``apply_bhns_misalignment`` commutes with the channel split.

The integration-level versions (with live Model A + COMPAS + MSSFR)
live in ``tests/sections/test_section_07c_*.py`` and ``..._08d_*.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_physics import MISALIGNMENT_SYSTEMATIC_FACTOR
from grb_rates import apply_bhns_misalignment


# ─────────────────────────────────────────────────────────────────────
# Shared FCI rate-grid fixture for the partition / non-negativity tests
# ─────────────────────────────────────────────────────────────────────
def _rate_inputs(seed: int = 0):
    """Build a small but realistic input bundle for ``compute_merger_rate``.

    Returns a dict of arrays plus a ``common`` kwargs dict so each test
    can call ``compute_merger_rate(COMPAS_Z=..., COMPAS_delay_times=...,
    COMPAS_weights=..., **common)`` per subset.

    Uses FCI's coherent ``(redshifts, times, time_first_SF)`` grid so the
    internal ``times_to_z`` interpolation never runs out of bounds.
    """
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
        find_metallicity_distribution,
        find_sfr,
    )

    rng = np.random.default_rng(seed)
    n = 60

    Z_grid = np.array([1e-4, 1e-3, 1e-2, 3e-2])
    Z = rng.choice(Z_grid, size=n)
    delays = rng.uniform(50.0, 5000.0, size=n)
    w = rng.uniform(0.5, 2.0, size=n)

    redshifts, _, times, time_first_SF, _, _ = calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.1, cosmology=Planck15
    )
    sfr = find_sfr(redshifts)
    dPdlogZ, mets, p_draw = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=float(np.log(Z_grid[0])),
        max_logZ_COMPAS=float(np.log(Z_grid[-1])),
    )

    common = dict(
        redshifts=redshifts,
        times=times,
        time_first_SF=time_first_SF,
        n_formed=sfr,
        p_draw=float(p_draw),
        dPdlogZ=dPdlogZ,
        metallicities=mets,
        smooth_sigma=0,
    )
    return {"Z": Z, "delays": delays, "w": w, "redshifts": redshifts, "common": common}


def _per_cell_rate(cl_mask, ch_mask, inputs):
    """Inline the Section 7c / 8d cell pattern for one (class, channel) cell.

    Mirrors the cell verbatim:

        m = cl_mask & ch_mask
        if not m.any():
            return np.zeros_like(redshifts)
        return compute_merger_rate(COMPAS_*=...[m], ...)
    """
    from grb_rates import compute_merger_rate

    m = cl_mask & ch_mask
    if not m.any():
        return np.zeros_like(inputs["redshifts"])
    return compute_merger_rate(
        COMPAS_Z=inputs["Z"][m],
        COMPAS_delay_times=inputs["delays"][m],
        COMPAS_weights=inputs["w"][m],
        **inputs["common"],
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Additivity tripwires A and B on synthetic input
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_compas
def test_class_channel_split_partition_synthetic():
    """sum_ch per class == per-class total, sum_cl per channel == per-channel total.

    Promotes the in-cell ``assert np.allclose(...)`` tripwires (Section 7c
    cells ~line 32 and ~line 39) to a pytest-level invariant on a
    synthetic 3 x 3 contingency table.  ``compute_merger_rate`` is a pure
    additive accumulator over its ``COMPAS_*`` axis, so disjoint masks
    that union to the full sample must produce rates that add to the
    all-sample rate at machine precision.
    """
    from grb_rates import compute_merger_rate

    inputs = _rate_inputs(seed=1)
    n = inputs["Z"].size

    # 3 classes partitioning the sample (disjoint, exhaustive).
    cls_id = np.arange(n) % 3
    cl_masks = {f"cl{i}": cls_id == i for i in range(3)}
    # 3 channels partitioning the sample, independent of the classes.
    ch_id = (np.arange(n) // 4) % 3
    ch_masks = {f"ch{j}": ch_id == j for j in range(3)}

    R_cell = {
        cl_lbl: {ch_lbl: _per_cell_rate(cl_m, ch_m, inputs) for ch_lbl, ch_m in ch_masks.items()}
        for cl_lbl, cl_m in cl_masks.items()
    }

    R_per_class = {
        cl_lbl: compute_merger_rate(
            COMPAS_Z=inputs["Z"][cl_m],
            COMPAS_delay_times=inputs["delays"][cl_m],
            COMPAS_weights=inputs["w"][cl_m],
            **inputs["common"],
        )
        for cl_lbl, cl_m in cl_masks.items()
    }
    R_per_channel = {
        ch_lbl: compute_merger_rate(
            COMPAS_Z=inputs["Z"][ch_m],
            COMPAS_delay_times=inputs["delays"][ch_m],
            COMPAS_weights=inputs["w"][ch_m],
            **inputs["common"],
        )
        for ch_lbl, ch_m in ch_masks.items()
    }

    for cl_lbl in cl_masks:
        R_sum_ch = sum(R_cell[cl_lbl].values())
        np.testing.assert_allclose(
            R_sum_ch,
            R_per_class[cl_lbl],
            rtol=1e-12,
            atol=0,
            err_msg=f"tripwire A failed for {cl_lbl!r}",
        )

    for ch_lbl in ch_masks:
        R_sum_cl = sum(R_cell[cl_lbl][ch_lbl] for cl_lbl in cl_masks)
        np.testing.assert_allclose(
            R_sum_cl,
            R_per_channel[ch_lbl],
            rtol=1e-12,
            atol=0,
            err_msg=f"tripwire B failed for {ch_lbl!r}",
        )


# ─────────────────────────────────────────────────────────────────────
# 2. Empty-cell short-circuit (no compas import path)
# ─────────────────────────────────────────────────────────────────────
def test_class_channel_empty_cell_short_circuits_to_zero():
    """An empty (class, channel) intersection yields a zero R(z) array.

    The cell guards against ``compute_merger_rate`` being called on a
    zero-length sample via ``if not m.any(): per_ch[ch] = zeros_like(z)``.
    This test asserts the guard fires and the returned array has the
    right shape and is exactly zero (not NaN, not 1e-300).  No COMPAS
    dependency: only the ``not m.any()`` branch executes.
    """
    redshifts = np.linspace(0.0, 10.0, 101)
    cl_mask = np.array([True, False, True, False])
    ch_mask = np.array([False, True, False, True])
    assert not (cl_mask & ch_mask).any(), "test setup broken: cells overlap"

    # Stand-in ``inputs`` dict; compute_merger_rate is never reached.
    inputs = {
        "Z": np.array([1e-3, 1e-3, 1e-3, 1e-3]),
        "delays": np.array([100.0, 200.0, 300.0, 400.0]),
        "w": np.array([1.0, 1.0, 1.0, 1.0]),
        "redshifts": redshifts,
        "common": None,
    }
    out = _per_cell_rate(cl_mask, ch_mask, inputs)
    assert out.shape == redshifts.shape
    assert np.all(out == 0.0), out


# ─────────────────────────────────────────────────────────────────────
# 3. All (class, channel) cells return non-negative R(z)
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_compas
def test_class_channel_all_cells_nonnegative():
    """Every (class, channel) R(z) curve is non-negative at every z.

    The y-axis of both 7c and 8d figures is log-scaled, so a negative
    value (e.g. from a future smoothing pathology or a wrong sign in
    apply_bhns_misalignment) would silently drop a curve out of view.
    Random 3 x 3 contingency over 60 synthetic systems; ``smooth_sigma=0``
    so the test isolates ``compute_merger_rate``'s own positivity rather
    than the Gaussian-filter boundary behaviour.
    """
    inputs = _rate_inputs(seed=2)
    n = inputs["Z"].size
    rng = np.random.default_rng(3)
    cls_id = rng.integers(0, 3, size=n)
    ch_id = rng.integers(0, 3, size=n)
    cl_masks = [cls_id == i for i in range(3)]
    ch_masks = [ch_id == j for j in range(3)]

    for i, cl_m in enumerate(cl_masks):
        for j, ch_m in enumerate(ch_masks):
            R = _per_cell_rate(cl_m, ch_m, inputs)
            assert np.all(R >= 0.0), f"cell (cl{i}, ch{j}) has negative R(z): min={R.min():.3e}"


# ─────────────────────────────────────────────────────────────────────
# 4. CH_PLOT legend filter (drops all-zero channels, preserves order)
# ─────────────────────────────────────────────────────────────────────
def test_ch_plot_filter_drops_only_all_zero_channels():
    """The Section 7c / 8d ``CH_PLOT`` list comprehension.

    The cells filter the legend down to channels with at least one
    nonzero curve in any class via

        [ch for ch in CH_KEYS if any(R[cl][ch].max() > 0 for cl in CL)]

    Asserts that (i) an all-zero channel is dropped, (ii) channels with
    at least one nonzero class curve are kept, and (iii) the surviving
    order matches the input order so the legend ordering stays stable
    across notebook runs.
    """
    z = np.linspace(0.0, 10.0, 50)
    R_by_cell = {
        "cl_a": {"I": np.ones_like(z), "II": np.zeros_like(z), "IV": 0.5 * np.ones_like(z)},
        "cl_b": {"I": 2 * np.ones_like(z), "II": np.zeros_like(z), "IV": np.zeros_like(z)},
    }
    keys = ["I", "II", "IV"]
    classes = ["cl_a", "cl_b"]

    survivors = [ch for ch in keys if any(R_by_cell[cl][ch].max() > 0 for cl in classes)]
    assert survivors == ["I", "IV"]

    # Edge case: every channel zero -> empty legend (cell still renders
    # the dotted per-class total).
    R_zero = {cl: {ch: np.zeros_like(z) for ch in keys} for cl in classes}
    empty = [ch for ch in keys if any(R_zero[cl][ch].max() > 0 for cl in classes)]
    assert empty == []


# ─────────────────────────────────────────────────────────────────────
# 5. apply_bhns_misalignment commutes with the channel split
# ─────────────────────────────────────────────────────────────────────
def test_misalignment_commutes_with_channel_split():
    """Scalar misalignment factor commutes with the (class, channel) sum.

    Section 8d applies ``apply_bhns_misalignment`` per cell (cell line
    ~2580) and the in-cell tripwire then asserts

        sum_ch apply_bhns_misalignment(R_intrinsic_cell)
        == apply_bhns_misalignment(sum_ch R_intrinsic_cell)

    which only holds because the misalignment correction is a scalar
    multiplier (Fragos+ 2010, Kawaguchi+ 2015, see
    ``grb_physics.MISALIGNMENT_SYSTEMATIC_FACTOR``).  This test pins the
    commutativity in isolation, so a future change that makes the
    correction redshift- or mass-dependent will trip here before it
    silently breaks the per-cell decomposition.
    """
    rng = np.random.default_rng(4)
    z = np.linspace(0.0, 10.0, 50)
    cells = [rng.uniform(0.1, 5.0, size=z.size) for _ in range(7)]

    lhs = sum(apply_bhns_misalignment(c) for c in cells)
    rhs = apply_bhns_misalignment(sum(cells))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=0)

    # Scalar factor matches the literature constant (the docstring above
    # only holds if the factor stays scalar; pin the value here so a
    # silent factor change is caught).
    assert apply_bhns_misalignment(1.0) == pytest.approx(MISALIGNMENT_SYSTEMATIC_FACTOR)
