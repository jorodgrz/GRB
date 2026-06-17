"""Behaviour and edge-case tests for ``grb_rates``.

Complements ``tests/unit/test_rates.py`` (compute_merger_rate /
detected_rate / calibrate_mean_mass_evolved / Wanderman-Piran /
Kroupa / Kish / beaming round-trips),
``tests/unit/test_phase4_helpers.py`` (marginalize_bh_spin /
observed_frame_rate / compute_eos_sensitivity / beamed_class_comparison),
``tests/sections/test_section_10_beaming.py`` (beaming + CLASS_THETA_J),
``tests/sections/test_section_05_metallicity_efficiency.py``
(formation_efficiency real-data), and the literature-anchor block in
``tests/anchors/test_literature_anchors.py``.

This file fills the gaps: the legacy ``marginalize`` helper,
``apply_bhns_misalignment``, ``frac4``, ``rate_label``,
``mcrit_sweep``, edge cases of the FCI wrappers, default behaviour of
``beamed_class_comparison``, and schema integrity of the two
published reference dicts (`OBSERVED_SGRB_RATES`,
`OBSERVED_RATES_BY_CLASS`).

References cited inline per test: Webbink (1984) ApJ 277, 355 (CE
energy formalism); Hurley et al. (2002) MNRAS 329, 897 (CE stability);
Levina et al. (2026) arXiv:2601.20202 (TNG MSSFR / SFR fits);
Broekgaarden et al. (2021) arXiv:2103.02608 (alpha_CE per model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# marginalize (legacy dict-mode helper)
# ---------------------------------------------------------------------------
def test_marginalize_dict_weighted_sum_identity():
    """``marginalize`` returns ``sum_a w_a * rate[a]`` over a spin dict.

    A 3-spin dict with weights summing to 1 must produce the
    weighted-average rate to machine precision.
    """
    from grb_rates import marginalize

    rates = {0.0: 1.0, 0.5: 2.0, 0.9: 4.0}
    weights = {0.0: 0.2, 0.5: 0.5, 0.9: 0.3}
    expected = 0.2 * 1.0 + 0.5 * 2.0 + 0.3 * 4.0
    assert marginalize(rates, weights) == pytest.approx(expected, rel=1e-12)


def test_marginalize_dict_array_elements_broadcast():
    """Per-spin rates may be 1-D arrays; the result broadcasts elementwise."""
    from grb_rates import marginalize

    rates = {
        0.0: np.array([1.0, 2.0]),
        0.5: np.array([3.0, 4.0]),
        0.9: np.array([5.0, 6.0]),
    }
    weights = {0.0: 0.2, 0.5: 0.5, 0.9: 0.3}
    expected = 0.2 * np.array([1.0, 2.0]) + 0.5 * np.array([3.0, 4.0]) + 0.3 * np.array([5.0, 6.0])
    np.testing.assert_allclose(marginalize(rates, weights), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# apply_bhns_misalignment
# ---------------------------------------------------------------------------
def test_apply_bhns_misalignment_default_factor_matches_grb_physics_constant():
    """``apply_bhns_misalignment`` defaults to ``MISALIGNMENT_SYSTEMATIC_FACTOR``."""
    from grb_physics import MISALIGNMENT_SYSTEMATIC_FACTOR
    from grb_rates import apply_bhns_misalignment

    rate = 10.0
    assert apply_bhns_misalignment(rate) == pytest.approx(
        rate * MISALIGNMENT_SYSTEMATIC_FACTOR, rel=1e-12
    )
    rate_arr = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        apply_bhns_misalignment(rate_arr),
        rate_arr * MISALIGNMENT_SYSTEMATIC_FACTOR,
        rtol=1e-12,
    )


def test_apply_bhns_misalignment_custom_factor_overrides_default():
    """Explicit ``factor=`` kwarg overrides the default and scales linearly."""
    from grb_rates import apply_bhns_misalignment

    rate = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        apply_bhns_misalignment(rate, factor=0.3),
        rate * 0.3,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# frac4
# ---------------------------------------------------------------------------
def test_frac4_four_components_sum_to_100_percent():
    """The four returned arrays sum to 100 (percent) entry-by-entry."""
    from grb_rates import frac4

    s, lo, bs, bl = frac4(
        np.array([1.0, 2.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    np.testing.assert_allclose(s + lo + bs + bl, np.array([100.0, 100.0]), rtol=1e-12)


def test_frac4_zero_total_returns_nan_no_zero_division():
    """A row with all-zero totals returns NaN for every component."""
    from grb_rates import frac4

    s, lo, bs, bl = frac4(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    for arr in (s, lo, bs, bl):
        assert np.isnan(arr[0]), arr


def test_frac4_scalar_inputs_return_scalar_outputs():
    """Scalar inputs give scalar outputs in percent."""
    from grb_rates import frac4

    s, lo, bs, bl = frac4(1.0, 1.0, 1.0, 1.0)
    assert s == pytest.approx(25.0)
    assert lo == pytest.approx(25.0)
    assert bs == pytest.approx(25.0)
    assert bl == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# rate_label
# ---------------------------------------------------------------------------
def test_rate_label_integer_format_for_values_above_one():
    """``rate_label(val)`` uses thousand-separator integer format when ``val >= 1``."""
    from grb_rates import rate_label

    assert rate_label(123.4) == "123"
    # Use a non-half value to avoid banker's rounding (Python's :,.0f
    # rounds 1234.5 to 1,234, not 1,235).
    assert rate_label(1234.7) == "1,235"


def test_rate_label_two_decimal_format_for_values_below_one():
    """``rate_label(val)`` uses two-decimal format when ``val < 1``."""
    from grb_rates import rate_label

    assert rate_label(0.05) == "0.05"
    assert rate_label(0.5) == "0.50"


def test_rate_label_boundary_at_one_uses_integer_format():
    """At exactly ``val == 1.0`` the ``>=`` branch fires and gives integer format."""
    from grb_rates import rate_label

    assert rate_label(1.0) == "1"


# ---------------------------------------------------------------------------
# mcrit_sweep
# ---------------------------------------------------------------------------
def _mcrit_sweep_inputs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    m_tot = rng.uniform(2.0, 3.5, size=n)
    q = rng.uniform(1.0, 1.5, size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    return m_tot, q, w


def test_mcrit_sweep_default_range_has_50_points():
    """``M_crit_range=None`` produces a default 50-point sweep spanning 2.3-3.5 Msun.

    Pinned by the docstring: 50 points dense enough to resolve the
    transition near the fiducial 2.8 Msun.
    """
    from grb_rates import mcrit_sweep

    m_tot, q, w = _mcrit_sweep_inputs()
    _, _, _, mc_range = mcrit_sweep(m_tot, q, w)
    assert mc_range.shape == (50,)
    assert mc_range[0] == pytest.approx(2.3, rel=1e-12)
    assert mc_range[-1] == pytest.approx(3.5, rel=1e-12)


def test_mcrit_sweep_returns_four_arrays_with_matching_shapes():
    """``mcrit_sweep`` returns ``(frac_I, frac_II, frac_L, M_crit_range)``
    arrays of identical shape."""
    from grb_rates import mcrit_sweep

    m_tot, q, w = _mcrit_sweep_inputs()
    out = mcrit_sweep(m_tot, q, w)
    assert len(out) == 4
    shape = out[0].shape
    for arr in out[1:]:
        assert arr.shape == shape


def test_mcrit_sweep_per_Mc_fractions_partition_unity():
    """The three class fractions sum to 1 at every M_crit point.

    The implementation uses ``M_tot < Mc`` and the complementary
    ``M_tot >= Mc`` split, with the latter sub-divided by ``q``.
    Every system therefore lands in exactly one class regardless of
    M_crit, so the three fractions partition unity to machine
    precision.
    """
    from grb_rates import mcrit_sweep

    m_tot, q, w = _mcrit_sweep_inputs()
    frac_I, frac_II, frac_L, _ = mcrit_sweep(m_tot, q, w)
    np.testing.assert_allclose(frac_I + frac_II + frac_L, 1.0, rtol=1e-12)


def test_mcrit_sweep_short_I_fraction_monotone_in_Mc():
    """The ``M_tot < Mc`` fraction is non-decreasing in M_crit.

    Cumulative mass invariant: raising M_crit can only pull more
    systems into the ``M_tot < Mc`` class.  A regression that
    introduced a sort or sign flip would break this monotonicity.
    """
    from grb_rates import mcrit_sweep

    m_tot, q, w = _mcrit_sweep_inputs(n=500, seed=11)
    frac_I, _, _, _ = mcrit_sweep(m_tot, q, w)
    assert (np.diff(frac_I) >= -1e-12).all(), (
        f"Short-I fraction not monotone non-decreasing in M_crit; "
        f"min diff = {np.diff(frac_I).min():.3e}"
    )


# ---------------------------------------------------------------------------
# formation_efficiency masks=None branch
# ---------------------------------------------------------------------------
def test_formation_efficiency_masks_none_returns_only_total_key():
    """Omitting ``masks=`` returns a single-key dict ``{'total': ...}``.

    Each bin is the weighted count divided by the per-metallicity
    star-forming mass ``mean_mass_evolved * f_i`` (intensive efficiency),
    not by the bare scalar total.
    """
    from grb_rates import formation_efficiency, metallicity_prior_mass_fraction

    metallicityGrid = np.array([0.001, 0.005, 0.01])
    Z_all = np.array([0.001, 0.001, 0.005, 0.01])
    w_all = np.array([1.0, 1.0, 1.0, 1.0])
    mme = 4.0
    eff = formation_efficiency(metallicityGrid, Z_all, w_all, mean_mass_evolved=mme)
    assert set(eff.keys()) == {"total"}
    assert eff["total"].shape == metallicityGrid.shape
    # bin 0 holds 2 of 4 unit-weight systems; the intensive efficiency
    # divides by the per-Z star-forming mass mean_mass_evolved * f_0.
    f = metallicity_prior_mass_fraction(metallicityGrid)
    assert eff["total"][0] == pytest.approx(2.0 / (mme * f[0]), rel=1e-12)


# ---------------------------------------------------------------------------
# metallicity_prior_mass_fraction
# ---------------------------------------------------------------------------
def test_metallicity_prior_mass_fraction_normalizes_and_preserves_order():
    """``f_i`` sums to 1, follows the input ordering, and is positive."""
    from grb_rates import metallicity_prior_mass_fraction

    grid = np.geomspace(1e-4, 0.03, 53)
    f = metallicity_prior_mass_fraction(grid)
    assert f.shape == grid.shape
    assert np.all(f > 0.0)
    assert f.sum() == pytest.approx(1.0, rel=1e-12)

    # Reversing the input reverses the output (no implicit sort leak).
    f_rev = metallicity_prior_mass_fraction(grid[::-1])
    np.testing.assert_allclose(f_rev[::-1], f, rtol=1e-12)


def test_metallicity_prior_mass_fraction_log_uniform_grid_is_flat():
    """A log-uniform grid gives equal interior mass fractions.

    Under uniform-in-ln Z sampling the prior mass per grid point is the
    log-Z cell width, so a log-uniform grid yields f_i = 1/(N-1) for the
    interior points and a half-cell at each edge (default bounds clamp to
    the grid min/max).
    """
    from grb_rates import metallicity_prior_mass_fraction

    n = 11
    grid = np.geomspace(1e-4, 1e-1, n)
    f = metallicity_prior_mass_fraction(grid)
    np.testing.assert_allclose(f[1:-1], 1.0 / (n - 1), rtol=1e-12)
    assert f[0] == pytest.approx(0.5 / (n - 1), rel=1e-12)
    assert f[-1] == pytest.approx(0.5 / (n - 1), rel=1e-12)


def test_metallicity_prior_mass_fraction_rejects_bad_bounds():
    """Empty grid and inverted bounds raise ``ValueError``."""
    from grb_rates import metallicity_prior_mass_fraction

    with pytest.raises(ValueError):
        metallicity_prior_mass_fraction(np.array([]))
    with pytest.raises(ValueError):
        metallicity_prior_mass_fraction(np.geomspace(1e-4, 0.03, 5), z_min=0.03, z_max=1e-4)


# ---------------------------------------------------------------------------
# per_system_rate_weights edge cases
# ---------------------------------------------------------------------------
def test_per_system_rate_weights_empty_input_returns_length_zero_array():
    """``per_system_rate_weights`` short-circuits on empty input.

    The early return prevents an empty-array NaN from leaking into
    rate computations downstream.
    """
    from grb_rates import per_system_rate_weights

    out = per_system_rate_weights(
        z_target=0.5,
        redshifts=np.linspace(0, 5, 10),
        times=np.linspace(13700, 100, 10),
        time_first_SF=100.0,
        n_formed=np.ones(10),
        p_draw=0.1,
        dPdlogZ=np.zeros((10, 1)),
        metallicities=np.array([0.01]),
        COMPAS_Z=np.array([]),
        COMPAS_delay_times=np.array([]),
        COMPAS_weights=np.array([]),
    )
    assert out.shape == (0,)


def test_per_system_rate_weights_all_out_of_range_returns_zeros():
    """When every binary's ``z_form`` predates star formation, every weight
    is zero and the output shape matches the input.

    Setting all delay times above the time-grid range puts every
    ``t_form = t_merge - delay`` below ``time_first_SF``, so the
    ``valid`` mask is empty and the routine returns its zero
    pre-allocation.
    """
    from grb_rates import per_system_rate_weights

    redshifts = np.linspace(0.0, 5.0, 100)
    times = np.linspace(13700.0, 100.0, 100)
    time_first_SF = 13000.0
    delays = np.full(4, 1e6)  # delays much larger than any t_merge
    out = per_system_rate_weights(
        z_target=0.5,
        redshifts=redshifts,
        times=times,
        time_first_SF=time_first_SF,
        n_formed=np.ones_like(redshifts),
        p_draw=0.1,
        dPdlogZ=np.ones((len(redshifts), 1)),
        metallicities=np.array([0.01]),
        COMPAS_Z=np.full(4, 0.01),
        COMPAS_delay_times=delays,
        COMPAS_weights=np.ones(4),
    )
    assert out.shape == (4,)
    np.testing.assert_array_equal(out, np.zeros(4))


# ---------------------------------------------------------------------------
# beamed_rate_mixed empty dict
# ---------------------------------------------------------------------------
def test_beamed_rate_mixed_empty_dict_returns_zero():
    """An empty rates dict produces ``0.0`` without raising.

    Catches a refactor that mis-implements the empty-iter sum.
    """
    from grb_rates import beamed_rate_mixed

    assert beamed_rate_mixed({}, {}) == 0.0


# ---------------------------------------------------------------------------
# beamed_class_comparison defaults
# ---------------------------------------------------------------------------
def test_beamed_class_comparison_default_theta_mapping_uses_class_theta_j():
    """``theta_j_deg_by_class=None`` maps sbGRB to ``CLASS_THETA_J['sbGRB']['fid']``
    and the three lbGRB classes to ``CLASS_THETA_J['lbGRB']['fid']``."""
    from grb_rates import CLASS_THETA_J, beamed_class_comparison

    R_int = {
        "sbGRB + blue KN": 100.0,
        "lbGRB + red KN (HMNS)": 50.0,
        "lbGRB + red KN (disk)": 10.0,
        "Faint lbGRB": 20.0,
    }
    df = beamed_class_comparison(R_int)
    sb_fid = CLASS_THETA_J["sbGRB"]["fid"]
    lb_fid = CLASS_THETA_J["lbGRB"]["fid"]
    assert df.loc["sbGRB + blue KN", "theta_j_deg"] == pytest.approx(sb_fid)
    assert df.loc["lbGRB + red KN (HMNS)", "theta_j_deg"] == pytest.approx(lb_fid)
    assert df.loc["lbGRB + red KN (disk)", "theta_j_deg"] == pytest.approx(lb_fid)
    assert df.loc["Faint lbGRB", "theta_j_deg"] == pytest.approx(lb_fid)


def test_beamed_class_comparison_unmapped_class_returns_nan_observed():
    """A class absent from ``OBSERVED_RATES_BY_CLASS`` carries NaN
    observed-rate columns and an empty reference string."""
    from grb_rates import beamed_class_comparison

    R_int = {"unknown class": 5.0}
    df = beamed_class_comparison(R_int, theta_j_deg_by_class={"unknown class": 10.0})
    row = df.loc["unknown class"]
    assert np.isnan(row["R_obs"])
    assert np.isnan(row["R_obs_lo"])
    assert np.isnan(row["R_obs_hi"])
    assert row["reference"] == ""


# ---------------------------------------------------------------------------
# compute_eos_sensitivity edge cases
# ---------------------------------------------------------------------------
def test_compute_eos_sensitivity_all_zero_weights_raises_value_error():
    """All-zero weights raise the documented ``ValueError``."""
    from grb_rates import compute_eos_sensitivity

    with pytest.raises(ValueError, match="weights sum to zero"):
        compute_eos_sensitivity(
            np.array([1.4, 1.4]),
            np.array([1.3, 1.3]),
            np.array([0.0, 0.0]),
        )


def test_compute_eos_sensitivity_custom_k_thresh_propagates_to_M_thresh_column():
    """``k_thresh=`` propagates: every row has ``M_thresh = k_thresh * M_TOV``."""
    from grb_rates import compute_eos_sensitivity

    rng = np.random.default_rng(0)
    m1 = rng.uniform(1.2, 2.0, size=200)
    m2 = rng.uniform(1.1, 2.0, size=200)
    w = rng.uniform(0.5, 2.0, size=200)
    k = 1.30
    table = compute_eos_sensitivity(m1, m2, w, k_thresh=k)
    np.testing.assert_allclose(table["M_thresh"], k * table["M_TOV"], rtol=1e-12)


def test_compute_eos_sensitivity_single_eos_subset_returns_one_row():
    """Passing a one-entry ``eos_models`` dict produces a one-row DataFrame."""
    from grb_physics import EOS_MODELS
    from grb_rates import compute_eos_sensitivity

    m1 = np.array([1.4, 1.5, 1.6])
    m2 = np.array([1.3, 1.3, 1.3])
    w = np.array([1.0, 1.0, 1.0])
    single = {"APR4": EOS_MODELS["APR4"]}
    table = compute_eos_sensitivity(m1, m2, w, eos_models=single)
    assert table.shape[0] == 1
    assert list(table.index) == ["APR4"]


# ---------------------------------------------------------------------------
# wanderman_piran_2015_Rz and observed_frame_rate shape preservation
# ---------------------------------------------------------------------------
def test_wanderman_piran_2015_Rz_shape_preserved_for_array_input():
    """A length-N array input returns length-N ``'R_best'`` / ``'R_lo'`` / ``'R_hi'``."""
    from grb_rates import wanderman_piran_2015_Rz

    z = np.linspace(0.0, 3.0, 10)
    out = wanderman_piran_2015_Rz(z)
    for key in ("R_best", "R_lo", "R_hi"):
        assert out[key].shape == z.shape


def test_observed_frame_rate_shape_preserved_for_array_input():
    """1-D inputs of length N give a length-N output."""
    from grb_rates import observed_frame_rate

    R_int = np.linspace(10.0, 1.0, 8)
    z = np.linspace(0.0, 2.0, 8)
    out = observed_frame_rate(R_int, z)
    assert out.shape == R_int.shape
    # Sanity: R(z=0) = R_int(0), R(z=1) = R_int(1)/2.
    assert out[0] == pytest.approx(R_int[0])
    assert out[-1] == pytest.approx(R_int[-1] / (1.0 + z[-1]))


# ---------------------------------------------------------------------------
# OBSERVED_SGRB_RATES schema
# ---------------------------------------------------------------------------
def test_observed_sgrb_rates_schema_has_required_keys():
    """Every reference entry carries the four documented keys."""
    from grb_rates import OBSERVED_SGRB_RATES

    required = {"R_obs", "R_obs_lo", "R_obs_hi", "note"}
    for name, entry in OBSERVED_SGRB_RATES.items():
        missing = required - set(entry.keys())
        assert not missing, f"OBSERVED_SGRB_RATES[{name!r}] missing keys: {missing}"


def test_observed_sgrb_rates_bounds_lo_le_obs_le_hi():
    """``R_obs_lo <= R_obs <= R_obs_hi`` for every reference."""
    from grb_rates import OBSERVED_SGRB_RATES

    for name, entry in OBSERVED_SGRB_RATES.items():
        lo = entry["R_obs_lo"]
        ob = entry["R_obs"]
        hi = entry["R_obs_hi"]
        assert lo <= ob <= hi, (
            f"OBSERVED_SGRB_RATES[{name!r}] bound order broken: lo={lo}, obs={ob}, hi={hi}"
        )


# ---------------------------------------------------------------------------
# OBSERVED_RATES_BY_CLASS schema
# ---------------------------------------------------------------------------
def test_observed_rates_by_class_schema_has_required_keys():
    """Every class entry has the five documented keys."""
    from grb_rates import OBSERVED_RATES_BY_CLASS

    required = {"R_obs", "R_obs_lo", "R_obs_hi", "reference", "caveat"}
    for label, entry in OBSERVED_RATES_BY_CLASS.items():
        missing = required - set(entry.keys())
        assert not missing, f"OBSERVED_RATES_BY_CLASS[{label!r}] missing keys: {missing}"


def test_observed_rates_by_class_bounds_lo_le_obs_le_hi_where_finite():
    """``R_obs_lo <= R_obs <= R_obs_hi`` where all three values are finite.

    The 'lbGRB + red KN (disk)' and 'Faint lbGRB' classes carry NaN
    rates by design (no published observed local rate); skip them.
    """
    from grb_rates import OBSERVED_RATES_BY_CLASS

    for label, entry in OBSERVED_RATES_BY_CLASS.items():
        lo, ob, hi = entry["R_obs_lo"], entry["R_obs"], entry["R_obs_hi"]
        if not (np.isfinite(lo) and np.isfinite(ob) and np.isfinite(hi)):
            continue
        assert lo <= ob <= hi, (
            f"OBSERVED_RATES_BY_CLASS[{label!r}] bound order broken: lo={lo}, obs={ob}, hi={hi}"
        )


# ---------------------------------------------------------------------------
# Sanity-keep: pandas import is used inside beamed_class_comparison; the
# top-level import here guards against a refactor that drops pandas.
# ---------------------------------------------------------------------------
def _pandas_import_guard():  # pragma: no cover
    _ = pd.DataFrame
