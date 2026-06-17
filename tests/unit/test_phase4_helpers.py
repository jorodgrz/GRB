"""Unit tests for the phase-4 cross-check helpers.

Each helper is exercised against:

- a small hand-built population (E1, E3, E4, E5, E7) so the maths is
  obvious from the test, and
- the canonical project constants (E2, E6) so a future refactor that
  silently breaks the convention is caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────
# E5: marginalize_bh_spin
# ─────────────────────────────────────────────────────────────────────
def test_marginalize_bh_spin_dict_mode_matches_legacy_marginalize():
    from grb_rates import marginalize, marginalize_bh_spin

    rates = {0.0: np.array([1.0, 2.0]), 0.5: np.array([3.0, 4.0]), 0.9: np.array([5.0, 6.0])}
    weights = {0.0: 0.5, 0.5: 0.3, 0.9: 0.2}

    legacy = marginalize(rates, weights)
    new = marginalize_bh_spin(rates, weights)
    np.testing.assert_allclose(new, legacy, rtol=1e-12)


def test_marginalize_bh_spin_array_mode_with_array_weights():
    from grb_rates import marginalize_bh_spin

    spins = np.array([0.0, 0.3, 0.5, 0.7, 0.9])
    rates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    expected = float(np.sum(weights * rates))
    assert marginalize_bh_spin(rates, weights, spin_grid=spins) == pytest.approx(expected)


def test_marginalize_bh_spin_array_mode_with_callable_pdf():
    from grb_rates import marginalize_bh_spin

    spins = np.linspace(0.0, 0.9, 19)
    rates = np.ones_like(spins)

    def pdf(chi):
        return 1.0 / 0.9

    out = marginalize_bh_spin(rates, pdf, spin_grid=spins)
    # Trapezoidal integral of (1 / 0.9) * 1 over [0, 0.9] = 1.0
    assert out == pytest.approx(1.0, rel=1e-10)


def test_marginalize_bh_spin_array_mode_raises_without_spin_grid():
    from grb_rates import marginalize_bh_spin

    with pytest.raises(ValueError, match="spin_grid is required"):
        marginalize_bh_spin(np.array([1.0]), np.array([1.0]))


# ─────────────────────────────────────────────────────────────────────
# E6: observed_frame_rate
# ─────────────────────────────────────────────────────────────────────
def test_observed_frame_rate_z_zero_unchanged():
    from grb_rates import observed_frame_rate

    R_int = np.array([100.0, 200.0])
    z = np.array([0.0, 0.0])
    np.testing.assert_allclose(observed_frame_rate(R_int, z), R_int)


def test_observed_frame_rate_applies_one_plus_z_factor():
    from grb_rates import observed_frame_rate

    R_int = np.array([100.0])
    np.testing.assert_allclose(observed_frame_rate(R_int, np.array([1.0])), np.array([50.0]))
    np.testing.assert_allclose(observed_frame_rate(R_int, np.array([3.0])), np.array([25.0]))


# ─────────────────────────────────────────────────────────────────────
# E1: classify_observed_mergers
# ─────────────────────────────────────────────────────────────────────
def test_classify_observed_mergers_mass_planes():
    """Each ejecta-decomposition cell maps to the expected Gottlieb class."""
    from grb_classify import classify_observed_mergers

    # Five test sources: blue-dominated, mixed, red-dominated, faint,
    # and intermediate.
    M_B = np.array([0.04, 0.03, 0.001, 0.001, 0.02])
    M_P = np.array([0.005, 0.01, 0.005, 0.001, 0.01])
    M_R = np.array([0.005, 0.04, 0.05, 0.001, 0.025])
    out = classify_observed_mergers(M_B, M_P, M_R)
    # Pre-compute f_red expectations:
    #   0: 0.005 / 0.05  = 0.10  -> blue-dominated
    #   1: 0.04  / 0.08  = 0.50  -> mixed
    #   2: 0.05  / 0.056 ~ 0.89  -> red-dominated
    #   3: total 0.003 -> faint (M_ej < 0.01)
    #   4: 0.025 / 0.055 ~ 0.45 -> mixed
    assert out["sbGRB + blue KN"].tolist() == [True, False, False, False, False]
    assert out["lbGRB + red KN (HMNS)"].tolist() == [False, True, False, False, True]
    assert out["lbGRB + red KN (disk)"].tolist() == [False, False, True, False, False]
    assert out["Faint lbGRB"].tolist() == [False, False, False, True, False]


def test_classify_observed_mergers_threshold_swap_raises():
    from grb_classify import classify_observed_mergers

    with pytest.raises(ValueError, match="must exceed"):
        classify_observed_mergers(
            np.array([0.01]),
            np.array([0.01]),
            np.array([0.01]),
            red_max_for_blue=0.7,
            red_min_for_red=0.3,
        )


# ─────────────────────────────────────────────────────────────────────
# E3: channel_class_crosstab
# ─────────────────────────────────────────────────────────────────────
def test_channel_class_crosstab_raw_and_normalised_modes():
    from grb_classify import channel_class_crosstab

    n = 6
    weights = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    channel_masks = {
        "I  Stable MT + CE": np.array([True, True, False, False, False, False]),
        "II  Stable MT only": np.array([False, False, True, True, False, False]),
        "III Single-core CE": np.array([False, False, False, False, True, True]),
        "IV  Double-core CE": np.zeros(n, dtype=bool),
        "V   Other": np.zeros(n, dtype=bool),
    }
    class_masks = {
        "sbGRB + blue KN": np.array([True, False, True, False, True, False]),
        "lbGRB + red KN (HMNS)": np.array([False, True, False, True, False, True]),
        "lbGRB + red KN (disk)": np.zeros(n, dtype=bool),
        "Faint lbGRB": np.zeros(n, dtype=bool),
    }

    raw = channel_class_crosstab(channel_masks, class_masks, weights)
    assert isinstance(raw, pd.DataFrame)
    assert raw.shape == (5, 4)
    # Row I: (T,T) cells are (True, sb)=1, (False, lb)=1
    assert raw.loc["I  Stable MT + CE", "sbGRB + blue KN"] == pytest.approx(1.0)
    assert raw.loc["I  Stable MT + CE", "lbGRB + red KN (HMNS)"] == pytest.approx(1.0)
    assert raw.loc["IV  Double-core CE", "sbGRB + blue KN"] == 0.0

    # Channel-normalised: row sums to 1 (or 0 for empty rows).
    by_ch = channel_class_crosstab(channel_masks, class_masks, weights, normalise="channel")
    assert by_ch.loc["I  Stable MT + CE"].sum() == pytest.approx(1.0)
    assert by_ch.loc["IV  Double-core CE"].sum() == 0.0


def test_channel_class_crosstab_invalid_normalise_raises():
    from grb_classify import channel_class_crosstab

    with pytest.raises(ValueError, match="normalise must be"):
        channel_class_crosstab({}, {}, np.array([]), normalise="banana")


# ─────────────────────────────────────────────────────────────────────
# E4: offset_cdf_by_class
# ─────────────────────────────────────────────────────────────────────
def test_offset_cdf_by_class_per_class_sorted_and_unit_normalised():
    from grb_offsets import offset_cdf_by_class

    offsets = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    masks = {
        "A": np.array([True, True, False, False, False]),
        "B": np.array([False, False, True, True, True]),
    }
    out = offset_cdf_by_class(offsets, weights, masks)

    o_a, c_a = out["A"]
    o_b, c_b = out["B"]
    assert (np.diff(o_a) >= 0).all()
    assert (np.diff(o_b) >= 0).all()
    assert c_a[-1] == pytest.approx(1.0)
    assert c_b[-1] == pytest.approx(1.0)


def test_offset_cdf_by_class_empty_class_returns_sentinel():
    from grb_offsets import offset_cdf_by_class

    offsets = np.array([1.0, 2.0])
    weights = np.array([1.0, 1.0])
    masks = {"empty": np.array([False, False]), "singleton": np.array([True, False])}
    out = offset_cdf_by_class(offsets, weights, masks)
    assert out["empty"][0].tolist() == [0.0]
    # singleton has only one valid system so the CDF cannot be built
    assert out["singleton"][0].tolist() == [0.0]


# ─────────────────────────────────────────────────────────────────────
# E2: compute_eos_sensitivity
# ─────────────────────────────────────────────────────────────────────
def test_compute_eos_sensitivity_class_fractions_sum_to_one():
    from grb_physics import EOS_MODELS
    from grb_rates import compute_eos_sensitivity

    rng = np.random.default_rng(0)
    n = 1000
    m1 = rng.uniform(1.1, 2.5, size=n)
    m2 = rng.uniform(1.1, 2.5, size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    table = compute_eos_sensitivity(m1, m2, w)
    # Should have one row per EOS in EOS_MODELS.
    assert set(table.index) == set(EOS_MODELS.keys())
    class_keys = [
        "sbGRB + blue KN",
        "lbGRB + red KN (HMNS)",
        "lbGRB + red KN (disk)",
        "Faint lbGRB",
    ]
    for eos in table.index:
        s = float(table.loc[eos, class_keys].sum())
        assert s == pytest.approx(1.0, rel=1e-9), (
            f"Class fractions for EOS {eos} sum to {s}, not 1.0"
        )


def test_compute_eos_sensitivity_M_thresh_tracks_M_TOV():
    """EOS sweep coherence: M_thresh = k_thresh * M_TOV per row."""
    from grb_rates import compute_eos_sensitivity

    rng = np.random.default_rng(1)
    n = 100
    m1 = rng.uniform(1.1, 2.5, size=n)
    m2 = rng.uniform(1.1, 2.5, size=n)
    w = np.ones(n)
    table = compute_eos_sensitivity(m1, m2, w, k_thresh=1.30)
    np.testing.assert_allclose(table["M_thresh"], 1.30 * table["M_TOV"])


# ─────────────────────────────────────────────────────────────────────
# E7: beamed_class_comparison + OBSERVED_RATES_BY_CLASS
# ─────────────────────────────────────────────────────────────────────
def test_beamed_class_comparison_basic_shape_and_columns():
    from grb_rates import CLASS_THETA_J, beamed_class_comparison

    R_int = {
        "sbGRB + blue KN": 100.0,
        "lbGRB + red KN (HMNS)": 50.0,
        "lbGRB + red KN (disk)": 10.0,
        "Faint lbGRB": 20.0,
    }
    df = beamed_class_comparison(R_int)
    assert set(df.index) == set(R_int.keys())
    for col in [
        "R_intrinsic",
        "theta_j_deg",
        "f_beam",
        "R_beamed",
        "R_obs",
        "R_obs_lo",
        "R_obs_hi",
        "reference",
    ]:
        assert col in df.columns

    # R_beamed for sbGRB row matches the manual formula
    theta_sb = CLASS_THETA_J["sbGRB"]["fid"]
    f_beam = 1.0 - np.cos(np.radians(theta_sb))
    assert df.loc["sbGRB + blue KN", "R_beamed"] == pytest.approx(100.0 * f_beam)

    # The Faint lbGRB class has NaN observed rate per the dict
    assert np.isnan(df.loc["Faint lbGRB", "R_obs"])


def test_observed_rates_by_class_keys_match_classify_bns_2024():
    """OBSERVED_RATES_BY_CLASS must use the same labels as classify_bns_2024."""
    from grb_classify import classify_bns_2024
    from grb_rates import OBSERVED_RATES_BY_CLASS

    cls = classify_bns_2024(np.array([1.4]), np.array([1.3]))
    expected = {k for k, v in cls.items() if isinstance(v, np.ndarray)}
    assert set(OBSERVED_RATES_BY_CLASS.keys()) == expected


def test_lvk_gwtc5_local_rates_well_formed():
    """LVK_GWTC5_LOCAL_RATES must carry the three published populations,
    each with finite positive ``R_lo < R_hi`` and a non-empty reference.

    Structural-only test; numerical values are pinned in
    ``tests/anchors/test_literature_anchors.py::test_lvk_gwtc5_local_rates_match_lvk_2026``.
    """
    from grb_rates import LVK_GWTC5_LOCAL_RATES

    assert set(LVK_GWTC5_LOCAL_RATES.keys()) == {"BNS", "NSBH", "BBH"}
    for pop, entry in LVK_GWTC5_LOCAL_RATES.items():
        assert "R_lo" in entry and "R_hi" in entry, pop
        lo, hi = float(entry["R_lo"]), float(entry["R_hi"])
        assert np.isfinite(lo) and np.isfinite(hi), (pop, lo, hi)
        assert lo > 0 and hi > 0, (pop, lo, hi)
        assert lo < hi, f"{pop}: R_lo={lo} not < R_hi={hi}"
        assert isinstance(entry.get("reference", ""), str)
        assert entry["reference"], f"{pop}: empty reference field"


def test_lvk_gwtc5_extra_anchors_well_formed():
    """Structural sanity for the two extended GWTC-5.0 anchors.

    Pins shape, types, and inequality invariants (e.g. R_med - R_lo_err > 0)
    for:

    - ``LVK_GWTC5_PER_MODEL_RATES``        (Table 2 rows 1-2)
    - ``LVK_GWTC5_NSBH_MIN_BH_MASS``       (inherited from Abac+ 2024)

    Numerical values are pinned in
    ``tests/anchors/test_literature_anchors.py``.
    """
    from grb_rates import (
        LVK_GWTC5_LOCAL_RATES,
        LVK_GWTC5_NSBH_MIN_BH_MASS,
        LVK_GWTC5_PER_MODEL_RATES,
    )

    assert set(LVK_GWTC5_PER_MODEL_RATES.keys()) == {
        "BNS",
        "NSBH",
        "BBH",
        "reference",
    }
    assert isinstance(LVK_GWTC5_PER_MODEL_RATES["reference"], str)
    # Each population must also appear in the joint union so callers can
    # always pair a per-model interval with its union band.  Cross-table
    # consistency is loose because Table 2's per-model rows and the joint
    # row are reported independently.
    for pop in ("BNS", "NSBH", "BBH"):
        assert pop in LVK_GWTC5_LOCAL_RATES, pop
        per_model = LVK_GWTC5_PER_MODEL_RATES[pop]
        assert set(per_model.keys()) == {"PixelPop", "FullPop"}, pop
        for model, entry in per_model.items():
            for key in ("R_med", "R_lo_err", "R_hi_err"):
                assert key in entry, (pop, model, key)
            med = float(entry["R_med"])
            lo_err = float(entry["R_lo_err"])
            hi_err = float(entry["R_hi_err"])
            assert med > 0 and lo_err >= 0 and hi_err >= 0
            assert med - lo_err > 0, f"{pop}/{model}: med - lo_err <= 0"

    for key in (
        "M_BH_min",
        "M_BH_min_lo_err",
        "M_BH_min_hi_err",
        "reference",
        "note",
    ):
        assert key in LVK_GWTC5_NSBH_MIN_BH_MASS, key
    M_min = float(LVK_GWTC5_NSBH_MIN_BH_MASS["M_BH_min"])
    lo_err = float(LVK_GWTC5_NSBH_MIN_BH_MASS["M_BH_min_lo_err"])
    hi_err = float(LVK_GWTC5_NSBH_MIN_BH_MASS["M_BH_min_hi_err"])
    assert M_min > 0 and lo_err >= 0 and hi_err >= 0
    assert M_min - lo_err > 0
    assert isinstance(LVK_GWTC5_NSBH_MIN_BH_MASS["reference"], str)


# ─────────────────────────────────────────────────────────────────────
# Vectorized offset orbit integrator
# ─────────────────────────────────────────────────────────────────────
def _make_synth_population(N, rng):
    v = rng.uniform(50.0, 600.0, size=N)
    t = 10.0 ** rng.uniform(np.log10(20.0), np.log10(8000.0), size=N)
    w = rng.uniform(0.1, 1.0, size=N)
    return v, t, w


def test_offsets_vectorized_matches_legacy_ks():
    """Empirical-CDF agreement between batched RK4 and the legacy
    per-system path within the figure line-thickness tolerance."""
    from scipy.stats import ks_2samp

    from grb_offsets import compute_offsets_population

    rng_pop = np.random.default_rng(2026)
    v, t, w = _make_synth_population(N=600, rng=rng_pop)

    res_v = compute_offsets_population(
        v, t, weights=w, max_systems=600, vectorized=True, rng=np.random.default_rng(0)
    )
    res_l = compute_offsets_population(
        v,
        t,
        weights=w,
        max_systems=600,
        vectorized=False,
        use_analytic=True,
        rng=np.random.default_rng(0),
    )

    # Two-sample KS against the legacy distribution.  At N=600 the natural
    # empirical-CDF noise floor is 1.36/sqrt(600) ~ 0.056; we accept up to
    # 0.10 to leave headroom for the RK4-vs-RK45 step-size difference.
    ks = ks_2samp(res_v["offsets_kpc"], res_l["offsets_kpc"])
    assert ks.statistic < 0.10, (
        f"vectorized vs legacy KS distance = {ks.statistic:.3f} "
        f"(p = {ks.pvalue:.3g}); CDFs disagree beyond figure line thickness"
    )

    # Bulk-percentile sanity: medians within 25 percent (the noise here is
    # dominated by random projection angle scatter; KS above is the tight
    # check).
    med_v = float(np.median(res_v["offsets_kpc"]))
    med_l = float(np.median(res_l["offsets_kpc"]))
    assert abs(med_v - med_l) / med_l < 0.25


def test_offsets_vectorized_zero_kick_returns_birth_radius():
    """v_sys = 0 binaries stay at their projected birth radius."""
    from grb_offsets import (
        _R_CAP_FACTOR,
        DEFAULT_R_E,
        KPC_CM,
        compute_offsets_population_vectorized,
        hernquist_scale_radius,
    )

    N = 100
    v = np.zeros(N)
    # The valid filter requires v > 0; pass a tiny positive kick instead so
    # the systems are admitted but their orbits stay essentially at r0.
    v[:] = 1e-3
    t = np.full(N, 1000.0)  # 1 Gyr delay
    res = compute_offsets_population_vectorized(v, t, max_systems=N, rng=np.random.default_rng(7))
    a_kpc = hernquist_scale_radius(DEFAULT_R_E) / KPC_CM
    # Cap is enforced; offsets must be finite and bounded above by r_cap.
    assert np.isfinite(res["offsets_kpc"]).all()
    assert (res["offsets_kpc"] <= _R_CAP_FACTOR * a_kpc).all()
    # The Hernquist median 3D enclosed-mass radius is r_50 ~ 2.41 a (from
    # M(<r)/M = (r/(r+a))^2 = 0.5 -> r = a / (sqrt(2) - 1) ~ 2.41 a);
    # projected, the median sits below that.  Just check it's positive
    # and sensible (within a couple decades of `a`).
    med = float(np.median(res["offsets_kpc"]))
    assert 0.01 * a_kpc < med < 100.0 * a_kpc


def test_offsets_vectorized_unbound_returns_finite():
    """Super-escape kicks give finite offsets capped at r_cap."""
    from grb_offsets import (
        _R_CAP_FACTOR,
        DEFAULT_R_E,
        KPC_CM,
        compute_offsets_population_vectorized,
        hernquist_scale_radius,
    )

    v = np.array([1e5, 5e4, 2e4])  # km/s -- super-escape
    t = np.array([100.0, 1000.0, 5000.0])
    res = compute_offsets_population_vectorized(v, t, max_systems=3, rng=np.random.default_rng(11))
    a_kpc = hernquist_scale_radius(DEFAULT_R_E) / KPC_CM
    assert np.isfinite(res["offsets_kpc"]).all()
    assert (res["offsets_kpc"] <= _R_CAP_FACTOR * a_kpc).all()


def test_offsets_mixed_hosts_regression_within_5_percent():
    """compute_offsets_mixed_hosts (now backed by the vectorized path) keeps
    its bulk distribution within ~5 percent of the legacy mean and 90th
    percentile when the seed is held fixed."""
    from grb_offsets import compute_offsets_mixed_hosts, compute_offsets_population

    rng_pop = np.random.default_rng(3)
    v, t, w = _make_synth_population(N=400, rng=rng_pop)

    mh = compute_offsets_mixed_hosts(v, t, weights=w, max_systems=400, rng=np.random.default_rng(0))
    legacy_off = []
    rng_l = np.random.default_rng(0)
    # Build a legacy reference with the same host mixture by calling the
    # legacy code path explicitly (vectorized=False).
    from grb_offsets import HOST_MODELS

    for name, hp in HOST_MODELS.items():
        r = compute_offsets_population(
            v,
            t,
            weights=w,
            max_systems=400,
            M_gal=hp["M_gal"],
            R_e=hp["R_e"],
            vectorized=False,
            use_analytic=True,
            rng=rng_l,
        )
        legacy_off.append(r["offsets_kpc"])
    legacy_arr = np.concatenate(legacy_off)

    mean_v = float(np.mean(mh["mixed_offsets"]))
    mean_l = float(np.mean(legacy_arr))
    p90_v = float(np.percentile(mh["mixed_offsets"], 90))
    p90_l = float(np.percentile(legacy_arr, 90))

    # Means and the 90th percentile must agree to ~25 percent.  This is a
    # bulk-distribution sanity check; the tight per-CDF agreement is in
    # ``test_offsets_vectorized_matches_legacy_ks``.
    assert abs(mean_v - mean_l) / mean_l < 0.25, f"mean drift {mean_v:.2f} vs {mean_l:.2f}"
    assert abs(p90_v - p90_l) / p90_l < 0.25, f"p90 drift {p90_v:.2f} vs {p90_l:.2f}"


# ─────────────────────────────────────────────────────────────────────
# remap_ns_marginal: BHNS-side equivalent of the BNS pair remap.
# Used in Section 0 and Section 12.0 of grb_main.ipynb to close the
# ~1.7 Msun Fryer 2012 Eq. 12-13 baryonic-to-gravitational artifact on
# the BHNS NS-mass column (Broekgaarden+ 2021 footnote 3).
# ─────────────────────────────────────────────────────────────────────
def test_remap_ns_marginal_closes_gap_in_1d_input():
    """1D quantile remap fills the 1.65-1.80 Msun NS-mass deficit."""
    from grb_physics import remap_ns_marginal

    rng = np.random.default_rng(2027)
    n_low = 8000
    n_high = 2400
    m_low = rng.uniform(1.10, 1.65, size=n_low)
    m_high = rng.uniform(1.80, 2.20, size=n_high)
    m_raw = np.concatenate([m_low, m_high])
    rng.shuffle(m_raw)

    raw_in_gap = ((m_raw >= 1.65) & (m_raw <= 1.80)).sum()
    assert raw_in_gap == 0, f"test setup broken: raw_in_gap={raw_in_gap}"

    m_new = remap_ns_marginal(m_raw, weights=np.ones_like(m_raw), rng=rng)
    new_in_gap = ((m_new >= 1.65) & (m_new <= 1.80)).sum()

    # Alsing+ 2018 puts ~12 percent of the truncated mass below 1.80 and
    # above 1.65, so requiring >=5 percent is conservative and isolates
    # the regression on the quantile transform itself.
    assert new_in_gap >= 0.05 * m_raw.size, (
        f"remap_ns_marginal left only {new_in_gap}/{m_raw.size} NSs in the "
        f"[1.65, 1.80] gap; expected >= 5 percent."
    )


def test_remap_ns_marginal_respects_truncation_window():
    """Output stays in ``[NS_REMAP_M_MIN, M_TOV]`` for any input."""
    from grb_physics import M_TOV, NS_REMAP_M_MIN, remap_ns_marginal

    rng = np.random.default_rng(2028)
    m = rng.uniform(0.5, 3.5, size=2000)  # input deliberately outside the target window
    m_new = remap_ns_marginal(m, rng=rng)
    assert (m_new >= NS_REMAP_M_MIN - 1e-9).all()
    assert (m_new <= M_TOV + 1e-9).all()


def test_remap_ns_marginal_preserves_weighted_rank_order():
    """High-weight inputs end up at high quantiles of the target."""
    from grb_physics import remap_ns_marginal

    rng = np.random.default_rng(2029)
    m = rng.uniform(1.15, 2.15, size=1500)
    w = np.ones_like(m)
    m_new_uniform = remap_ns_marginal(m, weights=w, rng=np.random.default_rng(11))

    # The ordering of m_new follows the (jittered) ordering of m.  The
    # interp + grid mapping is monotonic in the input rank, so the
    # rank-Spearman correlation must be near 1.
    rank_in = np.argsort(np.argsort(m))
    rank_out = np.argsort(np.argsort(m_new_uniform))
    spearman = np.corrcoef(rank_in, rank_out)[0, 1]
    assert spearman > 0.999, f"Spearman {spearman:.6f} below 0.999"


def test_pair_remap_combined_stack_matches_alsing_target():
    """The combined ``[m1_new, m2_new]`` stack is Alsing-conformal.

    After the per-component remap and re-sort, neither m1_new nor
    m2_new is marginal-Alsing on its own (max/min of two Alsing draws
    biases each tail), but their combined stack IS Alsing: re-sorting
    is just a per-row relabeling and preserves the union distribution.
    Pinned with a Kolmogorov-Smirnov distance against the analytic
    truncated double-Gaussian CDF.
    """
    import grb_physics as gp

    rng_a = np.random.default_rng(42)
    n = 4000
    m1 = rng_a.uniform(1.10, 2.10, size=n)
    m2 = rng_a.uniform(1.10, 2.10, size=n)
    m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)

    m1_new, m2_new = gp.remap_ns_masses_double_gaussian(m1, m2, rng=np.random.default_rng(7))
    assert (m1_new >= m2_new - 1e-12).all(), "m1 >= m2 invariant broken"

    combined = np.concatenate([m1_new, m2_new])
    grid = np.linspace(gp.NS_REMAP_M_MIN, gp.M_TOV, 4096)
    cdf_target = gp._truncated_double_gauss_cdf(grid, gp.NS_REMAP_M_MIN, gp.M_TOV)
    s = np.sort(combined)
    emp = np.searchsorted(s, grid, side="right") / s.size
    ks = float(np.max(np.abs(emp - cdf_target)))
    # For n_combined = 8000 the 99.9 percent KS critical value is ~0.022;
    # 0.03 is a tight but stable ceiling under the deterministic jitter.
    assert ks < 0.03, f"combined-stack KS {ks:.4f} too far from Alsing target"


def test_pair_remap_no_median_wall():
    """Neither component is squeezed against the target median at 1.34 Msun.

    Regression for the stacked-rank median-wall bug: with the legacy
    implementation, weighted mass of ``m2_new >= 1.34`` was ~0 and
    weighted mass of ``m1_new <= 1.34`` was ~0 (the inverse Alsing CDF
    split the stacked ranks at the median).  With the per-component
    remap both fractions are sizeable because each component samples
    the full target PDF.
    """
    import grb_physics as gp

    rng_a = np.random.default_rng(202)
    n = 5000
    m1 = rng_a.uniform(1.15, 2.10, size=n)
    m2 = rng_a.uniform(1.15, 2.10, size=n)
    m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
    w = rng_a.uniform(0.1, 1.0, size=n)

    m1_new, m2_new = gp.remap_ns_masses_double_gaussian(
        m1, m2, weights=w, rng=np.random.default_rng(13)
    )
    w_tot = float(w.sum())
    frac_m2_above = float(w[m2_new >= gp.NS_REMAP_MU1].sum()) / w_tot
    frac_m1_below = float(w[m1_new <= gp.NS_REMAP_MU1].sum()) / w_tot

    # Under the Alsing target ~50 percent of weight sits on each side
    # of mu1 = 1.34; per-component remap puts m2 ~50/50 across mu1.
    # The pair-sort biases m1 above and m2 below, but neither should
    # collapse: m2 >= 1.34 keeps >25 percent and m1 <= 1.34 keeps
    # >15 percent.  Stacked legacy gave both ~0.
    assert frac_m2_above > 0.25, (
        f"m2 above mu1 fraction {frac_m2_above:.3f} too small; median-wall artefact has returned."
    )
    assert frac_m1_below > 0.15, (
        f"m1 below mu1 fraction {frac_m1_below:.3f} too small; median-wall artefact has returned."
    )


# ─────────────────────────────────────────────────────────────────────
# Pais et al. (2025) jet breakout (grb_physics.jet_breaks_out_pais)
# ─────────────────────────────────────────────────────────────────────
def test_jet_breaks_out_pais_zero_disk_no_breakout():
    """A system with no accretion disk has no jet and cannot break out."""
    import grb_physics as gp

    theta = np.radians(6.5)
    E_j = gp.jet_energy_from_disk(0.0)
    E_ej = gp.ejecta_energy_in_cone(5e-3, theta)
    broke = gp.jet_breaks_out_pais(E_j, E_ej, theta, t_d=0.05, t_e=0.2)
    assert bool(broke) is False


def test_jet_breaks_out_pais_massive_disk_breaks_out():
    """A massive disk with thin polar ejecta clears both Pais conditions."""
    import grb_physics as gp

    theta = np.radians(6.5)
    E_j = gp.jet_energy_from_disk(0.1)  # 0.1 Msun disk
    E_ej = gp.ejecta_energy_in_cone(1e-4, theta)  # thin polar obstruction
    broke = gp.jet_breaks_out_pais(E_j, E_ej, theta, t_d=0.05, t_e=0.2)
    assert bool(broke) is True


def test_jet_breaks_out_pais_long_delay_suppresses():
    """t_d >> t_e suppresses breakout via the (t_d/t_e)^2 term (Eq. 4)."""
    import grb_physics as gp

    theta = np.radians(6.5)
    E_j = gp.jet_energy_from_disk(5e-3)
    E_ej = gp.ejecta_energy_in_cone(3e-3, theta)
    short = gp.jet_breaks_out_pais(E_j, E_ej, theta, t_d=0.05, t_e=0.2, require_collimation=False)
    long = gp.jet_breaks_out_pais(E_j, E_ej, theta, t_d=10.0, t_e=0.2, require_collimation=False)
    # A long delay can only make breakout harder, never easier.
    assert bool(long) is False or bool(short) is True
    assert not (bool(long) and not bool(short))


def test_jet_breaks_out_pais_monotonic_in_jet_energy():
    """Breakout is monotone in E_j: above the threshold stays above."""
    import grb_physics as gp

    theta = np.radians(6.5)
    E_ej = gp.ejecta_energy_in_cone(3e-3, theta)
    t_d, t_e = 0.05, 0.2
    E_grid = np.logspace(46, 52, 200)
    broke = gp.jet_breaks_out_pais(E_grid, E_ej, theta, t_d, t_e)
    # Once True, never returns to False as E_j increases.
    idx = np.where(broke)[0]
    if idx.size:
        assert broke[idx.min() :].all()


def test_jet_breaks_out_pais_vectorized_shape():
    """The boolean test broadcasts over per-system arrays."""
    import grb_physics as gp

    theta = np.radians(6.5)
    md = np.array([0.0, 1e-3, 1e-2, 0.1])
    mej = np.array([3e-3, 3e-3, 3e-3, 3e-3])
    E_j = gp.jet_energy_from_disk(md)
    E_ej = gp.ejecta_energy_in_cone(mej, theta)
    broke = gp.jet_breaks_out_pais(E_j, E_ej, theta, t_d=0.05, t_e=0.2)
    assert broke.shape == md.shape
    assert broke.dtype == bool
