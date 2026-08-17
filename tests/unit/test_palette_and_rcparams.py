"""Lock the project palette and ApJ rcParams against drift.

"Plotting Standards" mandates a fixed hex
palette and a canonical ``matplotlib.rcParams`` block.  This file
asserts both against ``grb_plot_style`` so silent edits to the
single source of truth fail loudly.
"""

from __future__ import annotations

import pytest

from grb_plot_style import (
    APJ_RCPARAMS,
    C_FAINT,
    C_LB_DISK,
    C_LB_HMNS,
    C_MODEL_A,
    C_MODEL_F,
    C_MODEL_G,
    C_MODEL_J,
    C_MODEL_K,
    C_NO_GRB,
    C_OBSERVED,
    C_SB_BLUE,
    C_WP15,
    CHANNEL_PALETTE,
    CLASS_LABELS,
    CLASS_LABELS_SHORT,
    CLASS_PALETTE,
    COMPONENT_PALETTE,
    ENGINE_PALETTE,
    MODEL_PALETTE,
    apply_apj_rcparams,
)

REFERENCE_PALETTE = {
    "C_SB_BLUE": "#06B6D4",
    "C_LB_HMNS": "#DC2626",
    "C_LB_DISK": "#DC2626",
    "C_FAINT": "#F59E0B",
    "C_NO_GRB": "#334155",
    "C_MODEL_A": "#1D4ED8",
    "C_MODEL_F": "#243c6e",
    "C_MODEL_G": "#7d7c78",
    "C_MODEL_J": "#d6c35d",
    "C_MODEL_K": "#DC2626",
    "C_WP15": "#6366F1",
    "C_OBSERVED": "#6B21A8",
}

MODULE_CONSTANTS = {
    "C_SB_BLUE": C_SB_BLUE,
    "C_LB_HMNS": C_LB_HMNS,
    "C_LB_DISK": C_LB_DISK,
    "C_FAINT": C_FAINT,
    "C_NO_GRB": C_NO_GRB,
    "C_MODEL_A": C_MODEL_A,
    "C_MODEL_F": C_MODEL_F,
    "C_MODEL_G": C_MODEL_G,
    "C_MODEL_J": C_MODEL_J,
    "C_MODEL_K": C_MODEL_K,
    "C_WP15": C_WP15,
    "C_OBSERVED": C_OBSERVED,
}


@pytest.mark.parametrize("name", sorted(REFERENCE_PALETTE))
def test_palette_hex_matches_reference(name):
    assert MODULE_CONSTANTS[name].lower() == REFERENCE_PALETTE[name].lower(), (
        f"{name} drift: module={MODULE_CONSTANTS[name]} reference={REFERENCE_PALETTE[name]}"
    )


def test_class_palette_keys_match_classify_bns_2024():
    """The four Gottlieb (2024) class labels are present in the palette
    plus a 'No GRB' fallback for grid background."""
    expected = {
        "sbGRB + blue KN",
        "lbGRB + red KN (HMNS)",
        "lbGRB + red KN (disk)",
        "Faint lbGRB",
        "No GRB",
    }
    assert set(CLASS_PALETTE) == expected


def test_model_palette_keys_match_broekgaarden_grid():
    assert set(MODEL_PALETTE) == {"A", "F", "G", "J", "K"}


def test_channel_palette_covers_broekgaarden_channels_I_to_V():
    """The five Broekgaarden+ 2021 formation channels each carry a hex."""
    assert set(CHANNEL_PALETTE) == {"I", "II", "III", "IV", "V"}
    assert CHANNEL_PALETTE["I"].lower() == "#3b82f6"
    assert CHANNEL_PALETTE["IV"].lower() == "#ef4444"
    distinct = {v.lower() for v in CHANNEL_PALETTE.values()}
    assert len(distinct) == len(CHANNEL_PALETTE), "channel colours are not distinct"


def test_engine_and_component_palettes_locked():
    """comparison.ipynb BH/HMNS engine and ejecta-composition palettes."""
    assert ENGINE_PALETTE["lb"].lower() == "#c0392b"
    assert ENGINE_PALETTE["sb"].lower() == "#1565c0"
    assert ENGINE_PALETTE["gw"].lower() == "#555555"
    assert set(COMPONENT_PALETTE) == {"R", "P", "B"}
    assert COMPONENT_PALETTE["R"].lower() == "#c0392b"


def _classifier_class_keys():
    """The seven class labels the classifiers emit (minus BHNS ``M_disk``)."""
    import numpy as np

    from grb_classify import classify_bhns, classify_bns_2024

    m1 = np.array([1.4, 1.6])
    m2 = np.array([1.3, 1.5])
    bns = set(classify_bns_2024(m1, m2))
    bhns = set(classify_bhns(np.array([8.0, 6.0]), np.array([1.4, 1.4])))
    bhns.discard("M_disk")
    return bns | bhns


def test_class_label_maps_share_the_canonical_keys():
    """Both label maps are keyed by exactly the classifier output keys."""
    keys = _classifier_class_keys()
    assert set(CLASS_LABELS) == keys
    assert set(CLASS_LABELS_SHORT) == keys


def test_class_labels_match_grid_class_labels():
    """The display values reproduce the GRID_CLASS_LABELS strings (plus the
    background 'No GRB'), so every figure names the classes identically."""
    from grb_classify import GRID_CLASS_LABELS

    assert set(CLASS_LABELS.values()) == set(GRID_CLASS_LABELS.values()) | {"No GRB"}


def test_short_labels_collapse_to_full_labels():
    """Stripping the newline from a short label recovers a string whose tokens
    all appear in the matching full label, so the two maps cannot drift apart."""
    for key, full in CLASS_LABELS.items():
        short = CLASS_LABELS_SHORT[key]
        for token in short.replace("\n", " ").split():
            assert token in full, (key, token, full)


def test_apj_rcparams_match_reference():
    assert APJ_RCPARAMS["font.size"] == 8
    assert APJ_RCPARAMS["mathtext.fontset"] == "cm"
    assert APJ_RCPARAMS["xtick.direction"] == "in"
    assert APJ_RCPARAMS["ytick.direction"] == "in"
    assert APJ_RCPARAMS["xtick.minor.visible"] is True
    assert APJ_RCPARAMS["ytick.minor.visible"] is True
    assert APJ_RCPARAMS["savefig.bbox"] == "tight"


def test_apply_apj_rcparams_updates_runtime():
    """``apply_apj_rcparams`` writes every entry from ``APJ_RCPARAMS``
    onto the active ``matplotlib.rcParams``."""
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    saved = {k: mpl.rcParams[k] for k in APJ_RCPARAMS}
    try:
        apply_apj_rcparams()
        for k, v in APJ_RCPARAMS.items():
            assert mpl.rcParams[k] == v, (k, mpl.rcParams[k], v)
    finally:
        mpl.rcParams.update(saved)


def test_model_grid_colors_cover_all_twenty_and_are_distinct():
    """The 20-model colour helper returns one distinct colour per variation."""
    pytest.importorskip("matplotlib")
    from grb_io import BROEKGAARDEN21_MODELS
    from grb_plot_style import model_grid_colors

    colors = model_grid_colors()
    assert set(colors) == set(BROEKGAARDEN21_MODELS), (
        "model_grid_colors does not cover exactly the registry suffixes."
    )
    rounded = {tuple(round(c, 6) for c in rgba) for rgba in colors.values()}
    assert len(rounded) == len(colors), "model_grid_colors produced duplicate colours."


def test_model_grid_family_order_matches_registry_families():
    """Every family used in the registry appears in the plotting family order."""
    from grb_io import BROEKGAARDEN21_MODELS
    from grb_plot_style import MODEL_GRID_FAMILY_ORDER

    reg_families = {m["family"] for m in BROEKGAARDEN21_MODELS.values()}
    assert reg_families <= set(MODEL_GRID_FAMILY_ORDER), (
        f"families {reg_families - set(MODEL_GRID_FAMILY_ORDER)} are in the "
        f"registry but missing from MODEL_GRID_FAMILY_ORDER."
    )


def test_per_dex_helper_applies_the_log10_jacobian_at_geometric_centres():
    """dp/dlog10(t) = t ln10 dp/dt, with t the geometric bin centre."""
    import numpy as np

    from grb_plot_style import weighted_hist_pdf, weighted_hist_pdf_per_dex

    rng = np.random.default_rng(0)
    bins = np.logspace(0.0, 4.0, 21)
    data = 10 ** rng.uniform(0.1, 3.9, 500)
    w = rng.uniform(0.5, 2.0, 500)
    mask = np.ones(500, dtype=bool)

    _, mu = weighted_hist_pdf(data, mask, w, bins, w.sum())
    mu_dex = weighted_hist_pdf_per_dex(data, mask, w, bins, w.sum())

    t_geom = np.sqrt(bins[:-1] * bins[1:])
    populated = mu > 0
    np.testing.assert_allclose(
        mu_dex[populated], (mu * t_geom * np.log(10.0))[populated], rtol=1e-12
    )
    # The geometric-centre Jacobian conserves area only to second order in the
    # bin width: integrating dp/dlog10(t) over dlog10(t) falls short of the
    # integral of dp/dt over dt by sqrt(r) ln(r) / (r - 1) for a grid of ratio
    # r.  Pinned so a future switch to the logarithmic mean is a deliberate act.
    r = bins[1] / bins[0]
    shortfall = np.sqrt(r) * np.log(r) / (r - 1.0)
    np.testing.assert_allclose(
        np.nansum(mu_dex * np.diff(np.log10(bins))),
        shortfall * (mu * np.diff(bins)).sum(),
        rtol=1e-12,
    )


def test_per_dex_helper_gaps_empty_bins_and_skips_empty_masks():
    """Empty bins come back as nan (a gap on log-y), an empty mask as None."""
    import numpy as np

    from grb_plot_style import weighted_hist_pdf_per_dex

    bins = np.array([1.0, 10.0, 100.0, 1000.0])
    data = np.array([2.0, 200.0])
    w = np.array([1.0, 1.0])

    mu_dex = weighted_hist_pdf_per_dex(data, np.ones(2, dtype=bool), w, bins, w.sum())
    assert np.isnan(mu_dex[1]), "unpopulated bin should be a nan gap, not 0"
    assert np.isfinite(mu_dex[[0, 2]]).all()

    assert weighted_hist_pdf_per_dex(data, np.zeros(2, dtype=bool), w, bins, 1.0) is None


def test_per_dex_helper_preserves_within_class_additivity():
    """Section 3b relies on channel curves summing to the per-class total."""
    import numpy as np

    from grb_plot_style import weighted_hist_pdf_per_dex

    rng = np.random.default_rng(7)
    bins = np.logspace(0.0, 3.0, 13)
    data = 10 ** rng.uniform(0.05, 2.95, 300)
    w = rng.uniform(0.5, 2.0, 300)
    part_a = rng.random(300) < 0.4
    part_b = ~part_a

    total = weighted_hist_pdf_per_dex(data, np.ones(300, dtype=bool), w, bins, w.sum())
    parts = [
        np.nan_to_num(weighted_hist_pdf_per_dex(data, p, w, bins, w.sum()))
        for p in (part_a, part_b)
    ]
    np.testing.assert_allclose(np.nan_to_num(total), parts[0] + parts[1], rtol=1e-12)


def test_per_dex_zero_fill_drops_holes_and_closes_the_support():
    """Interior holes and the bins flanking the support drop; nothing further out."""
    import numpy as np

    from grb_plot_style import weighted_hist_pdf_per_dex

    # 9 bins (0..8). Data lands in 2, 4, 6, so the support is 2..6 with holes at
    # 3 and 5, closing bins at 1 and 7, and bins 0 and 8 outside entirely.
    bins = np.logspace(0.0, 9.0, 10)
    data = np.array([1.5e2, 1.5e4, 1.5e6])
    w = np.ones(3)
    mask = np.ones(3, dtype=bool)

    plain = weighted_hist_pdf_per_dex(data, mask, w, bins, w.sum())
    assert np.isnan(plain[[0, 1, 3, 5, 7, 8]]).all(), "default leaves every empty bin a gap"

    filled = weighted_hist_pdf_per_dex(data, mask, w, bins, w.sum(), zero_fill=1e-9)
    assert (filled[[3, 5]] == 1e-9).all(), "interior holes must drop to the sentinel"
    assert (filled[[1, 7]] == 1e-9).all(), "the support must close at both ends"
    assert np.isnan(filled[[0, 8]]).all(), "no curve is laid along the floor past the closure"
    np.testing.assert_allclose(filled[[2, 4, 6]], plain[[2, 4, 6]], rtol=1e-12)


def test_per_dex_zero_fill_is_inert_without_holes_and_on_empty_masks():
    """A gapless curve is untouched, and an empty mask still returns None."""
    import numpy as np

    from grb_plot_style import weighted_hist_pdf_per_dex

    bins = np.logspace(0.0, 3.0, 4)
    data = np.array([5.0, 50.0, 500.0])
    w = np.ones(3)

    a = weighted_hist_pdf_per_dex(data, np.ones(3, dtype=bool), w, bins, w.sum())
    b = weighted_hist_pdf_per_dex(data, np.ones(3, dtype=bool), w, bins, w.sum(), zero_fill=1e-9)
    np.testing.assert_allclose(a, b, rtol=1e-12)

    assert (
        weighted_hist_pdf_per_dex(data, np.zeros(3, dtype=bool), w, bins, 1.0, zero_fill=1e-9)
        is None
    )


def test_per_dex_zero_fill_closes_an_end_only_when_a_bin_is_free_to_close_into():
    """A support reaching the array edge cannot close, so it must stay open."""
    import numpy as np

    from grb_plot_style import weighted_hist_pdf_per_dex

    bins = np.logspace(0.0, 5.0, 6)
    w = np.ones(2)
    mask = np.ones(2, dtype=bool)

    # Support in bins 1..2, so bin 0 and bin 3 flank it and both close.
    inner = weighted_hist_pdf_per_dex(
        np.array([1.5e1, 1.5e2]), mask, w, bins, w.sum(), zero_fill=1e-9
    )
    assert inner[0] == 1e-9 and inner[3] == 1e-9, "a flanked support must close both ends"
    assert np.isnan(inner[4]), "only the bin adjacent to the support closes"

    # Support touching the last bin: there is no bin beyond it, so it stays open.
    edge = weighted_hist_pdf_per_dex(
        np.array([1.5e1, 1.5e4]), mask, w, bins, w.sum(), zero_fill=1e-9
    )
    assert edge[0] == 1e-9, "the low end still closes"
    assert np.isfinite(edge[-1]), "a truncated high end is drawn, not closed away"
