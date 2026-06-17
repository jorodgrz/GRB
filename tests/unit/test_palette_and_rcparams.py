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
