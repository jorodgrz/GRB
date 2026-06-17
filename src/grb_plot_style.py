"""Canonical project palette and ApJ rcParams.

Fixed colour palette and a canonical
``matplotlib.rcParams`` block for every paper-bound figure.  This
module is the single source of truth: the palette dictionary below
is the project's locked palette, and ``apply_apj_rcparams()``
registers the rcParams the figure-generation code uses across
notebooks.

Tests in ``tests/unit/test_palette_and_rcparams.py`` lock the hex codes
and the rcParams settings against drift.

References
----------
Crameri, Shephard and Heron (2020), Nat. Commun. 11, 5444.
ApJ figure guidelines (single column 3.5 in, double column 7.0 in).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

C_SB_BLUE: str = "#06B6D4"
"""sbGRB plus blue KN (long-lived HMNS engine)."""

C_LB_HMNS: str = "#DC2626"
"""lbGRB plus red KN (short-lived HMNS engine)."""

C_LB_DISK: str = "#DC2626"
"""lbGRB plus red KN (disk; prompt collapse, massive disk)."""

C_FAINT: str = "#F59E0B"
"""Faint lbGRB (prompt collapse, small disk)."""

C_NO_GRB: str = "#334155"
"""No GRB / background."""

C_MODEL_A: str = "#1D4ED8"
"""Model A reference (blue, solid)."""

C_MODEL_F: str = "#243c6e"
"""Model F (alpha_CE=0.5; cividis 0.15, dashed)."""

C_MODEL_G: str = "#7d7c78"
"""Model G (alpha_CE=2.0; cividis 0.50, dash-dot)."""

C_MODEL_J: str = "#d6c35d"
"""Model J (M_NS,max=2.0; cividis 0.85, dotted)."""

C_MODEL_K: str = "#DC2626"
"""Model K reference (red, dash-dot-dot-dot)."""

C_WP15: str = "#6366F1"
"""Wanderman and Piran 2015 R(z) overlay (indigo, dashed)."""

C_OBSERVED: str = "#6B21A8"
"""Observed sample overlay (purple)."""


CLASS_PALETTE: dict[str, str] = {
    "sbGRB + blue KN": C_SB_BLUE,
    "lbGRB + red KN (HMNS)": C_LB_HMNS,
    "lbGRB + red KN (disk)": C_LB_DISK,
    "Faint lbGRB": C_FAINT,
    "No GRB": C_NO_GRB,
}

MODEL_PALETTE: dict[str, str] = {
    "A": C_MODEL_A,
    "F": C_MODEL_F,
    "G": C_MODEL_G,
    "J": C_MODEL_J,
    "K": C_MODEL_K,
}

MODEL_LINESTYLES: dict[str, object] = {
    "A": "-",
    "F": "--",
    "G": "-.",
    "J": ":",
    "K": (0, (3, 1, 1, 1)),
}


# Broekgaarden+ 2021 formation channels I-V (Sec. 4, Fig. 9).  Fixed once and
# reused wherever a figure splits a population by channel, so a reader tracks
# a channel across panels by colour.  Channels III and V are empty in the BNS
# and BHNS samples but are kept here so the mapping is complete and stable.
C_CHANNEL_I: str = "#3B82F6"
"""Channel I: classic isolated binary (stable MT then CE)."""

C_CHANNEL_II: str = "#10B981"
"""Channel II: stable mass transfer only (no CE)."""

C_CHANNEL_III: str = "#F59E0B"
"""Channel III: single-core CE."""

C_CHANNEL_IV: str = "#EF4444"
"""Channel IV: double-core CE."""

C_CHANNEL_V: str = "#8B5CF6"
"""Channel V: other / unclassified."""

CHANNEL_PALETTE: dict[str, str] = {
    "I": C_CHANNEL_I,
    "II": C_CHANNEL_II,
    "III": C_CHANNEL_III,
    "IV": C_CHANNEL_IV,
    "V": C_CHANNEL_V,
}


# BH-engine vs HMNS-engine observational comparison (comparison.ipynb).  This
# is a different observable axis from the four GRB classes, so it carries its
# own palette; the lbGRB / sbGRB marker reds and blues are chosen for
# protanopia separation rather than reused from CLASS_PALETTE.
ENGINE_PALETTE: dict[str, str] = {
    "lb": "#c0392b",  # lbGRB markers (red)
    "sb": "#1565c0",  # sbGRB markers (blue)
    "gw": "#555555",  # GW170817 marker (grey)
}

# Per-event ejecta composition (Rastinejad+ 2024 Table 3 decomposition).
COMPONENT_PALETTE: dict[str, str] = {
    "R": "#c0392b",  # lanthanide-rich (disk wind, MHD)
    "P": "#7d3c98",  # intermediate (disk wind, thermal)
    "B": "#1f78b4",  # lanthanide-poor (dynamical, not disk wind)
}


# Physics-knob families of the Broekgaarden+ 2021 grid, in the order the
# 20-model figures group the variations.  Mirrors the ``family`` field of
# ``grb_io.BROEKGAARDEN21_MODELS``; kept here so the plotting layer does not
# import the loader module.
MODEL_GRID_FAMILY_ORDER: list[str] = [
    "fiducial",
    "alpha_ce",
    "mt_efficiency",
    "case_bb",
    "ce_survival",
    "sn_engine",
    "ns_mass_cap",
    "pisn",
    "sn_kick",
    "bh_kick",
    "wr_winds",
]


def model_grid_colors(
    models: dict[str, dict] | None = None,
    cmap_name: str = "cividis",
) -> dict[str, tuple[float, float, float, float]]:
    """Assign one colour per model in the 20-variation Broekgaarden+ 2021 grid.

    The variations are ordered family-major (``MODEL_GRID_FAMILY_ORDER``,
    then registry order within a family) and sampled at evenly spaced points
    of a single perceptually uniform, colourblind-safe colormap (cividis by
    default; Crameri et al. 2020).  Family-major ordering puts the members of
    one physics knob in an adjacent hue band, so a reader tracks a family
    across the figure by colour while every model still gets a distinct
    shade.  Returns ``{suffix: rgba}`` for every model in ``models``.

    Parameters
    ----------
    models : dict, optional
        Registry mapping ``suffix -> {"family": ...}``.  Defaults to
        ``grb_io.BROEKGAARDEN21_MODELS`` (imported lazily to avoid a
        plot-to-loader import cycle).
    cmap_name : str
        Any perceptually uniform colormap; cividis and viridis are the
        project defaults.
    """
    import matplotlib

    if models is None:
        from grb_io import BROEKGAARDEN21_MODELS  # lazy: no import cycle

        models = BROEKGAARDEN21_MODELS

    fam_rank = {fam: i for i, fam in enumerate(MODEL_GRID_FAMILY_ORDER)}
    ordered = sorted(
        models,
        key=lambda s: (fam_rank.get(models[s]["family"], len(fam_rank)), list(models).index(s)),
    )
    cmap = matplotlib.colormaps[cmap_name]
    n = len(ordered)
    positions = np.linspace(0.05, 0.95, n) if n > 1 else np.array([0.5])
    return {suffix: tuple(cmap(pos)) for suffix, pos in zip(ordered, positions, strict=True)}


APJ_RCPARAMS: dict[str, object] = {
    "font.size": 8,
    "mathtext.fontset": "cm",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "savefig.bbox": "tight",
}


def apply_apj_rcparams(extra: dict[str, object] | None = None) -> None:
    """Register the canonical ApJ rcParams on the active matplotlib runtime.

    Parameters
    ----------
    extra : dict, optional
        Additional rcParams overrides applied on top of ``APJ_RCPARAMS``.
    """
    import matplotlib as mpl

    mpl.rcParams.update(APJ_RCPARAMS)
    if extra:
        mpl.rcParams.update(extra)


def weighted_hist_pdf(
    data: ArrayLike,
    mask: ArrayLike,
    w: ArrayLike,
    bins: ArrayLike,
    w_total: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Abundance-weighted PDF of ``data[mask]`` against ``bins``.

    Returns ``(bin_centers, density)`` with

        density = sum(w_in_bin) / (w_total * binw).

    Across mutually exclusive masks for the same population the per-class
    curve areas sum to the in-range weight fraction (= 1 when ``bins``
    covers the full support), so each curve's area equals that class's
    population fraction. Data outside ``[bins[0], bins[-1]]`` is silently
    dropped from the numerator but ``w_total`` is unchanged: the caller
    is responsible for choosing bins that cover the full support of every
    class plotted with the same ``w_total``.

    Returns ``None`` when the mask is empty so the caller can skip
    plotting without an empty-array branch.
    """
    data_arr = np.asarray(data)
    mask_arr = np.asarray(mask, dtype=bool)
    w_arr = np.asarray(w)
    bins_arr = np.asarray(bins)

    if not mask_arr.any():
        return None

    sub_d = data_arr[mask_arr]
    sub_w = w_arr[mask_arr]
    mu, _ = np.histogram(sub_d, bins=bins_arr, weights=sub_w)
    binw = np.diff(bins_arr)
    centers = 0.5 * (bins_arr[:-1] + bins_arr[1:])
    return centers, mu / (w_total * binw)
