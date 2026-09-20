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

C_PS26: str = "#6366F1"
"""Pracchia and Salafia 2026 R(z) overlay (indigo, dashed).  Reuses the
former WP15 hex slot because the two are the same dashed-empirical role
on the plot; both are never rendered simultaneously."""

C_DU25: str = "#14B8A6"
"""Du et al. 2025 Swift/BAT local-rate marker (teal square).  Chosen to
be perceptually orthogonal to C_SB_BLUE (Model A cyan), C_PS26 (indigo),
and C_OBSERVED (purple)."""

C_OBSERVED: str = "#6B21A8"
"""Observed sample overlay (purple)."""


CLASS_PALETTE: dict[str, str] = {
    "sbGRB + blue KN": C_SB_BLUE,
    "lbGRB + red KN (HMNS)": C_LB_HMNS,
    "lbGRB + red KN (disk)": C_LB_DISK,
    "Faint lbGRB": C_FAINT,
    "No GRB": C_NO_GRB,
}


# Canonical display labels, keyed by the classify_bns_2024 / classify_bhns
# output strings so a figure looks up its legend text by the same key it uses
# to index the mask dict.  Values match grb_classify.GRID_CLASS_LABELS so every
# panel names the seven classes identically; kept as literals here (not imported
# from grb_classify) so the plotting layer stays free of the classifier module,
# following MODEL_GRID_FAMILY_ORDER.  Drift is caught by
# tests/unit/test_palette_and_rcparams.py.
CLASS_LABELS: dict[str, str] = {
    "sbGRB + blue KN": "sbGRB + blue KN (HMNS)",
    "lbGRB + red KN (HMNS)": "lbGRB + red KN (HMNS)",
    "lbGRB + red KN (disk)": "lbGRB + red KN (BNS disk)",
    "Faint lbGRB": "Faint lbGRB (BNS)",
    "lbGRB + red KN (BHNS disk)": "lbGRB + red KN (BHNS disk)",
    "Faint lbGRB (BHNS)": "Faint lbGRB (BHNS)",
    "No GRB": "No GRB",
}

# Two-line forms for the crowded axes (the Section 11.4 engine-comparison
# tick labels, the Section 12 decision-tree boxes) where the full label
# overruns the width budget.
CLASS_LABELS_SHORT: dict[str, str] = {
    "sbGRB + blue KN": "sbGRB\n+ blue KN",
    "lbGRB + red KN (HMNS)": "lbGRB\n(HMNS)",
    "lbGRB + red KN (disk)": "lbGRB\n(BNS disk)",
    "Faint lbGRB": "Faint lbGRB\n(BNS)",
    "lbGRB + red KN (BHNS disk)": "lbGRB\n(BHNS disk)",
    "Faint lbGRB (BHNS)": "Faint lbGRB\n(BHNS)",
    "No GRB": "No GRB",
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
# a channel across panels by colour.  Channel V is empty in every model of the
# grid but is kept here so the mapping is complete and stable.
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

# Twenty separable colours for the 20-model grid, as an alternative to the
# cividis ramp of ``model_grid_colors``.  A sequential colormap sampled at 20
# points is the wrong tool for 20 categories: on the Section 6d / 7c rate
# panels the neighbouring cividis shades are not tellable apart, which is the
# whole point of drawing every model.
#
# Model A keeps ``C_MODEL_A``: it is the fiducial, it is drawn heavier than the
# rest, and a pastel reference would be the least visible line on the panel.
# The other 19 are the solution of a constrained search over the pastel box
# L* in [68, 90], C* in [14, 36] (CIELCh), maximising the minimum pairwise
# CIELab distance for normal vision subject to floors on that distance under
# simulated deuteranopia, protanopia and tritanopia (Vienot, Brettel and
# Mollon 1999 dichromat projection).  Pure CVD optimisation was tried first
# and rejected: it collapses the set onto the blue-yellow axis, the only axis
# a dichromat retains, and the result reads as monotonous to everyone else.
#
# Achieved separations, pinned by tests/unit/test_palette_and_rcparams.py:
# normal 17.1, deuteranopia 6.1, protanopia 5.7, tritanopia 4.3.  Twenty
# categories cannot be made fully CVD-safe (ColorBrewer stops at 12 for
# qualitative sets, 8 of them CVD-safe), so the legend order and the family
# hue bands carry identification where colour alone runs out.
#
# Colours are assigned family-major: the selected set is sorted by hue and
# handed out in ``MODEL_GRID_FAMILY_ORDER``, so one physics knob occupies one
# hue band and members within it run light to dark.
MODEL_GRID_PASTEL: dict[str, str] = {
    "A": C_MODEL_A,
    "alpha0p1": "#EAC7BF",
    "F": "#FFB495",
    "G": "#E8929B",
    "alpha10": "#C69F68",
    "B": "#F9E39E",
    "C": "#C6BF9C",
    "D": "#AEBD7C",
    "E": "#CDE8BC",
    "EH": "#9FC2AE",
    "H": "#6DBE99",
    "I": "#9CEAE0",
    "J": "#A2D5FA",
    "K": "#5BC9DC",
    "L": "#8FA8C4",
    "M": "#DCD3F0",
    "N": "#8AABE6",
    "O": "#E5B1E9",
    "fWR0p1": "#FFC6DD",
    "fWR5": "#C69FAF",
}


def model_grid_colors(
    models: dict[str, dict] | None = None,
    cmap_name: str = "cividis",
    style: str = "cividis",
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
        project defaults.  Ignored when ``style="pastel"``.
    style : {"cividis", "pastel"}
        ``"cividis"`` samples the colormap, which keeps a family band
        readable as a ramp but makes individual models hard to separate.
        ``"pastel"`` returns ``MODEL_GRID_PASTEL``, the separable
        categorical set the Section 6d / 7c rate panels use.
    """
    import matplotlib

    if style == "pastel":
        keys = list(models) if models is not None else list(MODEL_GRID_PASTEL)
        missing = [k for k in keys if k not in MODEL_GRID_PASTEL]
        if missing:
            raise KeyError(
                f"MODEL_GRID_PASTEL has no colour for {missing}; add one there "
                f"rather than falling back to a colormap sample, so the model "
                f"keeps one colour across every figure."
            )
        return {k: matplotlib.colors.to_rgba(MODEL_GRID_PASTEL[k]) for k in keys}
    if style != "cividis":
        raise ValueError(f"style must be 'cividis' or 'pastel', got {style!r}")

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


def weighted_hist_pdf_per_dex(
    data: ArrayLike,
    mask: ArrayLike,
    w: ArrayLike,
    bins: ArrayLike,
    w_total: float,
    zero_fill: float | None = None,
) -> NDArray[np.float64] | None:
    """``weighted_hist_pdf`` converted to a density per decade of ``data``.

    Returns ``dp/dlog10(x)`` on ``bins``, so that visual area under a curve
    drawn on a log-x axis reads as probability. This follows the t_inspiral
    panel of Broekgaarden+ 2021 Fig. 7, which plots R_form(log10 t_inspiral)
    for R_form(x) = dN/(dM_SFR dx), a density per dex of log10 t. Their Fig. 8
    is a different object and not the reference: a detection-weighted
    R_det(t_inspiral) in yr^-1 Gyr^-1 against linear t.

    The conversion evaluates the Jacobian at the geometric bin centre, the bin
    midpoint in the log coordinate being plotted:

        dp/dlog10(x) = x * ln(10) * dp/dx.

    Requires log-spaced ``bins``; on a linear grid the geometric centre is not
    the midpoint of the bin in log space and the conversion is biased.

    Area is conserved only to second order in the bin width: on a grid of ratio
    r the recovered probability is short by a factor sqrt(r) ln(r) / (r - 1),
    which is 0.35 percent on the 30-bin Section 3 grid. The factor depends on r
    alone, so it is a uniform rescaling of every bin and leaves the shape and
    the relative height of any two curves untouched. Use the logarithmic mean
    (x_hi - x_lo) / ln(x_hi / x_lo) in place of the geometric centre if an
    exactly normalised curve is ever needed.

    A power law dp/dx ~ x^alpha is drawn on this ordinate with slope alpha + 1,
    so the canonical delay-time index (alpha = -1 for gravitational-wave
    inspiral of a log-uniform separation distribution) appears flat rather than
    sloping at -1. Quote alpha from the underlying dp/dx if a caption states it.

    Zero-weight bins come back as ``nan`` by default, so a caller drawing on a
    log-y axis gets a gap. Pass ``zero_fill`` (typically an order of magnitude
    below the axis floor) to send an empty bin down instead: the curve drops
    off the bottom of the panel and returns, which reads as "no systems here"
    rather than leaving a disconnected segment floating in mid-panel.

    Filled are the bins interior to the support and the single empty bin
    flanking each end of it, so the curve closes downward where the
    distribution stops, the way a stepped histogram closes to its baseline.
    Bins further out stay ``nan``: the curve is never laid flat along the axis
    floor across a range it does not reach. A log axis cannot render zero,
    which is why the caller supplies a finite sentinel and lets the axis clip
    it, leaving a bare vertical drop.

    Closing an end asserts that the density really is zero just past the last
    populated bin. That is only true when ``bins`` spans the full support of
    ``data[mask]``; if the range clips the distribution the drop is an artifact
    of the binning, so verify coverage before reading an edge as an endpoint.

    Returns ``None`` for an empty mask, matching ``weighted_hist_pdf``.
    """
    h = weighted_hist_pdf(data, mask, w, bins, w_total)
    if h is None:
        return None
    _, mu = h
    bins_arr = np.asarray(bins)
    t_geom = np.sqrt(bins_arr[:-1] * bins_arr[1:])
    out = np.where(mu > 0, mu * t_geom * np.log(10.0), np.nan)

    populated = mu > 0
    if zero_fill is not None and populated.any():
        first = int(np.argmax(populated))
        last = populated.size - 1 - int(np.argmax(populated[::-1]))
        fill = ~populated
        fill[: max(first - 1, 0)] = False
        fill[last + 2 :] = False
        out[fill] = zero_fill
    return out


def plot_gated_curve(
    ax,
    x,
    y,
    ok,
    color,
    ls="-",
    label=None,
    lw=1.1,
    ms=2.5,
    marker="o",
    yerr=None,
    n_raw=None,
    y_ul=None,
    artifact=None,
    faded_alpha=0.35,
    alpha=1.0,
):
    """Draw ``y(x)`` over a quality gate, showing every bin the sample has.

    The gate (``ok``) sets how a bin is drawn, never whether it appears.
    Four states, matching what ``grb_rates.formation_efficiency_binned``
    returns:

    ``ok``
        filled marker joined by a solid line, the well-measured curve.
        ``marker`` sets the shape, which is how two curves sharing a
        colour stay separable once the points are dense enough to hide
        the linestyle.
    ``~ok`` with ``n_raw > 0``
        measured but imprecise: open marker with no connecting line, so it
        reads as a point estimate rather than as part of the curve.
        ``yerr`` is optional and adds a 1-sigma bar to that marker; without
        it the open marker carries the sample-support statement alone.  It
        may be a 1-D symmetric bar or a ``(2, N)`` [lower, upper] pair; on a
        log ordinate the asymmetric form is the honest one for a bin holding
        a few systems, since the symmetric bar reaches below zero.
    ``n_raw == 0``
        the subset produces nothing in this bin, which a log ordinate
        cannot show as a value.  Drawn as a downward-arrow upper limit at
        ``y_ul`` so an exact zero is distinguishable from an unmeasured bin.
    ``artifact``
        bins the caller has flagged as simulation artefact rather than
        physics.  Rendered at ``faded_alpha`` in whichever of the three
        states above applies, keeping the curve's own linestyle, so the
        reader sees the values and their provenance at once without two
        same-coloured classes merging.  The segment joining a flagged point
        to an unflagged neighbour is drawn in the faded style, so a step at
        the flag boundary reads as a step rather than as a gap in the data.
        ``alpha`` scales the whole curve, for a reference envelope that
        should not compete with the classes.

    Passing only the original arguments reproduces the plain gated curve.
    Returns the Line2D list of the main curve, so the caller's legend
    handling is unchanged.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    art = np.zeros_like(ok) if artifact is None else np.asarray(artifact, dtype=bool)
    n = None if n_raw is None else np.asarray(n_raw)

    drawable = np.isfinite(y) & (y > 0)
    solid = ok & drawable & ~art
    faded = ok & drawable & art

    handle = ax.plot(
        x,
        np.where(solid, y, np.nan),
        color=color,
        ls=ls,
        lw=lw,
        marker=marker,
        ms=ms,
        alpha=alpha,
        label=label,
    )
    if faded.any():
        # Keep the caller's linestyle here.  Overriding it with a dash made
        # two curves that share a colour and differ only by linestyle (the
        # two lbGRB + red KN classes) collapse onto each other inside the
        # flagged range, which on the Section 5 panel is a fifth of the BNS
        # weight.  Provenance is carried by the alpha and by the caller's
        # shaded band instead.
        ax.plot(
            x,
            np.where(faded, y, np.nan),
            color=color,
            ls=ls,
            lw=lw,
            marker=marker,
            ms=ms,
            alpha=faded_alpha * alpha,
            zorder=1,
        )

        # The solid and faded runs are two separate series, so the one segment
        # that spans the state change belongs to neither and the curve reads as
        # broken right at the flag boundary.  On the Section 5 panel that is
        # the CE-lambda step itself, the feature the shaded band exists to
        # show, and a gap there reads as missing data rather than as a jump.
        # Bridge each crossing with a line-only segment in the faded style;
        # NaN separators keep it to one artist however many crossings there
        # are, and no marker is drawn so neither endpoint is doubled.
        cross = np.flatnonzero((solid[:-1] & faded[1:]) | (faded[:-1] & solid[1:]))
        if cross.size:
            xb = np.full(3 * cross.size, np.nan)
            yb = np.full(3 * cross.size, np.nan)
            xb[0::3], xb[1::3] = x[cross], x[cross + 1]
            yb[0::3], yb[1::3] = y[cross], y[cross + 1]
            ax.plot(
                xb,
                yb,
                color=color,
                ls=ls,
                lw=lw,
                alpha=faded_alpha * alpha,
                zorder=1,
            )

    # Measured but below the precision gate.  Drawn whether or not the caller
    # supplies an uncertainty: the open marker is a statement about how many
    # systems back the bin, which stands on its own.
    if n is not None:
        # A (2, N) yerr is the asymmetric [lower, upper] pair; a 1-D yerr is
        # the symmetric bar.  Matplotlib wants the asymmetric form as (2, k)
        # after the subset, so index the columns rather than the rows.
        err = None if yerr is None else np.asarray(yerr, dtype=float)
        thin = (~ok) & drawable & (n > 0)
        for sub, a in ((thin & ~art, alpha), (thin & art, faded_alpha * alpha)):
            if not sub.any():
                continue
            if err is None:
                bar = None
            else:
                bar = err[:, sub] if err.ndim == 2 else err[sub]
            ax.errorbar(
                x[sub],
                y[sub],
                yerr=bar,
                fmt=marker,
                ms=ms,
                mfc="none",
                mec=color,
                ecolor=color,
                elinewidth=0.7,
                capsize=1.5,
                ls="none",
                alpha=a,
                zorder=1,
            )

    # Exact zeros, as upper limits.
    if y_ul is not None and n is not None:
        ul = np.asarray(y_ul, dtype=float)
        empty = (n == 0) & np.isfinite(ul) & (ul > 0)
        for sub, a in ((empty & ~art, alpha), (empty & art, faded_alpha * alpha)):
            if not sub.any():
                continue
            container = ax.errorbar(
                x[sub],
                ul[sub],
                yerr=ul[sub] * 0.30,
                uplims=True,
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                alpha=a,
                zorder=1,
            )
            # matplotlib draws the uplims caret at the default marker size,
            # which swamps the data markers on a crowded panel.
            for cap in container.lines[1]:
                cap.set_markersize(ms * 1.6)
    return handle
