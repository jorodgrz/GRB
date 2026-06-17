"""Section 7c of grb_main.ipynb: BNS R(z) per Gottlieb class split by Broekgaarden channel.

Six end-to-end checks on the live Model A pipeline, all reusing the
shared ``_model_cache.get_model("A")`` so the expensive load +
Alsing-remap + MEAN_MASS_EVOLVED calibration runs at most once per
session across Sections 7b / 7c / 8c / 8d / 12.

Layered with [tests/unit/test_class_channel_split.py](tests/unit/test_class_channel_split.py)
which exercises the same partition / non-negative / filter invariants
on synthetic input.  This file adds:

- both additivity tripwires (sum-over-channels and sum-over-classes)
  evaluated on the live Model A pipeline,
- three literature anchors on the per-class channel composition
  (Channel II essentially empty in BNS per Broekgaarden+ 2022 CE
  picture; Channels III + V together essentially empty; dominant
  channel per class is in {I, IV} under the fiducial alpha_CE = 1),
- a per-(class, channel) ``R(z = 0)`` regression pin so a future
  change to the COMPAS commit, the NS remap, the Levina+ 2026 MSSFR,
  or the calibration chain trips with a clear message.

Marker stack: ``requires_data`` + ``requires_compas`` + ``slow``.

References: Gottlieb et al. (2024), arXiv:2411.13657 (four-class
scheme); Broekgaarden et al. (2021), arXiv:2103.02608, Sec. 5
(channels I-V); Broekgaarden et al. (2022), arXiv:2112.05763,
Sec. 4.2 (BNS CE picture).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grb_classify import classify_bns_2024, classify_formation_channels  # noqa: E402
from grb_rates import compute_merger_rate  # noqa: E402
from tests.sections._model_cache import get_model as _get_model  # noqa: E402

_CLASS_KEYS = [
    "sbGRB + blue KN",
    "lbGRB + red KN (HMNS)",
    "lbGRB + red KN (disk)",
    "Faint lbGRB",
]
_CHANNEL_KEYS = [
    "I  Stable MT + CE",
    "II  Stable MT only",
    "III Single-core CE",
    "IV  Double-core CE",
    "V   Other",
]

# Per-(class, channel) R(z = 0) in Gpc^-3 yr^-1 for Model A under
# Levina+ 2026 TNG100-1 MSSFR + Planck 2015 cosmology, computed with
# ``smooth_sigma = 0`` (matches the cache contract; the notebook cell
# uses the default ``smooth_sigma = 30`` cosmetic Gaussian, which biases
# z = 0 upward by ~30% via the reflective boundary).  Pinned against the
# project's COMPAS commit ``81722d4``.  Channels not listed here have
# zero population in this run (Channels III and V are empty for BNS;
# Channel II is empty for BNS by the Broekgaarden+ 2022 CE picture)
# and are handled by ``_assert_pinned`` with a 1e-3 absolute tolerance.
_R0_PINS_BNS = {
    ("sbGRB + blue KN", "I  Stable MT + CE"): 16.6365,
    ("sbGRB + blue KN", "II  Stable MT only"): 0.0,
    ("sbGRB + blue KN", "III Single-core CE"): 0.0,
    ("sbGRB + blue KN", "IV  Double-core CE"): 5.14727,
    ("sbGRB + blue KN", "V   Other"): 0.0,
    ("lbGRB + red KN (HMNS)", "I  Stable MT + CE"): 3.93932,
    ("lbGRB + red KN (HMNS)", "II  Stable MT only"): 0.0,
    ("lbGRB + red KN (HMNS)", "III Single-core CE"): 0.0,
    ("lbGRB + red KN (HMNS)", "IV  Double-core CE"): 15.8076,
    ("lbGRB + red KN (HMNS)", "V   Other"): 0.0,
    ("lbGRB + red KN (disk)", "I  Stable MT + CE"): 0.871183,
    ("lbGRB + red KN (disk)", "II  Stable MT only"): 0.0,
    ("lbGRB + red KN (disk)", "III Single-core CE"): 0.0,
    ("lbGRB + red KN (disk)", "IV  Double-core CE"): 0.488862,
    ("lbGRB + red KN (disk)", "V   Other"): 0.0,
    ("Faint lbGRB", "I  Stable MT + CE"): 0.531779,
    ("Faint lbGRB", "II  Stable MT only"): 0.0,
    ("Faint lbGRB", "III Single-core CE"): 0.0,
    ("Faint lbGRB", "IV  Double-core CE"): 11.5617,
    ("Faint lbGRB", "V   Other"): 0.0,
}


def _assert_pinned(value: float, pinned: float, name: str) -> None:
    """Same idiom as ``test_section_06_formation_channels._assert_pinned``.

    Above 1e-3 Gpc^-3 yr^-1 the rate density is a population-level
    statistic, so a 5 percent relative tolerance is the right band for
    a regression tripwire.  Below 1e-3 the cell value is dominated by
    Monte-Carlo noise / a literally empty intersection, so a 1e-3
    absolute tolerance is the right band (``pytest.approx(0.0, rel=...)``
    is undefined for zero targets).
    """
    if pinned > 1e-3:
        assert value == pytest.approx(pinned, rel=5e-2), (
            f"{name} = {value:.6g} drifted from pin {pinned:.6g} by more than 5 percent (rtol)"
        )
    else:
        assert abs(value - pinned) < 1e-3, (
            f"{name} = {value:.6g} drifted from near-zero pin {pinned:.6g} by more than 1e-3 (atol)"
        )


def _build_per_cell_rates(mod):
    """Return ``(R_cell, R_per_class, R_per_channel, classes, channels)``.

    Computed once per test-session call.  Each test then operates on
    cached arrays so the per-test wall clock is dominated by the
    function's first invocation.
    """
    bns = mod["bns"]
    common = dict(
        redshifts=mod["redshifts"],
        times=mod["times"],
        time_first_SF=mod["time_first_SF"],
        n_formed=mod["sfr"] / mod["mme_bns"],
        p_draw=mod["p_draw"],
        dPdlogZ=mod["dPdlogZ"],
        metallicities=mod["metallicities"],
        smooth_sigma=0,
    )

    classes = classify_bns_2024(bns["m1"], bns["m2"])
    channels = classify_formation_channels(
        dblCE=bns["dblCE"],
        fc_CEE=bns["fc_CEE"],
        fc_mt_p1=bns["fc_mt_p1"],
        fc_mt_s1=bns["fc_mt_s1"],
        fc_mt_p1_K1=bns["fc_mt_p1_K1"],
        fc_mt_s1_K2=bns["fc_mt_s1_K2"],
    )

    def _R(mask):
        if not mask.any():
            return np.zeros_like(mod["redshifts"])
        return compute_merger_rate(
            COMPAS_Z=bns["metallicity"][mask],
            COMPAS_delay_times=bns["delay_time"][mask],
            COMPAS_weights=bns["weights"][mask],
            **common,
        )

    R_cell = {
        cl: {ch: _R(classes[cl] & channels[ch]) for ch in _CHANNEL_KEYS} for cl in _CLASS_KEYS
    }
    R_per_class = {cl: _R(classes[cl]) for cl in _CLASS_KEYS}
    R_per_channel = {ch: _R(channels[ch]) for ch in _CHANNEL_KEYS}
    return R_cell, R_per_class, R_per_channel, classes, channels


# ─────────────────────────────────────────────────────────────────────
# Session-scoped derived rates fixture
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def bns_class_channel_rates():
    mod = _get_model("A")
    R_cell, R_per_class, R_per_channel, _, _ = _build_per_cell_rates(mod)
    return {
        "mod": mod,
        "R_cell": R_cell,
        "R_per_class": R_per_class,
        "R_per_channel": R_per_channel,
    }


# ─────────────────────────────────────────────────────────────────────
# 1. Tripwire A: sum over channels per class == per-class total
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_channel_sum_equals_per_class_total(bns_class_channel_rates):
    """sum_ch R_cell[cl][ch] == R_per_class[cl] at rtol = 1e-12 for every BNS class.

    Mirrors the in-cell ``assert np.allclose(R_sum, merger_rates_BNS[cl_label])``
    tripwire on the live Model A pipeline.  ``compute_merger_rate`` is a
    pure additive accumulator over its COMPAS axis, so disjoint channel
    masks within a class must yield rates that sum to the per-class
    total at machine precision.
    """
    R_cell = bns_class_channel_rates["R_cell"]
    R_per_class = bns_class_channel_rates["R_per_class"]
    for cl in _CLASS_KEYS:
        R_sum = sum(R_cell[cl].values())
        np.testing.assert_allclose(
            R_sum,
            R_per_class[cl],
            rtol=1e-12,
            atol=0,
            err_msg=f"Section 7c tripwire A failed for class {cl!r}",
        )


# ─────────────────────────────────────────────────────────────────────
# 2. Tripwire B: sum over classes per channel == per-channel total
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_class_sum_equals_per_channel_total(bns_class_channel_rates):
    """sum_cl R_cell[cl][ch] == R_per_channel[ch] at rtol = 1e-12 for every channel.

    Mirrors the in-cell ``assert np.allclose(R_sum, merger_rates_ch[ch_label])``
    tripwire.  Same accumulator argument as tripwire A, transposed.  The
    Gottlieb four-class scheme is exhaustive over the BNS sample, so
    disjoint class masks within a channel must sum to the per-channel
    total.
    """
    R_cell = bns_class_channel_rates["R_cell"]
    R_per_channel = bns_class_channel_rates["R_per_channel"]
    for ch in _CHANNEL_KEYS:
        R_sum = sum(R_cell[cl][ch] for cl in _CLASS_KEYS)
        np.testing.assert_allclose(
            R_sum,
            R_per_channel[ch],
            rtol=1e-12,
            atol=0,
            err_msg=f"Section 7c tripwire B failed for channel {ch!r}",
        )


# ─────────────────────────────────────────────────────────────────────
# 3. All (class, channel) cells return non-negative R(z) everywhere
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_all_cells_nonnegative_everywhere(bns_class_channel_rates):
    """Every (class, channel) curve is >= 0 at every redshift.

    The Section 7c y-axis is log-scaled (``ax.set_yscale('log')``), so a
    negative value from a future regression would silently disappear
    instead of failing visibly on the figure.
    """
    R_cell = bns_class_channel_rates["R_cell"]
    for cl in _CLASS_KEYS:
        for ch in _CHANNEL_KEYS:
            R = R_cell[cl][ch]
            assert np.all(R >= 0.0), f"({cl!r}, {ch!r}) has negative R(z): min = {R.min():.3e}"


# ─────────────────────────────────────────────────────────────────────
# 4. Channel II contributes negligibly to every BNS class (Broekgaarden+ 2022)
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_channel_II_negligible_in_every_BNS_class(bns_class_channel_rates):
    """f_II / f_total < 1e-2 in every BNS class at z = 0.

    Anchor: Broekgaarden et al. (2022) arXiv:2112.05763, Sec. 4.2.  BNS
    progenitors must shrink the orbit by a CE episode to merge within a
    Hubble time, so the "stable MT only" channel is essentially empty
    for BNS in Model A (alpha_CE = 1, Fryer+ 2012 delayed engine).  A
    future COMPAS commit that bypasses this constraint (e.g. an updated
    stability criterion permitting stable-only BNS) trips this test.
    """
    R_cell = bns_class_channel_rates["R_cell"]
    R_per_class = bns_class_channel_rates["R_per_class"]
    iz0 = bns_class_channel_rates["mod"]["iz0"]
    for cl in _CLASS_KEYS:
        R_total = float(R_per_class[cl][iz0])
        if R_total <= 0.0:
            continue
        f_II = float(R_cell[cl]["II  Stable MT only"][iz0]) / R_total
        assert f_II < 1e-2, (
            f"Channel II is non-negligible in {cl!r}: "
            f"f_II = {f_II:.3e} >= 1e-2 (Broekgaarden+ 2022 CE picture)"
        )


# ─────────────────────────────────────────────────────────────────────
# 5. Channels III + V together negligible in every BNS class
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_channels_III_V_together_negligible(bns_class_channel_rates):
    """(f_III + f_V) / f_total < 1e-2 in every BNS class at z = 0.

    Same anchor family as Channel II: Channels III (single-core CE) and
    V (catch-all "Other") carry no measurable BNS weight under the
    fiducial Model A prescription.  The CH_PLOT legend filter in
    [grb_main.ipynb] Section 7c drops both from the legend; this test
    pins the underlying numerical fact, not just the visual choice.
    """
    R_cell = bns_class_channel_rates["R_cell"]
    R_per_class = bns_class_channel_rates["R_per_class"]
    iz0 = bns_class_channel_rates["mod"]["iz0"]
    for cl in _CLASS_KEYS:
        R_total = float(R_per_class[cl][iz0])
        if R_total <= 0.0:
            continue
        f_III = float(R_cell[cl]["III Single-core CE"][iz0]) / R_total
        f_V = float(R_cell[cl]["V   Other"][iz0]) / R_total
        assert (f_III + f_V) < 1e-2, (
            f"Channels III + V are non-negligible in {cl!r}: "
            f"f_III + f_V = {f_III + f_V:.3e} >= 1e-2"
        )


# ─────────────────────────────────────────────────────────────────────
# 6. Dominant channel per BNS class is in {I, IV}
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_dominant_channel_per_class_is_I_or_IV(bns_class_channel_rates):
    """argmax_ch R(z = 0) per BNS class is "I  Stable MT + CE" or "IV  Double-core CE".

    Model-A-specific anchor: under alpha_CE = 1, Fryer+ 2012 delayed
    engine and Levina+ 2026 TNG100-1 MSSFR, the only channels with
    appreciable BNS weight are I (stable MT followed by a CE) and IV
    (double-core CE; bulk of the symmetric near-equal-mass HMNS-engine
    BNS).  Re-pinning required if the test is ported to F / G / J / K.
    """
    R_cell = bns_class_channel_rates["R_cell"]
    iz0 = bns_class_channel_rates["mod"]["iz0"]
    allowed = {"I  Stable MT + CE", "IV  Double-core CE"}
    for cl in _CLASS_KEYS:
        per_ch = {ch: float(R_cell[cl][ch][iz0]) for ch in _CHANNEL_KEYS}
        dominant = max(per_ch, key=per_ch.get)
        assert dominant in allowed, (
            f"Class {cl!r}: dominant channel at z = 0 is {dominant!r} "
            f"(R = {per_ch[dominant]:.3f}); expected one of {sorted(allowed)}.  "
            f"Re-pin if Model has changed."
        )


# ─────────────────────────────────────────────────────────────────────
# 7. Per-(class, channel) R(z = 0) regression pin
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_R0_per_cell_pinned_to_model_A(bns_class_channel_rates):
    """Per-(class, channel) R(z = 0) within 5 percent of the COMPAS-81722d4 pin.

    Tolerance scheme: 5 percent relative above 1e-3 Gpc^-3 yr^-1, 1e-3
    absolute below.  Pin source: ``tests/sections/_extract_pins.py`` run
    against the project's pinned COMPAS commit ``81722d4`` and Levina+
    2026 TNG100-1 MSSFR, ``smooth_sigma = 0``.  A drift here flags one
    of: COMPAS-pin change, NS remap change, MSSFR change, calibration
    change in ``calibrate_mean_mass_evolved``.
    """
    R_cell = bns_class_channel_rates["R_cell"]
    iz0 = bns_class_channel_rates["mod"]["iz0"]
    for (cl, ch), pinned in _R0_PINS_BNS.items():
        value = float(R_cell[cl][ch][iz0])
        _assert_pinned(value, pinned, f"R0[{cl!r}, {ch!r}]")
