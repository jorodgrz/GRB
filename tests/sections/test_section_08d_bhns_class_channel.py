"""Section 8d of grb_main.ipynb: BHNS R(z) per Gottlieb class split by Broekgaarden channel.

End-to-end checks on the live Model A pipeline at fiducial BH spin
``A_BH_FID = 0.5``.  Same skeleton as
[tests/sections/test_section_07c_bns_class_channel.py](tests/sections/test_section_07c_bns_class_channel.py),
with two BHNS-specific additions:

- ``apply_bhns_misalignment`` is applied per cell (matching the cell
  pattern at ``grb_main.ipynb`` line ~2580); a dedicated test verifies
  the per-cell rate equals the half-intrinsic rate so the misalignment
  factor cannot drift to the wrong level (rate-level vs sample-level)
  without tripping a clear failure;
- the dominant-channel anchor for the No-GRB class is Channel I (Stable
  MT + CE), reflecting the Broekgaarden+ 2022 finding that the BHNS
  population is dominated by stable mass transfer.

Both reuse the shared ``_model_cache.get_model("A")`` and the new
``bhns_ch`` slot added there for the per-channel BHNS sample.

Marker stack: ``requires_data`` + ``requires_compas`` + ``slow``.

References: Gottlieb et al. (2023), arXiv:2309.00038, Sec. 4 (BHNS
disk-mass cuts); Broekgaarden et al. (2021), arXiv:2103.02608, Sec. 5
(channels I-V); Broekgaarden et al. (2022), arXiv:2112.05763, Sec. 4.2
(BHNS Channel I dominance); Fragos et al. (2010), ApJL 719, L79;
Kawaguchi et al. (2015), ApJ 807, 95 (BHNS misalignment).
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grb_classify import classify_bhns, classify_formation_channels  # noqa: E402
from grb_rates import apply_bhns_misalignment, compute_merger_rate  # noqa: E402
from tests.sections._model_cache import get_model as _get_model  # noqa: E402

A_BH_FID = inspect.signature(classify_bhns).parameters["a_BH"].default
"""Pull fiducial BH spin from ``classify_bhns`` rather than inlining 0.5,
so the section test stays in lockstep with whatever default the figure
code uses."""

_CLASS_KEYS = [
    "No GRB",
    "Faint lbGRB (BHNS)",
    "lbGRB + red KN (BHNS disk)",
]
_CHANNEL_KEYS = [
    "I  Stable MT + CE",
    "II  Stable MT only",
    "III Single-core CE",
    "IV  Double-core CE",
    "V   Other",
]

# Per-(class, channel) misalignment-corrected R(z = 0) in Gpc^-3 yr^-1
# for Model A at A_BH_FID = 0.5 under Levina+ 2026 TNG100-1 MSSFR +
# Planck 2015 cosmology, ``smooth_sigma = 0``.  Pinned against the
# project's COMPAS commit ``81722d4``.  Channels III and V are empty
# for BHNS in this run.  See ``_assert_pinned`` for tolerance rules.
_R0_PINS_BHNS = {
    ("No GRB", "I  Stable MT + CE"): 29.7408,
    ("No GRB", "II  Stable MT only"): 0.504796,
    ("No GRB", "III Single-core CE"): 0.0,
    ("No GRB", "IV  Double-core CE"): 0.135121,
    ("No GRB", "V   Other"): 0.0,
    ("Faint lbGRB (BHNS)", "I  Stable MT + CE"): 8.79915,
    ("Faint lbGRB (BHNS)", "II  Stable MT only"): 0.947377,
    ("Faint lbGRB (BHNS)", "III Single-core CE"): 0.0,
    ("Faint lbGRB (BHNS)", "IV  Double-core CE"): 0.155602,
    ("Faint lbGRB (BHNS)", "V   Other"): 0.0,
    ("lbGRB + red KN (BHNS disk)", "I  Stable MT + CE"): 1.34086,
    ("lbGRB + red KN (BHNS disk)", "II  Stable MT only"): 0.538304,
    ("lbGRB + red KN (BHNS disk)", "III Single-core CE"): 0.0,
    ("lbGRB + red KN (BHNS disk)", "IV  Double-core CE"): 0.0685578,
    ("lbGRB + red KN (BHNS disk)", "V   Other"): 0.0,
}


def _assert_pinned(value: float, pinned: float, name: str) -> None:
    """5 percent relative above 1e-3, 1e-3 absolute below.  Same as 7c."""
    if pinned > 1e-3:
        assert value == pytest.approx(pinned, rel=5e-2), (
            f"{name} = {value:.6g} drifted from pin {pinned:.6g} by more than 5 percent (rtol)"
        )
    else:
        assert abs(value - pinned) < 1e-3, (
            f"{name} = {value:.6g} drifted from near-zero pin {pinned:.6g} by more than 1e-3 (atol)"
        )


def _build_per_cell_rates(mod):
    """Return ``(R_cell, R_cell_intrinsic, R_per_class, R_per_channel, ...)``.

    The intrinsic per-cell rates are kept around for the half-intrinsic
    test below; the misalignment-corrected rates are what the figure
    plots.
    """
    bhns_ch = mod["bhns_ch"]
    common = dict(
        redshifts=mod["redshifts"],
        times=mod["times"],
        time_first_SF=mod["time_first_SF"],
        n_formed=mod["sfr"] / mod["mme_bhns"],
        p_draw=mod["p_draw"],
        dPdlogZ=mod["dPdlogZ"],
        metallicities=mod["metallicities"],
        smooth_sigma=0,
    )

    classes = classify_bhns(bhns_ch["M_BH"], bhns_ch["M_NS"], a_BH=A_BH_FID)
    channels = classify_formation_channels(
        dblCE=bhns_ch["dblCE"],
        fc_CEE=bhns_ch["fc_CEE"],
        fc_mt_p1=bhns_ch["fc_mt_p1"],
        fc_mt_s1=bhns_ch["fc_mt_s1"],
        fc_mt_p1_K1=bhns_ch["fc_mt_p1_K1"],
        fc_mt_s1_K2=bhns_ch["fc_mt_s1_K2"],
    )

    def _R_intrinsic(mask):
        if not mask.any():
            return np.zeros_like(mod["redshifts"])
        return compute_merger_rate(
            COMPAS_Z=bhns_ch["metallicity"][mask],
            COMPAS_delay_times=bhns_ch["delay_time"][mask],
            COMPAS_weights=bhns_ch["weights"][mask],
            **common,
        )

    R_cell_intrinsic = {
        cl: {ch: _R_intrinsic(classes[cl] & channels[ch]) for ch in _CHANNEL_KEYS}
        for cl in _CLASS_KEYS
    }
    R_cell = {
        cl: {ch: apply_bhns_misalignment(R_cell_intrinsic[cl][ch]) for ch in _CHANNEL_KEYS}
        for cl in _CLASS_KEYS
    }
    R_per_class_intrinsic = {cl: _R_intrinsic(classes[cl]) for cl in _CLASS_KEYS}
    R_per_class = {cl: apply_bhns_misalignment(R_per_class_intrinsic[cl]) for cl in _CLASS_KEYS}
    R_per_channel = {
        ch: apply_bhns_misalignment(_R_intrinsic(channels[ch])) for ch in _CHANNEL_KEYS
    }

    return {
        "R_cell": R_cell,
        "R_cell_intrinsic": R_cell_intrinsic,
        "R_per_class": R_per_class,
        "R_per_class_intrinsic": R_per_class_intrinsic,
        "R_per_channel": R_per_channel,
    }


@pytest.fixture(scope="module")
def bhns_class_channel_rates():
    mod = _get_model("A")
    cached = _build_per_cell_rates(mod)
    cached["mod"] = mod
    return cached


# ─────────────────────────────────────────────────────────────────────
# 1. Tripwire A: sum over channels per class == per-class total (misalignment-corrected)
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_08d_channel_sum_equals_misalignment_corrected_per_class_total(
    bhns_class_channel_rates,
):
    """sum_ch R_cell[cl][ch] == apply_bhns_misalignment(R_per_class_intrinsic[cl]).

    Mirrors the in-cell ``assert np.allclose(R_sum, apply_bhns_misalignment(...))``
    tripwire at notebook line ~2596.  Holds at rtol = 1e-12 because the
    misalignment correction is a scalar multiplier (Fragos+ 2010,
    Kawaguchi+ 2015) and therefore commutes with the per-channel sum.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    R_per_class = bhns_class_channel_rates["R_per_class"]
    for cl in _CLASS_KEYS:
        R_sum = sum(R_cell[cl].values())
        np.testing.assert_allclose(
            R_sum,
            R_per_class[cl],
            rtol=1e-12,
            atol=0,
            err_msg=f"Section 8d tripwire A failed for class {cl!r}",
        )


# ─────────────────────────────────────────────────────────────────────
# 2. Tripwire B: sum over classes per channel == per-channel total
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_08d_class_sum_equals_per_channel_total(bhns_class_channel_rates):
    """sum_cl R_cell[cl][ch] == R_per_channel[ch] at rtol = 1e-12 for every channel.

    The Gottlieb three-class BHNS scheme is exhaustive over the BHNS
    sample, so disjoint class masks within a channel sum to the
    per-channel total.  Both sides are misalignment-corrected.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    R_per_channel = bhns_class_channel_rates["R_per_channel"]
    for ch in _CHANNEL_KEYS:
        R_sum = sum(R_cell[cl][ch] for cl in _CLASS_KEYS)
        np.testing.assert_allclose(
            R_sum,
            R_per_channel[ch],
            rtol=1e-12,
            atol=0,
            err_msg=f"Section 8d tripwire B failed for channel {ch!r}",
        )


# ─────────────────────────────────────────────────────────────────────
# 3. Non-negativity at every z, every cell
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_08d_all_cells_nonnegative_everywhere(bhns_class_channel_rates):
    """Every (class, channel) curve >= 0 at every redshift.

    Same motivation as Section 7c: log-scaled y-axis hides negative
    values silently.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    for cl in _CLASS_KEYS:
        for ch in _CHANNEL_KEYS:
            R = R_cell[cl][ch]
            assert np.all(R >= 0.0), f"({cl!r}, {ch!r}) has negative R(z): min = {R.min():.3e}"


# ─────────────────────────────────────────────────────────────────────
# 4. Per-cell rate equals half the intrinsic per-cell rate
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_08d_each_cell_equals_half_intrinsic(bhns_class_channel_rates):
    """R_cell[cl][ch] == MISALIGNMENT_FACTOR * R_cell_intrinsic[cl][ch].

    Catches a future regression where the misalignment correction
    drifts from the rate level (population-averaged factor of 0.5) to
    the sample level (dropping individual systems).  The latter is
    physically wrong because the per-system misalignment angle is not
    in the COMPAS output (Fragos+ 2010 argue from spin-orbit dynamics,
    not from explicit kick orientations), so applying it sample-level
    would either double-count or under-count depending on the
    implementation.
    """
    from grb_physics import MISALIGNMENT_SYSTEMATIC_FACTOR

    R_cell = bhns_class_channel_rates["R_cell"]
    R_cell_intrinsic = bhns_class_channel_rates["R_cell_intrinsic"]
    for cl in _CLASS_KEYS:
        for ch in _CHANNEL_KEYS:
            np.testing.assert_allclose(
                R_cell[cl][ch],
                MISALIGNMENT_SYSTEMATIC_FACTOR * R_cell_intrinsic[cl][ch],
                rtol=1e-12,
                atol=0,
                err_msg=(
                    f"({cl!r}, {ch!r}) misalignment-corrected rate does "
                    f"not equal MISALIGNMENT_FACTOR x intrinsic"
                ),
            )


# ─────────────────────────────────────────────────────────────────────
# 5. Channels III + V together negligible in every BHNS class
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_08d_channels_III_V_together_negligible(bhns_class_channel_rates):
    """(f_III + f_V) / f_total < 1e-2 in every BHNS class at z = 0.

    Channels III (single-core CE) and V (catch-all "Other") carry no
    measurable BHNS weight in Model A; CH_PLOT filter in
    ``grb_main.ipynb`` Section 8d drops both from the legend.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    R_per_class = bhns_class_channel_rates["R_per_class"]
    iz0 = bhns_class_channel_rates["mod"]["iz0"]
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
# 6. Channel I dominates the No-GRB BHNS class
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_08d_channel_I_dominates_no_grb_class(bhns_class_channel_rates):
    """Channel I weight in the No-GRB class > 0.5 of that class total at z = 0.

    Anchor: Broekgaarden+ 2022 Sec. 4.2 finds that Channel I (stable MT
    followed by a CE) dominates the BHNS population in Model A, by an
    order of magnitude over the next-largest channel.  The No-GRB
    cell holds the bulk of the BHNS rate (NS swallowed whole, no
    accretion disk), so this is the cleanest place to pin the
    population-level dominance: 41 / (41 + 0.65 + 0.19) ~ 0.98 from the
    notebook print table.  Threshold of 0.5 keeps the test robust to
    small drifts in the calibration chain.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    R_per_class = bhns_class_channel_rates["R_per_class"]
    iz0 = bhns_class_channel_rates["mod"]["iz0"]
    cl = "No GRB"
    R_total = float(R_per_class[cl][iz0])
    f_I = float(R_cell[cl]["I  Stable MT + CE"][iz0]) / R_total
    assert f_I > 0.5, (
        f"Channel I no longer dominates the No-GRB BHNS class: "
        f"f_I = {f_I:.3f} <= 0.5.  Broekgaarden+ 2022 picture is broken."
    )


# ─────────────────────────────────────────────────────────────────────
# 7. Per-(class, channel) R(z = 0) regression pin
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_08d_R0_per_cell_pinned_to_model_A(bhns_class_channel_rates):
    """Per-(class, channel) misalignment-corrected R(z = 0) within 5 percent of pin.

    Same idiom as 7c: 5 percent relative above 1e-3 Gpc^-3 yr^-1, 1e-3
    absolute below.  Pinned against COMPAS commit ``81722d4`` + Levina+
    2026 TNG100-1 MSSFR at ``A_BH_FID = 0.5``, ``smooth_sigma = 0``.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    iz0 = bhns_class_channel_rates["mod"]["iz0"]
    for (cl, ch), pinned in _R0_PINS_BHNS.items():
        value = float(R_cell[cl][ch][iz0])
        _assert_pinned(value, pinned, f"R0[{cl!r}, {ch!r}]")
