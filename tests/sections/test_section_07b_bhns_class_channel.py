"""Section 7b of grb_main.ipynb: BHNS R(z) per Gottlieb class split by Broekgaarden channel.

End-to-end checks on the live Model A pipeline at fiducial BH spin
``A_BH_FID = 0.5``.  Same skeleton as
[tests/sections/test_section_06b_bns_class_channel.py](tests/sections/test_section_06b_bns_class_channel.py),
with two BHNS-specific additions:

- ``apply_bhns_misalignment`` is applied per cell to the two GRB classes
  only, as in Sections 7 and 7c: a No-GRB system swallows its NS and
  launches no jet, so there is nothing to misalign.  A dedicated test
  checks the GRB cells are half the intrinsic rate and the No-GRB cells
  equal it, so the factor cannot drift to the wrong class or level;
- the dominant-channel anchor for the No-GRB class is Channel I (Stable
  MT + CE), reflecting the Broekgaarden+ 2021 (Sec. 3.3, 4.1.3) finding
  that the classic channel dominates the BHNS population.

Both reuse the shared ``_model_cache.get_model("A")`` and the new
``bhns_ch`` slot added there for the per-channel BHNS sample.

Marker stack: ``requires_data`` + ``requires_compas`` + ``slow``.

References: Gottlieb et al. (2023), arXiv:2309.00038, Sec. 4 (BHNS
disk-mass cuts); Broekgaarden et al. (2021), arXiv:2103.02608, Sec. 5
(channels I-V) and Sec. 3.3, 4.1.3 (BHNS Channel I dominance); Fragos et al. (2010), ApJL 719, L79;
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

from grb_classify import (  # noqa: E402
    FORMATION_CHANNEL_INPUTS,
    classify_bhns,
    classify_formation_channels,
)
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

# Per-(class, channel) R(z = 0) as drawn, in Gpc^-3 yr^-1 (x0.5 misalignment
# on the two GRB classes, intrinsic for No GRB; the No-GRB pins doubled on
# 2026-10-22 when the halving was removed from that class) for Model A at A_BH_FID = 0.5 under the TNG100-1 S(Z, z) table +
# Planck 2015 cosmology, ``smooth_sigma = 0``.  Channels follow
# Broekgaarden+ 2021 Sec. 3.1 as reproduced in ``classify_formation_channels``
# (re-pinned 2026-10-21; the earlier pins predated the TNG100-1 table).
# See ``_assert_pinned`` for tolerance rules.
_R0_PINS_BHNS = {
    ("No GRB", "I  Stable MT + CE"): 47.2274,
    ("No GRB", "II  Stable MT only"): 1.41755,
    ("No GRB", "III Single-core CE"): 3.03421,
    ("No GRB", "IV  Double-core CE"): 0.378848,
    ("No GRB", "V   Other"): 2.97784,
    ("Faint lbGRB (BHNS)", "I  Stable MT + CE"): 5.29523,
    ("Faint lbGRB (BHNS)", "II  Stable MT only"): 0.494472,
    ("Faint lbGRB (BHNS)", "III Single-core CE"): 0.643221,
    ("Faint lbGRB (BHNS)", "IV  Double-core CE"): 0.010384,
    ("Faint lbGRB (BHNS)", "V   Other"): 0.0360843,
    ("lbGRB + red KN (BHNS disk)", "I  Stable MT + CE"): 0.626756,
    ("lbGRB + red KN (BHNS disk)", "II  Stable MT only"): 0.567003,
    ("lbGRB + red KN (BHNS disk)", "III Single-core CE"): 0.354838,
    ("lbGRB + red KN (BHNS disk)", "IV  Double-core CE"): 0.00091036,
    ("lbGRB + red KN (BHNS disk)", "V   Other"): 0.00337726,
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


_MISALIGNED = {"Faint lbGRB (BHNS)", "lbGRB + red KN (BHNS disk)"}


def _misalign(cl, R):
    """The 0.5x jet-misalignment factor, on the two GRB classes only."""
    return apply_bhns_misalignment(R) if cl in _MISALIGNED else R


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
    channels = classify_formation_channels(**{k: bhns_ch[k] for k in FORMATION_CHANNEL_INPUTS})

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
        cl: {ch: _misalign(cl, R_cell_intrinsic[cl][ch]) for ch in _CHANNEL_KEYS}
        for cl in _CLASS_KEYS
    }
    R_per_class_intrinsic = {cl: _R_intrinsic(classes[cl]) for cl in _CLASS_KEYS}
    R_per_class = {cl: _misalign(cl, R_per_class_intrinsic[cl]) for cl in _CLASS_KEYS}
    # Each channel integrated whole, then the No-GRB part restored to
    # intrinsic: 0.5 R(ch) + 0.5 R(ch & No GRB).  Built from different
    # integrations than R_cell, so tripwire B checks additivity for real.
    R_per_channel = {
        ch: apply_bhns_misalignment(_R_intrinsic(channels[ch]))
        + (1.0 - apply_bhns_misalignment(1.0)) * _R_intrinsic(classes["No GRB"] & channels[ch])
        for ch in _CHANNEL_KEYS
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
def test_section_07b_channel_sum_equals_misalignment_corrected_per_class_total(
    bhns_class_channel_rates,
):
    """sum_ch R_cell[cl][ch] equals the class total, corrected the same way.

    Holds at rtol = 1e-12 because the misalignment correction is a scalar
    multiplier per class (Fragos+ 2010, Kawaguchi+ 2015) and therefore
    commutes with the per-channel sum.
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
            err_msg=f"Section 7b tripwire A failed for class {cl!r}",
        )


# ─────────────────────────────────────────────────────────────────────
# 2. Tripwire B: sum over classes per channel == per-channel total
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07b_class_sum_equals_per_channel_total(bhns_class_channel_rates):
    """sum_cl R_cell[cl][ch] == R_per_channel[ch] at rtol = 1e-12 for every channel.

    The Gottlieb three-class BHNS scheme is exhaustive over the BHNS
    sample, so disjoint class masks within a channel sum to the
    per-channel total.  Both sides carry the factor on the GRB classes only.
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
            err_msg=f"Section 7b tripwire B failed for channel {ch!r}",
        )


# ─────────────────────────────────────────────────────────────────────
# 3. Non-negativity at every z, every cell
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07b_all_cells_nonnegative_everywhere(bhns_class_channel_rates):
    """Every (class, channel) curve >= 0 at every redshift.

    Same motivation as Section 6b: log-scaled y-axis hides negative
    values silently.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    for cl in _CLASS_KEYS:
        for ch in _CHANNEL_KEYS:
            R = R_cell[cl][ch]
            assert np.all(R >= 0.0), f"({cl!r}, {ch!r}) has negative R(z): min = {R.min():.3e}"


# ─────────────────────────────────────────────────────────────────────
# 4. GRB cells are half the intrinsic rate; No-GRB cells are intrinsic
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07b_misalignment_only_on_grb_classes(bhns_class_channel_rates):
    """R_cell = MISALIGNMENT_FACTOR x intrinsic for the two GRB classes, and
    R_cell = intrinsic for No GRB, which launches no jet to misalign.

    An earlier version halved No GRB too, drawing that panel a factor 2 low
    against Sections 7 and 7c.

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
        factor = MISALIGNMENT_SYSTEMATIC_FACTOR if cl in _MISALIGNED else 1.0
        for ch in _CHANNEL_KEYS:
            np.testing.assert_allclose(
                R_cell[cl][ch],
                factor * R_cell_intrinsic[cl][ch],
                rtol=1e-12,
                atol=0,
                err_msg=f"({cl!r}, {ch!r}) rate is not {factor} x intrinsic",
            )
    assert "No GRB" not in _MISALIGNED


# ─────────────────────────────────────────────────────────────────────
# 5. Channel III is material in every BHNS class
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07b_channel_III_material(bhns_class_channel_rates):
    """Channel III carries a real share of every populated BHNS class.

    It reads as empty whenever the single-core CE is missed, and
    Broekgaarden+ 2021 Table 5 puts f_III between 2.1 and 60 per cent of the
    detectable BHNS rate across their 420 variations, so an exactly-zero
    channel III is a bug signature rather than a result.  Channel V is not
    empty: it holds Case A first transfers and the secondary-first systems
    (Broekgaarden+ 2021 Sec. 3.1.5), 5.4 percent of the No GRB class.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    R_per_class = bhns_class_channel_rates["R_per_class"]
    iz0 = bhns_class_channel_rates["mod"]["iz0"]
    for cl in _CLASS_KEYS:
        R_total = float(R_per_class[cl][iz0])
        if R_total <= 0.0:
            continue
        f_III = float(R_cell[cl]["III Single-core CE"][iz0]) / R_total
        assert f_III > 1e-2, (
            f"Channel III is negligible in {cl!r}: f_III = {f_III:.3e}.  "
            f"Below 1 per cent the single-core CE is almost certainly being missed."
        )


# ─────────────────────────────────────────────────────────────────────
# 6. Channel I dominates the No-GRB BHNS class
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07b_channel_I_dominates_no_grb_class(bhns_class_channel_rates):
    """Channel I weight in the No-GRB class > 0.5 of that class total at z = 0.

    Anchor: Broekgaarden+ 2021 Sec. 3.3 and 4.1.3 find that Channel I (stable
    MT followed by a CE) dominates the BHNS yield, with f_I > 50 percent of
    the detectable rate in every model except their B, C and D.  The No-GRB
    cell holds the bulk of the BHNS rate (NS swallowed whole, no
    accretion disk), so this is the cleanest place to pin the
    population-level dominance: f_I = 0.86 on Model A.  Threshold of 0.5
    keeps the test robust to small drifts in the calibration chain.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    R_per_class = bhns_class_channel_rates["R_per_class"]
    iz0 = bhns_class_channel_rates["mod"]["iz0"]
    cl = "No GRB"
    R_total = float(R_per_class[cl][iz0])
    f_I = float(R_cell[cl]["I  Stable MT + CE"][iz0]) / R_total
    assert f_I > 0.5, (
        f"Channel I no longer dominates the No-GRB BHNS class: "
        f"f_I = {f_I:.3f} <= 0.5, against Broekgaarden+ 2021 Sec. 4.1.3."
    )


# ─────────────────────────────────────────────────────────────────────
# 7. Per-(class, channel) R(z = 0) regression pin
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07b_R0_per_cell_pinned_to_model_A(bhns_class_channel_rates):
    """Per-(class, channel) R(z = 0), as drawn, within 5 percent of pin.

    Same idiom as 7c: 5 percent relative above 1e-3 Gpc^-3 yr^-1, 1e-3
    absolute below.  Pinned against COMPAS commit ``81722d4`` + Levina+
    2026 TNG100-1 MSSFR at ``A_BH_FID = 0.5``, ``smooth_sigma = 0``.
    """
    R_cell = bhns_class_channel_rates["R_cell"]
    iz0 = bhns_class_channel_rates["mod"]["iz0"]
    for (cl, ch), pinned in _R0_PINS_BHNS.items():
        value = float(R_cell[cl][ch][iz0])
        _assert_pinned(value, pinned, f"R0[{cl!r}, {ch!r}]")
