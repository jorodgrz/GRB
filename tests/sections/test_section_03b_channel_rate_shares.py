"""Formation-channel shares of the Model A local merger rate, and their metallicity origin.

Section 3b draws channel shares weighted evenly over the 53 grid metallicities.
The number that compares with observation is the share of the z = 0 merger rate
on the fiducial TNG100-1 S(Z, z), and for BNS that share is set by where in
metallicity the local rate comes from.  These tests pin the rate-weighted
shares, two independent routes to them, and the metallicity structure behind
the BNS split, so the captions cannot drift from the pipeline.

Channel I BNS are the systems whose stripped primary explodes as an
electron-capture SN.  Broekgaarden+ 2021 (Sec. 2, after Hurley+ 2002) assign an
ECSN to a helium core of 1.6 to 2.25 Msun, and the yield of this route turns on
and off between neighbouring grid metallicities, which is what makes the BNS
channel split depend on metallicity in steps rather than smoothly.  The
loaders drop failed common-envelope ejections, which in Model A BNS come only
out of channels IV and V, so BNS channel I is the same system set with or
without that cut.
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

from grb_classify import FORMATION_CHANNEL_INPUTS, classify_formation_channels  # noqa: E402
from grb_rates import compute_merger_rate, per_system_rate_weights  # noqa: E402
from tests.sections._model_cache import get_model as _get_model  # noqa: E402

_I, _IV = "I  Stable MT + CE", "IV  Double-core CE"
# Grid points at or above 0.022; the grid has nothing between 0.0197 and 0.022.
_Z_HIGH = 0.021


def _setup(population: str):
    mod = _get_model("A")
    d = mod["bns"] if population == "BNS" else mod["bhns_ch"]
    ch = classify_formation_channels(**{k: d[k] for k in FORMATION_CHANNEL_INPUTS})
    common = dict(
        redshifts=mod["redshifts"],
        times=mod["times"],
        time_first_SF=mod["time_first_SF"],
        n_formed=mod["sfr"] / mod["M_SF"],
        p_draw=mod["p_draw"],
        dPdlogZ=mod["dPdlogZ"],
        metallicities=mod["metallicities"],
    )
    iz0 = int(np.argmin(np.abs(mod["redshifts"])))
    Z, t, w = d["metallicity"], d["delay_time"], d["weights"]

    def R0(mask):
        if not mask.any():
            return 0.0
        return compute_merger_rate(
            COMPAS_Z=Z[mask],
            COMPAS_delay_times=t[mask],
            COMPAS_weights=w[mask],
            smooth_sigma=0,
            **common,
        )[iz0]

    return d, ch, common, R0


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
@pytest.mark.parametrize(
    "population, expected",
    [
        ("BNS", {_I: 0.50923, _IV: 0.48171, "V   Other": 0.00898}),
        ("BHNS", {_I: 0.83268, "II  Stable MT only": 0.04991, "III Single-core CE": 0.07085}),
    ],
)
def test_channel_shares_of_the_local_rate(population, expected):
    """Channel shares of R(z = 0), two routes: per-channel cosmic integration and
    per-system rate weights summed over each channel."""
    d, ch, common, R0 = _setup(population)
    total = R0(np.ones(d["weights"].size, dtype=bool))
    per_channel = {c: R0(m) for c, m in ch.items()}
    np.testing.assert_allclose(sum(per_channel.values()), total, rtol=1e-12)

    w_sys = per_system_rate_weights(
        0.0,
        COMPAS_Z=d["metallicity"],
        COMPAS_delay_times=d["delay_time"],
        COMPAS_weights=d["weights"],
        **common,
    )
    np.testing.assert_allclose(w_sys.sum(), total, rtol=1e-6)
    for c, share in expected.items():
        assert per_channel[c] / total == pytest.approx(share, abs=2e-4)
        assert w_sys[ch[c]].sum() / w_sys.sum() == pytest.approx(share, abs=2e-4)


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_bns_channel_split_rests_on_the_highest_grid_metallicities():
    """A quarter of the BNS local rate comes from Z >= 0.022, where the
    double-core channel is 99 percent; below it channel I carries two thirds,
    and without the top grid point alone IV holds 39 percent."""
    d, ch, _, R0 = _setup("BNS")
    Z = d["metallicity"]
    total = R0(np.ones(Z.size, dtype=bool))
    high = Z >= _Z_HIGH
    low = ~high
    no_top = Z < Z.max()

    assert R0(high) / total == pytest.approx(0.231, abs=1e-3)
    assert R0(Z == Z.max()) / total == pytest.approx(0.149, abs=1e-3)
    assert R0(high & ch[_IV]) / R0(high) == pytest.approx(0.987, abs=1e-3)
    assert R0(low & ch[_I]) / R0(low) == pytest.approx(0.659, abs=1e-3)
    assert R0(low & ch[_IV]) / R0(low) == pytest.approx(0.330, abs=1e-3)
    assert R0(no_top & ch[_IV]) / R0(no_top) == pytest.approx(0.393, abs=1e-3)


@pytest.mark.requires_data
def test_bns_channel_one_is_the_electron_capture_route(bns_a_path):
    """Channel I BNS weight is carried by systems whose primary undergoes an ECSN
    (96 percent), every ECSN-primary BNS is in channel I, and that route switches
    on and off between neighbouring grid metallicities."""
    import h5py

    from grb_io import filter_systems, load_bns_with_channels
    from grb_physics import M_TOV, cold_stable_ns_mask

    d = load_bns_with_channels(bns_a_path, expected_model="A", expected_ns_max=2.5)
    with h5py.File(bns_a_path, "r") as f:
        dco = f["doubleCompactObjects"]
        sel = d["mask_merging"]
        np.testing.assert_array_equal(dco["weight"][()].ravel()[sel], d["weights"])
        d["ECSNPrimary"] = dco["ECSNPrimary"][()].ravel()[sel] == 1
    d = filter_systems(d, cold_stable_ns_mask(d["m1"], d["m2"], m_tov=M_TOV))
    ch = classify_formation_channels(**{k: d[k] for k in FORMATION_CHANNEL_INPUTS})
    w, Z, ecsn = d["weights"], d["metallicity"], np.asarray(d["ECSNPrimary"], bool)

    assert w[ch[_I] & ecsn].sum() / w[ch[_I]].sum() == pytest.approx(0.958, abs=1e-3)
    assert w[ecsn & ~ch[_I]].sum() == 0.0

    def ecsn_yield(z):
        return w[ecsn & np.isclose(Z, z, rtol=1e-3)].sum()

    # Off, on, off across neighbouring grid points 0.01018 -> 0.01137 and
    # 0.01971 -> 0.022.
    assert ecsn_yield(0.01137) > 20 * ecsn_yield(0.01018)
    assert ecsn_yield(0.01971) > 50 * ecsn_yield(0.022)
