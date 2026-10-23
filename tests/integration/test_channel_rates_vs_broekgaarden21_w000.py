"""Per-channel rates reproduce Broekgaarden et al. (2021) on all 20 models, and
their printed Sec. 3.1 channel percentages on Model A.

The channel labels are pinned to Broekgaarden+ 2021's own code in
``test_broekgaarden21_channels_reproduced.py``.  This file checks the other
half: run on their configuration (Neijssel et al. 2019 MSSFR, WMAP9), our
cosmic integration must give each channel the local rate their pipeline
stored, which is the file's per-system ``weights_intrinsic/w_000`` summed over
that channel.  Those weights include the failed common-envelope ejections
their Sec. 2 says were removed, so both comparisons load the raw file
(``drop_failed_ce=False``).  The Sec. 3.1 percentages cannot tell the two
samples apart: the ejections are 1.2 percent of the BHNS weight and both round
to the printed values.

The comparison runs at ``redshift_step = 0.001``.  At the 0.01 the science
figures use, FastCosmicIntegration zeroes the z = 0 rate of every system whose
delay lies within one redshift step (~140 Myr) of the longest delay that can
merge today, so the oldest systems drop out and the channels they sit in come
out low (alpha10 BHNS: one channel at 0.860).  At 0.001 that edge is gone.

Measured 2026-10-22 over all 40 files, channels above 1 percent of a model's
rate: totals 0.966 (F BHNS) to 0.997 (E BNS); single channels 0.952 (alpha10
BHNS) to 1.028 (alpha0p1 BHNS, channel II); channel shares within 0.47
percentage points.  The residual offset is largely common to all channels of
a model, larger for BHNS than BNS, and its source is not identified.  The
tolerances below sit just outside those extremes and are CODE HEURISTIC
regression bounds, not literature values.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from grb_classify import FORMATION_CHANNEL_INPUTS, classify_formation_channels
from grb_io import (
    ALL_MODEL_SUFFIXES,
    METALLICITY_GRID,
    load_bhns_with_channels,
    load_bns_with_channels,
)

_REDSHIFT_STEP = 0.001
_TOTAL_RATIO = (0.96, 1.01)
_CHANNEL_RATIO = (0.94, 1.04)
_SHARE_PP = 0.75
_MIN_SHARE = 0.01

_FILES = [
    f"COMPASCompactOutput_{kind}_{model}.h5"
    for kind in ("BNS", "BHNS")
    for model in ALL_MODEL_SUFFIXES
]


@pytest.fixture(scope="module")
def neijssel_wmap9():
    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed",
    )
    from astropy.cosmology import WMAP9

    from grb_rates import dPdlogZ_binned_to_grid, simulation_star_forming_mass

    grid = METALLICITY_GRID[np.concatenate(([True], np.diff(METALLICITY_GRID) != 0))]
    redshifts, _, times, time_first_SF, _, _ = fci.calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=_REDSHIFT_STEP, cosmology=WMAP9
    )
    sfr = fci.find_sfr(redshifts)  # Neijssel+ 2019 defaults, the w_000 MSSFR
    dPdlogZ, metallicities, p_draw = fci.find_metallicity_distribution(
        redshifts, min_logZ_COMPAS=np.log(grid[0]), max_logZ_COMPAS=np.log(grid[-1])
    )
    dPdlogZ = dPdlogZ_binned_to_grid(dPdlogZ, metallicities, grid, p_draw)
    return dict(
        redshifts=redshifts,
        times=times,
        time_first_SF=time_first_SF,
        n_formed=sfr / simulation_star_forming_mass(n_metallicities=grid.size),
        p_draw=p_draw,
        dPdlogZ=dPdlogZ,
        metallicities=metallicities,
    )


def _w000_per_system(path, mask_merging):
    """The file's z = 0 rate weight for each merging DCO row, joined by seed."""
    with h5py.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        seed = dco["seed" if "seed" in dco else "m_randomSeed"][...].squeeze()[mask_merging]
        w_seed = f["weights_intrinsic"]["SEED"][...].squeeze()
        w = f["weights_intrinsic"]["w_000"][...].squeeze()
    order = np.argsort(w_seed)
    pos = np.clip(np.searchsorted(w_seed[order], seed), 0, w_seed.size - 1)
    assert np.array_equal(w_seed[order][pos], seed), "w_000 is missing merging seeds"
    return w[order][pos]


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.parametrize("compas_file", _FILES, indirect=True)
def test_channel_local_rates_match_w000(compas_file, neijssel_wmap9):
    from grb_rates import per_system_rate_weights

    loader = load_bns_with_channels if "_BNS_" in compas_file else load_bhns_with_channels
    # The stored weights count the failed-CE ejections the loaders drop by default.
    d = loader(compas_file, drop_failed_ce=False)
    channels = classify_formation_channels(**{k: d[k] for k in FORMATION_CHANNEL_INPUTS})

    theirs = _w000_per_system(compas_file, d["mask_merging"])
    ours = per_system_rate_weights(
        0.0,
        COMPAS_Z=d["metallicity"],
        COMPAS_delay_times=d["delay_time"],
        COMPAS_weights=d["weights"],
        **neijssel_wmap9,
    )

    names = list(channels)
    R_theirs = np.array([theirs[m].sum() for m in channels.values()])
    R_ours = np.array([ours[m].sum() for m in channels.values()])
    share_theirs = R_theirs / R_theirs.sum()
    share_ours = R_ours / R_ours.sum()

    total = R_ours.sum() / R_theirs.sum()
    assert _TOTAL_RATIO[0] <= total <= _TOTAL_RATIO[1], f"total ratio {total:.4f}"

    for i in np.where(share_theirs > _MIN_SHARE)[0]:
        r = R_ours[i] / R_theirs[i]
        assert _CHANNEL_RATIO[0] <= r <= _CHANNEL_RATIO[1], f"{names[i]}: ratio {r:.4f}"

    worst = np.abs(share_ours - share_theirs).max() * 100.0
    assert worst <= _SHARE_PP, f"channel shares differ by up to {worst:.2f} pp"


# Broekgaarden et al. (2021) Sec. 3.1 headers: the share of detected Model A
# BHNS mergers (SFRD model 000, LVK at design sensitivity) per channel, as
# printed.  IV is printed as "< 1".
_B21_SEC31_DETECTED_PERCENT = {
    "I  Stable MT + CE": 86,
    "II  Stable MT only": 4,
    "III Single-core CE": 4,
    "V   Other": 6,
}


@pytest.mark.requires_data
def test_model_a_bhns_detected_channel_shares_match_broekgaarden21_sec31(bhns_a_path):
    """Our labels, weighted by the file's ``weights_detected/w_000``, round to
    the percentages Broekgaarden+ 2021 print in their Sec. 3.1 headers.

    That column is their detection-weighted rate: it sums to 10.83 yr^-1, the
    "R_det ~ 11 yr^-1 for an LVK network at design sensitivity" of their
    Sec. 4.1.2, and is checked here too.  Measured 2026-10-22: 85.54, 4.24,
    4.30, 0.33, 5.60 percent.  No cold-stability cut, since they apply none.
    """
    d = load_bhns_with_channels(bhns_a_path, expected_model="A", drop_failed_ce=False)
    channels = classify_formation_channels(**{k: d[k] for k in FORMATION_CHANNEL_INPUTS})
    with h5py.File(bhns_a_path, "r") as f:
        seed = f["doubleCompactObjects"]["seed"][...].squeeze()[d["mask_merging"]]
        w_seed = f["weights_detected"]["SEED"][...].squeeze()
        w_det = f["weights_detected"]["w_000"][...].squeeze()
    order = np.argsort(w_seed)
    pos = np.searchsorted(w_seed[order], seed)
    assert np.array_equal(w_seed[order][pos], seed), "w_000 is missing merging seeds"
    w = w_det[order][pos]

    assert round(float(w.sum())) == 11, f"detected rate {w.sum():.2f} yr^-1, paper ~11"
    share = {k: 100.0 * w[m].sum() / w.sum() for k, m in channels.items()}
    for k, printed in _B21_SEC31_DETECTED_PERCENT.items():
        assert round(share[k]) == printed, f"{k}: {share[k]:.2f} percent, paper {printed}"
    assert share["IV  Double-core CE"] < 1.0, f"IV: {share['IV  Double-core CE']:.2f} percent"
