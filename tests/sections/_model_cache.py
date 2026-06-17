"""Shared per-model load + calibration cache for Section 7b / 8c / 12 tests.

The cache mirrors the Section 12.0 setup cell of ``grb_main.ipynb`` end to
end (per-model load, Alsing remap with an independent RNG seed, per-model
``MEAN_MASS_EVOLVED`` calibration via the shared ``P_DRAW_BROEKGAARDEN21``)
and additionally retains the cosmology grid + Levina-TNG100-1 MSSFR
arrays needed by per-class follow-up rate computations.  The leading
underscore keeps pytest from discovering this file as a test module.

A single ``get_model(letter)`` call:

- loads the BNS + BHNS HDF5s,
- skips immediately via ``pytest.skip`` if either file or
  ``compas_python_utils`` is unavailable,
- applies the Alsing+ 2018 per-component remap,
- calibrates ``MEAN_MASS_EVOLVED`` per population, and
- runs ``compute_merger_rate`` once for the full BNS / BHNS samples.

The returned dict carries enough state for Sections 7b / 8c to layer
per-class subset rates on top without re-running the expensive load and
calibration steps.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pytest

MODEL_LETTERS = ["A", "F", "G", "J", "K"]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DATA_DIR = os.path.join(_REPO_ROOT, "Data")

_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def data_path(name: str) -> str:
    return os.path.join(_DATA_DIR, name)


def get_model(letter: str) -> Dict[str, Any]:
    """Load + calibrate Model ``letter`` and cache the result.

    Returns a dict with keys:

    - ``letter``            : Broekgaarden+ 2021 model letter
    - ``bns`` / ``bhns``    : the loader dicts (post Alsing remap)
    - ``bhns_ch``           : BHNS loaded with formation-channel columns
                              (post Alsing remap, same RNG seed as ``bhns``)
    - ``ns_max``            : embedded HDF5 attribute (2.0 / 2.5 / 3.0)
    - ``redshifts``, ``times``, ``time_first_SF`` : cosmology grid
    - ``sfr``               : Levina+ 2026 TNG100-1 SFR(z)
    - ``p_draw``, ``dPdlogZ``, ``metallicities`` : MSSFR arrays
    - ``mme_bns`` / ``mme_bhns`` : MEAN_MASS_EVOLVED per population
    - ``R_bns_z`` / ``R_bhns_z`` : full intrinsic R(z) arrays
    - ``R0_bns`` / ``R0_bhns``   : scalar values at z = 0
    - ``iz0``               : integer index of z = 0 in ``redshifts``

    Skips immediately if either the BNS or BHNS file for this letter
    is absent (partial-download support) or if ``compas_python_utils``
    is not importable.
    """
    if letter in _MODEL_CACHE:
        return _MODEL_CACHE[letter]

    fci = pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )

    bns_path = data_path(f"COMPASCompactOutput_BNS_{letter}.h5")
    bhns_path = data_path(f"COMPASCompactOutput_BHNS_{letter}.h5")
    for p in (bns_path, bhns_path):
        if not os.path.exists(p):
            pytest.skip(f"{os.path.basename(p)} not present in Data/")

    from astropy.cosmology import Planck15

    assert abs(Planck15.H0.value - 67.74) < 0.01

    from grb_io import (
        METALLICITY_GRID,
        load_bhns_with_channels,
        load_bhns_with_kicks,
        load_bns_with_channels,
        read_expected_local_rate,
    )
    from grb_physics import remap_ns_marginal, remap_ns_masses_double_gaussian
    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG100,
        calibrate_mean_mass_evolved,
        compute_merger_rate,
    )

    bns = load_bns_with_channels(path=bns_path, expected_model=letter)
    bhns = load_bhns_with_kicks(path=bhns_path, expected_model=letter)
    # Second BHNS load with formation-channel columns; needed by Section 8d
    # (per-class per-channel R(z)) and any other per-channel BHNS work.
    # Same physical sample as `bhns`, only the parsed columns differ.
    bhns_ch = load_bhns_with_channels(path=bhns_path, expected_model=letter)

    # Independent RNG seeds per population per model: matches Section 12.0
    # of grb_main.ipynb (`42 + i` for BNS, `43 + i` for BHNS).  bhns_ch
    # reuses the BHNS seed so both loader dicts carry the same remapped
    # M_NS draw and downstream rates are bit-identical.
    i = MODEL_LETTERS.index(letter)
    rng_bns = np.random.default_rng(42 + i)
    rng_bhns = np.random.default_rng(43 + i)
    bns["m1"], bns["m2"] = remap_ns_masses_double_gaussian(
        bns["m1"], bns["m2"], weights=bns["weights"], rng=rng_bns
    )
    bhns["M_NS"] = remap_ns_marginal(bhns["M_NS"], weights=bhns["weights"], rng=rng_bhns)
    bhns_ch["M_NS"] = remap_ns_marginal(
        bhns_ch["M_NS"], weights=bhns_ch["weights"], rng=np.random.default_rng(43 + i)
    )

    # bhns and bhns_ch must be the same underlying sample; the shared
    # columns are bit-identical because both loaders filter on
    # mergesInHubbleTimeFlag and stream the same HDF5.  This is a cheap
    # invariant on the cache contract.
    for _k in ("weights", "metallicity", "delay_time", "M_BH", "M_NS"):
        assert np.array_equal(bhns[_k], bhns_ch[_k]), (
            f"bhns and bhns_ch disagree on {_k!r}; _model_cache must load both from the same HDF5."
        )

    redshifts, _, times, time_first_SF, _, _ = fci.calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.01, cosmology=Planck15
    )
    sfr = fci.find_sfr(redshifts, **SFR_PARAMS_LEVINA26_TNG100)
    dPdlogZ, mets, p_draw = fci.find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=float(np.log(METALLICITY_GRID[0])),
        max_logZ_COMPAS=float(np.log(METALLICITY_GRID[-1])),
        **MSSFR_PARAMS_LEVINA26_TNG100,
    )
    p_draw = float(p_draw)

    mme_bns = calibrate_mean_mass_evolved(
        redshifts,
        times,
        time_first_SF,
        bns["metallicity"],
        bns["delay_time"],
        bns["weights"],
        read_expected_local_rate(bns_path),
        Z_min_COMPAS=METALLICITY_GRID[0],
        Z_max_COMPAS=METALLICITY_GRID[-1],
    )
    mme_bhns = calibrate_mean_mass_evolved(
        redshifts,
        times,
        time_first_SF,
        bhns["metallicity"],
        bhns["delay_time"],
        bhns["weights"],
        read_expected_local_rate(bhns_path),
        Z_min_COMPAS=METALLICITY_GRID[0],
        Z_max_COMPAS=METALLICITY_GRID[-1],
    )

    R_bns = compute_merger_rate(
        redshifts,
        times,
        time_first_SF,
        sfr / mme_bns,
        p_draw,
        dPdlogZ,
        mets,
        bns["metallicity"],
        bns["delay_time"],
        bns["weights"],
        smooth_sigma=0,
    )
    R_bhns = compute_merger_rate(
        redshifts,
        times,
        time_first_SF,
        sfr / mme_bhns,
        p_draw,
        dPdlogZ,
        mets,
        bhns["metallicity"],
        bhns["delay_time"],
        bhns["weights"],
        smooth_sigma=0,
    )
    iz0 = int(np.argmin(np.abs(redshifts)))
    out: Dict[str, Any] = {
        "letter": letter,
        "bns": bns,
        "bhns": bhns,
        "bhns_ch": bhns_ch,
        "ns_max": bns["ns_max"],
        "redshifts": redshifts,
        "times": times,
        "time_first_SF": time_first_SF,
        "sfr": sfr,
        "p_draw": p_draw,
        "dPdlogZ": dPdlogZ,
        "metallicities": mets,
        "mme_bns": mme_bns,
        "mme_bhns": mme_bhns,
        "R_bns_z": R_bns,
        "R_bhns_z": R_bhns,
        "R0_bns": float(R_bns[iz0]),
        "R0_bhns": float(R_bhns[iz0]),
        "iz0": iz0,
    }
    _MODEL_CACHE[letter] = out
    return out
