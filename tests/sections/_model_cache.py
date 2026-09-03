"""Shared per-model load + cosmic-integration cache for the Section 6 / 6b / 7 / 7b tests.

The cache reproduces the science path of ``grb_main.ipynb`` end to end, so a
test built on it is testing what the notebook actually computes:

- raw COMPAS gravitational masses, with no quantile remap;
- the cold-stability cut ``m_i <= M_TOV`` applied before any classifier sees
  the sample (Section 0), which is what makes the class fractions fractions of
  a valid partition;
- the run-configuration star-forming mass ``simulation_star_forming_mass()``,
  one number for both populations and every model, rather than a per-population
  ``calibrate_mean_mass_evolved`` fit;
- the COMPAS ``ClassMSSFR`` metallicity binning via ``dPdlogZ_binned_to_grid``,
  so star formation above the ``Z = 0.03`` grid ceiling is absorbed into the
  top grid point instead of being dropped.

The last two are not independent.  Calibrating the mass against each file's
``w_000`` anchor while feeding ``compute_merger_rate`` a pointwise ``dPdlogZ``
cancels one error against the other: the all-BNS total lands near the anchor
by construction while every per-class and per-channel split is wrong, because
the missing metallicity term is class-dependent.  Both have to move together.

It additionally retains the cosmology grid and the Levina TNG100-1 MSSFR
arrays needed by per-class follow-up rate computations.  The leading
underscore keeps pytest from discovering this file as a test module.

A single ``get_model(letter)`` call:

- loads the BNS + BHNS HDF5s,
- skips immediately via ``pytest.skip`` if either file or
  ``compas_python_utils`` is unavailable,
- applies the cold-stability cut to both populations,
- runs ``compute_merger_rate`` once for the full BNS / BHNS samples.

The returned dict carries enough state for the per-channel tests to layer
per-class subset rates on top without re-running the expensive load and
integration steps.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
# Lowercase, matching the repository layout.  A capitalised "Data" resolves on
# a case-insensitive macOS filesystem and silently misses on a Linux runner,
# where every test built on this cache would skip rather than fail.
_DATA_DIR = os.path.join(_REPO_ROOT, "data")

_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def data_path(name: str) -> str:
    return os.path.join(_DATA_DIR, name)


def get_model(letter: str) -> Dict[str, Any]:
    """Load Model ``letter`` on the notebook science path and cache the result.

    Returns a dict with keys:

    - ``letter``            : Broekgaarden+ 2021 model letter
    - ``bns`` / ``bhns``    : loader dicts, raw masses, cold-stability cut applied
    - ``bhns_ch``           : BHNS loaded with formation-channel columns, same cut
    - ``ns_max``            : embedded HDF5 attribute (2.0 / 2.5 / 3.0)
    - ``redshifts``, ``times``, ``time_first_SF`` : cosmology grid
    - ``sfr``               : Levina+ 2026 TNG100-1 SFR(z)
    - ``p_draw``, ``dPdlogZ``, ``metallicities`` : MSSFR arrays, ClassMSSFR-binned
    - ``M_SF``              : run-configuration star-forming mass, shared
    - ``mme_bns`` / ``mme_bhns`` : aliases of ``M_SF``, kept so callers that
                              divide ``sfr`` by a per-population mass still read
                              correctly; both are the same number by design
    - ``w000_bns`` / ``w000_bhns`` : each file's published local-rate anchor,
                              a reference column, never a normalisation
    - ``dropped_bns`` / ``dropped_bhns`` : weight fraction the cut discards
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
            pytest.skip(f"{os.path.basename(p)} not present in data/")

    from astropy.cosmology import Planck15

    assert abs(Planck15.H0.value - 67.74) < 0.01

    from grb_io import (
        METALLICITY_GRID,
        filter_systems,
        load_bhns_with_channels,
        load_bhns_with_kicks,
        load_bns_with_channels,
        read_expected_local_rate,
    )
    from grb_physics import M_TOV, cold_stable_ns_mask
    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG100,
        compute_merger_rate,
        dPdlogZ_binned_to_grid,
        simulation_star_forming_mass,
    )

    bns = load_bns_with_channels(path=bns_path, expected_model=letter)
    bhns = load_bhns_with_kicks(path=bhns_path, expected_model=letter)
    # Second BHNS load with formation-channel columns; needed by Section 7b
    # (per-class per-channel R(z)) and any other per-channel BHNS work.
    # Same physical sample as `bhns`, only the parsed columns differ.
    bhns_ch = load_bhns_with_channels(path=bhns_path, expected_model=letter)

    # Anchors are read before the cut: read_expected_local_rate sums w_000 over
    # every row in the file, so it covers the uncut population.  They are
    # reference values here, never a normalisation.
    w000_bns = read_expected_local_rate(bns_path)
    w000_bhns = read_expected_local_rate(bhns_path)

    # Cold-stability cut, Section 0 of the notebook.  COMPAS types a remnant as
    # an NS up to its own run-time M_NS_max (2.5 for Model A), which is above
    # the cold-stability limit M_TOV = 2.2 (Raaijmakers+ 2021).  classify_bns_2024
    # tests only M_tot and q, never the components, so a super-M_TOV "NS" left in
    # the sample would be classified as a BNS.  The cut runs before any
    # classifier.  Nothing here draws a random number.
    w_full_bns = float(bns["weights"].sum())
    w_full_bhns = float(bhns["weights"].sum())
    bns = filter_systems(bns, cold_stable_ns_mask(bns["m1"], bns["m2"], m_tov=M_TOV))
    keep_bhns = cold_stable_ns_mask(bhns["M_NS"], m_tov=M_TOV)
    bhns = filter_systems(bhns, keep_bhns)
    bhns_ch = filter_systems(bhns_ch, cold_stable_ns_mask(bhns_ch["M_NS"], m_tov=M_TOV))
    dropped_bns = 1.0 - bns["weights"].sum() / w_full_bns
    dropped_bhns = 1.0 - bhns["weights"].sum() / w_full_bhns

    # bhns and bhns_ch must be the same underlying sample; the shared
    # columns are bit-identical because both loaders filter on
    # mergesInHubbleTimeFlag and stream the same HDF5, and the cut above is a
    # deterministic function of M_NS.  This is a cheap invariant on the cache
    # contract.
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

    # COMPAS ClassMSSFR bin convention: the top and bottom simulated
    # metallicities absorb the star formation off the ends of the grid.  Without
    # this, 28.5 percent of z = 0 star formation above the Z = 0.03 ceiling is
    # assigned to nothing at all, and the shortfall is class-dependent.
    Z_grid_53 = METALLICITY_GRID[np.concatenate(([True], np.diff(METALLICITY_GRID) != 0))]
    dPdlogZ = dPdlogZ_binned_to_grid(dPdlogZ, mets, Z_grid_53, p_draw)

    # One star-forming mass: 53 grid points x about 1e6 binaries x 79.11 Msun of
    # Universe star formation per simulated binary.  A property of the sampler,
    # so it is the same for both populations and every model of the grid, and
    # nothing is fitted.
    M_SF = simulation_star_forming_mass(n_metallicities=Z_grid_53.size)

    R_bns = compute_merger_rate(
        redshifts,
        times,
        time_first_SF,
        sfr / M_SF,
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
        sfr / M_SF,
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
        "M_SF": M_SF,
        "mme_bns": M_SF,
        "mme_bhns": M_SF,
        "w000_bns": w000_bns,
        "w000_bhns": w000_bhns,
        "dropped_bns": dropped_bns,
        "dropped_bhns": dropped_bhns,
        "R_bns_z": R_bns,
        "R_bhns_z": R_bhns,
        "R0_bns": float(R_bns[iz0]),
        "R0_bhns": float(R_bhns[iz0]),
        "iz0": iz0,
    }
    _MODEL_CACHE[letter] = out
    return out
