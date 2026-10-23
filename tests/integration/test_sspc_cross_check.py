"""Independent check of the cosmic-rate convolution against SSPC.

``grb_rates.compute_merger_rate`` wraps COMPAS FastCosmicIntegration.  This
test recomputes the Model A BNS and BHNS merger-rate densities with
SyntheticStellarPopConvolve 0.4 (Hendriks et al. 2023, MNRAS 526, 4130),
which does its own time convolution, birth-redshift mapping and unit handling.

The two calculations share only the population sample and the metallicity
cell convention.  The S(Z, z) input to SSPC is built here, straight from the
raw TNG100-1 file, without ``tng_sfr_and_dPdlogZ`` or
``dPdlogZ_binned_to_grid``, so those two helpers are tested too.  The cells
are the COMPAS ``ClassMSSFR`` ones (log-midpoints between grid points, outer
edges 1e-90 and 1), because that is the convention the ``w_000`` anchors fix.

Two SSPC conventions matter.  ``metallicity_distribution_array`` is multiplied
by the *linear* cell width inside SSPC, so it takes dP/dZ, not a cell
probability.  And SSPC 0.4's sequential mode passes the storage routine the
wrapped ``{"convolution_results": ...}`` dict where its multiprocessing path
unwraps it first (``convolve_populations.py``); multiprocessing does not start
on macOS, so the test unwraps exactly that shape.

Result (2026-10-16): SSPC/ours is 0.993 (BNS) and 0.992 (BHNS) at z = 0,
rising to 1.002 by z = 4, unchanged to 5e-4 when SSPC's star-formation bins
shrink from dz = 0.01 to 0.002.  The residual is FastCosmicIntegration's own
discretisation, the ~1 percent z = 0 bias documented in
``compute_merger_rate`` (0.988 at z = 0, 1.000 by z ~ 4).
"""

from __future__ import annotations

import copy
import logging
import os

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_data, pytest.mark.requires_compas, pytest.mark.slow]

sspc = pytest.importorskip("syntheticstellarpopconvolve")

Z_FIRST_SF = 10.0
DZ_SFR = 0.01
Z_EVAL = (0.0, 0.5, 1.0, 2.0, 4.0)
# FastCosmicIntegration's documented discretisation bias is ~1 percent at
# z = 0; anything beyond 1.5 percent is a disagreement worth chasing.
RTOL = 0.015


def _population(kind, bns_path, bhns_path):
    from grb_io import filter_systems, load_bhns_with_channels, load_bns_with_channels
    from grb_physics import M_TOV, cold_stable_bhns_mask, cold_stable_ns_mask

    if kind == "BNS":
        d = load_bns_with_channels(path=bns_path, expected_model="A")
        return filter_systems(d, cold_stable_ns_mask(d["m1"], d["m2"], m_tov=M_TOV))
    d = load_bhns_with_channels(path=bhns_path, expected_model="A")
    return filter_systems(d, cold_stable_bhns_mask(d["M_BH"], d["M_NS"], m_tov=M_TOV))


def _rate_ours(d, grid, M_SF):
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
    )

    from grb_rates import (
        compute_merger_rate,
        dPdlogZ_binned_to_grid,
        fine_metallicity_grid,
        tng_sfr_and_dPdlogZ,
    )

    z, _, times, t_first, _, _ = calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.01, z_first_SF=Z_FIRST_SF, cosmology=Planck15
    )
    fine, p = fine_metallicity_grid(grid[0], grid[-1])
    sfr, dP = tng_sfr_and_dPdlogZ(z, fine, box="TNG100-1")
    R = compute_merger_rate(
        z,
        times,
        t_first,
        sfr / M_SF,
        p,
        dPdlogZ_binned_to_grid(dP, fine, grid, p),
        fine,
        d["metallicity"],
        d["delay_time"],
        d["weights"],
        smooth_sigma=0,
    )
    return {zz: float(R[int(np.argmin(np.abs(z - zz)))]) for zz in Z_EVAL}


def _sspc_sfr_info(grid):
    """TNG100-1 SFR and per-cell dP/dZ on redshift bins, from the raw file."""
    import astropy.units as u
    import h5py

    from grb_io import TNG_BOX_SIZE_MPC_H, TNG_SFRD_PATHS
    from grb_physics import H0_PLANCK15

    with h5py.File(TNG_SFRD_PATHS["TNG100-1"], "r") as f:
        zt, Z_edges, S = f["Redshifts"][()], f["MetalBins"][()], f["Sfr"][()]
    order = np.argsort(zt)
    zt, S = zt[order], S[order] / (TNG_BOX_SIZE_MPC_H["TNG100-1"] / (H0_PLANCK15 / 100) / 1e3) ** 3
    cum = np.concatenate([np.zeros((S.shape[0], 1)), np.cumsum(S, axis=1)], axis=1)

    cell = np.concatenate([[1e-90], np.exp(0.5 * (np.log(grid[:-1]) + np.log(grid[1:]))), [1.0]])
    zedges = np.arange(0.0, Z_FIRST_SF + DZ_SFR / 2, DZ_SFR)
    zc = 0.5 * (zedges[1:] + zedges[:-1])
    cum_z = np.stack([np.interp(zc, zt, cum[:, j]) for j in range(cum.shape[1])], axis=1)
    # Star formation spread uniformly in ln Z inside each TNG bin; cells below
    # the first TNG edge (1e-10) get nothing, as in the file.
    C = np.stack([np.interp(np.log(cell), np.log(Z_edges), row, left=0.0) for row in cum_z])
    sfr = C[:, -1]
    P_cell = np.diff(C, axis=1) / sfr[:, None]
    return {
        "redshift_bin_edges": zedges,
        "starformation_rate_array": sfr * u.Msun / u.yr / u.Gpc**3,
        "metallicity_bin_edges": cell,
        "metallicity_distribution_array": P_cell / np.diff(cell)[None, :],
    }


def _rate_sspc(d, grid, M_SF, tmp_path):
    import astropy.units as u
    import h5py
    import pandas as pd
    from astropy.cosmology import Planck15
    from syntheticstellarpopconvolve import (
        convolve,
        default_convolution_config,
        default_convolution_instruction,
    )
    from syntheticstellarpopconvolve import convolve_populations as cp
    from syntheticstellarpopconvolve.general_functions import generate_boilerplate_outputfile

    out = str(tmp_path / "sspc.h5")
    generate_boilerplate_outputfile(out)
    # Mergers per Msun formed at the system's metallicity: each grid point
    # carries M_SF / N of the run's star formation.
    pd.DataFrame(
        {
            "delay_time": d["delay_time"],
            "metallicity": d["metallicity"],
            "normalized_yield": d["weights"] * grid.size / M_SF,
        }
    ).to_hdf(out, key="input_data/pop")

    cfg = copy.copy(default_convolution_config)
    cfg["logger"].setLevel(logging.CRITICAL)
    cfg["output_filename"] = out
    cfg["tmp_dir"] = str(tmp_path / "tmp")
    cfg["redshift_interpolator_data_output_filename"] = str(tmp_path / "interp.p")
    cfg["cosmology"] = Planck15
    cfg["multiprocessing"] = False
    cfg["time_type"] = "redshift"
    cfg["redshift_interpolator_min_redshift"] = 0.0
    cfg["redshift_interpolator_max_redshift"] = Z_FIRST_SF
    cfg["SFR_info"] = _sspc_sfr_info(grid)
    eps = 1e-4
    cfg["convolution_redshift_bin_edges"] = np.array(
        sorted({0.0} | {x for zz in Z_EVAL for x in (max(zz - eps, 0.0), zz + eps)})
    )
    cfg["convolution_instructions"] = [
        {
            **default_convolution_instruction,
            "input_data_name": "pop",
            "output_data_name": "pop",
            "convolution_direction": "backward",
            "convolution_type": "integrate",
            "data_column_dict": {
                "delay_time": {"column_name": "delay_time", "unit": u.Myr},
                "metallicity": "metallicity",
                "normalized_yield": {"column_name": "normalized_yield", "unit": 1 / u.Msun},
            },
        }
    ]

    store = cp.handle_storing_convolution_results

    def store_unwrapped(config, grp, bin_center, convolution_results):
        if isinstance(convolution_results, dict) and set(convolution_results) == {
            "convolution_results"
        }:
            convolution_results = convolution_results["convolution_results"]
        return store(
            config=config, grp=grp, bin_center=bin_center, convolution_results=convolution_results
        )

    cp.handle_storing_convolution_results = store_unwrapped
    try:
        convolve(cfg)
    finally:
        cp.handle_storing_convolution_results = store

    with h5py.File(out, "r") as f:
        g = f["output_data/pop/pop/convolution_results"]
        by_centre = {float(k.split()[0]): float(g[k]["yield"][()].sum()) for k in g}
    return {zz: by_centre[min(by_centre, key=lambda c: abs(c - max(zz, eps)))] for zz in Z_EVAL}


@pytest.mark.parametrize("kind", ["BNS", "BHNS"])
def test_sspc_reproduces_the_model_a_rate_history(kind, bns_a_path, bhns_a_path, tmp_path):
    from grb_io import METALLICITY_GRID, TNG_SFRD_PATHS
    from grb_rates import simulation_star_forming_mass

    if not os.path.exists(TNG_SFRD_PATHS["TNG100-1"]):
        pytest.skip("TNG100-1 S(Z,z) table not in data/")
    grid = METALLICITY_GRID[np.concatenate(([True], np.diff(METALLICITY_GRID) != 0))]
    M_SF = simulation_star_forming_mass(n_metallicities=grid.size)
    d = _population(kind, bns_a_path, bhns_a_path)

    ours = _rate_ours(d, grid, M_SF)
    theirs = _rate_sspc(d, grid, M_SF, tmp_path)
    ratio = {zz: theirs[zz] / ours[zz] for zz in Z_EVAL}
    assert all(abs(r - 1.0) < RTOL for r in ratio.values()), ratio
    # The direction of FastCosmicIntegration's known bias: high at z = 0 and
    # vanishing by z ~ 4, so the ratio rises with redshift.
    assert ratio[0.0] < 1.0 < ratio[4.0] + 1e-3, ratio
