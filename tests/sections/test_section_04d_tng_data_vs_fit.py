"""Section 4d of grb_main.ipynb: Model A rates over the TNG S(Z, z) tables
against the Levina+ 2026 Table 1 fits.

Pins the TNG100-1 BNS local rates the caption quotes and the fact that the
fiducial (the table) sits below the fit.  BNS only: the BHNS integration
takes about 30 s per call.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_data, pytest.mark.requires_compas, pytest.mark.slow]


@pytest.fixture(scope="module")
def rates_tng100(bns_a_path):
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
        find_metallicity_distribution,
        find_sfr,
    )

    from grb_io import METALLICITY_GRID, filter_systems, load_bns_with_channels
    from grb_physics import M_TOV, cold_stable_ns_mask
    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        SFR_PARAMS_LEVINA26_TNG100,
        compute_merger_rate,
        dPdlogZ_binned_to_grid,
        simulation_star_forming_mass,
        tng_sfr_and_dPdlogZ,
    )

    b = load_bns_with_channels(path=bns_a_path, expected_model="A")
    b = filter_systems(b, cold_stable_ns_mask(b["m1"], b["m2"], m_tov=M_TOV))
    z, _, times, t_first, _, _ = calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.01, cosmology=Planck15
    )
    grid = METALLICITY_GRID[np.concatenate(([True], np.diff(METALLICITY_GRID) != 0))]
    M_SF = simulation_star_forming_mass(n_metallicities=grid.size)
    dP_fit, fine, p = find_metallicity_distribution(
        z,
        min_logZ_COMPAS=np.log(grid[0]),
        max_logZ_COMPAS=np.log(grid[-1]),
        **MSSFR_PARAMS_LEVINA26_TNG100,
    )
    sfr_dat, dP_dat = tng_sfr_and_dPdlogZ(z, fine, box="TNG100-1")
    out = {}
    for src, sfr, dP in (
        ("fit", find_sfr(z, **SFR_PARAMS_LEVINA26_TNG100), dP_fit),
        ("data", sfr_dat, dP_dat),
    ):
        out[src] = compute_merger_rate(
            z,
            times,
            t_first,
            sfr / M_SF,
            p,
            dPdlogZ_binned_to_grid(dP, fine, grid, p),
            fine,
            b["metallicity"],
            b["delay_time"],
            b["weights"],
            smooth_sigma=0,
        )
    return z, out


def test_tng100_local_bns_rates_match_the_caption(rates_tng100):
    _, R = rates_tng100
    assert R["fit"][0] == pytest.approx(46.91, abs=0.02)
    assert R["data"][0] == pytest.approx(38.34, abs=0.02)
    assert R["data"][0] / R["fit"][0] == pytest.approx(0.817, abs=0.002)


def test_table_and_fit_agree_away_from_z0(rates_tng100):
    z, R = rates_tng100
    for zz in (1.0, 2.0):
        i = int(np.argmin(np.abs(z - zz)))
        assert abs(R["data"][i] / R["fit"][i] - 1.0) < 0.06
