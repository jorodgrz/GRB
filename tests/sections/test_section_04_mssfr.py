"""Section 4 of grb_main.ipynb: Cosmic Integration / MSSFR Grid Setup.

Smoke-level invariants on the project fiducial MSSFR (Levina+ 2026 TNG100-1)
delivered through ``compas_python_utils.cosmic_integration.FastCosmicIntegration``.

Reference: Levina et al. (2026), arXiv:2601.20202, Eq. (3-6) and Table 1.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.requires_compas
@pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 2.0])
def test_levina_dPdlogZ_normalises_at_redshift(z):
    """FCI ``find_metallicity_distribution`` integrates to ~1 across a wide
    Z window for the Levina+ 2026 TNG100-1 parameters."""
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        find_metallicity_distribution,
    )

    from grb_rates import (
        MSSFR_PARAMS_LEVINA26_TNG100,
        check_dPdlogZ_normalization,
    )

    redshifts = np.array([z])
    dPdlogZ, mets, _ = find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(1e-4),
        max_logZ_COMPAS=np.log(0.03),
        **MSSFR_PARAMS_LEVINA26_TNG100,
    )
    norm = check_dPdlogZ_normalization(dPdlogZ, mets, rtol=0.05)
    assert np.all(np.isfinite(norm))


# ─────────────────────────────────────────────────────────────────────
# TNG-resolution sweep (Section 4b): forward pass on BNS A
# ─────────────────────────────────────────────────────────────────────
def _build_tng_sweep_setup(bns_a_path):
    """Run cosmic integration once per TNG variant on the BNS A sample.
    Helper for the two slow tests below; module-scoped fixture is
    avoided so each test reads as a self-contained scenario."""
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
        find_metallicity_distribution,
        find_sfr,
    )

    from grb_io import METALLICITY_GRID, load_bns, read_expected_local_rate
    from grb_rates import (
        LEVINA26_TNG_VARIANTS,
        calibrate_mean_mass_evolved,
        compute_merger_rate,
    )

    data = load_bns(path=bns_a_path)
    Z = data["metallicity"]
    delays = data["delay_time"]
    w = data["weights"]

    redshifts, _, times, time_first_SF, _, _ = calculate_redshift_related_params(
        max_redshift=10.0,
        redshift_step=0.05,
        cosmology=Planck15,
    )
    expected_local_rate = read_expected_local_rate(bns_a_path)
    mean_mass = calibrate_mean_mass_evolved(
        redshifts,
        times,
        time_first_SF,
        Z,
        delays,
        w,
        expected_local_rate,
        Z_min_COMPAS=METALLICITY_GRID[0],
        Z_max_COMPAS=METALLICITY_GRID[-1],
    )

    R_per_tng = {}
    for name, (sfr_p, mssfr_p) in LEVINA26_TNG_VARIANTS.items():
        sfr_v = find_sfr(redshifts, **sfr_p)
        dPdlogZ_v, mets_v, p_draw_v = find_metallicity_distribution(
            redshifts,
            min_logZ_COMPAS=np.log(METALLICITY_GRID[0]),
            max_logZ_COMPAS=np.log(METALLICITY_GRID[-1]),
            **mssfr_p,
        )
        R_per_tng[name] = compute_merger_rate(
            redshifts,
            times,
            time_first_SF,
            sfr_v / mean_mass,
            p_draw_v,
            dPdlogZ_v,
            mets_v,
            Z,
            delays,
            w,
            smooth_sigma=0,
        )
    return redshifts, R_per_tng


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_tng_sweep_R_local_ordering_on_bns(bns_a_path):
    """R_BNS(z=0) under TNG50 / TNG100 / TNG300 inherits the
    resolution-monotonicity Levina+ 2026 reports for BBH (R_TNG50 >
    R_TNG100 > R_TNG300 under the analytical fit, Table 2).  The
    monotonicity comes from the SFR normalisation and the metallicity
    PDF skewness, not from the DCO type, so it must hold on the BNS
    sample too."""
    redshifts, R_per_tng = _build_tng_sweep_setup(bns_a_path)
    iz0 = int(np.argmin(np.abs(redshifts)))
    R0 = {name: float(R[iz0]) for name, R in R_per_tng.items()}
    assert R0["TNG50-1"] > R0["TNG100-1"] > R0["TNG300-1"], R0


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_tng100_R_z_peak_near_z_2p5(bns_a_path):
    """Levina+ 2026 Sec. 3.2 reports the BBH merger rate peaks at
    z ~ 2.5 under TNG100-1.  Our TNG100-1 forward pass on the BNS
    sample has the same peak driver (SFR x MSSFR convolution), so
    R(z) should peak in [2.0, 3.0]; verifies the cosmic integration
    is qualitatively reproducing Levina's headline shape."""
    redshifts, R_per_tng = _build_tng_sweep_setup(bns_a_path)
    R = R_per_tng["TNG100-1"]
    # Restrict to z >= 0.5 so the rising branch near z = 0 does not
    # spuriously absorb the argmax.
    mask = redshifts >= 0.5
    z_peak = float(redshifts[mask][int(np.argmax(R[mask]))])
    assert 2.0 <= z_peak <= 3.0, z_peak
