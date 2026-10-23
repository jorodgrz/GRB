"""TNG S(Z, z) tables as the cosmic-integration input (``load_tng_sfrd``,
``tng_sfr_and_dPdlogZ``).

The synthetic-file tests run everywhere; the ``requires_data`` ones read the
three IllustrisTNG tables shipped with the Levina+ 2026 repository.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest


def _fine_grid():
    # find_metallicity_distribution's default fine grid: ln Z in [-12, 0], step 0.01.
    return np.exp(np.arange(-12.0, 0.0 + 0.01, 0.01))


def _write_table(path, sfr_box, z_desc, Z_edges):
    with h5py.File(path, "w") as f:
        f["Redshifts"] = z_desc
        f["MetalBins"] = Z_edges
        f["Sfr"] = sfr_box
        f["Metals"] = np.zeros_like(sfr_box)
        f["Lookbacktimes"] = np.zeros_like(z_desc)


@pytest.fixture
def synthetic_table(tmp_path):
    """Two snapshots (z = 2 and 0, stored descending as TNG does), three bins.

    The interior edges sit on fine-grid cell boundaries (ln Z = -7, -4.6) so
    the per-bin shares can be checked exactly.
    """
    Z_edges = np.array([1e-10, np.exp(-7.0), np.exp(-4.6), 1.0])
    z_desc = np.array([2.0, 0.0])
    sfr_box = np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    path = tmp_path / "SFRMetallicityFromGasWithMetalsTNG100-1.hdf5"
    _write_table(path, sfr_box, z_desc, Z_edges)
    return str(path), sfr_box, Z_edges


def test_load_tng_sfrd_divides_by_the_comoving_box_volume(synthetic_table):
    from grb_io import TNG_BOX_SIZE_MPC_H, load_tng_sfrd
    from grb_physics import H0_PLANCK15

    path, sfr_box, _ = synthetic_table
    d = load_tng_sfrd(box="TNG100-1", path=path)
    assert np.array_equal(d["z"], [0.0, 2.0])
    V_Gpc3 = (TNG_BOX_SIZE_MPC_H["TNG100-1"] / (H0_PLANCK15 / 100.0) / 1e3) ** 3
    assert np.allclose(d["sfrd"], sfr_box[::-1] / V_Gpc3, rtol=1e-12)


def test_load_tng_sfrd_rejects_an_unknown_box(synthetic_table):
    from grb_io import load_tng_sfrd

    with pytest.raises(ValueError, match="box must be one of"):
        load_tng_sfrd(box="TNG25", path=synthetic_table[0])


def test_tng_dPdlogZ_is_normalised_and_conserves_each_bin(synthetic_table):
    from grb_io import load_tng_sfrd
    from grb_rates import check_dPdlogZ_normalization, tng_sfr_and_dPdlogZ

    path, _, Z_edges = synthetic_table
    fine = _fine_grid()
    z = np.array([0.0, 1.0, 2.0])
    sfr, dP = tng_sfr_and_dPdlogZ(z, fine, box="TNG100-1", path=path)
    check_dPdlogZ_normalization(dP, fine, rtol=1e-10)

    per_bin = load_tng_sfrd(box="TNG100-1", path=path)["sfrd"]
    # Linear in z between snapshots, so z = 1 is the mean of the two.
    assert np.allclose(sfr, [per_bin[0].sum(), per_bin.sum() / 2, per_bin[1].sum()])

    # The share of star formation in the top bin, at each redshift.
    ln = np.log(fine)
    above = (dP[:, ln >= np.log(Z_edges[2]) - 1e-9] * 0.01).sum(axis=1)
    expected_top = np.array([1.0, 0.5, 0.0]) / np.array([2.0, 3.0, 4.0])
    assert np.allclose(above, expected_top, atol=1e-9)
    # The bottom bin reaches 1e-10, far below the first fine cell at e^-12;
    # the part below is folded into that cell, so the whole bin survives.
    below = (dP[:, ln < np.log(Z_edges[1]) - 1e-9] * 0.01).sum(axis=1)
    assert np.allclose(below, np.array([0.0, 1.5, 3.0]) / np.array([2.0, 3.0, 4.0]), atol=1e-9)


def test_tng_sfr_and_dPdlogZ_refuses_to_extrapolate_in_redshift(synthetic_table):
    from grb_rates import tng_sfr_and_dPdlogZ

    with pytest.raises(ValueError, match="outside"):
        tng_sfr_and_dPdlogZ(np.array([0.0, 3.0]), _fine_grid(), path=synthetic_table[0])


def _tng_files_present():
    from grb_io import TNG_SFRD_PATHS

    return all(os.path.exists(p) for p in TNG_SFRD_PATHS.values())


@pytest.mark.requires_data
@pytest.mark.skipif(not _tng_files_present(), reason="TNG S(Z,z) tables not in data/")
@pytest.mark.parametrize(
    "box, sfr0, above_003",
    # z = 0 SFRD in Msun/yr/Mpc^3 and the share of z = 0 star formation above
    # Z = 0.03, both from the TNG tables (computed 2026-10-16).
    [("TNG50-1", 0.02490, 0.240), ("TNG100-1", 0.01279, 0.221), ("TNG300-1", 0.00618, 0.133)],
)
def test_tng_tables_z0_values(box, sfr0, above_003):
    from grb_rates import check_dPdlogZ_normalization, tng_sfr_and_dPdlogZ

    fine = _fine_grid()
    z = np.arange(0.0, 10.0 + 1e-9, 0.01)
    sfr, dP = tng_sfr_and_dPdlogZ(z, fine, box=box)
    check_dPdlogZ_normalization(dP, fine, rtol=1e-10)
    assert sfr[0] / 1e9 == pytest.approx(sfr0, rel=2e-3)
    ln = np.log(fine)
    assert dP[0, ln >= np.log(0.03)].sum() * 0.01 == pytest.approx(above_003, abs=2e-3)


@pytest.mark.requires_data
@pytest.mark.skipif(not _tng_files_present(), reason="TNG S(Z,z) tables not in data/")
def test_levina_fit_overstates_tng100_star_formation_at_z0():
    """The Table 1 fit puts 34 percent more star formation at z = 0 than TNG100-1."""
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import find_sfr

    from grb_rates import SFR_PARAMS_LEVINA26_TNG100, tng_sfr_callable

    z = np.array([0.0])
    ratio = find_sfr(z, **SFR_PARAMS_LEVINA26_TNG100)[0] / tng_sfr_callable("TNG100-1")(0.0)
    assert ratio == pytest.approx(1.344, abs=0.01)


def test_cosmic_sf_fraction_rejects_half_a_fit():
    from grb_rates import MSSFR_PARAMS_LEVINA26_TNG100, cosmic_sf_mass_fraction_per_grid_point

    grid = np.geomspace(1e-4, 0.03, 5)
    with pytest.raises(ValueError, match="both"):
        cosmic_sf_mass_fraction_per_grid_point(
            grid, (grid[0], grid[-1]), mssfr_params=MSSFR_PARAMS_LEVINA26_TNG100
        )
    with pytest.raises(ValueError, match="not both"):
        cosmic_sf_mass_fraction_per_grid_point(
            grid, (grid[0], grid[-1]), {"a": 1}, {"mu0": 1}, tng_box="TNG100-1"
        )


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.skipif(not _tng_files_present(), reason="TNG S(Z,z) tables not in data/")
def test_cosmic_sf_fraction_defaults_to_the_tng100_table():
    from grb_rates import cosmic_sf_mass_fraction_per_grid_point

    grid = np.geomspace(1e-4, 0.03, 53)
    F_default = cosmic_sf_mass_fraction_per_grid_point(grid, (grid[0], grid[-1]))
    F_tng = cosmic_sf_mass_fraction_per_grid_point(grid, (grid[0], grid[-1]), tng_box="TNG100-1")
    assert np.allclose(F_default, F_tng)
    assert F_default.sum() == pytest.approx(1.0)
