"""Section 4e of grb_main.ipynb: the TNG100-1 S(Z, z) map against the Levina+
2026 fit.  Every number the caption quotes is recomputed here, from the table
and from Levina+ 2026 Eqs. 2-6 typed in directly, plus a replica of the
Levina plotting code that the side panel's convention follows."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_data, pytest.mark.requires_compas]

Z_MAX_4E = 14.05  # last point of Levina's np.arange(0, 14.1, 0.05)


def _levina_fit(z, Z):
    """S(z) dP/dlnZ from Levina+ 2026 Eqs. 2-6, TNG100-1 column of Table 1."""
    from scipy.stats import norm

    mu0, muz, w0, wz, al = 0.0247, -0.0521, 1.1509, 0.0477, -1.8801
    a, b, c, d = 0.0172, 1.4425, 4.5299, 6.2261
    w = (w0 * 10 ** (wz * z))[:, None]
    xi = np.log(mu0 * 10 ** (muz * z)[:, None] / (2 * norm.cdf(al * w / np.sqrt(1 + al**2))))
    x = (np.log(Z)[None, :] - (xi - w**2 / 2)) / w
    p = 2 / w * norm.pdf(x) * norm.cdf(al * x)
    p /= (p.sum(axis=1) * np.diff(np.log(Z))[0])[:, None]
    return (a * (1 + z) ** b / (1 + ((1 + z) / c) ** d))[:, None] * p


@pytest.fixture(scope="module")
def table():
    import astropy.units as u
    from astropy.cosmology import Planck15

    from grb_io import TNG_SFRD_PATHS, load_tng_sfrd

    if not os.path.exists(TNG_SFRD_PATHS["TNG100-1"]):
        pytest.skip("TNG100-1 S(Z,z) table not in data/")
    d = load_tng_sfrd("TNG100-1")
    d["t"] = Planck15.lookback_time(d["z"]).to(u.Gyr).value
    d["per_bin"] = d["sfrd"] / 1e9  # Msun / yr / Mpc^3 per bin
    d["dlnZ"] = np.diff(np.log(d["Z_edges"]))[0]
    return d


@pytest.fixture(scope="module")
def fit():
    import astropy.units as u
    from astropy.cosmology import Planck15

    from grb_physics import Z_SUN

    z = np.linspace(0.0, Z_MAX_4E, 1406)
    Z = np.exp(np.arange(np.log(1e-5 * Z_SUN), 0.0 + 1e-9, 0.01))
    return {"z": z, "Z": Z, "S": _levina_fit(z, Z), "t": Planck15.lookback_time(z).to(u.Gyr).value}


def test_star_formation_history(table, fit):
    S = table["per_bin"].sum(axis=1)
    i = int(np.argmax(S))
    assert np.round(table["z"][i - 1 : i + 2], 2).tolist() == [2.58, 2.73, 2.90]
    assert S[i] == pytest.approx(0.0875, abs=5e-5)
    assert S[0] == pytest.approx(0.0128, abs=5e-5)
    Sfit = fit["S"].sum(axis=1) * 0.01
    assert fit["z"][np.argmax(Sfit)] == pytest.approx(2.74, abs=0.01)
    assert Sfit.max() == pytest.approx(0.0885, abs=5e-5)
    assert Sfit[0] == pytest.approx(0.0172, abs=5e-5)


def _table_on_fit_grid(table, fit):
    dS = table["per_bin"] / table["dlnZ"]
    return np.column_stack([np.interp(fit["z"], table["z"], dS[:, j]) for j in range(dS.shape[1])])


def _quantiles(table, fit, tab, fz, q=(0.16, 0.5, 0.84)):
    from grb_physics import Z_SUN

    cum = np.concatenate([[0.0], np.cumsum(tab)])
    qt = np.exp(np.interp(q, cum / cum[-1], np.log(table["Z_edges"]))) / Z_SUN
    cf = np.cumsum(fz)
    return qt, np.interp(q, cf / cf[-1], fit["Z"]) / Z_SUN


def test_time_integrated_side_panel(table, fit):
    """The drawn marginal: stellar mass formed per ln Z, its medians and the
    table-to-fit bin ratios the caption quotes for the 2 to 5 Zsun bump."""
    from grb_physics import Z_SUN

    tab = np.trapezoid(_table_on_fit_grid(table, fit), fit["t"] * 1e9, axis=0)
    fz = np.trapezoid(fit["S"], fit["t"] * 1e9, axis=0)
    assert tab.max() / 1e8 == pytest.approx(2.27, abs=0.01)
    assert fz.max() / 1e8 == pytest.approx(2.31, abs=0.01)
    assert np.sort(tab)[-2] / tab.max() == pytest.approx(0.97, abs=0.005)

    qt, qf = _quantiles(table, fit, tab, fz)
    assert np.round(qt, 2).tolist() == pytest.approx([0.34, 0.96, 2.42], abs=0.006)
    assert np.round(qf, 2).tolist() == pytest.approx([0.35, 1.03, 2.48], abs=0.006)

    edges = table["Z_edges"]
    cf = np.concatenate([[0.0], np.cumsum(fz)]) * 0.01
    lnZf = np.concatenate([[np.log(fit["Z"][0])], np.log(fit["Z"]) + 0.01])
    fit_bins = np.diff(np.interp(np.log(edges), lnZf, cf)) / table["dlnZ"]
    lo = {round(e / Z_SUN, 2): i for i, e in enumerate(edges[:-1])}
    ratios = [tab[lo[k]] / fit_bins[lo[k]] for k in (1.52, 2.23, 4.80, 7.04)]
    assert np.round(ratios, 2).tolist() == pytest.approx([0.85, 1.08, 0.38, 0.14], abs=0.011)


def test_redshift_summed_side_panel(table, fit):
    """Levina's plotting convention, quoted in the caption for comparison."""
    from grb_physics import Z_SUN

    tab = np.trapezoid(_table_on_fit_grid(table, fit), fit["z"], axis=0)
    i = int(np.argmax(tab))
    edges = table["Z_edges"] / Z_SUN
    assert (round(edges[i], 2), round(edges[i + 1], 2)) == (0.70, 1.03)
    assert np.sort(tab)[-2] / tab.max() == pytest.approx(0.923, abs=0.002)
    assert tab.max() == pytest.approx(0.1286, abs=2e-4)
    fz = np.trapezoid(fit["S"], fit["z"], axis=0)
    assert fit["Z"][np.argmax(fz)] / Z_SUN == pytest.approx(0.99, abs=0.01)
    assert fz.max() == pytest.approx(0.1232, abs=2e-4)
    # Levina's axis: the same quantity summed over dz = 0.05.
    assert tab.max() / 0.05 == pytest.approx(2.57, abs=0.01)
    assert fz.max() / 0.05 == pytest.approx(2.46, abs=0.01)
    qt, qf = _quantiles(table, fit, tab, fz, q=(0.5,))
    assert qt[0] == pytest.approx(0.67, abs=0.005)
    assert qf[0] == pytest.approx(0.73, abs=0.005)


def test_mass_shares_quoted_in_the_caption(table, fit):
    from grb_io import METALLICITY_GRID
    from grb_physics import Z_SUN

    o = np.argsort(table["t"])
    w = np.trapezoid(table["per_bin"][o], table["t"][o], axis=0)
    cum = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
    lnZ = np.log(table["Z_edges"])
    assert 1 - np.interp(np.log(6 * Z_SUN), lnZ, cum) == pytest.approx(5.06e-3, rel=0.01)
    assert np.interp(np.log(METALLICITY_GRID[-1]), lnZ, cum) == pytest.approx(0.7971, abs=5e-4)
    assert np.interp(np.log(METALLICITY_GRID[0]), lnZ, cum) == pytest.approx(0.0062, abs=5e-4)

    wf = np.trapezoid(fit["S"], fit["t"], axis=0)
    wf = wf / wf.sum()
    assert wf[fit["Z"] > 6 * Z_SUN].sum() == pytest.approx(0.0161, abs=5e-4)
    assert wf[fit["Z"] < METALLICITY_GRID[-1]].sum() == pytest.approx(0.7871, abs=5e-4)
    assert wf[fit["Z"] < METALLICITY_GRID[0]].sum() == pytest.approx(0.0008, abs=1e-4)


def test_sub_1e4_star_formation_is_early(table):
    cum = np.concatenate([np.zeros((table["z"].size, 1)), np.cumsum(table["per_bin"], 1)], 1)
    below = np.array([np.interp(np.log(1e-4), np.log(table["Z_edges"]), r) for r in cum])
    m = below * np.abs(np.gradient(table["t"]))
    assert m[table["z"] > 3].sum() / m.sum() == pytest.approx(0.90, abs=0.01)
    assert table["z"][np.searchsorted(np.cumsum(m) / m.sum(), 0.5)] == pytest.approx(5.5, abs=0.1)


def test_fci_fit_matches_levina_equations():
    """The FastCosmicIntegration skew-log-normal plotted in the cell is Eqs. 3-6."""
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        find_metallicity_distribution,
    )

    from grb_rates import MSSFR_PARAMS_LEVINA26_TNG100

    z = np.array([0.0, 1.0, 2.73, 6.0])
    dP, Z, _ = find_metallicity_distribution(
        z, np.log(1e-4), np.log(0.03), min_logZ=np.log(1e-7), **MSSFR_PARAMS_LEVINA26_TNG100
    )
    eq = (
        _levina_fit(z, Z)
        / (0.0172 * (1 + z) ** 1.4425 / (1 + ((1 + z) / 4.5299) ** 6.2261))[:, None]
    )
    m = eq > 1e-6 * eq.max()
    assert np.max(np.abs(dP[m] / eq[m] - 1)) < 1e-4


def test_replica_of_levina_figure_code(table):
    """Their reader and plotting code (Fit_SFRD_TNG: readTNGdata,
    interpolate_TNGdata with Nmetals = 60, SFRD_2Dplot_sidepanels), with
    interp2d(kind='cubic') replaced by its fitpack equivalent."""
    import astropy.units as u
    import h5py
    from astropy.cosmology import Planck15
    from scipy.interpolate import RectBivariateSpline

    from grb_io import TNG_SFRD_PATHS

    with h5py.File(TNG_SFRD_PATHS["TNG100-1"], "r") as f:
        bins, lb, sfr, zs = (f[k][:] for k in ("MetalBins", "Lookbacktimes", "Sfr", "Redshifts"))
    centre = (bins[:-1] + bins[1:]) / 2
    S = sfr / (75 / 0.6774) ** 3 / np.diff(np.log(bins))[0]
    keep = centre > 1e-6
    spl = RectBivariateSpline(lb[::-1], centre[keep], S[:, keep][::-1], kx=3, ky=3, s=0)
    z_new = np.arange(0, 14.1, 0.05)
    lb_new = Planck15.lookback_time(z_new).to(u.Gyr).value
    met = np.logspace(np.log10(centre).min(), np.log10(centre).max(), 60)
    o = np.argsort(lb_new)
    grid = np.empty((met.size, z_new.size))
    grid[:, o] = spl(lb_new[o], met).T
    grid[grid < 0] = 0
    side_tab = grid.sum(axis=1)
    side_fit = _levina_fit(z_new, met).T.sum(axis=1)
    assert side_tab.max() == pytest.approx(2.57, abs=0.01)
    assert side_fit.max() == pytest.approx(2.45, abs=0.01)
    assert met[np.argmax(side_tab)] / 0.014 == pytest.approx(0.88, abs=0.01)
    assert met[np.argmax(side_fit)] / 0.014 == pytest.approx(0.88, abs=0.01)
