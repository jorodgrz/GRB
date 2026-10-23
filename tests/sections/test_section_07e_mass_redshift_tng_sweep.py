"""Section 7e: merger-rate-weighted masses overlaid in redshift, per TNG box.

Source-level checks on the notebook cell and caption, and data-bound checks of
the BNS mass floor and the TNG100-1 spike share the caption quotes.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_NB_PATH = os.path.join(_REPO_ROOT, "grb_main.ipynb")


def _section_cells():
    with open(_NB_PATH) as fh:
        cells = json.load(fh)["cells"]
    i = next(k for k, c in enumerate(cells) if "".join(c["source"]).startswith("## 7e."))
    code, caption = cells[i + 1], cells[i + 2]
    assert code["cell_type"] == "code" and "".join(caption["source"]).startswith("**Figure 7e.**")
    out = "".join("".join(o.get("text", "")) for o in code.get("outputs", []))
    return "".join(code["source"]), "".join(caption["source"]), out


def test_cell_follows_levina_conventions():
    src, _, _ = _section_cells()
    assert "Z_MERGE_7e = (0.2, 1.0, 2.0, 4.0, 6.0, 8.0)" in src
    assert "Z_FIRST_SF_LEVINA26 = 14.0" in src
    assert "z_first_SF=Z_FIRST_SF_LEVINA26" in src
    assert "TNG_BOXES_7e = ('TNG50-1', 'TNG100-1', 'TNG300-1')" in src
    assert "smooth_sigma=0" in src
    for banned in ("beamed_rate", "apply_bhns_misalignment", "find_sfr(", "MSSFR_PARAMS_LEVINA26"):
        assert banned not in src, f"{banned}: the sweep runs on the TNG tables, intrinsic rates"
    for ext in ("png", "pdf"):
        assert f"plots/mass_distributions_rate_weighted_tng_sweep.{ext}" in src


def test_caption_numbers_appear_in_the_stored_output():
    _, caption, out = _section_cells()
    assert out, "Section 7e has no stored output; run the cell"
    for s in (
        "97.510",
        "28.966",
        "9.401",
        "2.512",
        "145.086",
        "58.625",
        "2.009",
        "0.514",
        "53.3",
        "23.2",
        "20.9",
        "26.4",
        "2.72 [ 3.55]",
        "3.06 [ 3.89]",
        "9.34 [14.20]",
        "8.43 [14.43]",
        "11.28 [22.07]",
        "10.60 [16.79]",
        "10.80 [17.18]",
        "11.10 [20.97]",
        "     371",
    ):
        assert s in out, s
    for s in (
        "3.37",
        "2.47",
        "3.74",
        "3.91",
        "53.3",
        "23.2",
        "20.9",
        "26.4",
        "2.72",
        "3.06",
        "3.55",
        "3.89",
        "9.34",
        "8.43",
        "11.28",
        "14.20",
        "22.07",
        "16.79",
        "17.18",
        "20.97",
        "371",
        "2.5376",
        "2.5552",
        "1.2600",
        "1.2776",
        "6.5 percent",
        "8.3 percent",
        "30 percent",
        "Z = 0.0114, 0.0102, 0.0091 and 0.0082",
    ):
        assert s in caption, s
    assert not re.search("[–—]", caption), "no en or em dashes in captions"


@pytest.mark.requires_data
def test_bns_mass_floor_is_the_two_compas_ns_floors():
    """Min M_tot = ECSN mass (1.26) + lightest Fryer delayed NS (M_bar = 1.4)."""
    import grb_io
    from grb_physics import M_TOV, cold_stable_ns_mask

    m_ecsn = 1.26  # COMPAS constants.h MECS_REM, Belczynski+ 2008 Eq. 3
    # Fryer delayed, CO core < 2.5: proto 1.2 + fallback 0.2; M_bar = M + 0.075 M^2.
    m_delayed_min = (-1.0 + np.sqrt(1.0 + 4.0 * 0.075 * 1.4)) / (2.0 * 0.075)
    d = grb_io.load_bns(grb_io.DEFAULT_BNS_PATH)
    d = grb_io.filter_systems(d, cold_stable_ns_mask(d["m1"], d["m2"], m_tov=M_TOV))
    assert d["m2"].min() == pytest.approx(m_ecsn, abs=1e-5)
    assert (d["m1"] + d["m2"]).min() == pytest.approx(m_ecsn + m_delayed_min, abs=1e-4)


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_tng100_spike_share_recomputes():
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
    )

    import grb_io
    from grb_physics import M_TOV, cold_stable_ns_mask
    from grb_rates import (
        dPdlogZ_binned_to_grid,
        fine_metallicity_grid,
        per_system_rate_weights,
        simulation_star_forming_mass,
        tng_sfr_and_dPdlogZ,
    )

    z, _, t, tf, _, _ = calculate_redshift_related_params(
        max_redshift=14.0, redshift_step=0.01, z_first_SF=14.0, cosmology=Planck15
    )
    grid = grb_io.METALLICITY_GRID
    z53 = grid[np.concatenate(([True], np.diff(grid) != 0))]
    mets, p_draw = fine_metallicity_grid(
        *grb_io.verify_shared_metallicity_prior(grb_io.DEFAULT_BNS_PATH, grb_io.DEFAULT_BHNS_PATH)
    )
    sfr, dP = tng_sfr_and_dPdlogZ(z, mets, box="TNG100-1")
    dP = dPdlogZ_binned_to_grid(dP, mets, z53, p_draw)
    nf = sfr / simulation_star_forming_mass(n_metallicities=z53.size)
    d = grb_io.load_bns(grb_io.DEFAULT_BNS_PATH)
    d = grb_io.filter_systems(d, cold_stable_ns_mask(d["m1"], d["m2"], m_tov=M_TOV))
    m_tot = d["m1"] + d["m2"]
    spike = ((m_tot >= 2.5375) & (m_tot <= 2.5378)) | ((m_tot >= 2.5551) & (m_tot <= 2.5554))
    for zm, expected in ((0.2, 53.3), (8.0, 23.2)):
        w = per_system_rate_weights(
            zm, z, t, tf, nf, p_draw, dP, mets, d["metallicity"], d["delay_time"], d["weights"]
        )
        assert 100 * w[spike].sum() / w.sum() == pytest.approx(expected, abs=0.05)


# ─────────────────────────────────────────────────────────────────────
# Independent verification of the stored rates and of the KDE curves
# ─────────────────────────────────────────────────────────────────────
_Z_MERGE = (0.2, 1.0, 2.0, 4.0, 6.0, 8.0)
_BOXES = ("TNG50-1", "TNG100-1", "TNG300-1")
_BOX_MPC_H = {"TNG50-1": 35.0, "TNG100-1": 75.0, "TNG300-1": 205.0}  # Nelson+ 2019 Table 1
_TEETH_MSUN = (14.03, 14.975, 15.95, 16.9)


def _stored_rates():
    """R(z) table printed by the Section 7e cell, keyed by (population, box)."""
    _, _, out = _section_cells()
    stored = {}
    for line in out.splitlines():
        m = re.match(r"\s+(BNS|BHNS)\s+(TNG\S+)\s+([\d.\s]+)$", line)
        if m and "[" not in line and len(m.group(3).split()) == 6:
            stored.setdefault((m.group(1), m.group(2)), [float(v) for v in m.group(3).split()])
    assert len(stored) == 6, stored.keys()
    return stored


@pytest.fixture(scope="module")
def _samples():
    import grb_io
    from grb_physics import M_TOV, cold_stable_bhns_mask, cold_stable_ns_mask

    bns = grb_io.load_bns(grb_io.DEFAULT_BNS_PATH)
    bns = grb_io.filter_systems(bns, cold_stable_ns_mask(bns["m1"], bns["m2"], m_tov=M_TOV))
    bhns = grb_io.load_bhns(grb_io.DEFAULT_BHNS_PATH)
    bhns = grb_io.filter_systems(
        bhns, cold_stable_bhns_mask(bhns["M_BH"], bhns["M_NS"], m_tov=M_TOV)
    )
    grid = grb_io.METALLICITY_GRID
    return {"BNS": bns, "BHNS": bhns, "grid": grid[np.concatenate(([True], np.diff(grid) != 0))]}


def _independent_rate(d, grid, box, z_merge, emulate_fci=False):
    """R(z_merge) from the raw HDF5 table, sharing no code with grb_rates.

    Exact TNG-bin / COMPAS-cell overlaps (uniform in ln Z inside a TNG bin),
    exact formation redshift from astropy ages, star formation from z = 14.
    With ``emulate_fci`` the two FCI discretisations are reproduced: the
    formation rate read at ceil(z_form / 0.01) and COMPAS cell edges snapped
    to the 0.01 fine ln Z grid.
    """
    import h5py
    from astropy.cosmology import Planck15

    import grb_io
    from grb_rates import simulation_star_forming_mass

    with h5py.File(grb_io.TNG_SFRD_PATHS[box], "r") as f:
        z_tab = f["Redshifts"][...]
        edges = np.log(f["MetalBins"][...])
        sfr = f["Sfr"][...]
    o = np.argsort(z_tab)
    z_tab, sfr = z_tab[o], sfr[o] / (_BOX_MPC_H[box] / 0.6774 / 1e3) ** 3
    mid = 0.5 * (np.log(grid[:-1]) + np.log(grid[1:]))
    if emulate_fci:
        ln_fine = np.arange(-12.0, 0.01, 0.01)
        cell = np.concatenate(([-np.inf], ln_fine[np.searchsorted(ln_fine, mid)], [ln_fine[-1]]))
    else:
        cell = np.concatenate(([np.log(1e-90)], mid, [0.0]))
    lo = np.maximum(edges[:-1, None], cell[None, :-1])
    hi = np.minimum(edges[1:, None], cell[None, 1:])
    s_cell = sfr @ (np.clip(hi - lo, 0.0, None) / np.diff(edges)[:, None])

    zg = np.linspace(0.0, 14.0, 140001)
    age = Planck15.age(zg).to("Myr").value
    t_form = np.interp(z_merge, zg, age) - d["delay_time"]
    ok = t_form >= age[-1]
    z_form = np.interp(t_form[ok], age[::-1], zg[::-1])
    if emulate_fci:
        z_form = np.ceil(np.round(z_form / 0.01, 9)) * 0.01
    k = np.searchsorted(grid, d["metallicity"][ok])
    s = np.empty(z_form.size)
    for kk in np.unique(k):
        m = k == kk
        s[m] = np.interp(z_form[m], z_tab, s_cell[:, kk])
    return (
        (d["weights"][ok] * s).sum()
        * grid.size
        / simulation_star_forming_mass(n_metallicities=grid.size)
    )


@pytest.mark.requires_data
@pytest.mark.slow
def test_stored_rates_match_an_independent_integration(_samples):
    """All 36 stored R(z) values against a from-scratch integration of the raw tables.

    Plain: within 1 percent (measured 0.84 percent, the FCI redshift-slot and
    fine-grid discretisations).  With those two discretisations reproduced the
    gap closes to 1e-3 (measured 1.0e-3, BHNS z = 8, FCI's dropped first-SF slot).
    """
    stored = _stored_rates()
    for box in _BOXES:
        for pop in ("BNS", "BHNS"):
            for zm, r_nb in zip(_Z_MERGE, stored[pop, box]):
                r_plain = _independent_rate(_samples[pop], _samples["grid"], box, zm)
                r_fci = _independent_rate(
                    _samples[pop], _samples["grid"], box, zm, emulate_fci=True
                )
                assert abs(r_nb / r_plain - 1) < 0.01, (box, pop, zm, r_nb, r_plain)
                assert abs(r_nb / r_fci - 1) < 1.5e-3, (box, pop, zm, r_nb, r_fci)


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_kde_curves_follow_weighted_histograms(_samples):
    """Every Section 7e curve against a weighted histogram, at the cell's bandwidths."""
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
    )

    import grb_io
    from grb_plot_style import weighted_kde_per_dex
    from grb_rates import (
        dPdlogZ_binned_to_grid,
        fine_metallicity_grid,
        per_system_rate_weights,
        simulation_star_forming_mass,
        tng_sfr_and_dPdlogZ,
    )

    src, _, _ = _section_cells()
    assert "w=w_bns[~_in_spike_7e], bw=0.003" in src and "w=w_bhns, bw=0.01" in src
    grid = _samples["grid"]
    z, _, t, tf, _, _ = calculate_redshift_related_params(
        max_redshift=14.0, redshift_step=0.01, z_first_SF=14.0, cosmology=Planck15
    )
    mets, p_draw = fine_metallicity_grid(
        *grb_io.verify_shared_metallicity_prior(grb_io.DEFAULT_BNS_PATH, grb_io.DEFAULT_BHNS_PATH)
    )
    bns, bhns = _samples["BNS"], _samples["BHNS"]
    mt = bns["m1"] + bns["m2"]
    spike = ((mt >= 2.5375) & (mt <= 2.5378)) | ((mt >= 2.5551) & (mt <= 2.5554))
    cases = {"BNS": (bns, mt, ~spike, 0.003, 0.02), "BHNS": (bhns, bhns["M_BH"], None, 0.01, 0.05)}
    worst = {"BNS": 0.0, "BHNS": 0.0, "teeth": 0.0}
    for box in _BOXES:
        sfr, dP = tng_sfr_and_dPdlogZ(z, mets, box=box)
        dP = dPdlogZ_binned_to_grid(dP, mets, grid, p_draw)
        nf = sfr / simulation_star_forming_mass(n_metallicities=grid.size)
        for pop, (d, x, keep, bw, binw) in cases.items():
            keep = np.ones(x.size, bool) if keep is None else keep
            for zm in _Z_MERGE:
                rw = per_system_rate_weights(
                    zm,
                    z,
                    t,
                    tf,
                    nf,
                    p_draw,
                    dP,
                    mets,
                    d["metallicity"],
                    d["delay_time"],
                    d["weights"],
                )
                xk, wk = x[keep], rw[keep]
                xs, pdex = weighted_kde_per_dex(xk, wk, xk.min(), xk.max(), bw)
                lxs = np.log10(xs)
                dl = lxs[1] - lxs[0]
                # the plotted dR/dM integrates to the continuum rate
                assert (
                    abs(np.trapezoid(pdex * wk.sum() / (xs * np.log(10)), xs) / wk.sum() - 1) < 1e-3
                )
                e = np.arange(np.log10(xk.min()), np.log10(xk.max()) + binw, binw)
                kde_bin = np.array(
                    [pdex[(lxs >= a) & (lxs < b)].sum() * dl for a, b in zip(e[:-1], e[1:])]
                )
                h, _ = np.histogram(np.log10(xk), e, weights=wk)
                s2, _ = np.histogram(np.log10(xk), e, weights=wk**2)
                ok = np.divide(h**2, s2, out=np.zeros_like(h), where=s2 > 0) >= 100
                rel = np.abs(kde_bin[ok] * wk.sum() / h[ok] - 1)
                if pop == "BHNS":
                    near = np.array(
                        [
                            any(a - 0.3 < m < b + 0.3 for m in _TEETH_MSUN)
                            for a, b in zip(10 ** e[:-1][ok], 10 ** e[1:][ok])
                        ]
                    )
                    worst["teeth"] = max(worst["teeth"], rel[near].max(initial=0.0))
                    rel = rel[~near]
                worst[pop] = max(worst[pop], rel.max())
    # caption: 6.5 percent (BNS), 8.3 percent (BHNS), up to 30 percent at the teeth
    assert worst["BNS"] < 0.066, worst
    assert worst["BHNS"] < 0.084, worst
    assert worst["teeth"] < 0.305, worst


@pytest.mark.requires_data
def test_bhns_teeth_are_single_metallicity_pileups(_samples):
    """The 14 to 17 Msun pile-ups the KDE smooths are single-metallicity excesses.

    Each is 1.7 to 2.5 times denser (raw) than the 0.1 Msun just below it, and
    its dominant grid metallicity jumps from 2 to 12 percent of the systems
    below to 46 to 70 percent inside, the numbers the Figure 7e caption quotes.
    """
    d = _samples["BHNS"]
    x, Z = d["M_BH"], d["metallicity"]
    for (lo, hi), z_main in (
        ((14.04, 14.08), 0.01137),
        ((14.95, 15.00), 0.01018),
        ((15.90, 16.00), 0.00912),
        ((16.85, 16.95), 0.00817),
    ):
        tooth = (x >= lo) & (x < hi)
        below = (x >= lo - 0.15) & (x < lo - 0.05)
        density_ratio = (tooth.sum() / (hi - lo)) / (below.sum() / 0.1)
        share_tooth = (np.abs(Z[tooth] / z_main - 1) < 1e-3).mean()
        share_below = (np.abs(Z[below] / z_main - 1) < 1e-3).mean()
        assert 1.65 < density_ratio < 2.5, (lo, density_ratio)
        assert share_tooth > 0.45 and share_below < 0.13, (lo, share_tooth, share_below)


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_star_formation_onset_numbers_in_the_caption(_samples):
    """z = 10 onset against z = 14: -5 percent (BNS) and -20 percent (BHNS) at z = 8, < 0.5 percent at z <= 4."""
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
    )

    import grb_io
    from grb_rates import (
        dPdlogZ_binned_to_grid,
        fine_metallicity_grid,
        per_system_rate_weights,
        simulation_star_forming_mass,
        tng_sfr_and_dPdlogZ,
    )

    grid = _samples["grid"]
    mets, p_draw = fine_metallicity_grid(
        *grb_io.verify_shared_metallicity_prior(grb_io.DEFAULT_BNS_PATH, grb_io.DEFAULT_BHNS_PATH)
    )
    rates = {}
    for onset in (10.0, 14.0):
        z, _, t, tf, _, _ = calculate_redshift_related_params(
            max_redshift=onset, redshift_step=0.01, z_first_SF=onset, cosmology=Planck15
        )
        sfr, dP = tng_sfr_and_dPdlogZ(z, mets, box="TNG100-1")
        dP = dPdlogZ_binned_to_grid(dP, mets, grid, p_draw)
        nf = sfr / simulation_star_forming_mass(n_metallicities=grid.size)
        for pop in ("BNS", "BHNS"):
            d = _samples[pop]
            for zm in _Z_MERGE:
                rates[onset, pop, zm] = per_system_rate_weights(
                    zm,
                    z,
                    t,
                    tf,
                    nf,
                    p_draw,
                    dP,
                    mets,
                    d["metallicity"],
                    d["delay_time"],
                    d["weights"],
                ).sum()
    ratio = {k[1:]: rates[k] / rates[(14.0,) + k[1:]] for k in rates if k[0] == 10.0}
    assert ratio["BNS", 8.0] == pytest.approx(0.947, abs=0.002)
    assert ratio["BHNS", 8.0] == pytest.approx(0.798, abs=0.002)
    assert max(abs(1 - ratio[p, z]) for p in ("BNS", "BHNS") for z in (0.2, 1.0, 2.0, 4.0)) < 0.005
