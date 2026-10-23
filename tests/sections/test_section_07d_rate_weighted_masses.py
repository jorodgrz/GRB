"""Section 7d: merger-rate-weighted mass distributions by GRB class.

Source-level checks on the notebook cell, and a data-bound recomputation of
the numbers the Figure 7d caption quotes.
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
    i = next(k for k, c in enumerate(cells) if "".join(c["source"]).startswith("## 7d."))
    code, caption = cells[i + 1], cells[i + 2]
    assert code["cell_type"] == "code" and "".join(caption["source"]).startswith("**Figure 7d.**")
    out = "".join("".join(o.get("text", "")) for o in code.get("outputs", []))
    return "".join(code["source"]), "".join(caption["source"]), out


def test_cell_weights_by_merger_rate_and_stays_intrinsic():
    src, _, _ = _section_cells()
    assert "per_system_rate_weights(" in src
    assert "smooth_sigma=0" in src, "the area check must compare with the unsmoothed rate"
    for banned in ("beamed_rate", "apply_bhns_misalignment", "smooth_rate_curve"):
        assert banned not in src, f"{banned} has no place in a GW-comparable intrinsic rate"
    for ext in ("png", "pdf"):
        assert f"plots/mass_distributions_rate_weighted.{ext}" in src


def test_caption_numbers_appear_in_the_stored_output():
    """Every number the caption quotes from the cell is in the cell's printed output."""
    _, caption, out = _section_cells()
    assert out, "Section 7d has no stored output; run the cell"
    for s in (
        "25.5, 72.8, 93.4",
        "70.9, 189.2, 175.0",
        "67.4 -> 67.3%",
        "15.8 -> 15.5%",
        "11.3 -> 11.9%",
        "5.5 -> 5.4%",
        "formation 24.1%, merger rate 31.3%",
        "n_eff     361",
    ):
        assert s in out, s
    for s in (
        "52.6",
        "64.3",
        "59.0",
        "23.7",
        "9.6",
        "9.24",
        "9.36",
        "15.08",
        "14.12",
        "18.3",
        "4.4",
        "31.3",
        "24.1",
        "(361)",
        "137{,}482",
        "1{,}197{,}779",
    ):
        assert s in caption, s
    assert not re.search("[–—]", caption), "no en or em dashes in captions"


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_caption_shares_recompute_from_the_data():
    """Recompute the z = 0 class shares and the Q > 7 fraction independently of the notebook."""
    from astropy.cosmology import Planck15
    from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
        calculate_redshift_related_params,
    )

    import grb_io
    from grb_classify import classify_bhns, classify_bns_2024
    from grb_physics import A_BH_FID, M_TOV, cold_stable_bhns_mask, cold_stable_ns_mask
    from grb_rates import (
        dPdlogZ_binned_to_grid,
        fine_metallicity_grid,
        per_system_rate_weights,
        simulation_star_forming_mass,
        tng_sfr_and_dPdlogZ,
    )

    z, _, times, t_first, _, _ = calculate_redshift_related_params(
        max_redshift=10.0, redshift_step=0.01, cosmology=Planck15
    )
    grid = grb_io.METALLICITY_GRID
    z53 = grid[np.concatenate(([True], np.diff(grid) != 0))]
    mets, p_draw = fine_metallicity_grid(
        *grb_io.verify_shared_metallicity_prior(grb_io.DEFAULT_BNS_PATH, grb_io.DEFAULT_BHNS_PATH)
    )
    sfr, dP = tng_sfr_and_dPdlogZ(z, mets, box="TNG100-1")
    dP = dPdlogZ_binned_to_grid(dP, mets, z53, p_draw)
    n_formed = sfr / simulation_star_forming_mass(n_metallicities=z53.size)

    def rw(d, delays):
        return per_system_rate_weights(
            0.0,
            z,
            times,
            t_first,
            n_formed,
            p_draw,
            dP,
            mets,
            d["metallicity"],
            delays,
            d["weights"],
        )

    bns = grb_io.load_bns_with_channels(expected_model="A", expected_ns_max=2.5)
    bns = grb_io.filter_systems(bns, cold_stable_ns_mask(bns["m1"], bns["m2"], m_tov=M_TOV))
    sb = classify_bns_2024(bns["m1"], bns["m2"])["sbGRB + blue KN"]
    for delays, expected in ((bns["delay_time"], 67.4), (np.ones_like(bns["delay_time"]), 67.3)):
        w = rw(bns, delays)
        assert 100 * w[sb].sum() / w.sum() == pytest.approx(expected, abs=0.05)

    bhns = grb_io.load_bhns_with_channels(expected_model="A", expected_ns_max=2.5)
    bhns = grb_io.filter_systems(
        bhns, cold_stable_bhns_mask(bhns["M_BH"], bhns["M_NS"], m_tov=M_TOV)
    )
    w = rw(bhns, bhns["delay_time"])
    q_hi = bhns["M_BH"] / bhns["M_NS"] > 7
    assert 100 * w[q_hi].sum() / w.sum() == pytest.approx(31.3, abs=0.05)
    disk = classify_bhns(bhns["M_BH"], bhns["M_NS"], a_BH=A_BH_FID)["lbGRB + red KN (BHNS disk)"]
    assert 100 * w[disk].sum() / w.sum() == pytest.approx(4.4, abs=0.05)
