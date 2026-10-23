"""Section 6c: sbGRB + blue KN rate against the Swift short-GRB sample.

Two kinds of check.  Source-level checks on ``grb_main.ipynb`` pin the
conventions the figure depends on, each of which has been wrong before:
the Du et al. (2025) Table 5 rate is an observed rate and takes no beaming
factor; the two panel (a) bands carry the same jet-angle range, and the
Pracchia and Salafia (2026) R_0 interval is never stacked on top of it; the
LVK bar folds in Model A's sbGRB share of the z = 0 merger *rate*, not of the
formation weight; model E is left off panel (b).  Numeric checks recompute
the plotted values from the published constants, independently of the cell.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_NB_PATH = os.path.join(_REPO_ROOT, "grb_main.ipynb")
_CACHE_6D = os.path.join(_REPO_ROOT, "plots", "rate_bns_by_class_per_model.npz")


def _cell_6c():
    with open(_NB_PATH) as fh:
        nb = json.load(fh)
    cells = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "_binned_R_obs_6c" in "".join(c["source"])
    ]
    assert len(cells) == 1, f"expected exactly one Section 6c code cell, found {len(cells)}"
    return cells[0]


def _f_b(theta_deg):
    return 1.0 - np.cos(np.radians(theta_deg))


# ---------------------------------------------------------------------------
# Conventions in the cell source
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_du25_marker_takes_the_cosmology_rescale_only():
    """Table 5 is an observed rate; Du's 9.8 deg jet angle only enters their
    intrinsic conversion, so reprojecting it onto our angle is wrong."""
    src = _cell_6c()
    assert "_du25_R_median = _du25_R_median_raw * _du25_cosmo_scale\n" in src
    assert "_du25_fb_fid" not in src
    assert "lolims=True" in src.split("_du25_x_6c = 0.10")[1].split("_lvk_x_6c")[0]


@pytest.mark.unit
def test_panel_a_bands_share_the_jet_angle_range_and_r0_is_not_stacked():
    src = _cell_6c()
    for edge in ("lo", "fid", "hi"):
        assert f"_ps26_beamed_{edge} = _ps26_curve_6c['R_best'] * _f_beam_{edge}_6c" in src
    # The R_0 interval appears only unbeamed, in panel (b).
    assert "R_rate_lo'] * _f_beam" not in src
    assert "R_rate_hi'] * _f_beam" not in src
    assert (
        "ax_b.fill_between(\n    redshifts, _ps26_curve_6c['R_rate_lo'], _ps26_curve_6c['R_rate_hi']"
        in src
    )


@pytest.mark.unit
def test_lvk_bar_folds_the_rate_fraction_and_the_fiducial_beaming():
    src = _cell_6c()
    assert "_frac_sb_A_6c = float(_R_int_sb_6c[iz0] / merger_rates_BNS['All BNS'][iz0])" in src
    assert "w_bns[cls24" not in src.split("_frac_sb_A_6c =")[1].split("\n")[0]
    assert "_lvk_bns['R_lo'] * _frac_sb_A_6c * _f_beam_fid_6c" in src
    assert "_lvk_bns['R_hi'] * _frac_sb_A_6c * _f_beam_fid_6c" in src


@pytest.mark.unit
def test_swift_points_use_the_exact_volume_integral():
    src = _cell_6c()
    assert "quad(lambda z: _dVc_dz_Gpc3(z) / (1.0 + z), lo, hi)" in src
    assert "R = N / denom" in src
    assert "(1.0 + zbar) / denom" not in src


@pytest.mark.unit
def test_panel_b_omits_model_e_and_keeps_the_pracchia_curve_unsmoothed():
    src = _cell_6c()
    assert "_OMIT_6c = ('E',)" in src
    assert "smooth_rate_curve(_ps26" not in src
    assert "smooth_rate_curve(_R_A_beam_fid_6c, edge='linear')" in src


# ---------------------------------------------------------------------------
# Plotted values recomputed from the published constants
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_bright_points_recompute_from_fong22_redshifts():
    """69 measured-z classical bursts, 4 equal-count bins, 17.1 yr, 1.40 sr."""
    from astropy.cosmology import FlatLambdaCDM
    from scipy.integrate import quad

    from grb_offsets import (
        FONG22_SGRB_ASSUMED_Z1,
        FONG22_SGRB_NAMES,
        FONG22_SGRB_REDSHIFTS,
        SGRB_RECLASSIFIED_AS_LGRB_KN,
    )
    from grb_physics import H0_PLANCK15, OMEGA_M_PLANCK15
    from grb_rates import F_SKY_BAT, T_SWIFT_BRIGHT_YR

    keep = [
        z
        for n, z in zip(FONG22_SGRB_NAMES, FONG22_SGRB_REDSHIFTS)
        if n not in FONG22_SGRB_ASSUMED_Z1 and n not in SGRB_RECLASSIFIED_AS_LGRB_KN
    ]
    z = np.sort(np.asarray(keep))
    assert z.size == 69

    cosmo = FlatLambdaCDM(H0=H0_PLANCK15, Om0=OMEGA_M_PLANCK15)

    def dv_over_1pz(x):
        return 4 * np.pi * cosmo.differential_comoving_volume(x).value / 1e9 / (1 + x)

    parts = np.array_split(z, 4)
    edges = [z[0]] + [0.5 * (parts[i][-1] + parts[i + 1][0]) for i in range(3)] + [z[-1]]
    rates = [
        len(p) / (quad(dv_over_1pz, lo, hi)[0] * T_SWIFT_BRIGHT_YR * F_SKY_BAT)
        for p, lo, hi in zip(parts, edges[:-1], edges[1:])
    ]
    np.testing.assert_allclose(rates, [0.5400, 0.3962, 0.1409, 0.0321], rtol=2e-3)


@pytest.mark.unit
def test_du25_marker_values():
    from grb_physics import H0_PLANCK15
    from grb_rates import DU25_LOCAL_RATES_SWIFT_BAT

    vals = np.array([v[0] for v in DU25_LOCAL_RATES_SWIFT_BAT.values()])
    errs = np.array([v[1] for v in DU25_LOCAL_RATES_SWIFT_BAT.values()])
    scale = (H0_PLANCK15 / 70.0) ** 3
    got = np.array([np.median(vals), (vals - errs).min(), (vals + errs).max()]) * scale
    np.testing.assert_allclose(got, [0.5437, 0.3625, 1.1146], rtol=1e-3)


@pytest.mark.unit
def test_pracchia_salafia_values_at_z0_in_both_panels():
    from grb_rates import CLASS_THETA_J, pracchia_salafia_2026_Rz

    th = CLASS_THETA_J["sbGRB"]
    c = pracchia_salafia_2026_Rz(np.array([0.0]))
    r0 = float(c["R_best"][0])
    assert r0 == pytest.approx(10**1.92, rel=1e-9)
    # Panel (a): best-fit R_0 over 6, 16, 26 deg.
    np.testing.assert_allclose(
        [r0 * _f_b(th["lo"]), r0 * _f_b(th["fid"]), r0 * _f_b(th["hi"])],
        [0.4556, 3.222, 8.418],
        rtol=1e-3,
    )
    # Panel (b): the Table 2 90 percent CR, unbeamed.
    np.testing.assert_allclose(
        [c["R_rate_lo"][0], c["R_rate_hi"][0]], [10**1.00, 10**3.42], rtol=1e-9
    )


@pytest.mark.skipif(not os.path.exists(_CACHE_6D), reason="Section 6d cache not built")
def test_lvk_bar_endpoints_from_the_model_a_rate_fraction():
    """f_sb(0) = R_sb(0) / R_BNS(0) = 0.674 and f_b(16 deg) = 0.0387 put the
    panel (a) bar at [0.1332, 4.0392], with Model A's 0.6662 inside it."""
    from grb_rates import CLASS_THETA_J, LVK_GWTC5_LOCAL_RATES

    c = np.load(_CACHE_6D, allow_pickle=False)
    suffixes = [str(s) for s in c["suffixes"]]
    keys = [str(k) for k in c["class_keys"]]
    iz0 = int(np.argmin(np.abs(c["redshifts"])))
    r_cls = c["R_class"][suffixes.index("A"), :, iz0]
    r_sb0 = float(r_cls[keys.index("sbGRB + blue KN")])
    f_sb0 = r_sb0 / float(r_cls.sum())
    f_b = _f_b(CLASS_THETA_J["sbGRB"]["fid"])
    lvk = LVK_GWTC5_LOCAL_RATES["BNS"]

    assert f_sb0 == pytest.approx(0.674004, rel=1e-5)
    assert f_b == pytest.approx(0.038738, rel=1e-4)
    lo, hi = lvk["R_lo"] * f_sb0 * f_b, lvk["R_hi"] * f_sb0 * f_b
    np.testing.assert_allclose([lo, hi], [0.133160, 4.039180], rtol=1e-5)
    assert lo <= r_sb0 * f_b <= hi


@pytest.mark.skipif(not os.path.exists(_CACHE_6D), reason="Section 6d cache not built")
def test_panel_b_model_e_is_the_outlier_it_is_omitted_for():
    """E sits two decades below every other model at z = 0, which is the
    reason the caption gives for leaving it off the drawing."""
    c = np.load(_CACHE_6D, allow_pickle=False)
    suffixes = [str(s) for s in c["suffixes"]]
    keys = [str(k) for k in c["class_keys"]]
    iz0 = int(np.argmin(np.abs(c["redshifts"])))
    r_sb = c["R_class"][:, keys.index("sbGRB + blue KN"), iz0]
    i_e = suffixes.index("E")
    others = np.delete(r_sb, i_e)
    assert r_sb[i_e] == pytest.approx(0.072, abs=5e-4)
    assert others.min() / r_sb[i_e] > 100
