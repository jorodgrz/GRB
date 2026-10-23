"""Section 3c: sbGRB + blue KN delay times across the 20-model grid.

The fitted indices are what the caption quotes, so the real-data test pins
them: Model A under both weightings, the range and median over the 19
summarised models (E excluded at ESS 43), F as the one model steeper than
t^-2, the two shallower than t^-1, and the per-decade curvature of Model A.
Every index is fitted from the maximum of the model's own dp/dt to 14.1 Gyr,
the literature's form.  A source-level check keeps the cell on that rule.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_NB_PATH = os.path.join(_REPO_ROOT, "grb_main.ipynb")
_DATA = os.path.join(_REPO_ROOT, "data")

T_DELAY_MIN_MYR, T_DELAY_MAX_MYR = 3.0, 14100.0
T_FIT_HI_MYR = T_DELAY_MAX_MYR
KDE_BW_DEX = 0.04
# Paper-I keys; E is fitted but not summarised (Kish ESS 43).
NOT_SUMMARISED = ("E",)


def _cell_3c():
    with open(_NB_PATH) as fh:
        nb = json.load(fh)
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    hits = [s for s in cells if "delay_time_sbgrb_20models" in s]
    assert len(hits) == 1, "expected exactly one Section 3c code cell"
    return hits[0]


@pytest.mark.unit
def test_section_3c_cell_fits_from_the_dpdt_maximum():
    src = _cell_3c()
    assert "weighted_kde_per_dex" in src
    assert "KDE_BW_DEX = 0.04" in src
    assert "powerlaw_index_mle" in src
    assert "np.argmax(_d / _x)" in src, "start at the dp/dt maximum, not the per-dex one"
    assert "T_FIT_HI_MYR = T_DELAY_MAX_MYR" in src, "fit support matches the literature lines"
    assert "errorbar" not in src, "model indices are drawn without bars"
    assert "np.polyfit" not in src, "3c fits the unbinned sample, not bins"
    assert "cold_stable_ns_mask" in src
    assert "(mathrm{Gyr})" in src.replace("\\", ""), "abscissa is in Gyr"
    assert "mathrm{Gyr}^{-1}" in src.replace("\\", ""), "ordinate is dp/dt in Gyr^-1"
    assert "SGRB_DTD_POWERLAW_LITERATURE" in src, "literature comes from grb_rates"
    assert "cosmic_sf_mass_fraction_per_grid_point" in src


@pytest.mark.unit
def test_section_3c_omits_model_f_from_the_top_panel_only():
    """F (paper-I key EH) is left off the dp/dt panel but keeps its index below.

    The top-panel loop runs over the omission list, the bottom panel over every
    model, and the legend count follows the drawn set rather than a literal.
    """
    src = _cell_3c()
    assert "TOP_PANEL_OMIT_3c = ('EH',)" in src
    assert "_top_3c = [s for s in _fitted_3c if s not in TOP_PANEL_OMIT_3c]" in src
    assert "for _s in _top_3c:" in src
    assert "for _i, _s in enumerate(ALL_MODEL_SUFFIXES):" in src, "bottom panel keeps F"
    assert "f'{len(_top_3c) - 1} other models'" in src
    assert "'18 other models'" not in src


def _load_sb():
    from grb_classify import classify_bns_2024
    from grb_io import ALL_MODEL_SUFFIXES, filter_systems, load_bns
    from grb_physics import M_TOV, cold_stable_ns_mask

    out = {}
    for s in ALL_MODEL_SUFFIXES:
        path = os.path.join(_DATA, f"COMPASCompactOutput_BNS_{s}.h5")
        if not os.path.exists(path):
            pytest.skip(f"missing {path}")
        b = load_bns(path, expected_model=s)
        b = filter_systems(b, cold_stable_ns_mask(b["m1"], b["m2"], m_tov=M_TOV))
        m = classify_bns_2024(b["m1"], b["m2"], m_tov=M_TOV)["sbGRB + blue KN"]
        out[s] = (b["delay_time"][m], b["weights"][m])
    return out


def _peak(t, w):
    from grb_plot_style import weighted_kde_per_dex

    x, d = weighted_kde_per_dex(t, w, T_DELAY_MIN_MYR, T_DELAY_MAX_MYR, KDE_BW_DEX)
    return x[np.argmax(d / x)]  # dp/dt = (dp/dlog10 t) / (t ln 10)


def _fit(t, w):
    from grb_rates import powerlaw_index_mle

    t_pk = _peak(t, w)
    return t_pk, powerlaw_index_mle(t, w, t_pk, T_FIT_HI_MYR)[0]


@pytest.mark.requires_data
@pytest.mark.slow
def test_section_3c_every_model_has_complete_delay_times():
    for s, (t, w) in _load_sb().items():
        assert t.size > 0, s
        assert np.all(np.isfinite(t) & (t > 0)), s
        assert np.all(w > 0), s
        assert t.min() >= T_DELAY_MIN_MYR and t.max() <= T_DELAY_MAX_MYR, s


@pytest.mark.requires_data
@pytest.mark.slow
def test_section_3c_fitted_indices_match_caption():
    r = {s: _fit(t, w) for s, (t, w) in _load_sb().items()}
    assert all(np.isfinite(a) for _, a in r.values()), "every model is fitted"
    a = {s: x[1] for s, x in r.items() if s not in NOT_SUMMARISED}
    assert len(a) == 19
    assert r["A"][0] == pytest.approx(42.6, abs=0.1)
    assert a["A"] == pytest.approx(-1.3215, abs=0.002)
    peaks = np.array([r[s][0] for s in a])
    assert peaks.min() == pytest.approx(22.2, abs=0.1)  # paper-II J
    assert peaks.max() == pytest.approx(49.6, abs=0.1)  # D
    v = np.array(list(a.values()))
    # paper-II F is the one index past t^-2: its dp/dt falls four decades
    # within 0.15 Gyr of its peak.
    assert [k for k, x in a.items() if x < -2.0] == ["EH"]
    assert a["EH"] == pytest.approx(-7.35, abs=0.01)
    no_f = {k: x for k, x in a.items() if k != "EH"}
    assert min(no_f, key=no_f.get) == "B" and no_f["B"] == pytest.approx(-1.40, abs=0.005)
    assert max(no_f, key=no_f.get) == "alpha10" and no_f["alpha10"] == pytest.approx(
        -0.72, abs=0.005
    )
    assert sorted(k for k, x in a.items() if x > -1.0) == ["F", "alpha10"]  # paper-II H, J
    assert np.median(v) == pytest.approx(-1.33, abs=0.005)


def _alpha_scipy(t, w, t_lo, t_hi=T_FIT_HI_MYR):
    """Weighted truncated power-law MLE written out, without grb_rates."""
    from scipy.optimize import minimize_scalar

    s = (t >= t_lo) & (t <= t_hi)
    log_t, ws = np.log(t[s]), w[s]

    def nll(a):
        q = a + 1.0
        return -(ws * (a * log_t - np.log((t_hi**q - t_lo**q) / q))).sum()

    return minimize_scalar(nll, bounds=(-9.0, 1.0), method="bounded", options={"xatol": 1e-6}).x


@pytest.mark.requires_data
@pytest.mark.slow
def test_section_3c_indices_reproduced_by_an_independent_likelihood():
    """A, the steepest (B) and two shallowest (H, J) indices, refitted with scipy.

    Paper-I keys F and alpha10 print as paper-II H and J.  Peaks and indices are
    the Section 3c table values; the tolerances cover one step of the KDE grid,
    which moves B's weight above the peak by 0.001.
    """
    from grb_classify import classify_bns_2024
    from grb_io import filter_systems, load_bns
    from grb_physics import M_TOV, cold_stable_ns_mask

    expected = {
        "A": (42.6, 0.895, -1.322),
        "B": (39.3, 0.858, -1.405),
        "F": (46.3, 0.944, -0.903),
        "alpha10": (22.2, 0.990, -0.722),
    }
    for s, (pk, wf, alpha) in expected.items():
        path = os.path.join(_DATA, f"COMPASCompactOutput_BNS_{s}.h5")
        if not os.path.exists(path):
            pytest.skip(f"missing {path}")
        b = load_bns(path, expected_model=s)
        b = filter_systems(b, cold_stable_ns_mask(b["m1"], b["m2"], m_tov=M_TOV))
        m = classify_bns_2024(b["m1"], b["m2"], m_tov=M_TOV)["sbGRB + blue KN"]
        t, w = b["delay_time"][m], b["weights"][m]
        t_pk = _peak(t, w)
        assert t_pk == pytest.approx(pk, abs=0.1), s
        assert w[t >= t_pk].sum() / w.sum() == pytest.approx(wf, abs=0.002), s
        assert _alpha_scipy(t, w, t_pk) == pytest.approx(alpha, abs=0.003), s


@pytest.mark.requires_data
@pytest.mark.slow
def test_section_3c_model_a_tail_curves():
    """The caption's curvature statement: the decade after the peak is steeper
    than the 0.1 to 14.1 Gyr tail, so the peak-start index sits between them."""
    from grb_rates import powerlaw_index_mle

    t, w = _load_sb()["A"]
    t_pk, a = _fit(t, w)
    first = powerlaw_index_mle(t, w, t_pk, 100.0)[0]
    tail = powerlaw_index_mle(t, w, 100.0, T_FIT_HI_MYR)[0]
    assert first == pytest.approx(-1.95, abs=0.01)
    assert tail == pytest.approx(-1.24, abs=0.01)
    assert first < a < tail


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_3c_every_model_shallower_than_ps26_and_zevin():
    """The caption's comparison: every summarised index apart from F's, under
    either metallicity weighting, is shallower than the PS26 90 percent
    intervals and the Zevin range, and Model A's cosmic-Z index is -1.16 on
    the TNG100-1 table (-1.14 and -1.16 on TNG50-1 and TNG300-1)."""
    from grb_io import METALLICITY_GRID
    from grb_rates import (
        SGRB_DTD_POWERLAW_LITERATURE as lit,
    )
    from grb_rates import cosmic_sf_mass_fraction_per_grid_point

    grid = METALLICITY_GRID[np.concatenate(([True], np.diff(METALLICITY_GRID) != 0))]
    F = cosmic_sf_mass_fraction_per_grid_point(grid, (grid[0], grid[-1]), tng_box="TNG100-1")
    F_box = {
        bx: cosmic_sf_mass_fraction_per_grid_point(grid, (grid[0], grid[-1]), tng_box=bx)
        for bx in ("TNG50-1", "TNG300-1")
    }
    assert F[-1] == pytest.approx(0.217, abs=0.001)
    steepest_allowed = -min(
        lit["PS26_ELF"]["alpha_tau_lo"],
        lit["PS26_QUSJ"]["alpha_tau_lo"],
        lit["Zevin22"]["alpha_tau_lo"],
    )
    grid_a, cosmic_a = [], []
    for s, (t, w) in _load_sb().items():
        if s in NOT_SUMMARISED:
            continue
        Z = _load_Z(s)
        t_pk, c = _fit(t, w * F[np.searchsorted(grid, Z)] * grid.size)
        cosmic_a.append(c)
        if s == "A":
            assert t_pk == pytest.approx(44.7, abs=0.1)
            assert c == pytest.approx(-1.156, abs=0.002)
            k = np.searchsorted(grid, Z)
            a50 = _fit(t, w * F_box["TNG50-1"][k] * grid.size)[1]
            a300 = _fit(t, w * F_box["TNG300-1"][k] * grid.size)[1]
            assert a50 == pytest.approx(-1.138, abs=0.002)
            assert a300 == pytest.approx(-1.163, abs=0.002)
        if s == "EH":  # paper-II F, steeper than everything (caption)
            assert c < steepest_allowed
            continue
        grid_a.append(_fit(t, w)[1])
        grid_a.append(c)
    v = np.array(grid_a)
    assert np.all(v > steepest_allowed), (v.min(), steepest_allowed)
    assert np.median(cosmic_a) == pytest.approx(-1.16, abs=0.005)


def _load_Z(s):
    from grb_classify import classify_bns_2024
    from grb_io import filter_systems, load_bns
    from grb_physics import M_TOV, cold_stable_ns_mask

    b = load_bns(os.path.join(_DATA, f"COMPASCompactOutput_BNS_{s}.h5"), expected_model=s)
    b = filter_systems(b, cold_stable_ns_mask(b["m1"], b["m2"], m_tov=M_TOV))
    return b["metallicity"][classify_bns_2024(b["m1"], b["m2"], m_tov=M_TOV)["sbGRB + blue KN"]]


@pytest.mark.unit
def test_section_3c_cell_runs_the_ps26_forward_model():
    src = _cell_3c()
    assert "fit_pracchia_salafia_2026_powerlaw" in src and "sfr_delay_convolution_shape" in src
    assert "np.logspace(np.log10(0.005), np.log10(3.0)" in src, "tau_min grid follows PS26 prior"
    assert "np.arange(0.0, 5.0" in src, "alpha_tau grid follows PS26 prior"


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_3c_ps26_framework_returns_model_tail_index_for_model_a():
    """Under Madau and Fragos star formation, PS26's power law recovers Model A's
    tail index; under the TNG100-1 SFR the models use, it is shallower still.
    The templates use Madau and Fragos in both cases, as PS26 do."""
    from grb_rates import (
        fit_pracchia_salafia_2026_powerlaw,
        madau_fragos_2017_sfr,
        sfr_delay_convolution_shape,
        tng_sfr_callable,
    )

    mf_sfr = madau_fragos_2017_sfr
    t, w = _load_sb()["A"]
    z = np.linspace(0.0, 3.0, 61)
    A = np.round(np.arange(0.0, 5.0 + 1e-9, 0.05), 2)
    T = np.logspace(np.log10(0.005), np.log10(3.0), 121)
    mf = fit_pracchia_salafia_2026_powerlaw(
        z, sfr_delay_convolution_shape(z, t / 1e3, w, sfr=mf_sfr), A, T, sfr=mf_sfr
    )
    lv = fit_pracchia_salafia_2026_powerlaw(
        z,
        sfr_delay_convolution_shape(z, t / 1e3, w, sfr=tng_sfr_callable("TNG100-1")),
        A,
        T,
        sfr=mf_sfr,
    )
    assert mf["alpha_tau"] == pytest.approx(1.25)
    assert mf["tau_min_Gyr"] == pytest.approx(0.023, abs=0.003)
    assert lv["alpha_tau"] == pytest.approx(1.00)
    assert lv["at_tau_min_floor"]
    assert np.average(t, weights=w) / 1e3 == pytest.approx(1.19, abs=0.01)
