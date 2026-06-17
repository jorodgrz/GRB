"""Section 5b of grb_main.ipynb: eta(Z) per GRB class, split by channel.

eta(Z) from ``formation_efficiency`` shares a single mean-evolved-mass
denominator, so the per-channel curves are additive: within a class they must
sum to the per-class total bin by bin.  These tests pin that additivity on the
live Model A pipeline.  ``mean_mass_evolved`` is a positive constant that
cancels in the additivity check, so a placeholder value is used and no
calibration / FCI dependency is needed.

Marked ``requires_data``: skips cleanly without the Model A HDF5s.

References: Gottlieb et al. (2024), arXiv:2411.13657 (four-class scheme);
Broekgaarden et al. (2021), arXiv:2103.02608, Sec. 5 (channels I-V).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import (
    classify_bhns,
    classify_bns_2024,
    classify_formation_channels,
)
from grb_io import METALLICITY_GRID, load_bhns_with_channels, load_bns_with_channels
from grb_rates import LOGZ_MAX_PHYSICAL, formation_efficiency

_MEAN_MASS_PLACEHOLDER = 1.0e7  # cancels in the additivity ratio; units Msun

# Mirror of the Section 5 / 5b display binning so the test guards the exact
# coarse-bin construction the notebook plots.
_Z_SUN = 0.0142
_N_Z_BINS_COARSE = 15


def _coarse_grouping():
    uniq = np.concatenate(([True], np.diff(METALLICITY_GRID) != 0))
    Z_grid = METALLICITY_GRID[uniq]
    z_phys_max = (10.0**LOGZ_MAX_PHYSICAL) * _Z_SUN
    ok_grid = np.where(Z_grid <= z_phys_max)[0]
    grp = np.arange(len(ok_grid)) * _N_Z_BINS_COARSE // len(ok_grid)
    return Z_grid, ok_grid, grp


def _coarse_sw(Z, w, mask, Z_grid, ok_grid, grp):
    """Coarse-bin weighted sum for `mask` (mirrors notebook _bin_sums + _eta_binned)."""
    Zs = Z[mask]
    ws = w[mask].astype(float)
    idx = np.searchsorted(Z_grid, Zs)
    in_grid = (idx < len(Z_grid)) & np.isclose(Zs, Z_grid[np.clip(idx, 0, len(Z_grid) - 1)])
    Sw = np.bincount(idx[in_grid], weights=ws[in_grid], minlength=len(Z_grid))
    return np.bincount(grp, weights=Sw[ok_grid], minlength=_N_Z_BINS_COARSE)


def _assert_coarse_channels_sum_to_class(Z, w, class_masks, channel_masks):
    Z_grid, ok_grid, grp = _coarse_grouping()
    for cl_label, cl_mask in class_masks.items():
        channel_sum = sum(
            _coarse_sw(Z, w, cl_mask & m, Z_grid, ok_grid, grp) for m in channel_masks.values()
        )
        class_total = _coarse_sw(Z, w, cl_mask, Z_grid, ok_grid, grp)
        np.testing.assert_allclose(
            channel_sum,
            class_total,
            rtol=1e-12,
            atol=0,
            err_msg=f"coarse-bin channel eta does not sum to per-class total for {cl_label!r}",
        )


def _channels(d: dict) -> dict:
    return classify_formation_channels(
        dblCE=d["dblCE"],
        fc_CEE=d["fc_CEE"],
        fc_mt_p1=d["fc_mt_p1"],
        fc_mt_s1=d["fc_mt_s1"],
        fc_mt_p1_K1=d["fc_mt_p1_K1"],
        fc_mt_s1_K2=d["fc_mt_s1_K2"],
    )


def _assert_channels_sum_to_class(Z, w, class_masks, channel_masks):
    for cl_label, cl_mask in class_masks.items():
        masks = {c: cl_mask & m for c, m in channel_masks.items()}
        masks["__class_total__"] = cl_mask
        eff = formation_efficiency(
            METALLICITY_GRID, Z, w, masks=masks, mean_mass_evolved=_MEAN_MASS_PLACEHOLDER
        )
        channel_sum = sum(eff[c] for c in channel_masks)
        np.testing.assert_allclose(
            channel_sum,
            eff["__class_total__"],
            rtol=1e-12,
            atol=0,
            err_msg=f"channel eta(Z) curves do not sum to per-class total for {cl_label!r}",
        )


@pytest.mark.requires_data
def test_section_05b_bns_eta_channel_additivity(bns_a_path):
    bns = load_bns_with_channels(bns_a_path, expected_model="A", expected_ns_max=2.5)
    cls = classify_bns_2024(bns["m1"], bns["m2"])
    class_masks = {
        "sbGRB + blue KN": cls["sbGRB + blue KN"],
        "lbGRB (HMNS)": cls["lbGRB + red KN (HMNS)"],
        "lbGRB (disk)": cls["lbGRB + red KN (disk)"],
        "Faint lbGRB": cls["Faint lbGRB"],
    }
    _assert_channels_sum_to_class(bns["metallicity"], bns["weights"], class_masks, _channels(bns))


@pytest.mark.requires_data
def test_section_05b_bhns_eta_channel_additivity(bhns_a_path):
    bhns = load_bhns_with_channels(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    cbhns = classify_bhns(bhns["M_BH"], bhns["M_NS"], a_BH=0.5)
    class_masks = {
        "No GRB": cbhns["No GRB"],
        "Faint lbGRB (BHNS)": cbhns["Faint lbGRB (BHNS)"],
        "lbGRB + red KN (BHNS disk)": cbhns["lbGRB + red KN (BHNS disk)"],
    }
    _assert_channels_sum_to_class(
        bhns["metallicity"], bhns["weights"], class_masks, _channels(bhns)
    )


@pytest.mark.requires_data
def test_section_05b_bns_coarse_bin_additivity(bns_a_path):
    """The 15-bin coarse rebinning the notebook plots must stay additive."""
    bns = load_bns_with_channels(bns_a_path, expected_model="A", expected_ns_max=2.5)
    cls = classify_bns_2024(bns["m1"], bns["m2"])
    class_masks = {
        "sbGRB + blue KN": cls["sbGRB + blue KN"],
        "lbGRB (HMNS)": cls["lbGRB + red KN (HMNS)"],
        "lbGRB (disk)": cls["lbGRB + red KN (disk)"],
        "Faint lbGRB": cls["Faint lbGRB"],
    }
    _assert_coarse_channels_sum_to_class(
        bns["metallicity"], bns["weights"], class_masks, _channels(bns)
    )


@pytest.mark.requires_data
def test_section_05b_bhns_coarse_bin_additivity(bhns_a_path):
    bhns = load_bhns_with_channels(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    cbhns = classify_bhns(bhns["M_BH"], bhns["M_NS"], a_BH=0.5)
    class_masks = {
        "No GRB": cbhns["No GRB"],
        "Faint lbGRB (BHNS)": cbhns["Faint lbGRB (BHNS)"],
        "lbGRB + red KN (BHNS disk)": cbhns["lbGRB + red KN (BHNS disk)"],
    }
    _assert_coarse_channels_sum_to_class(
        bhns["metallicity"], bhns["weights"], class_masks, _channels(bhns)
    )
