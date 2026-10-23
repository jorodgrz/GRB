"""Section 3b of grb_main.ipynb: delay-time PDF per GRB class, split by channel.

The figure normalises each per-channel weighted PDF by the class in-range
weight, so within every class the populated channels must reconstruct the
per-class total curve.  These tests pin that within-class additivity on the
live Model A pipeline plus the underlying mask-partition fact, mirroring the
in-cell construction in [grb_main.ipynb] Section 3b.

Marked ``requires_data``: skips cleanly without the Model A HDF5s.  The
weighted-PDF helper itself is unit-tested in
[tests/unit](tests/unit); this file checks the notebook-level composition.

References: Gottlieb et al. (2024), arXiv:2411.13657 (four-class scheme);
Broekgaarden et al. (2021), arXiv:2103.02608, Sec. 3.1 (channels I-V).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import (
    FORMATION_CHANNEL_INPUTS,
    classify_bhns,
    classify_bns_2024,
    classify_formation_channels,
)
from grb_io import load_bhns_with_channels, load_bns_with_channels
from grb_plot_style import weighted_hist_pdf

# Same bin grid as the Section 3 / 3b cells.
_BINS_DELAY = np.logspace(np.log10(3.0), np.log10(14000), 30)


def _channels(d: dict) -> dict:
    return classify_formation_channels(**{k: d[k] for k in FORMATION_CHANNEL_INPUTS})


def _assert_within_class_additivity(data, w, class_masks, channel_masks):
    in_range = (data >= _BINS_DELAY[0]) & (data <= _BINS_DELAY[-1])
    for cl_label, cl_mask in class_masks.items():
        w_total = w[cl_mask & in_range].sum()
        if w_total <= 0:
            continue

        # Mask partition: channel weights within the class sum to the class.
        ch_weight_sum = sum(w[cl_mask & cm].sum() for cm in channel_masks.values())
        np.testing.assert_allclose(
            ch_weight_sum,
            w[cl_mask].sum(),
            rtol=1e-12,
            atol=0,
            err_msg=f"channel masks do not partition class {cl_label!r}",
        )

        # Per-channel PDFs (shared w_total) sum to the per-class PDF bin by bin.
        channel_sum = np.zeros(len(_BINS_DELAY) - 1)
        for cm in channel_masks.values():
            h = weighted_hist_pdf(data, cl_mask & cm, w, _BINS_DELAY, w_total)
            if h is not None:
                channel_sum += h[1]
        h_tot = weighted_hist_pdf(data, cl_mask, w, _BINS_DELAY, w_total)
        assert h_tot is not None, f"class {cl_label!r} is empty"
        np.testing.assert_allclose(
            channel_sum,
            h_tot[1],
            rtol=1e-12,
            atol=0,
            err_msg=f"channel PDFs do not sum to per-class PDF for {cl_label!r}",
        )


@pytest.mark.requires_data
def test_section_03b_bns_within_class_additivity(bns_a_path):
    bns = load_bns_with_channels(bns_a_path, expected_model="A", expected_ns_max=2.5)
    cls = classify_bns_2024(bns["m1"], bns["m2"])
    class_masks = {
        "sbGRB + blue KN": cls["sbGRB + blue KN"],
        "lbGRB (HMNS)": cls["lbGRB + red KN (HMNS)"],
        "lbGRB (disk)": cls["lbGRB + red KN (disk)"],
        "Faint lbGRB": cls["Faint lbGRB"],
    }
    _assert_within_class_additivity(bns["delay_time"], bns["weights"], class_masks, _channels(bns))


@pytest.mark.requires_data
def test_section_03b_bhns_within_class_additivity(bhns_a_path):
    bhns = load_bhns_with_channels(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    cbhns = classify_bhns(bhns["M_BH"], bhns["M_NS"], a_BH=0.5)
    class_masks = {
        "No GRB": cbhns["No GRB"],
        "Faint lbGRB (BHNS)": cbhns["Faint lbGRB (BHNS)"],
        "lbGRB + red KN (BHNS disk)": cbhns["lbGRB + red KN (BHNS disk)"],
    }
    _assert_within_class_additivity(
        bhns["delay_time"], bhns["weights"], class_masks, _channels(bhns)
    )


@pytest.mark.requires_data
@pytest.mark.parametrize("population", ["BNS", "BHNS"])
def test_section_03b_channel_iv_is_the_compas_double_core_flag(population, bns_a_path, bhns_a_path):
    """Channel IV and COMPAS's own doubleCommonEnvelopeFlag select the same systems.

    Broekgaarden+ 2021 define IV by the stellar types at the first RLOF (donor
    and companion both giants, unstable).  COMPAS sets the flag independently
    at CE onset, so the two must agree.  On Model A every IV system carries the
    flag and 99.96 (BNS) and 100 (BHNS) percent of flagged weight lands in IV.
    """
    if population == "BNS":
        d = load_bns_with_channels(bns_a_path, expected_model="A", expected_ns_max=2.5)
    else:
        d = load_bhns_with_channels(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    iv = _channels(d)["IV  Double-core CE"]
    flag = np.asarray(d["dblCE"]).astype(bool)
    w = d["weights"]
    assert w[iv & ~flag].sum() == 0.0
    assert w[iv & flag].sum() / w[flag].sum() > 0.999
