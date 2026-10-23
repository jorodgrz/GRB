"""Section 8c of grb_main.ipynb: Section 8a's kinematics across the 20 models.

Free drift needs no orbit integration, so unlike Section 8 it can be run over
the whole Broekgaarden+ 2021/2022 grid.  What the scan has to get right is the
bookkeeping around the edges: two models leave a GRB class unpopulated, one has
an effective sample size of 43, and one breaks Section 8a's claim that the
first leg is negligible.  Each of those is a result rather than a defect, so
each is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import classify_bhns, classify_bns_2024
from grb_io import (
    ALL_MODEL_SUFFIXES,
    filter_systems,
    load_bhns_with_kicks,
    load_bns_with_kicks,
)
from grb_offsets import ballistic_offset
from grb_physics import A_BH_FID, M_TOV, cold_stable_bhns_mask, cold_stable_ns_mask


def _wmedian(x, w, keep_zeros=False):
    ok = np.isfinite(x) & np.isfinite(w)
    if not keep_zeros:
        ok &= x > 0
    x, w = x[ok], w[ok]
    if x.size == 0 or w.sum() <= 0:
        return np.nan
    o = np.argsort(x)
    return float(np.interp(0.5, np.cumsum(w[o]) / w[o].sum(), x[o]))


def _bns(suffix):
    d = load_bns_with_kicks(path=f"data/COMPASCompactOutput_BNS_{suffix}.h5", expected_model=suffix)
    return filter_systems(d, cold_stable_ns_mask(d["m1"], d["m2"], m_tov=M_TOV))


def test_a_weighted_median_of_a_fraction_must_keep_its_zeros():
    """The leg-1 share is a bounded fraction, not a distance on a log axis.

    Section 8a's weighted-median helper drops non-positive values, which is
    right for an offset and wrong for a ratio that can legitimately be zero.
    On model F 11.6 percent of the weight has a first leg contributing nothing,
    and discarding it moves the reported share from 75.5 to 80.3 percent.
    """
    x = np.array([0.0, 0.0, 0.1, 0.9, 1.0])
    w = np.ones_like(x)
    assert _wmedian(x, w, keep_zeros=True) < _wmedian(x, w, keep_zeros=False)


@pytest.mark.requires_data
@pytest.mark.slow
def test_first_leg_is_negligible_on_the_fiducial_and_dominant_on_model_EH():
    """Section 8a's "the first leg is negligible" is a Model A statement.

    Unstable case BB with optimistic common envelope leaves binaries so tight
    that most of the weight merges within 0.1 Myr, so the SN1-to-SN2 interval
    becomes most of the drift.  This is what justifies keeping the two-leg
    treatment even though the fiducial does not need it.
    """
    shares = {}
    for suffix in ("A", "EH"):
        d = _bns(suffix)
        m = classify_bns_2024(d["m1"], d["m2"])["sbGRB + blue KN"]
        w = d["weights"][m]
        leg1 = (d["t_sn2"] - d["t_sn1"])[m]
        d_free = ballistic_offset(d["v_sys"][m], d["t_inspiral"][m], d["v_sys_1"][m], leg1)
        d_leg1 = ballistic_offset(d["v_sys_1"][m], leg1)
        shares[suffix] = 100.0 * _wmedian(d_leg1 / np.maximum(d_free, 1e-30), w, keep_zeros=True)
    assert shares["A"] < 1.0, shares
    assert shares["EH"] > 50.0, shares


@pytest.mark.requires_data
@pytest.mark.slow
def test_two_models_leave_the_bhns_disk_class_empty():
    """An unpopulated class is a property of the model, not a scan failure.

    Section 8c draws those as a missing marker rather than a point at zero, so
    the scan must be able to return NaN for a class with no systems.
    """
    empty = []
    for suffix in ALL_MODEL_SUFFIXES:
        d = load_bhns_with_kicks(
            path=f"data/COMPASCompactOutput_BHNS_{suffix}.h5", expected_model=suffix
        )
        d = filter_systems(d, cold_stable_bhns_mask(d["M_BH"], d["M_NS"], m_tov=M_TOV))
        masks = classify_bhns(d["M_BH"], d["M_NS"], a_BH=A_BH_FID)
        if masks["lbGRB + red KN (BHNS disk)"].sum() == 0:
            empty.append(suffix)
    assert set(empty) == {"E", "I"}, empty


@pytest.mark.requires_data
@pytest.mark.slow
def test_grid_moves_the_drift_time_far_more_than_the_kick():
    """The section's headline: a factor of 3 in speed against 238 in time.

    If this ever inverts, the claim that the common-envelope and mass-transfer
    families dominate the free-drift spread while the kick families do not has
    to be rewritten.
    """
    v, t = [], []
    for suffix in ALL_MODEL_SUFFIXES:
        d = _bns(suffix)
        m = classify_bns_2024(d["m1"], d["m2"])["sbGRB + blue KN"]
        w = d["weights"][m]
        v.append(_wmedian(d["v_sys"][m], w))
        t.append(_wmedian((d["t_sn2"] - d["t_sn1"] + d["t_inspiral"])[m], w))
    v, t = np.array(v), np.array(t)
    assert (t.max() / t.min()) > 20.0 * (v.max() / v.min()), (v.max() / v.min(), t.max() / t.min())
