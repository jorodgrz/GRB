"""End-to-end regression test for the BHNS rate vs LVK cross-check (Section 7 of ``grb_main.ipynb``).

This compares the Model A intrinsic BHNS local merger rate density
``R_BHNS(z = 0)`` against the LVK GWTC-5.0 90 percent credible interval
(LVK 2026, *GWTC-5.0: Population Properties of Merging Compact
Binaries*, arXiv:2605.27226; published NSBH band 6.7 - 32.8 Gpc^-3 yr^-1).

Pins two end-to-end claims:

1. ``test_section_07c_R0_BHNS_above_LVK_GWTC5_band`` -- the calibrated
   Model A BHNS intrinsic rate sits ABOVE the GWTC-5.0 NSBH 90 percent
   CR.  GWTC-5.0 added no new NSBH candidates, so the NSBH band tightened
   from the GWTC-4 9.1 - 84 Gpc^-3 yr^-1 to 6.7 - 32.8 Gpc^-3 yr^-1.  The
   unsmoothed Model A All-BHNS intrinsic rate (84.5 Gpc^-3 yr^-1) now
   sits a factor of ~2.6 above the upper edge: the panel's headline
   finding is precisely this overprediction.  Reconciliation would
   require either a downward shift in the COMPAS BHNS formation rate or
   a re-examination of the Foucart+ 2018 disk-mass calibration at high
   mass ratio.
2. ``test_section_07c_intrinsic_grb_class_subtotal_at_most_all_bhns``
   -- the two intrinsic GRB-class rates (lbGRB + Faint lbGRB at
   fiducial spin) sum to AT MOST the All-BHNS intrinsic rate (with a
   small numerical slack), since the No-GRB / NS-swallowed class
   contributes positively.  This is a one-sided sanity check, not an
   equality: per-class rates and the All-BHNS total are independent
   ``compute_merger_rate`` calls so they do not algebraically coincide
   even before the No-GRB class is subtracted.

Both tests reuse the shared ``_model_cache.get_model("A")`` so the
expensive per-model load + cosmic-integration calibration runs at most
once per session across Sections 7b, 8c, and 12.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grb_rates import LVK_GWTC5_LOCAL_RATES, compute_merger_rate  # noqa: E402
from tests.sections._model_cache import get_model as _get_model  # noqa: E402

# Fiducial BH spin for the Section 7 per-class panel, matching the
# Section 0 notebook constant ``A_BH_FID = 0.5``.  Sits at the centre of
# the Foucart+ 2018 calibration band [-0.5, 0.9].
A_BH_FID = 0.5


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_R0_BHNS_above_LVK_GWTC5_band():
    """Model A R_BHNS(z = 0) intrinsic sits above the GWTC-5.0 NSBH band.

    GWTC-5.0 (LVK 2026, arXiv:2605.27226) reports the NSBH local
    intrinsic merger rate density at 6.7 - 32.8 Gpc^-3 yr^-1 (90 percent
    CR); no new NSBH candidates were added relative to GWTC-4, so the
    band tightened from 9.1 - 84.  The unsmoothed Model A All-BHNS
    intrinsic rate (no misalignment, no beaming) is 84.5 Gpc^-3 yr^-1, a
    factor of ~2.6 above the upper edge: the panel's headline finding is
    this overprediction.  The upper ceiling of 100 below is a smoothing
    guard: a smoothed-R(z=0) quote-path regression reproduces the
    inflated ~115 Gpc^-3 yr^-1 (the gaussian_filter1d reflective-boundary
    artifact) and must fail this test.

    Source of truth: ``grb_rates.LVK_GWTC5_LOCAL_RATES['NSBH']``.
    """
    mod = _get_model("A")
    R0 = mod["R0_bhns"]
    lo = LVK_GWTC5_LOCAL_RATES["NSBH"]["R_lo"]
    hi = LVK_GWTC5_LOCAL_RATES["NSBH"]["R_hi"]
    assert hi < R0 < 100.0, (
        f"Model A R_BHNS(z=0) intrinsic = {R0:.2f} Gpc^-3 yr^-1 is not in "
        f"the expected overprediction range ({hi}, 100) above the GWTC-5.0 "
        f"NSBH 90% CR [{lo}, {hi}] (LVK 2026, arXiv:2605.27226).  A value "
        f"near 115 indicates the smoothed-R(z=0) quote-path regression; a "
        f"value inside the band would mean the BHNS overprediction is gone."
    )


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07c_intrinsic_grb_class_subtotal_at_most_all_bhns():
    """Sum of intrinsic GRB-class rates does not exceed the All-BHNS total.

    The Section 7 right panel shows lbGRB + Faint lbGRB at fiducial
    spin ``A_BH_FID = 0.5``; the All-BHNS bar additionally includes
    the No-GRB / NS-swallowed class.  Per-class and total rates are
    independent ``compute_merger_rate`` calls (each over its own
    sample subset) so they are not algebraically constrained to
    coincide, but the GRB-class subtotal cannot exceed the All-BHNS
    total within a few percent of numerical headroom.

    A regression that broke ``classify_bhns`` boolean exclusivity, or
    that accidentally double-counted systems via overlapping masks,
    would trip this test before the figure is generated.
    """
    from grb_classify import classify_bhns

    mod = _get_model("A")
    bhns = mod["bhns"]
    iz0 = mod["iz0"]

    cbh = classify_bhns(bhns["M_BH"], bhns["M_NS"], a_BH=A_BH_FID, clip_chi=0.9)

    R_classes = []
    for label in ("lbGRB + red KN (BHNS disk)", "Faint lbGRB (BHNS)"):
        mask = cbh[label]
        R_cls = compute_merger_rate(
            mod["redshifts"],
            mod["times"],
            mod["time_first_SF"],
            mod["sfr"] / mod["mme_bhns"],
            mod["p_draw"],
            mod["dPdlogZ"],
            mod["metallicities"],
            bhns["metallicity"][mask],
            bhns["delay_time"][mask],
            bhns["weights"][mask],
            smooth_sigma=0,
        )
        R_classes.append(float(R_cls[iz0]))

    R_grb_subtotal = float(np.sum(R_classes))
    R_total = float(mod["R_bhns_z"][iz0])

    # 1 percent numerical slack absorbs any cosmic-integration round-off
    # from running compute_merger_rate three times (whole sample plus two
    # subsets); the physical relation is R_grb_subtotal <= R_total.
    assert R_grb_subtotal <= R_total * 1.01, (
        f"Sum of intrinsic GRB-class BHNS R(z=0) = {R_grb_subtotal:.4e} "
        f"exceeds All-BHNS intrinsic total {R_total:.4e} (Model A); "
        f"classify_bhns class masks may have lost mutual exclusivity."
    )
