"""End-to-end regression test for Section 7b of ``grb_main.ipynb``.

Section 7b compares the Model A intrinsic BNS local merger rate density
``R_BNS(z = 0)`` from Section 7 against the LVK GWTC-5.0 90 percent
credible interval (LVK 2026, *GWTC-5.0: Population Properties of
Merging Compact Binaries*, arXiv:2605.27226; published BNS band 5.1 -
154.7 Gpc^-3 yr^-1, joint union of the PixelPop and FullPop models).

Pins two end-to-end claims:

1. ``test_section_07b_R0_BNS_inside_LVK_GWTC5_band`` -- the calibrated
   Model A BNS rate sits inside the GWTC-5.0 BNS 90 percent CR.
2. ``test_section_07b_per_class_rates_sum_to_total`` -- the four
   Gottlieb (2024) per-class intrinsic rates sum to the All-BNS total
   at ``rtol = 1e-12``; mirrors the in-notebook additivity tripwire so
   any future regression in ``classify_bns_2024`` or
   ``compute_merger_rate`` fails the test rather than silently
   distorting the figure.

Both reuse the shared ``_model_cache.get_model("A")`` so the expensive
per-model load + cosmic-integration calibration runs at most once per
session across Sections 7b, 8c, and 12.
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


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07b_R0_BNS_inside_LVK_GWTC5_band():
    """Model A R_BNS(z = 0) sits inside the GWTC-5.0 BNS 90 percent CR.

    GWTC-5.0 (LVK 2026, arXiv:2605.27226) reports the BNS local
    intrinsic merger rate density at 5.1 - 154.7 Gpc^-3 yr^-1 (90 percent
    CR, joint union of PixelPop and FullPop; 267 candidates through O4b,
    no new BNS).  The Section 7 panel reports the calibrated Model A
    All-BNS rate; this test pins the panel's headline claim that the
    prediction sits inside the published band.

    Source of truth: ``grb_rates.LVK_GWTC5_LOCAL_RATES['BNS']``.
    """
    mod = _get_model("A")
    R0 = mod["R0_bns"]
    lo = LVK_GWTC5_LOCAL_RATES["BNS"]["R_lo"]
    hi = LVK_GWTC5_LOCAL_RATES["BNS"]["R_hi"]
    assert lo <= R0 <= hi, (
        f"Model A R_BNS(z=0) = {R0:.2f} Gpc^-3 yr^-1 falls outside the "
        f"GWTC-5.0 BNS 90% CR [{lo}, {hi}] (LVK 2026, arXiv:2605.27226)."
    )


@pytest.mark.requires_data
@pytest.mark.requires_compas
@pytest.mark.slow
def test_section_07b_per_class_rates_sum_to_total():
    """Per-Gottlieb-2024 BNS class rates sum to the All-BNS total at z = 0.

    Mirrors the in-notebook ``assert np.allclose(...)`` from Section 7
    on the live cached data: ``classify_bns_2024`` exhaustively
    partitions the BNS sample into the four Gottlieb (2024) classes,
    and ``compute_merger_rate`` is linear in the sample, so the
    per-class rates must sum to the total at machine precision.  A
    regression in either function (or in the Alsing remap that runs
    upstream) would surface as a sub-1e-12 mismatch here.
    """
    from grb_classify import classify_bns_2024

    mod = _get_model("A")
    bns = mod["bns"]
    cls = classify_bns_2024(bns["m1"], bns["m2"])
    iz0 = mod["iz0"]

    R_total = float(mod["R_bns_z"][iz0])

    R_classes = []
    class_labels = [
        "sbGRB + blue KN",
        "lbGRB + red KN (HMNS)",
        "lbGRB + red KN (disk)",
        "Faint lbGRB",
    ]
    for label in class_labels:
        mask = cls[label]
        R_cls = compute_merger_rate(
            mod["redshifts"],
            mod["times"],
            mod["time_first_SF"],
            mod["sfr"] / mod["mme_bns"],
            mod["p_draw"],
            mod["dPdlogZ"],
            mod["metallicities"],
            bns["metallicity"][mask],
            bns["delay_time"][mask],
            bns["weights"][mask],
            smooth_sigma=0,
        )
        R_classes.append(float(R_cls[iz0]))

    R_sum = float(np.sum(R_classes))
    assert np.isclose(R_sum, R_total, rtol=1e-12, atol=0), (
        f"Per-class BNS R(z=0) sum {R_sum:.6e} does not equal All-BNS "
        f"{R_total:.6e}; the Gottlieb (2024) four-class partition is no "
        f"longer additive on production data."
    )
