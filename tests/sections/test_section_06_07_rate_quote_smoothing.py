"""Quoted R(z=0) values must come from unsmoothed rate arrays.

The ``compute_merger_rate`` display kernel (``smooth_sigma=30``,
``gaussian_filter1d`` with the default reflective boundary) folds the
rising side of R(z) back onto z = 0 and inflates the smoothed local
rate by ~30 percent at production sample sizes (Model A: BNS 55 -> 73,
BHNS 84 -> 115 Gpc^-3 yr^-1).  The calibration anchor was already
guarded (``calibrate_mean_mass_evolved`` passes ``smooth_sigma=0``);
this module pins the same contract for every notebook cell that prints
or tabulates an R(z=0) value, so the quoted local rates and the
Appendix per-model bars (Section 9, already ``smooth_sigma=0``)
cannot drift apart again.

Source-level checks on ``grb_main.ipynb``; no data files needed.
"""

from __future__ import annotations

import json
import os
import re

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_NB_PATH = os.path.join(_REPO_ROOT, "grb_main.ipynb")

_CALL = re.compile(r"compute_merger_rate\([^()]*\)")


def _code_cells():
    with open(_NB_PATH) as fh:
        nb = json.load(fh)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            yield i, "".join(cell["source"])


@pytest.mark.unit
def test_rate_quoting_cells_compute_unsmoothed():
    """Every cell that reads R[...iz0...] computes with smooth_sigma=0.

    A cell that both calls ``compute_merger_rate`` and indexes a rate
    array at ``iz0`` is a quote path: its z = 0 values reach printed
    summaries, Table 4/5 of the manuscript, or the LVK comparison
    panels.  Each such call must pass ``smooth_sigma=0`` explicitly;
    cosmetic smoothing belongs on display copies via
    ``smooth_rate_curve``.
    """
    offenders = []
    for i, src in _code_cells():
        if "compute_merger_rate(" not in src or "iz0" not in src:
            continue
        for call in _CALL.findall(src):
            if "smooth_sigma=0" not in call:
                offenders.append((i, " ".join(call.split())[:90]))
    assert not offenders, (
        "compute_merger_rate calls on an R(z=0) quote path without "
        f"smooth_sigma=0 (cell, call): {offenders}.  The sigma=30 kernel's "
        "reflective boundary inflates the smoothed R(z=0) by ~30 percent; "
        "smooth a display copy with grb_rates.smooth_rate_curve instead."
    )


@pytest.mark.unit
def test_quote_cells_do_not_index_smoothed_curves_at_iz0():
    """No quote cell reads smooth_rate_curve(...)[iz0]."""
    pattern = re.compile(r"smooth_rate_curve\([^()]*\)\s*\[\s*iz0", re.S)
    offenders = [i for i, src in _code_cells() if pattern.search(src)]
    assert not offenders, (
        f"cells {offenders} index a display-smoothed curve at iz0; quote "
        "R(z=0) from the unsmoothed array."
    )
