"""Integration tests for ``comparison.ipynb`` data plumbing.

Loads ``data/rastinejad_2024.csv`` via the shared ``rastinejad_csv_path``
fixture and asserts the column schema and the ``classify_observed_mergers``
contract that the comparison notebook consumes.

Reference: Rastinejad et al. (2024), ApJ 979, 190 (kilonova ejecta
decompositions); Gottlieb et al. (2024), arXiv:2411.13657 (four-class
scheme used to overlay observed-sample fractions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grb_classify import classify_observed_mergers

REQUIRED_COLUMNS = (
    "name",
    "M_B_med",
    "M_P_med",
    "M_R_med",
)


@pytest.mark.requires_data
def test_rastinejad_csv_has_required_columns(rastinejad_csv_path):
    df = pd.read_csv(rastinejad_csv_path, comment="#", skipinitialspace=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, (
        f"rastinejad_2024.csv missing required columns: {missing}; available={list(df.columns)}"
    )
    assert len(df) >= 5, f"expected at least 5 rows; got {len(df)}"


@pytest.mark.requires_data
def test_classify_observed_mergers_partitions_rastinejad_sample(
    rastinejad_csv_path,
):
    """Every Rastinejad row maps to exactly one Gottlieb (2024) class
    label.  Catches drift between ``classify_observed_mergers`` and the
    CSV ejecta-component conventions."""
    df = pd.read_csv(rastinejad_csv_path, comment="#", skipinitialspace=True)
    cls = classify_observed_mergers(
        df["M_B_med"].values,
        df["M_P_med"].values,
        df["M_R_med"].values,
    )

    label_keys = [
        "sbGRB + blue KN",
        "lbGRB + red KN (HMNS)",
        "lbGRB + red KN (disk)",
        "Faint lbGRB",
    ]
    stack = np.stack([np.asarray(cls[k], dtype=bool) for k in label_keys])
    overlap = stack.sum(axis=0)
    assert (overlap == 1).all(), (
        f"each row should map to exactly one class; counts={np.bincount(overlap)}"
    )
