"""Hard runtime asserts in ``grb_classify``.

A typo such as ``ns_max=2.4`` or a caller passing both ``k_thresh`` and
an inconsistent ``m_thresh`` would silently misclassify the entire BNS
or BHNS boundary, or the prompt-collapse threshold; both are hard errors.
"""

import warnings

import numpy as np
import pytest

from grb_classify import NS_MAX_FIDUCIAL, _resolve_m_thresh, classify_grid


def test_resolve_m_thresh_consistent_pair_returns_derived():
    val = _resolve_m_thresh(m_tov=2.2, m_thresh=2.794, k_thresh=1.27)
    assert val == pytest.approx(2.794, rel=1e-6)


def test_resolve_m_thresh_inconsistent_pair_raises():
    with pytest.raises(ValueError, match="Inconsistent prompt-collapse"):
        _resolve_m_thresh(m_tov=2.2, m_thresh=2.9, k_thresh=1.3)


def test_resolve_m_thresh_only_k_thresh_returns_product():
    val = _resolve_m_thresh(m_tov=2.0, m_thresh=None, k_thresh=1.3)
    assert val == pytest.approx(2.6, rel=1e-9)


def test_resolve_m_thresh_only_m_thresh_passes_through():
    assert _resolve_m_thresh(m_tov=2.2, m_thresh=2.8, k_thresh=None) == 2.8


@pytest.mark.parametrize("ns_max", [2.0, 2.5, 3.0])
def test_classify_grid_accepts_fiducial_ns_max(ns_max):
    g = np.linspace(0.8, 5.0, 12)
    M1, M2 = np.meshgrid(g, g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress the R_1p4_km warning
        out = classify_grid(M1, M2, ns_max=ns_max)
    assert out.shape == M1.shape


def test_classify_grid_rejects_off_fiducial_ns_max():
    g = np.linspace(0.8, 5.0, 12)
    M1, M2 = np.meshgrid(g, g)
    with pytest.raises(ValueError, match="not in fiducial"):
        classify_grid(M1, M2, ns_max=2.4)


def test_classify_grid_strict_ns_max_false_allows_off_fiducial():
    g = np.linspace(0.8, 5.0, 12)
    M1, M2 = np.meshgrid(g, g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = classify_grid(M1, M2, ns_max=2.4, strict_ns_max=False)
    assert out.shape == M1.shape


def test_classify_grid_warns_on_implicit_R_1p4():
    g = np.linspace(0.8, 6.0, 14)
    M1, M2 = np.meshgrid(g, g)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        classify_grid(M1, M2, ns_max=2.5)
    msgs = [str(item.message) for item in w]
    assert any("R_1p4_km not specified" in m for m in msgs), msgs


def test_classify_grid_R_1p4_supplied_suppresses_warning():
    g = np.linspace(0.8, 6.0, 14)
    M1, M2 = np.meshgrid(g, g)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        classify_grid(M1, M2, ns_max=2.5, R_1p4_km=11.9)
    msgs = [str(item.message) for item in w]
    assert not any("R_1p4_km not specified" in m for m in msgs), msgs


def test_NS_MAX_FIDUCIAL_matches_broekgaarden_models():
    """Models J / A / K = 2.0 / 2.5 / 3.0 Msun (Broekgaarden+ 2021)."""
    assert NS_MAX_FIDUCIAL == (2.0, 2.5, 3.0)
