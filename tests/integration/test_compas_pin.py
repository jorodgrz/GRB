"""Sanity tests for the pinned ``compas_python_utils`` install.

``compas_python_utils`` is pinned in ``environment.yml`` to a single
commit; a STROOPWAFEL weight-extraction bug there would make every
rate in the manuscript systematically wrong.  The escalating defenses:

1. ``test_compas_python_utils_imports``: package imports cleanly.
2. ``test_compas_cosmic_integration_module_imports``: the cosmic-
   integration submodule is reachable.
3. ``test_find_metallicity_distribution_recovers_known_distribution``:
   call ``find_metallicity_distribution`` against the COMPAS Model A
   metallicity window and assert (a) shape, (b) finite + non-negative
   entries, (c) per-redshift integration sums to unity, (d) the peak
   metallicity shifts with z per Neijssel et al. (2019) Eq. 7.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


@pytest.mark.requires_compas
def test_compas_python_utils_imports():
    try:
        compas_pkg = importlib.import_module("compas_python_utils")
    except ModuleNotFoundError as exc:
        pytest.skip(f"compas_python_utils not installed: {exc}")
    assert compas_pkg is not None


@pytest.mark.requires_compas
def test_compas_cosmic_integration_module_imports():
    """The cosmic-integration submodule is what grb_rates depends on."""
    try:
        ci = importlib.import_module("compas_python_utils.cosmic_integration.FastCosmicIntegration")
    except ModuleNotFoundError as exc:
        pytest.skip(f"FastCosmicIntegration not importable: {exc}")
    # find_metallicity_distribution was the canonical STROOPWAFEL hook
    # used in the 2021 archive; its presence is a smoke check that the
    # pinned commit still ships the API we depend on.
    assert hasattr(ci, "find_metallicity_distribution") or hasattr(ci, "find_sfr"), (
        "FastCosmicIntegration is missing the canonical helpers"
    )


@pytest.mark.requires_compas
@pytest.mark.requires_data
def test_find_metallicity_distribution_recovers_known_distribution(bns_a_path):
    """Upstream regression: Neijssel (2019) shape + redshift evolution.

    Anchors ``compas_python_utils.find_metallicity_distribution`` to
    four invariants that any future commit bump must continue to
    satisfy:

    1. Return shape is ``(len(z), n_logZ_bins)``.
    2. All entries are finite and non-negative.
    3. ``dPdlogZ[i].sum() * step_logZ == 1`` per redshift (the upstream
       renormalises the log-skew-normal to this convention).
    4. The peak metallicity at z = 0 sits at ``mu_0 = 0.035`` (within a
       factor of two), and at z = 2 has shifted to roughly
       ``0.035 * 10**(-0.23 * 2) ~ 0.0123`` (within the same factor).
       This catches a silent breakage of the Neijssel mu(z) evolution.

    The COMPAS Model A HDF5 is used only to read the simulation's
    ``min_logZ_COMPAS`` and ``max_logZ_COMPAS`` window so the upstream
    function is exercised against the same metallicity range the
    manuscript pipeline uses.  The shape and norm invariants are
    package-internal and would not change with COMPAS data drift; the
    redshift-evolution check is the physics anchor.
    """
    try:
        ci = importlib.import_module("compas_python_utils.cosmic_integration.FastCosmicIntegration")
    except ModuleNotFoundError as exc:
        pytest.skip(f"FastCosmicIntegration not importable: {exc}")
    if not hasattr(ci, "find_metallicity_distribution"):
        pytest.skip("find_metallicity_distribution not in this build")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_grb")

    import h5py as h5

    with h5.File(bns_a_path, "r") as f:
        Z = f["systems"]["Metallicity1"][...].squeeze()
    Z_min, Z_max = float(Z.min()), float(Z.max())
    min_logZ_COMPAS = float(np.log(Z_min))
    max_logZ_COMPAS = float(np.log(Z_max))

    z_grid = np.array([0.0, 1.0, 2.0])
    step_logZ = 0.01
    dPdlogZ, metals, p_draw = ci.find_metallicity_distribution(
        z_grid,
        min_logZ_COMPAS,
        max_logZ_COMPAS,
        step_logZ=step_logZ,
    )
    dPdlogZ = np.asarray(dPdlogZ)
    metals = np.asarray(metals)

    assert dPdlogZ.shape[0] == z_grid.size, (
        f"dPdlogZ first axis {dPdlogZ.shape[0]} != len(z) {z_grid.size}"
    )
    assert dPdlogZ.shape[1] == metals.size, (
        f"dPdlogZ second axis {dPdlogZ.shape[1]} != metals length {metals.size}"
    )

    assert np.all(np.isfinite(dPdlogZ)), "non-finite entries in dPdlogZ"
    assert np.all(dPdlogZ >= 0), "negative entries in dPdlogZ"

    row_integrals = dPdlogZ.sum(axis=-1) * step_logZ
    np.testing.assert_allclose(
        row_integrals,
        np.ones_like(row_integrals),
        rtol=1e-6,
        err_msg=(
            "upstream find_metallicity_distribution row integrals "
            "deviate from 1; STROOPWAFEL normalisation has drifted"
        ),
    )

    peaks = metals[np.argmax(dPdlogZ, axis=-1)]
    peak_z0 = float(peaks[0])
    peak_z2 = float(peaks[2])
    expected_peak_z0 = 0.035
    expected_peak_z2 = 0.035 * 10.0 ** (-0.23 * 2.0)
    assert 0.5 * expected_peak_z0 < peak_z0 < 2.0 * expected_peak_z0, (
        f"z=0 metallicity peak {peak_z0:.4f} not within factor of 2 of "
        f"Neijssel mu_0 = {expected_peak_z0:.4f}"
    )
    assert 0.5 * expected_peak_z2 < peak_z2 < 2.0 * expected_peak_z2, (
        f"z=2 metallicity peak {peak_z2:.4f} not within factor of 2 of "
        f"Neijssel mu(z=2) = {expected_peak_z2:.4f}"
    )

    p_draw_expected = 1.0 / (max_logZ_COMPAS - min_logZ_COMPAS)
    assert p_draw == pytest.approx(p_draw_expected, rel=1e-9), (
        f"p_draw_metallicity {p_draw:.6e} != 1 / (max_logZ_COMPAS - "
        f"min_logZ_COMPAS) = {p_draw_expected:.6e}"
    )
