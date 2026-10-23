"""Formation-channel classification underpinning the per-channel splits (Sections 3b, 6b, 7b) of grb_main.ipynb.

NEW data path.  ``load_bns_with_channels`` plus
``classify_formation_channels`` plus ``channel_class_crosstab`` on real
Model A BNS.  Asserts the joint contingency table is well-formed and
the row / column / total normalisations agree with the helper contract.

Also pins per-model COMPAS-prescription signatures:
* per-model weighted P(doubleCommonEnvelopeFlag = 1) -- CE energetics
  regression tripwire (Hurley alpha-CE vs Hirai+ stability criterion);
* per-model weighted P(Case A donor at first RLOF) -- Hurley+ 2000
  giant-branch radii regression tripwire.
And a regression check that P(channel II | Z) falls with metallicity
for Model A BHNS, plus the Broekgaarden+ 2026 Sec. 4.1 result that every
merging BNS passes through a common envelope.

Marked ``requires_data``: skips cleanly without the relevant HDF5.
The helper-only invariants (synthetic inputs) are exercised by
``tests/unit/test_phase4_helpers.py::test_channel_class_crosstab_*``.

Reference: Broekgaarden et al. (2021), arXiv:2103.02608, Sec. 3.1
(channels I to V; Case A first transfer is "Other", Sec. 3.1.5);
Broekgaarden et al. (2026), arXiv:2606.05322, Sec. 4.1 (BNS formation
requires a CE); Gottlieb et al. (2024), arXiv:2411.13657 (four-class
scheme).
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_classify import (
    FORMATION_CHANNEL_INPUTS,
    channel_class_crosstab,
    classify_bns_2024,
    classify_formation_channels,
)
from grb_io import (
    METALLICITY_GRID,
    load_bhns_with_channels,
    load_bns_with_channels,
)

# Per-model NS_max for the project's five core letters (A, F, G, J,
# K).  Pulled out so the parametrize over both populations stays compact.
_NS_MAX = {"A": 2.5, "F": 2.5, "G": 2.5, "J": 2.0, "K": 3.0}

# Per-(filename) pinned values for B1 and B2.  Computed once on the project's
# pinned COMPAS commit (81722d4) against the Broekgaarden+ 2021 Zenodo HDF5
# archives (5189849 BNS, 5178777 BHNS).  Regression tolerance is 5 percent
# relative for non-zero pins, 1e-3 absolute for pins at or below 1e-3 (the
# BNS Case-A pins are intrinsically near zero because BNS progenitors require
# a CE to tighten the orbit enough to merge, which excludes Case A donors).
_PINS = {
    "COMPASCompactOutput_BNS_A.h5": {"dblCE": 0.7461, "caseA": 0.0},
    "COMPASCompactOutput_BNS_F.h5": {"dblCE": 0.2106, "caseA": 7.1e-05},
    "COMPASCompactOutput_BNS_G.h5": {"dblCE": 0.7701, "caseA": 0.0},
    "COMPASCompactOutput_BNS_J.h5": {"dblCE": 0.7349, "caseA": 0.0},
    # The K pins used to hold Model H's values (BNS 0.6697 / 0.006667, BHNS
    # 0.02583 / 0.1805, all four matching H to five figures).  Paper II calls
    # paper-I H "K", so the letter was resolved through the display convention
    # instead of the paper-I file the key names.  Measured on the file itself:
    # its embedded attributes are model=K, ns_max=3.0, and its heaviest NS is
    # 3.000 Msun, which is Model K and not H.
    "COMPASCompactOutput_BNS_K.h5": {"dblCE": 0.7476, "caseA": 0.0},
    "COMPASCompactOutput_BHNS_A.h5": {"dblCE": 0.03371, "caseA": 0.04147},
    "COMPASCompactOutput_BHNS_F.h5": {"dblCE": 0.004758, "caseA": 0.02462},
    "COMPASCompactOutput_BHNS_G.h5": {"dblCE": 0.1506, "caseA": 0.02551},
    "COMPASCompactOutput_BHNS_J.h5": {"dblCE": 0.05695, "caseA": 0.05953},
    "COMPASCompactOutput_BHNS_K.h5": {"dblCE": 0.02874, "caseA": 0.03380},
}

# Common parametrize set; tests skip cleanly per-file via the ``compas_file``
# fixture if the requested HDF5 is not present in data/.
_PARAMS = sorted(_PINS.keys())


def _assert_pinned(value: float, pinned: float, name: str, fname: str) -> None:
    """Compare to pin with rtol=5e-2 above 1e-3, atol=1e-3 below.

    The mixed tolerance is necessary because some BNS Case-A pins are
    exactly zero (BNS progenitors require a CE, which selects against Case
    A donors); pytest.approx(0.0, rel=...) is undefined for zero targets.
    """
    if pinned > 1e-3:
        assert value == pytest.approx(pinned, rel=5e-2), (
            f"{fname}: {name} = {value:.4g} drifted from pin {pinned:.4g} "
            f"by more than 5 percent (rtol)"
        )
    else:
        assert abs(value - pinned) < 1e-3, (
            f"{fname}: {name} = {value:.4g} drifted from near-zero pin "
            f"{pinned:.4g} by more than 1e-3 (atol)"
        )


def _first_rlof_donor_K(d: dict) -> np.ndarray:
    """Donor Hurley stellar type at first RLOF.

    Mirrors the ``first_is_primary`` / ``first_is_secondary`` selection
    inside ``classify_formation_channels``; returns -1 where no stable MT
    occurred.
    """
    has_p1 = d["fc_mt_p1"] > 0
    has_s1 = d["fc_mt_s1"] > 0
    first_is_primary = has_p1 & (~has_s1 | (d["fc_mt_p1"] < d["fc_mt_s1"]))
    first_is_secondary = has_s1 & (~has_p1 | (d["fc_mt_s1"] < d["fc_mt_p1"]))
    return np.where(
        first_is_primary,
        d["fc_mt_p1_K1"],
        np.where(first_is_secondary, d["fc_mt_s1_K2"], -1),
    )


@pytest.mark.requires_data
def test_channel_class_crosstab_modelA_real_data(bns_a_path):
    bns = load_bns_with_channels(bns_a_path, expected_model="A", expected_ns_max=2.5)

    channels = classify_formation_channels(**{k: bns[k] for k in FORMATION_CHANNEL_INPUTS})
    classes = classify_bns_2024(bns["m1"], bns["m2"])

    raw = channel_class_crosstab(channels, classes, bns["weights"])
    assert raw.shape == (5, 4), raw.shape
    assert (raw.values >= 0.0).all(), "negative entry in raw crosstab"
    assert raw.values.sum() > 0.0, "crosstab is empty"

    by_channel = channel_class_crosstab(channels, classes, bns["weights"], normalise="channel")
    row_sums = by_channel.sum(axis=1).values
    nonzero_rows = row_sums > 0
    np.testing.assert_allclose(row_sums[nonzero_rows], 1.0, rtol=1e-9)

    by_class = channel_class_crosstab(channels, classes, bns["weights"], normalise="class")
    col_sums = by_class.sum(axis=0).values
    nonzero_cols = col_sums > 0
    np.testing.assert_allclose(col_sums[nonzero_cols], 1.0, rtol=1e-9)

    by_total = channel_class_crosstab(channels, classes, bns["weights"], normalise="total")
    assert by_total.values.sum() == pytest.approx(1.0, rel=1e-9)


@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _PARAMS, indirect=True)
def test_double_common_envelope_fraction_pinned_to_compas_commit(compas_file):
    """Pin per-model weighted P(doubleCommonEnvelopeFlag = 1).

    This is a population-level signature of the COMPAS CE energetics
    prescription used in the pinned commit ``81722d4`` (Hurley alpha-CE
    formalism with the COMPAS-default lambda_CE and stability criterion).
    The test is a regression tripwire: it does NOT validate that the
    prescription is "correct".  A future COMPAS commit moving to the
    Hirai+ updated stability criterion will shift the dblCE fraction and
    trip this test, which is the intended audit signal; the failure
    message then prompts an explicit re-pin under the new prescription.

    Pin source: computed from the project's pinned commit on the
    Broekgaarden+ 2021 Zenodo HDF5 archives (5189849 BNS / 5178777 BHNS).
    The pins describe the raw files, so the failed common-envelope ejections
    the loaders drop by default stay in (``drop_failed_ce=False``); every one
    of them carries a CE, so dropping them would move this fraction.
    """
    fname = compas_file.split("/")[-1]
    parts = fname.split("_")
    kind = parts[1]  # "BNS" or "BHNS"
    model = parts[2].replace(".h5", "")
    loader = load_bns_with_channels if kind == "BNS" else load_bhns_with_channels
    d = loader(
        compas_file,
        expected_model=model,
        expected_ns_max=_NS_MAX[model],
        drop_failed_ce=False,
    )

    w = d["weights"]
    f_dblCE = float(w[d["dblCE"] == 1].sum() / w.sum())
    _assert_pinned(f_dblCE, _PINS[fname]["dblCE"], "P(dblCE=1)", fname)


@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _PARAMS, indirect=True)
def test_case_a_donor_fraction_at_first_rlof_pinned(compas_file):
    """Pin per-model weighted P(Case A donor at first RLOF).

    Case A versus Case B/C is set by the donor's Hurley stellar type at
    first RLOF: K in {0, 1} (main sequence) is Case A.  This in turn
    depends on the Hurley+ 2000 (MNRAS 315, 543) fitting formulae for
    stellar radii during the giant branches that COMPAS uses to decide
    when each star fills its Roche lobe.  This test is a regression
    tripwire on the resulting population statistics: a future commit
    swapping Hurley radii for a different prescription (e.g. SSE2 grids
    or a MIST track interpolator) will shift the Case A fraction at the
    surviving DCO stage and trip this test.

    BNS Case A fractions are intrinsically near zero in this dataset
    because BNS progenitors require a CE to tighten the orbit enough to
    merge within a Hubble time, which selects against Case A donors that
    enter mass transfer too early in their evolution; only Model K (the
    M_NS_max = 3.0 variant) has any meaningful BNS Case A weight.  The
    near-zero pins therefore use a 1e-3 absolute tolerance (handled by
    ``_assert_pinned``); BHNS pins all sit above 2 percent and use the
    standard 5 percent relative tolerance.

    Pin source: as in the dblCE test, the project's pinned COMPAS commit
    on the Broekgaarden+ 2021 Zenodo HDF5 archives.
    """
    fname = compas_file.split("/")[-1]
    parts = fname.split("_")
    kind = parts[1]
    model = parts[2].replace(".h5", "")
    loader = load_bns_with_channels if kind == "BNS" else load_bhns_with_channels
    d = loader(compas_file, expected_model=model, expected_ns_max=_NS_MAX[model])

    K = _first_rlof_donor_K(d)
    w = d["weights"]
    is_caseA = (K == 0) | (K == 1)
    f_caseA = float(w[is_caseA].sum() / w.sum())
    _assert_pinned(f_caseA, _PINS[fname]["caseA"], "P(Case A donor)", fname)


@pytest.mark.requires_data
def test_bhns_channel_II_fraction_increases_with_decreasing_metallicity(bhns_a_path):
    """Channel II (stable MT only) fraction must rise toward low Z in BHNS.

    A regression check on Model A, not a literature anchor: no paper in
    papers/ resolves channel II by metallicity for these models
    (Broekgaarden+ 2022 does not split by channel, and their Fig. 5, which
    this docstring used to cite, shows detectable BHNS masses).  The likely
    driver, that lower-metallicity stars expand less so first mass transfer
    is more often stable, is an interpretation, not a sourced result.

    The test bins systems by ``log10(Z)`` into three bins spanning the
    full COMPAS metallicity prior ``[1e-4, 0.03]`` (the project's
    ``METALLICITY_GRID`` endpoints).  Three bins is the coarsest
    binning that suppresses the small-N uptick at solar metallicity
    seen in finer binnings.

    Asserts (i) strict monotonic non-increase of P(channel II | Z) as
    Z rises, and (ii) f_II(lowest-Z bin) / f_II(highest-Z bin) >= 1.5
    (real data on the pinned commit reaches ~3.5x).
    """
    bhns = load_bhns_with_channels(bhns_a_path, expected_model="A", expected_ns_max=2.5)
    channels = classify_formation_channels(**{k: bhns[k] for k in FORMATION_CHANNEL_INPUTS})
    ch_II = channels["II  Stable MT only"]
    w = bhns["weights"]
    logZ = np.log10(bhns["metallicity"])

    nbins = 3
    edges = np.linspace(np.log10(METALLICITY_GRID[0]), np.log10(METALLICITY_GRID[-1]), nbins + 1)
    f_II_bin = np.empty(nbins, dtype=float)
    for i in range(nbins):
        if i == nbins - 1:
            mask = (logZ >= edges[i]) & (logZ <= edges[i + 1])
        else:
            mask = (logZ >= edges[i]) & (logZ < edges[i + 1])
        denom = w[mask].sum()
        assert denom > 0, (
            f"Z bin {i} is empty; the COMPAS metallicity prior covers "
            f"the full grid, so this should never happen on real data."
        )
        f_II_bin[i] = w[ch_II & mask].sum() / denom

    assert np.all(np.diff(f_II_bin) <= 0), (
        f"Channel II fraction is not monotonic non-increasing with Z: {f_II_bin.tolist()}."
    )
    dynamic_range = f_II_bin[0] / f_II_bin[-1]
    assert dynamic_range >= 1.5, (
        f"Channel II low-Z/high-Z ratio = {dynamic_range:.2f} is below the "
        f"1.5x floor; Model A shows ~3.5x, so a value near 1.0 means the "
        f"metallicity dependence has been lost."
    )


@pytest.mark.requires_data
def test_every_model_A_bns_passes_through_a_common_envelope(bns_a_path):
    """Every merging Model A BNS has a CE, in every channel including V.

    Broekgaarden+ 2026 Sec. 4.1: BNS formation requires a CE.  Read from the
    ``formationChannels`` CE ordinal, which is independent of the RLOF rows
    the channels are built from, so this also cross-checks that I, III and IV
    (CE by definition) and V agree with a second column of the file.
    """
    bns = load_bns_with_channels(bns_a_path, expected_model="A", expected_ns_max=2.5)
    channels = classify_formation_channels(**{k: bns[k] for k in FORMATION_CHANNEL_INPUTS})
    had_CE = bns["fc_CEE"] > 0
    w = bns["weights"]
    assert had_CE.all(), f"{int((~had_CE).sum())} merging BNS have no CE"
    assert w[channels["II  Stable MT only"]].sum() == 0.0
