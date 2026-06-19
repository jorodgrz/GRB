"""Unit tests for ``grb_io.py``.

Covers the defensive-perimeter checks: NaN-weight guard, returned-dict
shape consistency, ``'population'`` tag, and the unmatched-seed
warning in ``_match_sn_to_dco``.

Fixtures build tiny synthetic HDF5 files mirroring the Broekgaarden
et al. (2021) COMPAS column layout so the suite is runnable in CI
without the multi-GB Zenodo archives; the real-data audit lives in
``tests/integration/test_grb_io_realdata.py``.
"""

from __future__ import annotations

import warnings

import h5py as h5
import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────
# Synthetic-HDF5 builders
# ─────────────────────────────────────────────────────────────────────
_DCO_COLS_BASE = (
    "M1",
    "M2",
    "weight",
    "Metallicity1",
    "mergesInHubbleTimeFlag",
    "tc",
    "tform",
)


def _write_dco_group(
    f, *, n_total=6, n_merging=4, st1_values=None, weights=None, masses=None, extra_cols=None
):
    """Write a minimal ``doubleCompactObjects`` group.

    Parameters
    ----------
    n_total : int
        Catalogue size before the merging mask.
    n_merging : int
        Number of rows with mergesInHubbleTimeFlag == 1; these are the
        first ``n_merging`` entries.
    st1_values : array-like, optional
        Values for ``stellarType1`` (defaults to all 14 = BH).
    weights : array-like, optional
        STROOPWAFEL weights (defaults to ``np.linspace(0.1, 1.0, n_total)``).
    masses : (M1, M2) tuple, optional
        Component-mass arrays of length ``n_total``.
    extra_cols : dict, optional
        Additional column name -> array, length ``n_total``.
    """
    dco = f.create_group("doubleCompactObjects")

    if masses is None:
        m1 = np.linspace(2.0, 8.0, n_total)
        m2 = np.linspace(1.2, 1.5, n_total)
    else:
        m1, m2 = masses

    if weights is None:
        weights = np.linspace(0.1, 1.0, n_total)
    if st1_values is None:
        st1_values = np.full(n_total, 14, dtype=int)

    mh = np.zeros(n_total, dtype=int)
    mh[:n_merging] = 1

    tc = np.linspace(10.0, 1000.0, n_total)
    tform = np.linspace(5.0, 50.0, n_total)
    Z = np.full(n_total, 0.0142)

    cols = {
        "M1": m1,
        "M2": m2,
        "weight": weights,
        "Metallicity1": Z,
        "mergesInHubbleTimeFlag": mh,
        "tc": tc,
        "tform": tform,
        "stellarType1": st1_values,
    }
    if extra_cols:
        cols.update(extra_cols)
    for name, arr in cols.items():
        dco.create_dataset(name, data=np.asarray(arr).reshape(-1, 1))
    return dco, n_merging


def _write_metadata(f, *, kind, model="A", ns_max=2.5):
    """Stamp the metadata that ``_validate_hdf5_metadata`` consumes."""
    f.attrs["kind"] = kind
    f.attrs["model"] = model
    f.attrs["ns_max"] = float(ns_max)


@pytest.fixture
def synthetic_bns_path(tmp_path):
    """Tiny BNS HDF5 file: 6 systems, 4 merging."""
    path = tmp_path / "bns_tiny.h5"
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model="A", ns_max=2.5)
        _write_dco_group(f, n_total=6, n_merging=4)
    return str(path)


@pytest.fixture
def synthetic_bhns_path(tmp_path):
    """Tiny BHNS HDF5 file with mixed stellarType1 ordering."""
    path = tmp_path / "bhns_tiny.h5"
    n_total = 6
    n_merging = 4
    # Mix stellarType1 so that M_BH/M_NS routing is non-trivial:
    # rows 0,2 have BH as M1; rows 1,3 have BH as M2.  Restrict
    # within the merging slice (first n_merging rows).
    st1 = np.array([14, 13, 14, 13, 14, 14], dtype=int)
    m1 = np.array([8.0, 1.4, 6.5, 1.3, 7.0, 9.0])
    m2 = np.array([1.4, 7.0, 1.3, 6.0, 1.4, 1.5])
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BHNS", model="A", ns_max=2.5)
        _write_dco_group(f, n_total=n_total, n_merging=n_merging, st1_values=st1, masses=(m1, m2))
    return str(path), n_merging, st1, m1, m2


@pytest.fixture
def synthetic_kicks_bns_path(tmp_path):
    """Tiny BNS kicks HDF5 with supernovae group; every DCO seed matched."""
    path = tmp_path / "bns_kicks_tiny.h5"
    n_total = 4
    n_merging = 3
    seeds = np.arange(100, 100 + n_total, dtype=np.int64)
    sn_seeds = np.repeat(seeds, 2)
    sn_time = np.tile([1.0, 2.0], n_total)
    sn_vsys = np.array(
        [
            10.0,
            110.0,
            20.0,
            220.0,
            30.0,
            330.0,
            40.0,
            440.0,
        ]
    )
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model="A", ns_max=2.5)
        _write_dco_group(
            f,
            n_total=n_total,
            n_merging=n_merging,
            extra_cols={
                "drawnKick1": np.full(n_total, 50.0),
                "drawnKick2": np.full(n_total, 60.0),
                "separationDCOFormation": np.full(n_total, 1e10),
                "eccentricityDCOFormation": np.full(n_total, 0.3),
                "seed": seeds,
            },
        )
        sn = f.create_group("supernovae")
        sn.create_dataset("randomSeed", data=sn_seeds.reshape(-1, 1))
        sn.create_dataset("systemicVelocity", data=sn_vsys.reshape(-1, 1))
        sn.create_dataset("time", data=sn_time.reshape(-1, 1))

    expected_last_vsys = np.array([110.0, 220.0, 330.0, 440.0])
    return str(path), seeds, expected_last_vsys, n_merging


@pytest.fixture
def synthetic_kicks_unmatched_path(tmp_path):
    """BNS kicks file where one DCO seed has no SN entry."""
    path = tmp_path / "bns_kicks_unmatched.h5"
    n_total = 3
    n_merging = 3
    dco_seeds = np.array([100, 101, 999], dtype=np.int64)
    # Only seeds 100 and 101 appear in supernovae; 999 is unmatched.
    sn_seeds = np.array([100, 100, 101, 101], dtype=np.int64)
    sn_time = np.array([1.0, 2.0, 1.0, 2.0])
    sn_vsys = np.array([10.0, 111.0, 20.0, 222.0])
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model="A", ns_max=2.5)
        _write_dco_group(
            f,
            n_total=n_total,
            n_merging=n_merging,
            extra_cols={
                "drawnKick1": np.full(n_total, 50.0),
                "drawnKick2": np.full(n_total, 60.0),
                "separationDCOFormation": np.full(n_total, 1e10),
                "eccentricityDCOFormation": np.full(n_total, 0.3),
                "seed": dco_seeds,
            },
        )
        sn = f.create_group("supernovae")
        sn.create_dataset("randomSeed", data=sn_seeds.reshape(-1, 1))
        sn.create_dataset("systemicVelocity", data=sn_vsys.reshape(-1, 1))
        sn.create_dataset("time", data=sn_time.reshape(-1, 1))
    return str(path)


# ─────────────────────────────────────────────────────────────────────
# Population tag
# ─────────────────────────────────────────────────────────────────────
def test_load_bns_returns_population_tag(synthetic_bns_path):
    from grb_io import load_bns

    out = load_bns(path=synthetic_bns_path)
    assert out["population"] == "BNS"


def test_load_bhns_returns_population_tag(synthetic_bhns_path):
    from grb_io import load_bhns

    path, *_ = synthetic_bhns_path
    out = load_bhns(path=path)
    assert out["population"] == "BHNS"


# ─────────────────────────────────────────────────────────────────────
# Returned-dict shape consistency
# ─────────────────────────────────────────────────────────────────────
def test_load_bns_dict_shape_consistent(synthetic_bns_path):
    from grb_io import load_bns

    out = load_bns(path=synthetic_bns_path)
    n = out["n_merging"]
    assert n == 4
    for k, v in out.items():
        if isinstance(v, np.ndarray) and k != "mask_merging":
            assert len(v) == n, f"{k!r} has length {len(v)}, expected {n}"
    assert len(out["mask_merging"]) == 6


def test_load_bhns_dict_shape_consistent(synthetic_bhns_path):
    from grb_io import load_bhns

    path, n_merging, *_ = synthetic_bhns_path
    out = load_bhns(path=path)
    assert out["n_merging"] == n_merging
    for k, v in out.items():
        if isinstance(v, np.ndarray) and k != "mask_merging":
            assert len(v) == n_merging, f"{k!r} has length {len(v)}, expected {n_merging}"


# ─────────────────────────────────────────────────────────────────────
# NaN-weight guard
# ─────────────────────────────────────────────────────────────────────
def test_load_bns_raises_on_nan_weight(tmp_path):
    """A NaN inside the merging slice of weight must abort the load."""
    from grb_io import load_bns

    path = tmp_path / "bns_nan_weight.h5"
    weights = np.array([0.1, 0.2, 0.3, np.nan, 0.5, 0.6])
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS")
        _write_dco_group(f, n_total=6, n_merging=4, weights=weights)

    with pytest.raises(ValueError, match="STROOPWAFEL weight has NaN"):
        load_bns(path=str(path))


def test_load_bns_passes_with_nan_outside_merging_mask(tmp_path):
    """NaN weights on non-merging rows must not trigger the guard."""
    from grb_io import load_bns

    path = tmp_path / "bns_nan_outside.h5"
    weights = np.array([0.1, 0.2, 0.3, 0.4, np.nan, 0.6])  # NaN at row 4
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS")
        _write_dco_group(f, n_total=6, n_merging=4, weights=weights)

    out = load_bns(path=str(path))
    assert np.isfinite(out["weights"]).all()


# ─────────────────────────────────────────────────────────────────────
# BHNS BH/NS type separation (Executor smoke)
# ─────────────────────────────────────────────────────────────────────
def test_load_bhns_type_separation(synthetic_bhns_path):
    """M_BH must follow stellarType1 == 14, regardless of mass order."""
    from grb_io import load_bhns

    path, n_merging, st1, m1, m2 = synthetic_bhns_path
    out = load_bhns(path=path)

    expected_M_BH = np.where(st1[:n_merging] == 14, m1[:n_merging], m2[:n_merging])
    expected_M_NS = np.where(st1[:n_merging] == 14, m2[:n_merging], m1[:n_merging])
    np.testing.assert_array_equal(out["M_BH"], expected_M_BH)
    np.testing.assert_array_equal(out["M_NS"], expected_M_NS)
    assert (out["M_BH"] >= out["M_NS"]).all(), (
        "Synthetic fixture violates BHNS hierarchy; BH should be heavier than NS in every row."
    )


# ─────────────────────────────────────────────────────────────────────
# _match_sn_to_dco vectorised join
# ─────────────────────────────────────────────────────────────────────
def test_match_sn_to_dco_last_per_seed(synthetic_kicks_bns_path):
    """Last SN per seed (highest time) must be returned; no warning fires."""
    from grb_io import _match_sn_to_dco

    path, seeds, expected_last_vsys, _ = synthetic_kicks_bns_path
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with h5.File(path, "r") as f:
            vsys = _match_sn_to_dco(f)

    np.testing.assert_array_equal(vsys, expected_last_vsys)
    sn_warnings = [w for w in caught if "DCO seeds had no matching SN" in str(w.message)]
    assert sn_warnings == [], (
        "All seeds matched; unmatched-SN warning should not fire. "
        f"Got: {[str(w.message) for w in sn_warnings]}"
    )


def test_match_sn_to_dco_warns_on_unmatched(synthetic_kicks_unmatched_path):
    """Unmatched DCO seeds must fire the missing-SN warning."""
    from grb_io import _match_sn_to_dco

    with pytest.warns(UserWarning, match="had no matching SN"):
        with h5.File(synthetic_kicks_unmatched_path, "r") as f:
            vsys = _match_sn_to_dco(f)

    # The third row is the orphan seed (999); v_sys must be NaN.
    assert np.isnan(vsys[2])
    assert np.isfinite(vsys[0]) and np.isfinite(vsys[1])


# ─────────────────────────────────────────────────────────────────────
# weighted_sample bias (Executor smoke)
# ─────────────────────────────────────────────────────────────────────
def test_weighted_sample_mean_within_5pct():
    """Weighted-sample mean recovers ``np.average`` within 5 percent.

    A subsampler that ignored ``weight`` would converge to the
    *unweighted* mean instead.  The test uses skewed weights that make
    the unweighted and weighted means measurably different.
    """
    from grb_io import weighted_sample

    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.standard_normal(n) + 5.0
    # Heavily skewed weights: the upper-tail samples dominate.
    w = np.exp(0.5 * x)

    mask = np.ones(n, dtype=bool)
    idx = weighted_sample(mask, w, n_target=10_000, rng=np.random.default_rng(7))
    sample_mean = float(np.mean(x[idx]))
    weighted_mean = float(np.average(x, weights=w))
    unweighted_mean = float(np.mean(x))

    # Sanity: the weighting must matter for this fixture.
    assert abs(weighted_mean - unweighted_mean) > 0.1, (
        "Test fixture is too symmetric; weighted_sample bias check "
        "would pass even if the subsampler ignored weights."
    )
    assert abs(sample_mean - weighted_mean) / abs(weighted_mean) < 0.05, (
        f"weighted_sample mean {sample_mean:.3f} deviates >5% from "
        f"np.average mean {weighted_mean:.3f}"
    )


def test_weighted_sample_handles_empty_mask():
    """Empty-mask call returns an empty index array, not a crash."""
    from grb_io import weighted_sample

    n = 10
    mask = np.zeros(n, dtype=bool)
    w = np.linspace(0.1, 1.0, n)
    idx = weighted_sample(mask, w, n_target=5)
    assert idx.size == 0


# ─────────────────────────────────────────────────────────────────────
# Loader return-dict carries `model` and `ns_max` (Section 14 prereq)
# ─────────────────────────────────────────────────────────────────────
# The Section 14 grid scan of grb_main.ipynb iterates over all 20
# Broekgaarden+ 2021 models and feeds bns['ns_max'] straight into classify_grid.
# These tests pin (a) the round-trip from the embedded HDF5 attribute
# to the loader output dict and (b) the model-substitution defense on
# the four variant loaders that previously skipped validation.
@pytest.mark.parametrize(
    "model_letter,ns_max", [("A", 2.5), ("F", 2.5), ("G", 2.5), ("J", 2.0), ("K", 3.0)]
)
def test_load_bns_returns_model_and_ns_max(tmp_path, model_letter, ns_max):
    from grb_io import load_bns

    path = tmp_path / f"bns_{model_letter}.h5"
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model=model_letter, ns_max=ns_max)
        _write_dco_group(f, n_total=4, n_merging=3)

    out = load_bns(path=str(path), expected_model=model_letter, expected_ns_max=ns_max)
    assert out["model"] == model_letter
    assert out["ns_max"] == ns_max


@pytest.mark.parametrize("model_letter,ns_max", [("A", 2.5), ("J", 2.0), ("K", 3.0)])
def test_load_bhns_returns_model_and_ns_max(tmp_path, model_letter, ns_max):
    from grb_io import load_bhns

    path = tmp_path / f"bhns_{model_letter}.h5"
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BHNS", model=model_letter, ns_max=ns_max)
        _write_dco_group(f, n_total=4, n_merging=3)

    out = load_bhns(path=str(path), expected_model=model_letter, expected_ns_max=ns_max)
    assert out["model"] == model_letter
    assert out["ns_max"] == ns_max


def test_load_bns_with_channels_validates_expected_model_mismatch(tmp_path):
    """The channels variant must raise on a mislabeled model attribute."""
    from grb_io import load_bns_with_channels

    path = tmp_path / "bns_channels_mislabel.h5"
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model="F", ns_max=2.5)
        _write_dco_group(
            f,
            n_total=3,
            n_merging=2,
            extra_cols={
                "M1ZAMS": np.full(3, 30.0),
                "M2ZAMS": np.full(3, 20.0),
                "doubleCommonEnvelopeFlag": np.zeros(3, int),
                "SemiMajorAxisPreCEE": np.full(3, 1e10),
                "SemiMajorAxisPostCEE": np.full(3, 1e8),
            },
        )
        fc = f.create_group("formationChannels")
        for col in (
            "mt_primary_ep1",
            "mt_primary_ep1_K1",
            "mt_secondary_ep1",
            "mt_secondary_ep1_K2",
            "CEE",
        ):
            fc.create_dataset(col, data=np.zeros((3, 1), int))

    with pytest.raises(ValueError, match="is model 'F', expected 'A'"):
        load_bns_with_channels(path=str(path), expected_model="A")


def test_load_bhns_with_kicks_validates_expected_model_mismatch(synthetic_kicks_bns_path):
    """The kicks variant must raise on a mislabeled model attribute.

    Reuses the BNS-kicks synthetic fixture (its ``stellarType1`` defaults
    to all-BH which is wrong for BHNS but does not affect the metadata
    validation that runs before the body of the loader).
    """
    from grb_io import load_bhns_with_kicks

    path, *_ = synthetic_kicks_bns_path
    with pytest.raises(ValueError, match="kind='BNS'"):
        # Synthetic file is BNS-kind; expected_model='K' would also raise
        # but the kind mismatch raises first.
        load_bhns_with_kicks(path=path, expected_model="K")


def test_load_bns_with_kicks_validates_expected_model_mismatch(synthetic_kicks_bns_path):
    """The BNS-kicks variant must raise on a mislabeled model attribute."""
    from grb_io import load_bns_with_kicks

    path, *_ = synthetic_kicks_bns_path
    with pytest.raises(ValueError, match="is model 'A', expected 'K'"):
        load_bns_with_kicks(path=path, expected_model="K")


def test_loader_returns_none_metadata_when_attributes_absent(tmp_path):
    """Backward compatibility: un-annotated archives return model=None."""
    from grb_io import load_bns

    path = tmp_path / "bns_unannotated.h5"
    with h5.File(path, "w") as f:
        _write_dco_group(f, n_total=3, n_merging=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence the "no metadata" warning
        out = load_bns(path=str(path))
    assert out["model"] is None
    assert out["ns_max"] is None


# ─────────────────────────────────────────────────────────────────────
# Broekgaarden+ 2021 model registry consistency
# ─────────────────────────────────────────────────────────────────────
def test_registry_matches_downloader_grid():
    """``BROEKGAARDEN21_MODELS`` and the downloader ``_GRID`` agree.

    The registry (consumed by the notebook grid scan) and the downloader
    grid (which renames the Zenodo zips on disk) are two independent
    transcriptions of the same Broekgaarden+ 2021 variation table.  If they
    drift, the scan loops over a model the downloader never produced, or the
    notebook silently skips one that is present.  Pin the suffix set and the
    per-model ``ns_max`` so either drift fails here, not mid-scan.
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    import download_compas_data as dl

    from grb_io import BROEKGAARDEN21_MODELS

    grid_suffixes = {row[1] for row in dl._GRID}  # project_suffix is index 1
    assert set(BROEKGAARDEN21_MODELS) == grid_suffixes, (
        f"registry vs downloader suffix mismatch: "
        f"registry-only={set(BROEKGAARDEN21_MODELS) - grid_suffixes}, "
        f"grid-only={grid_suffixes - set(BROEKGAARDEN21_MODELS)}."
    )
    grid_ns_max = {row[1]: row[5] for row in dl._GRID}  # ns_max is index 5
    for suffix, entry in BROEKGAARDEN21_MODELS.items():
        assert entry["ns_max"] == grid_ns_max[suffix], (
            f"ns_max mismatch for {suffix}: registry={entry['ns_max']}, "
            f"downloader={grid_ns_max[suffix]}."
        )


def test_registry_alpha_ce_matches_ce_prescription_constant():
    """Registry alpha_CE agrees with ``CE_PRESCRIPTION_BROEKGAARDEN21``.

    ``grb_rates.CE_PRESCRIPTION_BROEKGAARDEN21['alpha_CE']`` is the
    manuscript-core five-model alpha table.  Where the two overlap they must
    carry the same value, so a future edit to one does not silently
    contradict the other.
    """
    from grb_io import BROEKGAARDEN21_MODELS
    from grb_rates import CE_PRESCRIPTION_BROEKGAARDEN21

    for letter, alpha in CE_PRESCRIPTION_BROEKGAARDEN21["alpha_CE"].items():
        assert BROEKGAARDEN21_MODELS[letter]["alpha_ce"] == alpha, (
            f"alpha_CE mismatch for {letter}: registry="
            f"{BROEKGAARDEN21_MODELS[letter]['alpha_ce']}, "
            f"CE_PRESCRIPTION={alpha}."
        )


def test_all_model_suffixes_registry_order():
    """``ALL_MODEL_SUFFIXES`` mirrors ``BROEKGAARDEN21_MODELS`` in registry order.

    The grid scan keys the per-model Alsing-remap RNG seeds off this index, so
    the list must stay a faithful, de-duplicated mirror of the registry with
    the fiducial Model A leading.
    """
    from grb_io import ALL_MODEL_SUFFIXES, BROEKGAARDEN21_MODELS

    assert ALL_MODEL_SUFFIXES == list(BROEKGAARDEN21_MODELS)
    assert ALL_MODEL_SUFFIXES[0] == "A"
    assert set(ALL_MODEL_SUFFIXES) == set(BROEKGAARDEN21_MODELS)
    assert len(ALL_MODEL_SUFFIXES) == len(set(ALL_MODEL_SUFFIXES)) == 20
