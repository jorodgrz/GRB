"""Behaviour and edge-case tests for ``grb_io``.

Complements ``tests/unit/test_grb_io.py`` (NaN-weight guard, dict
shape, BH/NS type separation, weighted_sample bias, per-model
model / ns_max round-trip) and the data-bound real-data audit in
``tests/integration/test_grb_io_realdata.py``.  This file fills the
gaps: the four private validators as standalone functions, the
``_with_channels`` and ``_with_kicks`` loader bodies, the export
helpers, the ``METALLICITY_GRID`` schema, and the documented
edge cases for ``weighted_sample`` / ``log_jitter`` /
``verify_shared_metallicity_prior``.

All tests build tiny synthetic HDF5 files in ``tmp_path`` mirroring
the Broekgaarden+ 2021 COMPAS column layout.

References cited inline per test: Broekgaarden et al. (2021,
arXiv:2103.02608) Sec. 3.2 (metallicity prior; HDF5 schema);
Abbott et al. (2019 PRX 9, 011001; 2020 ApJL 892, L3) for
``OBSERVED_GW_EVENTS`` (pinned separately in
``tests/anchors/test_literature_anchors.py``).
"""

from __future__ import annotations

import warnings

import h5py as h5
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic HDF5 builders (parallel to tests/unit/test_grb_io.py).
# Kept local so the new file is self-contained.
# ---------------------------------------------------------------------------
def _write_dco_group(
    f,
    *,
    n_total=6,
    n_merging=4,
    st1_values=None,
    weights=None,
    masses=None,
    extra_cols=None,
):
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
    f.attrs["kind"] = kind
    f.attrs["model"] = model
    f.attrs["ns_max"] = float(ns_max)


def _write_formation_channels_group(f, n_total, *, fc_mt_p1=None, fc_CEE=None):
    """Add a ``formationChannels`` group sized to ``n_total``."""
    fc = f.create_group("formationChannels")
    arrays = {
        "mt_primary_ep1": fc_mt_p1 if fc_mt_p1 is not None else np.full(n_total, 3.0),
        "mt_primary_ep1_K1": np.zeros(n_total, dtype=int),
        "mt_secondary_ep1": np.zeros(n_total),
        "mt_secondary_ep1_K2": np.zeros(n_total, dtype=int),
        "CEE": fc_CEE if fc_CEE is not None else np.full(n_total, 5.0),
    }
    for name, arr in arrays.items():
        fc.create_dataset(name, data=np.asarray(arr).reshape(-1, 1))
    return fc


def _write_supernovae_group(f, dco_seeds, *, last_vsys=None):
    """Two SN rows per seed; the second has the larger time (so it is
    the post-2nd-SN systemic velocity)."""
    n = len(dco_seeds)
    sn_seeds = np.repeat(np.asarray(dco_seeds, dtype=np.int64), 2)
    sn_time = np.tile([1.0, 2.0], n)
    if last_vsys is None:
        last_vsys = 100.0 + 10.0 * np.arange(n, dtype=float)
    first_vsys = 10.0 + 1.0 * np.arange(n, dtype=float)
    sn_vsys = np.empty(2 * n, dtype=float)
    sn_vsys[0::2] = first_vsys
    sn_vsys[1::2] = last_vsys
    sn = f.create_group("supernovae")
    sn.create_dataset("randomSeed", data=sn_seeds.reshape(-1, 1))
    sn.create_dataset("systemicVelocity", data=sn_vsys.reshape(-1, 1))
    sn.create_dataset("time", data=sn_time.reshape(-1, 1))
    return last_vsys


# ---------------------------------------------------------------------------
# _validate_hdf5_metadata
# ---------------------------------------------------------------------------
def test_validate_hdf5_metadata_returns_none_pair_on_unannotated_file(tmp_path):
    """Un-annotated archives must return ``(None, None)`` and emit one
    UserWarning about the missing ``model`` attribute.

    The fallback path keeps backward compatibility with archives
    downloaded before ``tools/embed_model_metadata.py`` existed.
    """
    from grb_io import _METADATA_WARN_CACHE, _validate_hdf5_metadata

    path = tmp_path / "bns_unannotated.h5"
    with h5.File(path, "w") as f:
        _write_dco_group(f, n_total=2, n_merging=1)

    _METADATA_WARN_CACHE.discard(str(path))
    with pytest.warns(UserWarning, match="no embedded"):
        model, ns_max = _validate_hdf5_metadata(str(path))
    assert model is None and ns_max is None


def test_validate_hdf5_metadata_returns_pair_on_annotated_file(tmp_path):
    """Annotated archives must return ``(model_str, ns_max_float)``."""
    from grb_io import _validate_hdf5_metadata

    path = tmp_path / "bns_annotated.h5"
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model="A", ns_max=2.5)
        _write_dco_group(f, n_total=2, n_merging=1)
    model, ns_max = _validate_hdf5_metadata(
        str(path), expected_kind="BNS", expected_model="A", expected_ns_max=2.5
    )
    assert model == "A"
    assert ns_max == pytest.approx(2.5)


def test_validate_hdf5_metadata_raises_on_kind_mismatch(tmp_path):
    """A file stamped ``kind='BNS'`` but loaded with ``expected_kind='BHNS'``
    must raise ``ValueError`` mentioning the kind."""
    from grb_io import _validate_hdf5_metadata

    path = tmp_path / "bns_kind.h5"
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model="A", ns_max=2.5)
        _write_dco_group(f, n_total=2, n_merging=1)
    with pytest.raises(ValueError, match="kind='BNS'"):
        _validate_hdf5_metadata(str(path), expected_kind="BHNS")


def test_validate_hdf5_metadata_raises_on_ns_max_mismatch(tmp_path):
    """A file stamped ``ns_max=2.5`` loaded with ``expected_ns_max=2.0``
    must raise ``ValueError`` reporting both values."""
    from grb_io import _validate_hdf5_metadata

    path = tmp_path / "bns_ns_max.h5"
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model="A", ns_max=2.5)
        _write_dco_group(f, n_total=2, n_merging=1)
    with pytest.raises(ValueError, match="ns_max"):
        _validate_hdf5_metadata(str(path), expected_ns_max=2.0)


def test_validate_hdf5_metadata_warning_emits_once_per_path(tmp_path):
    """A second call on the same un-annotated path must not re-warn.

    The ``_METADATA_WARN_CACHE`` set deduplicates per-path warnings so
    the Section 14 grid scan over all 20 models does not spam the
    notebook log.
    """
    from grb_io import _METADATA_WARN_CACHE, _validate_hdf5_metadata

    path = tmp_path / "bns_warn_once.h5"
    with h5.File(path, "w") as f:
        _write_dco_group(f, n_total=2, n_merging=1)
    _METADATA_WARN_CACHE.discard(str(path))

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        _validate_hdf5_metadata(str(path))
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        _validate_hdf5_metadata(str(path))

    assert any("no embedded" in str(w.message) for w in first)
    assert not any("no embedded" in str(w.message) for w in second), (
        f"Second call re-emitted the metadata warning; second={[str(w.message) for w in second]}"
    )


# ---------------------------------------------------------------------------
# _validate_delay_times
# ---------------------------------------------------------------------------
def test_validate_delay_times_raises_when_max_above_1e5():
    """Delay times in yr (1e6x too large) must raise.

    Sanity-check guard against an accidental unit conversion in
    upstream tooling.
    """
    from grb_io import _validate_delay_times

    dt = np.array([1.0, 5e5])
    with pytest.raises(ValueError, match="delay_time range"):
        _validate_delay_times(dt)


def test_validate_delay_times_raises_on_negative_delay():
    """A negative entry indicates a broken tform / tc lookup."""
    from grb_io import _validate_delay_times

    dt = np.array([100.0, -5.0, 200.0])
    with pytest.raises(ValueError, match="delay_time range"):
        _validate_delay_times(dt)


def test_validate_delay_times_passes_on_typical_myr_range():
    """A sample inside the expected Myr range must pass silently."""
    from grb_io import _validate_delay_times

    dt = np.array([10.0, 100.0, 1000.0, 14000.0])
    _validate_delay_times(dt)


# ---------------------------------------------------------------------------
# _check_weights_no_nan
# ---------------------------------------------------------------------------
def test_check_weights_no_nan_raises_on_nan():
    """NaN weight raises ``ValueError`` whose message contains the basename."""
    from grb_io import _check_weights_no_nan

    w = np.array([0.1, 0.2, np.nan, 0.4])
    with pytest.raises(ValueError, match="poisoned weights"):
        _check_weights_no_nan(w, "/path/to/COMPASCompactOutput_BNS_A.h5")


def test_check_weights_no_nan_passes_on_finite_array():
    """All-finite array passes silently."""
    from grb_io import _check_weights_no_nan

    _check_weights_no_nan(np.linspace(0.1, 1.0, 10), "anything.h5")


# ---------------------------------------------------------------------------
# _validate_loader_dict
# ---------------------------------------------------------------------------
def test_validate_loader_dict_raises_on_length_mismatch():
    """A stray-length array in the loader dict raises ``AssertionError``."""
    from grb_io import _validate_loader_dict

    out = {
        "m1": np.zeros(4),
        "m2": np.zeros(4),
        "weights": np.zeros(3),  # wrong length
    }
    with pytest.raises(AssertionError, match="weights"):
        _validate_loader_dict(out, n_merging=4, path="dummy.h5")


def test_validate_loader_dict_excludes_mask_merging():
    """``mask_merging`` spans the pre-mask catalogue so its length
    intentionally differs from ``n_merging``; the validator must skip it.
    """
    from grb_io import _validate_loader_dict

    out = {
        "m1": np.zeros(4),
        "m2": np.zeros(4),
        "mask_merging": np.zeros(10, dtype=bool),  # spans pre-mask catalogue
    }
    _validate_loader_dict(out, n_merging=4, path="dummy.h5")


def test_validate_loader_dict_skips_non_array_values():
    """String / int / None values pass through; only ndarrays are checked."""
    from grb_io import _validate_loader_dict

    out = {
        "m1": np.zeros(4),
        "population": "BNS",
        "n_merging": 4,
        "model": None,
        "ns_max": None,
    }
    _validate_loader_dict(out, n_merging=4, path="dummy.h5")


# ---------------------------------------------------------------------------
# verify_shared_metallicity_prior
# ---------------------------------------------------------------------------
def _write_z_only(path, z_values):
    with h5.File(path, "w") as f:
        sys_grp = f.create_group("systems")
        sys_grp.create_dataset("Metallicity1", data=np.asarray(z_values).reshape(-1, 1))


def test_verify_shared_metallicity_prior_returns_range_on_match(tmp_path):
    """Two synthetic files with the same (Z_min, Z_max) return that pair."""
    from grb_io import verify_shared_metallicity_prior

    p_a = tmp_path / "a.h5"
    p_b = tmp_path / "b.h5"
    z = np.array([1e-4, 5e-3, 3e-2])
    _write_z_only(p_a, z)
    _write_z_only(p_b, z)
    out = verify_shared_metallicity_prior(str(p_a), str(p_b))
    assert out == (pytest.approx(1e-4), pytest.approx(3e-2))


def test_verify_shared_metallicity_prior_raises_on_mismatch(tmp_path):
    """Different Z ranges raise ``ValueError``."""
    from grb_io import verify_shared_metallicity_prior

    p_a = tmp_path / "a.h5"
    p_b = tmp_path / "b.h5"
    _write_z_only(p_a, [1e-4, 3e-2])
    _write_z_only(p_b, [2e-4, 3e-2])
    with pytest.raises(ValueError, match="Metallicity ranges differ"):
        verify_shared_metallicity_prior(str(p_a), str(p_b))


# ---------------------------------------------------------------------------
# read_expected_local_rate and read_metallicity_range
# ---------------------------------------------------------------------------
def test_read_expected_local_rate_sums_w_000_column(tmp_path):
    """``read_expected_local_rate`` returns the sum of ``weights_intrinsic/w_000``."""
    from grb_io import read_expected_local_rate

    path = tmp_path / "rate.h5"
    w_000 = np.array([1.5, 2.0, 3.5, 4.0])
    with h5.File(path, "w") as f:
        wi = f.create_group("weights_intrinsic")
        wi.create_dataset("w_000", data=w_000.reshape(-1, 1))
    out = read_expected_local_rate(str(path))
    assert out == pytest.approx(float(w_000.sum()), rel=1e-12)


def test_read_metallicity_range_returns_min_max_floats(tmp_path):
    """``read_metallicity_range`` returns ``(float(Z.min()), float(Z.max()))``."""
    from grb_io import read_metallicity_range

    path = tmp_path / "z_range.h5"
    z = np.array([0.0003, 0.005, 0.02, 0.001])
    _write_z_only(path, z)
    z_min, z_max = read_metallicity_range(str(path))
    assert isinstance(z_min, float) and isinstance(z_max, float)
    assert z_min == pytest.approx(float(z.min()))
    assert z_max == pytest.approx(float(z.max()))


# ---------------------------------------------------------------------------
# load_bns_with_channels body
# ---------------------------------------------------------------------------
def _write_bns_channels_file(path, *, n_total=6, n_merging=4, masses=None, fc_mt_p1=None):
    """Write a BNS HDF5 with the formationChannels group attached."""
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS", model="A", ns_max=2.5)
        _write_dco_group(
            f,
            n_total=n_total,
            n_merging=n_merging,
            masses=masses,
            extra_cols={
                "M1ZAMS": np.full(n_total, 30.0),
                "M2ZAMS": np.full(n_total, 20.0),
                "doubleCommonEnvelopeFlag": np.zeros(n_total, dtype=int),
                "SemiMajorAxisPreCEE": np.full(n_total, 1e10),
                "SemiMajorAxisPostCEE": np.full(n_total, 1e8),
            },
        )
        _write_formation_channels_group(f, n_total, fc_mt_p1=fc_mt_p1)


def test_load_bns_with_channels_returns_full_key_set(tmp_path):
    """Loader returns every documented key for the channels variant."""
    from grb_io import load_bns_with_channels

    path = tmp_path / "bns_ch.h5"
    _write_bns_channels_file(path)
    out = load_bns_with_channels(path=str(path))
    expected = {
        "m1",
        "m2",
        "weights",
        "metallicity",
        "delay_time",
        "n_merging",
        "mask_merging",
        "m1zams",
        "m2zams",
        "dblCE",
        "sep_preCE",
        "sep_postCE",
        "fc_mt_p1",
        "fc_mt_p1_K1",
        "fc_mt_s1",
        "fc_mt_s1_K2",
        "fc_CEE",
        "population",
        "model",
        "ns_max",
    }
    assert expected <= set(out.keys()), f"Missing keys: {expected - set(out.keys())}"


def test_load_bns_with_channels_sort_masses_true_enforces_m1_ge_m2(tmp_path):
    """``sort_masses=True`` (default) enforces ``m1 >= m2``."""
    from grb_io import load_bns_with_channels

    # Construct a file where M1 < M2 in some rows so the sort has work
    # to do.  The merging rows are the first n_merging entries.
    m1 = np.array([1.2, 2.0, 1.4, 1.3, 1.5, 1.6])
    m2 = np.array([1.8, 1.4, 1.5, 1.9, 1.3, 1.2])
    path = tmp_path / "bns_swap.h5"
    _write_bns_channels_file(path, masses=(m1, m2))
    out = load_bns_with_channels(path=str(path), sort_masses=True)
    assert (out["m1"] >= out["m2"]).all()


def test_load_bns_with_channels_does_not_reorder_formation_channel_columns(tmp_path):
    """``sort_masses=True`` swaps m1/m2 but leaves ``fc_mt_p1`` aligned
    with the original COMPAS primary.

    Documented caveat in the ``load_bns_with_channels`` body.  Pinning
    it via a fingerprint: ``fc_mt_p1`` carries a row-indexed sentinel
    array; after the loader reorders the masses it must NOT also
    reorder ``fc_mt_p1``, so the loaded ``fc_mt_p1`` equals the
    original COMPAS column on the merging slice.
    """
    from grb_io import load_bns_with_channels

    n_total = 6
    n_merging = 4
    # Half the rows have m1 < m2 (so the sort flips them).
    m1 = np.array([1.2, 2.0, 1.4, 1.3, 1.5, 1.6])
    m2 = np.array([1.8, 1.4, 1.5, 1.9, 1.3, 1.2])
    fc_mt_p1 = np.array([7.0, 7.1, 7.2, 7.3, 7.4, 7.5])  # row-indexed sentinel
    path = tmp_path / "bns_no_reorder.h5"
    _write_bns_channels_file(
        path, n_total=n_total, n_merging=n_merging, masses=(m1, m2), fc_mt_p1=fc_mt_p1
    )
    out = load_bns_with_channels(path=str(path), sort_masses=True)
    np.testing.assert_array_equal(out["fc_mt_p1"], fc_mt_p1[:n_merging])


# ---------------------------------------------------------------------------
# load_bhns_with_channels body
# ---------------------------------------------------------------------------
def _write_bhns_channels_file(path, *, st1=None, masses=None):
    n_total = 6
    n_merging = 4
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BHNS", model="A", ns_max=2.5)
        _write_dco_group(
            f,
            n_total=n_total,
            n_merging=n_merging,
            st1_values=st1,
            masses=masses,
            extra_cols={
                "doubleCommonEnvelopeFlag": np.zeros(n_total, dtype=int),
                "SemiMajorAxisPreCEE": np.full(n_total, 1e10),
            },
        )
        _write_formation_channels_group(f, n_total)


def test_load_bhns_with_channels_returns_full_key_set(tmp_path):
    """All documented BHNS channels keys present."""
    from grb_io import load_bhns_with_channels

    path = tmp_path / "bhns_ch.h5"
    _write_bhns_channels_file(path)
    out = load_bhns_with_channels(path=str(path))
    expected = {
        "M_BH",
        "M_NS",
        "weights",
        "metallicity",
        "delay_time",
        "n_merging",
        "mask_merging",
        "dblCE",
        "sep_preCE",
        "fc_mt_p1",
        "fc_mt_p1_K1",
        "fc_mt_s1",
        "fc_mt_s1_K2",
        "fc_CEE",
        "population",
        "model",
        "ns_max",
    }
    assert expected <= set(out.keys()), f"Missing keys: {expected - set(out.keys())}"


def test_load_bhns_with_channels_routes_BH_via_stellartype1(tmp_path):
    """``load_bhns_with_channels`` resolves M_BH from ``stellarType1==14``,
    matching ``load_bhns``."""
    from grb_io import load_bhns_with_channels

    st1 = np.array([14, 13, 14, 13, 14, 14], dtype=int)
    m1 = np.array([8.0, 1.4, 6.5, 1.3, 7.0, 9.0])
    m2 = np.array([1.4, 7.0, 1.3, 6.0, 1.4, 1.5])
    path = tmp_path / "bhns_st1.h5"
    _write_bhns_channels_file(path, st1=st1, masses=(m1, m2))
    out = load_bhns_with_channels(path=str(path))
    expected_M_BH = np.where(st1[:4] == 14, m1[:4], m2[:4])
    expected_M_NS = np.where(st1[:4] == 14, m2[:4], m1[:4])
    np.testing.assert_array_equal(out["M_BH"], expected_M_BH)
    np.testing.assert_array_equal(out["M_NS"], expected_M_NS)
    assert (out["M_BH"] >= out["M_NS"]).all()


# ---------------------------------------------------------------------------
# load_bns_with_kicks / load_bhns_with_kicks
# ---------------------------------------------------------------------------
def _write_kicks_file(path, *, kind="BNS", st1=None, masses=None):
    n_total = 4
    n_merging = 3
    dco_seeds = np.arange(100, 100 + n_total, dtype=np.int64)
    extras = {
        "drawnKick1": np.full(n_total, 50.0),
        "drawnKick2": np.full(n_total, 60.0),
        "separationDCOFormation": np.full(n_total, 1e10),
        "eccentricityDCOFormation": np.full(n_total, 0.3),
        "seed": dco_seeds,
    }
    with h5.File(path, "w") as f:
        _write_metadata(f, kind=kind, model="A", ns_max=2.5)
        _write_dco_group(
            f,
            n_total=n_total,
            n_merging=n_merging,
            st1_values=st1,
            masses=masses,
            extra_cols=extras,
        )
        last_vsys = _write_supernovae_group(f, dco_seeds)
    return last_vsys, n_merging


def test_load_bns_with_kicks_returns_full_key_set(tmp_path):
    """Keys include ``drawnKick1``, ``drawnKick2``, ``v_sys``, ``sep_DCO``,
    ``ecc_DCO`` per the documented signature."""
    from grb_io import load_bns_with_kicks

    path = tmp_path / "bns_kicks.h5"
    _write_kicks_file(path, kind="BNS")
    out = load_bns_with_kicks(path=str(path))
    expected = {
        "m1",
        "m2",
        "weights",
        "metallicity",
        "delay_time",
        "n_merging",
        "mask_merging",
        "drawnKick1",
        "drawnKick2",
        "v_sys",
        "sep_DCO",
        "ecc_DCO",
        "population",
        "model",
        "ns_max",
    }
    assert expected <= set(out.keys()), f"Missing keys: {expected - set(out.keys())}"


def test_load_bns_with_kicks_v_sys_matches_match_sn_to_dco(tmp_path):
    """The loader's ``v_sys`` array equals ``_match_sn_to_dco`` filtered
    by the merging mask."""
    from grb_io import _match_sn_to_dco, load_bns_with_kicks

    path = tmp_path / "bns_kicks_vsys.h5"
    last_vsys, n_merging = _write_kicks_file(path, kind="BNS")
    out = load_bns_with_kicks(path=str(path))
    with h5.File(path, "r") as f:
        vsys_full = _match_sn_to_dco(f)
    np.testing.assert_array_equal(out["v_sys"], vsys_full[:n_merging])


def test_load_bhns_with_kicks_returns_full_key_set(tmp_path):
    """BHNS variant: ``M_BH``, ``M_NS``, ``v_sys`` and the kinematic keys
    are present."""
    from grb_io import load_bhns_with_kicks

    st1 = np.array([14, 13, 14, 14], dtype=int)
    m1 = np.array([8.0, 1.4, 6.5, 7.0])
    m2 = np.array([1.4, 7.0, 1.3, 1.5])
    path = tmp_path / "bhns_kicks.h5"
    _write_kicks_file(path, kind="BHNS", st1=st1, masses=(m1, m2))
    out = load_bhns_with_kicks(path=str(path))
    expected = {
        "M_BH",
        "M_NS",
        "weights",
        "drawnKick1",
        "drawnKick2",
        "v_sys",
        "sep_DCO",
        "ecc_DCO",
        "population",
        "model",
        "ns_max",
    }
    assert expected <= set(out.keys()), f"Missing keys: {expected - set(out.keys())}"


def test_load_bhns_with_kicks_v_sys_matches_match_sn_to_dco(tmp_path):
    """BHNS variant: ``v_sys`` equals ``_match_sn_to_dco`` filtered by
    the merging mask."""
    from grb_io import _match_sn_to_dco, load_bhns_with_kicks

    st1 = np.array([14, 13, 14, 14], dtype=int)
    m1 = np.array([8.0, 1.4, 6.5, 7.0])
    m2 = np.array([1.4, 7.0, 1.3, 1.5])
    path = tmp_path / "bhns_kicks_vsys.h5"
    last_vsys, n_merging = _write_kicks_file(path, kind="BHNS", st1=st1, masses=(m1, m2))
    out = load_bhns_with_kicks(path=str(path))
    with h5.File(path, "r") as f:
        vsys_full = _match_sn_to_dco(f)
    np.testing.assert_array_equal(out["v_sys"], vsys_full[:n_merging])


# ---------------------------------------------------------------------------
# sort_masses=False counter-test
# ---------------------------------------------------------------------------
def test_load_bns_sort_masses_false_preserves_compas_ordering(tmp_path):
    """``sort_masses=False`` returns the masses as they appear in the
    COMPAS HDF5; the BNS file built here has ``m1 < m2`` in row 0."""
    from grb_io import load_bns

    m1 = np.array([1.2, 1.5, 1.4, 1.3, 1.6, 1.7])
    m2 = np.array([1.8, 1.4, 1.3, 1.5, 1.2, 1.1])
    path = tmp_path / "bns_no_sort.h5"
    with h5.File(path, "w") as f:
        _write_metadata(f, kind="BNS")
        _write_dco_group(f, n_total=6, n_merging=4, masses=(m1, m2))

    out = load_bns(path=str(path), sort_masses=False)
    np.testing.assert_array_equal(out["m1"], m1[:4])
    np.testing.assert_array_equal(out["m2"], m2[:4])
    # Counter-check: at least one row has m1 < m2.
    assert (out["m1"] < out["m2"]).any()


# ---------------------------------------------------------------------------
# METALLICITY_GRID schema
# ---------------------------------------------------------------------------
def test_metallicity_grid_length_is_53():
    """53 unique values; regression sentinel for an accidental
    insertion or removal that would shift every downstream
    metallicity-binned rate."""
    from grb_io import METALLICITY_GRID

    assert len(METALLICITY_GRID) == 53


def test_metallicity_grid_monotonically_increasing_no_duplicates():
    """Strictly monotonic in Z; a duplicate would silently double-count."""
    from grb_io import METALLICITY_GRID

    diffs = np.diff(METALLICITY_GRID)
    assert (diffs > 0).all(), (
        f"Non-monotonic METALLICITY_GRID; min diff = {diffs.min()}; "
        f"location = {int(np.argmin(diffs))}"
    )


def test_metallicity_grid_endpoints_match_broekgaarden_prior():
    """``METALLICITY_GRID[0] == 1e-4`` and ``METALLICITY_GRID[-1] == 3e-2``,
    matching the Broekgaarden+ 2021 Sec. 3.2 prior bounds."""
    from grb_io import METALLICITY_GRID

    assert METALLICITY_GRID[0] == pytest.approx(1e-4, rel=1e-12)
    assert METALLICITY_GRID[-1] == pytest.approx(3e-2, rel=1e-12)


# ---------------------------------------------------------------------------
# weighted_sample edge cases
# ---------------------------------------------------------------------------
def test_weighted_sample_zero_weight_fallback_returns_first_n_target():
    """All-zero weights fall back to the first ``n_target`` indices in
    the mask (uniform draw)."""
    from grb_io import weighted_sample

    n = 10
    mask = np.ones(n, dtype=bool)
    w = np.zeros(n)
    idx = weighted_sample(mask, w, n_target=5)
    np.testing.assert_array_equal(idx, np.arange(5))


def test_weighted_sample_n_target_larger_than_available_returns_all():
    """Requesting more samples than the mask population gives the full set."""
    from grb_io import weighted_sample

    n = 4
    mask = np.ones(n, dtype=bool)
    w = np.linspace(0.1, 1.0, n)
    idx = weighted_sample(mask, w, n_target=100, rng=np.random.default_rng(0))
    assert idx.size == n
    assert set(idx.tolist()) == set(range(n))


def test_weighted_sample_rng_determinism():
    """Fixed-seed reproducibility."""
    from grb_io import weighted_sample

    rng_factory = lambda: np.random.default_rng(123)  # noqa: E731
    n = 20
    mask = np.ones(n, dtype=bool)
    w = np.linspace(0.1, 1.0, n)
    a = weighted_sample(mask, w, n_target=8, rng=rng_factory())
    b = weighted_sample(mask, w, n_target=8, rng=rng_factory())
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# log_jitter
# ---------------------------------------------------------------------------
def test_log_jitter_output_stays_within_scale_band():
    """``log_jitter(Z, scale=s)`` returns values in ``Z * [10**-s, 10**s]``."""
    from grb_io import log_jitter

    rng = np.random.default_rng(0)
    Z = np.array([1e-4, 1e-3, 1e-2])
    scale = 0.04
    out = log_jitter(Z, scale=scale, rng=rng)
    ratio = out / Z
    assert (ratio >= 10**-scale - 1e-12).all()
    assert (ratio <= 10**scale + 1e-12).all()


def test_log_jitter_rng_determinism():
    """Fixed-seed reproducibility."""
    from grb_io import log_jitter

    Z = np.linspace(1e-4, 3e-2, 10)
    a = log_jitter(Z, scale=0.04, rng=np.random.default_rng(7))
    b = log_jitter(Z, scale=0.04, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# save_efficiencies and save_rates
# ---------------------------------------------------------------------------
def test_save_efficiencies_roundtrip(tmp_path):
    """``save_efficiencies`` + ``np.load`` recovers the original stack."""
    from grb_io import save_efficiencies

    arrays = [np.linspace(0, 1, 10), np.linspace(1, 2, 10), np.linspace(2, 3, 10)]
    path = tmp_path / "eff.npy"
    save_efficiencies(str(path), arrays, labels=["a", "b", "c"])
    loaded = np.load(str(path))
    assert loaded.shape == (3, 10)
    np.testing.assert_allclose(loaded, np.array(arrays))


def test_save_efficiencies_handles_labels_none(tmp_path):
    """``labels=None`` path saves silently without printing."""
    from grb_io import save_efficiencies

    arrays = [np.linspace(0, 1, 5), np.linspace(1, 2, 5)]
    path = tmp_path / "eff_no_labels.npy"
    save_efficiencies(str(path), arrays, labels=None)
    assert path.exists()
    loaded = np.load(str(path))
    assert loaded.shape == (2, 5)


def test_save_rates_includes_redshifts_row(tmp_path):
    """``save_rates`` stack starts with the redshift row, followed by
    named rate rows."""
    from grb_io import save_rates

    redshifts = np.linspace(0.0, 2.0, 6)
    rate_dict = {
        "All": np.linspace(10.0, 1.0, 6),
        "Class A": np.linspace(5.0, 0.5, 6),
    }
    path = tmp_path / "rates.npy"
    save_rates(str(path), redshifts, rate_dict)
    loaded = np.load(str(path))
    assert loaded.shape == (3, 6)
    np.testing.assert_array_equal(loaded[0], redshifts)
    np.testing.assert_array_equal(loaded[1], rate_dict["All"])
    np.testing.assert_array_equal(loaded[2], rate_dict["Class A"])
