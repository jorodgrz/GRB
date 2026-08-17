"""
Data loading and export helpers for COMPAS HDF5 compact-object outputs.

Standardises how BNS and BHNS populations are read from the COMPAS
doubleCompactObjects/formationChannels groups. All loaders filter to
merging systems (mergesInHubbleTimeFlag == 1) and return plain numpy arrays.
"""

import os
import warnings

import h5py as h5
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DEFAULT_BNS_PATH = os.path.join(_DATA_DIR, "COMPASCompactOutput_BNS_A.h5")
DEFAULT_BHNS_PATH = os.path.join(_DATA_DIR, "COMPASCompactOutput_BHNS_A.h5")
DEFAULT_BNS_K_PATH = os.path.join(_DATA_DIR, "COMPASCompactOutput_BNS_K.h5")

# Track which paths we've already complained about so the "no metadata"
# warning fires once per file per process instead of on every loader call.
_METADATA_WARN_CACHE: set[str] = set()


def _validate_hdf5_metadata(path, expected_kind=None, expected_model=None, expected_ns_max=None):
    """Validate Broekgaarden+ 2021 model metadata embedded by
    ``tools/embed_model_metadata.py``.

    If the attributes are absent, emit a one-time UserWarning per path
    and return without raising; this preserves backward compatibility
    with un-annotated archives.  If they are present and contradict
    the caller's expectation, raise ``ValueError`` so a mislabeled or
    renamed file fails loudly.

    Returns
    -------
    (actual_model, actual_ns_max) : tuple
        The values read from the HDF5 root attributes, or ``(None, None)``
        if the file carries no metadata.  Loaders attach these to their
        output dict so downstream code (``classify_grid(ns_max=...)``)
        does not need to re-open the HDF5 or maintain a per-letter literal
        table.
    """
    with h5.File(path, "r") as f:
        attrs = dict(f.attrs)

    if not attrs.get("model") and not attrs.get("ns_max"):
        if path not in _METADATA_WARN_CACHE:
            _METADATA_WARN_CACHE.add(path)
            warnings.warn(
                f"COMPAS HDF5 {os.path.basename(path)} has no embedded "
                f"model identifier; cannot validate against expected "
                f"({expected_kind}, model={expected_model}, "
                f"ns_max={expected_ns_max}).  Run "
                f"`python tools/embed_model_metadata.py` once per "
                f"download to enable validation.",
                stacklevel=3,
            )
        return None, None

    actual_kind = str(attrs.get("kind", "")) or None
    actual_model = str(attrs.get("model", "")) or None
    actual_ns_max = float(attrs["ns_max"]) if "ns_max" in attrs else None

    if expected_kind and actual_kind and actual_kind != expected_kind:
        raise ValueError(
            f"COMPAS HDF5 {os.path.basename(path)} has embedded "
            f"kind={actual_kind!r}, expected {expected_kind!r}.  "
            f"Either the file is mislabeled or the wrong loader was "
            f"called."
        )
    if expected_model and actual_model and actual_model != expected_model:
        raise ValueError(
            f"COMPAS HDF5 {os.path.basename(path)} is model "
            f"{actual_model!r}, expected {expected_model!r}."
        )
    if (
        expected_ns_max is not None
        and actual_ns_max is not None
        and not np.isclose(actual_ns_max, expected_ns_max)
    ):
        raise ValueError(
            f"COMPAS HDF5 {os.path.basename(path)} has ns_max="
            f"{actual_ns_max}, expected {expected_ns_max}."
        )
    return actual_model, actual_ns_max


def _validate_delay_times(dt, label=""):
    """Sanity-check that delay times are in Myr (expected range ~1-14000).

    COMPAS stores tform and tc in Myr.  If someone passes data in Gyr
    the values would be 1e3x too small; in yr they'd be 1e6x too large.
    """
    if dt.max() > 1e5 or dt.min() < 0:
        raise ValueError(
            f"delay_time range [{dt.min():.1f}, {dt.max():.1f}]{label} "
            "looks wrong; expected Myr (typical range 1-14000)"
        )


def _check_weights_no_nan(w_masked, path):
    """Hard fail if STROOPWAFEL weights contain NaN after the merging mask.

    A silent NaN propagates into ``np.average`` and weighted histograms
    as ``nan`` outputs, biasing every downstream rate, so we raise rather
    than warn.
    """
    if np.any(np.isnan(w_masked)):
        raise ValueError(
            f"STROOPWAFEL weight has NaN in {os.path.basename(path)} "
            f"after mergesInHubbleTimeFlag mask; refusing to return "
            f"poisoned weights."
        )


def _validate_loader_dict(out, n_merging, path):
    """Assert all per-system arrays in a loader dict have length ``n_merging``.

    ``mask_merging`` is excluded because it spans the full pre-mask
    catalogue.  Non-array values (the ``population`` string tag, the
    ``n_merging`` int) are skipped via the ``isinstance`` filter.

    Catches the copy-paste typo where one column is returned without
    ``[mask]``.
    """
    lengths = {
        k: len(v) for k, v in out.items() if isinstance(v, np.ndarray) and k != "mask_merging"
    }
    bad = {k: L for k, L in lengths.items() if n_merging != L}
    assert not bad, (
        f"Loader dict shape mismatch in {os.path.basename(path)}: "
        f"keys {bad} have wrong length, expected n_merging={n_merging}."
    )


def filter_systems(d, mask):
    """Apply a per-system boolean ``mask`` to every column of a loader dict.

    Returns a shallow copy in which each length-``n_merging`` array
    (``m1``, ``m2``, ``weights``, ``metallicity``, ``delay_time``, and
    every optional channel / kick column) is subset by ``mask``.  Scalar
    and metadata fields are carried through untouched except
    ``n_merging``, which is reset to ``int(mask.sum())``, and
    ``mask_merging``, which spans the full pre-merger-cut catalogue and is
    left as-is so the merger selection stays reconstructable.

    This exists so a downstream cut (the cold-stability ``m_i <= M_TOV``
    filter) is applied to the whole dict at one call site.  Masking
    ``m1`` / ``m2`` by hand and forgetting ``fc_mt_p1``, ``v_sys``, or
    ``metallicity`` would silently misalign the per-channel and offset
    arrays against the masses; routing every column through here makes
    that class of bug impossible.
    """
    mask = np.asarray(mask, dtype=bool)
    n = d.get("n_merging")
    out = {}
    for k, v in d.items():
        if k == "mask_merging":
            out[k] = v
        elif isinstance(v, np.ndarray) and n is not None and v.shape[:1] == (n,):
            out[k] = v[mask]
        else:
            out[k] = v
    out["n_merging"] = int(mask.sum())
    return out


def verify_shared_metallicity_prior(path_a, path_b):
    """Assert two COMPAS files use the same birth-metallicity prior.

    Both populations must span the same Z range for a single p_draw
    to be valid.  Returns the shared (Z_min, Z_max) tuple.
    """
    r_a = read_metallicity_range(path_a)
    r_b = read_metallicity_range(path_b)
    if r_a != r_b:
        raise ValueError(
            f"Metallicity ranges differ: {r_a} vs {r_b}; "
            "need separate dPdlogZ / p_draw per population"
        )
    return r_a


# ═══════════════════════════════════════════════════════════════════════════
# Simulation metadata
# ═══════════════════════════════════════════════════════════════════════════
def read_expected_local_rate(path):
    """Read the pre-computed local (z=0) intrinsic merger rate from a
    Broekgaarden et al. COMPAS HDF5 file.

    Returns the sum of the ``weights_intrinsic/w_000`` column, which
    represents the fiducial-model merger rate density at z = 0 in
    Gpc^-3 yr^-1.  This value is used to calibrate the per-population
    ``MEAN_MASS_EVOLVED`` normalization constant.
    """
    with h5.File(path, "r") as f:
        return float(f["weights_intrinsic"]["w_000"][...].sum())


def read_metallicity_range(path):
    """Return (Z_min, Z_max) of birth metallicities in the HDF5 file."""
    with h5.File(path, "r") as f:
        Z = f["systems"]["Metallicity1"][...].squeeze()
    return float(Z.min()), float(Z.max())


# ═══════════════════════════════════════════════════════════════════════════
# BNS data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_bns(path=None, sort_masses=True, expected_model=None, expected_ns_max=None):
    """Load merging BNS population from COMPAS HDF5 file.

    Parameters
    ----------
    path : str, optional
        Path to HDF5 file. Defaults to data/COMPASCompactOutput_BNS_A.h5.
    sort_masses : bool
        If True, m1 >= m2 is enforced (heavier/lighter).
    expected_model : str, optional
        Broekgaarden+ 2021 model letter ('A', 'J', 'K').  When the HDF5
        carries the embedded ``model`` attribute (see
        ``tools/embed_model_metadata.py``), the loader validates and
        raises if the file is mislabeled.  When the attribute is
        absent, a one-time UserWarning is emitted and the value is
        ignored (backward compatible with un-annotated archives).
    expected_ns_max : float, optional
        Validates the ``ns_max`` attribute the same way as
        ``expected_model`` (Models J / A / K = 2.0 / 2.5 / 3.0 Msun).

    Returns
    -------
    dict with keys:
        'm1', 'm2'         : component masses [Msun]
        'weights'          : STROOPWAFEL weights
        'metallicity'      : birth metallicity (Metallicity1)
        'delay_time'       : tform + tc [Myr]
        'n_merging'        : number of merging systems
        'mask_merging'     : boolean mask over the full catalogue
        'model'            : Broekgaarden+ 2021 model letter (or None
                             if the file carries no embedded metadata)
        'ns_max'           : maximum NS gravitational mass for this run
                             (Models J / A / K = 2.0 / 2.5 / 3.0 Msun).
                             Pass directly into ``classify_grid(ns_max=...)``
                             rather than maintaining a per-letter literal.
    """
    if path is None:
        path = DEFAULT_BNS_PATH

    actual_model, actual_ns_max = _validate_hdf5_metadata(
        path, expected_kind="BNS", expected_model=expected_model, expected_ns_max=expected_ns_max
    )

    with h5.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        m1 = dco["M1"][...].squeeze()
        m2 = dco["M2"][...].squeeze()
        w = dco["weight"][...].squeeze()
        Z = dco["Metallicity1"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()
        tc = dco["tc"][...].squeeze()
        tform = dco["tform"][...].squeeze()

    mask = mh == 1
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bns")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    m1_m, m2_m = m1[mask], m2[mask]
    if sort_masses:
        m1_out = np.maximum(m1_m, m2_m)
        m2_out = np.minimum(m1_m, m2_m)
    else:
        m1_out, m2_out = m1_m, m2_m

    out = {
        "m1": m1_out,
        "m2": m2_out,
        "weights": w_out,
        "metallicity": Z[mask],
        "delay_time": delay,
        "n_merging": int(mask.sum()),
        "mask_merging": mask,
        "population": "BNS",
        "model": actual_model,
        "ns_max": actual_ns_max,
    }
    _validate_loader_dict(out, out["n_merging"], path)
    return out


def load_bns_with_channels(path=None, sort_masses=True, expected_model=None, expected_ns_max=None):
    """Load merging BNS with formation-channel columns.

    Returns the same dict as load_bns, plus additional keys for the
    formation-channel classification:
        'dblCE', 'fc_CEE', 'fc_mt_p1', 'fc_mt_s1',
        'fc_mt_p1_K1', 'fc_mt_s1_K2',
        'm1zams', 'm2zams', 'sep_preCE', 'sep_postCE'

    ``expected_model`` / ``expected_ns_max`` behave as in ``load_bns``.
    """
    if path is None:
        path = DEFAULT_BNS_PATH

    actual_model, actual_ns_max = _validate_hdf5_metadata(
        path, expected_kind="BNS", expected_model=expected_model, expected_ns_max=expected_ns_max
    )

    with h5.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        m1 = dco["M1"][...].squeeze()
        m2 = dco["M2"][...].squeeze()
        w = dco["weight"][...].squeeze()
        Z = dco["Metallicity1"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()
        tc = dco["tc"][...].squeeze()
        tform = dco["tform"][...].squeeze()
        m1z = dco["M1ZAMS"][...].squeeze()
        m2z = dco["M2ZAMS"][...].squeeze()
        dblCE = dco["doubleCommonEnvelopeFlag"][...].squeeze()
        sep_pre = dco["SemiMajorAxisPreCEE"][...].squeeze()
        sep_post = dco["SemiMajorAxisPostCEE"][...].squeeze()
        dco_seed = dco["seed"][...].squeeze()

        fc = f["formationChannels"]
        fc_seed = fc["m_randomSeed"][...].squeeze()
        fc_mt_p1 = fc["mt_primary_ep1"][...].squeeze()
        fc_mt_p1_K1 = fc["mt_primary_ep1_K1"][...].squeeze()
        fc_mt_s1 = fc["mt_secondary_ep1"][...].squeeze()
        fc_mt_s1_K2 = fc["mt_secondary_ep1_K2"][...].squeeze()
        fc_CEE = fc["CEE"][...].squeeze()

    # Align channel columns to DCO rows by seed (positional join fails when the
    # channel group carries extra/duplicate rows, e.g. the no-PISN Model O).
    _fc_row = _fc_to_dco_index(fc_seed, dco_seed, path)
    fc_mt_p1 = fc_mt_p1[_fc_row]
    fc_mt_p1_K1 = fc_mt_p1_K1[_fc_row]
    fc_mt_s1 = fc_mt_s1[_fc_row]
    fc_mt_s1_K2 = fc_mt_s1_K2[_fc_row]
    fc_CEE = fc_CEE[_fc_row]

    mask = mh == 1
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bns_with_channels")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    m1_m, m2_m = m1[mask], m2[mask]
    if sort_masses:
        m1_out = np.maximum(m1_m, m2_m)
        m2_out = np.minimum(m1_m, m2_m)
    else:
        m1_out, m2_out = m1_m, m2_m

    # NOTE: sort_masses swaps m1/m2 so that m1 >= m2, but it does NOT
    # reorder the formation-channel columns (dblCE, fc_*, sep_*, ZAMS
    # masses).  Those columns still reference the *original* COMPAS
    # primary/secondary labelling, not the heavier/lighter compact object.
    # This is correct for formation-channel classification (which follows
    # COMPAS primary/secondary), but callers must not assume that
    # fc_mt_p1 corresponds to the heavier compact remnant.
    out = {
        "m1": m1_out,
        "m2": m2_out,
        "weights": w_out,
        "metallicity": Z[mask],
        "delay_time": delay,
        "n_merging": int(mask.sum()),
        "mask_merging": mask,
        "m1zams": m1z[mask],
        "m2zams": m2z[mask],
        "dblCE": dblCE[mask],
        "sep_preCE": sep_pre[mask],
        "sep_postCE": sep_post[mask],
        "fc_mt_p1": fc_mt_p1[mask],
        "fc_mt_p1_K1": fc_mt_p1_K1[mask],
        "fc_mt_s1": fc_mt_s1[mask],
        "fc_mt_s1_K2": fc_mt_s1_K2[mask],
        "fc_CEE": fc_CEE[mask],
        "population": "BNS",
        "model": actual_model,
        "ns_max": actual_ns_max,
    }
    _validate_loader_dict(out, out["n_merging"], path)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# BHNS data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_bhns(path=None, expected_model=None, expected_ns_max=None):
    """Load merging BHNS population from COMPAS HDF5 file.

    BH/NS identity is resolved using stellarType1 (14 = BH).

    Parameters
    ----------
    path : str, optional
        HDF5 path; defaults to data/COMPASCompactOutput_BHNS_A.h5.
    expected_model : str, optional
        See ``load_bns``.  When the embedded ``model`` attribute is
        present and differs, raises ``ValueError``.
    expected_ns_max : float, optional
        See ``load_bns``.

    Returns
    -------
    dict with keys:
        'M_BH', 'M_NS'    : component masses [Msun]
        'weights'          : STROOPWAFEL weights
        'metallicity'      : birth metallicity (Metallicity1)
        'delay_time'       : tform + tc [Myr]
        'n_merging'        : number of merging systems
        'mask_merging'     : boolean mask over the full catalogue
        'model'            : Broekgaarden+ 2021 model letter (or None)
        'ns_max'           : maximum NS gravitational mass for the run
    """
    if path is None:
        path = DEFAULT_BHNS_PATH

    actual_model, actual_ns_max = _validate_hdf5_metadata(
        path, expected_kind="BHNS", expected_model=expected_model, expected_ns_max=expected_ns_max
    )

    with h5.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        m1 = dco["M1"][...].squeeze()
        m2 = dco["M2"][...].squeeze()
        w = dco["weight"][...].squeeze()
        Z = dco["Metallicity1"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()
        tc = dco["tc"][...].squeeze()
        tform = dco["tform"][...].squeeze()
        st1 = dco["stellarType1"][...].squeeze()

    is_BH1 = st1 == 14
    M_BH = np.where(is_BH1, m1, m2)
    M_NS = np.where(is_BH1, m2, m1)

    mask = mh == 1
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bhns")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    out = {
        "M_BH": M_BH[mask],
        "M_NS": M_NS[mask],
        "weights": w_out,
        "metallicity": Z[mask],
        "delay_time": delay,
        "n_merging": int(mask.sum()),
        "mask_merging": mask,
        "population": "BHNS",
        "model": actual_model,
        "ns_max": actual_ns_max,
    }
    _validate_loader_dict(out, out["n_merging"], path)
    return out


def load_bhns_with_channels(path=None, expected_model=None, expected_ns_max=None):
    """Load merging BHNS with formation-channel columns.

    Same as load_bhns plus formation-channel keys for classify_formation_channels.
    ``expected_model`` / ``expected_ns_max`` behave as in ``load_bhns``.
    """
    if path is None:
        path = DEFAULT_BHNS_PATH

    actual_model, actual_ns_max = _validate_hdf5_metadata(
        path, expected_kind="BHNS", expected_model=expected_model, expected_ns_max=expected_ns_max
    )

    with h5.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        m1 = dco["M1"][...].squeeze()
        m2 = dco["M2"][...].squeeze()
        w = dco["weight"][...].squeeze()
        Z = dco["Metallicity1"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()
        tc = dco["tc"][...].squeeze()
        tform = dco["tform"][...].squeeze()
        st1 = dco["stellarType1"][...].squeeze()
        dblCE = dco["doubleCommonEnvelopeFlag"][...].squeeze()
        sep_pre = dco["SemiMajorAxisPreCEE"][...].squeeze()
        _seed_key = "seed" if "seed" in dco else "m_randomSeed"
        dco_seed = dco[_seed_key][...].squeeze()

        fc = f["formationChannels"]
        fc_seed = fc["m_randomSeed"][...].squeeze()
        fc_mt_p1 = fc["mt_primary_ep1"][...].squeeze()
        fc_mt_p1_K1 = fc["mt_primary_ep1_K1"][...].squeeze()
        fc_mt_s1 = fc["mt_secondary_ep1"][...].squeeze()
        fc_mt_s1_K2 = fc["mt_secondary_ep1_K2"][...].squeeze()
        fc_CEE = fc["CEE"][...].squeeze()

    # Align channel columns to DCO rows by seed (positional join fails when the
    # channel group carries extra/duplicate rows, e.g. the no-PISN Model O).
    _fc_row = _fc_to_dco_index(fc_seed, dco_seed, path)
    fc_mt_p1 = fc_mt_p1[_fc_row]
    fc_mt_p1_K1 = fc_mt_p1_K1[_fc_row]
    fc_mt_s1 = fc_mt_s1[_fc_row]
    fc_mt_s1_K2 = fc_mt_s1_K2[_fc_row]
    fc_CEE = fc_CEE[_fc_row]

    is_BH1 = st1 == 14
    M_BH = np.where(is_BH1, m1, m2)
    M_NS = np.where(is_BH1, m2, m1)

    mask = mh == 1
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bhns_with_channels")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    out = {
        "M_BH": M_BH[mask],
        "M_NS": M_NS[mask],
        "weights": w_out,
        "metallicity": Z[mask],
        "delay_time": delay,
        "n_merging": int(mask.sum()),
        "mask_merging": mask,
        "dblCE": dblCE[mask],
        "sep_preCE": sep_pre[mask],
        "fc_mt_p1": fc_mt_p1[mask],
        "fc_mt_p1_K1": fc_mt_p1_K1[mask],
        "fc_mt_s1": fc_mt_s1[mask],
        "fc_mt_s1_K2": fc_mt_s1_K2[mask],
        "fc_CEE": fc_CEE[mask],
        "population": "BHNS",
        "model": actual_model,
        "ns_max": actual_ns_max,
    }
    _validate_loader_dict(out, out["n_merging"], path)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# COMPAS metallicity grid
# ═══════════════════════════════════════════════════════════════════════════
# 53-element grid extracted from the Broekgaarden et al. (2021) Zenodo
# files (5189849 / 5178777, Models A and K, BNS and BHNS).  All four
# archives share this identical grid; verified by
# ``tests/integration/test_grb_io_realdata.py::test_metallicity_grid_matches_data``.
# The grid has 53 unique values (no duplicate at 0.03); do not
# reintroduce one when copy-editing.
METALLICITY_GRID = np.array(
    [
        0.0001,
        0.00011,
        0.00012,
        0.00014,
        0.00016,
        0.00017,
        0.00019,
        0.00022,
        0.00024,
        0.00027,
        0.0003,
        0.00034,
        0.00037,
        0.00042,
        0.00047,
        0.00052,
        0.00058,
        0.00065,
        0.00073,
        0.00081,
        0.0009,
        0.00101,
        0.00113,
        0.00126,
        0.0014,
        0.00157,
        0.00175,
        0.00195,
        0.00218,
        0.00243,
        0.00272,
        0.00303,
        0.00339,
        0.00378,
        0.00422,
        0.00471,
        0.00526,
        0.00587,
        0.00655,
        0.00732,
        0.00817,
        0.00912,
        0.01018,
        0.01137,
        0.01269,
        0.01416,
        0.01581,
        0.01765,
        0.01971,
        0.022,
        0.0244,
        0.02705,
        0.03,
    ]
)


# ═══════════════════════════════════════════════════════════════════════════
# Observed BNS gravitational-wave events
# ═══════════════════════════════════════════════════════════════════════════
# Component-mass anchors under the *low-spin* (|chi| < 0.05) prior from
# the LIGO/Virgo discovery papers, for figure annotations and quick sanity
# checks against classification thresholds; use the full posteriors
# (GWOSC) for any quantitative population work.  The published tables
# report 90% credible intervals rather than tabulated single-value
# medians for the individual component masses (the marginal posteriors
# are broad chirp-mass bananas, so a single "median M1" is not a well
# defined column), so each entry carries both the 90% CR (``M[12]_lo``
# and ``M[12]_hi``) and a representative central (M1, M2) point inside
# the ranges; annotations that draw the point without error bars will
# overstate how sharply the individual masses are measured.
OBSERVED_GW_EVENTS = {
    "GW170817": {
        # Representative central values inside the low-spin 90% CR,
        # consistent with the GWTC-1 catalogue (Abbott et al. 2019,
        # PRX 9, 031040); not directly tabulated in the discovery paper.
        "M1": 1.46,
        "M2": 1.27,
        # Low-spin (|chi| < 0.05) 90% credible ranges on the individual
        # component masses (Abbott et al. 2019, PRX 9, 011001, Table I).
        "M1_lo": 1.36,
        "M1_hi": 1.60,
        "M2_lo": 1.17,
        "M2_hi": 1.36,
        # Source-frame chirp mass and its 90% CR half-width, which is
        # the actually well-measured combination that defines the shape
        # of the joint 2D posterior (a thin banana along Mc = const in
        # the (M2, M1) plane).
        "Mc": 1.186,
        "Mc_err_90": 0.001,
        "reference": (
            "Low-spin (|chi| < 0.05) 90% CR from the discovery paper "
            "(Abbott et al. 2019, PRX 9, 011001, Table I); the plotted "
            "(M1, M2) are representative central values inside those "
            "ranges, consistent with the GWTC-1 catalogue."
        ),
    },
    "GW190425": {
        # Low-spin 90% CR midpoint on each component; Table 1 reports
        # intervals rather than tabulated single-value component-mass
        # estimates, so this is not a "median" in any strict sense.  Sits
        # ~0.02 Msun from the GWOSC low-spin 1D posterior medians.
        "M1": 1.73,
        "M2": 1.58,
        # Low-spin (|chi| < 0.05) 90% credible ranges on the individual
        # component masses (Abbott et al. 2020, ApJL 892, L3, Table 1).
        "M1_lo": 1.60,
        "M1_hi": 1.87,
        "M2_lo": 1.46,
        "M2_hi": 1.69,
        # Source-frame chirp mass and its 90% CR half-width; the joint
        # 2D posterior is a thin banana along Mc = const in (M2, M1).
        "Mc": 1.44,
        "Mc_err_90": 0.02,
        "reference": (
            "Low-spin (|chi| < 0.05) 90% CR from Abbott et al. 2020, "
            "ApJL 892, L3, Table 1; the plotted (M1, M2) is the CR "
            "midpoint since Table 1 reports intervals rather than "
            "tabulated component-mass medians (~0.02 Msun from the "
            "GWOSC low-spin 1D posterior medians)."
        ),
    },
}


# Observed NSBH GW events for annotation on the Section 1 BHNS mass
# plane.  Values are the published medians with 90% CRs on the individual
# component masses (Abbott et al. 2021, ApJL 915, L5, arXiv:2106.15163,
# Table I).  The paper defaults to a "high-spin" prior on the primary BH
# (a1 up to 0.99) and a "low-spin" prior on the secondary NS (a2 < 0.05);
# the medians pinned here are from that default column, which is what
# tables and figures throughout the discovery paper reference.  The two
# events are used only as figure annotations on the BHNS panel; anything
# quantitative should use the full posteriors from GWOSC.
OBSERVED_NSBH_EVENTS = {
    "GW200105": {
        # Published median from the discovery-paper Table I (high-spin
        # primary / low-spin secondary prior).
        "M_BH": 8.9,
        "M_NS": 1.9,
        # 90% credible ranges on the individual component masses
        # (Abbott et al. 2021, ApJL 915, L5, Table I).
        "M_BH_lo": 7.4,
        "M_BH_hi": 10.1,
        "M_NS_lo": 1.7,
        "M_NS_hi": 2.2,
        "reference": (
            "High-spin primary / low-spin secondary prior median with "
            "90% CR from Abbott et al. 2021, ApJL 915, L5, Table I "
            "(GW200105_162426)."
        ),
    },
    "GW200115": {
        # Published median from the discovery-paper Table I.
        "M_BH": 5.7,
        "M_NS": 1.5,
        # 90% credible ranges on the individual component masses.
        "M_BH_lo": 3.6,
        "M_BH_hi": 7.5,
        "M_NS_lo": 1.2,
        "M_NS_hi": 2.2,
        "reference": (
            "High-spin primary / low-spin secondary prior median with "
            "90% CR from Abbott et al. 2021, ApJL 915, L5, Table I "
            "(GW200115_042309)."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Broekgaarden+ 2021 model-variation grid
# ═══════════════════════════════════════════════════════════════════════════
# The 20 population-synthesis variations of Broekgaarden et al. (2021),
# arXiv:2103.02608 (Table 2; physics knobs in Sec. 3-5) and the paper-II
# companion arXiv:2112.05763.  Keyed by the project's on-disk filename suffix
# (the paper-I letter where the variation exists in paper I, otherwise a
# descriptive token).  This must stay consistent with ``_GRID`` in
# ``tools/download_compas_data.py``; the agreement is pinned by
# ``tests/unit/test_grb_io.py::test_registry_matches_downloader_grid``.
#
# Each entry records only the knob that the variation moves relative to the
# fiducial Model A: ``alpha_ce`` carries the common-envelope efficiency for
# every model (fiducial 1.0; Fryer 2012 *delayed* engine throughout except
# Model I), while ``beta`` / ``sigma_kick_kms`` / ``f_WR`` are ``None`` unless
# that specific knob is the one being varied, so the grid does not pin a
# fiducial value the paper leaves implicit (e.g. the adaptive mass-transfer
# efficiency or the Hobbs+ 2005 natal-kick sigma).
BROEKGAARDEN21_MODELS: dict[str, dict] = {
    "A": {
        "description": "Fiducial (alpha_CE=1.0, M_NS,max=2.5, delayed engine)",
        "family": "fiducial",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "B": {
        "description": "Mass-transfer efficiency beta=0.25",
        "family": "mt_efficiency",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": 0.25,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "C": {
        "description": "Mass-transfer efficiency beta=0.5",
        "family": "mt_efficiency",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": 0.5,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "D": {
        "description": "Mass-transfer efficiency beta=0.75",
        "family": "mt_efficiency",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": 0.75,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "E": {
        "description": "Unstable case BB mass transfer",
        "family": "case_bb",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "EH": {
        "description": "Unstable case BB plus optimistic CE (paper-II only)",
        "family": "case_bb",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "alpha0p1": {
        "description": "alpha_CE = 0.1 (paper-II only)",
        "family": "alpha_ce",
        "ns_max": 2.5,
        "alpha_ce": 0.1,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "F": {
        "description": "alpha_CE = 0.5",
        "family": "alpha_ce",
        "ns_max": 2.5,
        "alpha_ce": 0.5,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "G": {
        "description": "alpha_CE = 2.0",
        "family": "alpha_ce",
        "ns_max": 2.5,
        "alpha_ce": 2.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "alpha10": {
        "description": "alpha_CE = 10 (paper-II only)",
        "family": "alpha_ce",
        "ns_max": 2.5,
        "alpha_ce": 10.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "H": {
        "description": "Optimistic CE (HG donors survive CE)",
        "family": "ce_survival",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "I": {
        "description": "Rapid SN remnant prescription",
        "family": "sn_engine",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "rapid",
    },
    "J": {
        "description": "Maximum NS mass = 2.0 Msun",
        "family": "ns_mass_cap",
        "ns_max": 2.0,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "K": {
        "description": "Maximum NS mass = 3.0 Msun",
        "family": "ns_mass_cap",
        "ns_max": 3.0,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "L": {
        "description": "No (pulsational) PISN",
        "family": "pisn",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "M": {
        "description": "Core-collapse SN kick sigma = 100 km/s",
        "family": "sn_kick",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": 100.0,
        "f_WR": None,
        "engine": "delayed",
    },
    "N": {
        "description": "Core-collapse SN kick sigma = 30 km/s",
        "family": "sn_kick",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": 30.0,
        "f_WR": None,
        "engine": "delayed",
    },
    "O": {
        "description": "No BH natal kick (Blaauw kick only)",
        "family": "bh_kick",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": None,
        "engine": "delayed",
    },
    "fWR0p1": {
        "description": "Wolf-Rayet wind multiplier f_WR = 0.1 (paper-II only)",
        "family": "wr_winds",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": 0.1,
        "engine": "delayed",
    },
    "fWR5": {
        "description": "Wolf-Rayet wind multiplier f_WR = 5 (paper-II only)",
        "family": "wr_winds",
        "ns_max": 2.5,
        "alpha_ce": 1.0,
        "beta": None,
        "sigma_kick_kms": None,
        "f_WR": 5.0,
        "engine": "delayed",
    },
}

# Grid-scan order follows the registry (BROEKGAARDEN21_MODELS insertion order);
# the per-model Alsing-remap RNG seeds (42 + i for BNS, 43 + i for BHNS) are
# keyed off this index.
ALL_MODEL_SUFFIXES: list[str] = list(BROEKGAARDEN21_MODELS)

# Paper I suffix -> Broekgaarden+ 2022 paper II letter. The Zenodo records
# (5189849 BNS, 5178777 BHNS) package every catalog under the paper II letter,
# so figures keyed to the data release use these labels; the two conventions
# diverge after model E.
BROEKGAARDEN22_PAPER_II_LETTERS: dict[str, str] = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "EH": "F",
    "alpha0p1": "G",
    "F": "H",
    "G": "I",
    "alpha10": "J",
    "H": "K",
    "I": "L",
    "J": "M",
    "K": "N",
    "L": "O",
    "M": "P",
    "N": "Q",
    "O": "R",
    "fWR0p1": "S",
    "fWR5": "T",
}


# ═══════════════════════════════════════════════════════════════════════════
# Export helpers
# ═══════════════════════════════════════════════════════════════════════════
def save_efficiencies(filepath, arrays, labels=None):
    """Save efficiency arrays as a single .npy stack."""
    stack = np.array(arrays)
    np.save(filepath, stack)
    if labels:
        print(f"Saved {filepath}: rows = {labels}")


def save_rates(filepath, redshifts, rate_dict):
    """Save merger rate arrays (redshifts + named rates) as .npy stack."""
    rows = [redshifts] + [rate_dict[k] for k in rate_dict]
    labels = ["redshifts"] + list(rate_dict.keys())
    np.save(filepath, np.array(rows))
    print(f"Saved {filepath}: rows = {labels}")


# ═══════════════════════════════════════════════════════════════════════════
# Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════
def weighted_sample(mask, weight, n_target=12000, rng=None):
    """Subsample indices weighted by STROOPWAFEL weight for scatter plots."""
    if rng is None:
        rng = np.random.default_rng(42)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return idx
    w = weight[idx]
    if w.sum() == 0:
        return idx[: min(n_target, len(idx))]
    w = w / w.sum()
    n = min(n_target, len(idx))
    return rng.choice(idx, size=n, replace=False, p=w)


def log_jitter(Z, scale=0.04, rng=None):
    """Add small log-uniform jitter to metallicity values for scatter plots."""
    if rng is None:
        rng = np.random.default_rng(42)
    return Z * 10 ** rng.uniform(-scale, scale, size=len(Z))


def _fc_to_dco_index(fc_seed, dco_seed, path):
    """Row index into formationChannels for each doubleCompactObjects row,
    matched by random seed.  The groups are positionally aligned in most
    files, but some variations (e.g. the no-PISN Model O) write extra or
    duplicate channel rows, so a positional join misaligns or raises.  Join on
    seed, first occurrence per seed (duplicate channel rows in those files are
    identical in the classification columns)."""
    u_seed, first = np.unique(fc_seed, return_index=True)
    pos = np.clip(np.searchsorted(u_seed, dco_seed), 0, u_seed.size - 1)
    missing = u_seed[pos] != dco_seed
    if np.any(missing):
        raise KeyError(
            f"{os.path.basename(path)}: {int(missing.sum())} DCO seeds absent "
            f"from formationChannels; cannot seed-join."
        )
    return first[pos]


# ═══════════════════════════════════════════════════════════════════════════
# Kick / velocity loading (for kinematic offset analysis)
# ═══════════════════════════════════════════════════════════════════════════
def _match_sn_to_dco(f):
    """Join the per-supernova systemic velocities and times to DCO rows.

    The supernovae group holds two rows per system, one per SN.  Ordered by
    event time within each seed, the first row is the post-SN1 state and the
    second is the post-SN2 state that forms the DCO.  ``systemicVelocity`` is
    the cumulative center-of-mass speed after each kick, so ``v_sys_2`` is the
    post-second-SN value the older code returned as ``v_sys``.

    Returns a dict of full-length (pre-merger-mask) arrays aligned to the DCO
    table:

        ``v_sys_1`` / ``v_sys_2`` : COM speed [km/s] after SN1 / SN2.
        ``t_sn1`` / ``t_sn2``     : time of SN1 / SN2 [Myr].

    A DCO seed with no matching SN pair gets NaN in all four (dropped
    downstream via ``np.isfinite``).  A matched DCO seed whose SN group does
    not hold exactly two rows raises, because the two-leg kinematic treatment
    in ``grb_offsets`` depends on the SN1/SN2 pairing.

    Vectorized with numpy for performance on large populations.
    """
    sn = f["supernovae"]
    sn_seed = sn["randomSeed"][...].squeeze()
    sn_vsys = sn["systemicVelocity"][...].squeeze()
    sn_time = sn["time"][...].squeeze()

    dco = f["doubleCompactObjects"]
    dco_seed_key = "seed" if "seed" in dco else "m_randomSeed"
    dco_seed = dco[dco_seed_key][...].squeeze()

    order = np.lexsort((sn_time, sn_seed))
    sn_seed_s = sn_seed[order]
    sn_vsys_s = sn_vsys[order]
    sn_time_s = sn_time[order]

    u_seeds, first_idx, counts = np.unique(sn_seed_s, return_index=True, return_counts=True)
    n_seeds = len(u_seeds)

    # Per-seed SN1/SN2 arrays.  Only seeds with exactly two SN rows are filled;
    # a positional (idx, idx+1) read for any other count would straddle the
    # next seed's rows, so those stay NaN and are caught at match time below.
    v1 = np.full(n_seeds, np.nan)
    v2 = np.full(n_seeds, np.nan)
    t1 = np.full(n_seeds, np.nan)
    t2 = np.full(n_seeds, np.nan)
    two = counts == 2
    idx2 = first_idx[two]
    v1[two], t1[two] = sn_vsys_s[idx2], sn_time_s[idx2]
    v2[two], t2[two] = sn_vsys_s[idx2 + 1], sn_time_s[idx2 + 1]

    # A matched seed contributes SN1/SN2 states only if it carries exactly two
    # supernova rows.  Seeds with any other count (a few per model in the full
    # grid, from bookkeeping gaps in the supernovae group) are left NaN and
    # dropped downstream via np.isfinite, exactly like an unmatched seed; the
    # two-leg treatment needs a clean SN1/SN2 pair.
    match = np.searchsorted(u_seeds, dco_seed)
    match_clipped = np.clip(match, 0, n_seeds - 1)
    paired = (match < n_seeds) & (u_seeds[match_clipped] == dco_seed) & (counts[match_clipped] == 2)
    src = match_clipped[paired]

    def _gather(per_seed):
        out = np.full(len(dco_seed), np.nan)
        out[paired] = per_seed[src]
        return out

    n_missing = int((~paired).sum())
    if n_missing > 0:
        # NaN in v_sys silently disappears via np.isfinite(v_sys_km) in
        # compute_offsets_population (grb_offsets.py); surface it at
        # load time so the count is auditable in the notebook log.
        warnings.warn(
            f"_match_sn_to_dco ({os.path.basename(f.filename)}): "
            f"{n_missing} of {len(dco_seed)} DCO seeds had no matching "
            f"supernova pair; v_sys is NaN for those rows.  Downstream offset "
            f"analysis will silently drop them via np.isfinite() in "
            f"compute_offsets_population.",
            stacklevel=2,
        )

    return {
        "v_sys_1": _gather(v1),
        "v_sys_2": _gather(v2),
        "t_sn1": _gather(t1),
        "t_sn2": _gather(t2),
    }


def load_bns_with_kicks(path=None, sort_masses=True, expected_model=None, expected_ns_max=None):
    """Load merging BNS with kick/velocity columns for offset analysis.

    Returns the standard load_bns dict plus:
        'drawnKick1', 'drawnKick2', 'v_sys', 'sep_DCO', 'ecc_DCO'

    For the two-leg kinematic treatment (drift at the post-SN1 velocity until
    the second supernova, then at the post-SN2 velocity through the inspiral)
    it also returns:
        'v_sys_1', 'v_sys_2' : COM speed [km/s] after SN1 / SN2
        't_sn1', 't_sn2'     : time of SN1 / SN2 [Myr]
        't_inspiral'         : ``tc`` [Myr], the post-DCO gravitational-wave
                               inspiral time
    ``v_sys`` is kept as an alias of ``v_sys_2`` for backward compatibility.

    ``expected_model`` / ``expected_ns_max`` behave as in ``load_bns``.
    """
    if path is None:
        path = DEFAULT_BNS_PATH

    actual_model, actual_ns_max = _validate_hdf5_metadata(
        path, expected_kind="BNS", expected_model=expected_model, expected_ns_max=expected_ns_max
    )

    with h5.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        m1 = dco["M1"][...].squeeze()
        m2 = dco["M2"][...].squeeze()
        w = dco["weight"][...].squeeze()
        Z = dco["Metallicity1"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()
        tc = dco["tc"][...].squeeze()
        tform = dco["tform"][...].squeeze()
        dk1 = dco["drawnKick1"][...].squeeze()
        dk2 = dco["drawnKick2"][...].squeeze()
        sep = dco["separationDCOFormation"][...].squeeze()
        ecc = dco["eccentricityDCOFormation"][...].squeeze()

        sn = _match_sn_to_dco(f)

    mask = mh == 1
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bns_with_kicks")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    m1_m, m2_m = m1[mask], m2[mask]
    if sort_masses:
        m1_out = np.maximum(m1_m, m2_m)
        m2_out = np.minimum(m1_m, m2_m)
    else:
        m1_out, m2_out = m1_m, m2_m

    out = {
        "m1": m1_out,
        "m2": m2_out,
        "weights": w_out,
        "metallicity": Z[mask],
        "delay_time": delay,
        "n_merging": int(mask.sum()),
        "mask_merging": mask,
        "drawnKick1": dk1[mask],
        "drawnKick2": dk2[mask],
        "v_sys": sn["v_sys_2"][mask],
        "v_sys_1": sn["v_sys_1"][mask],
        "v_sys_2": sn["v_sys_2"][mask],
        "t_sn1": sn["t_sn1"][mask],
        "t_sn2": sn["t_sn2"][mask],
        "t_inspiral": tc[mask],
        "sep_DCO": sep[mask],
        "ecc_DCO": ecc[mask],
        "population": "BNS",
        "model": actual_model,
        "ns_max": actual_ns_max,
    }
    _validate_loader_dict(out, out["n_merging"], path)
    return out


def load_bhns_with_kicks(path=None, expected_model=None, expected_ns_max=None):
    """Load merging BHNS with kick/velocity columns for offset analysis.

    Returns the same two-leg kinematic columns as ``load_bns_with_kicks``
    (``v_sys_1``, ``v_sys_2``, ``t_sn1``, ``t_sn2``, ``t_inspiral``, with
    ``v_sys`` aliased to ``v_sys_2``).

    ``expected_model`` / ``expected_ns_max`` behave as in ``load_bhns``.
    """
    if path is None:
        path = DEFAULT_BHNS_PATH

    actual_model, actual_ns_max = _validate_hdf5_metadata(
        path, expected_kind="BHNS", expected_model=expected_model, expected_ns_max=expected_ns_max
    )

    with h5.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        m1 = dco["M1"][...].squeeze()
        m2 = dco["M2"][...].squeeze()
        w = dco["weight"][...].squeeze()
        Z = dco["Metallicity1"][...].squeeze()
        mh = dco["mergesInHubbleTimeFlag"][...].squeeze()
        tc = dco["tc"][...].squeeze()
        tform = dco["tform"][...].squeeze()
        st1 = dco["stellarType1"][...].squeeze()
        dk1 = dco["drawnKick1"][...].squeeze()
        dk2 = dco["drawnKick2"][...].squeeze()
        sep = dco["separationDCOFormation"][...].squeeze()
        ecc = dco["eccentricityDCOFormation"][...].squeeze()

        sn = _match_sn_to_dco(f)

    is_BH1 = st1 == 14
    M_BH = np.where(is_BH1, m1, m2)
    M_NS = np.where(is_BH1, m2, m1)
    mask = mh == 1
    delay = (tform + tc)[mask]
    _validate_delay_times(delay, " in load_bhns_with_kicks")

    w_out = w[mask]
    _check_weights_no_nan(w_out, path)

    out = {
        "M_BH": M_BH[mask],
        "M_NS": M_NS[mask],
        "weights": w_out,
        "metallicity": Z[mask],
        "delay_time": delay,
        "n_merging": int(mask.sum()),
        "mask_merging": mask,
        "drawnKick1": dk1[mask],
        "drawnKick2": dk2[mask],
        "v_sys": sn["v_sys_2"][mask],
        "v_sys_1": sn["v_sys_1"][mask],
        "v_sys_2": sn["v_sys_2"][mask],
        "t_sn1": sn["t_sn1"][mask],
        "t_sn2": sn["t_sn2"][mask],
        "t_inspiral": tc[mask],
        "sep_DCO": sep[mask],
        "ecc_DCO": ecc[mask],
        "population": "BHNS",
        "model": actual_model,
        "ns_max": actual_ns_max,
    }
    _validate_loader_dict(out, out["n_merging"], path)
    return out
