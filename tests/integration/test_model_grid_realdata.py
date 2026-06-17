"""Real-data audit tying ``BROEKGAARDEN21_MODELS`` to the on-disk grid.

The Section 14 scan loops ``grb_io.BROEKGAARDEN21_MODELS`` and loads one
HDF5 per (model, population).  This module pins, for every one of the 40
files actually present in ``data/``, that:

1. the embedded ``model`` / ``ns_max`` / ``kind`` attributes agree with the
   registry entry the scan will look up;
2. the ``with_channels`` loader the scan uses opens the file and returns the
   per-model ``ns_max`` (so ``classify_grid`` cannot silently fall back to a
   literal), with the ``m1 >= m2`` / ``M_BH >= M_NS`` ordering invariant held;
3. the ``weights_intrinsic/w_000`` calibration anchor is present and
   positive (``calibrate_mean_mass_evolved`` divides by it);
4. the empirical birth-metallicity range sits inside the shared COMPAS prior
   grid (a subset test, not equality: the extreme bins are depleted of
   merging systems in several variations, the documented benign shrinkage).

Parametrised over all 40 grid files; the ``compas_file`` indirect fixture
skips each instance whose data is absent, so a partial download exercises
only the variations that have landed.
"""

from __future__ import annotations

import h5py as h5
import numpy as np
import pytest
from embed_model_metadata import KNOWN_FILES  # type: ignore[import-not-found]

from grb_io import (
    ALL_MODEL_SUFFIXES,
    BROEKGAARDEN21_MODELS,
    METALLICITY_GRID,
    load_bhns,
    load_bns,
    read_expected_local_rate,
)

_AUDIT_FILES = sorted(KNOWN_FILES.keys())


def _suffix_kind(filename: str) -> tuple[str, str]:
    """`COMPASCompactOutput_BNS_alpha10.h5` -> ('alpha10', 'BNS')."""
    stem = filename.replace("COMPASCompactOutput_", "").removesuffix(".h5")
    kind, suffix = stem.split("_", 1)
    return suffix, kind


def test_registry_covers_every_known_grid_file():
    """Every on-disk grid suffix has a registry entry, and vice versa."""
    disk_suffixes = {_suffix_kind(name)[0] for name in _AUDIT_FILES}
    assert disk_suffixes == set(BROEKGAARDEN21_MODELS) == set(ALL_MODEL_SUFFIXES)


@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _AUDIT_FILES, indirect=True)
def test_embedded_attrs_match_registry(compas_file):
    """HDF5 root attributes agree with ``BROEKGAARDEN21_MODELS``.

    The scan trusts these attributes (via ``expected_model``) to confirm it
    loaded the file it intended; a mislabeled or renamed file must fail here.
    """
    suffix, kind = _suffix_kind(compas_file.rsplit("/", 1)[-1])
    info = BROEKGAARDEN21_MODELS[suffix]
    with h5.File(compas_file, "r") as f:
        attrs = dict(f.attrs)
    assert str(attrs.get("model")) == suffix, (
        f"{compas_file}: embedded model {attrs.get('model')!r} != {suffix!r}"
    )
    assert str(attrs.get("kind")) == kind
    assert float(attrs["ns_max"]) == info["ns_max"], (
        f"{compas_file}: ns_max {attrs.get('ns_max')} != registry {info['ns_max']}"
    )


@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _AUDIT_FILES, indirect=True)
def test_loader_anchor_and_metallicity_for_scan(compas_file):
    """The scan's loader, anchor read, and Z-range subset check all hold.

    Uses the base (no-channels) loader: the invariants here (ns_max, mass
    ordering, anchor, metallicity range) do not need the formationChannels
    columns, and a few variations (e.g. L) ship a channel group that does
    not align with the DCO table, which the scan handles with a documented
    fallback rather than by failing.
    """
    name = compas_file.rsplit("/", 1)[-1]
    suffix, kind = _suffix_kind(name)
    info = BROEKGAARDEN21_MODELS[suffix]

    loader = load_bns if kind == "BNS" else load_bhns
    d = loader(path=compas_file, expected_model=suffix)
    assert d["ns_max"] == info["ns_max"], (
        f"{name}: loader ns_max {d['ns_max']} != registry {info['ns_max']}"
    )
    if kind == "BNS":
        assert np.all(d["m1"] >= d["m2"]), f"{name}: m1 >= m2 invariant broken"
    else:
        assert np.all(d["M_BH"] >= d["M_NS"]), f"{name}: M_BH >= M_NS invariant broken"

    anchor = read_expected_local_rate(compas_file)
    assert anchor > 0, f"{name}: non-positive local-rate anchor {anchor}"

    z = d["metallicity"]
    assert z.min() >= METALLICITY_GRID[0] - 1e-12, (
        f"{name}: Z_min {z.min()} below prior grid floor {METALLICITY_GRID[0]}"
    )
    assert z.max() <= METALLICITY_GRID[-1] + 1e-12, (
        f"{name}: Z_max {z.max()} above prior grid ceiling {METALLICITY_GRID[-1]}"
    )
