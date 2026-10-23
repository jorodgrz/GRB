"""``classify_formation_channels`` reproduces Broekgaarden et al. (2021) system by system.

The reference is their own ``identify_formation_channels`` at the commit
pinned in ``grb_classify.BROEKGAARDEN21_CHANNEL_CODE``, fetched at test time
and checked against a SHA-256 so a moved or edited file cannot pass silently.
Their repository carries no licence, so the file is not vendored here.

Both classifiers run on every merging system of each grid file and the
labels must be identical, not just the weighted fractions: a fraction can
agree while two channels trade systems.  Their channels 1 to 4 are our I to
IV and their 0 ("other") is our V.  The 2022-09 revision of their script adds
channels 5 and 6 for Case A; it is not the paper's scheme and is not the
reference (see the ``BROEKGAARDEN21_CHANNEL_CODE`` note).

Verified 2026-10-21 on all 40 files of the grid: 0 mismatches.
"""

from __future__ import annotations

import hashlib
import importlib.util
import urllib.error
import urllib.request

import h5py
import numpy as np
import pytest

from grb_classify import FORMATION_CHANNEL_INPUTS, classify_formation_channels
from grb_io import ALL_MODEL_SUFFIXES, load_bhns_with_channels, load_bns_with_channels

_COMMIT = "2779b913b1090ea9265cd800bfb25f6cbdab114e"
_URL = (
    "https://raw.githubusercontent.com/FloorBroekgaarden/DCO_FormationChannels/"
    f"{_COMMIT}/Scripts/formation_channels.py"
)
_SHA256 = "009696b9ada46af30d4e1e48c8f0e14832be220ece6d8af7493ff3b4fea25b77"

_THEIR_CODE = {
    "I  Stable MT + CE": 1,
    "II  Stable MT only": 2,
    "III Single-core CE": 3,
    "IV  Double-core CE": 4,
    "V   Other": 0,
}

_FILES = [
    f"COMPASCompactOutput_{kind}_{model}.h5"
    for kind in ("BNS", "BHNS")
    for model in ALL_MODEL_SUFFIXES
]


@pytest.fixture(scope="module")
def broekgaarden21(tmp_path_factory):
    try:
        with urllib.request.urlopen(_URL, timeout=30) as r:
            src = r.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"cannot fetch the Broekgaarden+ 2021 reference code: {exc}")
    assert hashlib.sha256(src).hexdigest() == _SHA256, (
        "the pinned reference file changed content; refusing to compare against it"
    )
    path = tmp_path_factory.mktemp("broekgaarden21") / "formation_channels.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("broekgaarden21_formation_channels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_commit_matches_the_module_constant():
    from grb_classify import BROEKGAARDEN21_CHANNEL_CODE

    assert _COMMIT in BROEKGAARDEN21_CHANNEL_CODE


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("compas_file", _FILES, indirect=True)
def test_channels_match_broekgaarden21_code_for_every_system(compas_file, broekgaarden21):
    loader = load_bns_with_channels if "_BNS_" in compas_file else load_bhns_with_channels
    d = loader(compas_file)
    with h5py.File(compas_file, "r") as f:
        dco = f["doubleCompactObjects"]
        seed = dco["seed" if "seed" in dco else "m_randomSeed"][...].squeeze()
        theirs = broekgaarden21.identify_formation_channels(seed[d["mask_merging"]], f)

    channels = classify_formation_channels(**{k: d[k] for k in FORMATION_CHANNEL_INPUTS})
    ours = np.full(d["n_merging"], -1)
    for name, mask in channels.items():
        ours[mask] = _THEIR_CODE[name]

    bad = ours != theirs
    assert not bad.any(), (
        f"{compas_file}: {int(bad.sum())} of {bad.size} systems disagree; first pairs "
        f"(ours, theirs): {list(zip(ours[bad][:5], theirs[bad][:5]))}"
    )
