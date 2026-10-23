"""The loaders drop failed common-envelope ejections, and that is the right cut.

Broekgaarden+ 2021 (Sec. 2): "we remove binaries where the secondary star fills
its Roche lobe immediately after a CE event, as we treat those events as failed
CE ejections."  The Zenodo files keep those binaries, flagged only in
``commonEnvelopes/immediateRLOFAfterCEE``, and ``weights_intrinsic/w_000``
counts them.  These tests pin three things on the real Model A files:

1. what the flag means: COMPAS sets it when either star's post-CE radius
   reaches its Roche lobe (``BaseBinaryStar::SetPostCEEValues``), and the
   stored radii and Roche lobes reproduce it row for row.  The overflowing star
   is star 1 in nearly every flagged BNS row, so the paper's "secondary" is
   loose wording and a secondary-only cut would remove almost nothing;
2. how much it removes (a third of the merging BNS weight, 38 percent of the
   published BNS local rate);
3. that it is the cut Broekgaarden+ 2026 apply: their per-metallicity channel
   yields (FloorBroekgaarden/DCO_FormationChannels, fetched at test time and
   checked against a SHA-256) are a constant multiple of the public
   STROOPWAFEL weight once these systems are dropped, and of no other variant.

Verified 2026-10-22.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request

import h5py
import numpy as np
import pytest

from grb_io import failed_ce_shares, load_bhns, load_bns

_COMMIT = "ae577d7a3c384517ea7f08ebb2c79b76406c7ba8"
_CSV = {
    "BNS": (
        "formation_channels_per_Z_hdf5weights_NSNS.csv",
        "2dee77e16c915a0651789ecf934b46c13335daea8db85829caf03d099ae9bd78",
    ),
    "BHNS": (
        "formation_channels_per_Z_hdf5weights_BHNS.csv",
        "430f4e05a078b2f202a6ab649241056e17a2db6d1c9d23b2d6cb5a9ce8c26ca7",
    ),
}


def _overflow(path):
    """Per CE row: seed, flag, and whether star 1 / star 2 reach their Roche lobe."""
    with h5py.File(path, "r") as f:
        ce = f["commonEnvelopes"]
        g = {k: ce[k][...].ravel() for k in ce}
    o1 = g["radius1AfterStripping"] >= g["rocheLobe1to2PostCEE"]
    o2 = g["radius2AfterStripping"] >= g["rocheLobe2to1PostCEE"]
    return g["randomSeed"], g["immediateRLOFAfterCEE"].astype(bool), o1, o2


@pytest.mark.requires_data
@pytest.mark.parametrize(
    "kind, n_flag, n_star1_only, n_star2_only, n_both",
    [("BNS", 92874, 92860, 1, 13), ("BHNS", 3678, 3088, 20, 570)],
)
def test_flag_is_either_star_filling_its_roche_lobe_after_the_ce(
    kind, n_flag, n_star1_only, n_star2_only, n_both, request
):
    path = request.getfixturevalue("bns_a_path" if kind == "BNS" else "bhns_a_path")
    _, flag, o1, o2 = _overflow(path)
    np.testing.assert_array_equal(o1 | o2, flag)
    assert int(flag.sum()) == n_flag
    assert int((o1 & ~o2).sum()) == n_star1_only
    assert int((o2 & ~o1).sum()) == n_star2_only
    assert int((o1 & o2).sum()) == n_both


@pytest.mark.requires_data
@pytest.mark.parametrize(
    "kind, weight_share, w000_share", [("BNS", 0.3378, 0.3835), ("BHNS", 0.0118, 0.0020)]
)
def test_failed_ce_share_of_model_a(kind, weight_share, w000_share, request):
    path = request.getfixturevalue("bns_a_path" if kind == "BNS" else "bhns_a_path")
    got_w, got_r = failed_ce_shares(path)
    assert got_w == pytest.approx(weight_share, abs=1e-4)
    assert got_r == pytest.approx(w000_share, abs=1e-4)
    loader = load_bns if kind == "BNS" else load_bhns
    d, raw = loader(path), loader(path, drop_failed_ce=False)
    assert d["failed_ce_frac"] == got_w
    assert d["weights"].sum() == pytest.approx((1.0 - got_w) * raw["weights"].sum(), rel=1e-12)


def _fetch_csv(name, sha256):
    url = (
        "https://raw.githubusercontent.com/FloorBroekgaarden/DCO_FormationChannels/"
        f"{_COMMIT}/dataFiles/data_Fig_1/{name}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"cannot fetch the Broekgaarden+ 2026 table: {exc}")
    assert hashlib.sha256(raw).hexdigest() == sha256, "pinned table changed content"
    pd = pytest.importorskip("pandas")
    return pd.read_csv(io.BytesIO(raw), index_col=0)


@pytest.mark.requires_data
@pytest.mark.parametrize("kind", ["BNS", "BHNS"])
def test_dropping_failed_ce_reproduces_broekgaarden26_per_metallicity_yields(kind, request):
    """Her Model A total yield per grid metallicity, divided by our summed
    weight, must be one constant (her mass normalisation) across all 53 points
    under the cut, and must not be under the alternatives."""
    path = request.getfixturevalue("bns_a_path" if kind == "BNS" else "bhns_a_path")
    df = _fetch_csv(*_CSV[kind])
    z_her = np.array([float(s.replace("_", ".")) for s in df.index])
    her_all = df["A All  [Msun^{-1}]"].to_numpy()

    seed_ce, flag, o1, o2 = _overflow(path)
    with h5py.File(path, "r") as f:
        dco = f["doubleCompactObjects"]
        seed = dco["seed"][...].ravel()
        w = dco["weight"][...].ravel()
        z = dco["Metallicity1"][...].ravel()
        merging = dco["mergesInHubbleTimeFlag"][...].ravel() == 1

    def spread(dropped_rows):
        keep = merging & ~np.isin(seed, seed_ce[dropped_rows])
        ours = np.array([w[keep & np.isclose(z, zi, rtol=1e-3)].sum() for zi in z_her])
        ratio = her_all / ours
        return ratio.max() / ratio.min() - 1.0

    assert spread(flag) < 1e-12
    assert spread(np.zeros_like(flag)) > 0.04
    assert spread(o2) > 0.02
