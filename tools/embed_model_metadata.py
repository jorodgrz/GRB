#!/usr/bin/env python3
"""One-shot helper to embed Broekgaarden+ 2021 model identifiers as HDF5
root attributes on the COMPAS catalogues in ``Data/``.

The Zenodo files do not encode their model identity (A, J, K, ...) or
``ns_max`` value, so filename-only identification would let a renamed
or reprocessed file pass every load -> classify -> rate check silently.

Running this script once per fresh download writes the following
attributes onto the root group of each known catalogue:

    /attrs:
      model         : paper-I letter ('A' through 'O') or descriptor
                      ('alpha0p1', 'alpha10', 'EH', 'fWR0p1', 'fWR5')
                      for paper-II-only variations
      ns_max        : 2.5 (default), 2.0 (model J), or 3.0 (model K)  (Msun)
      mssfr_mu0     : 0.035                                  (Neijssel+ 2019 fiducial)
      zenodo_id     : '10.5281/zenodo.5189849' (BNS) or '...5178777' (BHNS)
      annotated_by  : 'tools/embed_model_metadata.py'
      annotated_at  : ISO timestamp

After this is run, ``grb_io.load_bns`` and ``load_bhns`` will validate
the caller's intent against these attributes (see the matching
loader-level checks).

Usage
-----
    python tools/embed_model_metadata.py                # all known files in Data/
    python tools/embed_model_metadata.py --dry-run      # log only
    python tools/embed_model_metadata.py path/to/file.h5 \\
        --model A --ns-max 2.5 --kind BNS \\
        --zenodo 10.5281/zenodo.5189849

The bulk run is normally chained automatically by
``tools/download_compas_data.py`` after each fresh download; running this
script manually is only needed when annotations need to be regenerated
for files already on disk.

This script is idempotent: rerunning overwrites the same attributes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

# h5py is imported lazily inside ``_annotate`` so that ``--dry-run`` and
# importers that only want the ``KNOWN_FILES`` table (for example
# ``tools/download_compas_data.py``) do not require the conda env to be
# active.

ZENODO_BNS_DOI = "10.5281/zenodo.5189849"
ZENODO_BHNS_DOI = "10.5281/zenodo.5178777"

# Project filename suffix -> (paper-I letter or descriptor, ns_max).
# Paper-I letters cover variations that exist in both Broekgaarden+ 2021
# papers; the five descriptors (alpha0p1, alpha10, EH, fWR0p1, fWR5)
# are paper-II-only knobs with no paper-I namesake.
_GRID = [
    ("A", "A", 2.5),
    ("B", "B", 2.5),
    ("C", "C", 2.5),
    ("D", "D", 2.5),
    ("E", "E", 2.5),
    ("EH", "EH", 2.5),  # paper-II only: unstable BB + opt CE
    ("alpha0p1", "alpha0p1", 2.5),  # paper-II only: alpha_CE = 0.1
    ("F", "F", 2.5),
    ("G", "G", 2.5),
    ("alpha10", "alpha10", 2.5),  # paper-II only: alpha_CE = 10
    ("H", "H", 2.5),
    ("I", "I", 2.5),
    ("J", "J", 2.0),
    ("K", "K", 3.0),
    ("L", "L", 2.5),
    ("M", "M", 2.5),
    ("N", "N", 2.5),
    ("O", "O", 2.5),
    ("fWR0p1", "fWR0p1", 2.5),  # paper-II only: f_WR = 0.1
    ("fWR5", "fWR5", 2.5),  # paper-II only: f_WR = 5
]


def _build_known_files() -> dict:
    """Cross-product the variation grid with both populations.

    Yields 40 entries: 20 variations x {BNS, BHNS}.  ``model`` carries the
    paper-I letter where the variation exists in paper I, otherwise the
    descriptor; this is the value ``grb_io._validate_hdf5_metadata``
    reads back when the loader is given ``expected_model``.
    """
    out = {}
    for kind, doi in (("BNS", ZENODO_BNS_DOI), ("BHNS", ZENODO_BHNS_DOI)):
        for suffix, model_attr, ns_max in _GRID:
            name = f"COMPASCompactOutput_{kind}_{suffix}.h5"
            out[name] = {
                "model": model_attr,
                "ns_max": ns_max,
                "kind": kind,
                "zenodo_id": doi,
            }
    return out


KNOWN_FILES = _build_known_files()

MSSFR_MU0_FIDUCIAL = 0.035  # Neijssel+ 2019


def _annotate(
    path: str, model: str, ns_max: float, kind: str, zenodo_id: str, dry_run: bool = False
) -> None:
    print(f"[{path}] model={model} ns_max={ns_max} kind={kind}")
    if dry_run:
        return
    import h5py  # deferred so --dry-run runs without the conda env active

    with h5py.File(path, "a") as f:
        f.attrs["model"] = model
        f.attrs["ns_max"] = ns_max
        f.attrs["kind"] = kind
        f.attrs["mssfr_mu0"] = MSSFR_MU0_FIDUCIAL
        f.attrs["zenodo_id"] = zenodo_id
        f.attrs["annotated_by"] = "tools/embed_model_metadata.py"
        f.attrs["annotated_at"] = dt.datetime.utcnow().isoformat() + "Z"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "path", nargs="?", default=None, help="Path to a single HDF5 file (default: walk Data/)."
    )
    p.add_argument(
        "--model",
        default=None,
        help="Paper-I letter (A through O) or paper-II-only "
        "descriptor (alpha0p1, alpha10, EH, fWR0p1, fWR5).",
    )
    p.add_argument("--ns-max", type=float, default=None)
    p.add_argument("--kind", choices=("BNS", "BHNS"), default=None)
    p.add_argument(
        "--zenodo", default=None, help="Zenodo DOI for this archive (e.g. 10.5281/zenodo.5189849)."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying files.",
    )
    args = p.parse_args(argv)

    if args.path is not None:
        if any(args.model is None for _ in (args.model, args.ns_max, args.kind, args.zenodo)):
            if (
                args.model is None
                or args.ns_max is None
                or args.kind is None
                or args.zenodo is None
            ):
                print(
                    "--model, --ns-max, --kind, and --zenodo are required when "
                    "annotating an explicit file.",
                    file=sys.stderr,
                )
                return 2
        _annotate(args.path, args.model, args.ns_max, args.kind, args.zenodo, dry_run=args.dry_run)
        return 0

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "Data")
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 2

    n_done = 0
    for name, meta in KNOWN_FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"[skip] {name} not present in Data/")
            continue
        _annotate(
            path,
            meta["model"],
            meta["ns_max"],
            meta["kind"],
            meta["zenodo_id"],
            dry_run=args.dry_run,
        )
        n_done += 1
    print(f"Done: annotated {n_done} HDF5 file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
