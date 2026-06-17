#!/usr/bin/env python3
"""Download COMPAS BNS and BHNS catalogues from Broekgaarden+ 2021 paper II.

Fetches the full 20-model grid for each population from Zenodo records
5189849 (BNS / NSNS in paper-II nomenclature) and 5178777 (BHNS), extracts
each zip into ``Data/``, renames the embedded HDF5 to the project's paper-I
letter convention where the model exists in paper I, and runs the
embed-metadata annotator so loaders can validate model identity at load
time.

Why a custom downloader instead of a generic Zenodo grabber
-----------------------------------------------------------
Both Zenodo records package data using **paper-II letter conventions**
(arXiv:2112.05763), while the project's filenames and
``tools/embed_model_metadata.py`` use **paper-I letters**
(arXiv:2103.02608).  The two bundled zips, ``fiducial.zip`` and
``unstableCaseBB.zip``, each contain two HDF5 files for two different
physics variations, and the same paper-II letter (e.g. K) refers to a
different variation than its paper-I namesake.  A naive "download and
unzip" workflow produces files that disagree with the project's filename
convention and silently mislabels which physics variation each h5
represents.  This script preserves the project-side paper-I letters and
adds descriptive suffixes (``alpha0p1``, ``alpha10``, ``EH``, ``fWR0p1``,
``fWR5``) for the five paper-II-only variations.

Usage
-----
::

    python tools/download_compas_data.py --dry-run            # show plan
    python tools/download_compas_data.py --tier 1 --confirm   # core 5 models
    python tools/download_compas_data.py --models J F G K     # explicit set
    python tools/download_compas_data.py --kind BNS --confirm # BNS only
    python tools/download_compas_data.py --confirm            # everything

After each h5 is extracted, model identity is written as HDF5 root
attributes (``model``, ``ns_max``, ``kind``, ``zenodo_id``,
``mssfr_mu0``) via :func:`tools.embed_model_metadata._annotate`.
Paper-II-only variations have no paper-I letter; their ``model`` attribute
is set to the descriptor (e.g. ``alpha10``) instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable

# Allow ``import embed_model_metadata`` without making ``tools`` a package.
# The annotator is intentionally a sibling module rather than a third-party
# install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed_model_metadata as _emm  # noqa: E402

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Data")
CACHE_DIR = os.path.join(DATA_DIR, "_cache")
MANIFEST_DIR = os.path.join(CACHE_DIR, "manifests")

# Zenodo concept identifiers (paper II of Broekgaarden+ 2021,
# arXiv:2112.05763).
ZENODO_BNS = 5189849
ZENODO_BHNS = 5178777

ZENODO_API = "https://zenodo.org/api/records/{record_id}"

# Confirmation gate: any plan whose total compressed download exceeds this
# requires ``--confirm`` to proceed.  47 GB is the full grid, so this
# threshold keeps "I just want to test it" runs from accidentally pulling
# the entire archive.
CONFIRM_THRESHOLD_BYTES = 5 * 1024**3  # 5 GB


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ModelEntry:
    """One physics variation, for one population, in the Zenodo grid.

    ``project_suffix``    Filename suffix used on disk, paper-I letter
                          where the variation exists in paper I, otherwise
                          a descriptive token (``alpha10`` etc.).
    ``kind``              ``BNS`` or ``BHNS``.
    ``zenodo_id``         Numeric Zenodo record ID, no DOI prefix.
    ``zip_name``          File entry within the Zenodo record.
    ``zip_dir``           Top-level directory created when the zip is
                          extracted; matches the zip stem.
    ``paper2_letter``     Letter that paper II uses for this variation.
                          Determines the internal h5 filename:
                          ``{zip_dir}/COMPASCompactOutput_{kind}_{letter}.h5``
    ``paper1_letter``     Paper I letter, or ``None`` for paper-II-only
                          variations.  Written as the ``model`` attribute
                          when present; otherwise the descriptor is used.
    ``ns_max``            Maximum NS gravitational mass (Msun) for this
                          variation; matters because three rows of the
                          grid vary it.
    ``description``       Short human-readable summary, prose form.
    ``tier``              1 (core), 2 (kick variants), 3 (lower priority),
                          4 (completeness).
    """

    project_suffix: str
    kind: str
    zenodo_id: int
    zip_name: str
    zip_dir: str
    paper2_letter: str
    paper1_letter: str | None
    ns_max: float
    description: str
    tier: int

    @property
    def internal_h5_path(self) -> str:
        """Path inside the zip to the desired h5 file."""
        return f"{self.zip_dir}/COMPASCompactOutput_{self.kind}_{self.paper2_letter}.h5"

    @property
    def output_filename(self) -> str:
        """On-disk filename under ``Data/`` after rename."""
        return f"COMPASCompactOutput_{self.kind}_{self.project_suffix}.h5"

    @property
    def output_path(self) -> str:
        return os.path.join(DATA_DIR, self.output_filename)

    @property
    def model_attr(self) -> str:
        """Value to write as the HDF5 ``model`` root attribute.

        Paper-I letter where it exists, descriptor otherwise.  This is
        what ``grb_io._validate_hdf5_metadata`` reads back.
        """
        return self.paper1_letter or self.project_suffix


# Single source of truth for the variation grid.  Each row describes one
# physics knob (the paper-II zip), with paper-I and project-side labels.
# (paper2_letter, project_suffix, paper1_letter, zip_name, zip_dir,
#  ns_max, description, tier)
_GRID = [
    ("A", "A", "A", "fiducial.zip", "fiducial", 2.5, "Fiducial", 1),
    (
        "B",
        "B",
        "B",
        "massTransferEfficiencyFixed_0_25.zip",
        "massTransferEfficiencyFixed_0_25",
        2.5,
        "Mass-transfer efficiency beta=0.25",
        4,
    ),
    (
        "C",
        "C",
        "C",
        "massTransferEfficiencyFixed_0_5.zip",
        "massTransferEfficiencyFixed_0_5",
        2.5,
        "Mass-transfer efficiency beta=0.5",
        4,
    ),
    (
        "D",
        "D",
        "D",
        "massTransferEfficiencyFixed_0_75.zip",
        "massTransferEfficiencyFixed_0_75",
        2.5,
        "Mass-transfer efficiency beta=0.75",
        4,
    ),
    (
        "E",
        "E",
        "E",
        "unstableCaseBB.zip",
        "unstableCaseBB",
        2.5,
        "Unstable case BB mass transfer",
        4,
    ),
    (
        "F",
        "EH",
        None,
        "unstableCaseBB.zip",
        "unstableCaseBB",
        2.5,
        "Unstable case BB plus optimistic CE (paper-II only)",
        4,
    ),
    ("G", "alpha0p1", None, "alpha0_1.zip", "alpha0_1", 2.5, "alpha_CE = 0.1 (paper-II only)", 4),
    ("H", "F", "F", "alpha0_5.zip", "alpha0_5", 2.5, "alpha_CE = 0.5", 1),
    ("I", "G", "G", "alpha2_0.zip", "alpha2_0", 2.5, "alpha_CE = 2.0", 1),
    ("J", "alpha10", None, "alpha10.zip", "alpha10", 2.5, "alpha_CE = 10 (paper-II only)", 4),
    ("K", "H", "H", "fiducial.zip", "fiducial", 2.5, "Optimistic CE (HG donors survive CE)", 3),
    ("L", "I", "I", "rapid.zip", "rapid", 2.5, "Rapid SN remnant prescription", 3),
    ("M", "J", "J", "maxNSmass2_0.zip", "maxNSmass2_0", 2.0, "Maximum NS mass = 2.0 Msun", 1),
    ("N", "K", "K", "maxNSmass3_0.zip", "maxNSmass3_0", 3.0, "Maximum NS mass = 3.0 Msun", 1),
    ("O", "L", "L", "noPISN.zip", "noPISN", 2.5, "No (pulsational) PISN", 4),
    (
        "P",
        "M",
        "M",
        "ccSNkick_100km_s.zip",
        "ccSNkick_100km_s",
        2.5,
        "Core-collapse SN kick sigma = 100 km/s",
        3,
    ),
    (
        "Q",
        "N",
        "N",
        "ccSNkick_30km_s.zip",
        "ccSNkick_30km_s",
        2.5,
        "Core-collapse SN kick sigma = 30 km/s",
        2,
    ),
    ("R", "O", "O", "noBHkick.zip", "noBHkick", 2.5, "No BH natal kick (Blaauw kick only)", 2),
    (
        "S",
        "fWR0p1",
        None,
        "wolf_rayet_multiplier_0_1.zip",
        "wolf_rayet_multiplier_0_1",
        2.5,
        "Wolf-Rayet wind multiplier f_WR = 0.1 (paper-II only)",
        4,
    ),
    (
        "T",
        "fWR5",
        None,
        "wolf_rayet_multiplier_5.zip",
        "wolf_rayet_multiplier_5",
        2.5,
        "Wolf-Rayet wind multiplier f_WR = 5 (paper-II only)",
        4,
    ),
]


def _build_registry() -> dict[tuple[str, str], ModelEntry]:
    """Cross-product the grid with both populations."""
    out: dict[tuple[str, str], ModelEntry] = {}
    for kind, zid in (("BNS", ZENODO_BNS), ("BHNS", ZENODO_BHNS)):
        for paper2, suffix, paper1, zip_name, zip_dir, ns_max, desc, tier in _GRID:
            entry = ModelEntry(
                project_suffix=suffix,
                kind=kind,
                zenodo_id=zid,
                zip_name=zip_name,
                zip_dir=zip_dir,
                paper2_letter=paper2,
                paper1_letter=paper1,
                ns_max=ns_max,
                description=desc,
                tier=tier,
            )
            out[(kind, suffix)] = entry
    return out


MODEL_REGISTRY: dict[tuple[str, str], ModelEntry] = _build_registry()


# ---------------------------------------------------------------------------
# Zenodo manifest fetching
# ---------------------------------------------------------------------------
def fetch_zenodo_manifest(zenodo_id: int, *, refresh: bool = False) -> dict[str, dict]:
    """Return ``{zip_name: {"url", "md5", "size"}}`` for ``zenodo_id``.

    Caches the parsed manifest under ``Data/_cache/manifests/<id>.json``
    so a typical run hits Zenodo at most twice (once per population).
    """
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    cache_path = os.path.join(MANIFEST_DIR, f"{zenodo_id}.json")
    if not refresh and os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)

    url = ZENODO_API.format(record_id=zenodo_id)
    with urllib.request.urlopen(url, timeout=60) as resp:
        record = json.loads(resp.read().decode("utf-8"))

    out: dict[str, dict] = {}
    for entry in record.get("files", []):
        checksum = entry.get("checksum", "")
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else ""
        out[entry["key"]] = {
            "url": entry["links"]["self"],
            "md5": md5,
            "size": int(entry["size"]),
        }
    with open(cache_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def human_size(n_bytes: int) -> str:
    """Return ``n_bytes`` as a 1024-base human string (e.g. ``1.23 GB``)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:6.2f} {unit}"
        size /= 1024
    return f"{n_bytes} B"


def md5_of_file(path: str, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Download with resume
# ---------------------------------------------------------------------------
def download_with_resume(url: str, dest: str, *, expected_md5: str, expected_size: int) -> None:
    """Stream ``url`` to ``dest`` with HTTP Range resume and MD5 check.

    Writes to ``<dest>.partial`` first.  On size or checksum mismatch
    raises ``RuntimeError`` and leaves the partial file in place so the
    next run can resume.

    Server-side resume requires the Zenodo CDN to honour ``Range``;
    Zenodo currently does, but if it does not, we fall back to a fresh
    download starting at byte zero.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    partial = dest + ".partial"
    start = os.path.getsize(partial) if os.path.exists(partial) else 0

    if start >= expected_size:
        # Existing partial is at least as big as the target file: probably
        # complete from a previous run that did not get to finalize.
        # Verify rather than redownload.
        start = 0

    headers = {}
    mode = "wb"
    if start > 0:
        headers["Range"] = f"bytes={start}-"
        mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    last_log = time.monotonic()
    log_every_s = 5.0
    bytes_done = start

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = getattr(resp, "status", 200)
            if start > 0 and status != 206:
                # Server ignored Range; restart cleanly so we do not append
                # the full body onto our partial bytes.
                print("  (server did not honour Range; restarting download)")
                start = 0
                bytes_done = 0
                mode = "wb"
                # Re-issue without the Range header.
                resp.close()
                with urllib.request.urlopen(url, timeout=120) as resp2:
                    _stream_body(
                        resp2, partial, mode, expected_size, bytes_done, log_every_s, last_log
                    )
            else:
                _stream_body(resp, partial, mode, expected_size, bytes_done, log_every_s, last_log)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e

    final_size = os.path.getsize(partial)
    if final_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {os.path.basename(dest)}: "
            f"got {final_size} bytes, expected {expected_size}"
        )

    actual_md5 = md5_of_file(partial)
    if expected_md5 and actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {os.path.basename(dest)}: got {actual_md5}, expected {expected_md5}"
        )

    os.replace(partial, dest)


def _stream_body(
    resp,
    dest_path: str,
    mode: str,
    total: int,
    bytes_done: int,
    log_every_s: float,
    last_log: float,
) -> None:
    chunk = 1024 * 1024
    name = os.path.basename(dest_path).removesuffix(".partial")
    with open(dest_path, mode) as fh:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            bytes_done += len(block)
            now = time.monotonic()
            if now - last_log >= log_every_s:
                pct = 100.0 * bytes_done / max(total, 1)
                sys.stderr.write(
                    f"\r  {name}: {human_size(bytes_done)} / {human_size(total)} ({pct:5.1f}%)"
                )
                sys.stderr.flush()
                last_log = now
    sys.stderr.write("\n")


# ---------------------------------------------------------------------------
# Zip extraction
# ---------------------------------------------------------------------------
def extract_zip(zip_path: str, internal_path: str, target_path: str) -> None:
    """Extract one entry from a Zenodo zip to ``target_path``.

    The zips ship with directory prefixes (``fiducial/``, ``alpha0_1/``)
    and use paper-II letters in the inner h5 filename.  ``internal_path``
    is the exact path inside the zip; we do not glob, so a missing entry
    is loud rather than silent.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if internal_path not in names:
            # Zenodo occasionally re-issues a zip without a leading
            # directory; try the basename as a fallback before erroring.
            base = os.path.basename(internal_path)
            candidates = [n for n in names if n.endswith("/" + base) or n == base]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Could not locate {internal_path!r} in "
                    f"{os.path.basename(zip_path)}; "
                    f"contents: {names}"
                )
            internal_path = candidates[0]

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with zf.open(internal_path) as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
def annotate(entry: ModelEntry, *, dry_run: bool = False) -> None:
    """Write model identity attributes to ``entry.output_path``.

    Reuses :func:`tools.embed_model_metadata._annotate` so the on-disk
    schema stays in lockstep with the existing one-shot annotator.
    Paper-II-only entries write the descriptor (e.g. ``alpha10``) as the
    ``model`` attribute, so ``grb_io.load_*`` validators still see a
    self-describing file.
    """
    zenodo_doi = f"10.5281/zenodo.{entry.zenodo_id}"
    _emm._annotate(  # noqa: SLF001 -- intentional reuse, single source of truth
        path=entry.output_path,
        model=entry.model_attr,
        ns_max=entry.ns_max,
        kind=entry.kind,
        zenodo_id=zenodo_doi,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Selection / filtering
# ---------------------------------------------------------------------------
def select_entries(
    *, kind: str | None, tier: int | None, suffixes: Iterable[str] | None
) -> list[ModelEntry]:
    """Apply CLI filters to ``MODEL_REGISTRY`` and return a stable list."""
    entries = list(MODEL_REGISTRY.values())
    if kind:
        entries = [e for e in entries if e.kind == kind]
    if tier is not None:
        entries = [e for e in entries if e.tier <= tier]
    if suffixes:
        wanted = set(suffixes)
        entries = [e for e in entries if e.project_suffix in wanted]
    entries.sort(key=lambda e: (e.kind, e.tier, e.project_suffix))
    return entries


# ---------------------------------------------------------------------------
# Plan printing
# ---------------------------------------------------------------------------
def print_plan(entries: list[ModelEntry], manifests: dict[int, dict[str, dict]]) -> int:
    """Print the per-entry plan and return the unique-zip total size."""
    # Group by zip so bundled zips are downloaded once.
    groups: dict[tuple[int, str], list[ModelEntry]] = {}
    for e in entries:
        groups.setdefault((e.zenodo_id, e.zip_name), []).append(e)

    total = 0
    print(
        f"{'kind':4s}  {'suffix':9s}  {'tier':4s}  {'ns_max':6s}  "
        f"{'zip':38s}  {'size':>9s}  description"
    )
    print("-" * 110)
    for entry in entries:
        size = manifests[entry.zenodo_id][entry.zip_name]["size"]
        # Only count one copy per (zenodo_id, zip_name): bundled zips
        # supply two h5 files but only one zip download.
        is_first_in_group = entry is groups[(entry.zenodo_id, entry.zip_name)][0]
        size_for_total = size if is_first_in_group else 0
        total += size_for_total
        size_label = human_size(size) if is_first_in_group else "(bundled)"
        print(
            f"{entry.kind:4s}  {entry.project_suffix:9s}  "
            f"{entry.tier:4d}  {entry.ns_max:6.1f}  "
            f"{entry.zip_name:38s}  {size_label:>9s}  {entry.description}"
        )
    print("-" * 110)
    print(f"Unique zips to download: {len(groups)} totalling {human_size(total)}")
    return total


# ---------------------------------------------------------------------------
# Per-entry processing pipeline
# ---------------------------------------------------------------------------
def process_group(
    zenodo_id: int,
    zip_name: str,
    entries: list[ModelEntry],
    manifests: dict[int, dict[str, dict]],
    *,
    force: bool,
    keep_zips: bool,
    verify: bool,
) -> None:
    """Download (if needed), then extract and annotate every entry that
    sources from a single zip."""
    # If every output already exists we still re-run annotate so a
    # previously aborted run (e.g. the conda env was not active and h5py
    # was missing during the first attempt) recovers cleanly without
    # forcing a redownload.  ``_annotate`` is idempotent.
    all_present = all(os.path.exists(e.output_path) for e in entries)
    if all_present and not force and not verify:
        print(f"[skip] {zenodo_id} {zip_name}: targets present, re-running annotation")
        for e in entries:
            print(f"       -> {e.output_filename}")
            annotate(e)
        return

    meta = manifests[zenodo_id][zip_name]
    zip_path = os.path.join(CACHE_DIR, f"zenodo_{zenodo_id}", zip_name)

    need_download = force or not (os.path.exists(zip_path) and md5_of_file(zip_path) == meta["md5"])
    if all_present and not force and verify:
        print(f"[verify] {zenodo_id} {zip_name}: targets exist; re-checking source zip")
        need_download = not (os.path.exists(zip_path) and md5_of_file(zip_path) == meta["md5"])
        if not need_download:
            print("  cached zip MD5 matches Zenodo manifest")
            return

    if need_download:
        print(f"[fetch] {zip_name} ({human_size(meta['size'])})")
        download_with_resume(
            meta["url"], zip_path, expected_md5=meta["md5"], expected_size=meta["size"]
        )
        print(f"  cached at {os.path.relpath(zip_path, REPO_ROOT)}")
    else:
        print(f"[cached] {zip_name}")

    for e in entries:
        if os.path.exists(e.output_path) and not force:
            print(f"  [keep] {e.output_filename}")
        else:
            print(f"  [extract] {e.internal_h5_path} -> {e.output_filename}")
            extract_zip(zip_path, e.internal_h5_path, e.output_path)
        annotate(e)

    if not keep_zips:
        os.remove(zip_path)
        print(f"  [clean] removed cached {zip_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--kind",
        choices=("BNS", "BHNS"),
        default=None,
        help="Limit to one population (default: both).",
    )
    p.add_argument(
        "--tier",
        type=int,
        choices=(1, 2, 3, 4),
        default=None,
        help="Inclusive tier filter; --tier 2 includes tiers 1 and 2.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="SUFFIX",
        help="Explicit list of project suffixes (e.g. J F G K).",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print the plan and budget; do not download."
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required acknowledgement when total > "
        f"{human_size(CONFIRM_THRESHOLD_BYTES).strip()}.",
    )
    p.add_argument(
        "--force", action="store_true", help="Re-download and re-extract even if outputs exist."
    )
    p.add_argument("--verify", action="store_true", help="Re-check cached zips against Zenodo MD5.")
    p.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep cached zips under Data/_cache/ after extraction.",
    )
    p.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Bypass the cached Zenodo manifest and re-fetch.",
    )
    return p.parse_args(argv)


def _check_h5py_available() -> bool:
    """Return True iff h5py imports.  Used as an early gate in main()."""
    try:
        import h5py  # noqa: F401

        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    entries = select_entries(kind=args.kind, tier=args.tier, suffixes=args.models)
    if not entries:
        print("No models match the requested filter.", file=sys.stderr)
        return 2

    # Fail fast if the env is missing h5py.  ``--dry-run`` does not need
    # it because annotation is skipped, but every other code path will
    # eventually call ``_annotate`` which lazy-imports h5py.  Catching it
    # here avoids the failure mode where one zip downloads, partially
    # extracts, then aborts with a confusing ImportError.
    if not args.dry_run and not _check_h5py_available():
        print(
            "ERROR: h5py is required for download and annotation but is "
            "not importable in the active Python environment.\n"
            "Activate the conda env first:\n"
            "    conda activate grb-env\n"
            "then re-run this command.  ``--dry-run`` works without "
            "h5py if you only want to preview the plan.",
            file=sys.stderr,
        )
        return 1

    needed_records = sorted({e.zenodo_id for e in entries})
    manifests = {
        zid: fetch_zenodo_manifest(zid, refresh=args.refresh_manifest) for zid in needed_records
    }

    total = print_plan(entries, manifests)

    if args.dry_run:
        return 0

    if total > CONFIRM_THRESHOLD_BYTES and not args.confirm:
        print(
            f"\nPlan exceeds {human_size(CONFIRM_THRESHOLD_BYTES).strip()}; "
            f"re-run with --confirm to proceed.",
            file=sys.stderr,
        )
        return 1

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    groups: dict[tuple[int, str], list[ModelEntry]] = {}
    for e in entries:
        groups.setdefault((e.zenodo_id, e.zip_name), []).append(e)

    for (zid, zip_name), group in sorted(groups.items()):
        try:
            process_group(
                zid,
                zip_name,
                group,
                manifests,
                force=args.force,
                keep_zips=args.keep_zips,
                verify=args.verify,
            )
        except KeyboardInterrupt:
            print(
                "\nInterrupted; partial download preserved at "
                f"{CACHE_DIR}/zenodo_{zid}/{zip_name}.partial",
                file=sys.stderr,
            )
            return 130
        except Exception as e:
            print(f"\n[error] {zip_name}: {e}", file=sys.stderr)
            return 1

    print("\nDone.  All requested files are present under Data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
