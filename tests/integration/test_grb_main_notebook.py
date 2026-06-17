"""End-to-end smoke test for `grb_main.ipynb`.

The literature-anchor tests in `test_literature_anchors.py` lock the
module-level constants and pure-function physics; this file locks the
*production-pipeline* output by re-running the master notebook through
`nbclient` and checking class-fraction summary lines, the rate
calibration anchor, and `Plots/` artifact freshness.

Mechanism:

- `nbclient.NotebookClient` executes `grb_main.ipynb` with a 600 s
  per-cell timeout against the project's working directory.
- Cell stdout is parsed for the class-fraction summary (printed via
  the `_summary` helper at the top of the notebook) and for the
  Section 4 cosmic-integration anchor lines (`BNS R_loc expected = ...`
  and `BHNS R_loc expected = ...`).
- Generated PDF / PNG files in `Plots/` are checked for non-zero
  size and an mtime newer than a marker file written before the
  notebook executes.

Marked `slow` + `requires_data` + `requires_compas`; skips cleanly on
machines without the COMPAS HDF5 catalogues or the upstream
`compas_python_utils` package.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

# Section 7 of the notebook prints `R_loc expected` lines using
# ``rate_label`` formatting which truncates >= 1 to "{int:,}" and < 1 to
# "{:.2f}".  Use a tolerant numeric regex.
_NUM = r"[-+]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?"

_BNS_SUMMARY_RE = re.compile(
    r"^BNS:\s+([\d,]+)\s+merging systems",
    re.MULTILINE,
)
_BHNS_SUMMARY_RE = re.compile(
    r"^BHNS\s*\([^)]+\):\s+([\d,]+)\s+merging systems",
    re.MULTILINE,
)
# Per-class lines look like
#   "  sbGRB + blue KN (long-lived HMNS)    :   123,456  (54.3% weighted)"
_CLASS_LINE_RE = re.compile(
    r"^\s{2}([A-Za-z+ /()\-]+?)\s*:\s*([\d,]+)\s+\(\s*(\d+(?:\.\d+)?)% weighted\s*\)$",
    re.MULTILINE,
)

# Section 4 cosmic-integration anchor lines.  rate_label may format the
# expected rate as either "33" (>= 1) or "0.33" (< 1).
_R_LOC_BNS_RE = re.compile(
    r"^BNS\s+R_loc expected\s*=\s*([\d.,]+)\s+Gpc\^-3 yr\^-1",
    re.MULTILINE,
)
_R_LOC_BHNS_RE = re.compile(
    r"^BHNS R_loc expected\s*=\s*([\d.,]+)\s+Gpc\^-3 yr\^-1",
    re.MULTILINE,
)


# Files the notebook is expected to (re)generate.  Both .pdf (vector,
# for paper) and .png (raster, 300 dpi, for previews) variants per the
# project plotting standards.  `channels_x_classes.csv` is included
# because the Section 11 follow-up cell exports the cross-tab to CSV
# alongside the heatmap.
_REQUIRED_PLOT_FILES = (
    "mass_plane_bhns.pdf",
    "mass_plane_bhns.png",
    "mass_plane_bns.png",
    "fig1_bns_mass_plane_modelA.pdf",
    "mass_distributions_by_class.pdf",
    "mass_distributions_by_class.png",
    "delay_time_distributions.pdf",
    "delay_time_distributions.png",
    "metallicity_dependence.pdf",
    "metallicity_dependence.png",
    "formation_channels_by_grb_class.pdf",
    "formation_channels_by_grb_class.png",
    "rate_bns_by_class.pdf",
    "rate_bns_by_class.png",
    "rate_bhns_spin_sensitivity.pdf",
    "rate_bhns_spin_sensitivity.png",
    "projected_offsets.pdf",
    "projected_offsets.png",
    "rate_beaming_comparison.pdf",
    "rate_beaming_comparison.png",
    "eos_sensitivity.pdf",
    "eos_sensitivity.png",
    "channels_x_classes.pdf",
    "channels_x_classes.png",
    "channels_x_classes.csv",
    "offset_cdf_by_class.pdf",
    "offset_cdf_by_class.png",
    "beamed_class_comparison.pdf",
    "beamed_class_comparison.png",
)


def _collect_stdout(nb) -> str:
    """Concatenate every stream-stdout output from every cell.

    nbclient stores cell outputs under ``cell.outputs`` with each entry
    a dict-like with ``output_type``.  ``stream`` outputs (stdout /
    stderr) carry the ``text`` field.  We keep stderr too (the COMPAS
    Foucart calibration warnings land there) to make the parsed log
    self-contained.
    """
    chunks = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append(out.get("text", ""))
            elif out.get("output_type") in {"display_data", "execute_result"}:
                # Some print(df) calls land in execute_result with
                # text/plain mime; concatenate so the regex picks them up.
                data = out.get("data", {})
                if "text/plain" in data:
                    chunks.append(data["text/plain"])
    return "".join(chunks)


def _parse_class_table(stdout: str, header_re: re.Pattern) -> dict[str, float]:
    """Find the population summary block and return {label: weighted_pct}.

    The notebook prints two summary blocks (BNS, then BHNS) using the
    same ``_summary`` helper; the regex selects the block whose header
    line matches ``header_re`` and walks the immediately-following
    indented per-class lines.
    """
    m = header_re.search(stdout)
    if m is None:
        raise AssertionError(
            f"Population summary header not found in notebook stdout "
            f"(regex {header_re.pattern!r}); the notebook may have "
            f"failed before the _summary helper printed."
        )

    tail = stdout[m.end() :]
    out: dict[str, float] = {}
    for cm in _CLASS_LINE_RE.finditer(tail):
        label = cm.group(1).strip()
        pct = float(cm.group(3))
        out[label] = pct
        # Stop at the first blank line: the next summary block starts
        # after a newline gap.
        end = cm.end()
        if "\n\n" in tail[end : end + 4]:
            break
    return out


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Project root (parent of `tests/`)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def required_compas_files(repo_root: Path) -> tuple[Path, ...]:
    """The four COMPAS HDF5 files required by `grb_main.ipynb` (Section 1)."""
    return (
        repo_root / "Data" / "COMPASCompactOutput_BNS_A.h5",
        repo_root / "Data" / "COMPASCompactOutput_BHNS_A.h5",
    )


@pytest.fixture(scope="session")
def executed_notebook(repo_root, required_compas_files):
    """Execute `grb_main.ipynb` once per test session, return the notebook object.

    Skips cleanly if any required Tier-1 COMPAS HDF5 file or
    `compas_python_utils` is missing.  Caches the executed notebook
    so multiple tests in this file reuse a single ~3 to 5 minute run.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip(
        "compas_python_utils.cosmic_integration.FastCosmicIntegration",
        reason="compas_python_utils not installed in this environment",
    )

    for fpath in required_compas_files:
        if not fpath.exists():
            pytest.skip(f"Required COMPAS file missing: {fpath}")

    notebook_path = repo_root / "grb_main.ipynb"
    assert notebook_path.exists(), notebook_path

    nb = nbformat.read(notebook_path, as_version=4)

    plots_dir = repo_root / "Plots"
    plots_dir.mkdir(exist_ok=True)
    marker = plots_dir / ".smoke_marker"
    marker.write_text(str(time.time()))
    marker_mtime = marker.stat().st_mtime

    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root)}},
        allow_errors=False,
    )
    client.execute()

    nb._smoke_marker_mtime = marker_mtime  # type: ignore[attr-defined]
    return nb


# ─────────────────────────────────────────────────────────────────────
# Class-fraction summary parser checks
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.requires_compas
def test_grb_main_bns_class_fractions_in_published_band(executed_notebook):
    """BNS Model A class fractions must lie in Gottlieb (2024) Fig. 3 bands.

    Gottlieb (2024) Fig. 3 (the 2024 hybrid four-class scheme) gives
    BNS class fractions that depend strongly on the assumed Galactic
    NS-mass distribution.  After the Alsing-Silva-Berti (2018) remap
    pulls Model A NS masses toward [1.34, 1.80] Msun, the four
    classes are roughly comparable in weight, with sbGRB + blue KN
    (long-lived HMNS, M_tot < 1.2 * M_TOV) typically the smallest
    bin (10 to 35 percent) and the lbGRB classes plus Faint lbGRB
    splitting the rest.  The bands below are deliberately loose:
    they catch catastrophic drift (classifier inversion, NS-mass
    remap silently disabled, four-class partition broken) but do
    not regression-pin specific percentages.
    """
    nb = executed_notebook
    stdout = _collect_stdout(nb)
    bns = _parse_class_table(stdout, _BNS_SUMMARY_RE)
    assert bns, (
        f"Failed to parse BNS class fractions from notebook stdout. "
        f"First 2000 chars of stdout:\n{stdout[:2000]}"
    )

    sb = bns.get("sbGRB + blue KN (long-lived HMNS)", 0.0)
    hmns = bns.get("lbGRB + red KN  (short-lived HMNS)", 0.0)
    disk = bns.get("lbGRB + red KN  (massive disk)", 0.0)
    faint = bns.get("Faint lbGRB    (small disk / prompt)", 0.0)
    total = sb + hmns + disk + faint

    # Four-class partition: total fractions must sum to ~100% to
    # within rounding, since classify_bns_2024 returns disjoint masks.
    assert 95.0 <= total <= 102.0, (
        f"BNS class fractions sum to {total:.1f}%, expected ~100% "
        f"for the Gottlieb (2024) four-class partition.  Parsed: {bns}"
    )
    # Each class non-trivial (catches "all systems collapsed into one
    # class" failure modes, e.g. M_TOV silently set to inf).
    for label, value in (
        ("sbGRB", sb),
        ("lbGRB+HMNS", hmns),
        ("lbGRB+disk", disk),
        ("Faint", faint),
    ):
        assert 1.0 <= value <= 75.0, (
            f"BNS {label} fraction = {value:.1f}% outside non-trivial sanity band [1, 75]%."
        )
    # sbGRB + blue KN must NOT dominate the population for Model A
    # after the Alsing remap: hmns_split = 1.2 * 2.2 = 2.64 Msun is
    # below the typical M_tot ~ 2.7-3.0 Msun; if it suddenly does
    # dominate, the remap or hmns_split is broken.
    assert sb <= 50.0, (
        f"sbGRB + blue KN fraction = {sb:.1f}% > 50%; for Model A "
        f"with Alsing remap, the long-lived HMNS class should be "
        f"sub-dominant (hmns_split = 1.2 * M_TOV = 2.64 Msun is "
        f"below typical post-remap M_tot ~ 2.7 to 3.0 Msun)."
    )


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.requires_compas
def test_grb_main_bhns_class_fractions_in_foucart_band(executed_notebook):
    """BHNS Model A at a_BH = 0.5: No-GRB must dominate, GRB classes < 25%.

    Foucart (2018) Eq. (4) at a_BH = 0.5 plus the Broekgaarden+ 2021
    Model A BHNS sample produces a population in which the majority
    of mergers swallow the NS without disruption (no GRB).  Under the
    Gottlieb 2024 hybrid relabel, ``Faint lbGRB (BHNS) + lbGRB + red
    KN (BHNS disk)`` carries a weighted fraction bounded above ~25
    percent in this regime; ``No GRB`` is the dominant class.

    The classify_bhns three classes do partition every system that
    has a finite M_disk, but a fraction (~15 to 20 percent for Model
    A) carries non-finite M_disk in the COMPAS catalogue (extreme Q,
    out-of-validity Foucart input) and so does not contribute to the
    summary.  Total parsed fractions therefore land in [80, 90]%.
    """
    nb = executed_notebook
    stdout = _collect_stdout(nb)
    bhns = _parse_class_table(stdout, _BHNS_SUMMARY_RE)
    assert bhns, (
        f"Failed to parse BHNS class fractions from notebook stdout. "
        f"First 2000 chars of stdout:\n{stdout[:2000]}"
    )

    no_grb = bhns.get("No GRB / KN    (NS swallowed)", 0.0)
    short = bhns.get("Faint lbGRB    (small disk)", 0.0)
    long_grb = bhns.get("lbGRB + red KN (massive disk)", 0.0)
    total = no_grb + short + long_grb

    # Three-class partition with ~10-20% unclassified (extreme Q
    # systems where Foucart 2018 is out of validity); accept [70, 102]%.
    assert 70.0 <= total <= 102.0, (
        f"BHNS class fractions sum to {total:.1f}%, expected in "
        f"[70, 102]% for the Gottlieb-2024-hybrid BHNS partition with "
        f"Foucart validity-cap unclassified residual.  Parsed: {bhns}"
    )
    assert no_grb >= 50.0, (
        f"No GRB fraction = {no_grb:.1f}% < 50% lower band; Foucart "
        f"(2018) at a_BH = 0.5 expects most BHNS systems to swallow "
        f"the NS."
    )
    assert (short + long_grb) <= 25.0, (
        f"GRB-producing BHNS fraction = {short + long_grb:.1f}% > 25% "
        f"upper band at a_BH = 0.5; Foucart (2018) calibration "
        f"prefers the bulk to swallow the NS."
    )


# ─────────────────────────────────────────────────────────────────────
# Cosmic-integration calibration anchor
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.requires_compas
def test_grb_main_local_rate_anchor_matches_broekgaarden_table3(executed_notebook, repo_root: Path):
    """Section 4 `R_loc expected` lines must equal `read_expected_local_rate`.

    The notebook prints

        BNS  R_loc expected = ... Gpc^-3 yr^-1, ...
        BHNS R_loc expected = ... Gpc^-3 yr^-1, ...

    These are the Broekgaarden+ (2021) Table 3 anchors recovered from
    `weights_intrinsic/w_000` and used to calibrate
    `MEAN_MASS_EVOLVED`.  The stand-alone helper
    `grb_io.read_expected_local_rate` reads the same column from the
    HDF5; this test asserts the printed and module-computed values
    agree to within 5 percent (the production rate-print uses
    `rate_label` formatting which rounds to 2 sig figs).
    """
    import sys

    sys.path.insert(0, str(repo_root))
    from grb_io import read_expected_local_rate

    nb = executed_notebook
    stdout = _collect_stdout(nb)

    m_bns = _R_LOC_BNS_RE.search(stdout)
    m_bhns = _R_LOC_BHNS_RE.search(stdout)
    assert m_bns is not None, (
        f"BNS R_loc expected line not in notebook stdout. First 2000 chars:\n{stdout[:2000]}"
    )
    assert m_bhns is not None, (
        f"BHNS R_loc expected line not in notebook stdout. First 2000 chars:\n{stdout[:2000]}"
    )

    R_bns_printed = float(m_bns.group(1).replace(",", ""))
    R_bhns_printed = float(m_bhns.group(1).replace(",", ""))

    R_bns_expected = read_expected_local_rate(repo_root / "Data" / "COMPASCompactOutput_BNS_A.h5")
    R_bhns_expected = read_expected_local_rate(repo_root / "Data" / "COMPASCompactOutput_BHNS_A.h5")

    rel_bns = abs(R_bns_printed - R_bns_expected) / R_bns_expected
    rel_bhns = abs(R_bhns_printed - R_bhns_expected) / R_bhns_expected

    assert rel_bns < 0.05, (
        f"Notebook BNS R_loc = {R_bns_printed:.2f} drifted from "
        f"Broekgaarden Table 3 anchor {R_bns_expected:.2f} by "
        f"{rel_bns * 100:.1f}% (>5% relative)."
    )
    assert rel_bhns < 0.05, (
        f"Notebook BHNS R_loc = {R_bhns_printed:.2f} drifted from "
        f"Broekgaarden Table 3 anchor {R_bhns_expected:.2f} by "
        f"{rel_bhns * 100:.1f}% (>5% relative)."
    )

    # Local-rate physical bands per Broekgaarden+ 2021 Sec. 4.
    # BNS: Models A/J/K range 10 to 300 Gpc^-3 yr^-1 across MSSFR /
    # metallicity-prior variations.  BHNS: 1 to 200 Gpc^-3 yr^-1.
    assert 5.0 < R_bns_expected < 500.0, R_bns_expected
    assert 0.5 < R_bhns_expected < 500.0, R_bhns_expected


# ─────────────────────────────────────────────────────────────────────
# Plots/ artifact freshness
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.requires_compas
def test_grb_main_plots_regenerated_with_recent_mtime(executed_notebook, repo_root: Path):
    """All `_REQUIRED_PLOT_FILES` must exist, have non-zero size, and be fresh.

    Catches three failure modes:
    1. The notebook silently bypassed a savefig call (file missing).
    2. The notebook ran but produced an empty PDF (zero size; e.g. a
       `bbox_inches='tight'` failure on an empty axis).
    3. The notebook reused a stale file from a previous run (mtime
       older than the marker file written before execution).
    """
    nb = executed_notebook
    plots_dir = repo_root / "Plots"
    marker_mtime = nb._smoke_marker_mtime  # type: ignore[attr-defined]

    missing = []
    empty = []
    stale = []
    for name in _REQUIRED_PLOT_FILES:
        p = plots_dir / name
        if not p.exists():
            missing.append(name)
            continue
        if p.stat().st_size == 0:
            empty.append(name)
            continue
        if p.stat().st_mtime < marker_mtime:
            stale.append(name)

    assert not missing, f"Missing Plots/ files after notebook run: {missing}"
    assert not empty, f"Empty Plots/ files after notebook run: {empty}"
    assert not stale, (
        f"Stale Plots/ files (mtime older than pre-run marker): {stale}; "
        f"the notebook may not have regenerated them."
    )
