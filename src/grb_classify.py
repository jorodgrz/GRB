"""
GRB and kilonova classification for compact binary mergers.

Implements the Gottlieb et al. (2024) hybrid four-class BNS scheme, the BHNS
disk-mass classifier (Foucart 2018 + Gottlieb 2023 disk thresholds, relabelled
to the 2024 hybrid taxonomy), a unified mass-plane grid classifier, and
Broekgaarden-style formation channel classification.
"""

import warnings

import numpy as np

from grb_physics import (
    HMNS_FACTOR_DEFAULT,
    M_THRESH,
    M_TOV,
    MDISK_LONG,
    MDISK_SHORT,
    Q_THRESH_BNS,
    foucart_disk_mass,
)


def _resolve_m_thresh(m_tov, m_thresh, k_thresh):
    """Reconcile the (m_tov, m_thresh, k_thresh) kwarg triple.

    If ``k_thresh`` is given, the prompt-collapse threshold is taken as
    ``k_thresh * m_tov`` so EOS sweeps that change ``m_tov`` move both
    thresholds together.  Passing an inconsistent ``m_thresh`` alongside
    ``k_thresh`` is ambiguous and now raises ``ValueError`` (was a
    warning until 2026-05-06; the silent override silently broke EOS
    sweep coherence when the caller forgot which value applied).
    """
    if k_thresh is None:
        return m_thresh
    derived = k_thresh * m_tov
    if m_thresh is not None and not np.isclose(m_thresh, derived):
        raise ValueError(
            f"Inconsistent prompt-collapse thresholds: k_thresh={k_thresh} "
            f"with m_tov={m_tov} implies m_thresh={derived:.3f}, but the "
            f"caller passed m_thresh={m_thresh}.  Pass only one of "
            f"(m_thresh, k_thresh) so EOS sweeps remain coherent."
        )
    return derived


# ═══════════════════════════════════════════════════════════════════════════
# BNS classification: Gottlieb et al. (2024) hybrid model
# ═══════════════════════════════════════════════════════════════════════════
def classify_bns_2024(
    m1,
    m2,
    m_tov=M_TOV,
    m_thresh=M_THRESH,
    q_thresh=Q_THRESH_BNS,
    hmns_factor=HMNS_FACTOR_DEFAULT,
    k_thresh=None,
):
    """Four-class BNS scheme (Gottlieb 2024).

    .. warning::
        ``hmns_factor`` (default 1.2) is a CODE HEURISTIC, not a value
        published by Gottlieb (2024).  It splits the long-lived HMNS
        (sbGRB) regime from the short-lived HMNS (lbGRB) regime at
        ``hmns_factor * m_tov`` and is motivated by Margalit and
        Metzger (2017, ApJL 850, L19, arXiv:1710.05938).  See
        ``grb_physics.HMNS_FACTOR_DEFAULT`` for the full discussion.

    Classes
    -------
    'sbGRB + blue KN' : Long-lived HMNS engine
        (M_tot < hmns_factor * M_TOV).
    'lbGRB + red KN (HMNS)' : Short-lived HMNS + BH engine
        (hmns_factor * M_TOV <= M_tot < M_thresh).
    'lbGRB + red KN (disk)' : Prompt collapse, massive disk
        (M_tot >= M_thresh and q >= q_thresh).
    'Faint lbGRB' : Prompt collapse, small disk
        (M_tot >= M_thresh and q < q_thresh).

    Parameters
    ----------
    m1, m2 : array-like
        Component masses (either ordering; heavier/lighter detected
        internally).
    m_tov : float, optional
        Maximum non-rotating NS mass [Msun]; default ``M_TOV`` (2.2).
    m_thresh : float, optional
        Prompt-collapse total-mass threshold [Msun]; default
        ``M_THRESH`` (~ 2.794).  Ignored if ``k_thresh`` is given.
    q_thresh : float, optional
        Mass ratio (>=1) above which the prompt-collapse remnant
        forms a massive disk; default ``Q_THRESH_BNS`` (1.2).
    hmns_factor : float, optional
        Multiplier on ``m_tov`` for the long-/short-lived HMNS split.
        Default 1.2 is a code heuristic (see ``HMNS_FACTOR_DEFAULT``);
        ``hmns_factor=1.0`` folds the short-lived HMNS class into
        prompt collapse entirely.
    k_thresh : float, optional
        If given, ``m_thresh`` is overridden to ``k_thresh * m_tov`` so
        that EOS sweeps move both thresholds together.

    There is no separate ``Q_NO_DISK = 1.05`` near-equal-mass "no-disk"
    boundary in Gottlieb (2023, 2024); GW170817-like systems (q ~ 1.0)
    were routed to "no GRB" in earlier revisions, which contradicts
    observations.  All prompt-collapse mergers below ``q_thresh`` now
    fall into 'Faint lbGRB'.
    """
    m1, m2 = np.asarray(m1), np.asarray(m2)
    m_heavy = np.maximum(m1, m2)
    m_light = np.minimum(m1, m2)
    M_tot = m_heavy + m_light
    q = m_heavy / m_light

    m_thresh = _resolve_m_thresh(m_tov, m_thresh, k_thresh)
    hmns_split = hmns_factor * m_tov

    return {
        "sbGRB + blue KN": (M_tot < hmns_split),
        "lbGRB + red KN (HMNS)": (M_tot >= hmns_split) & (M_tot < m_thresh),
        "lbGRB + red KN (disk)": (M_tot >= m_thresh) & (q >= q_thresh),
        "Faint lbGRB": (M_tot >= m_thresh) & (q < q_thresh),
    }


def bns_boundary_lines(
    m2,
    m_tov=M_TOV,
    m_thresh=M_THRESH,
    q_thresh=Q_THRESH_BNS,
    hmns_factor=HMNS_FACTOR_DEFAULT,
    k_thresh=None,
    m1_lim=None,
):
    """Compute the three Gottlieb (2024) BNS boundary curves on (M2, M1).

    Each line is parameterised by ``m2`` and clipped to the physically
    valid region ``m1 >= m2``.  Optional ``m1_lim = (m1_lo, m1_hi)``
    additionally clips to the figure's vertical range so callers do not
    need to repeat the masking algebra.

    Returns
    -------
    {'M_tot':   (m2_arr, m1_arr) for prompt-collapse boundary M_tot = M_thresh,
     'HMNS':    (m2_arr, m1_arr) for HMNS-lifetime split M_tot = hmns_factor * M_TOV,
     'q':       (m2_arr, m1_arr) for mass-ratio boundary m1 = q_thresh * m2}
    """
    m2 = np.asarray(m2, dtype=float)
    m_thresh = _resolve_m_thresh(m_tov, m_thresh, k_thresh)
    hmns_split = hmns_factor * m_tov

    def _clip(m1):
        ok = m1 >= m2
        if m1_lim is not None:
            lo, hi = m1_lim
            ok &= (m1 >= lo) & (m1 <= hi)
        return m2[ok], m1[ok]

    return {
        "M_tot": _clip(m_thresh - m2),
        "HMNS": _clip(hmns_split - m2),
        "q": _clip(q_thresh * m2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BHNS classification from disk mass
# ═══════════════════════════════════════════════════════════════════════════
def classify_bhns(M_BH, M_NS, a_BH=0.5, md_short=MDISK_SHORT, md_long=MDISK_LONG, **foucart_kw):
    """Classify BHNS mergers by Foucart disk mass under the Gottlieb 2024 hybrid.

    Disk-mass thresholds (0.01, 0.1 Msun) are Gottlieb (2023) Sec. 4 / Fig. 6;
    the class names follow the Gottlieb (2024) engine + ejecta hybrid so they
    line up with ``GRID_CLASS_LABELS`` entries 5 and 6 and with the BNS labels
    returned by ``classify_bns_2024``.

    Returns
    -------
    dict of boolean arrays:
        'No GRB'                       : M_disk < md_short
        'Faint lbGRB (BHNS)'           : md_short <= M_disk < md_long
        'lbGRB + red KN (BHNS disk)'   : M_disk >= md_long
    Also includes 'M_disk' key with the raw disk mass array.
    """
    M_BH, M_NS = np.asarray(M_BH), np.asarray(M_NS)
    M_disk = foucart_disk_mass(M_BH, M_NS, a_BH=a_BH, **foucart_kw)
    return {
        "No GRB": (M_disk < md_short),
        "Faint lbGRB (BHNS)": (M_disk >= md_short) & (M_disk < md_long),
        "lbGRB + red KN (BHNS disk)": (M_disk >= md_long),
        "M_disk": M_disk,
    }


def classify_bhns_spins(M_BH, M_NS, spins=(0.0, 0.3, 0.5, 0.7, 0.9), **kw):
    """Run classify_bhns for each spin value. Returns {spin: result_dict}."""
    return {a: classify_bhns(M_BH, M_NS, a_BH=a, **kw) for a in spins}


# ═══════════════════════════════════════════════════════════════════════════
# Unified mass-plane grid classifier (Gottlieb 2024 hybrid)
# ═══════════════════════════════════════════════════════════════════════════
GRID_CLASS_LABELS = {
    1: "Faint lbGRB (BNS)",
    2: "lbGRB + red KN (HMNS)",
    3: "sbGRB + blue KN",
    4: "lbGRB + red KN (BNS disk)",
    5: "Faint lbGRB (BHNS)",
    6: "lbGRB + red KN (BHNS disk)",
}
"""Integer class -> label map for ``classify_grid``.

Integer ``0`` is reserved for cells *outside* any classified region:
``m_light <= 0`` cells (grid edges where the lighter component has
no mass), BHNS cells with ``m_light < ns_min``, and the non-BNS /
non-BHNS region (m_heavy > ns_max with m_light also > ns_max, i.e.
BBH).  ``0`` is intentionally absent from the label map so plotting
code can mask it as background."""


NS_MAX_FIDUCIAL = (2.0, 2.5, 3.0)
"""Fiducial COMPAS ``M_NS_max`` values per Broekgaarden+ 2021
(arXiv:2103.02608): Model J = 2.0, Model A = 2.5, Model K = 3.0 Msun.
``classify_grid`` rejects any other value unless ``strict_ns_max=False``."""


def classify_grid(
    m1g,
    m2g,
    m_tov=M_TOV,
    m_thresh=M_THRESH,
    q_thresh=Q_THRESH_BNS,
    a_bh=0.5,
    hmns_factor=HMNS_FACTOR_DEFAULT,
    k_thresh=None,
    ns_max=None,
    ns_min=1.1,
    R_1p4_km=None,
    strict_ns_max=True,
):
    """Return an integer class map on a (M1, M2) grid.

    2024 Gottlieb hybrid model: all BH-powered jets produce lbGRBs;
    only long-lived HMNS (NS engine) produces sbGRBs.

    Classes (grouped by physical hierarchy; see ``GRID_CLASS_LABELS``):

        Background (default-fill):
        0 = unclassified          (outside BNS/BHNS regions, m_light < ns_min,
                                   or m_light <= 0)

        BNS region:
        1 = Faint lbGRB + red KN  (prompt collapse, small disk q < q_thresh)
        2 = lbGRB + red KN        (short-lived HMNS, BNS)
        3 = sbGRB + blue KN       (long-lived HMNS, BNS)
        4 = lbGRB + red KN        (BNS prompt collapse, massive disk q >= q_thresh)

        BHNS region:
        5 = Faint lbGRB + red KN  (BHNS, 0.01 <= Md < 0.1)
        6 = lbGRB + red KN        (BHNS, massive disk Md >= 0.1)

    Parameters
    ----------
    m_thresh : float, optional
        Prompt-collapse total-mass threshold [Msun]; default
        ``M_THRESH``.  Ignored if ``k_thresh`` is given.
    hmns_factor : float, optional
        Multiplier on ``m_tov`` for the long-/short-lived HMNS split.
        See ``classify_bns_2024`` for details.  Default 1.2.
    k_thresh : float, optional
        If given, ``m_thresh`` is overridden to ``k_thresh * m_tov`` so
        that EOS sweeps move both thresholds together.
    ns_max : float, optional
        Upper mass edge [Msun] of the BNS region on the grid: any
        component with mass > ns_max is treated as a BH.  This MUST
        match the ``M_NS_max`` configuration of the COMPAS run that
        produced your population (Broekgaarden+ 2021, arXiv:2103.02608:
        Models A/J/K use M_NS_max = 2.5 / 2.0 / 3.0 Msun).  If left as
        None, defaults to ``m_tov + 0.15`` and emits a UserWarning.
    ns_min : float, optional
        Lower mass [Msun] of the NS in BHNS systems.  Cells with
        ``m_light < ns_min`` are excluded from BHNS classification (and
        a warning is emitted reporting the count).  Default 1.1 Msun
        matches the lower truncation of the Alsing+ 2018 NS mass
        distribution (see ``NS_REMAP_M_MIN`` in ``grb_physics``); pass
        ``ns_min=0.8`` to be inclusive of ECSN/USSN NS, which reproduces
        the pre-patch behaviour.
    R_1p4_km : float, optional
        NS radius at 1.4 Msun [km], passed to ``foucart_disk_mass``
        for the BHNS branch.  If ``None`` and the grid contains BHNS
        cells, a UserWarning fires noting that the BHNS class 6
        (lbGRB + red KN, massive disk) inherits the ``foucart_disk_mass``
        default of 12.0 km and is therefore implicitly EOS-dependent.
        Pass an explicit value (e.g. ``EOS_MODELS['SFHo']['R_1p4']``)
        to remove the implicit dependence and to suppress the warning.
    strict_ns_max : bool, optional
        If True (default), ``ns_max`` must be in ``NS_MAX_FIDUCIAL``
        (2.0, 2.5, or 3.0 Msun) -- the only values used by published
        Broekgaarden+ 2021 COMPAS Models (J, A, K respectively).
        A typo such as ``ns_max=2.4`` would silently reshuffle the
        BNS / BHNS boundary and is now rejected with ``ValueError``.
        Pass ``strict_ns_max=False`` to opt out for sensitivity studies.
    """
    # Symmetrize the q convention so callers can pass either ordering
    # (or a rectangular meshgrid that includes both triangles).  This
    # matches classify_bns_2024's convention q = m_heavy / m_light and
    # avoids silently dropping cells where m2 > m1 into the default
    # "no GRB" class.  Cells with m2 > m1 are masked off downstream
    # in the visualization in the typical mass-plane use case.
    m1g = np.asarray(m1g, dtype=float)
    m2g = np.asarray(m2g, dtype=float)
    m_heavy = np.maximum(m1g, m2g)
    m_light = np.minimum(m1g, m2g)

    cls = np.full_like(m1g, 0, dtype=int)
    m_tot = m1g + m2g
    q = np.where(m_light > 0, m_heavy / m_light, 999.0)

    m_thresh = _resolve_m_thresh(m_tov, m_thresh, k_thresh)

    if ns_max is None:
        # Default is a pragmatic buffer ~0.15 Msun above M_TOV; with the
        # default m_tov=2.2 this gives ns_max ~ 2.35 Msun.  This is NOT
        # a physical NS upper mass: COMPAS runs typically impose a
        # tighter or different M_NS_max (Broekgaarden+ 2021 fiducial
        # 2.5 Msun, models J/K = 2.0/3.0 Msun).  Pass ns_max explicitly
        # to match the run configuration.
        ns_max = m_tov + 0.15
        warnings.warn(
            f"classify_grid: ns_max not specified; using default "
            f"m_tov + 0.15 = {ns_max:.2f} Msun.  This must match the "
            f"COMPAS run's M_NS_max (Broekgaarden+ 2021: 2.5 Msun "
            f"fiducial; models J/K = 2.0/3.0).  Pass ns_max explicitly "
            f"to suppress this warning.",
            stacklevel=2,
        )
    elif strict_ns_max and not any(np.isclose(ns_max, fid) for fid in NS_MAX_FIDUCIAL):
        # A typo such as ns_max=2.4 silently reassigns NS-NS systems
        # to BHNS or BBH; reject it.  Sensitivity studies that need
        # off-fiducial values must opt out via strict_ns_max=False.
        raise ValueError(
            f"classify_grid: ns_max={ns_max} not in fiducial "
            f"{NS_MAX_FIDUCIAL} Msun (Broekgaarden+ 2021 Models "
            f"J/A/K = 2.0/2.5/3.0).  Pass strict_ns_max=False to "
            f"override for sensitivity studies."
        )

    # BNS region: both components below the NS-mass cap, and the
    # lighter component must have positive mass.  Without the
    # ``m_light > 0`` guard, grid edges where m_light = 0 would fall
    # into the ``M_tot < hmns_split`` branch and be silently labeled
    # as 'sbGRB + blue KN' (class 3); the q guard at the top of this
    # function never fires because q is not consulted on that branch.
    # The BNS class is symmetric in (m_heavy, m_light) so we do not
    # require any ordering on the input; mirror cells across the
    # m1=m2 diagonal receive identical classifications.
    is_bns = (m_heavy <= ns_max) & (m_light <= ns_max) & (m_light > 0)
    hmns_split = hmns_factor * m_tov

    cls[is_bns & (m_tot < hmns_split)] = 3
    cls[is_bns & (m_tot >= hmns_split) & (m_tot < m_thresh)] = 2
    cls[is_bns & (m_tot >= m_thresh) & (q >= q_thresh)] = 4
    cls[is_bns & (m_tot >= m_thresh) & (q < q_thresh)] = 1

    # BHNS region: heavier component above the NS-mass cap (the BH),
    # lighter component at or above the NS-mass floor ``ns_min``.
    # Cells with 0 < m_light < ns_min are excluded (left at class 0)
    # and a warning is emitted; pass ns_min lower (e.g. 0.8 for ECSN/
    # USSN) to include them.
    bhns_candidate = (m_heavy > ns_max) & (m_light <= ns_max)
    excluded = bhns_candidate & (m_light > 0) & (m_light < ns_min)
    if np.any(excluded):
        n_excl = int(np.sum(excluded))
        warnings.warn(
            f"classify_grid: excluded {n_excl} cells with "
            f"m_light < {ns_min} Msun from BHNS region "
            f"(set ns_min lower to include them; 0.8 reproduces "
            f"the pre-patch default).",
            stacklevel=2,
        )

    is_bhns = bhns_candidate & (m_light >= ns_min)
    if np.any(is_bhns):
        if R_1p4_km is None:
            # The BHNS branch inherits foucart_disk_mass's default
            # R_1p4_km=12.0; this implicitly fixes the EOS softness for
            # class 6 (lbGRB + red KN, massive disk).  Surface this as
            # an EOS-implicit warning matching the ns_max style; pass
            # R_1p4_km explicitly (e.g. EOS_MODELS['SFHo']['R_1p4'])
            # to suppress.
            warnings.warn(
                "classify_grid: R_1p4_km not specified; BHNS branch "
                "inherits foucart_disk_mass default R_1p4_km=12.0 km, "
                "so class 6 (lbGRB + red KN, massive disk) is "
                "implicitly EOS-dependent.  Pass R_1p4_km explicitly "
                "(e.g. EOS_MODELS['SFHo']['R_1p4']) to suppress this "
                "warning.",
                stacklevel=2,
            )
            md = foucart_disk_mass(m_heavy[is_bhns], m_light[is_bhns], a_BH=a_bh)
        else:
            md = foucart_disk_mass(m_heavy[is_bhns], m_light[is_bhns], a_BH=a_bh, R_1p4_km=R_1p4_km)
        bhns_cls = np.zeros_like(md, dtype=int)
        bhns_cls[md >= MDISK_LONG] = 6
        bhns_cls[(md >= MDISK_SHORT) & (md < MDISK_LONG)] = 5
        cls[is_bhns] = bhns_cls

    return cls


# ═══════════════════════════════════════════════════════════════════════════
# Formation channel classification (Broekgaarden et al.)
# ═══════════════════════════════════════════════════════════════════════════
def classify_formation_channels(*, dblCE, fc_CEE, fc_mt_p1, fc_mt_s1, fc_mt_p1_K1, fc_mt_s1_K2):
    """Event-sequence formation channel classification.

    Parameters are 1-D arrays over the merging subset.

    Returns
    -------
    dict of boolean arrays:
        'I  Stable MT + CE', 'II  Stable MT only', 'III Single-core CE',
        'IV  Double-core CE', 'V   Other'
    """
    has_p1 = fc_mt_p1 > 0
    has_s1 = fc_mt_s1 > 0
    first_is_primary = has_p1 & (~has_s1 | (fc_mt_p1 < fc_mt_s1))
    first_is_secondary = has_s1 & (~has_p1 | (fc_mt_s1 < fc_mt_p1))
    no_stable_mt = ~has_p1 & ~has_s1

    # Case A donors (donor Hurley K = 0 or 1 at first RLOF) are folded into
    # I or II based on whether a CE follows, matching Broekgaarden+ 2022
    # Sec. 3.2 convention.  Donor stellar type at first RLOF is preserved on
    # fc_mt_p1_K1 / fc_mt_s1_K2 if a downstream analysis needs to slice on
    # it; the unused intermediate is dropped from this routine.
    had_CE = fc_CEE > 0

    first_rlof_time = np.where(
        first_is_primary, fc_mt_p1, np.where(first_is_secondary, fc_mt_s1, 999)
    )
    ce_first = had_CE & (no_stable_mt | (fc_CEE < first_rlof_time))

    ch_IV = dblCE == 1
    ch_other = ~ch_IV & no_stable_mt & ~had_CE
    ch_II = ~ch_IV & ~ch_other & ~had_CE
    ch_III = ~ch_IV & ~ch_other & ce_first
    ch_I = ~ch_IV & ~ch_other & had_CE & ~ce_first
    ch_other = ch_other | ~(ch_I | ch_II | ch_III | ch_IV)

    return {
        "I  Stable MT + CE": ch_I,
        "II  Stable MT only": ch_II,
        "III Single-core CE": ch_III,
        "IV  Double-core CE": ch_IV,
        "V   Other": ch_other,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Channel-by-class joint contingency
# ═══════════════════════════════════════════════════════════════════════════
def channel_class_crosstab(channel_masks, class_masks, weights, normalise=None):
    """STROOPWAFEL-weighted 5x4 (channel x GRB class) contingency table.

    Combines ``classify_formation_channels`` (Broekgaarden I to V) and
    ``classify_bns_2024`` (Gottlieb 4-class) into a single matrix
    showing which formation channels feed which GRB class.

    Parameters
    ----------
    channel_masks : dict[str, 1-D bool array]
        Output of ``classify_formation_channels`` (5 keys, length N).
    class_masks : dict[str, 1-D bool array]
        4-key Gottlieb output (e.g. from ``classify_bns_2024``).
        Non-mask keys like ``'M_disk'`` from ``classify_bhns`` are
        ignored automatically.
    weights : 1-D array, shape (N,)
        STROOPWAFEL weights aligned with the masks.
    normalise : {None, 'channel', 'class', 'total'}, optional
        Optional renormalisation of each cell:
        - None (default): raw weighted counts (sum_w over mask & class).
        - 'channel': each row sums to 1 (P(class | channel)).
        - 'class': each column sums to 1 (P(channel | class)).
        - 'total': all entries sum to 1 (joint distribution).

    Returns
    -------
    crosstab : pandas.DataFrame
        Rows are channel labels, columns are class labels.  Returned as
        a DataFrame (not a bare ndarray) so the row / column ordering
        is preserved for CSV export and heatmap labelling.
    """
    import pandas as pd

    weights = np.asarray(weights, dtype=float)
    class_keys = [
        k for k, v in class_masks.items() if isinstance(v, np.ndarray) and v.dtype == bool
    ]
    channel_keys = list(channel_masks.keys())

    matrix = np.zeros((len(channel_keys), len(class_keys)), dtype=float)
    for i, ch in enumerate(channel_keys):
        ch_mask = np.asarray(channel_masks[ch], dtype=bool)
        for j, cl in enumerate(class_keys):
            cl_mask = np.asarray(class_masks[cl], dtype=bool)
            matrix[i, j] = float(weights[ch_mask & cl_mask].sum())

    crosstab = pd.DataFrame(matrix, index=channel_keys, columns=class_keys)

    if normalise == "channel":
        row_sums = crosstab.sum(axis=1).replace(0.0, np.nan)
        crosstab = crosstab.div(row_sums, axis=0).fillna(0.0)
    elif normalise == "class":
        col_sums = crosstab.sum(axis=0).replace(0.0, np.nan)
        crosstab = crosstab.div(col_sums, axis=1).fillna(0.0)
    elif normalise == "total":
        total = crosstab.values.sum()
        if total > 0:
            crosstab = crosstab / total
    elif normalise is not None:
        raise ValueError(
            f"normalise must be None, 'channel', 'class', or 'total'; got {normalise!r}"
        )
    return crosstab


# ═══════════════════════════════════════════════════════════════════════════
# Observational classifier (Rastinejad et al. 2024 component decompositions)
# ═══════════════════════════════════════════════════════════════════════════
KN_RED_FRAC_BLUE_MAX = 0.30
"""Boundary between blue-dominated (sbGRB + blue KN) and mixed kilonova
ejecta in ``classify_observed_mergers``.  CODE HEURISTIC, not a value
published by Gottlieb (2024) or Rastinejad et al. (2024).  Below this
red-fraction the kilonova is interpreted as blue-dominated (long-lived
HMNS engine, no significant red-component lanthanide-rich ejecta)."""

KN_RED_FRAC_RED_MIN = 0.70
"""Boundary between mixed and red-dominated kilonova ejecta in
``classify_observed_mergers``.  CODE HEURISTIC; chosen symmetric
about ``KN_RED_FRAC_BLUE_MAX``.  Above this red-fraction the kilonova
is interpreted as red-dominated (post-merger disk wind from a prompt-
collapse remnant)."""

KN_M_EJ_FAINT_MAX = 0.01
"""Total ejecta-mass threshold below which the source is mapped to
'Faint lbGRB' [Msun].  Aligned with ``MDISK_SHORT`` (Gottlieb 2023,
Sec. 4): below 0.01 Msun of available mass there is no engine fuel for
a bright GRB or a detectable kilonova."""


def classify_observed_mergers(
    M_B,
    M_P,
    M_R,
    red_max_for_blue=KN_RED_FRAC_BLUE_MAX,
    red_min_for_red=KN_RED_FRAC_RED_MIN,
    m_ej_faint=KN_M_EJ_FAINT_MAX,
):
    """Map Rastinejad et al. (2024) ejecta decompositions to Gottlieb (2024) classes.

    .. warning::
        This is a phenomenological mapping that uses the kilonova
        ejecta colour decomposition (M_B blue, M_P purple, M_R red) as
        a proxy for the Gottlieb (2024) four-class engine taxonomy.
        The thresholds ``KN_RED_FRAC_BLUE_MAX = 0.30``,
        ``KN_RED_FRAC_RED_MIN = 0.70``, and ``KN_M_EJ_FAINT_MAX = 0.01``
        are CODE HEURISTICS, not values published by Gottlieb (2024)
        or Rastinejad et al. (2024).  Override per-call to test the
        sensitivity of any class-fraction comparison to the choice.

    Used by ``comparison.ipynb`` to overlay observed-sample class
    fractions on the model class-fraction predictions.

    Parameters
    ----------
    M_B, M_P, M_R : array-like
        Per-source posterior medians (or single-sample values) of the
        blue / purple / red kilonova ejecta components [Msun],
        following the Rastinejad et al. (2024, ApJ 979, 190) Table 3
        column convention.  Negative or NaN entries are passed through
        as NaN class labels.
    red_max_for_blue : float, optional
        Upper bound on the red fraction to count as 'sbGRB + blue KN'.
    red_min_for_red : float, optional
        Lower bound on the red fraction to count as 'lbGRB + red KN
        (disk)'.  Must satisfy ``red_min_for_red > red_max_for_blue``.
    m_ej_faint : float, optional
        Total ejecta below which the source is 'Faint lbGRB' [Msun].

    Returns
    -------
    dict of boolean arrays keyed by Gottlieb (2024) class label, plus
    'M_ej_total' and 'f_red' diagnostic arrays.
    """
    if not red_min_for_red > red_max_for_blue:
        raise ValueError(
            f"red_min_for_red ({red_min_for_red}) must exceed "
            f"red_max_for_blue ({red_max_for_blue})."
        )
    M_B = np.asarray(M_B, dtype=float)
    M_P = np.asarray(M_P, dtype=float)
    M_R = np.asarray(M_R, dtype=float)
    M_ej = M_B + M_P + M_R
    f_red = np.where(M_ej > 0, M_R / M_ej, np.nan)

    is_faint = M_ej < m_ej_faint
    is_blue_dom = (~is_faint) & (f_red < red_max_for_blue)
    is_red_dom = (~is_faint) & (f_red >= red_min_for_red)
    is_mixed = (~is_faint) & ~is_blue_dom & ~is_red_dom

    return {
        "sbGRB + blue KN": is_blue_dom,
        "lbGRB + red KN (HMNS)": is_mixed,
        "lbGRB + red KN (disk)": is_red_dom,
        "Faint lbGRB": is_faint,
        "M_ej_total": M_ej,
        "f_red": f_red,
    }
