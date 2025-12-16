
from __future__ import annotations
# -*- coding: utf-8 -*-
"""
ccp_core.py — Core Logic for CCP-Tag Hidden Quinone Discovery
-----------------------------------------------------------
Purpose
-------
Provides shared scientific logic used by both publication modules:
- Diagnostic fragment matching (shifted + free diagnostics)
- Anchor fragment checks
- Neutral-loss evaluation
- Isotope pattern heuristics
- Confidence scoring and evidence aggregation

This file contains the validated decision rules used throughout the workflow.
All thresholds and algorithms reflect the finalized
method described in the manuscript.

Usage
-----
Imported by:
    - module1_standards.py
    - module2_environmental.py

New users should not modify functions in this file unless intentionally altering
the scientific method.

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------------- Params shell ----------------
@dataclass
class CCPParams:
    # Tolerances
    USE_PPM: bool = True
    DIAG_PPM: float = 25.0 
    PPM_FOR_MERGE: float = 10.0 
    FRAG_MZ_MARGIN_DA: float = 0.5

    # Chemistry
    PROTON: float = 1.007276466812
    TAG_NEUTRAL_MASS: float = 1269.493187
    MIN_Q_MASS: float = 108.0

    # Diagnostics (neutral)
    DIAG_AA: Tuple[float, ...] = (
        105.042594, 133.037509, 146.069143, 165.078979, 174.111676,
        115.063329, 121.019751, 119.058244,
    )
    DIAG_SEQ: Tuple[float, ...] = (
        88.039305, 175.071333, 290.098276, 418.156854, 565.225268,
        721.326379, 818.379143, 933.406086, 1048.433029, 1151.442213,
        1269.493187, 1183.468428, 1096.4364, 981.409457, 853.350879,
        706.282465, 550.181354, 453.12859, 338.101647, 223.074704,
    )
    ANCHOR_DIAGS: Tuple[float, ...] = (706.282465, 818.379143, 933.406086, 1048.433029)

    # Evidence thresholds
    REL_INT_MIN: float = 0.01 
    TAG_REL_INT_MIN: float = 0.005
    DIAG_CHARGES: Tuple[int, ...] = (1, 2, 3, 4)
    MIN_DIAG_FOR_COS: int = 4
    HIGH_MASS_CUTOFF: float = 700.0

    # Gate
    FREE_MIN_DIAGS: int = 4
    FREE_INT_FLOOR: float = 0.01 

    # Isotopes
    ISO_RT_WIN_MIN: float = 0.5
    ISO_MZ_WIN_DA: float = 4.0
    ISO_PPM_TOL: float = 20.0
    ISO_INT_RELMIN: float = 0.0005

# ---------------- Basic math ----------------

def ppm_da(mz: float, ppm: float) -> float:
    return mz * ppm / 1e6

def within_tol(obs: float, theo: float, ppm: float, use_ppm: bool) -> bool:
    return abs(obs - theo) <= (ppm_da(theo, ppm) if use_ppm else 0.5)

# ---------------- MGF merge ----------------

def merge_peaks(peaklists: List[List[Tuple[float, float]]], p: CCPParams) -> List[Tuple[float, float]]:
    if not peaklists:
        return []
    arr = np.array([(mz, i) for pl in peaklists for mz, i in pl], float)
    if arr.size == 0:
        return []
    idx = np.argsort(arr[:, 0])
    mzs = arr[idx, 0]; ints = arr[idx, 1]
    out = []
    cm = mzs[0]; ci = ints[0]
    for mz, it in zip(mzs[1:], ints[1:]):
        tol = ppm_da(cm, p.PPM_FOR_MERGE) if p.USE_PPM else 0.1
        if abs(mz - cm) <= tol:
            if it > ci: cm, ci = mz, it
        else:
            out.append((cm, ci)); cm, ci = mz, it
    out.append((cm, ci))
    return out

# ---------------- Diagnostics (MS2) ----------------

def basepeak_intensity(merged: List[Tuple[float, float]]) -> float:
    return max((i for _, i in merged), default=0.0)


def _best_diag_intensity(merged: List[Tuple[float, float]], neu_mass: float, shift_neutral: float, precursor_mz: Optional[float], p: CCPParams) -> Tuple[float, Optional[int]]:
    best = 0.0; best_z = None
    for z in p.DIAG_CHARGES:
        theo = (neu_mass + shift_neutral + z*p.PROTON) / z
        if precursor_mz is not None and theo > (precursor_mz + p.FRAG_MZ_MARGIN_DA):
            continue
        for m, i in merged:
            if within_tol(m, theo, p.DIAG_PPM, p.USE_PPM) and i > best:
                best = i; best_z = z
    return best, best_z


def diag_hits(merged: List[Tuple[float, float]], shift_neutral: float, precursor_mz: Optional[float], p: CCPParams) -> Tuple[List[Tuple[float, int, float]], int, int, float]:
    bp = basepeak_intensity(merged) or 1.0
    thr = bp * p.REL_INT_MIN
    hits = []; seen = set()
    for neu in (p.DIAG_AA + p.DIAG_SEQ):
        best, z = _best_diag_intensity(merged, neu, shift_neutral, precursor_mz, p)
        if best >= thr:
            hits.append((neu, z or 1, best))
            seen.add(neu)
    cov_total = len(seen)
    cov_high  = sum(1 for neu in seen if neu >= p.HIGH_MASS_CUTOFF)
    return hits, cov_total, cov_high, thr


def make_sample_vector(merged: List[Tuple[float, float]], shift: float, precursor_mz: Optional[float], p: CCPParams):
    bp = basepeak_intensity(merged) or 1.0
    thr = bp * p.REL_INT_MIN
    vals = []; mask = []
    DIAG = p.DIAG_AA + p.DIAG_SEQ
    for neu in DIAG:
        best, _ = _best_diag_intensity(merged, neu, shift, precursor_mz, p)
        if best >= thr:
            vals.append(best); mask.append(1)
        else:
            vals.append(0.0); mask.append(0)
    vec = np.sqrt(np.array(vals, float))
    mask = np.array(mask, int)
    cov_total = int(mask.sum())
    cov_high  = int(sum(1 for i, neu in enumerate(DIAG) if mask[i] and neu >= p.HIGH_MASS_CUTOFF))
    return vec, mask, cov_total, cov_high


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.linalg.norm(a); rb = np.linalg.norm(b)
    return float(np.dot(a, b)/(ra*rb)) if ra>0 and rb>0 else 0.0


def coverage_weighted_cos(sample_vec: np.ndarray, mask: np.ndarray, p: CCPParams, ref_vec: Optional[np.ndarray] = None) -> Tuple[float, int]:
    if ref_vec is None:
        ref_vec = np.sqrt(np.ones_like(sample_vec, float))
    cov = int(mask.sum())
    if cov < p.MIN_DIAG_FOR_COS:
        return 0.0, cov
    cos = cosine(sample_vec, ref_vec)
    cw = cos * min(1.0, cov/8.0)
    return cw, cov


def count_anchor_hits(merged: List[Tuple[float, float]], shift_neutral: float, precursor_mz: Optional[float], p: CCPParams) -> int:
    bp = basepeak_intensity(merged) or 1.0
    thr = bp * p.REL_INT_MIN
    n = 0
    for neu in p.ANCHOR_DIAGS:
        ok = False
        for zz in p.DIAG_CHARGES:
            theo = (neu + shift_neutral + zz*p.PROTON)/zz
            if precursor_mz is not None and theo > (precursor_mz + p.FRAG_MZ_MARGIN_DA):
                continue
            inten = max((it for m, it in merged if within_tol(m, theo, p.DIAG_PPM, p.USE_PPM)), default=0.0)
            if inten >= thr:
                ok = True; break
        if ok: n += 1
    return n


def neutral_loss_signals(merged: List[Tuple[float, float]], precursor_neutral: Optional[float], q_mass: Optional[float], precursor_mz: Optional[float], p: CCPParams) -> Dict[str, float]:
    bp = basepeak_intensity(merged) or 1.0
    thr_tag = bp * p.TAG_REL_INT_MIN
    thr_gen = bp * p.REL_INT_MIN
    tag_i = 0.0; pmq_i = 0.0; has_tag = False; has_pmq = False
    # full-tag NL
    for z in p.DIAG_CHARGES:
        theo = (p.TAG_NEUTRAL_MASS + z*p.PROTON)/z
        if precursor_mz is not None and theo > (precursor_mz + p.FRAG_MZ_MARGIN_DA):
            continue
        inten = max((it for m, it in merged if within_tol(m, theo, p.DIAG_PPM, p.USE_PPM)), default=0.0)
        tag_i = max(tag_i, inten)
    has_tag = (tag_i >= thr_tag)
    # precursor - Q
    if q_mass is not None and precursor_neutral is not None:
        target_neutral = precursor_neutral - q_mass
        for z in p.DIAG_CHARGES:
            theo = (target_neutral + z*p.PROTON)/z
            if precursor_mz is not None and theo > (precursor_mz + p.FRAG_MZ_MARGIN_DA):
                continue
            inten = max((it for m, it in merged if within_tol(m, theo, p.DIAG_PPM, p.USE_PPM)), default=0.0)
            pmq_i = max(pmq_i, inten)
    has_pmq = (pmq_i >= thr_gen)
    return {"has_full_tag": has_tag, "full_tag_intensity": tag_i,
            "has_prec_minus_q": has_pmq, "prec_minus_q_intensity": pmq_i}

# ---------------- Gate & Decision ----------------

def pass_ccp_gate(merged: List[Tuple[float, float]], precursor_mz: Optional[float], p: CCPParams) -> Tuple[bool, int, Dict[str, float]]:
    bp = basepeak_intensity(merged) or 1.0
    thr_gate = bp * p.FREE_INT_FLOOR
    DIAG = p.DIAG_AA + p.DIAG_SEQ
    vals = []
    for neu in DIAG:
        best, _ = _best_diag_intensity(merged, neu, 0.0, precursor_mz, p)
        vals.append(best)
    n_free_gate = int(sum(1 for v in vals if v >= thr_gate))
    return (n_free_gate >= p.FREE_MIN_DIAGS), n_free_gate, {"thr_gate": thr_gate}


def decide_confidence(n_shifted: int, cw_shift: float, cw_free: float, cov_shift_high: int, anchor_hits: int, n_free_gate: int) -> str:
    cw_gain = (cw_shift or 0.0) - (cw_free or 0.0)
    if n_free_gate >= 4 and (
        (n_shifted >= 5)
        or (n_shifted >= 4 and (cw_gain >= 0.05 or cov_shift_high >= 2 or anchor_hits >= 2))
        or (n_free_gate >= 5 and cw_free >= 0.20 and n_shifted >= 1)
    ):
        return "High Confidence"
    if n_free_gate >= 4 and (
        (n_shifted >= 3 and (cw_shift >= 0.12 or cw_gain >= 0.06 or cov_shift_high >= 1 or anchor_hits >= 2))
        or (n_free_gate >= 4 and cw_free > 0.15 and n_shifted >= 1)
    ):
        return "Likely"
    if n_shifted >= 2 and (cw_shift >= 0.08 or cw_gain >= 0.04 or anchor_hits >= 1):
        return "Low Confidence"
    return "No Match"

# ======= MINIMAL ADDITIONS FOR CYS-ONLY GATING/DECISION =======

# Cysteine-containing CCP fragment neutrals (CT → SSDQFRPDDCT)
CYS_DIAG = (
    223.074704, 338.101647, 453.128590, 550.181354, 706.282465,
    853.350879, 981.409457, 1096.436400, 1183.468428, 1270.501012
)

# A small convenience to derive SUPPORT/ALL from your existing params
def get_diag_sets(p):
    """
    Returns (CYS_DIAG, SUPPORT_DIAG, ALL_DIAG).
    SUPPORT_DIAG = DIAG_AA + (DIAG_SEQ minus cysteine-containing subset).
    """
    # guard if DIAG_SEQ/DIAG_AA not present (should be in your params already)
    seq = tuple(getattr(p, "DIAG_SEQ", ()))
    aa  = tuple(getattr(p, "DIAG_AA", ()))
    support_seq = tuple(m for m in seq if m not in CYS_DIAG)
    SUPPORT_DIAG = aa + support_seq
    ALL_DIAG = CYS_DIAG + SUPPORT_DIAG
    return CYS_DIAG, SUPPORT_DIAG, ALL_DIAG

def _best_diag_intensity_basic(merged, neutral_mass, shift_neutral, precursor_mz, p):
    """
    Lightweight best-intensity finder using your existing conventions:
      theo m/z = (neutral + shift + z*H+)/z
    Relies on your existing 'within_tol' and 'DIAG_CHARGES' in params.
    """
    # Expect these to exist in your codebase already:
    from math import isfinite
    try:
        within_tol_fn = within_tol  # if within current module
    except NameError:
        raise RuntimeError("within_tol is required in ccp_core for diag matching")

    best = 0.0; best_z = None
    for z in getattr(p, "DIAG_CHARGES", (1,2,3,4)):
        theo = (neutral_mass + shift_neutral + z*p.PROTON) / z
        # keep your original precursor fence behavior: we do NOT enforce a hard cap here
        for m, i in merged:
            if within_tol_fn(m, theo, p.DIAG_PPM, p.USE_PPM) and i > best:
                best = i; best_z = z
    return best, best_z

def diag_hits_on_set(merged, shift_neutral, precursor_mz, p, diag_set):
    """
    Count matches on a specific diagnostic set at your standard relative threshold.
    Returns: (hits_list, cov_total, cov_high, threshold)
      - hits_list = [(neutral_mass, chosen_z, intensity), ...] for matched members of diag_set
      - cov_total = number of distinct diag_set members matched
      - cov_high  = count of matched members >= HIGH_MASS_CUTOFF
    """
    # Expect these to exist:
    try:
        bpi_fn = basepeak_intensity
    except NameError:
        raise RuntimeError("basepeak_intensity is required in ccp_core for diag matching")

    bp = bpi_fn(merged) or 1.0
    thr = bp * getattr(p, "REL_INT_MIN", 0.005)
    seen = set()
    hits = []
    for neu in diag_set:
        best, z = _best_diag_intensity_basic(merged, neu, shift_neutral, precursor_mz, p)
        if best >= thr:
            hits.append((neu, (z or 1), best))
            seen.add(neu)
    cov_total = len(seen)
    high_cut  = getattr(p, "HIGH_MASS_CUTOFF", 700.0)
    cov_high  = sum(1 for neu in seen if neu >= high_cut)
    return hits, cov_total, cov_high, thr

def pass_ccp_gate_cys(merged, precursor_mz, p):
    """
    Gate using *only* cysteine-containing diagnostics.
    Uses your existing FREE_MIN_DIAGS and FREE_INT_FLOOR.
    """
    try:
        bpi_fn = basepeak_intensity
    except NameError:
        raise RuntimeError("basepeak_intensity is required in ccp_core for gating")

    bp = bpi_fn(merged) or 1.0
    thr_gate = bp * getattr(p, "FREE_INT_FLOOR", 0.003)
    n = 0
    for neu in CYS_DIAG:
        best, _ = _best_diag_intensity_basic(merged, neu, 0.0, precursor_mz, p)
        if best >= thr_gate:
            n += 1
    min_diags = getattr(p, "FREE_MIN_DIAGS", 4)
    return (n >= min_diags), n, {"thr_gate": thr_gate}

# --- support-only free gate (gate ignores Cys) ---

def pass_ccp_gate_support_only(merged, precursor_mz, p):
    """
    Gate on NON-cysteine CCP diagnostics only (FREE space, i.e., shift = 0).
    Pass if the number of SUPPORT (non-Cys) diagnostics above FREE_INT_FLOOR
    is >= p.FREE_MIN_DIAGS (default 4).

    Returns:
        (pass_gate: bool, meta: dict)
        meta: {"n_support": int, "thr_gate": float}
    """
    # basepeak-scaled intensity floor (same idea as your original)
    bp = basepeak_intensity(merged) or 1.0
    thr_gate = bp * getattr(p, "FREE_INT_FLOOR", 0.003)

    # Use your existing diagnostic sets
    CYS_SET, SUPPORT_SET, ALL_SET = get_diag_sets(p)

    # Count SUPPORT-only FREE hits above the gate floor
    n_support = 0
    for neu in SUPPORT_SET:
        best, _ = _best_diag_intensity_basic(merged, neu, 0.0, precursor_mz, p)
        if best >= thr_gate:
            n_support += 1

    pass_gate = (n_support >= getattr(p, "FREE_MIN_DIAGS", 4))
    return pass_gate, {"n_support": n_support, "thr_gate": thr_gate}

# ---------------- Isotopes ----------------
try:
    from pyteomics import mzml as _mzml
    _HAS_MZML = True
    _MzMLReader = getattr(_mzml, "IndexedMzML", _mzml.MzML)
except Exception:
    _HAS_MZML = False
    _MzMLReader = None


def find_mzml(sample: str, mzml_dir: str) -> Optional[Path]:
    root = Path(mzml_dir)
    if not root.exists():
        return None
    pats = [f"{sample}*.mzML", f"{sample}*.mzML.gz"]
    cands: List[Path] = []
    for pat in pats: cands.extend(root.rglob(pat))
    if not cands:
        s_low = sample.lower()
        for q in root.rglob("*.mzML*"):
            try:
                if s_low in q.stem.lower(): cands.append(q)
            except Exception: continue
    if not cands:
        return None
    cands = sorted(cands, key=lambda x: (len(x.stem), str(x)))
    return cands[0]


def build_ms1_cache(reader, p: CCPParams):
    if reader is None: return None
    rts = []; mz_lists = []; int_lists = []
    try:
        for item in reader.iterfind("spectrum"):
            spec = item if isinstance(item, dict) else reader.get_by_id(item)
            ms_level = spec.get("ms level", spec.get("msLevel", None))
            if ms_level != 1: continue
            scan = spec.get("scanList", {}).get("scan", [{}])
            rt = None
            if scan and isinstance(scan, list): rt = scan[0].get("scan start time", None)
            if rt is None: continue
            try: rt_min = float(rt)/60.0
            except Exception: continue
            mzs = spec.get("m/z array"); ints = spec.get("intensity array")
            if mzs is None or ints is None: continue
            rts.append(rt_min)
            mz_lists.append(np.asarray(mzs, dtype=float))
            int_lists.append(np.asarray(ints, dtype=float))
    except Exception:
        return None
    if not rts: return None
    return (np.asarray(rts, dtype=float), mz_lists, int_lists)


def agg_ms1_from_cache(ms1_cache, target_mz: float, rt_center_min: float, p: CCPParams):
    if ms1_cache is None: return [], "no_mzml"
    if rt_center_min is None or target_mz is None: return [], "no_rt_or_mz"
    rts, mz_lists, int_lists = ms1_cache
    lo = rt_center_min - p.ISO_RT_WIN_MIN; hi = rt_center_min + p.ISO_RT_WIN_MIN
    idx_lo = int(np.searchsorted(rts, lo, side="left"))
    idx_hi = int(np.searchsorted(rts, hi, side="right"))
    raw = []
    for k in range(idx_lo, idx_hi):
        mzs = mz_lists[k]; ints = int_lists[k]
        mlo = target_mz - p.ISO_MZ_WIN_DA; mhi = target_mz + p.ISO_MZ_WIN_DA
        mask = (mzs >= mlo) & (mzs <= mhi)
        if mask.any():
            sel_m = mzs[mask]; sel_i = ints[mask]
            raw.extend((float(m), float(i)) for m, i in zip(sel_m, sel_i))
    if not raw: return [], "no_ms1_points"
    raw = np.asarray(raw, float)
    idx = np.argsort(raw[:, 0]); mzs = raw[idx, 0]; ints = raw[idx, 1]
    out = []; cm = mzs[0]; ci = ints[0]
    for mz, it in zip(mzs[1:], ints[1:]):
        tol = ppm_da(cm, p.ISO_PPM_TOL)
        if abs(mz - cm) <= tol: ci = max(ci, it)
        else: out.append((cm, ci)); cm, ci = mz, it
    out.append((cm, ci))
    bp = max(i for _, i in out)
    out = [(m, i) for (m, i) in out if i >= bp*p.ISO_INT_RELMIN]
    if not out: return [], "postfilter_all_below_threshold"
    return out, None


def isotope_series_score(agg_ms1_peaks: List[Tuple[float, float]], mono_mz: float, z: int, p: CCPParams) -> int:
    if not agg_ms1_peaks or mono_mz is None or z is None or z <= 0:
        return 0
    d = 1.003355 / z
    def present_at(target: float) -> bool:
        tol = ppm_da(target, p.ISO_PPM_TOL)
        for m, _ in agg_ms1_peaks:
            if abs(m - target) <= tol:
                return True
        return False
    best = 0
    for k0 in (0, 1, 2):
        targets = [mono_mz + (k0 + k)*d for k in (0, 1, 2, 3)]
        got = [present_at(t) for t in targets]
        consec = 0
        for ok in got:
            if ok: consec += 1
            else: break
        score = consec + (1 if consec >= 3 else 0)
        if score > best:
            best = score
            if best >= 4: break
    return best
