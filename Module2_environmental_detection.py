from __future__ import annotations
"""Module 2 — Environmental Detection (CCP-tag discovery)

Purpose
-------
Search environmental LC–MS/MS data for CCP-tagged quinone candidates using the
calibrated diagnostics from Module 1.

What it does (unchanged logic)
------------------------------
- Loads feature tables, mzML/MGF, and a sample manifest
- Checks shifted/free diagnostics, anchors, neutral losses; scores isotopes
- Aggregates evidence and assigns confidence calls
- Writes a comprehensive result table suitable for manuscript figures/supplement

Inputs/Paths
------------
Edit only the top-of-file path config (e.g., CCPParams) if your layout differs.

Outputs
-------
Results CSV(s) with evidence fields and confidence labels, ready for figures.
"""

# -*- coding: utf-8 -*-
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from ccp_core import (
    get_diag_sets, diag_hits_on_set,
    pass_ccp_gate_support_only,  # switched to SUPPORT-only gate (matches Step 2)
)

# === OUTPUT COLUMNS (identical to Step 2) ===
OUTPUT_COLUMNS = [
    "file", "standard", "feature_id",
    "assigned_charge", "precursor_mz", "precursor_neutral",
    "n_free_hits", "n_free_hits_cys",
    "n_shifted_hits", "n_shifted_hits_cys",
    "cov_free_high", "cov_free_high_cys",
    "cov_shift_high", "cov_shift_high_cys",
    "cw_free", "cw_shift",
    "anchor_hits", "nl_full_tag",
    "matched_standard_or_none", "putative_Q_mass", "decision",
    "iso_support_chosen", "isotope_ok_chosen",
 'ms1_feature_intensity', 'ms1_feature_rel_bp',
 "ms2_int_diag", "ms2_int_cys",
 "rt_min",
 ]

# -------------------------- Shared chemistry / library --------------------------
try:
    import ccp_core as core
except ImportError:
    core = None

@dataclass
class CCPParams:
    # I/O
    FEATURE_DIR: str = r"D:/UNR Research/PNNL Char Data/CCP_features"
    MGF_DIR: str     = r"D:/UNR Research/PNNL Char Data/CCP_mgf"
    MZML_DIR: str    = r"D:/UNR Research/PNNL Char Data/PNNL char mzml files"

    # Selection
    SAMPLES: Tuple[str, ...] = (
        "AJB_240612_pos_0002_1_CCP_22",
        "AJB_240612_pos_0002_2_CCP_25",
        "AJB_240612_pos_0050_1_CCP_28",
        "AJB_240612_pos_0050_2_CCP_31",
        "AJB_240627_pos_0002_CCP_new_4",
        "AJB_240627_pos_0007_CCP_new_11",
        "AJB_240627_pos_0068_CCP_new_7",
        "AJB_240627_pos_0070_CCP_new_15",
    )
    ONLY_SAMPLE: Optional[str] = None

    # Mass / ion chemistry
    USE_PPM: bool = True
    PPM_TOL: float = 25.0
    PPM_FOR_MERGE: float = 15.0
    DIAG_PPM: float = 25.0

    PROTON: float = 1.007276466812
    TAG_NEUTRAL_MASS: float = 1269.493187
    MIN_Q_MASS: float = 108.0

    # Diagnostics (neutral masses)
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
    REL_INT_MIN: float = 0.005
    TAG_REL_INT_MIN: float = 0.005
    DIAG_CHARGES: Tuple[int, ...] = (1, 2, 3, 4)
    MIN_DIAG_FOR_COS: int = 4
    HIGH_MASS_CUTOFF: float = 700.0

    # Gate (free diagnostics only)
    FREE_MIN_DIAGS: int = 4
    FREE_INT_FLOOR: float = 0.003

    # Charge tries
    TRY_CHARGES: Tuple[int, ...] = (1, 2, 3, 4)

    # Isotopes
    MS1_PRECACHE: bool = True
    ISO_RT_WIN_MIN: float = 2.0
    ISO_MZ_WIN_DA: float = 4.0
    ISO_PPM_TOL: float = 20.0
    ISO_INT_RELMIN: float = 0.0005
    ISO_HARD_GATE: bool = False

    # Standards (annotation only)
    Q_LIBRARY: Dict[str, float] = None
    Q_MASS_DA_TOL: float = 0.5
    STD_MIN_NSHIFT: int = 2
    STD_MIN_CW: float = 0.10

    # Misc
    FRAG_MZ_MARGIN_DA: float = 0.5

    # Adduct masses for library-consistency check
    PROTON_MASS: float = 1.007276
    NA_ADDUCT_MASS: float = 22.989218
    K_ADDUCT_MASS: float = 38.963158
    MZ_TOL_PPM: float = 20.0  # tolerance for precursor vs theoretical adduct m/z

    def __post_init__(self):
        if self.Q_LIBRARY is None:
            self.Q_LIBRARY = {
                "anthraquinone":        208.05243,
                "benzoquinone":         108.02113,
                "chloro-benzoquinone":  141.98216,
                "methyl-benzoquinone":  122.03678,
                "naphthoquinone":       158.03678,
            }

def _rt_from_row(r) -> float | None:
    """
    Return RT in minutes from the feature row, if any known RT-like column exists.
    We accept either minutes or seconds; if value looks like seconds (>100), convert to minutes.
    """
    import pandas as pd
    for k in ("rt_min_med","rt_min_min","rt_min_max",
              "rt","rt_min","rt_apex_min","rt_center","rt_mean","retention_time","scan_time_min"):
        if k in r and pd.notna(r[k]):
            try:
                v = float(r[k])
                return (v/60.0) if v > 100 else v
            except Exception:
                continue
    return None

# ---------- Math helpers ----------

def ppm_da(mz: float, ppm: float) -> float:
    return mz * ppm / 1e6

def within_tol(obs: float, theo: float, ppm: float, use_ppm: bool) -> bool:
    return abs(obs - theo) <= (ppm_da(theo, ppm) if use_ppm else 0.5)

ALL_DIAG = None  # built at runtime

# -------------------------- MGF parsing & merge --------------------------

def parse_mgf(fp: str) -> Dict[str, List[List[Tuple[float, float]]]]:
    spectra = {}
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        in_ions = False; params = {}; peaks = []
        for line in f:
            s = line.strip()
            if s == "BEGIN IONS":
                in_ions = True; params = {}; peaks = []
            elif s == "END IONS":
                fid = params.get("FEATURE_ID")
                if fid:
                    spectra.setdefault(str(fid), []).append(peaks[:])
                in_ions = False
            elif in_ions and '=' in s:
                k, v = s.split('=', 1); params[k] = v
            elif in_ions and s and ' ' in s and not s.startswith('#'):
                toks = s.split()
                try:
                    mz = float(toks[0]); inten = float(toks[1])
                    peaks.append((mz, inten))
                except Exception:
                    pass
    return spectra


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
            ci = max(ci, it)
        else:
            out.append((cm, ci))
            cm, ci = mz, it
    out.append((cm, ci))
    return out


def basepeak_intensity(merged: List[Tuple[float, float]]) -> float:
    return max((i for _, i in merged), default=0.0)

# -------------------------- Diagnostics (MS2) --------------------------

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


def make_sample_vector(merged: List[Tuple[float, float]], shift: float, precursor_mz: Optional[float], p: CCPParams):
    bp = basepeak_intensity(merged) or 1.0
    thr = bp * p.REL_INT_MIN
    vals = []; mask = []
    for neu in ALL_DIAG:
        best, _ = _best_diag_intensity(merged, neu, shift, precursor_mz, p)
        if best >= thr:
            vals.append(best); mask.append(1)
        else:
            vals.append(0.0); mask.append(0)
    vec = np.sqrt(np.array(vals, float))
    mask = np.array(mask, int)
    return vec, mask


def coverage_weighted_cos(sample_vec: np.ndarray, mask: np.ndarray, p: CCPParams, ref_vec: Optional[np.ndarray] = None) -> Tuple[float, int]:
    if ref_vec is None:
        ref_vec = np.sqrt(np.ones_like(sample_vec, float))
    cov = int(mask.sum())
    if cov < p.MIN_DIAG_FOR_COS:
        return 0.0, cov
    ra = np.linalg.norm(sample_vec); rb = np.linalg.norm(ref_vec)
    cos = float(np.dot(sample_vec, ref_vec)/(ra*rb)) if ra>0 and rb>0 else 0.0
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
    out = {"has_full_tag": False, "full_tag_intensity": 0.0, "has_prec_minus_q": False, "prec_minus_q_intensity": 0.0}
    # full-tag NL
    for z in p.DIAG_CHARGES:
        theo = (p.TAG_NEUTRAL_MASS + z*p.PROTON)/z
        if precursor_mz is not None and theo > (precursor_mz + p.FRAG_MZ_MARGIN_DA):
            continue
        inten = max((it for m, it in merged if within_tol(m, theo, p.DIAG_PPM, p.USE_PPM)), default=0.0)
        if inten >= thr_tag:
            out["has_full_tag"] = True
            out["full_tag_intensity"] = max(out["full_tag_intensity"], inten)
    # precursor - Q
    if q_mass is not None and precursor_neutral is not None:
        target = precursor_neutral - q_mass
        for z in p.DIAG_CHARGES:
            theo = (target + z*p.PROTON)/z
            if precursor_mz is not None and theo > (precursor_mz + p.FRAG_MZ_MARGIN_DA):
                continue
            inten = max((it for m, it in merged if within_tol(m, theo, p.DIAG_PPM, p.USE_PPM)), default=0.0)
            if inten >= thr_gen:
                out["has_prec_minus_q"] = True
                out["prec_minus_q_intensity"] = max(out["prec_minus_q_intensity"], inten)
    return out

# -------------------------- Confidence --------------------------

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


def precursor_is_plausible(precursor_mz: Optional[float], z: Optional[int], p: CCPParams) -> bool:
    if precursor_mz is None or z is None or z <= 0:
        return False
    min_mz = (p.TAG_NEUTRAL_MASS + p.MIN_Q_MASS + z*p.PROTON)/z
    tol = max(ppm_da(min_mz, 5), 0.01)
    return (precursor_mz + tol) >= min_mz

# -------------------------- Robust mzML support (MS1 / isotopes) --------------------------
try:
    from pyteomics import mzml as _mzml
    _HAS_MZML = True
    _MzMLReader = getattr(_mzml, "IndexedMzML", _mzml.MzML)
except Exception:
    _HAS_MZML = False
    _MzMLReader = None


def _find_mzml(sample: str, p: CCPParams) -> Optional[Path]:
    root = Path(p.MZML_DIR)
    if not root.exists():
        print(f"[WARN] MZML_DIR does not exist: {root}")
        return None
    pats = [f"{sample}*.mzML", f"{sample}*.mzML.gz"]
    cands: List[Path] = []
    for pat in pats:
        cands.extend(root.rglob(pat))
    if not cands:
        s_low = sample.lower()
        for q in root.rglob("*.mzML*"):
            try:
                if s_low in q.stem.lower():
                    cands.append(q)
            except Exception:
                continue
    if not cands:
        print(f"[WARN] No mzML found for sample '{sample}' under {root}")
        return None
    cands = sorted(cands, key=lambda x: (len(x.stem), str(x)))
    hit = cands[0]
    print(f"[INFO] mzML mapped: sample='{sample}' -> '{hit.name}'")
    return hit


def _build_ms1_cache(reader, p: CCPParams):
    if reader is None: return None
    rts = []; mz_lists = []; int_lists = []
    try:
        for item in reader.iterfind("spectrum"):
            spec = item if isinstance(item, dict) else reader.get_by_id(item)
            ms_level = spec.get("ms level", spec.get("msLevel", None))
            if ms_level != 1: continue
            scan = spec.get("scanList", {}).get("scan", [{}])
            rt = None
            if scan and isinstance(scan, list):
                rt = scan[0].get("scan start time", None)
            if rt is None: continue
            try:
                rt_min = float(rt) / 60.0  # seconds->minutes
            except Exception:
                continue
            mzs = spec.get("m/z array"); ints = spec.get("intensity array")
            if mzs is None or ints is None: continue
            rts.append(rt_min)
            mz_lists.append(np.asarray(mzs, dtype=float))
            int_lists.append(np.asarray(ints, dtype=float))
    except Exception:
        return None
    if not rts: return None
    return (np.asarray(rts, dtype=float), mz_lists, int_lists)


def _agg_ms1_from_cache(ms1_cache, target_mz: float, rt_center_min: float, p: CCPParams):
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


def _agg_ms1_around_indexed(reader, target_mz: float, rt_center_min: float, p: CCPParams):
    if reader is None: return [], "no_mzml"
    if rt_center_min is None or target_mz is None: return [], "no_rt_or_mz"
    raw = []
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
            if abs(rt_min - rt_center_min) > p.ISO_RT_WIN_MIN: continue
            mzs = spec.get("m/z array"); ints = spec.get("intensity array")
            if mzs is None or ints is None: continue
            lo = target_mz - p.ISO_MZ_WIN_DA; hi = target_mz + p.ISO_MZ_WIN_DA
            for m, i in zip(mzs, ints):
                if lo <= m <= hi:
                    raw.append((float(m), float(i)))
    except Exception:
        return [], "read_error"
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


def _isotope_series_score(agg_ms1_peaks: List[Tuple[float, float]], mono_mz: float, z: int, p: CCPParams) -> int:
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

def _ms1_feature_metrics(agg_ms1_peaks: List[Tuple[float, float]],
                         target_mz: Optional[float],
                         p: CCPParams) -> Tuple[float, float]:
    """
    Return (abs_intensity, rel_to_basepeak) for the MS1 signal at target_mz
    within the aggregated MS1 window used for isotopes.
    """
    if not agg_ms1_peaks or target_mz is None:
        return 0.0, 0.0
    bp = max(i for _, i in agg_ms1_peaks) or 1.0
    tol = ppm_da(target_mz, p.ISO_PPM_TOL) if p.USE_PPM else p.ISO_MZ_WIN_DA
    local = [i for m, i in agg_ms1_peaks if abs(m - target_mz) <= tol]
    if not local:
        return 0.0, 0.0
    imax = float(max(local))
    return imax, float(imax / bp)

# -------------------------- Free-CCP gate (SUPPORT-only) --------------------------
# handled via pass_ccp_gate_support_only from ccp_core

# -------------------------- Standards acceptance --------------------------

def _library_accepts(q_used: Optional[float], lib_mass: Optional[float], n_shift_obs: int, cw_shift_obs: float, p: CCPParams) -> bool:
    if (q_used is None) or (lib_mass is None):
        return False
    if abs(q_used - lib_mass) > p.Q_MASS_DA_TOL:
        return False
    return (n_shift_obs >= p.STD_MIN_NSHIFT) or (cw_shift_obs >= p.STD_MIN_CW)

# -------------------------- Main --------------------------

def main():
    p = CCPParams()
    global ALL_DIAG
    ALL_DIAG = p.DIAG_AA + p.DIAG_SEQ
    CYS_SET, SUPPORT_SET, ALL_SET = get_diag_sets(p)

    samples = [s for s in p.SAMPLES if (p.ONLY_SAMPLE is None or s == p.ONLY_SAMPLE)]
    if not samples:
        print("[WARN] No samples selected.")
        return

    rows: List[Dict[str, object]] = []

    for sample in samples:
        feat_csv = Path(p.FEATURE_DIR)/f"{sample}_features.csv"
        mgf_fp   = Path(p.MGF_DIR)/f"{sample}.mgf"
        if not feat_csv.exists() or not mgf_fp.exists():
            print(f"[WARN] Missing inputs for {sample}; skipping.")
            continue

        feats = pd.read_csv(feat_csv)
        mgf = parse_mgf(str(mgf_fp))

        # mzML (optional for isotopes)
        reader = None; ms1_cache = None
        mzml_path = None
        try:
            mzml_path = next((q for q in Path(p.MZML_DIR).rglob("*.mzML*") if sample.lower() in q.stem.lower()), None)
        except Exception:
            mzml_path = None

        if _HAS_MZML and mzml_path is not None:
            try:
                reader = _MzMLReader(str(mzml_path))
                print(f"[INFO] mzML reader: {_MzMLReader.__name__}")
            except Exception:
                try:
                    reader = _mzml.MzML(str(mzml_path))
                    print("[INFO] fallback mzml.MzML opened")
                except Exception as e2:
                    print(f"[WARN] Could not open mzML: {e2}")
                    reader = None
            if p.MS1_PRECACHE and reader is not None:
                print(f"[INFO] building MS1 cache for {sample} ...")
                ms1_cache = _build_ms1_cache(reader, p)
                if ms1_cache is not None:
                    try: reader.close()
                    except Exception: pass
                    reader = None
                    print(f"[INFO] MS1 cache built: {len(ms1_cache[0])} scans")

        for _, r in feats.iterrows():
            fid = str(r.get("id"))
            if fid not in mgf:
                continue
            merged = merge_peaks(mgf[fid], p)
            if not merged:
                continue

            mz_precursor = float(r.get("mz", np.nan)) if pd.notna(r.get("mz", np.nan)) else None
            reported_z = int(r.get("charge", 0)) if pd.notna(r.get("charge", np.nan)) and int(r.get("charge", 0))>0 else None
            rt_here      = _rt_from_row(r)

            # ---------- 1) FREE-CCP GATE (SUPPORT-only) ----------
            gate_ok, gate_meta = pass_ccp_gate_support_only(merged, mz_precursor, p)
            n_free_gate = gate_meta.get("n_support", 0)
            # --- NEW: defaults so they exist for every branch ---
            ms1_abs_intensity = 0.0
            ms1_rel_bp = 0.0
            if not gate_ok:
                # transparent row
                rows.append({
                    "file": sample,
                    "standard": "unknown",
                    "feature_id": fid,
                    "assigned_charge": (reported_z or 1),
                    "precursor_mz": mz_precursor,
                    "precursor_neutral": (mz_precursor*(reported_z or 1) - (reported_z or 1)*p.PROTON) if mz_precursor is not None else None,
                    "n_free_hits": n_free_gate,
                    "n_free_hits_cys": 0,
                    "n_shifted_hits": 0,
                    "n_shifted_hits_cys": 0,
                    "cov_free_high": 0,
                    "cov_free_high_cys": 0,
                    "cov_shift_high": 0,
                    "cov_shift_high_cys": 0,
                    "cw_free": 0.0,
                    "cw_shift": 0.0,
                    "anchor_hits": 0,
                    "nl_full_tag": False,
                    "matched_standard_or_none": "none",
                    "putative_Q_mass": None,
                    "decision": "Uncertain (gated)",
                    "iso_support_chosen": 0,
                    "isotope_ok_chosen": False,
                    "ms1_feature_intensity": ms1_abs_intensity,
                    "ms1_feature_rel_bp": ms1_rel_bp,
                    "rt_min": rt_here,
                })
                continue

            # ---------- 2) MS1 isotopes (compute ONCE per feature) ----------
            agg_ms1_once = []
            iso_scores_allz = {z: 0 for z in (1,2,3,4)}
            rt_min = None
            for k in ("rt_min_med", "rt_min_min", "rt_min_max", "rt", "rt_min", "rt_apex_min", "rt_center", "rt_mean", "retention_time"):
                if k in r and pd.notna(r[k]):
                    try: rt_min = float(r[k]); break
                    except Exception: pass
            if (rt_min is not None) and (mz_precursor is not None):
                if ms1_cache is not None:
                    rts = ms1_cache[0]
                    idx = int(np.searchsorted(rts, rt_min))
                    if idx <= 0: rt_for_ms1 = float(rts[0])
                    elif idx >= len(rts): rt_for_ms1 = float(rts[-1])
                    else:
                        rt_for_ms1 = float(rts[idx] if abs(rts[idx]-rt_min) < abs(rts[idx-1]-rt_min) else rts[idx-1])
                    agg_ms1_once, _ = _agg_ms1_from_cache(ms1_cache, mz_precursor, rt_for_ms1, p)
                elif reader is not None:
                    agg_ms1_once, _ = _agg_ms1_around_indexed(reader, mz_precursor, rt_min, p)
                if agg_ms1_once:
                    for z in (1,2,3,4):
                        iso_scores_allz[z] = _isotope_series_score(agg_ms1_once, mz_precursor, z, p)
            # --- NEW: MS1 feature metrics (intensity and relative-to-basepeak) ---
            ms1_abs_intensity, ms1_rel_bp = _ms1_feature_metrics(agg_ms1_once, mz_precursor, p)
            any_iso = any(s >= 1 for s in iso_scores_allz.values())
            best_z_hint = max(iso_scores_allz, key=lambda z: iso_scores_allz[z]) if agg_ms1_once else None
            best_iso_s  = iso_scores_allz.get(best_z_hint, 0) if best_z_hint else 0
            # --- NEW: compute MS1 metrics once (safe even if agg_ms1_once is empty) ---
            ms1_abs_intensity, ms1_rel_bp = _ms1_feature_metrics(agg_ms1_once, mz_precursor, p)

            # ---------- 3) Z hypotheses ----------
            z_hypos: List[int] = []
            if reported_z and reported_z not in z_hypos: z_hypos.append(reported_z)
            if best_z_hint and best_z_hint not in z_hypos: z_hypos.append(best_z_hint)
            for z in p.TRY_CHARGES:
                if z not in z_hypos: z_hypos.append(z)

            best_pick = None

            for z_assumed in z_hypos:
                if not precursor_is_plausible(mz_precursor, z_assumed, p):
                    continue

                precursor_neutral = (mz_precursor*z_assumed - z_assumed*p.PROTON) if mz_precursor is not None else None

                # FREE diagnostics (CYS set for legacy mapping of "n_free_hits")
                CYS_SET, SUPPORT_SET, ALL_SET = get_diag_sets(p)
                free_hits_cys, cov_free_cys, cov_free_hi_cys, _ = diag_hits_on_set(merged, 0.0, mz_precursor, p, CYS_SET)
                free_hits_all, cov_free_all, cov_free_hi_all, _ = diag_hits_on_set(merged, 0.0, mz_precursor, p, ALL_SET)
                svec_free, mask_free = make_sample_vector(merged, 0.0, mz_precursor, p)
                cw_free, _ = coverage_weighted_cos(svec_free, mask_free, p)

                # Candidate shifted hypotheses
                def shifted_metrics(q_mass):
                    if q_mass is None:
                        return dict(n_shifted=0, cov_shift_hi=0, cw_shift=0.0, anchor_hits=0, nl={"has_prec_minus_q": False, "has_full_tag": False})
                    sh_hits_all, cov_shift_all, cov_shift_hi_all, _ = diag_hits_on_set(merged, q_mass, mz_precursor, p, ALL_SET)
                    svec_shift, mask_shift = make_sample_vector(merged, q_mass, mz_precursor, p)
                    cw_shift, _ = coverage_weighted_cos(svec_shift, mask_shift, p)
                    anchors = count_anchor_hits(merged, q_mass, mz_precursor, p)
                    nl = neutral_loss_signals(merged, precursor_neutral, q_mass, mz_precursor, p)
                    return dict(
                        n_shifted=len(sh_hits_all),
                        cov_shift_hi=cov_shift_hi_all,
                        cw_shift=cw_shift,
                        anchor_hits=anchors,
                        nl=nl
                    )

                candidates = []
                # A) from precursor
                q_from_prec = None if precursor_neutral is None else float(precursor_neutral - p.TAG_NEUTRAL_MASS)
                if (q_from_prec is not None) and (q_from_prec >= p.MIN_Q_MASS):
                    candidates.append(("from_prec", q_from_prec, shifted_metrics(q_from_prec)))

                # B) adduct-aware library hypotheses (H/Na/K; z 1..3)
                if p.Q_LIBRARY and (mz_precursor is not None):
                    PROTON = p.PROTON_MASS; NA = p.NA_ADDUCT_MASS; K = p.K_ADDUCT_MASS
                    for name, q_lib in p.Q_LIBRARY.items():
                        expected_neutral = q_lib + p.TAG_NEUTRAL_MASS
                        best_ppm = None; is_consistent = False
                        for add_name, add_mass in (("H", PROTON), ("Na", NA), ("K", K)):
                            for z in (1,2,3):
                                theo_mz = (expected_neutral + add_mass)/z
                                dp = 1e6 * abs(mz_precursor - theo_mz)/max(theo_mz, 1e-12)
                                if (best_ppm is None) or (dp < best_ppm): best_ppm = dp
                                if dp <= p.MZ_TOL_PPM:
                                    is_consistent = True
                                    break
                            if is_consistent: break
                        if is_consistent:
                            candidates.append((f"from_lib:{name}", q_lib, shifted_metrics(q_lib)))

                # Pick the winner by (n_shifted, cw_shift, anchor_hits)
                if candidates:
                    label, q_used, met = max(candidates, key=lambda x: (x[2]["n_shifted"], x[2]["cw_shift"], x[2]["anchor_hits"]))
                    n_shifted = met["n_shifted"]
                    cov_shift_hi = met["cov_shift_hi"]
                    cw_shift = met["cw_shift"]
                    nl = met["nl"]
                    # CYS-only shifted metrics for reporting
                    sh_hits_cys, cov_shift_cys, cov_shift_hi_cys, _ = diag_hits_on_set(merged, q_used, mz_precursor, p, CYS_SET)
                    # NEW: ALL-diag shifted hits for intensity totals
                    sh_hits_all, _, _, _ = diag_hits_on_set(merged, q_used, mz_precursor, p, ALL_SET)
                    ms2_int_diag = float(sum(i for _, _, i in sh_hits_all))
                    ms2_int_cys  = float(sum(i for _, _, i in sh_hits_cys))
                else:
                    label = "none"; q_used = None
                    n_shifted = 0; cov_shift_hi = 0; cw_shift = 0.0
                    nl = {"has_full_tag": False, "has_prec_minus_q": False}
                    ms2_int_diag = 0.0
                    ms2_int_cys  = 0.0

                anchor_hits = count_anchor_hits(merged, (q_used if q_used is not None else 0.0), mz_precursor, p)
                decision = decide_confidence(n_shifted, cw_shift, cw_free, cov_shift_hi, anchor_hits, n_free_gate)

                iso_support_chosen = iso_scores_allz.get(z_assumed, 0) if agg_ms1_once else 0

                # Standards annotation uses chosen Q now
                matched_std = "none"
                if (q_used is not None) and p.Q_LIBRARY:
                    # 1) Mass pre-filter around the chosen Q (ppm if enabled; else DA)
                    if p.USE_PPM:
                        mass_ok = lambda qmass: abs(qmass - q_used) <= ppm_da(q_used, p.MZ_TOL_PPM)
                    else:
                        mass_ok = lambda qmass: abs(qmass - q_used) <= p.Q_MASS_DA_TOL
                
                    # 2) Build candidates that are mass-consistent with q_used
                    cand = []
                    for qname, qmass in p.Q_LIBRARY.items():
                        if not mass_ok(qmass):
                            continue
                        svec_lib, mask_lib = make_sample_vector(merged, qmass, mz_precursor, p)
                        cw_lib, _ = coverage_weighted_cos(svec_lib, mask_lib, p)
                        n_lib = int(mask_lib.sum())
                        # Rank: closest mass first, then evidence
                        delta_ppm = (abs(qmass - q_used) / max(q_used, 1e-12)) * 1e6
                        cand.append((delta_ppm, n_lib, cw_lib, qname, qmass))
                
                    # 3) Pick closest-in-mass library; tie-break on (n_lib, cw_lib)
                    if cand:
                        cand.sort(key=lambda t: (t[0], -t[1], -t[2]))  # smaller ppm first, then more coverage/cosine
                        _, _, _, best_name, best_mass = cand[0]
                        if _library_accepts(q_used, best_mass, n_shifted, cw_shift, p):
                            matched_std = best_name

                res = {
                    "file": sample,
                    "standard": "unknown",
                    "feature_id": fid,
                    "assigned_charge": z_assumed,
                    "precursor_mz": mz_precursor,
                    "precursor_neutral": precursor_neutral,
                    "n_free_hits": len(free_hits_all),
                    "n_free_hits_cys": len(free_hits_cys),
                    "n_shifted_hits": n_shifted,
                    "n_shifted_hits_cys": (len(sh_hits_cys) if 'sh_hits_cys' in locals() else 0),
                    "cov_free_high": (cov_free_hi_all if 'cov_free_hi_all' in locals() else 0),
                    "cov_free_high_cys": (cov_free_hi_cys if 'cov_free_hi_cys' in locals() else 0),
                    "cov_shift_high": cov_shift_hi,
                    "cov_shift_high_cys": (cov_shift_hi_cys if 'cov_shift_hi_cys' in locals() else 0),
                    "cw_free": cw_free,
                    "cw_shift": cw_shift,
                    "anchor_hits": anchor_hits,
                    "nl_full_tag": bool(nl.get("has_full_tag", False)),
                    "matched_standard_or_none": matched_std,
                    "putative_Q_mass": q_used,
                    "decision": decision,
                    "iso_support_chosen": int(iso_support_chosen),
                    "isotope_ok_chosen": bool(iso_support_chosen >= 1),
                    "ms2_int_diag": ms2_int_diag,
                    "ms2_int_cys":  ms2_int_cys,
                }

                # Rank preference: valid Q, decision strength, isotope hint, shifted metrics, free metrics
                rank_tuple = (
                    1 if (q_used is not None) else 0,
                    {"High Confidence":3, "Likely":2, "Low Confidence":1, "No Match":0}[decision],
                    1 if any_iso else 0,
                    res["n_shifted_hits"], res["cov_shift_high"], res["cw_shift"],
                    res["n_free_hits"], res["cov_free_high"], res["cw_free"],
                )
                if (best_pick is None) or (rank_tuple > best_pick[0]):
                    best_pick = (rank_tuple, res)

            if best_pick is None:
                z0 = reported_z if reported_z else 1
                rows.append({
                    "file": sample,
                    "standard": "unknown",
                    "feature_id": fid,
                    "assigned_charge": z0,
                    "precursor_mz": mz_precursor,
                    "precursor_neutral": (mz_precursor*z0 - z0*p.PROTON) if mz_precursor is not None else None,
                    "n_free_hits": n_free_gate,
                    "n_free_hits_cys": 0,
                    "n_shifted_hits": 0,
                    "n_shifted_hits_cys": 0,
                    "cov_free_high": 0,
                    "cov_free_high_cys": 0,
                    "cov_shift_high": 0,
                    "cov_shift_high_cys": 0,
                    "cw_free": 0.0,
                    "cw_shift": 0.0,
                    "anchor_hits": 0,
                    "nl_full_tag": False,
                    "matched_standard_or_none": "none",
                    "putative_Q_mass": None,
                    "decision": "No Match",
                    "iso_support_chosen": 0,
                    "isotope_ok_chosen": False,
                    "ms1_feature_intensity": ms1_abs_intensity,
                    "ms1_feature_rel_bp": ms1_rel_bp,
                    "ms2_int_diag": 0.0,
                    "ms2_int_cys":  0.0,
                    "rt_min": rt_here,
                })
            else:
                res = best_pick[1]
                rows.append(res)
                res["ms1_feature_intensity"] = ms1_abs_intensity
                res["ms1_feature_rel_bp"] = ms1_rel_bp

        try:
            if reader is not None:
                reader.close()
        except Exception:
            pass

    # === WRITE OUTPUT (Step 2–matched schema) ===
    out = pd.DataFrame(rows)


    # Keep only slim columns identical to Step 2
    cols = [c for c in OUTPUT_COLUMNS if c in out.columns]
    if not out.empty and cols:
        sort_cols = [c for c in ["file", "standard", "decision", "n_shifted_hits", "n_free_hits"] if c in out.columns]
        if sort_cols:
            # True = ascending, False = descending
            asc = [c not in ("n_shifted_hits", "n_free_hits") for c in sort_cols]
            out = out.sort_values(by=sort_cols, ascending=asc)
        out = out[cols]
    else:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)

    only = CCPParams().ONLY_SAMPLE
    out_path = f"step3_environmental_results_{only}.csv" if only else "step3_environmental_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  (rows={len(out)})")

if __name__ == "__main__":
    main()



# [behavior] Library hypotheses evaluation — applies *only* to the 5 library standards.
# This function tries Q (from library), Q−H, and Q+H hypotheses, scoring each with the same diagnostic model.
def evaluate_library_hypotheses(feature, lib_q_mass, params: CCPParams, spectrum, ms1_cache):
    candidates = []
    # Nominal variants: library Q, Q−H, Q+H
    variants = [
        ("from_lib", lib_q_mass),
        ("from_lib_minusH", lib_q_mass - 1.007276466812),
        ("from_lib_plusH", lib_q_mass + 1.007276466812),
    ]
    for tag, q_mass in variants:
        # SUPPORT-only gating and low-mass relaxations
        low_mass = is_low_mass_quinone(q_mass)
        # Compute diagnostics using core helpers; step-specific implementations should already be refactored to core.
        # Pseudocode placeholders below assume existing local glue code calls to:
        # - get_diag_sets / diag_hits_on_set
        # - make_sample_vector / coverage_weighted_cos / count_anchor_hits etc.
        try:
            diag_sets = get_diag_sets(params)
            hits = diag_hits_on_set(spectrum, q_mass, params, which="SUPPORT")
            gate_ok = pass_ccp_gate_support_only(hits, params, low_mass=low_mass)
            if not gate_ok:
                continue
            # Score shifted evidence (anchor counts + coverage-weighted cosine) via core utilities
            sample_vec = make_sample_vector(hits)
            cw, n_shifted = coverage_weighted_cos(sample_vec), count_anchor_hits(hits)
            # MS1 context
            ms1_int, ms1_rel_bp, iso_score = agg_ms1_from_cache(ms1_cache, feature, params)
            conf = decide_confidence(cw, n_shifted, low_mass=low_mass)
            candidates.append({
                "mode": tag,
                "q_mass": q_mass,
                "cw_shift": cw,
                "n_shifted": n_shifted,
                "ms1_feature_intensity": ms1_int,
                "ms1_feature_rel_bp": ms1_rel_bp,
                "isotope_series_score": iso_score,
                "confidence": conf,
            })
        except Exception as e:
            candidates.append({"mode": tag, "error": str(e)})
    # Pick best by (confidence, n_shifted, cw_shift)
    def rank_key(c):
        return (c.get("confidence", ""), c.get("n_shifted", 0), c.get("cw_shift", 0.0))
    candidates.sort(key=rank_key, reverse=True)
    return candidates


# [note] Non-target search uses unified SUPPORT-only gates and thresholds from core, but *does not* expand to Q±H hypotheses.
# This keeps runtime modest while preserving parity in gating logic with Step 2.

# step3_fragment_report.py
# Usage:
# reports file, neutral mass, charge, shift used, theoretical mz, intensity, and percent of total MS2 for each fragment matched to those in the library           
# reports feature 4563 by default, can change to any list of feature_ids

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

# --- import Step 3 + core helpers (no changes to your pipeline required) ---
import step3_environmental as S3   # provides CCPParams, parse_mgf, merge_peaks, etc.
from ccp_core import get_diag_sets, diag_hits_on_set  # fragment matching helpers

def _theo_mz(neutral_mass: float, shift_neutral: float, z: int, p: S3.CCPParams) -> float:
    return (neutral_mass + shift_neutral + z*p.PROTON) / z

def _collect_hits(merged: List[Tuple[float, float]],
                  precursor_mz: float | None,
                  q_mass: float | None,
                  p: S3.CCPParams) -> pd.DataFrame:
    """
    Build a tidy table of hits for:
      - free_all  (ALL diag set, no shift)
      - free_cys  (CYS-only, no shift)
      - shift_all (ALL diag set, shifted by q_mass)
      - shift_cys (CYS-only, shifted by q_mass)
    """
    CYS_SET, SUPPORT_SET, ALL_SET = get_diag_sets(p)  # core’s canonical sets

    # total MS2 intensity = sum of merged intensities
    total_i = sum(i for _, i in merged) or 1.0

    rows = []

    # FREE (no shift)
    for label, diag_set, shift in [
        ("free_all", ALL_SET, 0.0),
        ("free_cys", CYS_SET, 0.0),
    ]:
        hits, _, _, _ = diag_hits_on_set(merged, shift, precursor_mz, p, diag_set)
        for neu, z, inten in hits:
            rows.append(dict(
                bucket=label,
                neutral_mass=float(neu),
                z=int(z),
                shift_used=float(shift),
                theo_mz=_theo_mz(neu, shift, z, p),
                intensity=float(inten),
                pct_total_ms2=100.0 * float(inten)/total_i
            ))

    # SHIFTED (by best Q mass) if available
    if q_mass is not None:
        for label, diag_set, shift in [
            ("shift_all", ALL_SET, float(q_mass)),
            ("shift_cys", CYS_SET, float(q_mass)),
        ]:
            hits, _, _, _ = diag_hits_on_set(merged, shift, precursor_mz, p, diag_set)
            for neu, z, inten in hits:
                rows.append(dict(
                    bucket=label,
                    neutral_mass=float(neu),
                    z=int(z),
                    shift_used=float(shift),
                    theo_mz=_theo_mz(neu, shift, z, p),
                    intensity=float(inten),
                    pct_total_ms2=100.0 * float(inten)/total_i
                ))

    df = pd.DataFrame(rows, columns=[
        "bucket", "neutral_mass", "z", "shift_used", "theo_mz",
        "intensity", "pct_total_ms2"
    ]).sort_values(["bucket", "neutral_mass"]).reset_index(drop=True)

    return df

def _find_results_csv() -> Path:
    # Prefer ONLY_SAMPLE-specific output if present; otherwise the default file
    candidates = [Path("step3_environmental_results.csv")] + list(Path(".").glob("step3_environmental_results_*.csv"))
    for fp in candidates:
        if fp.exists():
            return fp
    raise FileNotFoundError("Could not find Step 3 results CSV (step3_environmental_results*.csv).")

def report_features(feature_ids: List[str | int]) -> None:
    p = S3.CCPParams()  # uses your configured paths/tolerances
    results_fp = _find_results_csv()
    res = pd.read_csv(results_fp, dtype={"feature_id": str})
    want = set(str(x) for x in feature_ids)

    # We’ll search all selected samples for these feature_ids
    out_tables: Dict[str, pd.DataFrame] = {}

    # Group by sample(file) to resolve the correct MGF once per file
    for file_name, group in res.groupby("file"):
        mgf_path = Path(p.MGF_DIR) / f"{file_name}.mgf"
        if not mgf_path.exists():
            print(f"[WARN] MGF not found for {file_name}: {mgf_path}")
            continue

        mgf = S3.parse_mgf(str(mgf_path))

        # Scan rows in this sample that match our feature_ids
        sub = group[group["feature_id"].astype(str).isin(want)]
        if sub.empty:
            continue

        for _, row in sub.iterrows():
            fid = str(row["feature_id"])
            if fid not in mgf:
                print(f"[WARN] feature_id {fid} has no MS2 in {mgf_path.name}; skipping")
                continue

            # Merge all replicate spectra for the feature (using the same algorithm as Step 3)
            merged = S3.merge_peaks(mgf[fid], p)
            if not merged:
                print(f"[WARN] feature_id {fid} has empty merged MS2; skipping")
                continue

            precursor_mz = float(row["precursor_mz"]) if pd.notna(row["precursor_mz"]) else None
            q_mass = float(row["putative_Q_mass"]) if pd.notna(row["putative_Q_mass"]) else None

            tbl = _collect_hits(merged, precursor_mz, q_mass, p)
            tbl.insert(0, "feature_id", fid)
            tbl.insert(1, "file", file_name)
            out_tables[f"{file_name}:{fid}"] = tbl

    if not out_tables:
        print("No matching features found with available MS2.")
        return

    # Print a concise console view and also write one CSV per feature
    out_dir = Path("fragment_reports")
    out_dir.mkdir(exist_ok=True)
    for key, df in out_tables.items():
        file_name, fid = key.split(":")
        csv_path = out_dir / f"fragment_report_{file_name}_{fid}.csv"
        df.to_csv(csv_path, index=False)
        print("\n" + "="*88)
        print(f"Feature {fid} — Sample: {file_name}")
        print("="*88)
        # Console preview (rounded for readability)
        print(df.assign(
            neutral_mass=lambda d: d["neutral_mass"].round(6),
            theo_mz=lambda d: d["theo_mz"].round(6),
            intensity=lambda d: d["intensity"].round(1),
            pct_total_ms2=lambda d: d["pct_total_ms2"].round(3),
        ).to_string(index=False))
        print(f"[WROTE] {csv_path}")

if __name__ == "__main__":
    feats = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [4563]
    report_features(feats)

# ======================================================
# Post-run MS² cysteine intensity summary 
# ======================================================
import pandas as pd
import numpy as np
from pathlib import Path

# Change this if your Step3 output is elsewhere
STEP3_RESULTS_PATH = Path("step3_environmental_results.csv")

# Load the results
try:
    df = pd.read_csv(STEP3_RESULTS_PATH)
except Exception as e:
    raise FileNotFoundError(f"Could not read {STEP3_RESULTS_PATH}: {e}")

# Filter to your target samples only
target_samples = {
    "AJB_240612_pos_0002_1_CCP_22",
    "AJB_240612_pos_0002_2_CCP_25",
    "AJB_240627_pos_0002_CCP_new_4",
}
df = df[df["file"].isin(target_samples)].copy()

# Check what columns exist
cols = df.columns.str.lower().tolist()

# --- Case 1: fragment intensities already summarized in the Step3 table ---
if {"ms2_int_cys", "ms2_int_diag"}.issubset(cols):
    df["frac_cys_of_diag"] = np.where(
        df["ms2_int_diag"] > 0,
        df["ms2_int_cys"] / df["ms2_int_diag"],
        np.nan,
    )
    hc = df[df["decision"].str.contains("High Confidence", case=False, na=False)].copy()
    med = np.nanmedian(hc["frac_cys_of_diag"])
    iqr25 = np.nanpercentile(hc["frac_cys_of_diag"], 25)
    iqr75 = np.nanpercentile(hc["frac_cys_of_diag"], 75)
    pct70 = 100 * np.nanmean(hc["frac_cys_of_diag"] >= 0.70)

    print("\n=== MS² cysteine diagnostic intensity summary ===")
    print(f"Samples analyzed: {', '.join(sorted(target_samples))}")
    print(f"High-confidence adducts: {len(hc)}")
    print(f"Median cysteine share of diagnostic MS² intensity: {med*100:.1f}% (IQR {iqr25*100:.1f}–{iqr75*100:.1f}%)")
    print(f"Features ≥70% cysteine share: {pct70:.1f}%")
else:
    print("\n[Info] Step3 results do not yet contain MS² intensity columns (`ms2_int_cys`, `ms2_int_diag`).")
    print("To enable the intensity-based metric, add the small snippet that records these in your loop,")
    print("then re-run Step3 once. After that, this block will compute the statistics automatically.")
