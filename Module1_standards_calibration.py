from __future__ import annotations
"""Module 1 — Standards (CCP-tag validation & calibration)

Purpose
-------
Validate CCP-tagged quinone standards and calibrate the evidence logic used
for environmental discovery.

What it does (unchanged logic)
------------------------------
- Loads standards data (mzML / spectra tables) and metadata
- Searches for CCP-tagged diagnostic ions, neutral losses, and anchor fragments
- Applies isotope and coverage checks
- Aggregates evidence per candidate/cluster
- Writes a tidy CSV summarizing evidence and confidence calls for each standard

Inputs/Paths
------------
Edit only the path constants at the top of this file if your layout differs.

Outputs
-------
Table with diagnostics/anchors/isotopes and confidence calls.
"""

from ccp_core import CCPParams
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from ccp_core import get_diag_sets, diag_hits_on_set, pass_ccp_gate_support_only

# === OUTPUT COLUMNS (slim CSV) ===
OUTPUT_COLUMNS = [
    "file", "standard", "feature_id",
    "assigned_charge", "precursor_mz", "precursor_neutral",
    # fragment counts (TOTAL + CYS-only)
    "n_free_hits", "n_free_hits_cys",
    "n_shifted_hits", "n_shifted_hits_cys",
    # keep only the HIGH flags
    "cov_free_high", "cov_free_high_cys",
    "cov_shift_high", "cov_shift_high_cys",
    # similarity + anchors + NL
    "cw_free", "cw_shift",
    "anchor_hits", "nl_full_tag",
    # standards + decision
    "matched_standard_or_none", "putative_Q_mass", "decision",
    # isotopes (keep only chosen)
    "iso_support_chosen", "isotope_ok_chosen",
    # absolute MS1 intensity at precursor m/z
    "ms1_feature_intensity",
    # feature_intensity / MS1 base peak
    "ms1_feature_rel_bp",
]

# ================== USER SETTINGS ==================
MZML_DIR_STD = r"D:/UNR Research/PNNL Char Data/Quinone STDs and CCP"

# Filenames WITHOUT extensions (the script will glob *<name>*.mzML / *.mzML.gz)
SAMPLES = [
    "POS_AQ_CCP",
    "POS_BQ_CCP",
    "POS_CBQ_CCP",
    "POS_MBQ_CCP",
    "POS_NQ_CCP",
]

# Map each standards run → expected library key (to avoid name drift)
EXPECTED = {
    "POS_AQ_CCP":  "anthraquinone",
    "POS_BQ_CCP":  "benzoquinone",
    "POS_CBQ_CCP": "chloro-benzoquinone",
    "POS_MBQ_CCP": "methyl-benzoquinone",
    "POS_NQ_CCP":  "naphthoquinone",
}
# ====================================================

# ---------- Prefer aligned core if present ----------
try:
    import ccp_core_aligned as core
except Exception:
    core = None

# ---------- Local fallbacks (kept identical to core) ----------
if core is None:
    @dataclass
    class CCPParams:
        USE_PPM: bool = True
        DIAG_PPM: float = 40.0
        PPM_FOR_MERGE: float = 15.0
        FRAG_MZ_MARGIN_DA: float = 0.5
        PROTON: float = 1.007276466812
        TAG_NEUTRAL_MASS: float = 1269.493187
        MIN_Q_MASS: float = 108.0
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
        REL_INT_MIN: float = 0.005
        TAG_REL_INT_MIN: float = 0.005
        DIAG_CHARGES: Tuple[int, ...] = (1, 2, 3, 4)
        MIN_DIAG_FOR_COS: int = 4
        HIGH_MASS_CUTOFF: float = 700.0
        FREE_MIN_DIAGS: int = 4
        FREE_INT_FLOOR: float = 0.003
        ISO_RT_WIN_MIN: float = 2.0
        ISO_MZ_WIN_DA: float = 4.0
        ISO_PPM_TOL: float = 20.0
        ISO_INT_RELMIN: float = 0.0005

    def ppm_da(mz: float, ppm: float) -> float:
        return mz * ppm / 1e6

    def within_tol(obs: float, theo: float, ppm: float, use_ppm: bool) -> bool:
        return abs(obs - theo) <= (ppm_da(theo, ppm) if use_ppm else 0.5)

    def merge_peaks(peaklists: List[List[Tuple[float, float]]], p: CCPParams) -> List[Tuple[float, float]]:
        if not peaklists: return []
        arr = np.array([(mz, i) for pl in peaklists for mz, i in pl], float)
        if arr.size == 0: return []
        idx = np.argsort(arr[:, 0]); mzs = arr[idx, 0]; ints = arr[idx, 1]
        out = []; cm = mzs[0]; ci = ints[0]
        for mz, it in zip(mzs[1:], ints[1:]):
            tol = ppm_da(cm, p.PPM_FOR_MERGE) if p.USE_PPM else 0.1
            if abs(mz - cm) <= tol:
                if it > ci: cm, ci = mz, it
            else:
                out.append((cm, ci)); cm, ci = mz, it
        out.append((cm, ci))
        return out

    def basepeak_intensity(merged: List[Tuple[float, float]]) -> float:
        return max((i for _, i in merged), default=0.0)

    def _best_diag_intensity(merged, neu_mass, shift_neutral, precursor_mz, p):
        best = 0.0; best_z = None
        for z in p.DIAG_CHARGES:
            theo = (neu_mass + shift_neutral + z*p.PROTON) / z
            if precursor_mz is not None and theo > (precursor_mz + p.FRAG_MZ_MARGIN_DA):
                continue
            for m, i in merged:
                if within_tol(m, theo, p.DIAG_PPM, p.USE_PPM) and i > best:
                    best = i; best_z = z
        return best, best_z

    def diag_hits(merged, shift_neutral, precursor_mz, p):
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

    def make_sample_vector(merged, shift, precursor_mz, p):
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

    def cosine(a, b):
        ra = np.linalg.norm(a); rb = np.linalg.norm(b)
        return float(np.dot(a, b)/(ra*rb)) if ra>0 and rb>0 else 0.0

    def coverage_weighted_cos(sample_vec, mask, p, ref_vec=None):
        if ref_vec is None:
            ref_vec = np.sqrt(np.ones_like(sample_vec, float))
        cov = int(mask.sum())
        if cov < p.MIN_DIAG_FOR_COS:
            return 0.0, cov
        cos = cosine(sample_vec, ref_vec)
        cw = cos * min(1.0, cov/8.0)
        return cw, cov

    def count_anchor_hits(merged, shift_neutral, precursor_mz, p):
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

    def neutral_loss_signals(merged, precursor_neutral, q_mass, precursor_mz, p):
        bp = basepeak_intensity(merged) or 1.0
        thr_tag = bp * p.TAG_REL_INT_MIN
        thr_gen = bp * p.REL_INT_MIN
        out = {"has_full_tag": False, "full_tag_intensity": 0.0, "has_prec_minus_q": False, "prec_minus_q_intensity": 0.0}
        for z in p.DIAG_CHARGES:
            theo = (p.TAG_NEUTRAL_MASS + z*p.PROTON)/z
            if precursor_mz is not None and theo > (precursor_mz + p.FRAG_MZ_MARGIN_DA):
                continue
            inten = max((it for m, it in merged if within_tol(m, theo, p.DIAG_PPM, p.USE_PPM)), default=0.0)
            if inten >= thr_tag:
                out["has_full_tag"] = True
                out["full_tag_intensity"] = max(out["full_tag_intensity"], inten)
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

    def pass_ccp_gate(merged, precursor_mz, p):
        bp = basepeak_intensity(merged) or 1.0
        thr_gate = bp * p.FREE_INT_FLOOR
        DIAG = p.DIAG_AA + p.DIAG_SEQ
        vals = []
        for neu in DIAG:
            best, _ = _best_diag_intensity(merged, neu, 0.0, precursor_mz, p)
            vals.append(best)
        n_free_gate = int(sum(1 for v in vals if v >= thr_gate))
        return (n_free_gate >= p.FREE_MIN_DIAGS), n_free_gate, {"thr_gate": thr_gate}

    def decide_confidence(n_shifted, cw_shift, cw_free, cov_shift_high, anchor_hits, n_free_gate):
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

    # isotopes
    try:
        from pyteomics import mzml as _mzml
        _HAS_MZML = True
        _MzMLReader = getattr(_mzml, "IndexedMzML", _mzml.MzML)
    except Exception:
        _HAS_MZML = False
        _MzMLReader = None

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

    def isotope_series_score(agg_ms1_peaks, mono_mz, z, p):
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

else:
    CCPParams = core.CCPParams
    ppm_da = core.ppm_da
    within_tol = core.within_tol
    merge_peaks = core.merge_peaks
    basepeak_intensity = core.basepeak_intensity
    _best_diag_intensity = core._best_diag_intensity
    diag_hits = core.diag_hits
    make_sample_vector = core.make_sample_vector
    coverage_weighted_cos = core.coverage_weighted_cos
    count_anchor_hits = core.count_anchor_hits
    neutral_loss_signals = core.neutral_loss_signals
    pass_ccp_gate = core.pass_ccp_gate
    decide_confidence = core.decide_confidence
    from pyteomics import mzml as _mzml
    _MzMLReader = getattr(core, "_MzMLReader", _mzml.MzML)
    agg_ms1_from_cache = core.agg_ms1_from_cache
    isotope_series_score = core.isotope_series_score

# ---------- mzML IO helpers ----------

def find_mzml(sample: str, root: Path) -> Optional[Path]:
    pats = [f"{sample}*.mzML", f"{sample}*.mzML.gz", f"*{sample}*.mzML", f"*{sample}*.mzML.gz"]
    cands: List[Path] = []
    for pat in pats:
        cands.extend(root.rglob(pat))
    if not cands:
        return None
    cands = sorted(cands, key=lambda x: (len(x.stem), str(x)))
    return cands[0]


def build_ms1_cache(reader, p: CCPParams):
    rts = []; mz_lists = []; int_lists = []
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
    if not rts: return None
    return (np.asarray(rts, dtype=float), mz_lists, int_lists)


def extract_ms2(reader) -> List[Dict[str, object]]:
    out = []
    for item in reader.iterfind("spectrum"):
        spec = item if isinstance(item, dict) else reader.get_by_id(item)
        ms_level = spec.get("ms level", spec.get("msLevel", None))
        if ms_level != 2: continue
        scan = spec.get("scanList", {}).get("scan", [{}])
        rt = None
        if scan and isinstance(scan, list):
            rt = scan[0].get("scan start time", None)
        if rt is None:
            continue
        try:
            rt_min = float(rt)/60.0
        except Exception:
            continue
        precs = spec.get("precursorList", {}).get("precursor", [])
        if not precs: continue
        sel = precs[0].get("selectedIonList", {}).get("selectedIon", [])
        if not sel: continue
        sel0 = sel[0]
        mz = sel0.get("selected ion m/z") or sel0.get("selectedIon m/z") or sel0.get("m/z")
        if mz is None:
            continue
        try:
            mz = float(mz)
        except Exception:
            continue
        z = sel0.get("charge state") or sel0.get("charge")
        try:
            z = int(z) if z is not None else None
        except Exception:
            z = None
        mzs = spec.get("m/z array"); ints = spec.get("intensity array")
        if mzs is None or ints is None:
            continue
        peaks = [(float(a), float(b)) for a, b in zip(mzs, ints) if b is not None]
        out.append({"scan_id": spec.get("id", None), "rt_min": rt_min, "precursor_mz": mz, "charge": z, "peaks": peaks})
    return out

# ---------- Clustering MS2 into pseudo-features ----------

def cluster_ms2(ms2: List[Dict[str, object]], p: CCPParams, ppm: float = 15.0, rt_win_min: float = 0.6):
    if not ms2: return []
    ms2 = sorted(ms2, key=lambda d: (d["precursor_mz"], d["rt_min"]))
    clusters: List[Dict[str, object]] = []
    for rec in ms2:
        mz = rec["precursor_mz"]; rt = rec["rt_min"]
        placed = False
        for cl in clusters:
            cmz = cl["mz_med"]; crt = cl["rt_med"]
            if abs(mz - cmz) <= ppm_da(cmz, ppm) and abs(rt - crt) <= rt_win_min:
                cl["mz_list"].append(mz)
                cl["rt_list"].append(rt)
                cl["z_list"].append(rec["charge"]) 
                cl["peaks_lists"].append(rec["peaks"]) 
                # update centroid
                cl["mz_med"] = float(np.median(cl["mz_list"]))
                cl["rt_med"] = float(np.median(cl["rt_list"]))
                placed = True
                break
        if not placed:
            clusters.append({
                "mz_list": [mz], "rt_list": [rt], "z_list": [rec["charge"]],
                "peaks_lists": [rec["peaks"]],
                "mz_med": float(mz), "rt_med": float(rt),
            })
    return clusters

# ---------- Library masses ----------
Q_LIBRARY = {
    "anthraquinone":        208.05243,
    "benzoquinone":         108.02113,
    "chloro-benzoquinone":  141.98216,
    "methyl-benzoquinone":  122.03678,
    "naphthoquinone":       158.03678,
}

def ppm(diff, ref):
    return 1e6 * abs(diff) / max(ref, 1e-9)

# ---------- Main ----------

def main():
    p = CCPParams()
    root = Path(MZML_DIR_STD)
    rows: List[Dict[str, object]] = []
    CYS_SET, SUPPORT_SET, ALL_SET = get_diag_sets(p)

    for sample in SAMPLES:
        mzml = find_mzml(sample, root)
        if not mzml or not mzml.exists():
            print(f"[WARN] mzML not found for {sample} under {root}")
            continue
        print(f"[INFO] Processing: {mzml.name}")
        reader = _MzMLReader(str(mzml))

        # Build MS1 cache for isotope scoring
        print("[INFO] Building MS1 cache...")
        ms1_cache = build_ms1_cache(reader, p)
        print("[INFO] MS1 cache:", "ok" if ms1_cache is not None else "none")

        # Extract MS2 and cluster into pseudo-features
        ms2 = extract_ms2(reader)
        clusters = cluster_ms2(ms2, p, ppm=15.0, rt_win_min=0.6)
        print(f"[INFO] Found {len(ms2)} MS2 scans; {len(clusters)} clusters")

        # Per-cluster scoring (LEGACY BEHAVIOR + Cys reporting)
        expected_std = EXPECTED.get(sample)
        for idx, cl in enumerate(clusters, 1):
            fid = f"{sample}_c{idx}"
            merged = merge_peaks(cl["peaks_lists"], p)
            if not merged:
                continue

            # mode charge (fallback to 1)
            z_counts = Counter([z for z in cl["z_list"] if z is not None and z > 0])
            z_assigned = int(z_counts.most_common(1)[0][0]) if z_counts else 1
            mz_precursor = float(cl["mz_med"]) if cl["mz_med"] else None
            precursor_neutral = (mz_precursor * z_assigned - z_assigned * p.PROTON) if mz_precursor is not None else None
            
            # --- NEW: base-peak & low-mass mode ---
            bp_merged = basepeak_intensity(merged) or 0.0
            
            # Identify the expected standard and its library mass
            q_lib_mass = Q_LIBRARY.get((expected_std or "").lower())
            LOW_MASS_MODE = bool(q_lib_mass is not None and q_lib_mass < 120.0)
            
            # ---- Low-mass mode toggle (internal only; no column) ----
            expected_std = EXPECTED.get(sample)
            q_lib_mass = Q_LIBRARY.get((expected_std or "").lower())
            LOW_MASS = bool(q_lib_mass is not None and q_lib_mass < 120.0)
            
            # Use a copy of params so we can lower thresholds without side effects
            p_use = p
            if LOW_MASS:
                p_use = CCPParams(**p.__dict__)  # shallow copy of your current params
                # Lower the free-floor and general rel intensity floor by ~40%
                p_use.FREE_INT_FLOOR = max(0.0015, p.FREE_INT_FLOOR * 0.6)
                p_use.REL_INT_MIN    = max(0.0015, p.REL_INT_MIN    * 0.6)

            # ---------------- GATE: SUPPORT-only FREE (non-Cys) ----------------
            gate_ok, gate_meta = pass_ccp_gate_support_only(merged, mz_precursor, p_use)
            n_free_gate = gate_meta["n_support"]  # for transparency in gated rows
            gate_thr_used = bp_merged * p_use.REL_INT_MIN
            
            if not gate_ok:
                rows.append({
                    "file": sample, "standard": expected_std or "unknown", "feature_id": fid,
                    "assigned_charge": z_assigned, "precursor_mz": mz_precursor,
                    "precursor_neutral": precursor_neutral,
                    "n_free_hits": n_free_gate, "n_free_hits_cys": 0,
                    "n_shifted_hits": 0, "n_shifted_hits_cys": 0,
                    "cov_free_high": 0, "cov_free_high_cys": 0,
                    "cov_shift_high": 0, "cov_shift_high_cys": 0,
                    "cw_free": 0.0, "cw_shift": 0.0,
                    "anchor_hits": 0, "nl_full_tag": False,
                    "matched_standard_or_none": "none",
                    "putative_Q_mass": None, "decision": "Uncertain (gated)",
                    "iso_support_chosen": 0, "isotope_ok_chosen": False,
                })
                continue
            
            # -------- SAFE DEFAULTS (exactly as in your file) --------
            free_hits_cys = []; free_hits_sup = []; free_hits_all = []
            sh_hits_cys = []; sh_hits_sup = []; sh_hits_all = []
            cov_free_cys = cov_free_hi_cys = 0
            cov_free_sup = cov_free_hi_sup = 0
            cov_free_all = cov_free_hi_all = 0
            cov_shift_cys = cov_shift_hi_cys = 0
            cov_shift_sup = cov_shift_hi_sup = 0
            cov_shift_all = cov_shift_hi_all = 0
            n_free = cov_free = cov_free_hi = 0
            n_shifted = cov_shift = cov_shift_hi = 0
            cw_free = 0.0; cw_shift = 0.0
            nl = {"has_full_tag": False, "full_tag_intensity": 0.0,
                  "has_prec_minus_q": False, "prec_minus_q_intensity": 0.0}
            
            # ---------------- FREE (shift = 0) ----------------
            free_hits_cys, cov_free_cys, cov_free_hi_cys, _ = diag_hits_on_set(merged, 0.0, mz_precursor, p_use, CYS_SET)
            free_hits_sup, cov_free_sup, cov_free_hi_sup, _ = diag_hits_on_set(merged, 0.0, mz_precursor, p_use, SUPPORT_SET)
            free_hits_all, cov_free_all, cov_free_hi_all, _ = diag_hits_on_set(merged, 0.0, mz_precursor, p_use, ALL_SET)
            svec_free, mask_free, _, _ = make_sample_vector(merged, 0.0, mz_precursor, p_use)
            cw_free, _ = coverage_weighted_cos(svec_free, mask_free, p_use)
            
            # ---------------- SHIFTED candidates (unchanged) ----------------
            q_from_prec = None if precursor_neutral is None else float(precursor_neutral - p.TAG_NEUTRAL_MASS)
            q_valid     = (q_from_prec is not None) and (q_from_prec >= p.MIN_Q_MASS)
            
            # After computing q_from_prec (whatever formula you currently use)
            q_lib = Q_LIBRARY.get((expected_std or "").lower())
            def plausible(qp, qlib):
                # strict mass agreement for standards: precursor-derived Q must be within ±2 Da of library Q
                return (qp is not None) and (qlib is not None) and (abs(qp - qlib) <= 2.0)
            
            def shifted_metrics(q_mass):
                """Compute shifted diagnostics/cosine/anchors for a given Q mass hypothesis."""
                if q_mass is None:
                    return dict(
                        n_shifted=0, cov_shift=0, cov_shift_hi=0,
                        cw_shift=0.0, anchor_hits=0,
                        nl={"has_prec_minus_q": False}
                    )
            
                # All diagnostics (use p_use so low-mass mode applies)
                sh_hits_all, cov_shift_all, cov_shift_hi_all, _ = diag_hits_on_set(
                    merged, q_mass, mz_precursor, p_use, ALL_SET
                )
            
                # Coverage-weighted cosine on the shifted vector
                svec_shift, mask_shift, _, _ = make_sample_vector(merged, q_mass, mz_precursor, p_use)
                cw_shift, _ = coverage_weighted_cos(svec_shift, mask_shift, p_use)
            
                # Anchors and neutral-loss checks
                anchors = count_anchor_hits(merged, q_mass, mz_precursor, p_use)
                nl = neutral_loss_signals(merged, precursor_neutral, q_mass, mz_precursor, p_use)
            
                return dict(
                    n_shifted=len(sh_hits_all),
                    cov_shift=cov_shift_all,
                    cov_shift_hi=cov_shift_hi_all,
                    cw_shift=cw_shift,
                    anchor_hits=anchors,
                    nl=nl
                )
            
            candidates = []
            if plausible(q_from_prec, q_lib):
                candidates.append(("from_prec", q_from_prec, shifted_metrics(q_from_prec)))
            # (keep your library candidates too)
            
            # -------- Library hypothesis with M±1 precursor check + Q±H variants --------
            q_from_lib = Q_LIBRARY.get((expected_std or "").lower())
            if q_from_lib is not None and mz_precursor is not None:
                expected_neutral = q_from_lib + p.TAG_NEUTRAL_MASS
                ppm_tol = getattr(p, "MZ_TOL_PPM", 20.0)
                PROTON = getattr(p, "PROTON_MASS", 1.007276)
                NA     = getattr(p, "NA_ADDUCT_MASS", 22.989218)
                K      = getattr(p, "K_ADDUCT_MASS", 38.963158)
            
                def ppm(a, b): return 1e6 * abs(a - b) / max(b, 1e-12)
            
                # Precursor adduct consistency with M±1 tolerance (informational; not required to add candidates)
                lib_is_consistent = False
                best_ppm = None
                best_hyp = None
                for adduct_name, add_mass in (("H", PROTON), ("Na", NA), ("K", K)):
                    for z in (1, 2, 3):
                        mono = (expected_neutral + add_mass) / z
                        c13  = 1.003355 / z
                        for iso_k in (-1, 0, +1):
                            expected_mz = mono + iso_k * c13
                            dp = ppm(mz_precursor, expected_mz)
                            if (best_ppm is None) or (dp < best_ppm):
                                best_ppm = dp
                                best_hyp = (z, adduct_name if iso_k == 0 else f"{adduct_name} (M{iso_k:+d})")
                            if dp <= ppm_tol:
                                lib_is_consistent = True
                                break
                        if lib_is_consistent: break
                    if lib_is_consistent: break
            
                # Evaluate Q, Q−H, Q+H regardless of adduct consistency; let shifted evidence decide
                q_minus_H = q_from_lib - PROTON
                q_plus_H  = q_from_lib + PROTON
            
                candidates.append(("from_lib",        q_from_lib, shifted_metrics(q_from_lib)))
                candidates.append(("from_lib_minusH", q_minus_H,  shifted_metrics(q_minus_H)))
                candidates.append(("from_lib_plusH",  q_plus_H,   shifted_metrics(q_plus_H)))
            
            # ---------------- Pick winner & decide ----------------
            label = None
            putative_Q_mass = None
            if candidates:
                label, q_used, met = max(
                    candidates,
                    key=lambda x: (x[2]["n_shifted"], x[2]["cw_shift"], x[2]["anchor_hits"])
                )
                n_shifted    = met["n_shifted"]
                cov_shift    = met["cov_shift"]
                cov_shift_hi = met["cov_shift_hi"]
                cw_shift     = met["cw_shift"]
                nl           = met["nl"]
                anchor_hits  = met["anchor_hits"]
            
                # Split shifted hits for reporting (Cys vs SUPPORT) with p_use
                sh_hits_cys,  cov_shift_cys,  cov_shift_hi_cys,  _ = diag_hits_on_set(
                    merged, q_used, mz_precursor, p_use, CYS_SET
                )
                sh_hits_sup,  cov_shift_sup,  cov_shift_hi_sup,  _ = diag_hits_on_set(
                    merged, q_used, mz_precursor, p_use, SUPPORT_SET
                )
                n_shifted_hits_cys = len(sh_hits_cys)
            
                # Report canonical library mass when winner is library-derived; keep raw if from_prec
                if label.startswith("from_lib"):
                    putative_Q_mass = q_from_lib
                else:
                    putative_Q_mass = q_used
            
                # Decision: for <120 Da, ignore anchors and allow n_shifted ≥ 1
                if LOW_MASS and (putative_Q_mass is not None) and (putative_Q_mass < 120.0):
                    decision = "Likely" if n_shifted >= 1 else decide_confidence(
                        n_shifted, cw_shift, cw_free, cov_shift_hi, 0, n_free_gate
                    )
                else:
                    decision = decide_confidence(
                        n_shifted, cw_shift, cw_free, cov_shift_hi, anchor_hits, n_free_gate
                    )
            
                # ---------------- Standards label (require precursor arithmetic within ±2 Da) ----------------
                matched_std = "none"
                if q_from_lib is not None:
                    # Always check mass using the precursor-derived quinone, not the reported mass
                    q_from_prec_for_check = q_from_prec  # (computed earlier as precursor_neutral - TAG_NEUTRAL_MASS)
                    mass_ok = (q_from_prec_for_check is not None) and (abs(q_from_prec_for_check - q_from_lib) <= 2.0)
                    evidence_ok = (n_shifted >= (1 if LOW_MASS else 2)) or (cw_shift >= 0.10)
                    if mass_ok and evidence_ok:
                        matched_std = (expected_std or "library_std")
                    else:
                        # (Optional) if mass fails, hard-downgrade decision so it won’t look like a valid call
                        if decision in ("High Confidence", "Likely"):
                            decision = "No Match"
            
                # ---------------- Isotopes (unchanged) ----------------
                iso_support_best = 0; iso_support_chosen = 0; isotope_ok_any = False; isotope_ok_chosen = False
                if ms1_cache is not None and mz_precursor is not None:
                    agg_ms1_once, _ = agg_ms1_from_cache(ms1_cache, mz_precursor, cl["rt_med"], p)
                    if agg_ms1_once:
                        scores = {z: isotope_series_score(agg_ms1_once, mz_precursor, z, p) for z in (1, 2, 3, 4)}
                        iso_support_best = max(scores.values())
                        iso_support_chosen = scores.get(z_assigned, 0)
                        isotope_ok_any = iso_support_best >= 1
                        isotope_ok_chosen = iso_support_chosen >= 1
            
                # --- MS1 intensities for comparison ---
                ms1_feature_intensity = 0.0
                ms1_feature_rel_bp = 0.0
                if (ms1_cache is not None) and (mz_precursor is not None):
                    agg_ms1_once, _ = agg_ms1_from_cache(ms1_cache, mz_precursor, cl["rt_med"], p)
                    if agg_ms1_once:
                        bp_ms1 = 0.0; best = 0.0
                        for m, i in agg_ms1_once:
                            if i > bp_ms1: bp_ms1 = i
                        tol_ppm = getattr(p, "ISO_PPM_TOL", 8.0)
                        ppm_da_loc = lambda m, ppm: (ppm/1e6)*max(m, 1e-12)
                        for m, i in agg_ms1_once:
                            if abs(m - mz_precursor) <= ppm_da_loc(mz_precursor, tol_ppm):
                                if i > best: best = i
                        ms1_feature_intensity = float(best)
                        ms1_feature_rel_bp = float((best / bp_ms1) if bp_ms1 > 0 else 0.0)
            
                # --------------- Append success row ---------------
                rows.append({
                    "file": sample,
                    "standard": expected_std or "unknown",
                    "feature_id": fid,
                    "assigned_charge": z_assigned,
                    "precursor_mz": mz_precursor,
                    "precursor_neutral": precursor_neutral,
                    "n_free_hits": len(free_hits_all),
                    "n_free_hits_cys": len(free_hits_cys),
                    "n_shifted_hits": n_shifted,
                    "n_shifted_hits_cys": n_shifted_hits_cys,
                    "cov_free_high": cov_free_hi_all,
                    "cov_free_high_cys": cov_free_hi_cys,
                    "cov_shift_high": cov_shift_hi,
                    "cov_shift_high_cys": cov_shift_hi_cys,
                    "cw_free": cw_free,
                    "cw_shift": cw_shift,
                    "anchor_hits": anchor_hits,
                    "nl_full_tag": bool(nl.get("has_full_tag", False)),
                    "matched_standard_or_none": matched_std,
                    "putative_Q_mass": putative_Q_mass,
                    "decision": decision,
                    "iso_support_chosen": int(iso_support_chosen),
                    "isotope_ok_chosen": bool(isotope_ok_chosen),
                    "ms1_feature_intensity": ms1_feature_intensity,
                    "ms1_feature_rel_bp": ms1_feature_rel_bp,
                })
            
            else:
                # -------- No candidates: write a minimal row (rare) --------
                rows.append({
                    "file": sample, "standard": expected_std or "unknown", "feature_id": fid,
                    "assigned_charge": z_assigned, "precursor_mz": mz_precursor,
                    "precursor_neutral": precursor_neutral,
                    "n_free_hits": len(free_hits_all), "n_free_hits_cys": len(free_hits_cys),
                    "n_shifted_hits": 0, "n_shifted_hits_cys": 0,
                    "cov_free_high": cov_free_hi_all, "cov_free_high_cys": cov_free_hi_cys,
                    "cov_shift_high": 0, "cov_shift_high_cys": 0,
                    "cw_free": cw_free, "cw_shift": 0.0,
                    "anchor_hits": 0, "nl_full_tag": False,
                    "matched_standard_or_none": "none",
                    "putative_Q_mass": None, "decision": "No Match",
                    "iso_support_chosen": 0, "isotope_ok_chosen": False,
                    "ms1_feature_intensity": 0.0, "ms1_feature_rel_bp": 0.0,
                })


        try:
            reader.close()
        except Exception:
                    pass
        
        out = pd.DataFrame(rows)
        
        # Keep only the slimmed output columns
        cols = [c for c in OUTPUT_COLUMNS if c in out.columns]
        if not out.empty and cols:
            # Optional: sort with a stable, compact order
            sort_cols = [c for c in ["file","standard","decision","n_shifted_hits","n_free_hits"] if c in out.columns]
            if sort_cols:
                asc = [(c not in ("n_shifted_hits","n_free_hits")) for c in sort_cols]
                out = out.sort_values(sort_cols, ascending=asc)
            out = out[cols]
        else:
            out = pd.DataFrame(columns=OUTPUT_COLUMNS)
        
        out.to_csv("step2_standard_validation.csv", index=False)
        print(f"Wrote: step2_standard_validation.csv (rows={len(out)})")

if __name__ == "__main__":
    main()


