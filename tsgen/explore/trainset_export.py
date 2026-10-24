from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from tsgen.data.xyz import Frame, write_xyz_multiframe
from tsgen.utils.geometry import pair_distance_fingerprint, fingerprint_distance
from tsgen.utils.molecular_identity import molecular_graph, same_molecular_identity
from tsgen.explore.config import ExploreCfg
from tsgen.explore.oracle_gaussian import GaussianJobSpec, GaussianOracle
from tsgen.trainset.reaxff_io import LEGACY_BGF_ATOM_FORMAT, LEGACY_BGF_CONECT_FORMAT


_Z_TO_SYMBOL = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 53: "I",
}


def _safe_name(s: str, max_len: int = 96) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:max_len] or "unknown"


def read_single_xyz(path: Path) -> Frame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid xyz file: {path}")
    n = int(lines[0].strip())
    comment = lines[1].strip()
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2:2 + n]:
        p = line.split()
        if len(p) < 4:
            continue
        symbols.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    if len(symbols) != n:
        raise ValueError(f"XYZ atom count mismatch in {path}: expected {n}, got {len(symbols)}")
    return Frame(symbols=symbols, coords=np.asarray(coords, dtype=float), comment=comment)


def _parse_scf_energy_hartree(text: str) -> Optional[float]:
    vals = re.findall(r"SCF Done:\s+E\([A-Za-z0-9]+\)\s*=\s*([-]?\d+\.\d+)", text)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except Exception:
        return None


def parse_gaussian_optimized_frame(log_path: Path, fallback_symbols: Optional[list[str]] = None) -> Optional[Frame]:
    """Parse the last Standard/Input orientation block from a Gaussian log.

    This returns the final geometry available in the log. It works for Opt jobs
    and also for SP logs that print an orientation block.
    """
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    lines = text.splitlines()
    starts: list[int] = []
    for i, line in enumerate(lines):
        if "Standard orientation:" in line or "Input orientation:" in line:
            starts.append(i)
    if not starts:
        return None

    for start in reversed(starts):
        # Find dashed separators; Gaussian block has separator, header, separator, rows, separator.
        dashes = []
        for j in range(start + 1, min(start + 20, len(lines))):
            if re.match(r"\s*-{10,}\s*$", lines[j]):
                dashes.append(j)
            if len(dashes) >= 2:
                break
        if len(dashes) < 2:
            continue
        row_start = dashes[1] + 1
        symbols: list[str] = []
        coords: list[list[float]] = []
        for j in range(row_start, len(lines)):
            if re.match(r"\s*-{10,}\s*$", lines[j]):
                break
            p = lines[j].split()
            if len(p) < 6:
                continue
            try:
                z = int(p[1])
                x, y, zc = float(p[3]), float(p[4]), float(p[5])
            except Exception:
                continue
            sym = _Z_TO_SYMBOL.get(z)
            if sym is None and fallback_symbols and len(fallback_symbols) > len(symbols):
                sym = fallback_symbols[len(symbols)]
            if sym is None:
                sym = f"X{z}"
            symbols.append(sym)
            coords.append([x, y, zc])
        if symbols:
            comment = f"optimized_from={log_path.parent.name} E_hartree={_parse_scf_energy_hartree(text) or ''}"
            return Frame(symbols=symbols, coords=np.asarray(coords, dtype=float), comment=comment)
    return None


_COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02,
}


def _formula_from_frame(frame: Frame) -> str:
    counts: dict[str, int] = {}
    for sym in frame.symbols:
        counts[str(sym)] = counts.get(str(sym), 0) + 1
    # Hill-like for CHO; C,H first, then alphabetical.
    order: list[str] = []
    if "C" in counts:
        order.append("C")
    if "H" in counts:
        order.append("H")
    order.extend([k for k in sorted(counts) if k not in set(order)])
    return "".join(f"{el}{counts[el] if counts[el] != 1 else ''}" for el in order)


def _bond_order_for_pair(a: str, b: str, d: float) -> str:
    pair = "-".join(sorted((a, b)))
    # Conservative distance rules for gas-phase optimized CHO structures.  These
    # are used only for family labels/coverage, not for quantum chemistry.
    if pair == "C-C":
        if d <= 1.25:
            return "triple"
        if d <= 1.43:
            return "double"
        if d <= 1.72:
            return "single"
    if pair == "C-O":
        if d <= 1.18:
            return "triple"
        if d <= 1.32:
            return "double"
        if d <= 1.68:
            return "single"
    if pair == "O-O":
        if d <= 1.32:
            return "double"
        if d <= 1.72:
            return "single"
    if pair == "C-H" and d <= 1.25:
        return "single"
    if pair == "H-O" and d <= 1.18:
        return "single"
    if pair == "H-H" and d <= 0.90:
        return "single"
    # Generic fallback for uncommon allowed elements.
    ra = _COVALENT_RADII.get(a)
    rb = _COVALENT_RADII.get(b)
    if ra is not None and rb is not None and d <= 1.25 * (ra + rb) + 0.25:
        return "single"
    return "none"


def _infer_bonds(frame: Frame) -> list[dict]:
    syms = list(frame.symbols)
    xyz = np.asarray(frame.coords, dtype=float)
    bonds: list[dict] = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            order = _bond_order_for_pair(str(syms[i]), str(syms[j]), d)
            if order != "none":
                a, b = sorted((str(syms[i]), str(syms[j])))
                bonds.append({"i": i, "j": j, "pair": f"{a}-{b}", "order": order, "distance": d})
    return bonds


def _cycle_rank(n: int, bonds: list[dict]) -> int:
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for b in bonds:
        union(int(b["i"]), int(b["j"]))
    comps = len({find(i) for i in range(n)}) if n else 0
    return max(0, len(bonds) - n + comps)


def _chemical_family_descriptor(frame: Frame, species_id: str) -> dict:
    formula = _formula_from_species(species_id) or _formula_from_frame(frame)
    counts = _parse_formula_counts(formula)
    nC = int(counts.get("C", 0)); nH = int(counts.get("H", 0)); nO = int(counts.get("O", 0))
    bonds = _infer_bonds(frame)
    cc_single = sum(1 for b in bonds if b["pair"] == "C-C" and b["order"] == "single")
    cc_double = sum(1 for b in bonds if b["pair"] == "C-C" and b["order"] == "double")
    cc_triple = sum(1 for b in bonds if b["pair"] == "C-C" and b["order"] == "triple")
    co_single = sum(1 for b in bonds if b["pair"] == "C-O" and b["order"] == "single")
    co_double = sum(1 for b in bonds if b["pair"] == "C-O" and b["order"] == "double")
    co_triple = sum(1 for b in bonds if b["pair"] == "C-O" and b["order"] == "triple")
    oh_single = sum(1 for b in bonds if b["pair"] == "H-O" and b["order"] == "single")
    oo_single = sum(1 for b in bonds if b["pair"] == "O-O" and b["order"] == "single")
    oo_double = sum(1 for b in bonds if b["pair"] == "O-O" and b["order"] == "double")
    h2_bond = sum(1 for b in bonds if b["pair"] == "H-H")
    ring_rank = _cycle_rank(len(frame.symbols), bonds)
    c_degrees: dict[int, int] = {}
    for b in bonds:
        i, j = int(b["i"]), int(b["j"])
        syms = list(frame.symbols)
        if syms[i] == "C" and syms[j] == "C":
            c_degrees[i] = c_degrees.get(i, 0) + 1
            c_degrees[j] = c_degrees.get(j, 0) + 1
    branched = any(v >= 3 for v in c_degrees.values())
    ring = ring_rank > 0
    size = "C0" if nC == 0 else ("C1" if nC == 1 else "C2" if nC == 2 else "C3-C4" if nC <= 4 else "C5-C9" if nC <= 9 else "C10plus")

    if formula in {"H2", "O2", "H2O", "CO", "CO2", "CH4", "C2H2", "C2H4", "C2H6"}:
        family = f"benchmark_{formula}"
    elif nC == 0:
        if formula.startswith("H") and nO == 0:
            family = "C0_hydrogen"
        elif nO > 0 and nH > 0:
            family = "C0_oxygen_hydride"
        else:
            family = "C0_oxygen"
    elif nO > 0:
        tags = []
        if co_triple:
            tags.append("CO_triple")
        if co_double:
            tags.append("carbonyl")
        if co_single:
            tags.append("C-O_single")
        if oh_single:
            tags.append("O-H")
        if oo_single or oo_double:
            tags.append("O-O")
        if not tags:
            tags.append("oxygenated")
        family = f"{size}_" + "+".join(tags[:3])
    else:
        if cc_triple:
            unsat = "alkyne"
        elif cc_double:
            unsat = "alkene_or_cycloalkene"
        else:
            unsat = "alkane_or_saturated"
        topo = "cyclic" if ring else ("branched" if branched else "linear_or_small")
        family = f"{size}_hydrocarbon_{unsat}_{topo}"
    return {
        "formula": formula,
        "family": family,
        "n_C": nC, "n_H": nH, "n_O": nO,
        "n_bonds": len(bonds),
        "cc_single": cc_single, "cc_double": cc_double, "cc_triple": cc_triple,
        "co_single": co_single, "co_double": co_double, "co_triple": co_triple,
        "oh_single": oh_single, "oo_single": oo_single, "oo_double": oo_double,
        "h2_bond": h2_bond,
        "ring_rank": ring_rank,
        "branched_carbon": bool(branched),
    }


def _energy_sort_value(rec: CandidateRecord) -> float:
    # Relative per-species energies are suitable for the heuristic family cap;
    # absolute molecular energies are not comparable across different formulas.
    if rec.e_kcalmol_rel is not None and np.isfinite(rec.e_kcalmol_rel):
        return float(rec.e_kcalmol_rel)
    if rec.e_kcalmol_abs is not None and np.isfinite(rec.e_kcalmol_abs):
        return float(rec.e_kcalmol_abs)
    return float('inf')


def _duplicate_energy_sort_value(rec: CandidateRecord) -> float:
    # Within a duplicate cluster/formula, prefer the actual optimized energy.
    if rec.opt_e_hartree is not None and np.isfinite(rec.opt_e_hartree):
        return float(rec.opt_e_hartree) * 627.5094740631
    if rec.e_kcalmol_abs is not None and np.isfinite(rec.e_kcalmol_abs):
        return float(rec.e_kcalmol_abs)
    if rec.e_kcalmol_rel is not None and np.isfinite(rec.e_kcalmol_rel):
        return float(rec.e_kcalmol_rel)
    return float("inf")


def _curate_final_geo_records(
    records: list[CandidateRecord],
    frames: list[Frame],
    cfg: ExploreCfg,
    run_dir: Path | None = None,
) -> tuple[list[CandidateRecord], list[Frame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply final post-optimization duplicate and chemical-family filters.

    The input records/frames are already the structures eligible for GEO.  This
    function is the final safety gate that makes the one-structure XYZ export and
    trainset.geo consistent and non-stale.
    """
    sel_cfg = cfg.export.selection
    dedup_cfg = getattr(sel_cfg, "post_opt_dedup", None)
    fam_cfg = getattr(sel_cfg, "chemical_family_filter", None)
    dedup_method = str(getattr(dedup_cfg, "method", "molecular_identity"))

    entries: list[dict] = []
    for order, (rec, fr) in enumerate(zip(records, frames)):
        desc = _chemical_family_descriptor(fr, rec.species_id)
        entries.append({
            "order": order,
            "rec": rec,
            "frame": fr,
            "fp": pair_distance_fingerprint(fr),
            # Unknown radii can omit bonds/stereo centers. Keep such structures
            # separate rather than treating an incomplete graph as identity.
            "graph": (molecular_graph(fr, _infer_bonds(fr))
                      if dedup_method == "molecular_identity" and set(fr.symbols).issubset(_COVALENT_RADII)
                      else None),
            **desc,
            "keep_after_dedup": True,
            "duplicate_of": "",
            "dedup_distance": np.nan,
            "dedup_method": dedup_method,
            "keep_after_family": True,
            "final_keep": True,
            "final_reason": "keep: final GEO representative",
        })

    # 1) Post-optimization duplicate removal.
    if dedup_cfg is not None and bool(getattr(dedup_cfg, "enabled", True)):
        tol = float(getattr(dedup_cfg, "fingerprint_tol", 0.03) or 0.03)
        scope = str(getattr(dedup_cfg, "scope", "within_formula") or "within_formula")
        keep_mode = str(getattr(dedup_cfg, "keep", "lowest_energy") or "lowest_energy")
        def scope_key(e: dict) -> str:
            if scope == "global":
                return "__global__"
            if scope == "within_family":
                return str(e["family"])
            return str(e["formula"])
        grouped: dict[str, list[dict]] = {}
        for e in entries:
            grouped.setdefault(scope_key(e), []).append(e)
        for _, group in grouped.items():
            ordered = sorted(group, key=lambda e: (_duplicate_energy_sort_value(e["rec"]), e["order"])) if keep_mode == "lowest_energy" else sorted(group, key=lambda e: e["order"])
            kept_group: list[dict] = []
            for e in ordered:
                dup = None; dup_dist = float("inf")
                for old in kept_group:
                    if e['formula'] != old['formula']:
                        continue
                    if (e['rec'].charge, e['rec'].multiplicity) != (old['rec'].charge, old['rec'].multiplicity):
                        continue
                    d = fingerprint_distance(e["fp"], old["fp"])
                    duplicate = (same_molecular_identity(e["graph"], old["graph"])
                                 if dedup_method == "molecular_identity" else d <= tol)
                    if duplicate:
                        dup = old; dup_dist = d; break
                if dup is not None:
                    e["keep_after_dedup"] = False
                    e["duplicate_of"] = dup["rec"].id
                    e["dedup_distance"] = dup_dist
                    e["final_keep"] = False
                    match_kind = "molecular identity" if dedup_method == "molecular_identity" else "geometry"
                    e["final_reason"] = f"drop: post-opt duplicate of {dup['rec'].id} ({match_kind}; fp_distance={dup_dist:.5f})"
                else:
                    kept_group.append(e)

    # 2) Optional per-formula cap, after true duplicate removal.  Protected formulas
    # are not capped here because they are usually small reference molecules.
    if fam_cfg is not None:
        max_per_formula = getattr(fam_cfg, "max_per_formula", None)
        protected_formulas = {str(x) for x in list(getattr(fam_cfg, "protected_formulas", []) or [])}
        if max_per_formula is not None:
            try:
                max_pf = int(max_per_formula)
            except Exception:
                max_pf = 0
            if max_pf > 0:
                groups: dict[str, list[dict]] = {}
                for e in entries:
                    if e["final_keep"] and str(e["formula"]) not in protected_formulas:
                        groups.setdefault(str(e["formula"]), []).append(e)
                for formula, group in groups.items():
                    ordered = sorted(group, key=lambda e: (_energy_sort_value(e["rec"]), e["order"]))
                    for e in ordered[max_pf:]:
                        e["keep_after_family"] = False
                        e["final_keep"] = False
                        e["final_reason"] = f"drop: max_per_formula={max_pf} exceeded for {formula}"

    # 3) Chemical-family representative filter.  This is intentionally less strict
    # than formula-level filtering: different formulas/isomers may remain if they
    # are representative or protected.
    if fam_cfg is not None and bool(getattr(fam_cfg, "enabled", False)) and str(getattr(fam_cfg, "stage", "post_opt")) != "off":
        try:
            max_per_family = max(1, int(getattr(fam_cfg, "max_per_family", 3) or 3))
        except Exception:
            max_per_family = 3
        protected_formulas = {str(x) for x in list(getattr(fam_cfg, "protected_formulas", []) or [])}
        protected_families = {str(x) for x in list(getattr(fam_cfg, "protected_families", []) or [])}
        groups: dict[str, list[dict]] = {}
        for e in entries:
            if e["final_keep"]:
                groups.setdefault(str(e["family"]), []).append(e)
        for family, group in groups.items():
            # Protected small benchmark molecules are kept even if the family cap
            # would otherwise remove them.
            nonprotected = [e for e in group if str(e["formula"]) not in protected_formulas and family not in protected_families]
            ordered = sorted(nonprotected, key=lambda e: (_energy_sort_value(e["rec"]), e["order"]))
            for e in ordered[max_per_family:]:
                e["keep_after_family"] = False
                e["final_keep"] = False
                e["final_reason"] = f"drop: chemical-family quota exceeded ({family}, max_per_family={max_per_family})"

    final_entries = sorted([e for e in entries if e["final_keep"]], key=lambda e: e["order"])
    final_records = [e["rec"] for e in final_entries]
    final_frames = [e["frame"] for e in final_entries]

    decision_rows = []
    for e in entries:
        rec = e["rec"]
        row = _row(rec, bool(e["final_keep"]), str(e["final_reason"]))
        row.update({
            "formula": e["formula"], "chemical_family": e["family"],
            "n_C": e["n_C"], "n_H": e["n_H"], "n_O": e["n_O"],
            "cc_single": e["cc_single"], "cc_double": e["cc_double"], "cc_triple": e["cc_triple"],
            "co_single": e["co_single"], "co_double": e["co_double"], "co_triple": e["co_triple"],
            "oh_single": e["oh_single"], "oo_single": e["oo_single"], "oo_double": e["oo_double"],
            "ring_rank": e["ring_rank"], "branched_carbon": e["branched_carbon"],
            "keep_after_dedup": bool(e["keep_after_dedup"]),
            "duplicate_of": e["duplicate_of"],
            "dedup_distance": e["dedup_distance"],
            "dedup_method": e["dedup_method"],
            "identity_match_available": e["graph"] is not None,
            "keep_after_family": bool(e["keep_after_family"]),
            "final_keep_for_geo": bool(e["final_keep"]),
        })
        decision_rows.append(row)
    decisions = pd.DataFrame(decision_rows)
    dup_df = decisions.loc[decisions.get("duplicate_of", "") != ""].copy() if not decisions.empty else pd.DataFrame()
    if not decisions.empty:
        kept_dec = decisions.loc[decisions["final_keep_for_geo"].astype(bool)].copy()
        fam_summary = kept_dec.groupby("chemical_family").agg(
            n_structures=("id", "count"),
            n_formulas=("formula", "nunique"),
            formulas=("formula", lambda x: ";".join(sorted(set(map(str, x))))),
        ).reset_index().sort_values(["n_structures", "chemical_family"], ascending=[False, True])
    else:
        fam_summary = pd.DataFrame(columns=["chemical_family", "n_structures", "n_formulas", "formulas"])
    return final_records, final_frames, decisions, dup_df, fam_summary


def _geo_index_row(rec: CandidateRecord, frame: Frame, optimized: bool, reason: str = "geo_source") -> dict:
    row = _row(rec, True, reason)
    row.update(_chemical_family_descriptor(frame, rec.species_id))
    row["optimized_geometry"] = bool(optimized)
    return row


@dataclass
class CandidateRecord:
    id: str
    species_id: str
    iter: int
    is_seed: bool
    e_hartree: float | None
    e_kcalmol_abs: float | None
    e_kcalmol_rel: float | None
    charge: int | None
    multiplicity: int | None
    job_dir: Path
    source_xyz: Path
    frame: Frame
    fingerprint: tuple
    opt_e_hartree: float | None = None


def _resolve_job_dir(run_dir: Path, job_dir_raw: str, iid: str) -> Path | None:
    candidates: list[Path] = []
    if job_dir_raw:
        p = Path(str(job_dir_raw))
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(run_dir / p)
            candidates.append(run_dir.parent / p)
    if iid:
        candidates.extend(sorted(run_dir.glob(f"iter_*/gaussian/jobs/job_{iid}")))
        candidates.extend(sorted(run_dir.glob(f"iter_*/gaussian/jobs/job_*{iid}*")))
    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
        except Exception:
            pass
    return None



_ATOMIC_NUMBERS_SIMPLE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Br": 35, "I": 53,
}


def _parse_formula_counts(species_id: str) -> dict[str, int]:
    formula = str(species_id).split("_", 1)[0]
    counts: dict[str, int] = {}
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return counts


def _electron_count_from_species(species_id: str, charge: int | None = 0) -> int | None:
    counts = _parse_formula_counts(species_id)
    if not counts:
        return None
    total = 0
    for sym, n in counts.items():
        z = _ATOMIC_NUMBERS_SIMPLE.get(sym)
        if z is None:
            return None
        total += z * int(n)
    try:
        total -= int(charge or 0)
    except Exception:
        pass
    return total


def _looks_open_shell(rec: "CandidateRecord") -> bool:
    # Prefer the actual Gaussian multiplicity label.  Multiplicity > 1 is the
    # strongest indicator that the candidate is open-shell/radical-like.
    try:
        if rec.multiplicity is not None and int(rec.multiplicity) != 1:
            return True
    except Exception:
        pass
    # Fallback for older labels: odd electron count under neutral charge is
    # necessarily open-shell.
    ne = _electron_count_from_species(rec.species_id, rec.charge)
    if ne is not None and ne % 2 == 1:
        return True
    return False


def _formula_from_species(species_id: str) -> str:
    return str(species_id).split("_", 1)[0]


def _composition_filter_reason(rec: "CandidateRecord", cfg: ExploreCfg) -> Optional[str]:
    """Return a drop reason if user-defined formula/element filters reject rec."""
    sel_cfg = cfg.export.selection
    formula = _formula_from_species(rec.species_id)
    counts = _parse_formula_counts(rec.species_id)

    include_pats = list(getattr(sel_cfg, "include_formula_regex", []) or [])
    if include_pats:
        matched = False
        for pat in include_pats:
            try:
                if re.search(str(pat), formula):
                    matched = True
                    break
            except re.error:
                continue
        if not matched:
            return f"drop: formula {formula} does not match include_formula_regex"

    for pat in list(getattr(sel_cfg, "exclude_formula_regex", []) or []):
        try:
            if re.search(str(pat), formula):
                return f"drop: formula {formula} matches exclude_formula_regex={pat!r}"
        except re.error:
            continue

    allowed = getattr(sel_cfg, "allowed_elements", None)
    if allowed is not None:
        allowed_set = {str(x) for x in allowed}
        extra = sorted(set(counts) - allowed_set)
        if extra:
            return f"drop: elements {extra} not in allowed_elements={sorted(allowed_set)}"

    required = {str(x) for x in list(getattr(sel_cfg, "required_elements", []) or [])}
    missing = sorted([el for el in required if int(counts.get(el, 0)) <= 0])
    if missing:
        return f"drop: missing required_elements={missing}"

    excluded = {str(x) for x in list(getattr(sel_cfg, "exclude_elements", []) or [])}
    present_excluded = sorted([el for el in excluded if int(counts.get(el, 0)) > 0])
    if present_excluded:
        return f"drop: contains excluded_elements={present_excluded}"

    for el, mn in dict(getattr(sel_cfg, "min_element_counts", {}) or {}).items():
        try:
            if int(counts.get(str(el), 0)) < int(mn):
                return f"drop: {el} count {int(counts.get(str(el), 0))} < min_element_counts[{el}]={int(mn)}"
        except Exception:
            continue

    for el, mx in dict(getattr(sel_cfg, "max_element_counts", {}) or {}).items():
        try:
            if int(counts.get(str(el), 0)) > int(mx):
                return f"drop: {el} count {int(counts.get(str(el), 0))} > max_element_counts[{el}]={int(mx)}"
        except Exception:
            continue

    return None


def _selection_filter_reason(rec: "CandidateRecord", cfg: ExploreCfg) -> Optional[str]:
    sel_cfg = cfg.export.selection
    comp_reason = _composition_filter_reason(rec, cfg)
    if comp_reason:
        return comp_reason
    allowed = getattr(sel_cfg, "allowed_multiplicities", None)
    if allowed is not None:
        try:
            mult = int(rec.multiplicity) if rec.multiplicity is not None else None
        except Exception:
            mult = None
        allowed_int = {int(x) for x in allowed}
        if mult not in allowed_int:
            return f"drop: multiplicity {mult} not in allowed_multiplicities={sorted(allowed_int)}"
    if bool(getattr(sel_cfg, "exclude_open_shell", False)) and _looks_open_shell(rec):
        return "drop: open-shell/radical excluded from trainset/GEO export"
    for pat in list(getattr(sel_cfg, "exclude_species_regex", []) or []):
        try:
            if re.search(str(pat), rec.species_id):
                return f"drop: species_id matches exclude_species_regex={pat!r}"
        except re.error:
            continue
    return None

def _load_labeled_candidates(run_dir: Path, cfg: ExploreCfg) -> tuple[list[CandidateRecord], dict[str, int]]:
    labels_csv = run_dir / "labels" / "labels.csv"
    if not labels_csv.exists():
        return [], {"no_labels_csv": 1}
    df = pd.read_csv(labels_csv)
    if df.empty:
        return [], {"labels_empty": 1}

    # Seed map from current pool index, if available.
    seed_map: dict[str, bool] = {}
    pool_csv = run_dir / "pool" / "pool_index.csv"
    if pool_csv.exists():
        try:
            pdf = pd.read_csv(pool_csv)
            if "id" in pdf.columns:
                seed_map = dict(zip(pdf["id"].astype(str), pdf.get("is_seed", pd.Series([False] * len(pdf))).astype(bool)))
        except Exception:
            seed_map = {}

    status_ok = {"ok", "labeled", "success", "done"}
    records: list[CandidateRecord] = []
    skipped = {"not_ok": 0, "seed": 0, "missing_job_dir": 0, "missing_structure": 0, "parse_xyz_failed": 0}
    source_policy = str(cfg.export.source).strip()

    for _, r in df.iterrows():
        st = str(r.get("status", "ok")).lower()
        if st and st not in status_ok:
            skipped["not_ok"] += 1
            continue
        iid = str(r.get("id", ""))
        if not iid:
            continue
        is_seed = bool(seed_map.get(iid, False))
        if source_policy == "non_seed_labeled" and is_seed:
            skipped["seed"] += 1
            continue
        job_dir = _resolve_job_dir(run_dir, str(r.get("job_dir", "") or ""), iid)
        if job_dir is None:
            skipped["missing_job_dir"] += 1
            continue
        xyz = job_dir / "structure.xyz"
        if not xyz.exists():
            skipped["missing_structure"] += 1
            continue
        try:
            fr = read_single_xyz(xyz)
        except Exception:
            skipped["parse_xyz_failed"] += 1
            continue
        def _float_or_none(x):
            try:
                if pd.isna(x):
                    return None
                return float(x)
            except Exception:
                return None
        def _int_or_none(x):
            try:
                if pd.isna(x):
                    return None
                return int(x)
            except Exception:
                return None
        records.append(CandidateRecord(
            id=iid,
            species_id=str(r.get("species_id", "unknown")),
            iter=int(r.get("iter", -1)) if str(r.get("iter", "")).strip() != "" else -1,
            is_seed=is_seed,
            e_hartree=_float_or_none(r.get("e_hartree", None)),
            e_kcalmol_abs=_float_or_none(r.get("e_kcalmol_abs", None)),
            e_kcalmol_rel=_float_or_none(r.get("e_kcalmol_rel", None)),
            charge=_int_or_none(r.get("charge", None)),
            multiplicity=_int_or_none(r.get("multiplicity", None)),
            job_dir=job_dir,
            source_xyz=xyz,
            frame=fr,
            fingerprint=pair_distance_fingerprint(fr),
        ))
    return records, skipped


def _select_representatives(records: list[CandidateRecord], cfg: ExploreCfg) -> tuple[list[CandidateRecord], pd.DataFrame]:
    sel_cfg = cfg.export.selection
    max_per_species = int(sel_cfg.max_per_species)
    max_total = int(sel_cfg.max_total) if sel_cfg.max_total is not None else None
    fp_tol = float(sel_cfg.fingerprint_tol)
    max_rel = sel_cfg.max_rel_kcalmol
    bins = max(1, int(sel_cfg.energy_bins))

    rows: list[dict] = []
    kept: list[CandidateRecord] = []

    # User-defined chemistry filters and energy sanity filter first.
    filtered: list[CandidateRecord] = []
    for rec in records:
        filter_reason = _selection_filter_reason(rec, cfg)
        if filter_reason:
            rows.append(_row(rec, keep=False, reason=filter_reason))
            continue
        if max_rel is not None and rec.e_kcalmol_rel is not None and rec.e_kcalmol_rel > float(max_rel):
            rows.append(_row(rec, keep=False, reason=f"drop: e_kcalmol_rel>{float(max_rel):.3f}"))
            continue
        filtered.append(rec)

    by_species: dict[str, list[CandidateRecord]] = {}
    for rec in filtered:
        by_species.setdefault(rec.species_id, []).append(rec)

    for sp, group in sorted(by_species.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        # Prefer low/medium/high energy bins, then diversity inside each bin.
        group = sorted(group, key=lambda r: (math.inf if r.e_kcalmol_rel is None else r.e_kcalmol_rel, r.id))
        if len(group) == 0:
            continue
        energies = np.array([0.0 if r.e_kcalmol_rel is None else float(r.e_kcalmol_rel) for r in group], dtype=float)
        if bins <= 1 or len(group) <= bins:
            bin_ids = np.zeros(len(group), dtype=int)
        else:
            qs = np.quantile(energies, np.linspace(0, 1, bins + 1))
            # Make internal cutpoints unique enough for digitize.
            cuts = qs[1:-1]
            bin_ids = np.digitize(energies, cuts, right=True)

        chosen_for_sp: list[CandidateRecord] = []
        # Round-robin over bins to avoid selecting only low-energy conformers.
        while len(chosen_for_sp) < min(max_per_species, len(group)):
            progressed = False
            for b in range(bins):
                if len(chosen_for_sp) >= min(max_per_species, len(group)):
                    break
                cand = [r for r, bid in zip(group, bin_ids) if int(bid) == b and r not in chosen_for_sp]
                for rec in cand:
                    if all(fingerprint_distance(rec.fingerprint, old.fingerprint) > fp_tol for old in chosen_for_sp):
                        chosen_for_sp.append(rec)
                        progressed = True
                        break
            if not progressed:
                break
        chosen_ids = {r.id for r in chosen_for_sp}
        for rec in group:
            if rec.id in chosen_ids:
                rows.append(_row(rec, keep=True, reason="keep: representative"))
                kept.append(rec)
            else:
                reason = "drop: duplicate/similar within species or species quota exceeded"
                rows.append(_row(rec, keep=False, reason=reason))

    # Apply global cap after species balancing.
    if max_total is not None and len(kept) > max_total:
        # Keep diverse species order; stable low-energy first within current kept ordering.
        keep_ids = {r.id for r in kept[:max_total]}
        new_rows = []
        for row in rows:
            if row.get("keep_for_trainset") and row["id"] not in keep_ids:
                row = dict(row)
                row["keep_for_trainset"] = False
                row["reason"] = "drop: global max_total exceeded"
            new_rows.append(row)
        rows = new_rows
        kept = [r for r in kept if r.id in keep_ids]

    return kept, pd.DataFrame(rows)


def _row(rec: CandidateRecord, keep: bool, reason: str) -> dict:
    return {
        "id": rec.id,
        "species_id": rec.species_id,
        "iter": rec.iter,
        "is_seed": rec.is_seed,
        "e_hartree": rec.e_hartree,
        "opt_e_hartree": rec.opt_e_hartree,
        "e_kcalmol_abs": rec.e_kcalmol_abs,
        "e_kcalmol_rel": rec.e_kcalmol_rel,
        "charge": rec.charge,
        "multiplicity": rec.multiplicity,
        "open_shell_like": _looks_open_shell(rec),
        "natoms": len(rec.frame.symbols),
        "keep_for_trainset": bool(keep),
        "reason": reason,
        "source_xyz": str(rec.source_xyz),
        "job_dir": str(rec.job_dir),
    }


def _write_representative_xyz(run_dir: Path, records: list[CandidateRecord], table: pd.DataFrame) -> None:
    out_dir = run_dir / "exports" / "trainset_candidates"
    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    for old in xyz_dir.glob("*.xyz"):
        old.unlink()

    rows = []
    frames = []
    for rec in records:
        name = f"{_safe_name(rec.species_id)}__{rec.id}.xyz"
        dst = xyz_dir / name
        shutil.copyfile(rec.source_xyz, dst)
        frames.append(rec.frame)
        rows.append({"id": rec.id, "xyz_file": str(dst.relative_to(run_dir))})
    if frames:
        write_xyz_multiframe(out_dir / "trainset_candidates.xyz", frames)
    else:
        (out_dir / "trainset_candidates.xyz").write_text("", encoding="utf-8")

    file_map = {r["id"]: r["xyz_file"] for r in rows}
    if not table.empty:
        table = table.copy()
        table["candidate_xyz_file"] = table["id"].astype(str).map(file_map).fillna("")
        table.to_csv(out_dir / "trainset_candidates.csv", index=False)
    else:
        pd.DataFrame(columns=["id", "species_id", "keep_for_trainset", "reason", "candidate_xyz_file"]).to_csv(out_dir / "trainset_candidates.csv", index=False)

    (out_dir / "README.txt").write_text(
        "TSGen Explore trainset candidate export.\n"
        f"Representative structures kept: {len(records)}\n"
        "These are DFT-labeled, de-duplicated candidate structures selected from labels.csv.\n"
        "They are not necessarily geometry-optimized unless optimization/geo export is enabled and completed.\n",
        encoding="utf-8",
    )


def _write_single_xyz(path: Path, frame: Frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(frame.symbols)}\n")
        f.write((frame.comment or "").strip() + "\n")
        for sym, xyz in zip(frame.symbols, np.asarray(frame.coords, dtype=float)):
            f.write(f"{sym:2s}  {float(xyz[0]): .10f}  {float(xyz[1]): .10f}  {float(xyz[2]): .10f}\n")


def _gaussian_log_status(log_path: Path) -> str:
    """Return ok|failed|pending based only on the current input.log content."""
    if not log_path.exists():
        return "pending"
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return "pending"
    if "Normal termination" in text:
        return "ok"
    if "Error termination" in text:
        return "failed"
    return "pending"


def _resolve_opt_job_dir(run_dir: Path, opt_root: Path, rec: CandidateRecord, existing_map: dict[str, str]) -> Path:
    """Resolve optimization job dir robustly.

    Earlier versions wrote opt_job_dir as a relative path such as
    outputs_explore/explore_run_xxxx/exports/..., which is valid only from the
    original working directory.  On resume from a different cwd, rebuilding the
    deterministic path from run_dir is safer.  We still try the recorded path as
    a compatibility fallback.
    """
    candidates = [opt_root / f"opt_{rec.id}"]
    raw = existing_map.get(rec.id, "")
    if raw:
        p = Path(raw)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(run_dir / raw)
            candidates.append(run_dir.parent / raw)
            # If raw already starts with the run directory name, run_dir.parent/raw is usually correct.
    candidates.extend(sorted(opt_root.glob(f"opt_*{rec.id}*")))
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                return c
        except Exception:
            continue
    return opt_root / f"opt_{rec.id}"


def _prepare_or_collect_opt_jobs(run_dir: Path, records: list[CandidateRecord], cfg: ExploreCfg) -> tuple[list[dict], list[tuple[CandidateRecord, Frame]]]:
    """Prepare/submit optimization jobs and collect completed optimized geometries.

    This function is deliberately idempotent:
      - first call: writes/submits opt jobs;
      - later resume: re-scans input.log and updates opt_jobs.csv;
      - it never trusts the previous opt_jobs.csv status as the source of truth.
    """
    opt_cfg = cfg.export.optimization
    if not bool(opt_cfg.enabled):
        return [], []

    opt_root = run_dir / "exports" / "trainset_candidates" / "opt_jobs"
    opt_root.mkdir(parents=True, exist_ok=True)

    # Existing manifest is used only for jobid/path preservation, not for status.
    existing_jobid: dict[str, str] = {}
    existing_dir: dict[str, str] = {}
    manifest = run_dir / "exports" / "trainset_candidates" / "opt_jobs.csv"
    if manifest.exists():
        try:
            old = pd.read_csv(manifest)
            if "id" in old.columns:
                if "opt_jobid" in old.columns:
                    existing_jobid = dict(zip(old["id"].astype(str), old["opt_jobid"].fillna("").astype(str)))
                if "opt_job_dir" in old.columns:
                    existing_dir = dict(zip(old["id"].astype(str), old["opt_job_dir"].fillna("").astype(str)))
        except Exception:
            pass

    # Clone Gaussian config but with optimization-specific route/mode if provided.
    gcfg = cfg.labeling.gaussian.model_copy(deep=True)
    if opt_cfg.route:
        gcfg.route = str(opt_cfg.route)
    if opt_cfg.mode is not None:
        gcfg.mode = opt_cfg.mode
    oracle = GaussianOracle(gcfg)

    rows: list[dict] = []
    opt_pairs: list[tuple[CandidateRecord, Frame]] = []
    for rec in records:
        job_dir = _resolve_opt_job_dir(run_dir, opt_root, rec, existing_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Write input exactly once if absent.
        if not (job_dir / "input.gjf").exists():
            oracle.write_job(job_dir, GaussianJobSpec(id=rec.id, species_id=rec.species_id, frame=rec.frame))

        log = job_dir / "input.log"
        status = _gaussian_log_status(log)
        jobid = existing_jobid.get(rec.id, "")

        # Submit only if there is no completed/failed log and the job was not already submitted.
        if bool(opt_cfg.submit) and gcfg.mode == "slurm" and status == "pending" and not (job_dir / "sbatch.stdout").exists():
            try:
                jid = oracle.submit(job_dir)
                jobid = jid or jobid
            except Exception:
                pass

        # Recompute after possible submission; do not rely on stale opt_jobs.csv.
        status = _gaussian_log_status(log)
        opt_frame = None
        opt_energy = None
        parse_note = ""
        if status == "ok" and log.exists():
            try:
                text = log.read_text(errors="ignore")
                opt_energy = _parse_scf_energy_hartree(text)
            except Exception:
                opt_energy = None
            opt_frame = parse_gaussian_optimized_frame(log, fallback_symbols=rec.frame.symbols)
            if opt_frame is not None:
                # Frame is frozen in tsgen.data.xyz, so do not mutate comment in place.
                # Construct a new Frame carrying the optimized coordinates and provenance.
                opt_frame = Frame(
                    symbols=list(opt_frame.symbols),
                    coords=np.asarray(opt_frame.coords, dtype=float),
                    comment=f"id={rec.id} species={rec.species_id} optimized=true E_hartree={opt_energy or ''}",
                )
                _write_single_xyz(job_dir / "optimized_structure.xyz", opt_frame)
                opt_pairs.append((replace(rec, opt_e_hartree=opt_energy), opt_frame))
            else:
                parse_note = "normal_termination_but_no_orientation_parsed"

        rows.append({
            "id": rec.id,
            "species_id": rec.species_id,
            "opt_job_dir": str(job_dir),
            "opt_status": status,
            "opt_jobid": jobid,
            "opt_e_hartree": opt_energy,
            "has_optimized_geometry": opt_frame is not None,
            "parse_note": parse_note,
            "log_path": str(log) if log.exists() else "",
            "optimized_xyz": str(job_dir / "optimized_structure.xyz") if opt_frame is not None else "",
        })

    pd.DataFrame(rows).to_csv(run_dir / "exports" / "trainset_candidates" / "opt_jobs.csv", index=False)
    return rows, opt_pairs


def _opt_status_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"ok": 0, "failed": 0, "pending": 0, "with_geometry": 0, "total": len(rows)}
    for row in rows:
        st = str(row.get("opt_status", "pending") or "pending").lower()
        if st not in {"ok", "failed", "pending"}:
            st = "pending"
        counts[st] += 1
        if bool(row.get("has_optimized_geometry", False)):
            counts["with_geometry"] += 1
    return counts


def _wait_for_opt_jobs(run_dir: Path, records: list[CandidateRecord], cfg: ExploreCfg, logger=None) -> tuple[list[dict], list[tuple[CandidateRecord, Frame]]]:
    """Prepare/collect optimization jobs, optionally blocking until logs finish.

    This provides a one-shot workflow for users who prefer not to check squeue
    and manually resume.  It polls Gaussian input.log files, not Slurm itself,
    so it works across slurm/local/manual environments as long as logs appear in
    the job directories.
    """
    rows, pairs = _prepare_or_collect_opt_jobs(run_dir, records, cfg)
    opt_cfg = cfg.export.optimization
    if not bool(opt_cfg.enabled) or not bool(getattr(opt_cfg, "wait_for_completion", False)):
        return rows, pairs
    if not records:
        return rows, pairs

    poll = max(5, int(getattr(opt_cfg, "poll_seconds", 60) or 60))
    max_wait = max(0, int(getattr(opt_cfg, "max_wait_seconds", 86400) or 0))
    require_all_ok = bool(getattr(opt_cfg, "require_all_ok", True))
    deadline = time.time() + max_wait if max_wait > 0 else None

    while True:
        counts = _opt_status_counts(rows)
        total = counts["total"] or len(records)
        msg = (
            f"Optimization wait: ok={counts['ok']}, failed={counts['failed']}, "
            f"pending={counts['pending']}, optimized_geometry={counts['with_geometry']}/{total}"
        )
        if logger is not None:
            try:
                logger.info(msg)
            except Exception:
                print(f"[TSGen] {msg}", flush=True)
        else:
            print(f"[TSGen] {msg}", flush=True)

        if counts["pending"] == 0:
            if (not require_all_ok) or counts["failed"] == 0:
                return rows, pairs
            return rows, pairs
        if deadline is not None and time.time() >= deadline:
            if logger is not None:
                try:
                    logger.warning("Optimization wait timed out; run resume later to collect remaining opt jobs.")
                except Exception:
                    pass
            return rows, pairs
        time.sleep(poll)
        rows, pairs = _prepare_or_collect_opt_jobs(run_dir, records, cfg)


def _write_centralized_optimized_xyz(run_dir: Path, frames: list[Frame], records: list[CandidateRecord], optimized: bool) -> None:
    """Write one xyz per final GEO structure into a user-friendly directory."""
    out = run_dir / "exports" / ("optimized_xyz" if optimized else "geo_source_xyz")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.xyz"):
        try:
            old.unlink()
        except Exception:
            pass
    rows = []
    for rec, fr in zip(records, frames):
        name = f"{_safe_name(rec.species_id)}__{rec.id}.xyz"
        dst = out / name
        _write_single_xyz(dst, fr)
        rows.append({
            "id": rec.id,
            "species_id": rec.species_id,
            "optimized": bool(optimized),
            "xyz_file": str(dst.relative_to(run_dir)),
            "e_hartree": rec.e_hartree,
            "opt_e_hartree": rec.opt_e_hartree,
            "e_kcalmol_rel": rec.e_kcalmol_rel,
            "job_dir": str(rec.job_dir),
        })
    pd.DataFrame(rows, columns=['id', 'species_id', 'optimized', 'xyz_file', 'e_hartree',
                               'opt_e_hartree', 'e_kcalmol_rel', 'job_dir']).to_csv(out / "index.csv", index=False)


def _safe_copy(src: Path, dst: Path) -> None:
    try:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    except Exception:
        pass


def _write_analysis_products(
    run_dir: Path,
    records: list[CandidateRecord],
    kept: list[CandidateRecord],
    candidate_table: pd.DataFrame,
    opt_rows: list[dict],
    geo_records: list[CandidateRecord] | None = None,
    frames_written: int = 0,
    logger=None,
) -> None:
    """Write paper-friendly CSV tables, basic PNG figures, and a run report."""
    analysis = run_dir / "exports" / "analysis"
    tables = analysis / "tables"
    figs = analysis / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # 1) Pool species composition.
    pool_csv = run_dir / "pool" / "pool_index.csv"
    pool_count = 0
    if pool_csv.exists():
        try:
            pool_df = pd.read_csv(pool_csv)
            pool_count = len(pool_df)
            if "species_id" in pool_df.columns:
                pool_species = pool_df["species_id"].astype(str).value_counts().rename_axis("species_id").reset_index(name="n_pool")
                pool_species["formula"] = pool_species["species_id"].astype(str).map(_formula_from_species)
                pool_species.to_csv(tables / "pool_species_counts.csv", index=False)
                pool_formula = pool_df["species_id"].astype(str).map(_formula_from_species).value_counts().rename_axis("formula").reset_index(name="n_pool")
                pool_formula.to_csv(tables / "pool_formula_counts.csv", index=False)
                cc_rows = []
                for sid in pool_df["species_id"].astype(str):
                    d = _parse_formula_counts(sid)
                    cc_rows.append({"n_C": int(d.get("C", 0)), "n_O": int(d.get("O", 0))})
                if cc_rows:
                    cdf = pd.DataFrame(cc_rows)
                    cdist = cdf.groupby(["n_C", "n_O"]).size().reset_index(name="n_pool")
                    cdist.to_csv(tables / "pool_carbon_oxygen_distribution.csv", index=False)
        except Exception:
            pass

    # 2) Labeled data and energy distribution.
    labels_csv = run_dir / "labels" / "labels.csv"
    labels_count = 0
    if labels_csv.exists():
        try:
            lab = pd.read_csv(labels_csv)
            labels_count = len(lab)
            lab.to_csv(tables / "labels_for_plotting.csv", index=False)
            if "species_id" in lab.columns:
                agg = lab.groupby("species_id").agg(
                    n_labeled=("id", "count"),
                    min_e_rel=("e_kcalmol_rel", "min"),
                    median_e_rel=("e_kcalmol_rel", "median"),
                    max_e_rel=("e_kcalmol_rel", "max"),
                ).reset_index()
                agg.to_csv(tables / "labeled_species_energy_summary.csv", index=False)
            if "iter" in lab.columns:
                by_iter = lab.groupby("iter").agg(
                    n_labeled=("id", "count"),
                    n_species=("species_id", "nunique"),
                ).reset_index()
                by_iter["cumulative_labeled"] = by_iter["n_labeled"].cumsum()
                by_iter.to_csv(tables / "labeled_by_iteration.csv", index=False)
            if "species_id" in lab.columns:
                fam_rows = []
                for _, rr in lab.iterrows():
                    sid = str(rr.get("species_id", ""))
                    d = _parse_formula_counts(sid)
                    fam_rows.append({"id": rr.get("id", ""), "species_id": sid, "formula": _formula_from_species(sid), "n_C": int(d.get("C", 0)), "n_O": int(d.get("O", 0))})
                if fam_rows:
                    lf = pd.DataFrame(fam_rows)
                    lf.to_csv(tables / "labeled_formula_metadata.csv", index=False)
                    lf.groupby(["n_C", "n_O"]).size().reset_index(name="n_labeled").to_csv(tables / "labeled_carbon_oxygen_distribution.csv", index=False)
        except Exception:
            lab = pd.DataFrame()
    else:
        lab = pd.DataFrame()

    # 3) Trainset candidate tables.
    if not candidate_table.empty:
        candidate_table.to_csv(tables / "trainset_candidate_decisions.csv", index=False)
    if kept:
        kept_df = pd.DataFrame([_row(r, True, "kept") for r in kept])
        kept_df.to_csv(tables / "trainset_kept_structures.csv", index=False)
        comp_rows = []
        for r in kept:
            d = _parse_formula_counts(r.species_id)
            comp_rows.append({"id": r.id, "species_id": r.species_id, **{f"n_{k}": v for k, v in sorted(d.items())}})
        pd.DataFrame(comp_rows).to_csv(tables / "trainset_composition_counts.csv", index=False)

    # 4) Final GEO curation tables produced by post-optimization filtering.
    geo_dir = run_dir / "exports" / "reaxff_geo"
    for fname in ["geo_index.csv", "postopt_curation_decisions.csv", "postopt_duplicate_clusters.csv", "geo_family_summary.csv"]:
        src = geo_dir / fname
        if src.exists():
            try:
                pd.read_csv(src).to_csv(tables / fname, index=False)
            except Exception:
                pass
    # Decision reason counts are useful as selection-funnel plotting data.
    try:
        dec = pd.read_csv(geo_dir / "postopt_curation_decisions.csv")
        if not dec.empty and "final_reason" in dec.columns:
            dec["decision"] = dec["final_keep_for_geo"].map(lambda x: "keep" if bool(x) else "drop")
            dec.groupby(["decision", "final_reason"]).size().reset_index(name="count").to_csv(tables / "postopt_decision_reason_counts.csv", index=False)
    except Exception:
        pass
    if not candidate_table.empty and "reason" in candidate_table.columns:
        try:
            candidate_table.groupby(["keep_for_trainset", "reason"]).size().reset_index(name="count").to_csv(tables / "trainset_decision_reason_counts.csv", index=False)
        except Exception:
            pass

    # 5) Funnel summary.
    opt_counts = _opt_status_counts(opt_rows)
    geo_count = len(geo_records or []) if geo_records is not None else int(frames_written or 0)
    funnel = pd.DataFrame([
        {"stage": "pool", "count": pool_count},
        {"stage": "labeled_ok", "count": labels_count},
        {"stage": "candidate_records_loaded", "count": len(records)},
        {"stage": "trainset_representatives", "count": len(kept)},
        {"stage": "opt_jobs_ok", "count": opt_counts.get("ok", 0)},
        {"stage": "optimized_geometries", "count": opt_counts.get("with_geometry", 0)},
        {"stage": "geo_structures", "count": geo_count},
    ])
    funnel.to_csv(tables / "selection_funnel.csv", index=False)

    # 5) Plots.  Best-effort; CSVs are the authoritative plotting data.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        funnel_csv = tables / "selection_funnel.csv"
        if funnel_csv.exists():
            fdf = pd.read_csv(funnel_csv)
            if not fdf.empty and {"stage", "count"}.issubset(fdf.columns):
                fig, ax = plt.subplots(figsize=(7.2, max(3.4, 0.30 * len(fdf))))
                ax.barh(fdf["stage"].astype(str), fdf["count"].astype(float))
                ax.set_xlabel("Count")
                ax.set_ylabel("Workflow stage")
                ax.set_title("Active-learning to GEO selection funnel")
                fig.tight_layout()
                fig.savefig(figs / "selection_funnel.png", dpi=300)
                plt.close(fig)

        if labels_csv.exists() and not lab.empty and "species_id" in lab.columns:
            top = lab["species_id"].astype(str).value_counts().head(15)
            fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.28 * len(top))))
            top.sort_values().plot(kind="barh", ax=ax)
            ax.set_xlabel("Number of DFT-labeled structures")
            ax.set_ylabel("Species")
            ax.set_title("Top labeled species")
            fig.tight_layout()
            fig.savefig(figs / "labeled_species_top15.png", dpi=300)
            plt.close(fig)

        if labels_csv.exists() and not lab.empty and "iter" in lab.columns:
            by_iter = lab.groupby("iter").size().reset_index(name="n_labeled")
            by_iter["cumulative_labeled"] = by_iter["n_labeled"].cumsum()
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(by_iter["iter"], by_iter["cumulative_labeled"], marker="o")
            ax.set_xlabel("Active-learning iteration")
            ax.set_ylabel("Cumulative DFT labels")
            ax.set_title("DFT label growth")
            fig.tight_layout()
            fig.savefig(figs / "dft_labels_vs_iteration.png", dpi=300)
            plt.close(fig)

        if labels_csv.exists() and not lab.empty and "e_kcalmol_rel" in lab.columns:
            vals = pd.to_numeric(lab["e_kcalmol_rel"], errors="coerce").dropna()
            if len(vals) > 0:
                fig, ax = plt.subplots(figsize=(6.5, 4))
                ax.hist(vals, bins=min(20, max(5, int(len(vals) ** 0.5))))
                ax.set_xlabel("Relative DFT energy (kcal/mol)")
                ax.set_ylabel("Count")
                ax.set_title("Energy distribution of labeled structures")
                fig.tight_layout()
                fig.savefig(figs / "labeled_energy_distribution.png", dpi=300)
                plt.close(fig)

        if kept:
            kdf = pd.DataFrame([_row(r, True, "kept") for r in kept])
            vc = kdf["species_id"].astype(str).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.28 * len(vc))))
            vc.sort_values().plot(kind="barh", ax=ax)
            ax.set_xlabel("Structures kept for trainset")
            ax.set_ylabel("Species")
            ax.set_title("Curated trainset composition")
            fig.tight_layout()
            fig.savefig(figs / "trainset_species_composition.png", dpi=300)
            plt.close(fig)

        # Final GEO family coverage: this is the most relevant paper-ready plot
        # for a curated ReaxFF equilibrium-geometry GEO export.
        gf = tables / "geo_family_summary.csv"
        if gf.exists():
            gfd = pd.read_csv(gf)
            if not gfd.empty and "chemical_family" in gfd.columns and "n_structures" in gfd.columns:
                gfd = gfd.sort_values("n_structures", ascending=True)
                fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.32 * len(gfd))))
                ax.barh(gfd["chemical_family"].astype(str), gfd["n_structures"].astype(float))
                ax.set_xlabel("Structures in final GEO")
                ax.set_ylabel("Chemical family")
                ax.set_title("Final GEO chemical-family coverage")
                fig.tight_layout()
                fig.savefig(figs / "geo_chemical_family_coverage.png", dpi=300)
                plt.close(fig)

        sf = tables / "selection_funnel.csv"
        # sf may not exist yet here, so plot directly from funnel below in a second pass.
    except Exception as exc:
        (analysis / "plot_warnings.txt").write_text(f"Plot generation failed: {exc}\n", encoding="utf-8")

    # 6) Human-readable summary and a stable final/ layout.
    summary = {
        "pool_structures": pool_count,
        "dft_labeled_records": labels_count,
        "candidate_records_loaded": len(records),
        "trainset_representatives": len(kept),
        "opt_jobs_total": opt_counts.get("total", 0),
        "opt_jobs_ok": opt_counts.get("ok", 0),
        "optimized_geometries": opt_counts.get("with_geometry", 0),
        "geo_structures": geo_count,
    }
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (analysis / "report.md").write_text(
        "# TSGen Explore Run Report\n\n"
        "## Summary\n\n"
        f"- Candidate pool structures: **{pool_count}**\n"
        f"- DFT-labeled records: **{labels_count}**\n"
        f"- Trainset representatives kept: **{len(kept)}**\n"
        f"- Optimization jobs OK: **{opt_counts.get('ok', 0)}/{opt_counts.get('total', 0)}**\n"
        f"- Optimized geometries parsed: **{opt_counts.get('with_geometry', 0)}/{opt_counts.get('total', 0)}**\n"
        f"- Curated optimized structures written: **{geo_count}**\n\n"
        "## Key files\n\n"
        "- `exports/optimized_xyz/`  # recommended input for `tsgen trainset`\n"
        "- `exports/final/structures/`\n"
        "- `exports/analysis/tables/`\n"
        "- `exports/analysis/figures/`\n",
        encoding="utf-8",
    )
    if logger is not None:
        try:
            logger.info(
                f"Export summary | labels={labels_count}, representatives={len(kept)}, "
                f"opt_ok={opt_counts.get('ok', 0)}/{opt_counts.get('total', 0)}, geo={geo_count}"
            )
        except Exception:
            pass


def _write_final_layout(run_dir: Path) -> None:
    """Create a clean, user-facing final/ directory without moving raw outputs.

    The final directory is rebuilt from scratch on every export to guarantee that
    final/structures is exactly consistent with reaxff_geo/geo_index.csv and
    trainset.geo.
    """
    exports = run_dir / "exports"
    final = exports / "final"
    if final.exists():
        try:
            shutil.rmtree(final)
        except Exception:
            pass
    (final / "tables").mkdir(parents=True, exist_ok=True)
    (final / "figures").mkdir(parents=True, exist_ok=True)
    (final / "structures").mkdir(parents=True, exist_ok=True)

    _safe_copy(exports / "reaxff_geo" / "trainset.geo", final / "trainset.geo")
    _safe_copy(exports / "reaxff_geo" / "optimized_structures.xyz", final / "optimized_structures.xyz")
    _safe_copy(exports / "reaxff_geo" / "candidate_structures.xyz", final / "candidate_structures.xyz")
    _safe_copy(exports / "reaxff_geo" / "geo_index.csv", final / "tables" / "geo_index.csv")
    _safe_copy(exports / "reaxff_geo" / "postopt_curation_decisions.csv", final / "tables" / "postopt_curation_decisions.csv")
    _safe_copy(exports / "reaxff_geo" / "postopt_duplicate_clusters.csv", final / "tables" / "postopt_duplicate_clusters.csv")
    _safe_copy(exports / "reaxff_geo" / "geo_family_summary.csv", final / "tables" / "geo_family_summary.csv")
    _safe_copy(exports / "trainset_candidates" / "trainset_candidates.csv", final / "tables" / "trainset_candidates.csv")
    _safe_copy(exports / "analysis" / "summary.json", final / "summary.json")
    _safe_copy(exports / "analysis" / "report.md", final / "report.md")

    src_tables = exports / "analysis" / "tables"
    if src_tables.exists():
        for f in src_tables.glob("*.csv"):
            _safe_copy(f, final / "tables" / f.name)
    src_figs = exports / "analysis" / "figures"
    if src_figs.exists():
        for f in src_figs.glob("*.png"):
            _safe_copy(f, final / "figures" / f.name)
    opt_xyz = exports / "optimized_xyz"
    if opt_xyz.exists():
        for f in opt_xyz.glob("*.xyz"):
            _safe_copy(f, final / "structures" / f.name)
    src_index = opt_xyz / "index.csv"
    if src_index.exists():
        _safe_copy(src_index, final / "tables" / "structure_files_index.csv")
    if opt_xyz.exists():
        for f in opt_xyz.glob("*.csv"):
            if f.name == "index.csv":
                continue
            _safe_copy(f, final / "tables" / f.name)

def _write_geo_atom_line(i: int, sym: str, x: float, y: float, z: float) -> str:
    return (
        f"HETATM {i:5d} {sym:<5s} RES A {1:<5d}"
        f"{x:10.5f}{y:10.5f}{z:10.5f} {sym:<5s}{0:3d}{0:2d} {0.0:8.5f}"
    )


def _formula_from_species_for_geo(species_id: str) -> str:
    return str(species_id).split("_", 1)[0] or "mol"


def write_reaxff_geo(path: Path, frames: list[Frame], records: list[CandidateRecord], optimized: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    formula_counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for rec in records:
        f = _formula_from_species_for_geo(rec.species_id)
        totals[f] = totals.get(f, 0) + 1
    for idx, (fr, rec) in enumerate(zip(frames, records), start=1):
        formula = _formula_from_species_for_geo(rec.species_id)
        formula_counts[formula] = formula_counts.get(formula, 0) + 1
        if totals.get(formula, 0) > 1:
            descrp = f"{formula}_opt_{formula_counts[formula]:02d}"
        else:
            descrp = f"{formula}_opt"
        lines.append("BIOGRF 200")
        lines.append(f"DESCRP {descrp}")
        lines.append(f"REMARK TSGen Explore optimized={optimized} source_id={rec.id}")
        lines.append("RUTYPE NORMAL RUN")
        lines.append(LEGACY_BGF_ATOM_FORMAT)
        for i, (sym, xyz) in enumerate(zip(fr.symbols, np.asarray(fr.coords, dtype=float)), start=1):
            lines.append(_write_geo_atom_line(i, sym, float(xyz[0]), float(xyz[1]), float(xyz[2])))
        lines.append(LEGACY_BGF_CONECT_FORMAT)
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_trainset_products(run_dir: Path, cfg: ExploreCfg, logger=None) -> None:
    """Post-process AL labels into trainset candidates and optional Opt/GEO products.

    This function is idempotent and safe to call after every collect/export pass.
    It keeps the original AL outputs intact and adds additional curated outputs
    only when cfg.export.scheme requests them.
    """
    scheme = str(cfg.export.scheme).strip()
    if scheme in {"validated", "none", "off"}:
        return

    records, skipped = _load_labeled_candidates(run_dir, cfg)
    out_dir = run_dir / "exports" / "trainset_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    kept, table = _select_representatives(records, cfg)
    _write_representative_xyz(run_dir, kept, table)

    # Optimization is optional and can be manual/slurm. On first run it prepares
    # or submits jobs; on later resume it collects completed opt logs.  If
    # optimization.wait_for_completion is enabled, this call blocks and polls
    # Gaussian logs to provide a one-shot workflow.
    opt_rows, opt_pairs = _wait_for_opt_jobs(run_dir, kept, cfg, logger=logger)

    geo_cfg = cfg.export.geo
    geo_records_for_analysis: list[CandidateRecord] = []
    frames_written_for_analysis = 0

    # v6 policy: Explore is an equilibrium-structure collector, not the final
    # ReaxFF trainset writer.  By default it writes curated one-XYZ-per-structure
    # files only.  A GEO file is emitted only when export.geo.enabled is explicitly
    # true; the downstream trainset builder can read the XYZ directory directly.
    if not bool(geo_cfg.enabled):
        opt_records = [r for r, _ in opt_pairs]
        opt_frames = [fr for _, fr in opt_pairs]
        pre_count = len(opt_frames)
        curation_decisions = pd.DataFrame(columns=['id', 'final_keep_for_geo', 'reason'])
        duplicate_clusters = pd.DataFrame(columns=['id', 'duplicate_of', 'dedup_distance'])
        family_summary = pd.DataFrame(columns=['chemical_family', 'n_structures', 'n_formulas', 'formulas'])
        if opt_frames:
            opt_records, opt_frames, curation_decisions, duplicate_clusters, family_summary = _curate_final_geo_records(
                opt_records, opt_frames, cfg, run_dir=run_dir
            )
        # These are derived caches, never Gaussian job/log directories. Clear
        # even if no current Opt result exists, so final cannot inherit stale XYZ
        # or an old GEO from a different export configuration/selection.
        exports_root = (run_dir / 'exports').resolve()
        for name in ['optimized_xyz', 'geo_source_xyz', 'reaxff_geo']:
            cached = run_dir / 'exports' / name
            if cached.exists():
                if cached.resolve().parent != exports_root:
                    raise ValueError(f'Refusing to clear export cache outside exports: {cached}')
                shutil.rmtree(cached)
        _write_centralized_optimized_xyz(run_dir, opt_frames, opt_records, optimized=True)
        xyz_dir = run_dir / 'exports' / 'optimized_xyz'
        curation_decisions.to_csv(xyz_dir / 'postopt_curation_decisions.csv', index=False)
        duplicate_clusters.to_csv(xyz_dir / 'postopt_duplicate_clusters.csv', index=False)
        family_summary.to_csv(xyz_dir / 'chemical_family_summary.csv', index=False)
        (xyz_dir / 'README.txt').write_text(
            'TSGen Explore curated optimized XYZ export.\n'
            'Only current-selection optimized geometries are written.\n'
            f'Structures available before final curation: {pre_count}\n'
            f'Structures written as individual XYZ files: {len(opt_frames)}\n'
            f'Optimization enabled: {bool(cfg.export.optimization.enabled)}\n',
            encoding='utf-8',
        )
        geo_records_for_analysis = list(opt_records)
        frames_written_for_analysis = len(opt_frames)

    if bool(geo_cfg.enabled):
        geo_dir = run_dir / "exports" / "reaxff_geo"
        geo_dir.mkdir(parents=True, exist_ok=True)

        opt_records = [r for r, _ in opt_pairs]
        opt_frames = [fr for _, fr in opt_pairs]
        all_optimized = len(opt_frames) == len(kept) and len(kept) > 0

        if bool(geo_cfg.prefer_optimized) and all_optimized:
            frames = opt_frames
            geo_records = opt_records
            optimized = True
            geo_mode = "optimized_all"
        elif bool(geo_cfg.prefer_optimized) and opt_frames and not bool(geo_cfg.allow_unoptimized_fallback):
            # Strict optimized mode, but only a subset parsed successfully.  Write
            # the optimized subset instead of leaving users with no GEO at all;
            # README and geo_index make the partial status explicit.
            frames = opt_frames
            geo_records = opt_records
            optimized = True
            geo_mode = "optimized_partial"
        elif bool(geo_cfg.allow_unoptimized_fallback):
            frames = [r.frame for r in kept]
            geo_records = kept
            optimized = False
            geo_mode = "unoptimized_fallback"
        else:
            frames = []
            geo_records = []
            optimized = False
            geo_mode = "waiting_for_optimized_geometries"

        # Final GEO curation after geometry optimization/fallback.  This is the
        # authoritative set for trainset.geo, optimized_structures.xyz,
        # exports/optimized_xyz, and exports/final/structures.
        pre_curation_count = len(frames)
        curation_decisions = pd.DataFrame()
        duplicate_clusters = pd.DataFrame()
        family_summary = pd.DataFrame()
        if frames:
            geo_records, frames, curation_decisions, duplicate_clusters, family_summary = _curate_final_geo_records(
                geo_records, frames, cfg, run_dir=run_dir
            )
            if pre_curation_count != len(frames):
                geo_mode = f"{geo_mode}_curated"

        # Clean stale final-source XYZ directories before writing the authoritative
        # final GEO set.
        for stale_dir in [run_dir / "exports" / "optimized_xyz", run_dir / "exports" / "geo_source_xyz"]:
            if stale_dir.exists():
                try:
                    shutil.rmtree(stale_dir)
                except Exception:
                    pass

        if frames:
            write_reaxff_geo(geo_dir / str(geo_cfg.filename), frames, geo_records, optimized=optimized)
            # Remove stale alternative merged xyz files to avoid ambiguity.
            for stale_name in ["optimized_structures.xyz", "candidate_structures.xyz"]:
                stale_path = geo_dir / stale_name
                if stale_path.exists():
                    try:
                        stale_path.unlink()
                    except Exception:
                        pass
            merged_name = "optimized_structures.xyz" if optimized else "candidate_structures.xyz"
            write_xyz_multiframe(geo_dir / merged_name, frames)
            _write_centralized_optimized_xyz(run_dir, frames, geo_records, optimized=optimized)
        else:
            # Remove stale GEO if a previous fallback file existed but current
            # strict configuration says no valid GEO should be emitted yet.
            for stale in [geo_dir / str(geo_cfg.filename), geo_dir / "optimized_structures.xyz", geo_dir / "candidate_structures.xyz"]:
                if stale.exists():
                    try:
                        stale.unlink()
                    except Exception:
                        pass

        # Rich final GEO metadata for scientific plotting and auditability.
        pd.DataFrame([_geo_index_row(r, fr, optimized, "geo_source") for r, fr in zip(geo_records, frames)]).to_csv(geo_dir / "geo_index.csv", index=False)
        if not curation_decisions.empty:
            curation_decisions.to_csv(geo_dir / "postopt_curation_decisions.csv", index=False)
        else:
            pd.DataFrame().to_csv(geo_dir / "postopt_curation_decisions.csv", index=False)
        if not duplicate_clusters.empty:
            duplicate_clusters.to_csv(geo_dir / "postopt_duplicate_clusters.csv", index=False)
        else:
            pd.DataFrame(columns=["id", "duplicate_of", "dedup_distance"]).to_csv(geo_dir / "postopt_duplicate_clusters.csv", index=False)
        if not family_summary.empty:
            family_summary.to_csv(geo_dir / "geo_family_summary.csv", index=False)
        else:
            pd.DataFrame(columns=["chemical_family", "n_structures", "n_formulas", "formulas"]).to_csv(geo_dir / "geo_family_summary.csv", index=False)

        geo_records_for_analysis = list(geo_records)
        frames_written_for_analysis = len(frames)
        n_ok = sum(1 for row in opt_rows if str(row.get("opt_status", "")).lower() == "ok")
        n_with_geom = sum(1 for row in opt_rows if bool(row.get("has_optimized_geometry", False)))
        (geo_dir / "README.txt").write_text(
            "TSGen Explore ReaxFF GEO export.\n"
            f"GEO written: {bool(frames)}\n"
            f"Structures in GEO: {len(frames)}\n"
            f"Uses optimized geometries: {optimized}\n"
            f"GEO mode: {geo_mode}\n"
            f"Representative structures requested before Opt/GEO curation: {len(kept)}\n"
            f"Structures available before final post-opt curation: {pre_curation_count}\n"
            f"Structures removed by final curation: {max(0, pre_curation_count - len(frames))}\n"
            f"Optimization logs with Normal termination: {n_ok}/{len(kept)}\n"
            f"Optimized geometries parsed: {n_with_geom}/{len(kept)}\n"
            "If optimization is enabled but not all opt jobs are complete, rerun/resume after Gaussian Opt jobs finish.\n"
            "If logs are complete but parsed geometries are zero, inspect exports/trainset_candidates/opt_jobs.csv parse_note.\n"
            "The GEO file is a BIOGRF/BGF-like coordinate file; keep labels.csv/geo_index.csv as the energy provenance table.\n",
            encoding="utf-8",
        )

    # Publication/reporting products and normalized final layout.
    # These tables/figures are derived from the current export. Optional outputs
    # absent this time must not survive from a previous GEO-enabled selection.
    analysis_cache = run_dir / 'exports' / 'analysis'
    if analysis_cache.exists():
        if analysis_cache.resolve().parent != (run_dir / 'exports').resolve():
            raise ValueError(f'Refusing to clear analysis cache outside exports: {analysis_cache}')
        shutil.rmtree(analysis_cache)
    try:
        _write_analysis_products(
            run_dir, records, kept, table, opt_rows,
            geo_records=geo_records_for_analysis,
            frames_written=frames_written_for_analysis,
            logger=logger,
        )
        _write_final_layout(run_dir)
    except Exception as exc:
        if logger is not None:
            try:
                logger.warning(f"Analysis/final layout export failed: {exc}")
            except Exception:
                pass

    summary = {
        "scheme": scheme,
        "labels_seen": len(records),
        "representatives_kept": len(kept),
        "skipped": skipped,
        "optimization_enabled": bool(cfg.export.optimization.enabled),
        "optimization_records": len(opt_rows),
        "exclude_open_shell": bool(getattr(cfg.export.selection, "exclude_open_shell", False)),
        "allowed_multiplicities": getattr(cfg.export.selection, "allowed_multiplicities", None),
        "exclude_species_regex": list(getattr(cfg.export.selection, "exclude_species_regex", []) or []),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if logger is not None:
        try:
            logger.info(f"Trainset post-processing: kept {len(kept)} representative structures from {len(records)} labeled candidates.")
        except Exception:
            pass
