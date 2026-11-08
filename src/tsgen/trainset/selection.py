from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from tsgen.trainset.chem import (
    Bond,
    adjacency,
    angle_deg,
    connected_component,
    dihedral_deg,
    edge_in_cycle,
    family_from_graph,
    formula_from_symbols,
    infer_bonds,
)
from tsgen.trainset.config import CoordinateSelectionCfg
from tsgen.trainset.io import sanitize_id
from tsgen.trainset.models import CoordinateSpec, EquilibriumStructure


def _priority(eq: EquilibriumStructure, cfg: CoordinateSelectionCfg) -> float:
    n_atoms = len(eq.frame.symbols)
    score = 0.0
    if eq.formula in set(cfg.protected_formulas or []):
        score += 1000.0
    if cfg.prefer_small_molecules:
        score += max(0.0, 200.0 - 2.0 * n_atoms)
    # mild preference for oxygenated and benchmark-like structures because these are often sparse
    if "O" in eq.frame.symbols:
        score += 20.0
    return score


def select_coordinates(eqs: Sequence[EquilibriumStructure], cfg: CoordinateSelectionCfg) -> tuple[list[CoordinateSpec], list[dict]]:
    """Select representative bond/angle/torsion scan coordinates.

    The goal is coverage, not exhaustive enumeration.  Quotas are per coordinate
    family and per parent structure, preventing high-frequency structures from
    dominating the scan training set.
    """
    candidates: list[CoordinateSpec] = []
    audit: list[dict] = []

    for eq in eqs:
        frame = eq.frame
        coords = np.asarray(frame.coords, dtype=float)
        bonds = infer_bonds(frame)
        adj = adjacency(len(frame.symbols), bonds)
        parent_score = _priority(eq, cfg)
        bond_by_edge = {tuple(sorted((b.i, b.j))): b for b in bonds}

        # Bond scans
        if cfg.bond_scans.enabled:
            allowed_fams = set(cfg.bond_scans.families or [])
            for b in bonds:
                if allowed_fams and b.family not in allowed_fams:
                    audit.append(_audit(eq, "bond", b.family, (b.i, b.j), b.distance, "drop_family_not_enabled"))
                    continue
                cid = sanitize_id(f"bnd_{eq.parent_id}_{b.i+1}_{b.j+1}_{b.family}")
                candidates.append(CoordinateSpec(
                    coord_id=cid,
                    parent_id=eq.parent_id,
                    scan_type="bond",
                    family=b.family,
                    atom_indices=(b.i, b.j),
                    atom_symbols=(frame.symbols[b.i], frame.symbols[b.j]),
                    eq_value=float(b.distance),
                    priority=parent_score + _family_bonus(b.family),
                ))

        # Angle scans
        if cfg.angle_scans.enabled:
            for j in range(len(frame.symbols)):
                neigh = sorted(adj[j])
                for a_idx in range(len(neigh)):
                    for b_idx in range(a_idx + 1, len(neigh)):
                        i, k = neigh[a_idx], neigh[b_idx]
                        val = angle_deg(coords, i, j, k)
                        if not np.isfinite(val) or val < cfg.angle_scans.min_angle_deg or val > cfg.angle_scans.max_angle_deg:
                            audit.append(_audit(eq, "angle", "angle", (i, j, k), val, "drop_angle_out_of_range"))
                            continue
                        fam = angle_family(frame.symbols, bond_by_edge, i, j, k)
                        cid = sanitize_id(f"ang_{eq.parent_id}_{i+1}_{j+1}_{k+1}_{fam}")
                        candidates.append(CoordinateSpec(
                            coord_id=cid,
                            parent_id=eq.parent_id,
                            scan_type="angle",
                            family=fam,
                            atom_indices=(i, j, k),
                            atom_symbols=(frame.symbols[i], frame.symbols[j], frame.symbols[k]),
                            eq_value=float(val),
                            priority=parent_score + _family_bonus(fam),
                        ))

        # Torsion scans
        if cfg.torsion_scans.enabled:
            seen_paths = set()
            for j in range(len(frame.symbols)):
                for k in sorted(adj[j]):
                    if j >= k:
                        continue
                    center_b = bond_by_edge.get(tuple(sorted((j, k))))
                    if center_b is None:
                        continue
                    if cfg.torsion_scans.exclude_double_bond_center and center_b.order_label in {"double", "triple"}:
                        audit.append(_audit(eq, "torsion", center_b.family, (j, k), center_b.distance, "drop_double_or_triple_center"))
                        continue
                    if cfg.torsion_scans.exclude_triple_bond_related and center_b.order_label == "triple":
                        audit.append(_audit(eq, "torsion", center_b.family, (j, k), center_b.distance, "drop_triple_center"))
                        continue
                    if cfg.torsion_scans.exclude_ring_torsions and edge_in_cycle(len(frame.symbols), bonds, j, k):
                        audit.append(_audit(eq, "torsion", center_b.family, (j, k), center_b.distance, "drop_ring_torsion"))
                        continue
                    lefts = sorted(x for x in adj[j] if x != k)
                    rights = sorted(x for x in adj[k] if x != j)
                    for i in lefts:
                        for l in rights:
                            path = (i, j, k, l)
                            canon = min(path, tuple(reversed(path)))
                            if canon in seen_paths:
                                continue
                            seen_paths.add(canon)
                            # Gaussian rejects or becomes unstable for dihedrals that
                            # involve near-linear adjacent angles.  Filter them before
                            # generating ModRedundant scan jobs.
                            a1 = angle_deg(coords, i, j, k)
                            a2 = angle_deg(coords, j, k, l)
                            lo = float(cfg.torsion_scans.reject_near_zero_angle_deg)
                            hi = float(cfg.torsion_scans.reject_near_linear_angle_deg)
                            if (not np.isfinite(a1)) or (not np.isfinite(a2)) or a1 < lo or a2 < lo or a1 > hi or a2 > hi:
                                audit.append(_audit(eq, "torsion", "near_linear_torsion", path, float("nan"), "drop_near_linear_torsion"))
                                continue
                            if cfg.torsion_scans.exclude_triple_bond_related:
                                b_left = bond_by_edge.get(tuple(sorted((i, j))))
                                b_right = bond_by_edge.get(tuple(sorted((k, l))))
                                if (b_left and b_left.order_label == "triple") or (b_right and b_right.order_label == "triple"):
                                    audit.append(_audit(eq, "torsion", "triple_related_torsion", path, float("nan"), "drop_triple_bond_related"))
                                    continue
                            if cfg.torsion_scans.exclude_terminal_hydrogen_rotation:
                                # Skip pure methyl H-C-C-H-like torsions only if both terminal atoms are H.
                                # H-C-C-H can still be important, but exhaustive H rotations are redundant; keep one via quota.
                                pass
                            val = dihedral_deg(coords, i, j, k, l)
                            if not np.isfinite(val):
                                audit.append(_audit(eq, "torsion", "torsion", path, float("nan"), "drop_invalid_torsion"))
                                continue
                            fam = torsion_family(frame.symbols, bond_by_edge, i, j, k, l)
                            cid = sanitize_id(f"tor_{eq.parent_id}_{i+1}_{j+1}_{k+1}_{l+1}_{fam}")
                            candidates.append(CoordinateSpec(
                                coord_id=cid,
                                parent_id=eq.parent_id,
                                scan_type="torsion",
                                family=fam,
                                atom_indices=path,
                                atom_symbols=(frame.symbols[i], frame.symbols[j], frame.symbols[k], frame.symbols[l]),
                                eq_value=float(val),
                                priority=parent_score + _family_bonus(fam),
                            ))

    selected: list[CoordinateSpec] = []
    selected_keys = set()
    per_type_family: Dict[tuple[str, str], int] = defaultdict(int)
    per_parent: Dict[str, int] = defaultdict(int)
    total_per_type: Dict[str, int] = defaultdict(int)

    def limits(c: CoordinateSpec) -> tuple[int, int | None]:
        if c.scan_type == "bond":
            return int(cfg.bond_scans.max_per_family), cfg.bond_scans.max_total
        if c.scan_type == "angle":
            return int(cfg.angle_scans.max_per_family), cfg.angle_scans.max_total
        return int(cfg.torsion_scans.max_per_family), cfg.torsion_scans.max_total

    # Sort by scan type then family priority.  Keep chemically diverse structures.
    candidates.sort(key=lambda c: (-c.priority, c.scan_type, c.family, c.parent_id, c.atom_indices))
    for c in candidates:
        fam_lim, type_lim = limits(c)
        key = (c.scan_type, c.family)
        if per_type_family[key] >= fam_lim:
            audit.append(_coord_audit(c, "drop_family_quota"))
            continue
        if type_lim is not None and total_per_type[c.scan_type] >= int(type_lim):
            audit.append(_coord_audit(c, "drop_type_quota"))
            continue
        if per_parent[c.parent_id] >= int(cfg.max_coordinates_per_parent):
            audit.append(_coord_audit(c, "drop_parent_quota"))
            continue
        # Avoid selecting identical local coordinate on same parent/family.
        uniq = (c.parent_id, c.scan_type, c.family, c.atom_indices)
        if uniq in selected_keys:
            audit.append(_coord_audit(c, "drop_duplicate_coordinate"))
            continue
        selected.append(c)
        selected_keys.add(uniq)
        per_type_family[key] += 1
        per_parent[c.parent_id] += 1
        total_per_type[c.scan_type] += 1
        audit.append(_coord_audit(c, "keep"))

    return selected, audit


def angle_family(symbols: Sequence[str], bond_by_edge: Dict[Tuple[int, int], Bond], i: int, j: int, k: int) -> str:
    b1 = bond_by_edge.get(tuple(sorted((i, j))))
    b2 = bond_by_edge.get(tuple(sorted((j, k))))
    ends = sorted([symbols[i], symbols[k]])
    center = symbols[j]
    order_part = ""
    if b1 and b2:
        order_part = f"_{b1.order_label}_{b2.order_label}"
    return f"{ends[0]}-{center}-{ends[1]}{order_part}"


def torsion_family(symbols: Sequence[str], bond_by_edge: Dict[Tuple[int, int], Bond], i: int, j: int, k: int, l: int) -> str:
    center = bond_by_edge.get(tuple(sorted((j, k))))
    seq = f"{symbols[i]}-{symbols[j]}-{symbols[k]}-{symbols[l]}"
    rev = f"{symbols[l]}-{symbols[k]}-{symbols[j]}-{symbols[i]}"
    base = min(seq, rev)
    if center:
        return f"{base}_center_{center.family}"
    return base


def _family_bonus(fam: str) -> float:
    bonus = 0.0
    for key in ["C-O", "O-H", "O-O", "C-C_double", "C-C_triple", "C-O_double", "C-O_triple"]:
        if key in fam:
            bonus += 10.0
    return bonus


def _audit(eq: EquilibriumStructure, scan_type: str, family: str, atoms: tuple[int, ...], value: float, decision: str) -> dict:
    return {
        "parent_id": eq.parent_id,
        "formula": eq.formula,
        "scan_type": scan_type,
        "family": family,
        "atom_indices_1based": " ".join(str(a + 1) for a in atoms),
        "eq_value": value,
        "decision": decision,
    }


def _coord_audit(c: CoordinateSpec, decision: str) -> dict:
    return {
        "coord_id": c.coord_id,
        "parent_id": c.parent_id,
        "scan_type": c.scan_type,
        "family": c.family,
        "atom_indices_1based": " ".join(str(a + 1) for a in c.atom_indices),
        "atom_symbols": "-".join(c.atom_symbols),
        "eq_value": c.eq_value,
        "priority": c.priority,
        "decision": decision,
    }
