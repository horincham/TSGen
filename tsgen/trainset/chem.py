from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from tsgen.data.xyz import Frame

# Minimal element data for organic/combustion use.  Extend as needed.
ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}


@dataclass(frozen=True)
class Bond:
    i: int
    j: int
    distance: float
    family: str
    order_label: str


def formula_from_symbols(symbols: Sequence[str]) -> str:
    counts = Counter(symbols)
    # Hill-style for organic molecules: C, H, then alphabetical remainder.
    parts: List[str] = []
    ordered: List[str] = []
    if "C" in counts:
        ordered.append("C")
    if "H" in counts:
        ordered.append("H")
    ordered.extend(sorted(k for k in counts if k not in {"C", "H"}))
    for el in ordered:
        n = counts[el]
        parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts)


def element_counts(symbols: Sequence[str]) -> Dict[str, int]:
    return dict(Counter(symbols))


def total_electrons(symbols: Sequence[str], charge: int = 0) -> int:
    try:
        return int(sum(ATOMIC_NUMBERS[s] for s in symbols) - int(charge))
    except KeyError as exc:
        raise ValueError(f"Unsupported element for electron count: {exc.args[0]}") from exc


def resolve_charge_multiplicity(frame: Frame, charge_cfg="auto", mult_cfg="auto") -> tuple[int, int, int]:
    q = 0 if isinstance(charge_cfg, str) and charge_cfg.lower() == "auto" else int(charge_cfg)
    ne = total_electrons(frame.symbols, q)
    if isinstance(mult_cfg, str) and mult_cfg.lower() == "auto":
        m = 1 if ne % 2 == 0 else 2
    else:
        m = int(mult_cfg)
    parity_ok = ((ne % 2 == 0) and (m % 2 == 1)) or ((ne % 2 == 1) and (m % 2 == 0))
    if not parity_ok:
        raise ValueError(f"Incompatible charge/multiplicity: q={q}, mult={m}, electrons={ne}")
    return q, m, ne


def passes_chemistry(frame: Frame, cfg) -> tuple[bool, str]:
    counts = element_counts(frame.symbols)
    elems = set(counts)
    allowed = set(cfg.allowed_elements or [])
    if allowed and not elems.issubset(allowed):
        return False, f"elements {sorted(elems)} not subset of allowed {sorted(allowed)}"
    excluded = set(cfg.exclude_elements or [])
    bad = elems.intersection(excluded)
    if bad:
        return False, f"excluded elements present: {sorted(bad)}"
    required = set(cfg.required_elements or [])
    if required and not required.issubset(elems):
        return False, f"missing required elements: {sorted(required - elems)}"
    for el, nmin in (cfg.min_element_counts or {}).items():
        if counts.get(el, 0) < int(nmin):
            return False, f"{el} count {counts.get(el, 0)} < {nmin}"
    for el, nmax in (cfg.max_element_counts or {}).items():
        if counts.get(el, 0) > int(nmax):
            return False, f"{el} count {counts.get(el, 0)} > {nmax}"
    if getattr(cfg, "exclude_open_shell", False):
        q = 0 if str(getattr(cfg, "charge", "auto")).lower() == "auto" else int(getattr(cfg, "charge"))
        if total_electrons(frame.symbols, q) % 2 != 0:
            return False, "odd electron count / likely open shell"
    return True, "ok"


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def infer_bonds(frame: Frame, bond_tol: float = 1.20, extra_tol: float = 0.12) -> List[Bond]:
    coords = np.asarray(frame.coords, dtype=float)
    n = len(frame.symbols)
    bonds: List[Bond] = []
    for i in range(n):
        si = frame.symbols[i]
        ri = COVALENT_RADII.get(si, 0.75)
        for j in range(i + 1, n):
            sj = frame.symbols[j]
            rj = COVALENT_RADII.get(sj, 0.75)
            d = float(np.linalg.norm(coords[j] - coords[i]))
            cutoff = bond_tol * (ri + rj) + extra_tol
            if d <= cutoff:
                fam, order = classify_bond(si, sj, d)
                bonds.append(Bond(i=i, j=j, distance=d, family=fam, order_label=order))
    return bonds


def classify_bond(a: str, b: str, d: float) -> tuple[str, str]:
    pair = "-".join(sorted([a, b]))
    if pair == "C-H":
        return "C-H", "single"
    if pair == "H-O":
        return "O-H", "single"
    if pair == "H-H":
        return "H-H", "single"
    if pair == "O-O":
        return "O-O_single", "single"
    if pair == "C-C":
        if d < 1.25:
            return "C-C_triple", "triple"
        if d < 1.42:
            return "C-C_double", "double"
        return "C-C_single", "single"
    if pair == "C-O":
        if d < 1.17:
            return "C-O_triple", "triple"
        if d < 1.30:
            return "C-O_double", "double"
        return "C-O_single", "single"
    return f"{pair}_bond", "single"


def adjacency(n: int, bonds: Sequence[Bond], skip: Optional[tuple[int, int]] = None) -> List[Set[int]]:
    adj = [set() for _ in range(n)]
    sk = tuple(sorted(skip)) if skip else None
    for b in bonds:
        edge = tuple(sorted((b.i, b.j)))
        if sk is not None and edge == sk:
            continue
        adj[b.i].add(b.j)
        adj[b.j].add(b.i)
    return adj


def connected_component(adj: Sequence[Set[int]], start: int) -> Set[int]:
    seen: Set[int] = {start}
    q: deque[int] = deque([start])
    while q:
        v = q.popleft()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    return seen


def edge_in_cycle(n: int, bonds: Sequence[Bond], i: int, j: int) -> bool:
    adj = adjacency(n, bonds, skip=(i, j))
    return j in connected_component(adj, i)


def angle_deg(coords: np.ndarray, i: int, j: int, k: int) -> float:
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    c = np.clip(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def dihedral_deg(coords: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    p0, p1, p2, p3 = coords[[i, j, k, l]]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    n = np.linalg.norm(b1)
    if n < 1e-12:
        return float("nan")
    b1 = b1 / n
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv < 1e-12 or nw < 1e-12:
        return float("nan")
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def rotate_points(coords: np.ndarray, indices: Iterable[int], origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    out = np.array(coords, dtype=float, copy=True)
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return out
    u = axis / norm
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    # Rodrigues rotation matrix
    ux, uy, uz = u
    R = np.array([
        [c + ux*ux*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c), uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)],
    ])
    for idx in indices:
        out[idx] = origin + R @ (coords[idx] - origin)
    return out


def min_nonbonded_distance(frame: Frame, bonds: Sequence[Bond]) -> float:
    bonded = {tuple(sorted((b.i, b.j))) for b in bonds}
    coords = np.asarray(frame.coords, dtype=float)
    n = len(frame.symbols)
    dmin = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            d = float(np.linalg.norm(coords[j] - coords[i]))
            if d < dmin:
                dmin = d
    return dmin if dmin < float("inf") else 999.0


def family_from_graph(frame: Frame, bonds: Sequence[Bond]) -> str:
    counts = element_counts(frame.symbols)
    formula = formula_from_symbols(frame.symbols)
    bfamilies = Counter(b.family for b in bonds)
    c = counts.get("C", 0)
    o = counts.get("O", 0)
    if formula in {"H2", "O2", "H2O", "CO", "CO2", "CH4", "C2H2", "C2H4", "C2H6"}:
        return f"benchmark_{formula}"
    size = "C0" if c == 0 else "C1" if c == 1 else "C2" if c == 2 else "C3-C4" if c <= 4 else "C5-C9"
    if o > 0:
        if bfamilies.get("C-O_triple", 0):
            motif = "carbon_monoxide"
        elif bfamilies.get("C-O_double", 0):
            motif = "carbonyl_or_CO2"
        elif bfamilies.get("O-O_single", 0):
            motif = "peroxide"
        elif bfamilies.get("O-H", 0):
            motif = "alcohol_or_water"
        else:
            motif = "ether_or_CHO"
        return f"{size}_oxygenated_{motif}"
    if bfamilies.get("C-C_triple", 0):
        motif = "alkyne"
    elif bfamilies.get("C-C_double", 0):
        motif = "alkene_or_cycloalkene"
    elif c > 0:
        motif = "alkane_or_saturated"
    else:
        motif = "hydrogen_only"
    return f"{size}_hydrocarbon_{motif}"
