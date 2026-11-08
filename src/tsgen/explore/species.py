from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
from typing import List, Literal, Tuple

import numpy as np

from tsgen.data.xyz import Frame


_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
}


def _formula(symbols: List[str]) -> str:
    c = Counter(symbols)
    # Hill-like ordering: C, H, then alphabetical
    keys: List[str] = []
    if "C" in c:
        keys.append("C")
    if "H" in c:
        keys.append("H")
    for k in sorted(c.keys()):
        if k not in keys:
            keys.append(k)
    out = []
    for k in keys:
        n = c[k]
        out.append(f"{k}{n if n != 1 else ''}")
    return "".join(out)


def _bond_graph(
    symbols: List[str], coords: np.ndarray, bond_tol: float
) -> List[List[int]]:
    n = len(symbols)
    radii = np.array([_RADII.get(s, 0.8) for s in symbols], dtype=float)
    # Evaluate each pair once without a full distance matrix.
    adj = [[] for _ in range(n)]
    for i in range(n):
        ri = radii[i]
        # compute distances to j>i
        diff = coords[i + 1 :] - coords[i]
        d = np.linalg.norm(diff, axis=1)
        rcut = (ri + radii[i + 1 :]) * bond_tol
        bonded = np.where(d < rcut)[0]
        for idx in bonded:
            j = i + 1 + int(idx)
            adj[i].append(j)
            adj[j].append(i)
    return adj


def _connected_components(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        comps.append(comp)
    return comps


# Overlapping windows in angstroms allow distorted and aromatic configurations.
_CC_LENGTHS = ((1.27, 1.72), (1.20, 1.48), (1.10, 1.30))
_LINEAR_COSINE_LIMIT = np.cos(np.deg2rad(150.0))


@lru_cache(maxsize=8192)
def _match_bond_orders(
    remaining: tuple[int, ...], edges: tuple[tuple[int, int, tuple[int, ...]], ...]
) -> bool:
    """Assign C-C bond increments above single bonds to fill carbon valences."""
    if not edges:
        return not any(remaining)
    minimum = [0] * len(remaining)
    maximum = [0] * len(remaining)
    for i, j, increments in edges:
        minimum[i] += increments[0]
        minimum[j] += increments[0]
        maximum[i] += increments[-1]
        maximum[j] += increments[-1]
    if any(not low <= need <= high for need, low, high in zip(remaining, minimum, maximum)):
        return False
    i, j, increments = edges[0]
    for extra in increments:
        if extra > min(remaining[i], remaining[j]):
            break
        updated = list(remaining)
        updated[i] -= extra
        updated[j] -= extra
        if _match_bond_orders(tuple(updated), edges[1:]):
            return True
    return False


def neutral_hydrocarbon_drop_reason(frame: Frame, bond_tol: float) -> str:
    """Screen geometries against a neutral H/C valence and bond-length model."""
    symbols = frame.symbols
    if not set(symbols) <= {"C", "H"}:
        return "neutral hydrocarbon screen requires H/C elements"
    graph = _bond_graph(symbols, frame.coords, bond_tol)
    if len(_connected_components(graph)) != 1:
        return "neutral hydrocarbon screen requires one connected molecule"
    carbon_atoms = [i for i, symbol in enumerate(symbols) if symbol == "C"]
    carbon = {atom: index for index, atom in enumerate(carbon_atoms)}
    remaining = tuple(4 - len(graph[i]) for i in carbon)
    if any(need < 0 for need in remaining) or any(
        len(graph[i]) != 1 for i, sym in enumerate(symbols) if sym == "H"
    ):
        return "neutral hydrocarbon valence incompatible with connectivity"
    edges = []
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            if i >= j:
                continue
            length = float(np.linalg.norm(frame.coords[i] - frame.coords[j]))
            if i in carbon and j in carbon:
                increments = tuple(k for k, (lo, hi) in enumerate(_CC_LENGTHS) if lo <= length <= hi)
                if not increments:
                    return "C-C bond length outside hydrocarbon screen"
                edges.append((carbon[i], carbon[j], increments))
            else:
                is_ch = i in carbon or j in carbon
                lo, hi = (0.90, 1.25) if is_ch else (0.60, 0.90)
                if not lo <= length <= hi:
                    pair = "C-H" if is_ch else "H-H"
                    return f"{pair} bond length outside hydrocarbon screen"
    if not _match_bond_orders(remaining, tuple(edges)):
        return "neutral hydrocarbon valence incompatible with bond orders"
    for i in carbon_atoms:
        if len(graph[i]) == 2:
            a, b = frame.coords[graph[i]] - frame.coords[i]
            limit = _LINEAR_COSINE_LIMIT * np.linalg.norm(a) * np.linalg.norm(b)
            if np.dot(a, b) > limit:
                return "two-coordinate carbon is outside the near-linear geometry screen"
    return ""


def _component_hash(symbols: List[str], adj: List[List[int]], comp: List[int]) -> str:
    # Create a lightweight, order-invariant graph fingerprint:
    # multiset of (label, degree, neighbor-label multiset)
    tokens: List[str] = []
    for v in comp:
        lbl = symbols[v]
        deg = len([u for u in adj[v] if u in set(comp)])
        nbl = sorted(symbols[u] for u in adj[v] if u in set(comp))
        tokens.append(f"{lbl}|{deg}|{','.join(nbl)}")
    tokens.sort()
    h = hashlib.sha256(";".join(tokens).encode("utf-8")).hexdigest()[:8]
    return h


FingerprintVersion = Literal["legacy", "wl_v2"]


def _graph_hash(
    symbols: List[str],
    adj: List[List[int]],
    comp: List[int],
    version: FingerprintVersion,
) -> str:
    if version == "legacy":
        return _component_hash(symbols, adj, comp)
    if version != "wl_v2":
        raise ValueError(f"Unknown species fingerprint version: {version}")
    members = set(comp)
    labels = {v: symbols[v] for v in comp}
    for _ in range(len(comp)):
        refined = {}
        for v in comp:
            neighbors = sorted(labels[u] for u in adj[v] if u in members)
            token = labels[v] + "|" + ",".join(neighbors)
            refined[v] = hashlib.sha256(token.encode("utf-8")).hexdigest()
        labels = refined
    digest = hashlib.sha256(
        ";".join(sorted(labels.values())).encode("utf-8")
    ).hexdigest()[:16]
    return f"v2{digest}"


def species_signature(
    frame: Frame,
    bond_tol: float = 1.2,
    max_components: int | None = None,
    *,
    fingerprint_version: FingerprintVersion = "wl_v2",
) -> str:
    """Return a composition and graph fingerprint, invariant to atom ordering.

    Version 2 refines atom labels over the connected graph. It is a graph
    fingerprint, so equality alone does not establish graph isomorphism.
    """
    syms = frame.symbols
    coords = frame.coords
    if len(syms) == 0:
        return "EMPTY"

    adj = _bond_graph(syms, coords, bond_tol=bond_tol)
    comps = _connected_components(adj)
    comps = sorted(comps, key=lambda c: len(c), reverse=True)
    limit = (
        4
        if max_components is None and fingerprint_version == "legacy"
        else max_components
    )
    parts: List[str] = []
    for comp in comps[:limit]:
        s_comp = [syms[i] for i in comp]
        f = _formula(s_comp)
        gh = _graph_hash(syms, adj, comp, fingerprint_version)
        parts.append(f"{f}_{gh}")
    if fingerprint_version == "wl_v2":
        parts.sort()
    return "+".join(parts)


def split_frame_components(
    frame: Frame,
    bond_tol: float = 1.2,
    *,
    fingerprint_version: FingerprintVersion = "wl_v2",
) -> List[Tuple[Frame, str]]:
    """Split a (possibly multi-molecule) frame into connected fragments.

    Each connected component is converted into a new Frame, preserving a minimal comment.
    The returned species_id corresponds to <formula>_<graphhash> for that component.
    """
    syms = frame.symbols
    coords = frame.coords
    if len(syms) == 0:
        return []

    adj = _bond_graph(syms, coords, bond_tol=bond_tol)
    comps = _connected_components(adj)
    out: List[Tuple[Frame, str]] = []
    for comp in comps:
        comp = sorted(comp)
        s_comp = [syms[i] for i in comp]
        c_comp = coords[np.array(comp, dtype=int)]
        f = _formula(s_comp)
        gh = _graph_hash(syms, adj, comp, fingerprint_version)
        sid = f"{f}_{gh}"
        out.append(
            (
                Frame(
                    symbols=s_comp,
                    coords=c_comp,
                    comment=f"{sid} | parent: {frame.comment}",
                ),
                sid,
            )
        )
    return out
