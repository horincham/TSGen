from __future__ import annotations

from collections import Counter
import hashlib
from typing import Dict, List, Tuple

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
        out.append(f"{k}{n if n!=1 else ''}")
    return "".join(out)


def _bond_graph(symbols: List[str], coords: np.ndarray, bond_tol: float) -> List[List[int]]:
    n = len(symbols)
    radii = np.array([_RADII.get(s, 0.8) for s in symbols], dtype=float)
    # pairwise distances (n,n) is acceptable up to ~2000 atoms; for safety we use blocks
    adj = [[] for _ in range(n)]
    for i in range(n):
        ri = radii[i]
        # compute distances to j>i
        diff = coords[i+1:] - coords[i]
        d = np.linalg.norm(diff, axis=1)
        rcut = (ri + radii[i+1:]) * bond_tol
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


def species_signature(frame: Frame, bond_tol: float = 1.2, max_components: int = 4) -> str:
    """Return a species signature for a (possibly multi-molecule) frame.

    The signature is a '+'-joined list of the largest components, each encoded as
    <formula>_<graphhash>.
    """
    syms = frame.symbols
    coords = frame.coords
    if len(syms) == 0:
        return "EMPTY"

    adj = _bond_graph(syms, coords, bond_tol=bond_tol)
    comps = _connected_components(adj)
    comps = sorted(comps, key=lambda c: len(c), reverse=True)
    parts: List[str] = []
    for comp in comps[:max_components]:
        s_comp = [syms[i] for i in comp]
        f = _formula(s_comp)
        gh = _component_hash(syms, adj, comp)
        parts.append(f"{f}_{gh}")
    return "+".join(parts)


def split_frame_components(frame: Frame, bond_tol: float = 1.2) -> List[Tuple[Frame, str]]:
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
        gh = _component_hash(syms, adj, comp)
        sid = f"{f}_{gh}"
        out.append((Frame(symbols=s_comp, coords=c_comp, comment=f"{sid} | parent: {frame.comment}"), sid))
    return out
