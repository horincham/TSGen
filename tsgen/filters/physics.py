from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np
from tsgen.data.xyz import Frame

def pairwise_min_dist(coords: np.ndarray) -> float:
    n = coords.shape[0]
    mind = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < mind:
                mind = d
    return mind


def any_extreme_angle(
    symbols: List[str],
    coords: np.ndarray,
    angle_limits_deg: Tuple[float, float],
    bond_scale: float = 1.25,
    max_bond_angstrom: float = 2.25,
) -> bool:
    """
    Bonded-only angle sanity check (robust for small molecules).

    The previous "all-triplet" style checks (even with a loose neighbor cutoff)
    can falsely reject most candidates for small scanned structures because many
    non-bonded triplets become nearly collinear.

    Here we:
      1) build a simple covalent bond graph from interatomic distances using
         covalent radii and a scale factor;
      2) only evaluate angles i-j-k where i and k are *bonded neighbors* of j.

    This preserves physical validity while avoiding over-filtering.
    """
    lo, hi = angle_limits_deg
    n = coords.shape[0]
    if n < 3:
        return False

    # Covalent radii (Å), common elements for TSGen use-cases.
    # (Approximate; extend as needed.)
    rcov = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
        "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
        "Br": 1.20, "I": 1.39, "Al": 1.21,
        "Na": 1.66, "K": 2.03,
    }
    def r(sym: str) -> float:
        return rcov.get(sym, 0.77)  # conservative fallback

    # Distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    # Build bonded neighbors
    neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = float(dist[i, j])
            # exclude nearly overlapping atoms; min_distance handled elsewhere
            if dij <= 1e-12:
                continue
            cutoff = min(max_bond_angstrom, bond_scale * (r(symbols[i]) + r(symbols[j])))
            if dij <= cutoff:
                neighbors[i].append(j)
                neighbors[j].append(i)

    # Check only bonded angles around central atom j using bonded neighbor pairs
    for j in range(n):
        nb = neighbors[j]
        if len(nb) < 2:
            continue
        cj = coords[j]
        for a in range(len(nb) - 1):
            i = nb[a]
            v1 = coords[i] - cj
            n1 = np.linalg.norm(v1) + 1e-12
            for b in range(a + 1, len(nb)):
                k = nb[b]
                v2 = coords[k] - cj
                n2 = np.linalg.norm(v2) + 1e-12
                c = float(np.dot(v1, v2) / (n1 * n2))
                c = max(-1.0, min(1.0, c))
                ang = float(np.degrees(np.arccos(c)))
                if ang < lo or ang > hi:
                    return True

    return False


@dataclass
class PhysicsFilter:
    min_distance_angstrom: float
    angle_limits_deg: Tuple[float, float]

    def apply(self, frames: List[Frame]):
        kept: List[Frame] = []
        flags: List[Dict[str, Any]] = []
        for idx, fr in enumerate(frames):
            reasons: List[str] = []

            md = pairwise_min_dist(fr.coords)
            if md < self.min_distance_angstrom:
                reasons.append(f"min_dist<{self.min_distance_angstrom:.2f} (got {md:.2f})")

            ea = any_extreme_angle(fr.symbols, fr.coords, self.angle_limits_deg)
            if ea:
                reasons.append(f"extreme_angle_outside{self.angle_limits_deg}")

            flags.append({
                "frame_index": idx,
                "keep": len(reasons) == 0,
                "reasons": ";".join(reasons),
                "min_dist": float(md),
                "extreme_angle": bool(ea),
            })

            if len(reasons) == 0:
                kept.append(fr)

        return kept, flags
