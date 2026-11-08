from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np
from tsgen.data.xyz import Frame
from tsgen.filters.geometry import infer_reference_bonds, local_angle_range, reference_bond_ratio

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
    """Check local bonded angles; diatomic/isolated atoms have no angle."""
    bonds = infer_reference_bonds(symbols, coords, bond_scale, max_bond_angstrom)
    low, high = local_angle_range(coords, bonds)
    return bool(low < angle_limits_deg[0]-1e-10 or high > angle_limits_deg[1]+1e-10)


@dataclass
class PhysicsFilter:
    min_distance_angstrom: float
    angle_limits_deg: Tuple[float, float]
    max_bond_ratio: float | None = None
    bond_scale: float = 1.25
    max_bond_angstrom: float = 2.25

    def apply(self, frames: List[Frame], reference_bonds=None):
        if reference_bonds is not None and len(reference_bonds) != len(frames):
            raise ValueError('One reference bond list is required per frame')
        if self.max_bond_ratio is not None and reference_bonds is None:
            raise ValueError('rho screening requires fixed reference bonds')
        kept: List[Frame] = []
        flags: List[Dict[str, Any]] = []
        for idx, fr in enumerate(frames):
            reasons: List[str] = []
            coords = np.asarray(fr.coords)
            if coords.shape != (len(fr.symbols), 3) or not np.isfinite(coords).all():
                raise ValueError(f'Invalid coordinates in frame {idx}')

            md = pairwise_min_dist(fr.coords)
            if md < self.min_distance_angstrom:
                reasons.append(f"min_dist<{self.min_distance_angstrom:.2f} (got {md:.2f})")

            bonds = (reference_bonds[idx] if reference_bonds is not None else
                     infer_reference_bonds(fr.symbols, coords, self.bond_scale, self.max_bond_angstrom))
            theta_min, theta_max = local_angle_range(coords, bonds)
            lo, hi = self.angle_limits_deg
            # Numerical roundoff at the equality boundary must not reject it.
            ea = bool(theta_min < lo-1e-10 or theta_max > hi+1e-10)
            if ea:
                reasons.append(f"extreme_angle_outside{self.angle_limits_deg}")
            rho = reference_bond_ratio(fr.symbols, coords, bonds) if reference_bonds is not None else float('nan')
            if self.max_bond_ratio is not None and rho > self.max_bond_ratio+1e-12:
                reasons.append(f'rho_max>{self.max_bond_ratio:g} (got {rho:.6g})')

            flags.append({
                "frame_index": idx,
                "keep": len(reasons) == 0,
                "reasons": ";".join(reasons),
                "min_dist": float(md),
                "extreme_angle": bool(ea),
                "theta_min": theta_min,
                "rho_max": rho,
                "reference_bond_count": len(bonds) if reference_bonds is not None else None,
                "rho_status": ('no_reference' if reference_bonds is None else
                               'no_reference_bonds' if not bonds else 'evaluated'),
            })

            if len(reasons) == 0:
                kept.append(fr)

        return kept, flags
