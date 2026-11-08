"""Geometric descriptors for finite, non-periodic molecular candidates."""
from itertools import combinations
import math
import numpy as np

# Cordero et al., Dalton Trans. (2008), 2832, doi:10.1039/B801115J.
# Carbon uses the sp3 value, consistently with the existing TSGen graph.
COVALENT_RADII = {
    'H': .31, 'C': .76, 'N': .71, 'O': .66, 'F': .57,
    'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
    'Br': 1.20, 'I': 1.39, 'Al': 1.21, 'Na': 1.66, 'K': 2.03,
}


def radii(symbols):
    unknown = sorted(set(symbols) - COVALENT_RADII.keys())
    if unknown:
        raise ValueError(f'No documented covalent radius for {unknown}; extend COVALENT_RADII explicitly')
    return np.array([COVALENT_RADII[s] for s in symbols])


def infer_reference_bonds(symbols, coords, bond_scale=1.25, max_bond_angstrom=2.25):
    """Infer a graph from a supplied parent geometry, never a stretched candidate for rho."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (len(symbols), 3) or not np.isfinite(coords).all():
        raise ValueError('Reference coordinates must be a finite N x 3 matrix')
    if not all(math.isfinite(v) and v > 0 for v in (bond_scale, max_bond_angstrom)):
        raise ValueError('Reference bond cutoffs must be finite and positive')
    rs = radii(symbols)
    return [(i, j) for i, j in combinations(range(len(symbols)), 2)
            if 1e-12 < np.linalg.norm(coords[i]-coords[j]) <= min(max_bond_angstrom, bond_scale*(rs[i]+rs[j]))]


def local_angle_range(coords, bonds):
    neighbors = [[] for _ in coords]
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    angles = []
    for j, adjacent in enumerate(neighbors):
        for i, k in combinations(adjacent, 2):
            a, b = coords[i]-coords[j], coords[k]-coords[j]
            norm = np.linalg.norm(a)*np.linalg.norm(b)
            if norm <= 1e-24:
                angles.append(0.)
            else:
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b)/norm, -1., 1.)))))
    return (min(angles), max(angles)) if angles else (math.nan, math.nan)


def reference_bond_ratio(symbols, coords, bonds):
    rs = radii(symbols)
    return max((float(np.linalg.norm(coords[i]-coords[j])/(rs[i]+rs[j])) for i, j in bonds), default=math.nan)
