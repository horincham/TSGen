from __future__ import annotations

from typing import List
import numpy as np
from tsgen.data.xyz import Frame

Z_TABLE = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "Si": 14, "P": 15, "S": 16, "Cl": 17,
}

def coulomb_matrix_eigs(frame: Frame, max_atoms: int = 32) -> np.ndarray:
    syms = frame.symbols
    coords = frame.coords
    n = len(syms)
    if n > max_atoms:
        syms = syms[:max_atoms]
        coords = coords[:max_atoms]
        n = max_atoms

    Z = np.array([Z_TABLE.get(s, 0) for s in syms], dtype=float)
    M = np.zeros((max_atoms, max_atoms), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0.5 * (Z[i] ** 2.4)
            else:
                r = np.linalg.norm(coords[i] - coords[j]) + 1e-12
                M[i, j] = (Z[i] * Z[j]) / r

    eigs = np.linalg.eigvalsh(M)
    eigs = np.sort(eigs)[::-1]
    return eigs
