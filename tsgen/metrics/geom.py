from __future__ import annotations

"""Geometric metrics for publication-grade observability.

This module exists to make Panel C of Figure 3 reproducible:
reaction-coordinate values must be written explicitly as numeric columns.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from tsgen.data.xyz import Frame
from tsgen.config import ReactionCoordsConfig


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a


def bond_length(fr: Frame, i: int, j: int) -> float:
    r = fr.coords[j] - fr.coords[i]
    return float(np.linalg.norm(r))


def angle_deg(fr: Frame, i: int, j: int, k: int) -> float:
    # angle i-j-k at vertex j
    v1 = _vec(fr.coords[j], fr.coords[i])
    v2 = _vec(fr.coords[j], fr.coords[k])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def dihedral_deg(fr: Frame, i: int, j: int, k: int, l: int) -> float:
    # standard torsion i-j-k-l
    p0, p1, p2, p3 = fr.coords[i], fr.coords[j], fr.coords[k], fr.coords[l]
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1n = np.linalg.norm(b1)
    if b1n == 0.0:
        return float("nan")
    b1u = b1 / b1n

    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u

    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)
    return float(np.degrees(np.arctan2(y, x)))


@dataclass
class GeomMetricsResult:
    columns: List[str]
    rows: List[Dict[str, float]]


def compute_geom_metrics(frames: List[Frame], ids: List[str], it: int, rcfg: ReactionCoordsConfig) -> GeomMetricsResult:
    """Compute user-defined reaction-coordinate values for each frame.

    Parameters
    ----------
    frames / ids
        Must be aligned 1:1.
    it
        Iteration index (recorded as a column).
    rcfg
        Reaction coordinate definitions from config.
    """
    if len(frames) != len(ids):
        raise ValueError("frames and ids must be aligned")

    cols: List[str] = ["id", "iter"]
    # preserve user-provided names (stable column identifiers for plotting)
    for b in rcfg.bonds:
        cols.append(b.name)
    for a in rcfg.angles:
        cols.append(a.name)
    for d in rcfg.dihedrals:
        cols.append(d.name)

    out_rows: List[Dict[str, float]] = []
    for fr, cid in zip(frames, ids):
        r: Dict[str, float] = {"id": cid, "iter": float(it)}
        for b in rcfg.bonds:
            r[b.name] = bond_length(fr, b.i, b.j)
        for a in rcfg.angles:
            r[a.name] = angle_deg(fr, a.i, a.j, a.k)
        for d in rcfg.dihedrals:
            r[d.name] = dihedral_deg(fr, d.i, d.j, d.k, d.l)
        out_rows.append(r)

    return GeomMetricsResult(columns=cols, rows=out_rows)
