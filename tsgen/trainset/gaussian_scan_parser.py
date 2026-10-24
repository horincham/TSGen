"""Dependency-light parsing helpers for Gaussian relaxed scan logs."""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from tsgen.data.xyz import Frame


def parse_completed_opt_records(text: str) -> list[tuple[float, Frame]]:
    """Return one converged energy/geometry record per Gaussian scan step.

    Gaussian commonly prints both ``Optimization completed`` and
    ``Stationary point found`` for one constrained optimization.  A record is
    emitted only when both a new SCF energy and orientation have appeared since the
    previous completion marker.
    """
    records: list[tuple[float, Frame]] = []
    lines = text.splitlines()
    last_e: Optional[float] = None
    last_fr: Optional[Frame] = None
    energy_since_record = False
    geometry_since_record = False
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.search(
            r"SCF Done:\s+E\([A-Za-z0-9]+\)\s*=\s*([-+]?\d+\.\d+)", line
        )
        if match:
            last_e = float(match.group(1))
            energy_since_record = True
        if "Standard orientation:" in line or "Input orientation:" in line:
            try:
                k = i + 5
                symbols: list[str] = []
                coordinates: list[list[float]] = []
                while k < len(lines):
                    current = lines[k]
                    if current.strip().startswith("-----"):
                        break
                    parts = current.split()
                    if len(parts) >= 6 and parts[0].isdigit():
                        symbols.append(_atomic_symbol(int(parts[1])))
                        coordinates.append(
                            [float(parts[3]), float(parts[4]), float(parts[5])]
                        )
                    k += 1
                if symbols and coordinates:
                    last_fr = Frame(
                        symbols=symbols,
                        coords=np.asarray(coordinates, dtype=float),
                        comment="source=gaussian_scan_step",
                    )
                    geometry_since_record = True
                i = k
            except (ValueError, IndexError):
                pass
        is_completion = (
            "Optimization completed" in line or "Stationary point found" in line
        )
        if is_completion and energy_since_record and geometry_since_record and last_e is not None and last_fr is not None:
            records.append((last_e, last_fr))
            energy_since_record = False
            geometry_since_record = False
        i += 1
    return records


def measure_scan_coordinate(
    frame: Frame,
    scan_type: str,
    atom_indices_1based: list[int] | tuple[int, ...],
) -> float:
    indices = [int(value) - 1 for value in atom_indices_1based]
    xyz = np.asarray(frame.coords, dtype=float)
    if scan_type == "bond" and len(indices) == 2:
        return float(np.linalg.norm(xyz[indices[0]] - xyz[indices[1]]))
    if scan_type == "angle" and len(indices) == 3:
        first = xyz[indices[0]] - xyz[indices[1]]
        second = xyz[indices[2]] - xyz[indices[1]]
        cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    if scan_type == "torsion" and len(indices) == 4:
        p0, p1, p2, p3 = (xyz[index] for index in indices)
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))
    raise ValueError(
        f"Unsupported {scan_type!r} coordinate with {len(indices)} atom indices"
    )


def validate_scan_record_alignment(
    records: list[tuple[float, Frame]],
    targets: list[float],
    scan_type: str,
    atom_indices_1based: list[int] | tuple[int, ...],
) -> list[float]:
    """Raise when a scan label does not match the exported geometry."""
    if len(records) != len(targets):
        raise ValueError(
            f"Gaussian scan produced {len(records)} converged records for "
            f"{len(targets)} requested targets"
        )
    tolerance = 0.03 if scan_type == "bond" else 1.0
    actual_values: list[float] = []
    mismatches: list[str] = []
    for index, (target, (_, frame)) in enumerate(zip(targets, records)):
        if not math.isfinite(float(target)) or not np.isfinite(frame.coords).all():
            raise ValueError(f'Nonfinite scan target/geometry at point {index}')
        actual = measure_scan_coordinate(frame, scan_type, atom_indices_1based)
        if not math.isfinite(actual):
            raise ValueError(f'Undefined scan coordinate at point {index}')
        actual_values.append(actual)
        if scan_type == "torsion":
            delta = abs(((actual - float(target) + 180.0) % 360.0) - 180.0)
        else:
            delta = abs(actual - float(target))
        if delta > tolerance:
            mismatches.append(
                f"point {index}: target={float(target):.6g}, geometry={actual:.6g}"
            )
    if mismatches:
        raise ValueError(
            "Gaussian scan labels are not aligned with constrained geometries: "
            + "; ".join(mismatches[:5])
        )
    return actual_values


def _atomic_symbol(number: int) -> str:
    symbols = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
    }
    return symbols.get(number, str(number))
