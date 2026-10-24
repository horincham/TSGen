from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tsgen.data.xyz import Frame


@dataclass
class EquilibriumStructure:
    parent_id: str
    source_path: str
    formula: str
    species_id: str
    family: str
    frame: Frame
    charge: int = 0
    multiplicity: int = 1
    n_electrons: int = 0
    scan_prefix: str = ""


@dataclass
class CoordinateSpec:
    coord_id: str
    parent_id: str
    scan_type: str  # bond|angle|torsion
    family: str
    atom_indices: Tuple[int, ...]  # 0-based internal
    atom_symbols: Tuple[str, ...]
    eq_value: float
    priority: float = 0.0
    reason: str = "selected"


@dataclass
class ScanPoint:
    id: str
    parent_id: str
    coord_id: str
    scan_type: str  # equilibrium|bond|angle|torsion
    family: str
    point_index: int
    target_value: float
    eq_value: float
    atom_indices: Tuple[int, ...]
    frame: Frame
    status: str = "planned"
    job_dir: str = ""
    jobid: str = ""
    e_hartree: Optional[float] = None
    e_kcalmol_abs: Optional[float] = None
    e_kcalmol_rel: Optional[float] = None
    parse_note: str = ""

