from __future__ import annotations

from typing import Sequence
import hashlib

import numpy as np

from tsgen.data.xyz import Frame
from tsgen.trainset.chem import (
    adjacency,
    angle_deg,
    connected_component,
    dihedral_deg,
    infer_bonds,
    min_nonbonded_distance,
    rotate_points,
)
from tsgen.trainset.config import ScanGenerationCfg
from tsgen.trainset.io import sanitize_id
from tsgen.trainset.reaxff_io import coordinate_label, format_scan_value, make_scan_step_id, sanitize_reaxff_id
from tsgen.trainset.models import CoordinateSpec, EquilibriumStructure, ScanPoint


def generate_scan_points(
    eqs: Sequence[EquilibriumStructure],
    coords: Sequence[CoordinateSpec],
    cfg: ScanGenerationCfg,
) -> tuple[list[ScanPoint], list[dict]]:
    """Generate Gaussian jobs for the selected scan coordinates.

    For ``relaxed_modredundant`` this deliberately generates *one Gaussian job per
    selected internal coordinate*, not one job per grid point.  The Gaussian input
    uses the canonical ModRedundant scan syntax after the Cartesian coordinates:

        i j [k [l]] S nsteps step

    The first geometry is moved to the requested start value, so Gaussian produces
    a continuous relaxed PES from start to stop.  This is the format expected by
    Gaussian/Sobereva-style redundant-coordinate relaxed scans and keeps job names
    meaningful at the PES-curve level.
    """
    eq_by_id = {e.parent_id: e for e in eqs}
    out: list[ScanPoint] = []
    audit: list[dict] = []

    # Equilibrium references remain single-point jobs; scan energies are reported
    # relative to these reference energies in trainset.in.
    for eq in eqs:
        # The equilibrium reference identifier must match DESCRP/trainset.in.
        spid = sanitize_reaxff_id(eq.parent_id)
        frame = Frame(
            symbols=list(eq.frame.symbols),
            coords=np.array(eq.frame.coords, dtype=float),
            comment=f"id={spid} parent={eq.parent_id} scan_type=equilibrium",
        )
        p = ScanPoint(
            id=spid,
            parent_id=eq.parent_id,
            coord_id="equilibrium",
            scan_type="equilibrium",
            family=eq.family,
            point_index=0,
            target_value=0.0,
            eq_value=0.0,
            atom_indices=tuple(),
            frame=frame,
        )
        p.reaxff_id = spid  # type: ignore[attr-defined]
        p.scan_prefix = eq.scan_prefix  # type: ignore[attr-defined]
        out.append(p)

    for c in coords:
        eq = eq_by_id[c.parent_id]
        if cfg.mode == "relaxed_modredundant":
            job = _generate_relaxed_scan_job(eq, c, cfg)
            if job is None:
                audit.append(_coord_audit(eq, c, "drop_cannot_generate_start_geometry", None))
                continue
            bonds = infer_bonds(job.frame)
            dmin = min_nonbonded_distance(job.frame, bonds)
            if dmin < float(cfg.min_nonbonded_distance):
                audit.append(_point_audit(job, "drop_min_nonbonded_distance", dmin))
                continue
            out.append(job)
            audit.append(_point_audit(job, "keep_relaxed_modredundant_scan_job", dmin))
        else:
            # Retain legacy rigid-point generation for debugging/fallback only.
            if c.scan_type == "bond":
                points = _generate_bond_points(eq, c, cfg)
            elif c.scan_type == "angle":
                points = _generate_angle_points(eq, c, cfg)
            elif c.scan_type == "torsion":
                points = _generate_torsion_points(eq, c, cfg)
            else:
                points = []
            for p in points:
                bonds = infer_bonds(p.frame)
                dmin = min_nonbonded_distance(p.frame, bonds)
                if dmin < float(cfg.min_nonbonded_distance):
                    audit.append(_point_audit(p, "drop_min_nonbonded_distance", dmin))
                    continue
                out.append(p)
                audit.append(_point_audit(p, "keep_rigid_point", dmin))
    return out, audit


def scan_grid_values(scan_type: str, cfg: ScanGenerationCfg, family: str | None = None) -> list[float]:
    if scan_type == "bond":
        start, stop, step = _bond_range_for_family(cfg, family)
        return _grid(start, stop, step, include_endpoint=True)
    if scan_type == "angle":
        return _grid(cfg.angle.start_deg, cfg.angle.stop_deg, cfg.angle.step_deg, include_endpoint=True)
    if scan_type == "torsion":
        return _grid(cfg.torsion.start_deg, cfg.torsion.stop_deg, cfg.torsion.step_deg, include_endpoint=cfg.torsion.include_endpoint)
    return []


def scan_start_stop_step_nsteps(scan_type: str, cfg: ScanGenerationCfg, family: str | None = None) -> tuple[float, float, float, int, list[float]]:
    vals = scan_grid_values(scan_type, cfg, family=family)
    if len(vals) < 2:
        raise ValueError(f"{scan_type} scan grid must contain at least two values")
    step = float(vals[1] - vals[0])
    # In Gaussian ModRedundant S syntax, nsteps is the number of increments after
    # the initial geometry, so total saved scan points should be nsteps + 1.
    nsteps = len(vals) - 1
    return float(vals[0]), float(vals[-1]), step, int(nsteps), [float(v) for v in vals]


def _generate_relaxed_scan_job(eq: EquilibriumStructure, c: CoordinateSpec, cfg: ScanGenerationCfg) -> ScanPoint | None:
    start, stop, step, nsteps, vals = scan_start_stop_step_nsteps(c.scan_type, cfg, family=c.family)
    if c.scan_type == "bond":
        fr = _set_bond(eq.frame, c.atom_indices, start)
    elif c.scan_type == "angle":
        fr = _set_angle(eq.frame, c.atom_indices, start)
    elif c.scan_type == "torsion":
        fr = _set_torsion(eq.frame, c.atom_indices, start)
    else:
        return None
    if fr is None:
        return None
    jid = _scan_job_id(c, eq, start, stop, step)
    fr = Frame(
        symbols=list(fr.symbols),
        coords=np.array(fr.coords, dtype=float),
        comment=(
            f"id={jid} parent={eq.parent_id} coord={c.coord_id} scan_type={c.scan_type} "
            f"family={sanitize_id(c.family)} start={start:.6f} stop={stop:.6f} step={step:.6f} nsteps={nsteps}"
        ),
    )
    p = ScanPoint(
        id=jid,
        parent_id=eq.parent_id,
        coord_id=c.coord_id,
        scan_type=c.scan_type,
        family=c.family,
        point_index=0,
        target_value=float(start),
        eq_value=float(c.eq_value),
        atom_indices=c.atom_indices,
        frame=fr,
    )
    # Dynamic attributes are intentionally attached here so older dataclass rows
    # and CSV writers stay backward-compatible.
    p.scan_start = float(start)  # type: ignore[attr-defined]
    p.scan_stop = float(stop)  # type: ignore[attr-defined]
    p.scan_step = float(step)  # type: ignore[attr-defined]
    p.scan_nsteps = int(nsteps)  # type: ignore[attr-defined]
    p.scan_values = vals  # type: ignore[attr-defined]
    p.scan_prefix = _scan_reaxff_prefix(c, eq)  # type: ignore[attr-defined]
    p.scan_group_comment = _scan_group_comment(c, eq)  # type: ignore[attr-defined]
    p.scan_unit = "A" if c.scan_type == "bond" else "deg"  # type: ignore[attr-defined]
    return p


def _grid(start: float, stop: float, step: float, include_endpoint: bool = True) -> list[float]:
    if step <= 0:
        raise ValueError("scan step must be positive")
    vals: list[float] = []
    x = float(start)
    stop = float(stop)
    eps = abs(step) * 1e-7
    if include_endpoint:
        while x <= stop + eps:
            vals.append(round(float(x), 10))
            x += float(step)
    else:
        while x < stop - eps:
            vals.append(round(float(x), 10))
            x += float(step)
    return vals


def _set_bond(frame: Frame, atoms: tuple[int, ...], target: float) -> Frame | None:
    i, j = atoms
    coords = np.array(frame.coords, dtype=float)
    bonds = infer_bonds(frame)
    adj = adjacency(len(frame.symbols), bonds, skip=(i, j))
    side_j = connected_component(adj, j)
    side_i = connected_component(adj, i)
    moving = side_j if len(side_j) <= len(side_i) else side_i
    anchor = i if moving is side_j else j
    moving_atom = j if moving is side_j else i
    direction = coords[moving_atom] - coords[anchor]
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
    direction = direction / norm
    current = float(np.linalg.norm(coords[moving_atom] - coords[anchor]))
    delta = float(target) - current
    new_coords = np.array(coords, copy=True)
    for a in moving:
        new_coords[a] += delta * direction
    return Frame(symbols=list(frame.symbols), coords=new_coords, comment=frame.comment)


def _set_angle(frame: Frame, atoms: tuple[int, ...], target: float) -> Frame | None:
    i, j, k = atoms
    coords = np.array(frame.coords, dtype=float)
    eq_ang = angle_deg(coords, i, j, k)
    bonds = infer_bonds(frame)
    adj = adjacency(len(frame.symbols), bonds, skip=(j, k))
    moving = connected_component(adj, k)
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    axis = np.cross(v2, v1)
    if np.linalg.norm(axis) < 1e-10:
        axis = _orthogonal_axis(v2)
    delta = np.radians(float(target) - float(eq_ang))
    c1 = rotate_points(coords, moving, coords[j], axis, delta)
    c2 = rotate_points(coords, moving, coords[j], axis, -delta)
    a1 = abs(angle_deg(c1, i, j, k) - target)
    a2 = abs(angle_deg(c2, i, j, k) - target)
    return Frame(symbols=list(frame.symbols), coords=(c1 if a1 <= a2 else c2), comment=frame.comment)


def _set_torsion(frame: Frame, atoms: tuple[int, ...], target: float) -> Frame | None:
    i, j, k, l = atoms
    coords = np.array(frame.coords, dtype=float)
    eq_tor = dihedral_deg(coords, i, j, k, l)
    bonds = infer_bonds(frame)
    adj = adjacency(len(frame.symbols), bonds, skip=(j, k))
    moving = connected_component(adj, k)
    axis = coords[k] - coords[j]
    origin = coords[j]
    delta = _angle_delta_deg(float(target), float(eq_tor))
    c1 = rotate_points(coords, moving, origin, axis, np.radians(delta))
    c2 = rotate_points(coords, moving, origin, axis, np.radians(-delta))
    a1 = abs(_angle_delta_deg(dihedral_deg(c1, i, j, k, l), target))
    a2 = abs(_angle_delta_deg(dihedral_deg(c2, i, j, k, l), target))
    return Frame(symbols=list(frame.symbols), coords=(c1 if a1 <= a2 else c2), comment=frame.comment)


def _bond_range_for_family(cfg: ScanGenerationCfg, family: str | None) -> tuple[float, float, float]:
    if bool(getattr(cfg.bond, "use_family_ranges", False)) and family:
        ranges = getattr(cfg.bond, "family_ranges", {}) or {}
        item = ranges.get(family)
        if item is not None:
            try:
                return float(item.get("start", cfg.bond.start)), float(item.get("stop", cfg.bond.stop)), float(item.get("step", cfg.bond.step))
            except Exception:
                pass
    return float(cfg.bond.start), float(cfg.bond.stop), float(cfg.bond.step)


# Legacy rigid point helpers retained for backward compatibility/debugging.
def _generate_bond_points(eq: EquilibriumStructure, c: CoordinateSpec, cfg: ScanGenerationCfg) -> list[ScanPoint]:
    vals = scan_grid_values("bond", cfg, family=c.family)
    pts = []
    for idx, v in enumerate(vals, start=1):
        fr = _set_bond(eq.frame, c.atom_indices, v)
        if fr is None:
            continue
        pid = _scan_point_id("bnd", eq.parent_id, c.coord_id, idx, v)
        pts.append(ScanPoint(pid, eq.parent_id, c.coord_id, "bond", c.family, idx, float(v), c.eq_value, c.atom_indices, fr))
    return pts


def _generate_angle_points(eq: EquilibriumStructure, c: CoordinateSpec, cfg: ScanGenerationCfg) -> list[ScanPoint]:
    vals = scan_grid_values("angle", cfg)
    pts = []
    for idx, v in enumerate(vals, start=1):
        fr = _set_angle(eq.frame, c.atom_indices, v)
        if fr is None:
            continue
        pid = _scan_point_id("ang", eq.parent_id, c.coord_id, idx, v)
        pts.append(ScanPoint(pid, eq.parent_id, c.coord_id, "angle", c.family, idx, float(v), c.eq_value, c.atom_indices, fr))
    return pts


def _generate_torsion_points(eq: EquilibriumStructure, c: CoordinateSpec, cfg: ScanGenerationCfg) -> list[ScanPoint]:
    vals = scan_grid_values("torsion", cfg)
    pts = []
    for idx, v in enumerate(vals, start=1):
        fr = _set_torsion(eq.frame, c.atom_indices, v)
        if fr is None:
            continue
        pid = _scan_point_id("tor", eq.parent_id, c.coord_id, idx, v)
        pts.append(ScanPoint(pid, eq.parent_id, c.coord_id, "torsion", c.family, idx, float(v), c.eq_value, c.atom_indices, fr))
    return pts


def _scan_job_id(c: CoordinateSpec, eq: EquilibriumStructure, start: float, stop: float, step: float) -> str:
    # Job directory names are filesystem-safe, but still human-readable.  The
    # final ReaxFF DESCRP identifiers for individual scan points are generated
    # separately as e.g. C9H18_01_C1_C2_1.50.
    atoms = coordinate_label(eq.frame.symbols, c.atom_indices, with_indices=True)
    prefix = sanitize_reaxff_id(eq.scan_prefix or eq.formula, max_len=32)
    rng = f"{_fmt_value(start)}_to_{_fmt_value(stop)}_by_{_fmt_value(step)}"
    base = f"scan_{c.scan_type}_{prefix}_{atoms}_{rng}"
    return sanitize_id(base, max_len=120)


def _scan_reaxff_prefix(c: CoordinateSpec, eq: EquilibriumStructure) -> str:
    atoms = coordinate_label(eq.frame.symbols, c.atom_indices, with_indices=True)
    prefix = sanitize_reaxff_id(eq.scan_prefix or eq.formula, max_len=32)
    return sanitize_reaxff_id(f"{prefix}_{atoms}", max_len=72)


def _scan_group_comment(c: CoordinateSpec, eq: EquilibriumStructure) -> str:
    atoms = "-".join(coordinate_label(eq.frame.symbols, c.atom_indices, with_indices=True).split("_"))
    label = "torsion" if c.scan_type == "torsion" else c.scan_type
    return f"{eq.scan_prefix or eq.formula} {label} {atoms} ({c.family})"

def _scan_point_id(prefix: str, parent_id: str, coord_id: str, idx: int, target: float) -> str:
    h = hashlib.sha1(f"{parent_id}|{coord_id}|{idx}|{target}".encode("utf-8")).hexdigest()[:8]
    p = sanitize_id(parent_id, max_len=30)
    return sanitize_id(f"{prefix}_{p}_{_fmt_value(target)}_p{idx:02d}_{h}", max_len=90)


def _short_parent(parent_id: str) -> str:
    parts = sanitize_id(parent_id, max_len=80).split("_")
    if len(parts) >= 3:
        return sanitize_id("_".join(parts[:3]), max_len=32)
    return sanitize_id(parent_id, max_len=32)


def _fmt_value(x: float) -> str:
    s = f"{float(x):.2f}".replace("-", "m").replace(".", "p")
    return s


def _angle_delta_deg(target: float, current: float) -> float:
    return ((target - current + 180.0) % 360.0) - 180.0


def _orthogonal_axis(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if abs(v[0]) < abs(v[1]):
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    return np.cross(v, a)


def _point_audit(p: ScanPoint, decision: str, min_nb: float | None) -> dict:
    return {
        "id": p.id,
        "parent_id": p.parent_id,
        "coord_id": p.coord_id,
        "scan_type": p.scan_type,
        "family": p.family,
        "point_index": p.point_index,
        "scan_start": getattr(p, "scan_start", p.target_value),
        "scan_stop": getattr(p, "scan_stop", None),
        "scan_step": getattr(p, "scan_step", None),
        "scan_nsteps": getattr(p, "scan_nsteps", None),
        "eq_value": p.eq_value,
        "atom_indices_1based": " ".join(str(a + 1) for a in p.atom_indices),
        "min_nonbonded_distance": min_nb,
        "decision": decision,
    }


def _coord_audit(eq: EquilibriumStructure, c: CoordinateSpec, decision: str, min_nb: float | None) -> dict:
    return {
        "parent_id": eq.parent_id,
        "coord_id": c.coord_id,
        "scan_type": c.scan_type,
        "family": c.family,
        "atom_indices_1based": " ".join(str(a + 1) for a in c.atom_indices),
        "eq_value": c.eq_value,
        "min_nonbonded_distance": min_nb,
        "decision": decision,
    }
