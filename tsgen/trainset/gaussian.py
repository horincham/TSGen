from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from tsgen.data.xyz import Frame, write_xyz_multiframe
from tsgen.trainset.chem import ATOMIC_NUMBERS, resolve_charge_multiplicity
from tsgen.trainset.config import ExecutionCfg, GaussianCfg
from tsgen.trainset.models import ScanPoint
from tsgen.trainset.reaxff_io import make_scan_step_id, sanitize_reaxff_id
from tsgen.trainset.gaussian_scan_parser import (
    parse_completed_opt_records,
    validate_scan_record_alignment,
)

HARTREE_TO_KCALMOL = 627.5094740631
ATOMIC_SYMBOLS = {v: k for k, v in ATOMIC_NUMBERS.items()}


class GaussianScanRunner:
    def __init__(self, cfg: GaussianCfg, scan_mode: str = "relaxed_modredundant"):
        self.cfg = cfg
        self.scan_mode = scan_mode

    def write_job(self, job_dir: Path, point: ScanPoint) -> None:
        job_dir.mkdir(parents=True, exist_ok=True)
        write_xyz_multiframe(job_dir / "input_guess.xyz", [point.frame])
        q, m, ne = resolve_charge_multiplicity(point.frame, self.cfg.charge, self.cfg.multiplicity)
        self._write_gjf(job_dir / "input.gjf", point, q, m)
        if self.cfg.mode == "slurm":
            self._write_slurm(job_dir)
        scan_values = getattr(point, "scan_values", None)
        meta = {
            "id": point.id,
            "parent_id": point.parent_id,
            "coord_id": point.coord_id,
            "scan_type": point.scan_type,
            "family": point.family,
            "target_value": point.target_value,
            "eq_value": point.eq_value,
            "atom_indices": list(point.atom_indices),
            "atom_indices_1based": [a + 1 for a in point.atom_indices],
            "modredundant_scan_line": _scan_line(point) if point.scan_type != "equilibrium" and self.scan_mode == "relaxed_modredundant" else "",
            "scan_start": getattr(point, "scan_start", None),
            "scan_stop": getattr(point, "scan_stop", None),
            "scan_step": getattr(point, "scan_step", None),
            "scan_nsteps": getattr(point, "scan_nsteps", None),
            "scan_values": scan_values,
            "scan_prefix": getattr(point, "scan_prefix", ""),
            "scan_group_comment": getattr(point, "scan_group_comment", ""),
            "scan_unit": getattr(point, "scan_unit", ""),
            "reaxff_id": getattr(point, "reaxff_id", point.id),
            "expected_points": (None if point.scan_type == "equilibrium" else (len(scan_values) if isinstance(scan_values, list) else None)),
            "route": self.cfg.reference_route if point.scan_type == "equilibrium" else self.cfg.route,
            "charge": q,
            "multiplicity": m,
            "n_electrons": ne,
            "mode": self.cfg.mode,
            "scan_mode": self.scan_mode,
            "nproc": self.cfg.nproc,
            "mem": self.cfg.mem,
        }
        (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _write_gjf(self, path: Path, point: ScanPoint, charge: int, multiplicity: int) -> None:
        frame = point.frame
        route = self.cfg.reference_route if point.scan_type == "equilibrium" else self.cfg.route
        if self.scan_mode == "relaxed_modredundant" and point.scan_type != "equilibrium" and "modredundant" not in route.lower():
            route = route.rstrip() + " Opt=ModRedundant"
        lines = [f"%nprocshared={int(self.cfg.nproc)}"]
        if self.cfg.mem:
            lines.append(f"%mem={self.cfg.mem}")
        if getattr(self.cfg, "write_chk", True):
            lines.append(f"%chk={getattr(self.cfg, 'chk_name', 'input.chk')}")
        lines.append(str(route).rstrip())
        title = f"TSGen ReaxFF relaxed ModRedundant scan {point.id}"
        if point.scan_type == "equilibrium":
            title = f"TSGen ReaxFF equilibrium reference {point.id}"
        lines.extend(["", title, "", f"{int(charge)} {int(multiplicity)}"])
        for sym, (x, y, z) in zip(frame.symbols, frame.coords):
            lines.append(f"{sym:2s}  {x: .10f}  {y: .10f}  {z: .10f}")
        lines.append("")
        if self.scan_mode == "relaxed_modredundant" and point.scan_type != "equilibrium":
            scan = _scan_line(point)
            if scan:
                # Gaussian relaxed PES scan in redundant internal coordinates.
                # Syntax: N1 N2 [N3 [N4]] S nsteps stepsize
                lines.append(scan)
                lines.append("")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_slurm(self, job_dir: Path) -> None:
        scfg = self.cfg.slurm
        lines = ["#!/bin/bash"]
        if scfg.partition:
            lines.append(f"#SBATCH -p {scfg.partition}")
        lines.append(f"#SBATCH -J tsgen_tr_{job_dir.name[:40]}")
        lines.append("#SBATCH -o gaussian.out")
        lines.append("#SBATCH -e gaussian.err")
        if scfg.time:
            lines.append(f"#SBATCH -t {scfg.time}")
        if scfg.cpus_per_task:
            lines.append(f"#SBATCH --cpus-per-task={int(scfg.cpus_per_task)}")
        if scfg.mem:
            lines.append(f"#SBATCH --mem={scfg.mem}")
        lines.append("")
        lines.extend(scfg.extra_lines or [])
        if scfg.extra_lines:
            lines.append("")
        lines.append(f"{self.cfg.g16_cmd} < input.gjf > input.log")
        if getattr(self.cfg, "write_chk", True) and getattr(self.cfg, "auto_formchk", True):
            formchk = str(getattr(self.cfg, "formchk_cmd", "formchk"))
            chk = str(getattr(self.cfg, "chk_name", "input.chk"))
            lines.append(f"if command -v {formchk} >/dev/null 2>&1; then {formchk} {chk} input.fchk || true; fi")
        lines.append("")
        (job_dir / "submit.slurm").write_text("\n".join(lines), encoding="utf-8")

    def submit(self, job_dir: Path) -> Optional[str]:
        if self.cfg.mode == "manual":
            return None
        if self.cfg.mode == "local":
            proc = subprocess.run(
                f"{self.cfg.g16_cmd} < input.gjf > input.log",
                cwd=str(job_dir), shell=True, capture_output=True, text=True,
            )
            (job_dir / "local.stdout").write_text(proc.stdout or "", encoding="utf-8")
            (job_dir / "local.stderr").write_text(proc.stderr or "", encoding="utf-8")
            self._try_formchk(job_dir)
            return None
        proc = subprocess.run(["sbatch", "submit.slurm"], cwd=str(job_dir), capture_output=True, text=True)
        (job_dir / "sbatch.stdout").write_text(proc.stdout or "", encoding="utf-8")
        (job_dir / "sbatch.stderr").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            return None
        m = re.search(r"Submitted batch job\s+(\d+)", proc.stdout or "")
        return m.group(1) if m else None

    def _try_formchk(self, job_dir: Path) -> None:
        if not (getattr(self.cfg, "write_chk", True) and getattr(self.cfg, "auto_formchk", True)):
            return
        chk = job_dir / str(getattr(self.cfg, "chk_name", "input.chk"))
        if not chk.exists():
            return
        cmd = str(getattr(self.cfg, "formchk_cmd", "formchk"))
        exe = shutil.which(cmd) or (cmd if Path(cmd).exists() else None)
        if exe is None:
            return
        proc = subprocess.run([exe, chk.name, "input.fchk"], cwd=str(job_dir), capture_output=True, text=True)
        (job_dir / "formchk.stdout").write_text(proc.stdout or "", encoding="utf-8")
        (job_dir / "formchk.stderr").write_text(proc.stderr or "", encoding="utf-8")


def _fmt_gaussian_float(x: float) -> str:
    """Format numeric ModRedundant step sizes with an explicit decimal point.

    Some Gaussian builds reject integer-looking angle/torsion step sizes such as
    ``10`` or ``30`` in ``S n step`` scan lines.  ``10.0`` and ``30.0`` are
    accepted consistently, while non-integer steps remain compact, e.g. ``0.25``.
    """
    s = f"{float(x):.8f}".rstrip("0").rstrip(".")
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


def _scan_line(point: ScanPoint) -> str:
    atoms = [a + 1 for a in point.atom_indices]
    nsteps = int(getattr(point, "scan_nsteps", 0) or 0)
    step = float(getattr(point, "scan_step", 0.0) or 0.0)
    if nsteps <= 0 or abs(step) < 1e-12:
        return ""
    step_s = _fmt_gaussian_float(step)
    # Sobereva/Gaussian ModRedundant scan syntax in redundant internal coordinates:
    # atom1 atom2 [atom3 [atom4]] S steps stepsize.  Do not use freeze lines here.
    if point.scan_type == "bond" and len(atoms) == 2:
        return f"{atoms[0]} {atoms[1]} S {nsteps:d} {step_s}"
    if point.scan_type == "angle" and len(atoms) == 3:
        return f"{atoms[0]} {atoms[1]} {atoms[2]} S {nsteps:d} {step_s}"
    if point.scan_type == "torsion" and len(atoms) == 4:
        return f"{atoms[0]} {atoms[1]} {atoms[2]} {atoms[3]} S {nsteps:d} {step_s}"
    return ""


def job_status(job_dir: Path) -> tuple[str, Optional[float], str]:
    log = job_dir / "input.log"
    if not log.exists():
        return "pending", None, "missing input.log"
    try:
        text = log.read_text(errors="ignore")
    except Exception as exc:
        return "pending", None, f"cannot read log: {exc}"
    e = parse_scf_energy_hartree(text)
    if "Normal termination" in text and e is not None:
        fr = parse_final_orientation_frame(text, job_dir)
        if fr is not None:
            write_xyz_multiframe(job_dir / "optimized_structure.xyz", [fr])
        # If this is a ModRedundant scan, also expand individual scan-step structures/energies.
        try:
            rows = expand_scan_job(job_dir)
            if rows:
                return "ok", e, f"normal termination; relaxed scan points parsed={len(rows)}"
        except ValueError as exc:
            return "failed", e, f"normal termination; scan validation failed: {exc}"
        if fr is not None:
            return "ok", e, "normal termination; optimized geometry parsed"
        inp = job_dir / "input_guess.xyz"
        if inp.exists():
            return "ok", e, "normal termination; no orientation parsed, input geometry retained"
        return "ok", e, "normal termination; no geometry parsed"
    if "Error termination" in text:
        return "failed", e, _diagnose_gaussian_error(text)
    if e is not None:
        return "running", e, "energy found but no normal termination"
    return "pending", None, "no parsed energy"


def _diagnose_gaussian_error(text: str) -> str:
    patterns = [
        ("Wanted a floating point number", "Gaussian ModRedundant input format error: wanted floating point number"),
        ("FormBX had a problem", "Gaussian internal-coordinate/FormBX failure"),
        ("Linear angle in Bend", "Gaussian redundant-coordinate failure: linear angle in Bend"),
        ("l9999", "Gaussian optimization did not converge (l9999)"),
        ("Optimization stopped", "Gaussian optimization stopped before convergence"),
        ("Maximum number of optimization cycles", "Gaussian optimization reached max cycles"),
        ("Error termination", "Gaussian error termination"),
    ]
    for pat, msg in patterns:
        if pat.lower() in text.lower():
            return msg
    # Last non-empty lines are often the most useful clue.
    tail = [ln.strip() for ln in text.splitlines()[-20:] if ln.strip()]
    return "Gaussian error termination" + ((": " + tail[-1]) if tail else "")


def parse_scf_energy_hartree(text: str) -> Optional[float]:
    matches = re.findall(r"SCF Done:\s+E\([A-Za-z0-9]+\)\s*=\s*([-+]?\d+\.\d+)", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def parse_final_orientation_frame(text: str, job_dir: Optional[Path] = None) -> Optional[Frame]:
    blocks = parse_orientation_blocks(text)
    if not blocks:
        return None
    fr = blocks[-1]
    cid = job_dir.name if job_dir is not None else "gaussian_final"
    return Frame(symbols=fr.symbols, coords=fr.coords, comment=f"id={cid} source=gaussian_final_orientation")


def parse_orientation_blocks(text: str) -> list[Frame]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if "Standard orientation:" in line or "Input orientation:" in line]
    frames: list[Frame] = []
    for start in starts:
        try:
            k = start + 5
            symbols: list[str] = []
            coords: list[list[float]] = []
            while k < len(lines):
                line = lines[k]
                if line.strip().startswith("-----"):
                    break
                parts = line.split()
                if len(parts) >= 6 and parts[0].isdigit():
                    anum = int(parts[1])
                    sym = ATOMIC_SYMBOLS.get(anum, str(anum))
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    symbols.append(sym)
                    coords.append([x, y, z])
                k += 1
            if symbols and coords:
                frames.append(Frame(symbols=symbols, coords=np.array(coords, dtype=float), comment="source=gaussian_orientation"))
        except Exception:
            continue
    return frames


def _parse_completed_opt_records(text: str) -> list[tuple[float, Frame]]:
    """Best-effort parser for relaxed scan points in Gaussian optimization logs.

    We record the last SCF energy and last orientation seen before each optimization
    completion marker.  This is more appropriate than taking every SCF energy because
    one constrained optimization step usually contains many SCF cycles.
    """
    return parse_completed_opt_records(text)


def expand_scan_job(job_dir: Path) -> list[dict]:
    meta_p = job_dir / "meta.json"
    log_p = job_dir / "input.log"
    if not meta_p.exists() or not log_p.exists():
        return []
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("scan_type") == "equilibrium":
        return []
    values = meta.get("scan_values") or []
    if not isinstance(values, list) or not values:
        return []
    text = log_p.read_text(errors="ignore")
    if "Normal termination" not in text:
        return []
    records = _parse_completed_opt_records(text)
    n = len(values)
    # SCF cycles and Input/Standard orientation blocks are NOT scan points.
    # Never zip their tails as a fallback: one optimization prints many of each.
    if len(records) != n:
        raise ValueError(
            f"Gaussian scan produced {len(records)} converged records; expected {n}. "
            "Partial or over-complete scans are not exported to ReaxFF."
        )
    validate_scan_record_alignment(
        records,
        [float(value) for value in values],
        str(meta.get("scan_type") or ""),
        [int(value) for value in meta.get("atom_indices_1based", [])],
    )
    step_dir = job_dir / "scan_step_structures"
    step_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, (target, rec) in enumerate(zip(values, records), start=0):
        e, fr = rec
        step_id = _step_id_from_meta(meta, float(target), idx)
        fr = Frame(symbols=list(fr.symbols), coords=np.array(fr.coords, dtype=float), comment=f"id={step_id} parent={meta.get('parent_id')} source={meta['id']} target={float(target):.6f}")
        xyz = step_dir / f"{_file_safe_step_id(step_id)}.xyz"
        write_xyz_multiframe(xyz, [fr])
        rows.append({
            "id": step_id,
            "reaxff_id": step_id,
            "scan_job_id": meta.get("id"),
            "parent_id": meta.get("parent_id"),
            "coord_id": meta.get("coord_id"),
            "scan_type": meta.get("scan_type"),
            "family": meta.get("family"),
            "scan_prefix": meta.get("scan_prefix", ""),
            "scan_group_comment": meta.get("scan_group_comment", ""),
            "scan_unit": meta.get("scan_unit", ""),
            "point_index": idx,
            "target_value": float(target),
            "eq_value": meta.get("eq_value"),
            "atom_indices_1based": " ".join(str(x) for x in meta.get("atom_indices_1based", [])),
            "job_dir": str(job_dir),
            "status": "ok",
            "e_hartree": float(e),
            "e_kcalmol_abs": float(e) * HARTREE_TO_KCALMOL,
            "has_optimized_geometry": True,
            "optimized_xyz": str(xyz),
            "parse_note": "relaxed ModRedundant scan step parsed",
        })
    (job_dir / "scan_points.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _step_id_from_meta(meta: dict, target: float, idx: int) -> str:
    prefix = str(meta.get("scan_prefix") or meta.get("id") or "scan")
    scan_type = str(meta.get("scan_type") or "")
    return make_scan_step_id(prefix, float(target), scan_type)


def _file_safe_step_id(step_id: str) -> str:
    return sanitize_reaxff_id(step_id).replace(".", "p")


def collect_job_table(jobs_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not jobs_dir.exists():
        return rows
    for jd in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        meta_p = jd / "meta.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        else:
            meta = {"id": jd.name}
        status, e, note = job_status(jd)
        row = dict(meta)
        opt_xyz = jd / "optimized_structure.xyz"
        row.update({
            "job_dir": str(jd),
            "status": status,
            "e_hartree": e,
            "e_kcalmol_abs": None if e is None else e * HARTREE_TO_KCALMOL,
            "has_optimized_geometry": opt_xyz.exists(),
            "optimized_xyz": str(opt_xyz) if opt_xyz.exists() else "",
            "parse_note": note,
        })
        sb = jd / "sbatch.stdout"
        if sb.exists():
            m = re.search(r"Submitted batch job\s+(\d+)", sb.read_text(errors="ignore"))
            if m:
                row["jobid"] = m.group(1)
        rows.append(row)
    return rows


def collect_scan_point_table(jobs_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not jobs_dir.exists():
        return rows
    for jd in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        meta_p = jd / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("scan_type") == "equilibrium":
            status, e, note = job_status(jd)
            opt_xyz = jd / "optimized_structure.xyz"
            if not opt_xyz.exists():
                opt_xyz = jd / "input_guess.xyz"
            rows.append({
                **meta,
                "reaxff_id": meta.get("reaxff_id", meta.get("id")),
                "job_dir": str(jd),
                "status": status,
                "e_hartree": e,
                "e_kcalmol_abs": None if e is None else e * HARTREE_TO_KCALMOL,
                "has_optimized_geometry": opt_xyz.exists(),
                "optimized_xyz": str(opt_xyz) if opt_xyz.exists() else "",
                "parse_note": note,
            })
        else:
            try:
                rows.extend(expand_scan_job(jd))
            except ValueError as exc:
                # One incomplete/misaligned scan must not corrupt or block the
                # valid scans. Preserve an explicit non-exportable audit row.
                rows.append({
                    **meta,
                    'job_dir': str(jd),
                    'status': 'invalid_scan',
                    'e_hartree': None,
                    'e_kcalmol_abs': None,
                    'has_optimized_geometry': False,
                    'optimized_xyz': '',
                    'parse_note': f'scan validation failed: {exc}',
                })
    return rows


def wait_for_jobs(jobs_dir: Path, cfg: ExecutionCfg, echo=print) -> None:
    start = time.time()
    while True:
        rows = collect_job_table(jobs_dir)
        n = len(rows)
        ok = sum(1 for r in rows if r.get("status") == "ok")
        failed = sum(1 for r in rows if r.get("status") == "failed")
        pending = n - ok - failed
        echo(f"Trainset Gaussian wait: ok={ok}, failed={failed}, pending/running={pending}, total={n}")
        if n == 0:
            return
        ok_frac = ok / max(1, n)
        if cfg.require_all_ok and ok == n:
            return
        if (not cfg.require_all_ok) and ok_frac >= float(cfg.min_ok_frac):
            return
        # Critical: if every job has reached a terminal state, do not keep
        # polling until max_wait_seconds.  Report that collection can proceed,
        # even if the failure fraction is larger than min_ok_frac.
        if pending <= 0:
            echo("Trainset Gaussian wait: all jobs reached terminal state; collecting completed and failed jobs now.")
            return
        if time.time() - start > int(cfg.max_wait_seconds):
            echo("Trainset Gaussian wait: timeout reached; resume later to collect remaining jobs.")
            return
        time.sleep(max(1, int(cfg.poll_seconds)))
