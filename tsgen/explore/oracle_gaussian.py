from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from tsgen.data.xyz import Frame, write_xyz_multiframe
from tsgen.explore.config import GaussianCfg
from tsgen.explore.label_store import HARTREE_TO_KCALMOL, LabelRecord


# Minimal atomic number table (extend if needed)
_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "I": 53,
}


def _atomic_number(sym: str) -> int:
    z = _ATOMIC_NUMBERS.get(sym)
    if z is None:
        raise ValueError(
            f"Unsupported element symbol {sym!r} in Gaussian oracle. "
            "Please extend _ATOMIC_NUMBERS in oracle_gaussian.py."
        )
    return z


def _compute_total_electrons(frame: Frame, charge: int) -> int:
    # Ne = sum(Z_i) - charge
    return int(sum(_atomic_number(s) for s in frame.symbols) - int(charge))


def _resolve_charge_mult(frame: Frame, charge_cfg, mult_cfg) -> tuple[int, int, int]:
    """
    Resolve (charge, multiplicity) for Gaussian.
    - charge_cfg can be int or "auto" (defaults to 0).
    - mult_cfg can be int or "auto" (parity-based: even Ne -> 1, odd Ne -> 2).
    Returns: (charge, multiplicity, n_electrons)
    """
    # charge
    if isinstance(charge_cfg, str) and charge_cfg.lower() == "auto":
        q = 0
    else:
        q = int(charge_cfg)

    ne = _compute_total_electrons(frame, q)

    # multiplicity
    if isinstance(mult_cfg, str) and mult_cfg.lower() == "auto":
        m = 1 if (ne % 2 == 0) else 2
    else:
        m = int(mult_cfg)

    # Validate parity: even Ne -> odd mult; odd Ne -> even mult
    parity_ok = ((ne % 2 == 0) and (m % 2 == 1)) or ((ne % 2 == 1) and (m % 2 == 0))
    if not parity_ok:
        # Do NOT auto-correct explicit multiplicity; make the error obvious early.
        raise ValueError(
            f"Incompatible (charge={q}, multiplicity={m}) for electron count Ne={ne}. "
            "Rule: even Ne -> odd multiplicity (1,3,5...), odd Ne -> even multiplicity (2,4,6...). "
            "Set multiplicity: auto (recommended) or choose a compatible value."
        )

    if m < 1:
        raise ValueError(f"Invalid multiplicity {m}. Must be >= 1.")

    return q, m, ne


@dataclass(frozen=True)
class GaussianJobSpec:
    id: str
    species_id: str
    frame: Frame


class GaussianOracle:
    """
    Gaussian oracle:
      - write_job(): create job folder with structure.xyz, input.gjf, submit.slurm, meta.json
      - submit(): submit via SLURM (sbatch) or prepare local run
      - collect_labels(): parse Gaussian log and return LabelRecord list (ok only)
    """

    def __init__(self, cfg: GaussianCfg):
        self.cfg = cfg

    # -------------------------
    # Job writing
    # -------------------------
    def write_job(self, job_dir: Path, spec: GaussianJobSpec) -> None:
        job_dir.mkdir(parents=True, exist_ok=True)

        # Write structure (single-frame xyz)
        xyz_path = job_dir / "structure.xyz"
        write_xyz_multiframe(xyz_path, [spec.frame])

        # Resolve charge/mult *from structure*
        q, m, ne = _resolve_charge_mult(spec.frame, self.cfg.charge, self.cfg.multiplicity)

        # Write Gaussian input
        gjf_path = job_dir / "input.gjf"
        self._write_gjf(gjf_path, spec.frame, charge=q, multiplicity=m)

        # Write SLURM script if needed
        if self.cfg.mode == "slurm":
            self._write_slurm(job_dir, gjf_path)

        # Meta for auditing / parsing
        meta = {
            "id": spec.id,
            "species_id": spec.species_id,
            "route": self.cfg.route,
            "charge_cfg": self.cfg.charge,
            "multiplicity_cfg": self.cfg.multiplicity,
            "charge": q,
            "multiplicity": m,
            "n_electrons": ne,
            "mode": self.cfg.mode,
            "nproc": self.cfg.nproc,
            "mem": self.cfg.mem,
        }
        (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _write_gjf(self, gjf: Path, frame: Frame, charge: int, multiplicity: int) -> None:
        # Gaussian mem syntax: accept "8GB" or "8GB"/"8Gb"; keep as provided
        mem = str(self.cfg.mem).strip()
        nproc = int(self.cfg.nproc)

        lines: list[str] = []
        lines.append(f"%nprocshared={nproc}")
        if mem:
            lines.append(f"%mem={mem}")
        lines.append(str(self.cfg.route).rstrip())
        lines.append("")  # blank line
        lines.append("TSGen oracle single-point")
        lines.append("")  # blank line
        lines.append(f"{int(charge)} {int(multiplicity)}")

        for sym, (x, y, z) in zip(frame.symbols, frame.coords):
            lines.append(f"{sym:2s}  {x: .10f}  {y: .10f}  {z: .10f}")

        lines.append("")  # blank line
        lines.append("")  # end of file

        gjf.write_text("\n".join(lines), encoding="utf-8")

    def _write_slurm(self, job_dir: Path, gjf: Path) -> Path:
        slurm = job_dir / "submit.slurm"

        # SlurmCfg is a Pydantic object; access attributes
        scfg = self.cfg.slurm
        part = getattr(scfg, "partition", None)
        time_lim = getattr(scfg, "time", None)
        cpus = getattr(scfg, "cpus_per_task", None)
        mem = getattr(scfg, "mem", None)
        extra_lines = list(getattr(scfg, "extra_lines", []) or [])

        job_name = f"tsgen_al_{job_dir.name}"
        out = "gaussian.out"
        err = "gaussian.err"

        lines: list[str] = ["#!/bin/bash"]
        if part:
            lines.append(f"#SBATCH -p {part}")
        lines.append(f"#SBATCH -J {job_name}")
        lines.append(f"#SBATCH -o {out}")
        lines.append(f"#SBATCH -e {err}")
        if time_lim:
            lines.append(f"#SBATCH -t {time_lim}")
        if cpus:
            lines.append(f"#SBATCH --cpus-per-task={int(cpus)}")
        if mem:
            lines.append(f"#SBATCH --mem={mem}")

        lines.append("")
        # module load etc.
        lines.extend(extra_lines)
        if extra_lines:
            lines.append("")

        # Run Gaussian
        # Some clusters require exporting GAUSS_SCRDIR, etc. user can add via extra_lines.
        g16 = str(self.cfg.g16_cmd or "g16").strip()
        lines.append(f"{g16} < {gjf.name} > input.log")
        lines.append("")

        slurm.write_text("\n".join(lines), encoding="utf-8")
        return slurm

    # -------------------------
    # Submission
    # -------------------------
    def submit(self, job_dir: Path) -> Optional[str]:
        """
        Submit a prepared job. Returns SLURM jobid if submitted, else None.
        In local mode, returns None.
        """
        if self.cfg.mode != "slurm":
            # local mode: do not submit here
            return None

        slurm = job_dir / "submit.slurm"
        if not slurm.exists():
            raise FileNotFoundError(f"Missing submit script: {slurm}")

        # Important: use cwd=job_dir and pass only script filename
        proc = subprocess.run(
            ["sbatch", slurm.name],
            cwd=str(job_dir),
            capture_output=True,
            text=True,
        )

        # Persist outputs for audit/debug
        (job_dir / "sbatch.stdout").write_text(proc.stdout or "", encoding="utf-8")
        (job_dir / "sbatch.stderr").write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            return None

        # Parse jobid from stdout: "Submitted batch job 123456"
        m = re.search(r"Submitted batch job\s+(\d+)", proc.stdout or "")
        return m.group(1) if m else None

    # -------------------------
    # Collection / parsing
    # -------------------------
    def job_status(self, job_dir: Path) -> str:
        """Return one of: pending|ok|failed based on Gaussian log presence/content."""
        log = job_dir / "input.log"
        if not log.exists():
            return "pending"
        try:
            t = log.read_text(errors="ignore")
        except Exception:
            return "pending"
        if "Normal termination" in t:
            return "ok"
        if "Error termination" in t or "Error termination" in t:
            return "failed"
        return "pending"


    def collect_labels(self, jobs_dir: Path, iter_idx: int) -> list[LabelRecord]:
        """
        Scan jobs_dir/job_*/input.log and return LabelRecord list for jobs that succeeded.
        Only "ok" records are returned (failed jobs are simply ignored here).
        """
        if not jobs_dir.exists():
            return []

        records: list[LabelRecord] = []
        for job in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
            meta_p = job / "meta.json"
            log_p = job / "input.log"
            if not (meta_p.exists() and log_p.exists()):
                continue

            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            text = log_p.read_text(errors="ignore")

            # Skip failed Gaussian logs
            if "Error termination" in text or "impossible" in text:
                continue

            e = self._parse_scf_energy_hartree(text)
            if e is None:
                continue

            e_kcal = float(e) * float(HARTREE_TO_KCALMOL)

            rec = LabelRecord(
                id=str(meta.get("id", "")),
                species_id=str(meta.get("species_id", "")),
                e_hartree=float(e),
                e_kcalmol_abs=float(e_kcal),
                e_kcalmol_rel=float(0.0),
                iter=int(iter_idx),
                job_dir=str(job),
                log_path=str(log_p),
                status="ok",
                route=str(meta.get("route", "")),
                charge=int(meta.get("charge", 0)),
                multiplicity=int(meta.get("multiplicity", 1)),
            )
            records.append(rec)

        return records

    @staticmethod
    def _parse_scf_energy_hartree(text: str) -> Optional[float]:
        """
        Parse "SCF Done:  E(RB3LYP) =  -xxx" or similar.
        Returns energy in Hartree, or None.
        """
        # Common Gaussian line: "SCF Done:  E(RM062X) =  -385.123456789     A.U. after ..."
        m = re.search(r"SCF Done:\s+E\([A-Za-z0-9]+\)\s*=\s*([-]?\d+\.\d+)", text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None