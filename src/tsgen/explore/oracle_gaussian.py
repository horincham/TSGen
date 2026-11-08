from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from tsgen.data.xyz import Frame, write_xyz_multiframe
from tsgen.explore.config import GaussianCfg
from tsgen.explore.label_store import HARTREE_TO_KCALMOL, LabelRecord
from tsgen.outerloop.io import atomic_json
from tsgen.trainset.gaussian import _process_exit_code, _scheduler_terminal_failure


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

    def __init__(self, cfg: GaussianCfg, timeout_seconds: float | None = None):
        self.cfg = cfg
        self.timeout_seconds = timeout_seconds

    # Job writing
    def write_job(self, job_dir: Path, spec: GaussianJobSpec) -> None:
        job_dir.mkdir(parents=True, exist_ok=True)

        # Write structure (single-frame xyz)
        xyz_path = job_dir / "structure.xyz"
        write_xyz_multiframe(xyz_path, [spec.frame], precision=16)

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
        mem = str(self.cfg.mem).strip()
        nproc = int(self.cfg.nproc)

        lines: list[str] = []
        lines.append(f"%nprocshared={nproc}")
        if mem:
            lines.append(f"%mem={mem}")
        lines.append(str(self.cfg.route).rstrip())
        lines.append("")  # blank line
        lines.append("TSGen DFT job")
        lines.append("")  # blank line
        lines.append(f"{int(charge)} {int(multiplicity)}")

        for sym, (x, y, z) in zip(frame.symbols, frame.coords):
            lines.append(f"{sym:2s}  {x: .16f}  {y: .16f}  {z: .16f}")

        lines.append("")  # blank line
        lines.append("")  # end of file

        gjf.write_text("\n".join(lines), encoding="utf-8")

    def _write_slurm(self, job_dir: Path, gjf: Path) -> Path:
        slurm = job_dir / "submit.slurm"

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

        lines.extend(["", 'cd "${SLURM_SUBMIT_DIR:-$PWD}"', "set -e"])
        lines.extend(extra_lines)
        if extra_lines:
            lines.append("")

        g16 = str(self.cfg.g16_cmd or "g16").strip()
        lines.append("set +e")
        lines.append(f"{g16} < {gjf.name} > input.log")
        lines.extend(['status=$?', 'printf "%s\\n" "$status" > gaussian.exitcode', 'exit "$status"', ""])

        with slurm.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines))
        return slurm

    # Submission
    def submit(self, job_dir: Path, *, allow_calculation_failure: bool = False) -> Optional[str]:
        """
        Execute locally, submit once to Slurm, or retain manual inputs.
        """
        if self.cfg.mode == "manual":
            return None
        if self.cfg.mode == "local":
            marker = job_dir / "local_execution.json"
            if marker.exists():
                state = json.loads(marker.read_text(encoding="utf-8"))
                if state.get("status") != "RETURNED":
                    raise RuntimeError(f"Gaussian execution was interrupted: {job_dir}; inspect the process and logs before retrying")
                return None
            if self.job_status(job_dir) != "pending":
                return None
            argv = shlex.split(self.cfg.g16_cmd)
            if not argv:
                raise ValueError("Gaussian executable command is empty")
            with marker.open("x", encoding="utf-8") as handle:
                json.dump({"status": "STARTED", "argv": argv}, handle)
            try:
                with (job_dir / "input.gjf").open("rb") as source, \
                        (job_dir / "input.log").open("wb") as out, \
                        (job_dir / "stderr.log").open("wb") as err:
                    completed = subprocess.run(
                        argv, cwd=job_dir, stdin=source, stdout=out, stderr=err,
                        check=False, timeout=self.timeout_seconds,
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                atomic_json(marker, {"status": "INTERRUPTED", "argv": argv, "error": str(exc)})
                raise RuntimeError(f"Gaussian execution did not complete: {job_dir}: {exc}") from exc
            atomic_json(marker, {"status": "RETURNED", "argv": argv, "returncode": completed.returncode})
            (job_dir / "gaussian.exitcode").write_text(str(completed.returncode), encoding="utf-8")
            if completed.returncode and not allow_calculation_failure:
                raise RuntimeError(f"Gaussian exited with exit code {completed.returncode}: {job_dir}; inspect stderr.log")
            return None

        previous = job_dir / "sbatch.stdout"
        if previous.exists():
            accepted = re.search(r"Submitted batch job\s+(\d+)", previous.read_text(encoding="utf-8"))
            if accepted:
                return accepted.group(1)
        if self.job_status(job_dir) != "pending":
            return None

        slurm = job_dir / "submit.slurm"
        if not slurm.exists():
            raise FileNotFoundError(f"Missing submit script: {slurm}")

        # Gaussian inputs and logs are relative to the Slurm submission directory.
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
            raise RuntimeError(f"Slurm rejected Gaussian job {job_dir}: {proc.stderr.strip()}")

        # Parse jobid from stdout: "Submitted batch job 123456"
        m = re.search(r"Submitted batch job\s+(\d+)", proc.stdout or "")
        if m is None:
            raise RuntimeError(f"Slurm did not return a job ID for {job_dir}; inspect sbatch.stdout before retrying")
        return m.group(1)

    # Collection / parsing
    def job_status(self, job_dir: Path) -> str:
        """Return one of: pending|ok|failed based on Gaussian log presence/content."""
        exit_code = _process_exit_code(job_dir)
        if exit_code is not None and exit_code != 0:
            return "failed"
        marker = job_dir / "local_execution.json"
        if marker.exists():
            state = json.loads(marker.read_text(encoding="utf-8"))
            if state.get("status") == "INTERRUPTED" or state.get("returncode", 0) != 0:
                return "failed"
        log = job_dir / "input.log"
        if not log.exists():
            return "failed" if exit_code is not None or _scheduler_terminal_failure(job_dir) else "pending"
        try:
            t = log.read_text(errors="ignore")
        except Exception:
            return "pending"
        normal = t.rfind("Normal termination")
        error = t.rfind("Error termination")
        if error > normal:
            return "failed"
        if normal >= 0 and self._parse_scf_energy_hartree(t) is not None:
            return "ok"
        return "failed" if exit_code is not None or _scheduler_terminal_failure(job_dir) else "pending"


    def collect_labels(self, jobs_dir: Path, iter_idx: int) -> list[LabelRecord]:
        """
        Scan jobs_dir/job_*/input.log and return LabelRecord list for jobs that succeeded.
        Only "ok" records are returned (failed jobs are simply ignored here).
        """
        if not jobs_dir.exists():
            return []

        return [
            record
            for job in sorted(jobs_dir.iterdir())
            if job.is_dir() and (record := self.collect_label(job, iter_idx)) is not None
        ]

    def collect_label(self, job_dir: Path, iter_idx: int) -> LabelRecord | None:
        meta_path = job_dir / "meta.json"
        log_path = job_dir / "input.log"
        if not meta_path.is_file() or self.job_status(job_dir) != "ok":
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        energy = self._parse_scf_energy_hartree(log_path.read_text(errors="ignore"))
        if energy is None:
            return None
        return LabelRecord(
            id=str(meta.get("id", "")),
            species_id=str(meta.get("species_id", "")),
            e_hartree=energy,
            e_kcalmol_abs=energy * HARTREE_TO_KCALMOL,
            e_kcalmol_rel=0.0,
            iter=iter_idx,
            job_dir=str(job_dir),
            log_path=str(log_path),
            status="ok",
            route=str(meta.get("route", "")),
            charge=int(meta.get("charge", 0)),
            multiplicity=int(meta.get("multiplicity", 1)),
        )

    @staticmethod
    def _parse_scf_energy_hartree(text: str) -> Optional[float]:
        """
        Parse "SCF Done:  E(RB3LYP) =  -xxx" or similar.
        Returns energy in Hartree, or None.
        """
        matches = re.findall(
            r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eEdD][-+]?\d+)?)", text
        )
        return float(matches[-1].replace("D", "E").replace("d", "e")) if matches else None
