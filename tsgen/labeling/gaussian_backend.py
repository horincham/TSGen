from __future__ import annotations

"""Gaussian16 backend (Slurm/local/manual) for publication-grade labeling.

Design goals:
  - Inputs and execution are fully reproducible from run_dir/config.yaml.
  - No fabricated labels: energies are parsed only from Gaussian outputs.
  - Portable: supports Slurm submission (default), local workstation execution,
    and a "manual" mode (write inputs only).
  - Resume-friendly: if some logs are missing, TSGen will stop and can be resumed.

Directory layout (per iter):
  iter_xxx/label_tasks/<id>/
    <id>.gjf
    <id>.log
    <id>.sbatch           (slurm/manual)
    task.json
  iter_xxx/label_tasks/tasks.json
  iter_xxx/label_tasks/results.json
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
import subprocess
import time

from tsgen.data.xyz import Frame


_HARTREE_TO_KCAL_MOL = 627.509474


# Minimal Z table for electron counting (extend if needed)
_Z: Dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Br": 35, "I": 53, "Al": 13, "Na": 11, "K": 19,
}


def _parse_gaussian_energy(text: str) -> Optional[float]:
    m = re.search(r"SCF Done:\s+E\([^\)]+\)\s*=\s*([-+]?\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    for pat in [
        r"Final energy is\s+([-+]?\d+\.\d+)",
        r"Total Energy,\s*E\s*=\s*([-+]?\d+\.\d+)",
    ]:
        m2 = re.search(pat, text)
        if m2:
            return float(m2.group(1))
    return None


def _parse_gaussian_status(text: str) -> tuple[str, Optional[str]]:
    """Classify Gaussian log status.

    Returns:
        (status, reason)
        status in {"ok", "failed", "present_unparsed"}
    """
    # Success marker (keeps this lightweight and robust across routes)
    if "Normal termination of Gaussian" in text:
        return "ok", None

    # Hard-failure markers (common Gaussian failures)
    if "End of file in ZSymb." in text:
        return "failed", "End of file in ZSymb (input truncated / missing blank lines)"
    if "Error termination" in text:
        # Capture a short nearby message if present
        m = re.search(r"Error termination(?:.*\n){0,6}", text)
        snippet = (m.group(0).strip() if m else "Error termination")
        return "failed", snippet[:240]

    # Log exists but neither success nor explicit failure
    return "present_unparsed", None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_comment_charge_mult(comment: str) -> tuple[Optional[int], Optional[int]]:
    if not comment:
        return None, None
    low = comment.lower().replace(",", " ")
    c = None
    m = None
    for tok in low.split():
        if tok.startswith("charge="):
            try:
                c = int(tok.split("=", 1)[1])
            except Exception:
                pass
        if tok.startswith("mult=") or tok.startswith("multiplicity="):
            try:
                m = int(tok.split("=", 1)[1])
            except Exception:
                pass
    return c, m


def _guess_charge_mult(fr: Frame, charge_cfg: Any, mult_cfg: Any) -> tuple[int, int]:
    # Priority: xyz comment > explicit config int > auto
    c_comment, m_comment = _read_comment_charge_mult(fr.comment or "")
    c: Optional[int] = c_comment
    m: Optional[int] = m_comment

    if isinstance(charge_cfg, int):
        c = charge_cfg
    if isinstance(mult_cfg, int):
        m = mult_cfg

    if c is None:
        if charge_cfg == "auto":
            c = 0
        else:
            c = int(charge_cfg)

    if m is None:
        if mult_cfg == "auto":
            zsum = 0
            for s in fr.symbols:
                if s not in _Z:
                    raise KeyError(f"Unknown element '{s}' for electron counting. Extend _Z in gaussian_backend.py")
                zsum += _Z[s]
            ne = zsum - c
            m = 1 if (ne % 2 == 0) else 2
        else:
            m = int(mult_cfg)
    return int(c), int(m)


def _write_sbatch(path: Path, job_name: str, prefix: str, slurm_cfg: Dict[str, Any], extra_lines: List[str]) -> None:
    partition = slurm_cfg.get("partition", "compute")
    time_limit = slurm_cfg.get("time", "02:00:00")
    cpus_per_task = int(slurm_cfg.get("cpus_per_task", 8))
    mem = str(slurm_cfg.get("mem", "16G"))

    lines: List[str] = []
    lines.append("#!/bin/bash")
    lines.append(f"#SBATCH -J {job_name}")
    lines.append(f"#SBATCH -p {partition}")
    lines.append(f"#SBATCH -t {time_limit}")
    lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    lines.append(f"#SBATCH --mem={mem}")
    lines.append(f"#SBATCH -o {prefix}.log")
    lines.append(f"#SBATCH -e {prefix}.err")
    lines.append("")
    for l in extra_lines or []:
        lines.append(l)
    lines.append("")
    lines.append(f"g16 < {prefix}.gjf > {prefix}.log")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _sbatch_submit(cwd: Path, sbatch_file: str) -> str:
    res = subprocess.run(["sbatch", sbatch_file], cwd=str(cwd), capture_output=True, text=True, check=True)
    return res.stdout.strip().split()[-1]


def _in_squeue(jobid: str) -> bool:
    # -h: no header. If output empty -> job not running/pending.
    res = subprocess.run(["squeue", "-j", jobid, "-h"], capture_output=True, text=True)
    return bool((res.stdout or "").strip())


def _wait_job(jobid: str, poll_sec: int = 30, timeout_sec: Optional[int] = None) -> None:
    t0 = time.time()
    while True:
        if not _in_squeue(jobid):
            return
        if timeout_sec is not None and (time.time() - t0) > timeout_sec:
            raise TimeoutError(f"Gaussian job {jobid} still present in squeue after {timeout_sec}s")
        time.sleep(poll_sec)


def _run_local(cwd: Path, g16_cmd: str, prefix: str) -> None:
    cmd = f"{g16_cmd} < {prefix}.gjf > {prefix}.log"
    subprocess.run(cmd, cwd=str(cwd), shell=True, check=True)


def _slurm_mem_to_gaussian(mem: str) -> str:
    """Normalize Slurm-style memory strings to Gaussian %mem format.

    Examples:
      - "16G" -> "16GB"
      - "16000M" -> "16000MB"
      - "8GB" -> "8GB" (unchanged)

    Gaussian accepts a variety of units; we standardize to GB/MB for clarity.
    """
    s = str(mem).strip()
    if not s:
        return s
    m = re.match(r"^(\d+)([KkMmGg][Bb]?)$", s)
    if not m:
        return s
    n, u = m.group(1), m.group(2).lower()
    if u in ("g", "gb"):
        return f"{n}GB"
    if u in ("m", "mb"):
        return f"{n}MB"
    if u in ("k", "kb"):
        return f"{n}KB"
    return s


def _summarize_task_status(task_dir: Path, ids: List[str]) -> dict:
    """Quickly summarize task states for progress logging."""
    n_total = len(ids)
    n_ok = 0
    n_failed = 0
    n_has_log = 0
    n_pending = 0
    for cid in ids:
        d = task_dir / cid
        tjson = d / "task.json"
        status = None
        if tjson.exists():
            try:
                status = json.loads(tjson.read_text(encoding="utf-8")).get("status")
            except Exception:
                status = None
        if (d / f"{cid}.log").exists():
            n_has_log += 1
        if status == "ok":
            n_ok += 1
        elif status == "failed":
            n_failed += 1
        else:
            n_pending += 1
    return {
        "total": n_total,
        "ok": n_ok,
        "failed": n_failed,
        "pending": n_pending,
        "has_log": n_has_log,
    }


@dataclass
class GaussianBackend:
    """Gaussian16 backend supporting slurm/local/manual."""

    mode: str = "slurm"  # slurm|local|manual
    route: str = "# M062X/6-311G(d) SP"
    charge: Any = "auto"
    multiplicity: Any = "auto"
    nproc: int = 8
    mem: str = "8GB"
    g16_cmd: str = "g16"
    slurm: Dict[str, Any] = None
    slurm_extra_lines: List[str] = None
    poll_sec: int = 30

    def generate_tasks(self, task_dir: Path, selected: List[Frame], ids: List[str], layer_cfg: Dict[str, Any]) -> None:
        task_dir.mkdir(parents=True, exist_ok=True)
        gcfg = (layer_cfg.get("gaussian") or {})
        # allow cfg override via config.yaml
        mode = gcfg.get("mode", self.mode)
        route = gcfg.get("route", self.route)
        charge_cfg = gcfg.get("charge", self.charge)
        mult_cfg = gcfg.get("multiplicity", self.multiplicity)
        nproc = int(gcfg.get("nproc", self.nproc))
        mem = str(gcfg.get("mem", self.mem))
        g16_cmd = str(gcfg.get("g16_cmd", self.g16_cmd))
        slurm_cfg = dict((gcfg.get("slurm") or (self.slurm or {})))
        extra_lines = list(gcfg.get("slurm", {}).get("extra_lines", []) or (self.slurm_extra_lines or []))

        # Keep Slurm and Gaussian resource declarations consistent unless explicitly overridden.
        user_set_mem = "mem" in gcfg
        user_set_nproc = "nproc" in gcfg
        if str(mode).lower() == "slurm":
            if (not user_set_nproc) and (slurm_cfg or {}).get("cpus_per_task"):
                try:
                    nproc = int(slurm_cfg.get("cpus_per_task"))
                except Exception:
                    pass
            if (not user_set_mem) and (slurm_cfg or {}).get("mem"):
                mem = _slurm_mem_to_gaussian(str(slurm_cfg.get("mem")))

        meta = {
            "backend": "gaussian",
            "ids": ids,
            "created_at": _now(),
            "mode": mode,
            "route": route,
            "charge": charge_cfg,
            "multiplicity": mult_cfg,
            "nproc": nproc,
            "mem": mem,
            "g16_cmd": g16_cmd,
            "slurm": slurm_cfg,
        }
        (task_dir / "tasks.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        for fr, cid in zip(selected, ids):
            d = task_dir / cid
            d.mkdir(parents=True, exist_ok=True)

            # guess charge/mult per structure (auto uses electron parity)
            c, m = _guess_charge_mult(fr, charge_cfg, mult_cfg)

            prefix = cid
            gjf = d / f"{prefix}.gjf"
            # Write Gaussian input
            lines: List[str] = []
            # Always write chk; use prefix to keep naming aligned with files
            lines.append(f"%chk={prefix}.chk")
            lines.append(f"%nprocshared={nproc}")
            lines.append(f"%mem={mem}")
            r = route.strip()
            if not r.startswith("#"):
                r = "# " + r
            lines.append(r)
            lines.append("")
            lines.append(prefix)
            lines.append("")
            lines.append(f"{c} {m}")
            for s, (x, y, z) in zip(fr.symbols, fr.coords):
                lines.append(f"{s:2s} {x: .8f} {y: .8f} {z: .8f}")
            # Gaussian is sensitive to file termination. Ensure at least two
            # blank lines and a trailing newline (avoids 'End of file in ZSymb.').
            lines.append("")
            lines.append("")
            gjf.write_text("\n".join(lines).rstrip() + "\n\n", encoding="utf-8")

            task_meta = {
                "id": cid,
                "prefix": prefix,
                "route": r,
                "charge": c,
                "multiplicity": m,
                "mode": mode,
                "created_at": _now(),
            }

            # Write sbatch for slurm/manual
            if mode in ("slurm", "manual"):
                sb = d / f"{prefix}.sbatch"
                _write_sbatch(sb, job_name=prefix, prefix=prefix, slurm_cfg=slurm_cfg, extra_lines=extra_lines)
                task_meta["sbatch"] = sb.name

            (d / "task.json").write_text(json.dumps(task_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_and_wait(self, task_dir: Path) -> None:
        meta_path = task_dir / "tasks.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mode = meta.get("mode", self.mode)
        ids: List[str] = list(meta.get("ids", []))

        if mode == "manual":
            return

        if mode == "local":
            g16_cmd = meta.get("g16_cmd", self.g16_cmd)
            for cid in ids:
                d = task_dir / cid
                if not d.exists():
                    continue
                log = d / f"{cid}.log"
                if log.exists():
                    continue
                _run_local(d, g16_cmd=g16_cmd, prefix=cid)
            return

        if mode == "slurm":
            jobids: List[str] = []
            t_submit0 = time.time()
            for cid in ids:
                d = task_dir / cid
                if not d.exists():
                    continue
                log = d / f"{cid}.log"
                if log.exists():
                    continue
                tjson = d / "task.json"
                tj = json.loads(tjson.read_text(encoding="utf-8")) if tjson.exists() else {"id": cid, "prefix": cid}
                if tj.get("jobid"):
                    jobids.append(str(tj["jobid"]))
                    continue
                sbatch_name = tj.get("sbatch", f"{cid}.sbatch")
                jobid = _sbatch_submit(d, sbatch_name)
                tj["jobid"] = jobid
                tj["submitted_at"] = _now()
                tjson.write_text(json.dumps(tj, indent=2, ensure_ascii=False), encoding="utf-8")
                jobids.append(jobid)

            if jobids:
                print(f"[INFO] dft.slurm.submitted n_jobs={len(jobids)} poll_sec={self.poll_sec}")

            # Wait all submitted jobs, but also emit progress so the run does not appear stuck.
            # We consider a job 'done' once it disappears from squeue (success or failure).
            while True:
                in_queue = 0
                for jid in list(jobids):
                    if _in_squeue(jid):
                        in_queue += 1
                stat = _summarize_task_status(task_dir, ids)
                elapsed_min = (time.time() - t_submit0) / 60.0
                print(
                    f"[INFO] dft.slurm.poll in_queue={in_queue}/{len(jobids)} "
                    f"task_ok={stat['ok']} task_failed={stat['failed']} task_pending={stat['pending']} "
                    f"has_log={stat['has_log']} elapsed_min={elapsed_min:.1f}"
                )
                if in_queue == 0:
                    break
                time.sleep(self.poll_sec)
            return

        raise ValueError(f"Unknown gaussian mode: {mode}")

    def parse_results(self, task_dir: Path) -> Dict[str, Dict[str, Any]]:
        meta_path = task_dir / "tasks.json"
        if not meta_path.exists():
            return {}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids: List[str] = list(meta.get("ids", []))

        results: Dict[str, Dict[str, Any]] = {}
        for cid in ids:
            d = task_dir / cid
            if not d.exists():
                continue
            log = d / f"{cid}.log"
            if not log.exists():
                # pending/unavailable
                continue
            text = log.read_text(errors="ignore")
            status, reason = _parse_gaussian_status(text)
            e = _parse_gaussian_energy(text)

            # Update per-task metadata for robust resume and audit
            tjson = d / "task.json"
            tj = json.loads(tjson.read_text(encoding="utf-8")) if tjson.exists() else {"id": cid, "prefix": cid}
            tj["log"] = log.name
            tj["parsed_at"] = _now()

            if status == "ok" and e is not None:
                rec = {
                    "status": "ok",
                    "source_file": str(log),
                    "energy_hartree": float(e),
                    "energy_kcal_mol": float(e) * _HARTREE_TO_KCAL_MOL,
                }
                tj["status"] = "ok"
                tj["energy_hartree"] = float(e)
            elif status == "failed":
                rec = {
                    "status": "failed",
                    "source_file": str(log),
                    "fail_reason": reason or "Gaussian failed",
                }
                tj["status"] = "failed"
                tj["fail_reason"] = rec["fail_reason"]
            else:
                # Log present but not parseable yet
                rec = {"status": "present_unparsed", "source_file": str(log)}
                if reason:
                    rec["note"] = reason
                tj["status"] = "present_unparsed"

            tjson.write_text(json.dumps(tj, indent=2, ensure_ascii=False), encoding="utf-8")
            results[cid] = rec

        (task_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return results
