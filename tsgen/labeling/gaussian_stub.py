from __future__ import annotations

"""Gaussian task writer + passive parser.

This backend is intentionally *execution-agnostic*: TSGen writes input decks and
parses finished outputs, but does not attempt to submit jobs on a specific HPC
scheduler (SLURM/PBS/LFS). This keeps the workflow reproducible and portable.

For publication-grade data, labels must originate from the Gaussian output files
(.log/.out). TSGen never fabricates energies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re

from tsgen.data.xyz import Frame


_HARTREE_TO_KCAL_MOL = 627.509474  # commonly used conversion


def _parse_gaussian_energy(text: str) -> Optional[float]:
    # Preferred: SCF Done energy
    m = re.search(r"SCF Done:\s+E\([^\)]+\)\s*=\s*([-+]?\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    # Fallback patterns (rare)
    for pat in [
        r"Final energy is\s+([-+]?\d+\.\d+)",
        r"Total Energy,\s*E\s*=\s*([-+]?\d+\.\d+)",
    ]:
        m2 = re.search(pat, text)
        if m2:
            return float(m2.group(1))
    return None


@dataclass
class GaussianStubBackend:
    """Minimal Gaussian backend suitable for real data generation.

    - Writes one subdirectory per configuration id, containing <id>.gjf
    - Expects external execution to produce <id>.log (or .out)
    - Parses energies into task_dir/results.json
    """

    route: str = "#p B3LYP/6-31G(d) SP"
    charge: int = 0
    multiplicity: int = 1
    output_exts: tuple[str, ...] = (".log", ".out")

    def generate_tasks(self, task_dir: Path, selected: List[Frame], ids: List[str], layer_cfg: Dict[str, Any]) -> None:
        task_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": "gaussian",
            "ids": ids,
            "layer_cfg": layer_cfg,
            "route": self.route,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
        }
        (task_dir / "tasks.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        for fr, cid in zip(selected, ids):
            d = task_dir / cid
            d.mkdir(parents=True, exist_ok=True)
            gjf = d / f"{cid}.gjf"
            lines: List[str] = []
            lines.append(self.route)
            lines.append("")
            lines.append(cid)
            lines.append("")
            lines.append(f"{self.charge} {self.multiplicity}")
            for s, (x, y, z) in zip(fr.symbols, fr.coords):
                lines.append(f"{s:2s} {x: .6f} {y: .6f} {z: .6f}")
            lines.append("")
            gjf.write_text("\n".join(lines), encoding="utf-8")

    def run_and_wait(self, task_dir: Path) -> None:
        # no-op: intended for HPC/external execution
        return

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
            out_file = None
            for ext in self.output_exts:
                p = d / f"{cid}{ext}"
                if p.exists():
                    out_file = p
                    break
            if out_file is None:
                continue
            text = out_file.read_text(errors="ignore")
            e = _parse_gaussian_energy(text)
            if e is None:
                results[cid] = {"status": "present_unparsed", "source_file": str(out_file)}
                continue

            results[cid] = {
                "status": "ok",
                "source_file": str(out_file),
                "energy_hartree": float(e),
                "energy_kcal_mol": float(e) * _HARTREE_TO_KCAL_MOL,
            }

        (task_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return results
