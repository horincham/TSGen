from __future__ import annotations

import re
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Literal


ChargeMethod = Literal["adch", "hirshfeld"]

DEFAULT_ADCH_OUTPUT_FILES = [
    "adch_charges.txt",
    "adch.out",
    "multiwfn_adch.out",
    "charges.adch",
    "atomic_charges.txt",
]
DEFAULT_ADCH_WAVEFUNCTION_PATTERNS = ["*.fchk", "*.wfn", "*.wfx", "*.molden", "*.mwfn"]


_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[DEde][-+]?\d+)?"


def _to_float(s: str) -> float:
    return float(str(s).replace("D", "E").replace("d", "e"))


def _validate_natoms(charges: list[float], natoms: int | None) -> list[float]:
    if natoms is not None and len(charges) != int(natoms):
        raise ValueError(f"Parsed {len(charges)} charges, expected {natoms}")
    return charges


def parse_adch_charges(text: str, natoms: int | None = None) -> list[float]:
    """Parse ADCH-style atomic charge tables, typically exported by Multiwfn.

    The parser intentionally accepts simple three-column tables because users
    often save ADCH output as cleaned text:

        1 C -0.123456
        2 H  0.061728
    """
    charges: list[float] = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.search(r"\b(sum|total)\b", line, flags=re.I):
            continue
        m = re.match(rf"^\s*(\d+)\s+([A-Za-z]{{1,3}})\s+({_FLOAT})\b", line)
        if m:
            charges.append(_to_float(m.group(3)))
    return _validate_natoms(charges, natoms)


def parse_hirshfeld_charges(text: str, natoms: int | None = None) -> list[float]:
    """Parse Gaussian Hirshfeld population charges.

    Gaussian variants differ slightly in headings; we start after the first
    "Hirshfeld charges" line and read rows like:

        1  C  -0.130000  0.000000 ...
    """
    lines = str(text).splitlines()
    start = None
    for i, line in enumerate(lines):
        if "hirshfeld charges" in line.lower():
            start = i + 1
            break
    if start is None:
        raise ValueError("No Gaussian Hirshfeld charge block found")

    charges: list[float] = []
    for raw in lines[start:]:
        line = raw.strip()
        if not line:
            if charges:
                break
            continue
        if line.lower().startswith(("tot", "sum")):
            break
        m = re.match(rf"^\s*(\d+)\s+([A-Za-z]{{1,3}})\s+({_FLOAT})\b", line)
        if m:
            charges.append(_to_float(m.group(3)))
            if natoms is not None and len(charges) >= int(natoms):
                break
    return _validate_natoms(charges, natoms)


def _read_first_existing(paths: Iterable[Path]) -> tuple[Path, str] | None:
    for p in paths:
        if p.exists() and p.is_file():
            return p, p.read_text(encoding="utf-8", errors="ignore")
    return None


def _quote_for_shell(path: Path | str) -> str:
    s = str(path)
    if os.name == "nt":
        return subprocess.list2cmdline([s])
    return shlex.quote(s)


def _adch_output_paths(job_dir: Path, output_files: Iterable[str] | None = None) -> list[Path]:
    names = list(output_files or DEFAULT_ADCH_OUTPUT_FILES)
    return [job_dir / name for name in names]


def _find_wavefunction_file(job_dir: Path, patterns: Iterable[str] | None = None) -> Path | None:
    for pattern in patterns or DEFAULT_ADCH_WAVEFUNCTION_PATTERNS:
        hits = sorted(job_dir.glob(pattern))
        for p in hits:
            if p.is_file():
                return p
    return None


def run_adch_postprocess(
    job_dir: Path,
    *,
    command_template: str,
    natoms: int,
    wavefunction_patterns: Iterable[str] | None = None,
    output_files: Iterable[str] | None = None,
    timeout_seconds: int = 600,
) -> Path | None:
    """Run a user-configured ADCH postprocessing command in a job directory.

    Multiwfn menu scripts differ across local installations, so TSGen does not
    hard-code a menu sequence.  Instead it supplies robust placeholders for a
    site-validated command template and then parses the resulting ADCH table.
    """
    jd = Path(job_dir)
    wf = _find_wavefunction_file(jd, wavefunction_patterns)
    if wf is None:
        return None
    output_paths = _adch_output_paths(jd, output_files)
    output_file = output_paths[0]
    values = {
        "job_dir": str(jd),
        "job_dir_q": _quote_for_shell(jd),
        "wavefunction_file": str(wf),
        "wavefunction_file_q": _quote_for_shell(wf),
        "input_log": str(jd / "input.log"),
        "input_log_q": _quote_for_shell(jd / "input.log"),
        "output_file": str(output_file),
        "output_file_q": _quote_for_shell(output_file),
        "natoms": int(natoms),
    }
    try:
        command = str(command_template).format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown ADCH command placeholder: {exc}") from exc
    proc = subprocess.run(
        command,
        cwd=str(jd),
        shell=True,
        capture_output=True,
        text=True,
        timeout=max(1, int(timeout_seconds)),
    )
    (jd / "adch_postprocess.stdout").write_text(proc.stdout or "", encoding="utf-8")
    (jd / "adch_postprocess.stderr").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        return None
    found = _read_first_existing(output_paths)
    return found[0] if found is not None else None


def load_reference_charges(
    job_dir: Path,
    *,
    natoms: int,
    method: ChargeMethod = "adch",
    fallback_method: Literal["hirshfeld", "none"] = "hirshfeld",
    adch_command: str | None = None,
    adch_wavefunction_patterns: Iterable[str] | None = None,
    adch_output_files: Iterable[str] | None = None,
    adch_timeout_seconds: int = 600,
) -> tuple[list[float], str]:
    """Load preferred reference charges for a Gaussian/ReaxFF trainset job.

    ADCH is not produced by Gaussian itself; TSGen therefore looks for common
    Multiwfn ADCH text exports in the job directory.  If ADCH output is absent
    and fallback is enabled, Hirshfeld charges are parsed from ``input.log``.
    """
    jd = Path(job_dir)
    if method == "adch":
        output_paths = _adch_output_paths(jd, adch_output_files)
        found = _read_first_existing(output_paths)
        if found is None and adch_command:
            run_adch_postprocess(
                jd,
                command_template=adch_command,
                natoms=natoms,
                wavefunction_patterns=adch_wavefunction_patterns,
                output_files=adch_output_files,
                timeout_seconds=adch_timeout_seconds,
            )
            found = _read_first_existing(output_paths)
        if found is not None:
            _p, text = found
            return parse_adch_charges(text, natoms=natoms), "adch"
        if fallback_method == "none":
            raise FileNotFoundError(f"No ADCH charge output found in {jd}")

    if method == "hirshfeld" or fallback_method == "hirshfeld":
        log = jd / "input.log"
        if not log.exists():
            raise FileNotFoundError(f"No Gaussian log found for Hirshfeld fallback: {log}")
        return parse_hirshfeld_charges(log.read_text(encoding="utf-8", errors="ignore"), natoms=natoms), "hirshfeld"

    raise ValueError(f"Unsupported charge method/fallback combination: {method}/{fallback_method}")
