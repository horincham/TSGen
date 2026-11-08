from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess

from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.paths import SensitivityPaths


def _pythonpath_line():
    # Workers also work with an uninstalled source checkout when submitted
    # from the dedicated slurm directory rather than the TSGen root.
    source_root = Path(__file__).resolve().parents[2].as_posix()
    return f'export PYTHONPATH={shlex.quote(source_root)}:"${{PYTHONPATH:-}}"'


@dataclass(frozen=True)
class SlurmArrayBatch:
    script: Path
    pending_file: Path
    sample_ids: list[str]


class SlurmSubmissionLimitError(RuntimeError):
    """Scheduler rejected the submission because the job-count quota is full."""


def write_pending_list(
    paths: SensitivityPaths,
    pending_ids: list[str],
    filename: str = "pending_samples.txt",
) -> Path:
    paths.slurm_dir.mkdir(parents=True, exist_ok=True)
    output = paths.slurm_dir / filename
    output.write_text("\n".join(pending_ids) + ("\n" if pending_ids else ""), encoding="utf-8", newline="\n")
    return output


def write_manual_script(paths: SensitivityPaths, pending_ids: list[str]) -> Path:
    pending = write_pending_list(paths, pending_ids)
    output = paths.slurm_dir / "run_pending_samples.sh"
    output.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'{_pythonpath_line()}\n'
        f'RUN_DIR={shlex.quote(paths.run_dir.resolve().as_posix())}\n'
        f'while IFS= read -r SAMPLE_ID; do\n'
        '  [ -z "${SAMPLE_ID}" ] && continue\n'
        '  python -m tsgen.cli sensitivity worker --run-dir "${RUN_DIR}" --sample-id "${SAMPLE_ID}"\n'
        f'done < {shlex.quote(pending.resolve().as_posix())}\n',
        encoding="utf-8",
        newline="\n",
    )
    try:
        output.chmod(0o755)
    except OSError:
        pass
    return output


def write_slurm_array(
    paths: SensitivityPaths,
    cfg: SensitivityConfig,
    pending_ids: list[str],
    *,
    script_name: str = "sensitivity_array.sh",
    pending_name: str = "pending_samples.txt",
) -> Path:
    if not pending_ids:
        raise ValueError("No unfinished samples are available for Slurm submission")
    pending_file = write_pending_list(paths, pending_ids, pending_name)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    slurm = cfg.execution.slurm
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=tsgen-hsic",
        f"#SBATCH --array=0-{len(pending_ids) - 1}%{slurm.array_parallelism}",
        f"#SBATCH --time={slurm.time}",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --mem={slurm.mem}",
    ]
    if slurm.partition:
        lines.append(f"#SBATCH --partition={slurm.partition}")
    directives = [line for line in slurm.extra_lines if line.lstrip().startswith('#SBATCH')]
    setup_lines = [line for line in slurm.extra_lines if not line.lstrip().startswith('#SBATCH')]
    lines.extend(directives)
    # Scheduler opens these before starting Python; the directory must exist.
    # Set these last so extra_lines cannot accidentally restore root-level logs.
    log_dir = paths.logs_dir.resolve().as_posix()
    lines.extend([
        f'#SBATCH --output="{log_dir}/slurm-%A_%a.out"',
        f'#SBATCH --error="{log_dir}/slurm-%A_%a.err"',
    ])
    lines.extend(
        [
            "set -euo pipefail",
            *setup_lines,
            _pythonpath_line(),
            f'RUN_DIR={shlex.quote(paths.run_dir.resolve().as_posix())}',
            f'SAMPLE_ID="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" '
            f'"${{RUN_DIR}}/slurm/{pending_file.name}")"',
            'python -m tsgen.cli sensitivity worker --run-dir "${RUN_DIR}" --sample-id "${SAMPLE_ID}"',
        ]
    )
    output = paths.slurm_dir / script_name
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    try:
        output.chmod(0o755)
    except OSError:
        pass
    return output


def write_slurm_arrays(
    paths: SensitivityPaths,
    cfg: SensitivityConfig,
    pending_ids: list[str],
    *,
    submission_tag: str | None = None,
) -> list[SlurmArrayBatch]:
    if not pending_ids:
        raise ValueError("No unfinished samples are available for Slurm submission")
    max_size = cfg.execution.slurm.max_array_size
    chunks = [
        pending_ids[start : start + max_size]
        for start in range(0, len(pending_ids), max_size)
    ]
    batches = []
    for index, sample_ids in enumerate(chunks, start=1):
        if submission_tag is not None:
            suffix = f"{submission_tag}_batch_{index:04d}"
            script_name = f"sensitivity_array_{suffix}.sh"
            pending_name = f"pending_samples_{suffix}.txt"
        elif len(chunks) == 1:
            script_name = "sensitivity_array.sh"
            pending_name = "pending_samples.txt"
        else:
            suffix = f"{index:04d}"
            script_name = f"sensitivity_array_{suffix}.sh"
            pending_name = f"pending_samples_{suffix}.txt"
        script = write_slurm_array(
            paths,
            cfg,
            sample_ids,
            script_name=script_name,
            pending_name=pending_name,
        )
        batches.append(
            SlurmArrayBatch(
                script=script,
                pending_file=paths.slurm_dir / pending_name,
                sample_ids=sample_ids,
            )
        )
    return batches


def submit_slurm(script: str | Path) -> str:
    script = Path(script).resolve()
    completed = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        message=f"sbatch failed: {completed.stderr.strip()}"
        if any(code in completed.stderr for code in (
            'QOSMaxSubmitJobPerUserLimit','QOSMaxSubmitJobPerAccountLimit',
            'QOSGrpSubmitJobsLimit','AssocMaxSubmitJobLimit','AssocGrpSubmitJobsLimit')):
            raise SlurmSubmissionLimitError(message)
        raise RuntimeError(message)
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"\d+", job_id):
        raise RuntimeError(f"Could not parse Slurm job ID from: {completed.stdout.strip()}")
    return job_id


def slurm_job_is_active(job_id: str) -> bool | None:
    try:
        completed = subprocess.run(
            ["squeue", "--noheader", "--jobs", str(job_id), "--format=%T"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())
