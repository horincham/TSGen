from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.paths import SensitivityPaths


@dataclass(frozen=True)
class SlurmArrayBatch:
    script: Path
    pending_file: Path
    sample_ids: list[str]


def write_pending_list(
    paths: SensitivityPaths,
    pending_ids: list[str],
    filename: str = "pending_samples.txt",
) -> Path:
    paths.slurm_dir.mkdir(parents=True, exist_ok=True)
    output = paths.slurm_dir / filename
    output.write_text("\n".join(pending_ids) + ("\n" if pending_ids else ""), encoding="utf-8")
    return output


def write_manual_script(paths: SensitivityPaths, pending_ids: list[str]) -> Path:
    pending = write_pending_list(paths, pending_ids)
    output = paths.slurm_dir / "run_pending_samples.sh"
    output.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'RUN_DIR="{paths.run_dir.resolve().as_posix()}"\n'
        f'while IFS= read -r SAMPLE_ID; do\n'
        '  [ -z "${SAMPLE_ID}" ] && continue\n'
        '  python -m tsgen.cli sensitivity worker --run-dir "${RUN_DIR}" --sample-id "${SAMPLE_ID}"\n'
        f'done < "{pending.resolve().as_posix()}"\n',
        encoding="utf-8",
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
    lines.extend(slurm.extra_lines)
    lines.extend(
        [
            "set -euo pipefail",
            f'RUN_DIR="{paths.run_dir.resolve().as_posix()}"',
            f'SAMPLE_ID="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" '
            f'"${{RUN_DIR}}/slurm/{pending_file.name}")"',
            'python -m tsgen.cli sensitivity worker --run-dir "${RUN_DIR}" --sample-id "${SAMPLE_ID}"',
        ]
    )
    output = paths.slurm_dir / script_name
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    completed = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"sbatch failed: {completed.stderr.strip()}")
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
