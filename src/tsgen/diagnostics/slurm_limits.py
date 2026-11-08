"""Change only Slurm execution policy on an existing, prepared sensitivity run."""

import copy
import json
from pathlib import Path
import uuid
import yaml


def update(
    run_dir: Path, max_array_size: int, array_parallelism: int, apply: bool = False
):
    if max_array_size < 1 or array_parallelism < 1:
        raise ValueError("Limits must be positive")
    config = run_dir / "config.yaml"
    original = config.read_bytes()
    before = yaml.safe_load(original)
    after = copy.deepcopy(before)
    execution = after.setdefault("execution", {})
    if execution.get("mode") != "slurm":
        raise ValueError("This helper requires an existing Slurm run")
    execution["wait_for_completion"] = True
    policy = execution.setdefault("slurm", {})
    policy.update(
        max_array_size=max_array_size,
        array_parallelism=array_parallelism,
        sequential_batches=True,
    )
    report = dict(
        config=str(config.resolve()),
        apply=apply,
        execution_changes=dict(
            wait_for_completion=True,
            max_array_size=max_array_size,
            array_parallelism=array_parallelism,
            sequential_batches=True,
        ),
        backup=None,
    )
    assert {k: v for k, v in before.items() if k != "execution"} == {
        k: v for k, v in after.items() if k != "execution"
    }
    if apply:
        backup = config.with_name("config.yaml.before_qos_" + uuid.uuid4().hex[:12])
        with backup.open("xb") as handle:
            handle.write(original)
        temporary = config.with_name("config.yaml.qos_" + uuid.uuid4().hex + ".tmp")
        temporary.write_text(
            yaml.safe_dump(after, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        temporary.replace(config)
        report["backup"] = str(backup)
    return report
