"""Collect a bounded diagnostic archive; never modify runs or launch calculations."""

from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def reason_group(reason):
    return re.sub(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", "<number>", reason)


def collect(run_dir, output, per_reason=3, max_groups=20, max_file_mb=32):
    run = Path(run_dir).resolve()
    output = Path(output).resolve()
    if not (run / "config.yaml").is_file():
        raise ValueError("Run config.yaml not found")
    if output.is_relative_to(run):
        raise ValueError("Place the diagnostic archive outside the source run")
    if per_reason < 1 or max_groups < 1 or max_file_mb < 1:
        raise ValueError("Collection limits must be positive")
    if output.exists():
        raise FileExistsError(f"Preserve existing archive: {output}")
    rows = []
    groups = defaultdict(list)
    successful = []
    counts = Counter()
    for folder in sorted((run / "samples").glob("sample_*")):
        if not folder.is_dir():
            continue
        status = read_json(folder / "status.json")
        state = status.get("status", "missing_status")
        reason = status.get("failure_reason") or ""
        if state not in {"success", "prepared", "pending", "submitted", "running"}:
            result = read_json(folder / "result.json")
            state = result.get("status", state)
            reason = result.get("failure_reason") or reason
        counts[state] += 1
        row = dict(sample_id=folder.name, status=state, failure_reason=reason)
        rows.append(row)
        if state == "success":
            successful.append(folder.name)
        elif state not in {"prepared", "pending", "submitted", "running"}:
            groups[reason_group(reason or "(no reason recorded)")].append(row)
    selected = set(successful[:2])
    summaries = []
    ranked = sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    for index, (group, items) in enumerate(ranked):
        examples = items[:per_reason] if index < max_groups else []
        selected.update(row["sample_id"] for row in examples)
        summaries.append(
            dict(
                reason_group=group,
                count=len(items),
                original_reason=items[0]["failure_reason"],
                sample_examples=[row["sample_id"] for row in examples],
            )
        )
    report = dict(
        run_dir=str(run),
        created=datetime.datetime.now().isoformat(),
        counts=dict(counts),
        total_samples=len(rows),
        failure_groups=summaries,
        selected_sample_ids=sorted(selected),
        per_reason=per_reason,
        max_groups=max_groups,
        original_run_unchanged=True,
        calculations_launched=0,
        skipped_files=[],
        source_hashes={},
    )
    files = []
    for relative in [
        "config.yaml",
        "state.json",
        "provenance.json",
        "inputs/ffield",
        "inputs/geo",
        "inputs/trainset.in",
        "inputs/control",
        "inputs/params-ml.candidates",
        "inputs/objective_manifest.json",
        "inputs/training_item_coverage.json",
        "inputs/sampling_quantization.json",
        "tables/parameter_samples.csv",
        "tables/sample_losses.csv",
        "tables/sample_status.csv",
        "slurm/submissions.json",
        "slurm/submission_limit_events.json",
    ]:
        files.append(run / relative)
    names = [
        "result.json",
        "status.json",
        "sample.json",
        "ffield",
        "fort.13",
        "fort.99",
        "stdout.log",
        "stderr.log",
        "fort.8",
    ]
    for ident in ["baseline", *sorted(selected)]:
        files.extend(run / "samples" / ident / name for name in names)
    # Include scheduler output when a worker stops before writing sample logs.
    for directory in (run / "logs", run / "slurm"):
        for suffix in ("out", "err"):
            files.extend(sorted(directory.glob(f"*.{suffix}"))[:5])
    sources = []
    try:
        spec = importlib.util.find_spec("tsgen")
        if spec and spec.submodule_search_locations:
            root = Path(next(iter(spec.submodule_search_locations)))
            report["loaded_package_path"] = str(root)
            sources = [
                root / "sensitivity" / name
                for name in [
                    "runner.py",
                    "fort_parser.py",
                    "slurm.py",
                    "evaluation_integrity.py",
                ]
            ]
    except (ImportError, ValueError):
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz", dereference=True) as archive:
        for path in files:
            if not path.is_file():
                continue
            if (
                not path.resolve().is_relative_to(run)
                or path.stat().st_size > max_file_mb * 1024**2
            ):
                report["skipped_files"].append(str(path))
                continue
            archive.add(path, arcname=path.relative_to(run).as_posix(), recursive=False)
        for path in sources:
            if path.is_file():
                report["source_hashes"][path.name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                archive.add(path, arcname="loaded_source/" + path.name, recursive=False)

        def add_bytes(name, payload):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream, fieldnames=["sample_id", "status", "failure_reason"]
        )
        writer.writeheader()
        writer.writerows(rows)
        add_bytes(
            "diagnosis/all_sample_statuses.csv", stream.getvalue().encode("utf-8")
        )
        add_bytes(
            "diagnosis/summary.json",
            json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
        )
    return dict(
        archive=str(output),
        bytes=output.stat().st_size,
        counts=dict(counts),
        failure_groups=len(groups),
        sample_logs_included=len(selected),
        run_unchanged=True,
        calculations_launched=0,
    )
