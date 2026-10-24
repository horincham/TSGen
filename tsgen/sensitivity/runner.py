from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import pandas as pd

from tsgen.sensitivity.config import (
    SensitivityConfig,
    load_sensitivity_config,
    save_sensitivity_config,
)
from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.fort_parser import (
    build_evaluation_result,
    item_loss_rows,
    loss_summary_row,
)
from tsgen.sensitivity.models import EvaluationResult, ItemLoss, SampleRecord, TrainingItem
from tsgen.sensitivity.params_ml import (
    build_auto_candidates,
    build_rule_candidates,
    candidate_audit_rows,
    parse_params_ml,
    validate_parameter_bounds,
    write_params_ml,
)
from tsgen.sensitivity.paths import SensitivityPaths
from tsgen.sensitivity.sampling import generate_samples, sample_table, write_sample_ffield
from tsgen.sensitivity.slurm import (
    slurm_job_is_active,
    submit_slurm,
    write_manual_script,
    write_slurm_arrays,
)
from tsgen.sensitivity.trainset_map import (
    parse_geo_inventory,
    parse_trainset,
    validate_training_inputs,
    write_reaxff_compatible_geo,
)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)


def _atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_argv(command: str | list[str]) -> list[str]:
    if isinstance(command, list):
        return [str(part) for part in command]
    return shlex.split(command, posix=os.name != "nt")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path, default: dict | None = None) -> dict | None:
    try:
        return _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(default) if default is not None else None


def _evaluation_from_dict(payload: dict) -> EvaluationResult:
    data = dict(payload)
    data["item_losses"] = tuple(ItemLoss(**item) for item in data.get("item_losses", []))
    return EvaluationResult(**data)


class SensitivityRun:
    def __init__(
        self,
        cfg: SensitivityConfig,
        paths: SensitivityPaths,
        config_source: Path | None = None,
    ):
        self.cfg = cfg
        self.paths = paths
        self.config_source = config_source
        self.document: FFieldDocument | None = None
        self.bounds = []
        self.training_items: list[TrainingItem] = []
        self.samples: list[SampleRecord] = []

    @classmethod
    def from_config(cls, path: str | Path) -> "SensitivityRun":
        source = Path(path).resolve()
        cfg = load_sensitivity_config(source)
        return cls(cfg, SensitivityPaths.next(cfg.run.out_dir, cfg.run.name), source)

    @classmethod
    def from_run_dir(cls, path: str | Path) -> "SensitivityRun":
        paths = SensitivityPaths.from_run_dir(path)
        if not paths.config_file.exists():
            raise FileNotFoundError(f"Missing sensitivity run config: {paths.config_file}")
        run = cls(load_sensitivity_config(paths.config_file), paths, paths.config_file)
        run._load_prepared_metadata()
        return run

    def _require_real_inputs(self) -> None:
        missing = []
        for name in ("ffield", "geo", "trainset_in", "control"):
            value = getattr(self.cfg.inputs, name)
            if value is None or not Path(value).is_file():
                missing.append(f"inputs.{name}={value}")
        if self.cfg.parameter_space.mode == "from_params_ml":
            path = self.cfg.parameter_space.params_ml
            if path is None or not path.is_file():
                missing.append(f"parameter_space.params_ml={path}")
        if missing:
            raise FileNotFoundError("Missing sensitivity inputs: " + "; ".join(missing))
        if self.cfg.reaxff.executable_source is not None and not self.cfg.reaxff.executable_source.is_file():
            raise FileNotFoundError(f"Missing ReaxFF executable/wrapper source: {self.cfg.reaxff.executable_source}")

    def doctor(self, run_baseline: bool = False) -> dict:
        self._require_real_inputs()
        document = FFieldDocument.from_path(self.cfg.inputs.ffield)
        with self.cfg.inputs.ffield.open("r", encoding="utf-8", newline="") as handle:
            original_ffield = handle.read()
        if document.render({}) != original_ffield:
            raise ValueError("ffield no-change round trip is not identical")
        inventory = parse_geo_inventory(self.cfg.inputs.geo)
        items = parse_trainset(self.cfg.inputs.trainset_in)
        validate_training_inputs(inventory, items)
        if self.cfg.parameter_space.mode == "from_params_ml":
            bounds = parse_params_ml(self.cfg.parameter_space.params_ml)
            validate_parameter_bounds(
                bounds,
                document,
                self.cfg.parameter_space.require_initial_inside_bounds,
            )
        elif self.cfg.parameter_space.mode == "auto_from_ffield":
            bounds, audit = build_auto_candidates(
                document, self.cfg.parameter_space, inventory, items
            )
            validate_parameter_bounds(
                bounds,
                document,
                self.cfg.parameter_space.require_initial_inside_bounds,
            )
        else:
            bounds, audit = build_rule_candidates(
                document, self.cfg.parameter_space, inventory, items
            )
            unresolved = [row for row in audit if row["status"] == "unresolved_bound"]
            if unresolved:
                raise ValueError(f"{len(unresolved)} rule-selected parameters have unresolved bounds")
            validate_parameter_bounds(bounds, document)
        report = {
            "status": "ok",
            "n_parameters_in_ffield": len(document.parameters),
            "n_candidates": len(bounds),
            "n_structures": len(inventory),
            "n_training_items": len(items),
            "sections": {
                section: sum(item.section == section for item in items)
                for section in ("ENERGY", "GEOMETRY", "CHARGE")
            },
        }
        if run_baseline:
            report["baseline"] = "baseline is executed by prepare in the persistent run directory"
        return report

    def _snapshot_inputs(self) -> None:
        mapping = {
            "ffield": self.cfg.inputs.ffield,
            "trainset.in": self.cfg.inputs.trainset_in,
            "control": self.cfg.inputs.control,
        }
        for name, source in mapping.items():
            shutil.copy2(source, self.paths.inputs_dir / name)
        write_reaxff_compatible_geo(
            self.cfg.inputs.geo,
            self.paths.inputs_dir / "geo",
        )
        if self.cfg.reaxff.executable_source is not None:
            shutil.copy2(
                self.cfg.reaxff.executable_source,
                self.paths.inputs_dir / self.cfg.reaxff.executable_source.name,
            )

    def _load_source_model(self) -> tuple[FFieldDocument, dict[str, set[str]], list[TrainingItem]]:
        document = FFieldDocument.from_path(self.paths.inputs_dir / "ffield")
        inventory = parse_geo_inventory(self.paths.inputs_dir / "geo")
        items = parse_trainset(self.paths.inputs_dir / "trainset.in")
        validate_training_inputs(inventory, items)
        return document, inventory, items

    def _write_training_items(self) -> None:
        atomic_write_json(
            self.paths.inputs_dir / "training_items.json",
            [asdict(item) for item in self.training_items],
        )

    def _quantize_samples_for_ffield(self, records: list[SampleRecord]) -> list[SampleRecord]:
        quantized_records = []
        unique_values = {
            bound.coordinate.parameter_id: set()
            for bound in self.bounds
        }
        changed_values = 0
        maximum_absolute_change = 0.0
        maximum_relative_change = 0.0
        for record in records:
            physical_values = []
            normalized_values = []
            for bound, requested in zip(self.bounds, record.physical_values, strict=True):
                actual = self.document.quantize(bound.coordinate, requested)
                physical_values.append(actual)
                unique_values[bound.coordinate.parameter_id].add(actual)
                normalized_values.append(
                    (actual - bound.lower_bound) / (bound.upper_bound - bound.lower_bound)
                )
                absolute_change = abs(actual - requested)
                relative_change = absolute_change / max(abs(requested), 1.0e-12)
                if absolute_change:
                    changed_values += 1
                maximum_absolute_change = max(maximum_absolute_change, absolute_change)
                maximum_relative_change = max(maximum_relative_change, relative_change)
            quantized_records.append(
                SampleRecord(
                    sample_id=record.sample_id,
                    normalized_values=tuple(normalized_values),
                    physical_values=tuple(physical_values),
                    is_initial=record.is_initial,
                )
            )
        atomic_write_json(
            self.paths.inputs_dir / "sampling_quantization.json",
            {
                "sample_count": len(records),
                "parameter_count": len(self.bounds),
                "values_checked": len(records) * len(self.bounds),
                "values_changed_by_ffield_precision": changed_values,
                "maximum_absolute_change": maximum_absolute_change,
                "maximum_relative_change": maximum_relative_change,
                "hsic_uses_quantized_values": True,
                "unique_levels_by_parameter": {
                    parameter_id: len(values)
                    for parameter_id, values in unique_values.items()
                },
                "constant_after_quantization": [
                    parameter_id
                    for parameter_id, values in unique_values.items()
                    if len(values) == 1
                ],
            },
        )
        return quantized_records

    def _stage_common(self, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        for name in ("geo", "trainset.in", "control"):
            source = self.paths.inputs_dir / name
            target = work_dir / name
            if target.exists() or target.is_symlink():
                target.unlink()
            if self.cfg.reaxff.stage_mode == "symlink":
                try:
                    target.symlink_to(os.path.relpath(source, work_dir))
                    continue
                except OSError:
                    pass
            shutil.copy2(source, target)
        executable = self.cfg.reaxff.executable_source
        if executable is not None:
            source = self.paths.inputs_dir / executable.name
            target = work_dir / executable.name
            self._stage_alias(source, target)
            try:
                target.chmod(0o755)
            except OSError:
                pass
        self._stage_alias(
            self.paths.inputs_dir / "training_items.json",
            work_dir / "training_items.json",
        )
        self._stage_alias(work_dir / "geo", work_dir / "fort.3")
        (work_dir / "fort.20").write_text("0\n", encoding="ascii")
        (work_dir / "fort.35").write_text(f"{self.cfg.run.seed}\n", encoding="ascii")

    def _prepare_work_dir(self, work_dir: Path, record: SampleRecord) -> None:
        self._stage_common(work_dir)
        write_sample_ffield(self.document, self.bounds, record, work_dir / "ffield")
        self._stage_alias(work_dir / "ffield", work_dir / "fort.4")
        atomic_write_json(work_dir / "sample.json", asdict(record))
        atomic_write_json(work_dir / "status.json", {"sample_id": record.sample_id, "status": "prepared"})

    def _stage_alias(self, source: Path, target: Path) -> None:
        if target.exists() or target.is_symlink():
            target.unlink()
        if self.cfg.reaxff.stage_mode == "symlink":
            try:
                target.symlink_to(os.path.relpath(source, target.parent))
                return
            except OSError:
                pass
        shutil.copy2(source, target)

    def prepare(self, run_baseline: bool = True) -> None:
        self._require_real_inputs()
        self.paths.ensure()
        save_sensitivity_config(self.cfg, self.paths.config_file)
        self._snapshot_inputs()
        self.document, inventory, self.training_items = self._load_source_model()
        pd.DataFrame(self.document.catalog_rows()).to_csv(self.paths.parameter_catalog, index=False)
        if self.cfg.parameter_space.mode == "from_params_ml":
            supplied = parse_params_ml(self.cfg.parameter_space.params_ml)
            self.bounds = validate_parameter_bounds(
                supplied,
                self.document,
                self.cfg.parameter_space.require_initial_inside_bounds,
            )
            audit = candidate_audit_rows(self.bounds, self.document)
        elif self.cfg.parameter_space.mode == "auto_from_ffield":
            self.bounds, audit = build_auto_candidates(
                self.document,
                self.cfg.parameter_space,
                inventory,
                self.training_items,
            )
        else:
            self.bounds, audit = build_rule_candidates(
                self.document,
                self.cfg.parameter_space,
                inventory,
                self.training_items,
            )
        pd.DataFrame(audit).to_csv(self.paths.candidate_parameter_audit, index=False)
        unresolved = [row for row in audit if row.get("status") == "unresolved_bound"]
        if unresolved:
            raise ValueError(
                f"{len(unresolved)} candidate bounds are unresolved; inspect "
                f"{self.paths.candidate_parameter_audit}"
            )
        validate_parameter_bounds(
            self.bounds,
            self.document,
            self.cfg.parameter_space.require_initial_inside_bounds,
        )
        write_params_ml(self.paths.inputs_dir / "params-ml.candidates", self.bounds)
        self._write_training_items()
        initial = [self.document.get(bound.coordinate) for bound in self.bounds]
        generated_samples = generate_samples(
            self.bounds,
            initial,
            self.cfg.sampling.samples,
            self.cfg.run.seed,
            self.cfg.sampling.center_initial_ffield,
        )
        self.samples = self._quantize_samples_for_ffield(generated_samples)
        _atomic_dataframe(
            self.paths.parameter_samples,
            sample_table(self.samples, self.bounds, self.cfg.run.seed),
        )
        if run_baseline:
            print("[sensitivity] running baseline ReaxFF evaluation", flush=True)
            baseline = SampleRecord("baseline", tuple(), tuple(initial), True)
            self._prepare_work_dir(self.paths.baseline_dir, baseline)
            baseline_result = self._execute_work_dir(self.paths.baseline_dir, "baseline")
            if baseline_result.status != "success":
                raise RuntimeError(
                    f"Baseline ReaxFF evaluation failed: {baseline_result.failure_reason}"
                )
            atomic_write_json(
                self.paths.inputs_dir / "training_item_coverage.json",
                {
                    "declared_trainset_rows": len(self.training_items),
                    "reaxff_objective_rows": len(baseline_result.item_losses),
                    "coverage_fraction": (
                        len(baseline_result.item_losses) / len(self.training_items)
                        if self.training_items
                        else 0.0
                    ),
                    "authority": "fort.99 rows contributing to the fort.13 objective",
                },
            )
            print(
                "[sensitivity] baseline succeeded: "
                f"{len(baseline_result.item_losses)} objective rows",
                flush=True,
            )
        for index, record in enumerate(self.samples, start=1):
            self._prepare_work_dir(self.paths.sample_dir(record.sample_id), record)
            if index % 250 == 0 or index == len(self.samples):
                print(
                    f"[sensitivity] prepared {index}/{len(self.samples)} sample directories",
                    flush=True,
                )
        self._write_provenance()
        atomic_write_json(
            self.paths.state_file,
            {"stage": "prepared", "n_requested": len(self.samples), "run_dir": str(self.paths.run_dir)},
        )

    def _write_provenance(self) -> None:
        hashes = {
            path.name: sha256_file(path)
            for path in self.paths.inputs_dir.iterdir()
            if path.is_file()
        }
        atomic_write_json(
            self.paths.provenance_file,
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "python": sys.version,
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "command": command_argv(self.cfg.reaxff.command),
                "seed": self.cfg.run.seed,
                "sampling": self.cfg.sampling.model_dump(mode="json"),
                "analysis": self.cfg.analysis.model_dump(mode="json"),
                "input_sha256": hashes,
            },
        )

    def _execute_work_dir(self, work_dir: Path, sample_id: str) -> EvaluationResult:
        result_file = work_dir / "result.json"
        existing_payload = _read_optional_json(result_file)
        if existing_payload is not None and self.cfg.execution.reuse_completed:
            existing = _evaluation_from_dict(existing_payload)
            if existing.status == "success":
                return existing
        status_file = work_dir / "status.json"
        status_payload = _read_optional_json(status_file, {})
        status_payload.update({"sample_id": sample_id, "status": "running"})
        atomic_write_json(status_file, status_payload)
        for output_name in (
            self.cfg.reaxff.expected_total_output,
            self.cfg.reaxff.expected_item_output,
        ):
            stale_output = work_dir / output_name
            if stale_output.exists():
                stale_output.unlink()
        started = time.perf_counter()
        return_code = None
        try:
            completed = subprocess.run(
                command_argv(self.cfg.reaxff.command),
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.cfg.reaxff.timeout_seconds,
                shell=False,
                check=False,
            )
            return_code = completed.returncode
            (work_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (work_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                result = EvaluationResult(
                    sample_id,
                    "failed",
                    None,
                    elapsed_seconds=elapsed,
                    failure_reason=f"ReaxFF returned {completed.returncode}",
                    return_code=completed.returncode,
                    ffield_sha256=sha256_file(work_dir / "ffield"),
                )
            else:
                result = build_evaluation_result(
                    sample_id,
                    work_dir / self.cfg.reaxff.expected_total_output,
                    work_dir / self.cfg.reaxff.expected_item_output,
                    self.training_items,
                    elapsed,
                    completed.returncode,
                    sha256_file(work_dir / "ffield"),
                )
        except subprocess.TimeoutExpired as exc:
            result = EvaluationResult(
                sample_id,
                "timeout",
                None,
                elapsed_seconds=time.perf_counter() - started,
                failure_reason=f"ReaxFF timeout after {exc.timeout} seconds",
                return_code=return_code,
                ffield_sha256=sha256_file(work_dir / "ffield"),
            )
        except Exception as exc:
            result = EvaluationResult(
                sample_id,
                "parse_error",
                None,
                elapsed_seconds=time.perf_counter() - started,
                failure_reason=f"{type(exc).__name__}: {exc}",
                return_code=return_code,
                ffield_sha256=sha256_file(work_dir / "ffield"),
            )
        atomic_write_json(result_file, result.to_dict())
        status_payload = _read_optional_json(status_file, {})
        status_payload.update(
            {
                "sample_id": sample_id,
                "status": result.status,
                "failure_reason": result.failure_reason,
            }
        )
        atomic_write_json(
            status_file,
            status_payload,
        )
        return result

    def _load_prepared_metadata(self) -> None:
        if not self.paths.inputs_dir.exists():
            return
        self.document = FFieldDocument.from_path(self.paths.inputs_dir / "ffield")
        self.bounds = validate_parameter_bounds(
            parse_params_ml(self.paths.inputs_dir / "params-ml.candidates"),
            self.document,
            self.cfg.parameter_space.require_initial_inside_bounds,
        )
        payload = _read_json(self.paths.inputs_dir / "training_items.json")
        self.training_items = [
            TrainingItem(
                item_index=item["item_index"],
                section=item["section"],
                structure_ids=tuple(item["structure_ids"]),
                item_label=item["item_label"],
                reference_value=item["reference_value"],
                weight=item["weight"],
            )
            for item in payload
        ]
        table = pd.read_csv(self.paths.parameter_samples)
        normalized_columns = [column for column in table if column.startswith("normalized__")]
        physical_columns = [column for column in table if column.startswith("physical__")]
        self.samples = [
            SampleRecord(
                row.sample_id,
                tuple(float(row[column]) for column in normalized_columns),
                tuple(float(row[column]) for column in physical_columns),
                bool(row.is_initial),
            )
            for _, row in table.iterrows()
        ]

    def evaluate_sample(self, sample_id: str) -> EvaluationResult:
        if not self.training_items:
            self._load_prepared_metadata()
        work_dir = self.paths.sample_dir(sample_id)
        if not work_dir.is_dir():
            raise FileNotFoundError(f"Unknown sensitivity sample: {sample_id}")
        return self._execute_work_dir(work_dir, sample_id)

    def _submission_history(self) -> list[dict[str, Any]]:
        history_file = self.paths.slurm_dir / "submissions.json"
        if not history_file.exists():
            return []
        history = json.loads(history_file.read_text(encoding="utf-8"))
        return history if isinstance(history, list) else []

    def _reconcile_slurm_activity(self) -> None:
        jobs = {
            (entry.get("submission_round"), entry.get("batch")): str(entry["job_id"])
            for entry in self._submission_history()
            if entry.get("job_id") is not None
        }
        active_by_job: dict[str, bool | None] = {}
        for record in self.samples:
            status_file = self.paths.sample_dir(record.sample_id) / "status.json"
            status = _read_optional_json(status_file)
            if status is None:
                continue
            if status.get("status") not in {"submitted", "running"}:
                continue
            job_id = jobs.get(
                (status.get("submission_round"), status.get("array_batch"))
            )
            if job_id is None:
                continue
            if job_id not in active_by_job:
                active_by_job[job_id] = slurm_job_is_active(job_id)
            if active_by_job[job_id] is False:
                atomic_write_json(
                    status_file,
                    {
                        "sample_id": record.sample_id,
                        "status": "prepared",
                        "recovery_reason": f"Slurm job {job_id} is no longer active",
                    },
                )

    def unfinished_sample_ids(self) -> list[str]:
        pending = []
        for record in self.samples:
            result_file = self.paths.sample_dir(record.sample_id) / "result.json"
            result = _read_optional_json(result_file)
            if result is not None and result.get("status") == "success":
                continue
            status_file = self.paths.sample_dir(record.sample_id) / "status.json"
            status = _read_optional_json(status_file, {}).get("status", "pending")
            if status not in {"submitted", "running"}:
                pending.append(record.sample_id)
        return pending

    def active_sample_ids(self) -> list[str]:
        active = []
        for record in self.samples:
            status_file = self.paths.sample_dir(record.sample_id) / "status.json"
            status = _read_optional_json(status_file)
            if status is not None and status.get("status") in {"submitted", "running"}:
                active.append(record.sample_id)
        return active

    def execute_local(self) -> None:
        pending = self.unfinished_sample_ids()
        with ThreadPoolExecutor(max_workers=self.cfg.execution.workers) as executor:
            list(executor.map(self.evaluate_sample, pending))
        self.collect()

    def resume(self) -> None:
        if not self.samples:
            self._load_prepared_metadata()
        mode = self.cfg.execution.mode
        if mode == "slurm":
            self._reconcile_slurm_activity()
            if self.cfg.execution.slurm.sequential_batches:
                active = self.active_sample_ids()
                if active:
                    if self.cfg.execution.wait_for_completion:
                        self._wait_for_sample_ids(active)
                    else:
                        self.collect()
                        return
        pending = self.unfinished_sample_ids()
        if not pending:
            self.collect()
            return
        if mode == "local":
            self.execute_local()
        elif mode == "manual":
            write_manual_script(self.paths, pending)
        else:
            history_file = self.paths.slurm_dir / "submissions.json"
            submissions = self._submission_history()
            submission_round = len(submissions) + 1
            batches = write_slurm_arrays(
                self.paths,
                self.cfg,
                pending,
                submission_tag=f"submission_{submission_round:04d}",
            )
            if self.cfg.execution.submit:
                for batch_index, batch in enumerate(batches, start=1):
                    previous_statuses = {}
                    for sample_id in batch.sample_ids:
                        status_file = self.paths.sample_dir(sample_id) / "status.json"
                        previous_statuses[sample_id] = _read_optional_json(
                            status_file,
                            {"sample_id": sample_id, "status": "pending"},
                        )
                        atomic_write_json(
                            status_file,
                            {
                                "sample_id": sample_id,
                                "status": "submitted",
                                "submission_round": submission_round,
                                "array_batch": batch_index,
                            },
                        )
                    try:
                        job_id = submit_slurm(batch.script)
                    except Exception:
                        for sample_id, previous in previous_statuses.items():
                            atomic_write_json(
                                self.paths.sample_dir(sample_id) / "status.json",
                                previous,
                            )
                        raise
                    print(
                        f"[sensitivity] submitted Slurm array job {job_id} "
                        f"(batch {batch_index}/{len(batches)}, "
                        f"{len(batch.sample_ids)} samples)",
                        flush=True,
                    )
                    submissions.append(
                        {
                            "submission_round": submission_round,
                            "batch": batch_index,
                            "job_id": job_id,
                            "script": str(batch.script),
                            "pending_file": str(batch.pending_file),
                            "n_samples": len(batch.sample_ids),
                        }
                    )
                    atomic_write_json(
                        history_file,
                        submissions,
                    )
                    if self.cfg.execution.slurm.sequential_batches:
                        if self.cfg.execution.wait_for_completion:
                            self._wait_for_sample_ids(
                                batch.sample_ids,
                                job_id=job_id,
                            )
                        elif batch_index < len(batches):
                            break

    def _wait_for_sample_ids(
        self,
        sample_ids: list[str],
        job_id: str | None = None,
    ) -> None:
        terminal_states = {"success", "failed", "timeout", "parse_error"}
        deadline = time.monotonic() + self.cfg.execution.max_wait_seconds
        while True:
            counts = {
                state: 0
                for state in (
                    "submitted",
                    "running",
                    "success",
                    "failed",
                    "timeout",
                    "parse_error",
                )
            }
            for sample_id in sample_ids:
                status_file = self.paths.sample_dir(sample_id) / "status.json"
                state = _read_optional_json(status_file, {}).get(
                    "status", "pending"
                )
                counts[state] = counts.get(state, 0) + 1
            terminal_count = sum(counts.get(state, 0) for state in terminal_states)
            print(
                "[sensitivity] Slurm wave progress: "
                f"success={counts['success']} running={counts['running']} "
                f"submitted={counts['submitted']} failed="
                f"{counts['failed'] + counts['timeout'] + counts['parse_error']} "
                f"total={len(sample_ids)}",
                flush=True,
            )
            if terminal_count >= len(sample_ids):
                self.collect()
                return
            if job_id is not None and slurm_job_is_active(job_id) is False:
                for sample_id in sample_ids:
                    status_file = self.paths.sample_dir(sample_id) / "status.json"
                    status = _read_optional_json(
                        status_file,
                        {"sample_id": sample_id},
                    )
                    if status.get("status") in terminal_states:
                        continue
                    status.update(
                        {
                            "sample_id": sample_id,
                            "status": "parse_error",
                            "failure_reason": (
                                f"Slurm job {job_id} ended without a terminal "
                                "sample status"
                            ),
                        }
                    )
                    atomic_write_json(status_file, status)
                self.collect()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out while waiting for the current ReaxFF Slurm wave. "
                    f"Resume with: python -m tsgen.cli sensitivity resume "
                    f"--run-dir {self.paths.run_dir}"
                )
            time.sleep(self.cfg.execution.poll_seconds)

    def _wait_for_terminal_samples(self) -> None:
        terminal_states = ("success", "failed", "timeout", "parse_error")
        deadline = time.monotonic() + self.cfg.execution.max_wait_seconds
        while True:
            report = self.status()
            terminal_count = sum(report.get(state, 0) for state in terminal_states)
            print(
                "[sensitivity] Slurm progress: "
                f"success={report['success']} running={report['running']} "
                f"submitted={report['submitted']} failed="
                f"{report['failed'] + report['timeout'] + report['parse_error']} "
                f"total={report['n_requested']}",
                flush=True,
            )
            if terminal_count >= report["n_requested"]:
                self.collect()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out while waiting for the ReaxFF Slurm array. "
                    f"Inspect {self.paths.sample_status} or resume with: "
                    f"python -m tsgen.cli sensitivity resume --run-dir {self.paths.run_dir}"
                )
            time.sleep(self.cfg.execution.poll_seconds)

    def collect(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        status_rows = []
        loss_rows = []
        item_rows = []
        for record in self.samples:
            work_dir = self.paths.sample_dir(record.sample_id)
            status_file = work_dir / "status.json"
            status = _read_optional_json(
                status_file,
                {"sample_id": record.sample_id, "status": "pending"},
            )
            status_rows.append(status)
            result_file = work_dir / "result.json"
            result_payload = _read_optional_json(result_file)
            if result_payload is not None:
                result = _evaluation_from_dict(result_payload)
                loss_rows.append(loss_summary_row(result))
                item_rows.extend(item_loss_rows(result))
        statuses = pd.DataFrame(status_rows)
        losses = pd.DataFrame(loss_rows)
        items = pd.DataFrame(item_rows)
        _atomic_dataframe(self.paths.sample_status, statuses)
        _atomic_dataframe(self.paths.sample_losses, losses)
        _atomic_dataframe(self.paths.sample_item_losses, items)
        return losses, items

    def status(self) -> dict[str, Any]:
        if not self.samples:
            self._load_prepared_metadata()
        counts = {
            state: 0
            for state in (
                "pending",
                "prepared",
                "submitted",
                "running",
                "success",
                "failed",
                "timeout",
                "parse_error",
            )
        }
        for record in self.samples:
            status_file = self.paths.sample_dir(record.sample_id) / "status.json"
            state = _read_optional_json(status_file, {}).get("status", "pending")
            counts[state] = counts.get(state, 0) + 1
        total = len(self.samples)
        success_fraction = counts["success"] / total if total else 0.0
        analysis_ready = (
            counts["success"] >= self.cfg.analysis.subsample_size
            and success_fraction >= self.cfg.analysis.minimum_success_fraction
        )
        return {
            "run_dir": str(self.paths.run_dir),
            "n_requested": total,
            **counts,
            "success_fraction": success_fraction,
            "analysis_ready": analysis_ready,
            "resume_command": f"python -m tsgen.cli sensitivity resume --run-dir {self.paths.run_dir}",
        }

    def _finalize_execution(self) -> None:
        wait_for_slurm = (
            self.cfg.execution.mode == "slurm"
            and self.cfg.execution.submit
            and self.cfg.execution.wait_for_completion
        )
        if wait_for_slurm:
            self._wait_for_terminal_samples()
        report = self.status()
        if report["analysis_ready"]:
            self.analyze()
            if self.cfg.outputs.write_figures:
                self.plot()
            if self.cfg.outputs.export_indeedopt_bundle:
                self.export_indeedopt()
        elif wait_for_slurm:
            raise RuntimeError(
                "ReaxFF sampling finished, but too few evaluations succeeded for analysis: "
                f"{report['success']}/{report['n_requested']} successful "
                f"({report['success_fraction']:.1%})."
            )

    def resume_and_finalize(self) -> None:
        self.resume()
        self._finalize_execution()

    def run(self) -> None:
        self.prepare()
        self.resume_and_finalize()

    def analyze(self):
        from tsgen.sensitivity.analysis import analyze_run
        return analyze_run(self.paths, self.cfg)

    def plot(self):
        from tsgen.sensitivity.plotting import plot_sensitivity_figure
        return plot_sensitivity_figure(self.paths)

    def export_indeedopt(self):
        from tsgen.sensitivity.indeedopt import export_indeedopt_bundle
        return export_indeedopt_bundle(self.paths, self.cfg)
