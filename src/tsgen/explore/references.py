from __future__ import annotations

import csv
import json
import re
from logging import Logger, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tsgen.data.xyz import Frame, read_xyz_multiframe, write_xyz_multiframe
from tsgen.explore.config import GaussianCfg
from tsgen.explore.label_store import LabelStore
from tsgen.explore.oracle_gaussian import (
    GaussianJobSpec,
    GaussianOracle,
    _resolve_charge_mult,
)
from tsgen.explore.species import species_signature
from tsgen.outerloop.methods import _method_identity
from tsgen.trainset.gaussian import parse_final_orientation_frame
from tsgen.utils.geometry import composition_key


_SPECIES_ID = re.compile(r"[A-Za-z0-9_+.-]+")


def export_optimized_references(labels: LabelStore, output: Path) -> int:
    """Export the finalized species references from their saved provenance."""
    energies = labels.energy_references()
    records = [(sid, record) for sid, record in sorted(labels.reference_metadata().items())
               if sid in energies]
    if not records:
        return 0

    output.mkdir(parents=True, exist_ok=True)
    fields = ["geometry_id", "charge", "multiplicity", "energy_hartree", "route", "log_sha256"]
    with TemporaryDirectory(dir=output) as staging:
        staged = Path(staging)
        with (staged / "index.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["species_id", "xyz_file", *fields])
            writer.writeheader()
            for sid, record in records:
                if not _SPECIES_ID.fullmatch(sid):
                    raise ValueError(f"Invalid species ID: {sid!r}")
                filename = f"{sid}.xyz"
                comment = (
                    f"id={record['geometry_id']} species={sid} source=gaussian_opt "
                    f"charge={record['charge']} multiplicity={record['multiplicity']}"
                )
                frame = Frame(record["symbols"], np.asarray(record["coords"], dtype=float), comment)
                write_xyz_multiframe(staged / filename, [frame], precision=16)
                writer.writerow(dict(species_id=sid, xyz_file=filename, **{key: record[key] for key in fields}))
        for path in staged.glob("*.xyz"):
            path.replace(output / path.name)
        (staged / "index.csv").replace(output / "index.csv")
    return len(records)


class OptimizedReferences:
    """Prepare one geometry optimization per newly acquired species."""

    def __init__(
        self,
        cfg: GaussianCfg,
        bond_tol: float,
        jobs_dir: Path,
        labels: LabelStore,
        timeout_seconds: float | None = None,
        fingerprint_version: str = "wl_v2",
        logger: Logger | None = None,
    ):
        self.jobs_dir = jobs_dir
        self.bond_tol = bond_tol
        self.labels = labels
        self.fingerprint_version = fingerprint_version
        self.logger = logger or getLogger(__name__)
        self.failures: dict[str, str] = {}
        route = re.sub(
            r"(?<![\w/-])sp(?![\w/-])", "", cfg.route, flags=re.IGNORECASE
        ).strip()
        self.oracle = GaussianOracle(
            cfg.model_copy(update={"route": f"{route} Opt"}), timeout_seconds
        )

    def job_dir(self, species_id: str) -> Path:
        if not _SPECIES_ID.fullmatch(species_id):
            raise ValueError(f"Invalid species ID: {species_id!r}")
        return self.jobs_dir / f"job_{species_id}"

    def prepare(self, candidate: GaussianJobSpec, *, reference_frame: Frame | None = None) -> None:
        """Prepare the species reference, optionally starting from another configuration."""
        if candidate.species_id in self.labels.energy_references():
            metadata = self.labels.reference_metadata().get(candidate.species_id)
            if metadata is not None:
                q, m, _ = _resolve_charge_mult(
                    candidate.frame,
                    self.oracle.cfg.charge,
                    self.oracle.cfg.multiplicity,
                )
                if (q, m) != (metadata["charge"], metadata["multiplicity"]):
                    raise ValueError(
                        f"Electronic state differs from the optimized reference for {candidate.species_id}"
                    )
                if _method_identity(self.oracle.cfg.route) != _method_identity(
                    metadata["route"]
                ):
                    raise ValueError(
                        f"DFT method differs from the optimized reference for {candidate.species_id}"
                    )
            return
        directory = self.job_dir(candidate.species_id)
        if not (directory / "meta.json").exists():
            frame = candidate.frame
            if reference_frame is not None:
                if any(
                    species_signature(
                        geometry, self.bond_tol, fingerprint_version=self.fingerprint_version
                    ) != candidate.species_id
                    for geometry in (candidate.frame, reference_frame)
                ):
                    raise ValueError(f"Reference geometry fingerprint differs for {candidate.species_id}")
                cfg = self.oracle.cfg
                sampled_state = _resolve_charge_mult(candidate.frame, cfg.charge, cfg.multiplicity)[:2]
                reference_state = _resolve_charge_mult(reference_frame, cfg.charge, cfg.multiplicity)[:2]
                if sampled_state != reference_state:
                    raise ValueError(f"Electronic state differs for reference geometry of {candidate.species_id}")
                if composition_key(candidate.frame) != composition_key(reference_frame):
                    raise ValueError(f"Composition differs for reference geometry of {candidate.species_id}")
                frame = reference_frame
            spec = GaussianJobSpec(
                id=candidate.species_id,
                species_id=candidate.species_id,
                frame=frame,
            )
            self.oracle.write_job(directory, spec)
        self.oracle.submit(directory, allow_calculation_failure=True)

    def active_job_dir(self, species_id: str) -> Path:
        directory = self.job_dir(species_id)
        retry = directory / "cartesian"
        return retry if (retry / "meta.json").is_file() else directory

    def _prepare_cartesian_retry(self, directory: Path, species_id: str) -> Path:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        route = re.sub(r"\bOpt$", "Opt=Cartesian", meta["route"], flags=re.IGNORECASE)
        if route == meta["route"]:
            raise ValueError(f"Cannot change the reference optimization coordinates: {directory}")
        cfg = self.oracle.cfg.model_copy(update={
            "route": route, "charge": meta["charge"], "multiplicity": meta["multiplicity"],
        })
        frame, = read_xyz_multiframe(directory / "structure.xyz")
        retry = directory / "cartesian"
        GaussianOracle(cfg, self.oracle.timeout_seconds).write_job(
            retry, GaussianJobSpec(species_id, species_id, frame)
        )
        self.logger.warning("[references] Retrying %s with Opt=Cartesian | %s", species_id, retry)
        return retry

    def collect(self) -> None:
        refs = self.labels.energy_references()
        provenance = self.labels.reference_metadata()
        failures = {}
        for directory in sorted(self.jobs_dir.glob("job_*")):
            if not (directory / "meta.json").is_file():
                continue
            species_id = directory.name.removeprefix("job_")
            if species_id in refs and species_id in provenance:
                continue
            active = self.active_job_dir(species_id)
            status = self.oracle.job_status(active)
            if status == "pending":
                self.oracle.submit(active, allow_calculation_failure=True)
                status = self.oracle.job_status(active)
            log_path = active / "input.log"
            if (
                active == directory and status == "failed" and log_path.is_file()
                and "Error in internal coordinate system" in log_path.read_text(errors="replace")
            ):
                active = self._prepare_cartesian_retry(directory, species_id)
                self.oracle.submit(active, allow_calculation_failure=True)
                log_path = active / "input.log"
                status = self.oracle.job_status(active)
            if status == "pending":
                continue
            reason = "Gaussian optimization failed"
            if status == "ok":
                record = self.oracle.collect_label(active, iter_idx=-1)
                log = log_path.read_text(errors="replace")
                frame = parse_final_orientation_frame(log)
                if record is None or "Optimization completed" not in log or frame is None:
                    reason = "No converged geometry"
                elif species_signature(
                    frame, self.bond_tol, fingerprint_version=self.fingerprint_version
                ) != species_id:
                    reason = "Optimized geometry changed the molecular graph"
                else:
                    self.labels.record_reference(record, frame)
                    continue
            failures[species_id] = f"{reason} | {log_path}"
            if self.failures.get(species_id) != failures[species_id]:
                self.logger.warning("[references] %s unavailable | %s", species_id, failures[species_id])
        self.failures = failures
