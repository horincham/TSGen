from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


HARTREE_TO_KCALMOL = 627.5094740631


@dataclass(frozen=True)
class LabelRecord:
    """A single oracle label (Gaussian energy)."""

    id: str
    species_id: str
    e_hartree: float
    e_kcalmol_abs: float
    # relative energy target used for surrogate training
    e_kcalmol_rel: float
    # bookkeeping / provenance
    iter: int
    job_dir: str
    log_path: str
    status: str = "ok"  # ok|failed
    route: str = ""
    charge: int = 0
    multiplicity: int = 1


class LabelStore:
    """DFT labels and fixed species-reference energies with a CSV export."""

    def __init__(
        self,
        labels_jsonl: Path,
        *,
        reference_path: Path | None = None,
        excluded_ids=None,
        require_optimized: bool = False,
    ):
        self.labels_jsonl = labels_jsonl
        self.labels_csv = labels_jsonl.with_suffix(".csv")
        self.labels_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.reference_path = reference_path
        self.require_optimized = require_optimized
        self.excluded_ids = set(excluded_ids or [])

    @property
    def reference_metadata_path(self) -> Path:
        if self.reference_path is None:
            raise ValueError("A reference path is required for optimized references")
        return self.reference_path.with_suffix(".provenance.json")

    def energy_references(self) -> dict[str, float]:
        if self.reference_path is None or not self.reference_path.exists():
            return {}
        refs = json.loads(self.reference_path.read_text(encoding="utf-8"))
        if not isinstance(refs, dict) or not all(
            isinstance(v, (int, float)) and np.isfinite(v) for v in refs.values()
        ):
            raise ValueError(
                "Energy references must be a mapping of species IDs to finite kcal/mol energies"
            )
        return refs

    def reference_metadata(self) -> dict[str, dict]:
        if self.reference_path is None or not self.reference_metadata_path.exists():
            return {}
        metadata = json.loads(self.reference_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(
                "Reference provenance must be a mapping of species IDs to records"
            )
        from tsgen.utils.hashing import structure_hash

        energies = self.energy_references()
        for species, record in metadata.items():
            coords = np.asarray(record["coords"], dtype=float)
            symbols = record["symbols"]
            if coords.shape != (len(symbols), 3) or not np.isfinite(coords).all():
                raise ValueError(f"Invalid optimized reference geometry for {species}")
            if structure_hash(symbols, coords) != record["geometry_id"]:
                raise ValueError(f"Optimized reference geometry changed for {species}")
            energy = float(record["energy_kcalmol"])
            if not np.isfinite(energy) or (
                species in energies
                and not np.isclose(energy, energies[species], rtol=0.0, atol=1e-9)
            ):
                raise ValueError(
                    f"Fixed optimized reference energy changed for {species}"
                )
        return metadata

    def record_reference(self, record: LabelRecord, frame) -> None:
        """Freeze a converged species optimization and its calculation provenance."""
        from tsgen.data.xyz import Frame, write_xyz_multiframe
        from tsgen.outerloop.io import atomic_json, sha256
        from tsgen.utils.hashing import structure_hash

        energies = self.energy_references()
        metadata = self.reference_metadata()
        species = record.species_id
        value = float(record.e_kcalmol_abs)
        if not np.isfinite(value) or not np.isclose(
            value, record.e_hartree * HARTREE_TO_KCALMOL, rtol=0.0, atol=1e-8
        ):
            raise ValueError(f"Invalid optimized reference energy for {species}")
        if species in energies and not np.isclose(
            energies[species], value, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                f"A fixed optimized reference already exists for {species}"
            )
        geometry_id = structure_hash(frame.symbols, frame.coords)
        directory = Path(record.job_dir).resolve()
        geometry = directory / "optimized_reference.xyz"
        log = Path(record.log_path).resolve()
        reference = dict(
            energy_kcalmol=value,
            energy_hartree=float(record.e_hartree),
            geometry_id=geometry_id,
            symbols=list(frame.symbols),
            coords=frame.coords.tolist(),
            charge=int(record.charge),
            multiplicity=int(record.multiplicity),
            route=record.route,
            job_dir=str(directory),
            log_path=str(log),
            log_sha256=sha256(log),
            geometry_path=str(geometry),
        )
        if species in metadata and metadata[species] != reference:
            raise ValueError(
                f"A fixed optimized reference geometry or calculation changed for {species}"
            )
        directory.mkdir(parents=True, exist_ok=True)
        write_xyz_multiframe(
            geometry,
            [Frame(frame.symbols, frame.coords, f"id={geometry_id} species={species}")],
            precision=16,
        )
        metadata[species] = reference
        # Writing provenance first lets collection finish the energy-map update after interruption.
        atomic_json(self.reference_metadata_path, metadata)
        atomic_json(self.reference_path, energies | {species: value})

    def _validate_reference_states(self, labels: pd.DataFrame) -> None:
        if labels.empty:
            return
        from tsgen.outerloop.methods import _method_identity

        metadata = self.reference_metadata()
        for species, group in labels.groupby("species_id"):
            states = set(
                zip(group["charge"].astype(int), group["multiplicity"].astype(int))
            )
            if len(states) > 1:
                raise ValueError(
                    f"Labels for {species} use different charge or multiplicity"
                )
            routes = [route for route in group["route"] if str(route).strip()]
            if len({_method_identity(route) for route in routes}) > 1:
                raise ValueError(f"Labels for {species} use different DFT methods")
            reference = metadata.get(str(species))
            if reference is None:
                continue
            if states != {(reference["charge"], reference["multiplicity"])}:
                raise ValueError(
                    f"Optimized reference and labels have different electronic states for {species}"
                )
            method = _method_identity(reference["route"])
            if any(_method_identity(route) != method for route in group["route"]):
                raise ValueError(
                    f"Optimized reference and labels use different DFT methods for {species}"
                )

    def _check_references(self, species_ids, refs: dict[str, float]) -> None:
        missing = sorted(set(species_ids) - refs.keys())
        if missing:
            raise ValueError(
                f"Missing optimized DFT reference for species: {', '.join(missing)}"
            )

    def load(self) -> pd.DataFrame:
        if not self.labels_jsonl.exists():
            return pd.DataFrame(
                columns=[
                    "id",
                    "species_id",
                    "e_hartree",
                    "e_kcalmol_abs",
                    "e_kcalmol_rel",
                    "iter",
                    "job_dir",
                    "log_path",
                    "status",
                    "route",
                    "charge",
                    "multiplicity",
                ]
            )
        rows = []
        for line in self.labels_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(
                columns=[
                    "id",
                    "species_id",
                    "e_hartree",
                    "e_kcalmol_abs",
                    "e_kcalmol_rel",
                    "iter",
                    "job_dir",
                    "log_path",
                    "status",
                    "route",
                    "charge",
                    "multiplicity",
                ]
            )
        df = pd.DataFrame(rows)
        # last-write-wins by id (resubmissions)
        df = (
            df.sort_values(["id", "iter"])
            .drop_duplicates("id", keep="last")
            .reset_index(drop=True)
        )
        if {"species_id", "e_kcalmol_abs"}.issubset(df.columns):
            if self.reference_path is None:
                df["e_kcalmol_rel"] = self.compute_relative_energies(
                    df, reference="per_species_min"
                )
            else:
                refs = self.energy_references()
                if self.require_optimized:
                    self._check_references(df["species_id"].astype(str), refs)
                    self._validate_reference_states(df)
                    df["reference_hartree"] = (
                        df["species_id"].astype(str).map(refs) / HARTREE_TO_KCALMOL
                    )
                    metadata = self.reference_metadata()
                    for column, key in (
                        ("reference_geometry_path", "geometry_path"),
                        ("reference_geometry_id", "geometry_id"),
                        ("reference_log_path", "log_path"),
                        ("reference_route", "route"),
                    ):
                        df[column] = df["species_id"].map(
                            {sid: row[key] for sid, row in metadata.items()}
                        )
                    df["reference_provenance_path"] = str(
                        self.reference_metadata_path.resolve()
                    )
                else:
                    training = df[~df["id"].astype(str).isin(self.excluded_ids)]
                    for species, group in training.groupby("species_id"):
                        refs.setdefault(
                            str(species), float(group["e_kcalmol_abs"].min())
                        )
                    from tsgen.outerloop.io import atomic_json

                    atomic_json(self.reference_path, refs)
                df["e_kcalmol_rel"] = df["e_kcalmol_abs"].astype(float) - df[
                    "species_id"
                ].astype(str).map(refs)
        return df

    def append(self, records: Iterable[LabelRecord]) -> None:
        recs = list(records)
        if not recs:
            return
        if self.require_optimized:
            self._check_references(
                (r.species_id for r in recs), self.energy_references()
            )
            current = self.load()
            incoming = pd.DataFrame([asdict(r) for r in recs])
            candidate = (
                pd.concat([current, incoming], ignore_index=True)
                if not current.empty
                else incoming
            )
            self._validate_reference_states(candidate)
        with self.labels_jsonl.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        # refresh CSV for convenience
        df = self.load()
        df.to_csv(self.labels_csv, index=False)

    @staticmethod
    def compute_relative_energies(
        df_labels: pd.DataFrame, reference: str = "per_species_min"
    ) -> pd.Series:
        """Compute relative energies (kcal/mol) from absolute kcal/mol.

        reference:
          - per_species_min: for each species, subtract current minimum absolute energy
          - global_min: subtract global minimum
        """
        if df_labels.empty:
            return pd.Series([], dtype=float)
        if reference == "global_min":
            ref = float(df_labels["e_kcalmol_abs"].min())
            return df_labels["e_kcalmol_abs"].astype(float) - ref

        # per_species_min
        mins = df_labels.groupby("species_id")["e_kcalmol_abs"].min()
        return df_labels.apply(
            lambda r: float(r["e_kcalmol_abs"]) - float(mins.loc[str(r["species_id"])]),
            axis=1,
        )
