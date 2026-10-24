from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


HARTREE_TO_KCALMOL = 627.5094740631  # CODATA (approx), sufficient for AL bookkeeping


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
    """Append-only label store.

    Stored as JSONL to preserve provenance and enable robust resume.
    A CSV export is also written for analysis/plotting.
    """

    def __init__(self, labels_jsonl: Path):
        self.labels_jsonl = labels_jsonl
        self.labels_csv = labels_jsonl.with_suffix(".csv")
        self.labels_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.labels_jsonl.exists():
            return pd.DataFrame(columns=[
                "id","species_id","e_hartree","e_kcalmol_abs","e_kcalmol_rel",
                "iter","job_dir","log_path","status","route","charge","multiplicity"
            ])
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
            return pd.DataFrame(columns=[
                "id","species_id","e_hartree","e_kcalmol_abs","e_kcalmol_rel",
                "iter","job_dir","log_path","status","route","charge","multiplicity"
            ])
        df = pd.DataFrame(rows)
        # last-write-wins by id (resubmissions)
        df = df.sort_values(["id","iter"]).drop_duplicates("id", keep="last").reset_index(drop=True)
        # Relative energies are derived from the complete current label set.
        # Recompute on every load so the append-only JSONL cannot retain stale
        # per-species references from the iteration in which a row was added.
        if {"species_id", "e_kcalmol_abs"}.issubset(df.columns):
            df["e_kcalmol_rel"] = self.compute_relative_energies(df, reference="per_species_min")
        return df

    def append(self, records: Iterable[LabelRecord]) -> None:
        recs = list(records)
        if not recs:
            return
        with self.labels_jsonl.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        # refresh CSV for convenience
        df = self.load()
        df.to_csv(self.labels_csv, index=False)

    @staticmethod
    def compute_relative_energies(df_labels: pd.DataFrame, reference: str = "per_species_min") -> pd.Series:
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
        return df_labels.apply(lambda r: float(r["e_kcalmol_abs"]) - float(mins.loc[str(r["species_id"])]), axis=1)
