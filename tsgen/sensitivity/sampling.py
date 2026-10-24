from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.models import ParameterBound, SampleRecord


def generate_samples(
    bounds: list[ParameterBound],
    initial_values: list[float],
    samples: int,
    seed: int,
    include_initial: bool,
) -> list[SampleRecord]:
    if not bounds:
        raise ValueError("At least one parameter bound is required")
    if len(bounds) != len(initial_values):
        raise ValueError("Initial vector length does not match parameter bounds")
    if samples < 2:
        raise ValueError("At least two samples are required")
    lower = np.array([bound.lower_bound for bound in bounds], dtype=float)
    upper = np.array([bound.upper_bound for bound in bounds], dtype=float)
    initial = np.array(initial_values, dtype=float)
    normalized_initial = (initial - lower) / (upper - lower)
    if np.any(normalized_initial < -1e-12) or np.any(normalized_initial > 1 + 1e-12):
        raise ValueError("Initial parameter vector lies outside sampling bounds")
    lhs_count = samples - 1 if include_initial else samples
    normalized = qmc.LatinHypercube(d=len(bounds), seed=seed).random(lhs_count)
    if include_initial:
        normalized = np.vstack([np.clip(normalized_initial, 0, 1), normalized])
    physical = qmc.scale(normalized, lower, upper)
    return [
        SampleRecord(
            sample_id=f"sample_{index:06d}",
            normalized_values=tuple(float(value) for value in normalized[index]),
            physical_values=tuple(float(value) for value in physical[index]),
            is_initial=bool(include_initial and index == 0),
        )
        for index in range(samples)
    ]


def _bounds_hash(bounds: list[ParameterBound]) -> str:
    payload = [
        [bound.coordinate.parameter_id, bound.lower_bound, bound.upper_bound]
        for bound in bounds
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def sample_table(records: list[SampleRecord], bounds: list[ParameterBound], seed: int = 0) -> pd.DataFrame:
    digest = _bounds_hash(bounds)
    rows = []
    for record in records:
        row = {
            "sample_id": record.sample_id,
            "is_initial": record.is_initial,
            "seed": seed,
            "method": "latin_hypercube",
            "bounds_sha256": digest,
        }
        for bound, normalized, physical in zip(
            bounds, record.normalized_values, record.physical_values, strict=True
        ):
            label = bound.coordinate.parameter_id
            row[f"normalized__{label}"] = normalized
            row[f"physical__{label}"] = physical
        rows.append(row)
    return pd.DataFrame(rows)


def write_sample_ffield(
    document: FFieldDocument,
    bounds: list[ParameterBound],
    record: SampleRecord,
    path: str | Path,
) -> None:
    updates = {
        bound.coordinate: value
        for bound, value in zip(bounds, record.physical_values, strict=True)
    }
    document.write(path, updates)

