"""Read-only GEO audit: invariant redundancy and coordinate labels in readable IDs.

Usage: tsgen diagnostics audit-geo folder/geo --out reports/geo_audit
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from tsgen.trainset.io import read_frames_from_reaxff_geo
from tsgen.trainset.gaussian_scan_parser import measure_scan_coordinate
from tsgen.utils.geometry import (
    composition_key,
    pair_distance_fingerprint,
    fingerprint_distance,
)


def audit_geo(path: Path, out: Path, tolerance: float = 0.0001) -> dict:
    frames = read_frames_from_reaxff_geo(path)
    raw_names = [
        line.split(maxsplit=1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("DESCRP ")
    ]
    if len(raw_names) == len(frames):
        # The generic input loader sanitizes dots; audit the original IDs.
        frames = [(name, frame) for name, (_, frame) in zip(raw_names, frames)]
    groups, duplicates, coordinates = {}, [], []
    equilibrium = 0
    for name, frame in frames:
        equilibrium += int("_opt" in name)
        key, fp = composition_key(frame), pair_distance_fingerprint(frame)
        group = groups.setdefault(key, [])
        for old_name, old_fp in group:
            distance = fingerprint_distance(fp, old_fp)
            if distance <= tolerance:
                duplicates.append(
                    dict(
                        id=name, duplicate_of=old_name, fingerprint_distance_A=distance
                    )
                )
                break
        else:
            group.append((name, fp))
        # Legacy readable IDs include atom labels and target values. Hashed IDs
        # require scan_energy_curves.csv; never invent targets for those.
        match = re.search(r"((?:[A-Z][a-z]?\d+_){2,4})([-+]?\d+(?:\.\d+)?)$", name)
        if not match:
            continue
        atoms = [int(x) for x in re.findall(r"[A-Z][a-z]?(\d+)_", match.group(1))]
        stype = {2: "bond", 3: "angle", 4: "torsion"}[len(atoms)]
        target = float(match.group(2))
        try:
            actual = measure_scan_coordinate(frame, stype, atoms)
        except (ValueError, IndexError) as exc:
            coordinates.append(
                dict(
                    id=name,
                    scan_type=stype,
                    target=target,
                    error=str(exc),
                    mismatch=True,
                )
            )
            continue
        delta = (
            abs(((actual - target + 180.0) % 360.0) - 180.0)
            if stype == "torsion"
            else abs(actual - target)
        )
        coordinates.append(
            dict(
                id=name,
                scan_type=stype,
                target=target,
                actual=actual,
                delta=delta,
                mismatch=delta > (0.03 if stype == "bond" else 1.0),
            )
        )
    summary = dict(
        input_file=str(path.resolve()),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        n_structures=len(frames),
        n_equilibrium_by_id=equilibrium,
        fingerprint_tolerance_A=tolerance,
        n_redundant=len(duplicates),
        n_unique_greedy=len(frames) - len(duplicates),
        n_readable_scan_labels=len(coordinates),
        n_mismatched_labels=sum(r["mismatch"] for r in coordinates),
        note="Distance fingerprints are heuristic. Readable IDs provide targets, not Gaussian-log provenance. No input files modified.",
    )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        duplicates, columns=["id", "duplicate_of", "fingerprint_distance_A"]
    ).to_csv(out / "duplicates.csv", index=False)
    pd.DataFrame(coordinates).to_csv(out / "coordinate_labels.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
