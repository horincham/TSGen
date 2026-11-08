from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from tsgen.explore.config import HoldoutCfg


def _groups(table: pd.DataFrame, X: np.ndarray, cfg: HoldoutCfg) -> list[str]:
    if cfg.groups_file:
        manifest = pd.read_csv(Path(cfg.groups_file), dtype=str, keep_default_na=False)
        if not {"id", "split_group"}.issubset(manifest.columns) or manifest["id"].duplicated().any():
            raise ValueError("Groups CSV needs unique id and split_group columns")
        ids = table["id"].map(manifest.set_index("id")["split_group"])
        if ids.isna().any() or ids.eq("").any():
            raise ValueError("Groups CSV must cover every geometrically eligible candidate")
        return ids.tolist()

    parents = list(range(len(table)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(i, j):
        a, b = find(i), find(j)
        parents[max(a, b)] = min(a, b)

    trajectory = table.loc[~table["is_seed"]]
    for _, rows in trajectory.groupby("trajectory"):
        frames = rows.sort_values("trajectory_frame")["trajectory_frame"]
        previous = None
        for index, frame in frames.items():
            if previous is not None and frame - previous[1] <= cfg.adjacent_frames:
                join(index, previous[0])
            previous = index, frame

    for indices in table.groupby("species_id").indices.values():
        if len(indices) < 2:
            continue
        features = X[indices]
        neighbors = NearestNeighbors(radius=cfg.similarity_threshold, metric="euclidean")
        neighbors.fit(features)
        for i, nearby in enumerate(neighbors.radius_neighbors(return_distance=False)):
            for j in nearby:
                join(indices[i], indices[j])

    return [str(table.at[find(i), "id"]) for i in range(len(table))]


def grouped_holdout(table: pd.DataFrame, X: np.ndarray, cfg: HoldoutCfg, seed: int) -> pd.DataFrame:
    """Assign a fixed test set by whole groups, stratified by species and SOAP cluster."""
    eligible = table["passed_gating"].to_numpy(dtype=bool)
    candidates = table.loc[eligible].copy().reset_index(drop=True)
    features = X[eligible]
    if cfg.size >= len(candidates):
        raise ValueError("Held-out count must leave an acquisition pool")
    candidates["split_group"] = _groups(candidates, features, cfg)
    n_clusters = min(cfg.soap_clusters, len(candidates))
    candidates["soap_cluster"] = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(features)
    candidates["stratum"] = list(zip(candidates["species_id"], candidates["soap_cluster"]))
    strata = candidates["stratum"].value_counts(normalize=True)
    fractions = strata.to_numpy()
    targets = np.arange(cfg.size + 1)[:, None] * fractions

    # Retain the best partial allocation of each exact size. Groups containing
    # seed geometries remain in acquisition for every allocation.
    states = {0: ([], np.zeros(len(strata)), 0.0)}
    groups = list(candidates.groupby("split_group", sort=True))
    np.random.default_rng(seed).shuffle(groups)
    for group, rows in groups:
        if rows["is_seed"].any() or len(rows) > cfg.size:
            continue
        counts = rows["stratum"].value_counts().reindex(strata.index, fill_value=0).to_numpy()
        for size, (chosen, seen, _) in list(states.items()):
            total = size + len(rows)
            if total > cfg.size:
                continue
            prospective = seen + counts
            error = float(((prospective - targets[total]) ** 2).sum())
            if total in states and error >= states[total][2]:
                continue
            states[total] = (chosen + [group], prospective, error)
    if cfg.size not in states:
        raise ValueError(f"Cannot reserve exactly {cfg.size} structures using whole groups; inspect group sizes")
    candidates["is_holdout"] = candidates["split_group"].isin(states[cfg.size][0])
    return candidates[["id", "split_group", "soap_cluster", "is_holdout"]]

def label_reserved_set(run_dir: Path, cfg) -> tuple[int, int, Path]:
    """Prepare/collect Gaussian labels outside the acquisition budget."""
    from tsgen.data.xyz import read_xyz_multiframe
    from tsgen.explore.label_store import LabelStore
    from tsgen.explore.oracle_gaussian import GaussianJobSpec, GaussianOracle
    from tsgen.explore.references import OptimizedReferences
    from tsgen.outerloop.io import atomic_json
    from tsgen.utils.hashing import structure_hash

    gaussian = cfg.labeling.gaussian
    if gaussian.reference_mode != "optimized":
        raise ValueError("Held-out DFT labels require optimized species references")
    holdout = pd.read_csv(run_dir / "holdout.csv", dtype={"id": str, "species_id": str})
    frames = read_xyz_multiframe(run_dir / "holdout.xyz")
    if len(holdout) != len(frames) or holdout["id"].tolist() != [f.get_id() for f in frames]:
        raise ValueError("Holdout XYZ does not match holdout.csv")
    if holdout["id"].tolist() != [structure_hash(f.symbols, f.coords) for f in frames]:
        raise ValueError("Holdout XYZ coordinates do not match the reserved IDs")

    evaluation = run_dir / "evaluation"
    evaluation_refs = evaluation / "energy_references.json"
    labels = LabelStore(evaluation / "labels" / "labels.jsonl", reference_path=evaluation_refs, require_optimized=True)
    training = LabelStore(run_dir / "labels" / "labels.jsonl", reference_path=run_dir / "energy_references.json")
    fixed = labels.energy_references()
    acquired = {sid: value for sid, value in training.energy_references().items() if sid in set(holdout["species_id"])}
    if any(sid in fixed and fixed[sid] != value for sid, value in acquired.items()):
        raise ValueError("Evaluation and acquisition use different optimized energies. Run test labeling after acquisition.")
    if acquired:
        atomic_json(evaluation_refs, fixed | acquired)
        inherited_metadata = {sid: record for sid, record in training.reference_metadata().items() if sid in acquired}
        if inherited_metadata:
            evaluation_metadata = labels.reference_metadata()
            for sid, record in inherited_metadata.items():
                if sid in evaluation_metadata and evaluation_metadata[sid] != record:
                    raise ValueError(f"Evaluation and acquisition use different optimized geometries for {sid}")
            atomic_json(labels.reference_metadata_path, evaluation_metadata | inherited_metadata)

    oracle = GaussianOracle(gaussian)
    refs = OptimizedReferences(gaussian, cfg.species.bond_tol, evaluation / "references" / "jobs", labels,
                              timeout_seconds=cfg.loop.max_wait_seconds,
                              fingerprint_version=cfg.species.fingerprint_version)
    for row, frame in zip(holdout.itertuples(), frames):
        if not all(c in "0123456789abcdef" for c in row.id) or len(row.id) != 16:
            raise ValueError(f"Invalid candidate ID: {row.id}")
        job = evaluation / "jobs" / f"job_{row.id}"
        spec = GaussianJobSpec(row.id, row.species_id, frame)
        refs.prepare(spec)
        if not (job / "meta.json").exists():
            oracle.write_job(job, spec)
            oracle.submit(job)
    refs.collect()
    fixed = labels.energy_references()
    existing = set(labels.load()["id"])
    ready = [record for record in oracle.collect_labels(evaluation / "jobs", iter_idx=-1)
             if record.id not in existing and record.species_id in fixed]
    labels.append(ready)
    return len(existing) + len(ready), len(holdout), labels.labels_csv
