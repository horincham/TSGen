from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from tsgen.explore.acquisition import _min_dist_to_set
from tsgen.explore.config import ExploreCfg
from tsgen.explore.label_store import LabelStore
from tsgen.explore.surrogate import build_surrogate
from tsgen.outerloop.io import atomic_json
from tsgen.outerloop.methods import _method_identity


def evaluate_acquisition(run_dir: Path, cfg: ExploreCfg) -> Path:
    """Fit the final acquisition GP and evaluate independently labeled test IDs."""
    table = pd.read_csv(run_dir / "pool" / "pool_index.csv", dtype={"id": str})
    features = np.load(run_dir / "pool" / "features.npy", allow_pickle=False)
    holdout = pd.read_csv(run_dir / "holdout.csv", dtype={"id": str})
    train_store = LabelStore(run_dir / "labels" / "labels.jsonl",
                             reference_path=run_dir / "energy_references.json", require_optimized=True)
    train = train_store.load() if train_store.labels_jsonl.exists() else pd.read_csv(
        train_store.labels_csv, dtype={"id": str}
    )
    evaluation = run_dir / "evaluation"
    test_store = LabelStore(evaluation / "labels" / "labels.jsonl",
                            reference_path=evaluation / "energy_references.json", require_optimized=True)
    test = test_store.load() if test_store.labels_jsonl.exists() else pd.read_csv(
        test_store.labels_csv, dtype={"id": str}
    )

    if len(table) != len(features) or table["id"].duplicated().any():
        raise ValueError("Pool rows and descriptors must be aligned by unique IDs")
    for labels in (train, test):
        if labels.empty or labels["id"].duplicated().any() or not labels["status"].eq("ok").all():
            raise ValueError("Evaluation requires nonempty unique successful DFT labels")
        if not np.isfinite(labels[["e_kcalmol_rel", "reference_hartree"]]).all().all():
            raise ValueError("Evaluation requires relative labels with fixed optimized references")
    if len(holdout) != cfg.holdout.size or set(test["id"]) != set(holdout["id"]):
        raise ValueError("Every fixed test ID must have a successful independent label")
    if set(train["id"]) & set(holdout["id"]):
        raise ValueError("Acquisition labels overlap the independent test set")
    all_labels = pd.concat([train, test], ignore_index=True)
    if (all_labels.groupby("species_id")["reference_hartree"].nunique() > 1).any():
        raise ValueError("A species has inconsistent optimized references across evaluation subsets")
    for species, labels in all_labels.groupby("species_id"):
        for column in ("charge", "multiplicity"):
            if column in labels and labels[column].nunique() > 1:
                raise ValueError(f"A species has different electronic states across evaluation subsets: {species}")
        if "route" in labels and len({_method_identity(route) for route in labels["route"]}) > 1:
            raise ValueError(f"DFT methods differ across evaluation subsets for {species}")

    by_id = table.reset_index().set_index("id")
    matched = by_id.loc[all_labels["id"]]
    if not matched["passed_gating"].all() or not matched["species_id"].to_numpy().tolist() == all_labels["species_id"].tolist():
        raise ValueError("Evaluation labels disagree with the screened pool index")
    X_train = features[by_id.loc[train["id"], "index"]]
    y_train = train["e_kcalmol_rel"].to_numpy(dtype=float)
    X_test = features[by_id.loc[test["id"], "index"]]
    model = build_surrogate(cfg.surrogate)
    model.fit(X_train, y_train)
    d_nn_train = np.full(len(train), np.nan)
    if len(train) > 1:
        nn = NearestNeighbors(n_neighbors=1).fit(X_train)
        d_nn_train = np.sqrt(nn.kneighbors()[0][:, 0] ** 2 + 1e-12)
    d_nn_test = _min_dist_to_set(X_test, X_train)

    predictions = []
    summary = {}
    for subset, labels, X, distances in (("train", train, X_train, d_nn_train), ("test", test, X_test, d_nn_test)):
        predicted, sigma = model.predict(X)
        rows = labels[["id", "species_id", "e_kcalmol_rel"]].copy()
        rows["subset"] = subset
        rows["predicted_kcalmol"] = predicted
        rows["sigma_kcalmol"] = sigma
        rows["abs_error_kcalmol"] = np.abs(predicted - rows["e_kcalmol_rel"])
        rows["d_nn"] = distances
        rows["carbon_atoms"] = rows["species_id"].map(_carbon_count)
        centered = rows["e_kcalmol_rel"] - rows["e_kcalmol_rel"].mean()
        denominator = float((centered ** 2).sum())
        summary[subset] = {
            "n": len(rows),
            "mae_kcalmol": float(rows["abs_error_kcalmol"].mean()),
            "r2": 1 - float((rows["abs_error_kcalmol"] ** 2).sum()) / denominator if denominator else None,
        }
        predictions.append(rows)

    results = pd.concat(predictions, ignore_index=True)
    results.to_csv(evaluation / "predictions.csv", index=False)
    results.groupby(["subset", "carbon_atoms"])["abs_error_kcalmol"].agg(
        n="size", mae_kcalmol="mean"
    ).reset_index().to_csv(evaluation / "carbon_mae.csv", index=False)
    result = evaluation / "summary.json"
    atomic_json(result, summary)
    return result


def _carbon_count(species_id: str) -> int:
    match = re.search(r"C(\d*)(?![a-z])", species_id.split("_", 1)[0])
    return int(match.group(1) or "1") if match else 0
