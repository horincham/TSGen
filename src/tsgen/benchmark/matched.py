"""Replay acquisition strategies against a fixed DFT oracle and test set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

from tsgen.explore.acquisition import Acquisition
from tsgen.explore.config import AcquisitionCfg, SurrogateCfg
from tsgen.explore.surrogate.gp import GaussianProcessSurrogate


def _farthest(features: np.ndarray, candidates: np.ndarray, labeled: list[int], k: int) -> list[int]:
    trial = features[candidates]
    _, distances = pairwise_distances_argmin_min(trial, features[labeled])
    selected = []
    for _ in range(k):
        idx = int(np.argmax(distances))
        selected.append(int(candidates[idx]))
        distances = np.minimum(distances, np.linalg.norm(trial - trial[idx], axis=1))
        distances[idx] = -np.inf
    return selected


def run_benchmark(
    pool: pd.DataFrame,
    features: np.ndarray,
    energies: pd.Series,
    test_ids: list[str],
    initial_ids: list[str],
    budgets: list[int],
    *,
    seed: int,
    acquisition_cfg: AcquisitionCfg,
    surrogate_cfg: SurrogateCfg,
) -> list[dict]:
    """Evaluate each budget using the same initial labels, oracle and holdout."""
    if pool["id"].duplicated().any() or len(pool) != len(features) or not energies.index.is_unique:
        raise ValueError("Pool, descriptors and oracle must have unambiguous structure IDs")
    rows = {value: idx for idx, value in enumerate(pool["id"])}
    if not set(energies.index) <= rows.keys() or not np.isfinite(energies.to_numpy(dtype=float)).all():
        raise ValueError("Oracle energies must be finite and keyed by eligible pool IDs")
    if not test_ids or len(test_ids) != len(set(test_ids)) or not set(test_ids) <= set(energies.index):
        raise ValueError("Every unique fixed test ID needs a successful oracle label")
    available = set(energies.index) - set(test_ids)
    if not initial_ids or len(initial_ids) != len(set(initial_ids)) or not set(initial_ids) <= available:
        raise ValueError("Supply unique initial IDs with oracle labels outside the fixed test set")
    if not budgets or sorted(set(budgets)) != budgets or not len(initial_ids) <= budgets[0] <= budgets[-1] <= len(available):
        raise ValueError("Budgets must increase from at least the initial count to at most the acquisition pool size")
    if surrogate_cfg.kind != "gp" or acquisition_cfg.method != "sigma":
        raise ValueError("The matched TSGen benchmark requires GP uncertainty acquisition")
    if acquisition_cfg.representative is not None:
        raise ValueError("The matched benchmark requires the study acquisition policy without representative selection")
    if acquisition_cfg.diversity.method != "farthest_in_descriptor":
        raise ValueError("TSGen benchmark requires descriptor novelty and FPS reranking")
    if acquisition_cfg.k_select < 1 or acquisition_cfg.preselect_factor < 1:
        raise ValueError("Benchmark batch size and shortlist factor must be positive")
    if surrogate_cfg.gp.max_train < budgets[-1]:
        raise ValueError("GP max_train must cover the largest benchmark label budget")
    if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-7):
        raise ValueError("The benchmark requires full L2-normalized SOAP descriptors")

    initial = [rows[value] for value in initial_ids]
    test = [rows[value] for value in test_ids]
    domain = np.array(sorted(rows[value] for value in available), dtype=int)
    y = energies.reindex(pool["id"]).to_numpy(dtype=float)
    acq = Acquisition(acquisition_cfg)
    rng = np.random.default_rng(seed)
    results = []
    for strategy in ("tsgen", "random", "fps"):
        labeled = initial.copy()
        for budget in budgets:
            while len(labeled) < budget:
                candidates = domain[~np.isin(domain, labeled)]
                k = min(acquisition_cfg.k_select, budget - len(labeled))
                if strategy == "random":
                    selected = rng.choice(candidates, size=k, replace=False).tolist()
                else:
                    if strategy == "tsgen":
                        surrogate = GaussianProcessSurrogate(surrogate_cfg.gp)
                        surrogate.fit(features[labeled], y[labeled])
                        mu, sigma = surrogate.predict(features[candidates])
                        trial = pool.iloc[candidates].assign(mu=mu, sigma=sigma)
                        counts = pool.iloc[labeled]["species_id"].value_counts().to_dict()
                        scores = acq.compute(trial, features[candidates], features[labeled], counts)["score"]
                        order = np.argsort(-scores.to_numpy(), kind="stable")
                        candidates = candidates[order[: acquisition_cfg.preselect_factor * k]]
                    selected = _farthest(features, candidates, labeled, k)
                labeled.extend(selected)
            surrogate = GaussianProcessSurrogate(surrogate_cfg.gp)
            surrogate.fit(features[labeled], y[labeled])
            predicted, _ = surrogate.predict(features[test])
            results.append({
                "strategy": strategy,
                "budget": budget,
                "seed": seed,
                "mae_kcalmol": float(np.mean(np.abs(predicted - y[test]))),
                "train_ids": pool.iloc[labeled]["id"].tolist(),
            })
    return results
