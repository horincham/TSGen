from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsgen.explore.acquisition import Acquisition
from tsgen.explore.config import AcquisitionCfg, ExploreCfg
from tsgen.explore.io import ExplorePaths
from tsgen.explore.runner import ExploreRunner


def test_adaptive_cold_start_pairs_seeds_with_trajectory_representatives(tmp_path):
    cfg = ExploreCfg.model_validate({
        "candidate_pool": {"balance": {"enabled": True, "method": "representative"}},
        "acquisition": {"representative": {"seed_slots": 2, "new_species_slots": 0}},
    })
    runner = ExploreRunner(cfg, ExplorePaths(tmp_path))
    runner.pool.df = pd.DataFrame({
        "id": ["s1", "s2", "a", "b", "x"],
        "is_seed": [True, True, False, False, False],
        "species_id": ["A", "B", "A", "B", "X"],
        "support_frames": [0, 0, 7, 6, 1],
    })
    runner.pool.X = np.array([[0.], [1.], [.2], [.8], [9.]])
    chosen = runner._select_next_batch(k=4)
    assert set(chosen) == {0, 1, 2, 3}


def test_adaptive_flat_gp_uses_supported_coverage_and_records_it(tmp_path, monkeypatch):
    cfg = ExploreCfg.model_validate({
        "candidate_pool": {"balance": {"enabled": True, "method": "representative"}},
        "acquisition": {"representative": {"new_species_slots": 0}},
    })
    runner = ExploreRunner(cfg, ExplorePaths(tmp_path))
    runner.pool.df = pd.DataFrame({
        "id": ["a", "b", "isolated"],
        "is_seed": [False, False, False],
        "species_id": ["A", "B", "X"],
        "support_frames": [7, 6, 1],
    })
    runner.pool.X = np.array([[.2], [.8], [9.]])
    monkeypatch.setattr(runner.labels, "load", lambda: pd.DataFrame({"id": ["train"], "species_id": ["A"]}))
    model = SimpleNamespace(predict=lambda x: (np.zeros(len(x)), np.ones(len(x))))
    monkeypatch.setattr(runner, "_fit_surrogate", lambda *args: (model, np.array([[0.]])))
    chosen = runner._select_next_batch(k=2)
    assert set(chosen) == {0, 1}
    audit = pd.read_csv(tmp_path / "iter_000" / "scored.csv")
    assert set(audit["selection_mode"]) == {"representative_coverage"}
    assert set(audit.loc[audit["selected"], "id"]) == {"a", "b"}


def test_stable_scoring_keeps_constant_terms_out_of_the_score():
    candidates = pd.DataFrame({"species_id": ["new", "old", "new"], "mu": 0., "sigma": .1})
    scored = Acquisition(AcquisitionCfg()).compute(
        candidates, np.array([[.1], [.2], [.4]]), np.array([[0.]]), {"old": 2}, stable_scaling=True,
    )
    assert not scored["u"].any()
    assert not scored["acq"].any()
    assert np.max(np.abs(scored["new_species"])) <= 1.
