import numpy as np
import pandas as pd

from tsgen.explore.representative import has_informative_uncertainty, select_representatives


def test_bootstrap_pairs_seed_species_and_reserves_supported_discovery_slot():
    table = pd.DataFrame(
        {
            "species_id": ["paired", "paired", "other", "singleton"],
            "support_frames": [np.nan, 5, 3, 1],
            "is_seed": [True, False, False, False],
        },
        index=[11, 12, 13, 14],
    )
    X = np.array([[0.0], [0.4], [0.6], [100.0]])
    selected = select_representatives(
        table, X, 2, X[:1], set(), min_support_frames=2, new_species_slots=1,
        paired_species={"paired"},
    )
    assert selected == [1, 2]


def test_unsupported_exploration_is_limited_and_can_leave_batch_short():
    table = pd.DataFrame({
        "species_id": ["observed", "observed", "rare-a", "rare-b"],
        "support_frames": [5, 1, 1, 1],
        "is_seed": [False] * 4,
    })
    X = np.array([[0.5], [0.8], [1.0], [1.0]])
    selected = select_representatives(
        table, X, 4, np.array([[0.0]]), {"observed"},
        min_support_frames=2, new_species_slots=1,
    )
    assert selected == [2, 0]


def test_gp_shortlist_preserves_score_start_and_diversifies_supported_batch():
    table = pd.DataFrame({
        "species_id": ["same", "same", "covered", "distant", "singleton"],
        "support_frames": [3, 3, 3, 3, 1],
        "is_seed": [False] * 5,
    })
    X = np.array([[0.0], [3.0], [0.1], [2.0], [100.0]])
    selected = select_representatives(
        table, X, 2, np.empty((0, 1)), set(), min_support_frames=2, new_species_slots=0,
        scores=np.array([10.0, 9.9, 9.0, 8.0, 1.e5]), preselect_factor=2,
    )
    assert selected == [0, 3]


def test_uncertainty_requires_a_finite_and_nontrivial_spread():
    assert not has_informative_uncertainty(np.full(676, 0.1811965582))
    assert not has_informative_uncertainty(np.linspace(0.18, 0.180001, 40))
    assert not has_informative_uncertainty(np.r_[0.0, np.ones(99)])
    assert not has_informative_uncertainty(np.array([0.2, np.nan]))
    assert has_informative_uncertainty(np.linspace(0.1, 0.2, 40))
