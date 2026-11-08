import numpy as np
import pandas as pd

from tsgen.data.xyz import Frame, write_xyz_multiframe
from tsgen.explore.config import (
    CandidatePoolCfg,
    DescriptorCfg,
    PoolBalanceCfg,
    SeedXYZSourceCfg,
    SpeciesCfg,
    TrajXYZSourceCfg,
)
from tsgen.explore.pool import CandidatePool
from tsgen.explore.surrogate.features import compute_features


def _diatomics(*lengths):
    positions = [[x, 0.0, 0.0] for i, length in enumerate(lengths) for x in (10.0 * i, 10.0 * i + length)]
    return Frame(["H"] * len(positions), np.asarray(positions), "")


def test_representative_balance_preserves_seeds_and_counts_distinct_cluster_frames(tmp_path):
    seed = tmp_path / "seed.xyz"
    traj_a = tmp_path / "a.xyz"
    traj_b = tmp_path / "b.xyz"
    write_xyz_multiframe(seed, [_diatomics(0.55)])
    write_xyz_multiframe(traj_a, [_diatomics(0.40, 0.41), _diatomics(0.43), _diatomics(0.69, 0.70)])
    write_xyz_multiframe(traj_b, [_diatomics(0.42), _diatomics(0.72)])
    cfg = CandidatePoolCfg(
        sources=[SeedXYZSourceCfg(path=str(seed)), TrajXYZSourceCfg(path=str(traj_a)), TrajXYZSourceCfg(path=str(traj_b))],
        balance=PoolBalanceCfg(enabled=True, by="species", max_per_group=3, method="representative"),
    )
    descriptor = DescriptorCfg(backend="coulomb_eigs", max_atoms=2, normalization="none")
    pool = CandidatePool(cfg, descriptor, SpeciesCfg())
    pool.initialize(tmp_path / "pool")

    # The low-distance cluster spans two frame-zero records in different files
    # and one more frame. Both copies within the same frame count only once.
    assert pool.df["is_seed"].tolist() == [True, False, False]
    assert pool.df["support_frames"].tolist() == [0, 3, 2]
    assert pool.df["pool_idx"].tolist() == [0, 1, 2]
    lengths = [np.linalg.norm(frame.coords[0] - frame.coords[1]) for frame in pool.frames]
    np.testing.assert_allclose(lengths, [0.55, 0.41, 0.70])
    np.testing.assert_allclose(pool.X, compute_features(pool.frames, descriptor))
    persisted = pd.read_csv(tmp_path / "pool" / "pool_index.csv")
    assert persisted["support_frames"].tolist() == [0, 3, 2]


def test_representative_balance_ignores_empty_clusters(tmp_path):
    trajectory = tmp_path / "trajectory.xyz"
    frames = [_diatomics(length) for length in [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]]
    write_xyz_multiframe(
        trajectory,
        [Frame(frame.symbols, frame.coords + [i, 0.0, 0.0], "") for i, frame in enumerate(frames)],
    )
    cfg = CandidatePoolCfg(
        sources=[TrajXYZSourceCfg(path=str(trajectory))],
        balance=PoolBalanceCfg(enabled=True, by="species", max_per_group=3, method="representative"),
    )
    pool = CandidatePool(cfg, DescriptorCfg(backend="coulomb_eigs", max_atoms=2), SpeciesCfg())
    pool.initialize(tmp_path / "pool")

    assert len(pool.frames) == len(pool.df) == len(pool.X) == 2
    assert pool.df["trajectory_frame"].tolist() == [0, 4]
    assert pool.df["support_frames"].tolist() == [4, 3]


def test_small_recurrent_species_is_not_split_into_single_frame_clusters(tmp_path):
    trajectory = tmp_path / "trajectory.xyz"
    write_xyz_multiframe(trajectory, [_diatomics(length) for length in (0.55, 0.56, 0.57)])
    cfg = CandidatePoolCfg(
        sources=[TrajXYZSourceCfg(path=str(trajectory))],
        balance=PoolBalanceCfg(enabled=True, method="representative", max_per_group=12),
    )
    pool = CandidatePool(cfg, DescriptorCfg(backend="coulomb_eigs", max_atoms=2), SpeciesCfg())
    pool.initialize(tmp_path / "pool")
    assert pool.df["support_frames"].tolist() == [3]
