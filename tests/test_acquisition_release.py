from pathlib import Path

import pytest

from tsgen.explore.config import GaussianCfg
from tsgen.explore.label_store import HARTREE_TO_KCALMOL, LabelRecord, LabelStore
from tsgen.explore.oracle_gaussian import GaussianOracle


def test_quickstart_chemistry_gate_checks_valence_and_bond_stretch():
    import numpy as np
    from tsgen.data.xyz import Frame, read_xyz_multiframe
    from tsgen.explore.config import PoolChemistryFilterCfg, load_explore_config
    from tsgen.explore.pool import _pool_chemistry_drop_reason

    root = Path(__file__).resolve().parents[1]
    config = load_explore_config(root / "examples/quickstart.yaml").candidate_pool.chemistry_filter
    acetylene = Frame(["H", "C", "C", "H"], np.array([[0., 0., z] for z in (0., 1.06, 2.26, 3.32)]), "")
    bent_coords = acetylene.coords.copy()
    bent_coords[0] = [.92, 0., .53]
    bent_acetylene = Frame(acetylene.symbols, bent_coords, "")
    vinylidene = Frame(
        ["C", "C", "H", "H"],
        np.array([[0., 0., 0.], [1.29, 0., 0.], [-.54, .94, 0.], [-.54, -.94, 0.]]), "",
    )
    carbon_dimer = Frame(["C", "C"], np.array([[0., 0., 0.], [1.43, 0., 0.]]), "")
    assert not _pool_chemistry_drop_reason(acetylene, config)
    assert "linear" in _pool_chemistry_drop_reason(bent_acetylene, config)
    assert "valence" in _pool_chemistry_drop_reason(vinylidene, config)
    assert "valence" in _pool_chemistry_drop_reason(carbon_dimer, config)
    assert not _pool_chemistry_drop_reason(vinylidene, PoolChemistryFilterCfg(enabled=True))

    ethane, = read_xyz_multiframe(root / "examples/data/seeds/ethane.xyz")
    first, second = (i for i, symbol in enumerate(ethane.symbols) if symbol == "C")
    bond = ethane.coords[second] - ethane.coords[first]
    length = np.linalg.norm(bond)
    second_group = np.linalg.norm(ethane.coords - ethane.coords[second], axis=1) < np.linalg.norm(
        ethane.coords - ethane.coords[first], axis=1
    )
    stretched = ethane.coords.copy()
    stretched[second_group] += (1.78 - length) * bond / length
    assert "bond length" in _pool_chemistry_drop_reason(Frame(ethane.symbols, stretched, ""), config)
    assert not _pool_chemistry_drop_reason(ethane, config)


def test_reference_neighborhood_filters_trajectory_geometry_before_selection(tmp_path):
    from tsgen.explore.config import load_explore_config
    from tsgen.explore.pool import CandidatePool

    (tmp_path / "reference.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
    (tmp_path / "trajectory.xyz").write_text(
        "2\nnear reference\nH 0 0 0\nH 0 0 0.72\n"
        "2\nshort H2\nH 0 0 0\nH 0 0 0.62\n"
        "2\nunknown species\nC 0 0 0\nC 0 0 1.40\n"
    )
    config = tmp_path / "example.yaml"
    config.write_text(
        "candidate_pool:\n"
        "  sources:\n"
        f"    - {{type: traj_xyz, path: '{(tmp_path / 'trajectory.xyz').as_posix()}'}}\n"
        "  chemistry_filter:\n"
        "    enabled: true\n"
        "    reference_neighborhood:\n"
        "      xyz_file: reference.xyz\n"
        "      max_pair_distance_delta: 0.05\n"
    )
    cfg = load_explore_config(config)
    pool = CandidatePool(cfg.candidate_pool, cfg.descriptor, cfg.species, pool_dir=tmp_path)
    frames, table = pool._build_frames_and_df()
    assert len(frames) == 1
    assert table["trajectory_frame"].tolist() == [0]
    audit = (tmp_path / "pool_prefilter_dropped.csv").read_text()
    assert "reference distance exceeded" in audit
    assert "species absent from reference library" in audit


def test_optimized_reference_is_required_and_fixed(tmp_path):
    references = tmp_path / "energy_references.json"
    store = LabelStore(tmp_path / "labels.jsonl", reference_path=references, require_optimized=True)
    row = LabelRecord("candidate", "H2_example", -1.0, -HARTREE_TO_KCALMOL, 0., 0, "", "")
    with pytest.raises(ValueError, match="Missing optimized"):
        store.append([row])
    assert not store.labels_jsonl.exists()
    references.write_text('{"H2_example": -630}')
    store.append([row])
    assert store.load().iloc[0]["e_kcalmol_rel"] == pytest.approx(630 - HARTREE_TO_KCALMOL)
    assert references.read_text() == '{"H2_example": -630}'


def test_gaussian_collector_uses_final_energy_and_requires_normal_termination(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "meta.json").write_text('{"id":"candidate","species_id":"species"}')
    scf = " SCF Done:  E(RM062X) = -1.1 A.U.\n SCF Done:  E(RM062X) = -1.2 A.U.\n"
    (job / "input.log").write_text(scf)
    oracle = GaussianOracle(GaussianCfg())
    assert oracle.collect_labels(tmp_path, iter_idx=0) == []
    (job / "input.log").write_text(scf + " Normal termination of Gaussian 16\n")
    assert oracle.collect_labels(tmp_path, iter_idx=0)[0].e_hartree == -1.2


def test_holdout_reserves_whole_groups_without_seeds(tmp_path):
    import numpy as np
    import pandas as pd
    from tsgen.explore.config import HoldoutCfg
    from tsgen.explore.holdout import grouped_holdout

    table = pd.DataFrame({
        "id": list("abcdef"),
        "species_id": ["H2", "H2", "CH", "CH", "H2", "CH"],
        "passed_gating": [True] * 6,
        "is_seed": [True] + [False] * 5,
        "trajectory": ["seed"] + ["md"] * 5,
        "trajectory_frame": [0, 0, 1, 6, 7, 11],
    })
    features = np.asarray([[0.0], [0.1], [2.0], [3.0], [3.2], [1.9]])
    cfg = HoldoutCfg(size=2, soap_clusters=2, similarity_threshold=0.15)
    split = grouped_holdout(table, features, cfg, 123)
    assert split["is_holdout"].sum() == 2
    assert not split.loc[0, "is_holdout"]
    assert split.groupby("split_group")["is_holdout"].nunique().eq(1).all()
    assert split.loc[1, "split_group"] == split.loc[2, "split_group"]
    assert split.loc[2, "split_group"] == split.loc[5, "split_group"]


def test_coverage_stops_after_descriptor_novelty_is_exhausted():
    import numpy as np
    from tsgen.explore.config import CandidatePoolCfg, DescriptorCfg, SpeciesCfg
    from tsgen.explore.pool import CandidatePool

    pool = CandidatePool(CandidatePoolCfg(), DescriptorCfg(), SpeciesCfg())
    X = np.asarray([[0.], [0.01], [0.1], [0.25], [0.26]])
    assert pool._coverage_selection(X, [0], [1, 2, 3, 4], 4, 0.15) == [4]


def test_manual_acquisition_waits_for_opt_and_keeps_fixed_reference(tmp_path):
    from tsgen.explore.config import ExploreCfg
    from tsgen.explore.io import ExplorePaths
    from tsgen.explore.runner import ExploreRunner

    seed = tmp_path / "seed.xyz"
    seed.write_text("2\nH2 seed\nH 0 0 0\nH 0.70 0 0\n")
    trajectory = tmp_path / "trajectory.xyz"
    trajectory.write_text("2\nH2 candidate\nH 0 0 0\nH 0.72 0 0\n")
    cfg = ExploreCfg.model_validate({
        "run": {"max_iters": 2, "dft_budget": 2},
        "candidate_pool": {"sources": [{"type": "seed_xyz", "path": str(seed)},
                                        {"type": "traj_xyz", "path": str(trajectory)}]},
        "descriptor": {"backend": "coulomb_eigs", "max_atoms": 2, "normalization": "l2"},
        "labeling": {"gaussian": {"reference_mode": "optimized"}},
        "acquisition": {"k_select": 1},
        "anchors": {"enabled": False},
    })
    paths = ExplorePaths(tmp_path / "run")
    runner = ExploreRunner(cfg, paths)
    runner.run()
    first = next(paths.jobs_dir(0).iterdir())
    ref = next((paths.run_dir / "references" / "jobs").iterdir())
    assert "M062X/6-311G(d,p)" in (first / "input.gjf").read_text()
    assert "Opt" in (ref / "input.gjf").read_text()

    def gaussian_log(energy):
        return f" SCF Done:  E(RM062X) = {energy} A.U.\n Normal termination of Gaussian 16\n"

    (first / "input.log").write_text(gaussian_log(-1.0))
    runner = ExploreRunner(cfg, paths)
    runner.run()
    assert runner.labels.load().empty
    assert not paths.jobs_dir(1).exists()
    (ref / "input.log").write_text(
        " Standard orientation:\n --\n Center\n Number\n --\n"
        " 1 1 0 0.0 0.0 0.0\n 2 1 0 0.70 0.0 0.0\n -----\n"
        " Optimization completed.\n" + gaussian_log(-1.1)
    )
    runner = ExploreRunner(cfg, paths)
    runner.run()
    second = next(paths.jobs_dir(1).iterdir())
    (second / "input.log").write_text(gaussian_log(-1.05))
    runner = ExploreRunner(cfg, paths)
    runner.run()
    labels = runner.labels.load().set_index("e_hartree")
    assert len(labels) == 2
    assert labels.loc[-1., "e_kcalmol_rel"] == pytest.approx(0.1 * HARTREE_TO_KCALMOL)
    assert labels.loc[-1.05, "e_kcalmol_rel"] == pytest.approx(0.05 * HARTREE_TO_KCALMOL)
    assert runner.state.dft_used == 2
    assert runner.state.iter == 2


def test_holdout_labeling_reuses_reference_without_touching_training(tmp_path):
    import json
    import numpy as np
    from tsgen.data.xyz import read_xyz_multiframe
    from tsgen.explore.config import ExploreCfg
    from tsgen.explore.holdout import label_reserved_set
    from tsgen.utils.hashing import structure_hash

    iid = structure_hash(["H", "H"], np.array([[0., 0., 0.], [.7400504, 0., 0.]]))
    (tmp_path / "holdout.csv").write_text(f"id,species_id\n{iid},H2_example\n")
    xyz = tmp_path / "holdout.xyz"
    xyz.write_text(f"2\nid={iid}\nH 0 0 0\nH 0.740050 0 0\n")
    ref = tmp_path / "energy_references.json"
    ref.write_text(json.dumps({"H2_example": -630.0}))
    cfg = ExploreCfg.model_validate({"labeling": {"gaussian": {"reference_mode": "optimized"}}})
    with pytest.raises(ValueError, match="coordinates do not match"):
        label_reserved_set(tmp_path, cfg)
    xyz.write_text(f"2\nid={iid}\nH 0 0 0\nH 0.7400504 0 0\n")
    assert label_reserved_set(tmp_path, cfg)[:2] == (0, 1)
    job = tmp_path / "evaluation" / "jobs" / f"job_{iid}"
    assert (job / "input.gjf").exists()
    frame = read_xyz_multiframe(job / "structure.xyz")[0]
    assert structure_hash(frame.symbols, frame.coords) == iid
    assert not (tmp_path / "evaluation" / "references").exists()
    (job / "input.log").write_text(" SCF Done: E(RM062X) = -1.0 A.U.\n Normal termination of Gaussian 16\n")
    collected, expected, csv = label_reserved_set(tmp_path, cfg)
    assert (collected, expected) == (1, 1)
    assert iid in csv.read_text()
    assert not (tmp_path / "labels" / "labels.jsonl").exists()


def test_prospective_evaluation_uses_only_acquired_labels(tmp_path):
    import json
    import numpy as np
    import pandas as pd
    from tsgen.explore.config import ExploreCfg
    from tsgen.explore.evaluation import evaluate_acquisition

    for name in ("pool", "labels", "evaluation/labels"):
        (tmp_path / name).mkdir(parents=True)
    ids = list("abcd")
    table = pd.DataFrame({"id": ids, "species_id": ["C2H6_graph"] * 4, "passed_gating": [True] * 4})
    table.to_csv(tmp_path / "pool" / "pool_index.csv", index=False)
    np.save(tmp_path / "pool" / "features.npy", [[0.], [0.3], [0.2], [0.1]])
    table.loc[2:, ["id"]].to_csv(tmp_path / "holdout.csv", index=False)
    labels = table[["id", "species_id"]].assign(
        status="ok", e_kcalmol_rel=[0., 0.3, 0.2, 0.1], reference_hartree=-1.
    )
    labels.iloc[:2].to_csv(tmp_path / "labels" / "labels.csv", index=False)
    labels.iloc[2:].to_csv(tmp_path / "evaluation" / "labels" / "labels.csv", index=False)
    cfg = ExploreCfg.model_validate({"holdout": {"size": 2, "similarity_threshold": 0.01}})
    result = evaluate_acquisition(tmp_path, cfg)
    summary = json.loads(result.read_text())
    assert summary["train"]["n"] == summary["test"]["n"] == 2
    predictions = pd.read_csv(result.parent / "predictions.csv")
    assert predictions.loc[predictions["subset"] == "train", "d_nn"].min() == pytest.approx(0.3)
    labels.to_csv(tmp_path / "labels" / "labels.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_acquisition(tmp_path, cfg)
