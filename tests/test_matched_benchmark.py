import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tsgen.benchmark.matched import run_benchmark
from tsgen.cli import app
from tsgen.explore.config import AcquisitionCfg, SurrogateCfg


def run(initial=("a",), budgets=(2, 4), acquisition_cfg=None):
    pool = pd.DataFrame(
        {"id": ["a", "b", "c", "d", "e", "test"], "species_id": ["s"] * 6}
    )
    theta = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 0.2])
    features = np.column_stack([np.cos(theta), np.sin(theta)])
    energies = pd.Series([0, 1, 3, 5, 10, 2], index=pool["id"], dtype=float)
    return run_benchmark(
        pool, features, energies, ["test"], list(initial), list(budgets),
        seed=41, acquisition_cfg=acquisition_cfg or AcquisitionCfg(k_select=2), surrogate_cfg=SurrogateCfg()
    )


def test_matched_budgets_exclude_fixed_test_and_share_initial_labels():
    rows = run()
    assert {(r["strategy"], r["budget"]) for r in rows} == {
        (strategy, budget) for strategy in ["tsgen", "random", "fps"] for budget in [2, 4]
    }
    for row in rows:
        assert row["train_ids"][0] == "a"
        assert len(set(row["train_ids"])) == row["budget"]
        assert "test" not in row["train_ids"]
        assert np.isfinite(row["mae_kcalmol"])
    fps = next(row for row in rows if row["strategy"] == "fps" and row["budget"] == 2)
    assert fps["train_ids"] == ["a", "e"]


def test_fixed_test_label_cannot_be_used_as_seed():
    with pytest.raises(ValueError, match="initial"):
        run(initial=("test",))


def test_matched_benchmark_requires_study_acquisition_policy():
    cfg = AcquisitionCfg.model_validate({"representative": {}})
    with pytest.raises(ValueError, match="study acquisition"):
        run(acquisition_cfg=cfg)


def test_benchmark_cli_uses_saved_screen_and_rejects_group_leakage(tmp_path):
    import json

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (tmp_path / "config.yaml").write_text(
        "descriptor: {backend: soap, normalization: l2}\n"
        "holdout: {size: 1, groups_file: groups.csv}\n"
        "acquisition: {k_select: 1}\n"
    )
    (pool_dir / "pool_index.csv").write_text("id,species_id\na,H2\nb,H2\nc,H2\ntest,H2\n")
    (pool_dir / "geometry_screening.csv").write_text(
        "id,passed_gating\na,True\nb,True\nc,False\ntest,True\n"
    )
    angles = np.arange(4) * 0.2
    np.save(pool_dir / "features.npy", np.column_stack([np.cos(angles), np.sin(angles)]))
    (tmp_path / "holdout.csv").write_text("id\ntest\n")
    split = tmp_path / "pool_split.csv"
    split.write_text("id,split_group,is_holdout\na,a,False\nb,b,False\ntest,t,True\n")
    oracle = tmp_path / "oracle.csv"
    oracle.write_text("id,e_kcalmol_rel\na,0\nb,1\ntest,2\n")
    initial = tmp_path / "initial.txt"
    initial.write_text("a\n")
    out = tmp_path / "report.json"
    args = [
        "benchmark", "run", "--run-dir", str(tmp_path), "--oracle", str(oracle),
        "--initial-ids", str(initial), "--budget", "2", "--output", str(out),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["test_ids"] == ["test"]
    assert report["screened_pool_size"] == 3
    assert report["oracle_acquisition_size"] == 2
    assert len(report["results"]) == 3
    split.write_text("id,split_group,is_holdout\na,a,False\nb,t,False\ntest,t,True\n")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "spans both acquisition and test" in result.output
