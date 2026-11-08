from pathlib import Path

from tsgen.explore.config import ExploreCfg, load_explore_config


def test_energy_target_defaults_to_fixed_optimized_relative_energy():
    cfg = ExploreCfg()
    assert cfg.surrogate.target == 'e_kcalmol_rel'
    assert cfg.descriptor.normalization == 'l2'
    assert cfg.labeling.gaussian.reference_mode == 'optimized'
    assert cfg.labeling.backend == 'gaussian'
    assert not cfg.anchors.enabled


def test_quickstart_uses_numbered_explore_runs(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    quickstart = load_explore_config(Path("examples/quickstart.yaml")).run
    assert (Path(quickstart.out_dir), quickstart.name) == (Path("output/quickstart"), "explore")
