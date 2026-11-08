from types import SimpleNamespace

from tsgen.sensitivity.paths import SensitivityPaths
from tsgen.sensitivity.slurm import write_manual_script, write_slurm_array


def test_scheduler_scripts_and_sample_lists_use_unix_newlines(tmp_path):
    paths = SensitivityPaths(tmp_path / "run")
    settings = SimpleNamespace(
        array_parallelism=2, time="01:00:00", cpus_per_task=1, mem="1G",
        partition=None, extra_lines=[],
    )
    cfg = SimpleNamespace(execution=SimpleNamespace(slurm=settings))
    pending_ids = ["sample_000001", "sample_000002"]
    scripts = [write_manual_script(paths, pending_ids), write_slurm_array(paths, cfg, pending_ids)]
    for path in [*scripts, paths.slurm_dir / "pending_samples.txt"]:
        data = path.read_bytes()
        assert b"\r" not in data
        assert data.endswith(b"\n")
    assert b"#SBATCH --array=0-1%2\n" in scripts[1].read_bytes()
