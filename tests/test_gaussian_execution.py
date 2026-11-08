import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pytest

from tsgen.data.xyz import Frame
from tsgen.explore.config import GaussianCfg
from tsgen.explore.oracle_gaussian import GaussianJobSpec, GaussianOracle


def prepare(tmp_path, script, mode="local"):
    executable = tmp_path / "engine.py"
    executable.write_text(script, encoding="utf-8")
    command = shlex.join([sys.executable, str(executable)])
    oracle = GaussianOracle(GaussianCfg(mode=mode, g16_cmd=command))
    job = tmp_path / "job"
    frame = Frame(["H", "H"], np.array([[0., 0., 0.], [.74, 0., 0.]]), "")
    oracle.write_job(job, GaussianJobSpec("candidate", "species", frame))
    return oracle, job


def test_local_gaussian_executes_stdin_and_runs_once(tmp_path):
    oracle, job = prepare(tmp_path, """from pathlib import Path
import sys
assert 'M062X/6-311G(d,p)' in sys.stdin.read()
counter = Path('runs.txt')
counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')
print(' SCF Done: E(RM062X) = -1.2 A.U.')
print(' Normal termination of Gaussian 16')
""")
    assert oracle.submit(job) is None
    assert oracle.job_status(job) == "ok"
    records = oracle.collect_labels(tmp_path, 0)
    assert records[0].e_hartree == -1.2
    oracle.submit(job)
    assert (job / "runs.txt").read_text() == "1"


def test_nonzero_local_exit_invalidates_normal_looking_log(tmp_path):
    oracle, job = prepare(tmp_path, """import sys
print(' SCF Done: E(RM062X) = -1.2 A.U.')
print(' Normal termination of Gaussian 16')
sys.exit(2)
""")
    with pytest.raises(RuntimeError, match="exit code 2"):
        oracle.submit(job)
    assert oracle.job_status(job) == "failed"
    assert oracle.collect_labels(tmp_path, 0) == []


def test_slurm_script_runs_in_job_directory_and_preserves_exit_status(tmp_path):
    oracle, job = prepare(tmp_path, "", mode="slurm")
    script = (job / "submit.slurm").read_bytes()
    assert b"\r\n" not in script
    assert b"cd " in script
    assert b"gaussian.exitcode" in script
    text = script.decode()
    assert "exit \"$status\"" in text


def test_submitted_slurm_job_is_not_submitted_again(tmp_path, monkeypatch):
    import subprocess

    oracle, job = prepare(tmp_path, "", mode="slurm")
    submissions = []

    def submit(argv, **kwargs):
        submissions.append(argv)
        return subprocess.CompletedProcess(argv, 0, "Submitted batch job 12345\n", "")

    monkeypatch.setattr("tsgen.explore.oracle_gaussian.subprocess.run", submit)
    assert oracle.submit(job) == "12345"
    assert oracle.submit(job) == "12345"
    assert len(submissions) == 1


def test_final_error_termination_wins_over_an_earlier_normal_exit(tmp_path):
    oracle, job = prepare(tmp_path, "", mode="manual")
    (job / "input.log").write_text(
        " SCF Done: E(RM062X) = -1.2 A.U.\n Normal termination\n Error termination\n"
    )
    assert oracle.job_status(job) == "failed"
    assert oracle.collect_labels(tmp_path, 0) == []


@pytest.mark.parametrize("retry_reference", [False, True])
def test_local_acquisition_completes_two_rounds_and_reuses_one_optimized_reference(tmp_path, retry_reference):
    from tsgen.explore.config import ExploreCfg
    from tsgen.explore.io import ExplorePaths
    from tsgen.explore.runner import ExploreRunner

    seed = tmp_path / "seed.xyz"
    seed.write_text("2\nH2 seed\nH 0 0 0\nH 0.70 0 0\n")
    trajectory = tmp_path / "trajectory.xyz"
    trajectory.write_text("2\nH2 candidate\nH 0 0 0\nH 0.72 0 0\n")
    executable = tmp_path / "engine command.py"
    executable.write_text(f"retry_reference = {retry_reference!r}\n" + """import sys
text = sys.stdin.read()
if retry_reference and ' Opt\\n' in text:
    print(' Error in internal coordinate system.')
    print(' Error termination via Lnk1e in l103.exe')
    sys.exit(1)
if ' Opt' in text:
    print(' Standard orientation:\\n --\\n Center\\n Number\\n --')
    print(' 1 1 0 0.0 0.0 0.0\\n 2 1 0 0.70 0.0 0.0\\n -----')
    print(' Optimization completed.')
    energy = -1.1
else:
    energy = -1.0 if '0.7000000000000000' in text else -1.05
print(f' SCF Done: E(RM062X) = {energy} A.U.')
print(' Normal termination of Gaussian 16')
""", encoding="utf-8")
    cfg = ExploreCfg.model_validate({
        "run": {"max_iters": 2, "dft_budget": 2},
        "candidate_pool": {"sources": [{"type": "seed_xyz", "path": str(seed)},
                                        {"type": "traj_xyz", "path": str(trajectory)}]},
        "descriptor": {"backend": "coulomb_eigs", "max_atoms": 2, "normalization": "l2"},
        "labeling": {"gaussian": {"mode": "local", "reference_mode": "optimized",
                                   "g16_cmd": shlex.join([sys.executable, str(executable)])}},
        "acquisition": {"k_select": 1},
        "anchors": {"enabled": False},
        "loop": {"mode": "auto"},
    })
    paths = ExplorePaths(tmp_path / "run")
    runner = ExploreRunner(cfg, paths)
    runner.run()
    labels = runner.labels.load().sort_values("e_hartree")
    assert len(labels) == 2
    assert runner.state.dft_used == 2
    assert runner.state.iter == 2
    assert labels["reference_hartree"].tolist() == pytest.approx([-1.1, -1.1])
    assert len(list(paths.run_dir.glob("references/jobs/job_*"))) == 1
    refs = runner.labels.reference_metadata()
    expected_route = "Opt=Cartesian" if retry_reference else "Opt"
    assert next(iter(refs.values()))["route"].endswith(f" {expected_route}")
    sid, = refs
    from tsgen.data.xyz import read_xyz_multiframe

    exported, = read_xyz_multiframe(paths.exports_dir / "optimized_references" / f"{sid}.xyz")
    assert np.linalg.norm(exported.coords[0] - exported.coords[1]) == pytest.approx(0.70)
    from tsgen.explore.surrogate.gp import GaussianProcessSurrogate

    final = GaussianProcessSurrogate.load(paths.run_dir / "surrogate/model_final.pkl")
    assert final.model.X_train_.shape[0] == 2
    before = (paths.run_dir / "energy_references.json").read_bytes()
    ExploreRunner(cfg, paths).run()
    assert (paths.run_dir / "energy_references.json").read_bytes() == before


def test_manual_acquisition_reports_wait_and_exports_prepared_batch(tmp_path, capsys):
    from tsgen.data.xyz import read_xyz_multiframe
    from tsgen.explore.config import ExploreCfg
    from tsgen.explore.io import ExplorePaths
    from tsgen.explore.runner import ExploreRunner

    seed = tmp_path / "seed.xyz"
    seed.write_text("2\nH2 seed\nH 0 0 0\nH 0.74 0 0\n", encoding="utf-8")
    cfg = ExploreCfg.model_validate({
        "run": {"max_iters": 1, "dft_budget": 1},
        "candidate_pool": {"sources": [{"type": "seed_xyz", "path": str(seed)}]},
        "descriptor": {"backend": "coulomb_eigs", "max_atoms": 2},
        "acquisition": {"k_select": 1},
    })
    paths = ExplorePaths(tmp_path / "run")
    runner = ExploreRunner(cfg, paths)
    runner.run()
    runner.run()
    log = capsys.readouterr().err
    assert "[pool]" in log and "[round 1/1]" in log
    assert "Awaiting Gaussian logs" in log
    assert (paths.run_dir / "run.log.jsonl").is_file()
    assert len(read_xyz_multiframe(paths.selected_xyz)) == 1
    assert not paths.reaxff_geo_dir.exists()
    assert not (paths.run_dir / "surrogate/model_final.pkl").exists()


def test_reference_resumes_after_a_rejected_slurm_submission(tmp_path, monkeypatch):
    import subprocess
    from tsgen.explore.label_store import LabelStore
    from tsgen.explore.references import OptimizedReferences

    store = LabelStore(tmp_path / "labels.jsonl", reference_path=tmp_path / "energy_references.json",
                       require_optimized=True)
    refs = OptimizedReferences(GaussianCfg(mode="slurm"), 1.2, tmp_path / "refs", store)
    frame = Frame(["H", "H"], np.array([[0., 0., 0.], [.74, 0., 0.]]), "")
    spec = GaussianJobSpec("candidate", "H2_graph", frame)
    responses = iter([(1, "", "QOSMaxSubmitJobPerUserLimit"), (0, "Submitted batch job 456", "")])

    def submit(argv, **kwargs):
        status, out, err = next(responses)
        return subprocess.CompletedProcess(argv, status, out, err)

    monkeypatch.setattr("tsgen.explore.oracle_gaussian.subprocess.run", submit)
    with pytest.raises(RuntimeError, match="Slurm rejected"):
        refs.prepare(spec)
    assert (refs.job_dir(spec.species_id) / "meta.json").exists()
    refs.collect()
    assert "456" in (refs.job_dir(spec.species_id) / "sbatch.stdout").read_text()


@pytest.mark.parametrize("same_species", [True, False])
def test_slurm_resume_waits_for_reference_retry_after_sp_budget(tmp_path, monkeypatch, same_species):
    from tsgen.explore.config import ExploreCfg
    from tsgen.explore.io import ExplorePaths
    from tsgen.explore.runner import ExploreRunner

    seed = tmp_path / "seed.xyz"
    seed.write_text("2\nH2 seed\nH 0 0 0\nH 0.74 0 0\n")
    cfg = ExploreCfg.model_validate({
        "run": {"max_iters": 1, "dft_budget": 1},
        "candidate_pool": {"sources": [{"type": "seed_xyz", "path": str(seed)}]},
        "descriptor": {"backend": "coulomb_eigs", "max_atoms": 2},
        "labeling": {"gaussian": {"mode": "manual", "reference_mode": "optimized"}},
        "anchors": {"enabled": False},
    })
    paths = ExplorePaths(tmp_path / "run")
    prepared = ExploreRunner(cfg, paths)
    prepared.run()
    original = next(paths.run_dir.glob("references/jobs/job_*"))
    failed_log = " Error in internal coordinate system.\n Error termination via Lnk1e in l103.exe\n"
    (original / "input.log").write_text(failed_log)
    sp = next(paths.jobs_dir(0).glob("job_*"))
    (sp / "input.log").write_text(" SCF Done: E(RM062X) = -1.0 A.U.\n Normal termination\n")
    cfg.labeling.gaussian.mode = "slurm"
    cfg.loop.mode = "auto"

    import subprocess
    submitted = []

    def sbatch(argv, *, cwd, **kwargs):
        assert argv == ["sbatch", "submit.slurm"]
        submitted.append(Path(cwd))
        return subprocess.CompletedProcess(argv, 0, "Submitted batch job 987\n", "")

    def finish_on_poll(_):
        retry, = submitted
        bond_length = 0.74 if same_species else 0.90
        (retry / "input.log").write_text(
            " Standard orientation:\n --\n Center\n Number\n --\n"
            f" 1 1 0 0.0 0.0 0.0\n 2 1 0 {bond_length} 0.0 0.0\n -----\n"
            " Optimization completed.\n SCF Done: E(RM062X) = -1.1 A.U.\n Normal termination\n"
        )

    monkeypatch.setattr("tsgen.explore.oracle_gaussian.subprocess.run", sbatch)
    monkeypatch.setattr("time.sleep", finish_on_poll)
    runner = ExploreRunner(cfg, paths)
    runner.run()
    assert submitted == [original / "cartesian"]
    assert (original / "input.log").read_text() == failed_log
    assert runner.state.dft_used == 1
    assert runner.state.failed == int(not same_species)
    assert (paths.run_dir / "surrogate/model_final.pkl").exists() == same_species
    labels = runner.labels.load()
    if same_species:
        assert labels["reference_hartree"].tolist() == pytest.approx([-1.1])
    else:
        assert labels.empty and not runner.labels.energy_references()
    ExploreRunner(cfg, paths).run()
    assert len(submitted) == 1


def test_resume_recovers_reserved_budget_from_prepared_jobs_without_a_manifest(tmp_path):
    from tsgen.explore.config import ExploreCfg
    from tsgen.explore.io import ExplorePaths
    from tsgen.explore.runner import ExploreRunner

    seed = tmp_path / "seed.xyz"
    seed.write_text("2\nseed\nH 0 0 0\nH 0.70 0 0\n")
    cfg = ExploreCfg.model_validate({
        "run": {"max_iters": 2, "dft_budget": 2},
        "candidate_pool": {"sources": [{"type": "seed_xyz", "path": str(seed)}]},
        "descriptor": {"backend": "coulomb_eigs", "max_atoms": 2, "normalization": "l2"},
        "labeling": {"gaussian": {"mode": "manual", "reference_mode": "optimized"}},
        "anchors": {"enabled": False},
    })
    paths = ExplorePaths(tmp_path / "run")
    prepared = ExploreRunner(cfg, paths)
    prepared._ensure_pool()
    row = prepared.pool.df.iloc[0]
    spec = GaussianJobSpec(row["id"], row["species_id"], prepared.pool.frames[0])
    prepared.reference_oracle.prepare(spec)
    prepared.oracle.write_job(paths.jobs_dir(0) / f"job_{row['id']}", spec)
    resumed = ExploreRunner(cfg, paths)
    resumed.run()
    assert resumed.state.dft_used == 1
    assert resumed.state.iter == 1
    assert resumed.pool.df.iloc[0]["status"] == "prepared"
    assert not paths.jobs_dir(1).exists()


def test_frozen_reference_rejects_changed_electronic_state_before_sp_submission(tmp_path):
    from types import SimpleNamespace
    from tsgen.explore.references import OptimizedReferences

    labels = SimpleNamespace(
        energy_references=lambda: {"H2_graph": -600.},
        reference_metadata=lambda: {"H2_graph": {
            "charge": 0, "multiplicity": 1, "route": "# M062X/6-311G(d,p) Opt",
        }},
    )
    cfg = GaussianCfg(charge=1, multiplicity=2)
    refs = OptimizedReferences(cfg, 1.2, tmp_path, labels)
    frame = Frame(["H", "H"], np.array([[0., 0., 0.], [.74, 0., 0.]]), "")
    with pytest.raises(ValueError, match="Electronic state differs"):
        refs.prepare(GaussianJobSpec("candidate", "H2_graph", frame))
