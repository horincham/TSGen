import json
import tarfile
from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from tsgen.cli import app
from tsgen.data.xyz import Frame
from tsgen.diagnostics.audit_geo import audit_geo
from tsgen.diagnostics.sensitivity import collect
from tsgen.diagnostics.slurm_limits import update
from tsgen.trainset.reaxff_io import write_geo_file


def test_geo_audit_reports_redundancy_without_changing_input(tmp_path):
    geo = tmp_path / "geo"
    frame = Frame(["H", "H"], np.array([[0., 0., 0.], [.74, 0., 0.]]), "")
    write_geo_file(geo, [{"id": "one", "frame": frame}, {"id": "two", "frame": frame}])
    before = geo.read_bytes()
    summary = audit_geo(geo, tmp_path / "audit")
    assert summary["n_structures"] == 2
    assert summary["n_redundant"] == 1
    assert geo.read_bytes() == before


def test_slurm_limits_preserve_scientific_settings_and_keep_backup(tmp_path):
    config = tmp_path / "config.yaml"
    original = {"sampling": {"samples": 10000}, "analysis": {"cumulative_threshold": .95},
                "execution": {"mode": "slurm", "slurm": {"partition": "compute"}}}
    config.write_text(yaml.safe_dump(original), encoding="utf-8")
    before = config.read_bytes()
    preview = update(tmp_path, 12, 3)
    assert not preview["apply"] and config.read_bytes() == before
    result = update(tmp_path, 12, 3, apply=True)
    changed = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert changed["sampling"] == original["sampling"]
    assert changed["analysis"] == original["analysis"]
    assert changed["execution"]["slurm"]["partition"] == "compute"
    assert changed["execution"]["slurm"]["max_array_size"] == 12
    assert changed["execution"]["slurm"]["array_parallelism"] == 3
    assert Path(result["backup"]).read_bytes() == before


def test_diagnosis_archive_contains_actual_failure_and_cli_is_registered(tmp_path):
    run = tmp_path / "run"
    sample = run / "samples" / "sample_000001"
    sample.mkdir(parents=True)
    (run / "config.yaml").write_text("execution: {mode: slurm}\n", encoding="utf-8")
    (sample / "status.json").write_text(json.dumps({"status": "failed", "failure_reason": "engine exit 1"}))
    (run / "logs").mkdir()
    (run / "logs" / "slurm-123_0.err").write_text("worker failed before startup\n", encoding="utf-8")
    (run / "slurm").mkdir()
    (run / "slurm" / "legacy.err").write_text("legacy scheduler message\n", encoding="utf-8")
    output = tmp_path / "diagnosis.tar.gz"
    result = collect(run, output)
    assert result["counts"] == {"failed": 1}
    with tarfile.open(output) as archive:
        summary = json.load(archive.extractfile("diagnosis/summary.json"))
        assert "logs/slurm-123_0.err" in archive.getnames()
        assert "slurm/legacy.err" in archive.getnames()
    assert summary["failure_groups"][0]["original_reason"] == "engine exit 1"
    commands = CliRunner().invoke(app, ["diagnostics", "--help"])
    assert commands.exit_code == 0
    assert "audit-geo" in commands.output
    assert "collect-sensitivity" in commands.output
    assert "slurm-limits" in commands.output
