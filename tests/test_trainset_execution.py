from pathlib import Path
import sys

from tsgen.trainset.config import GaussianCfg
from tsgen.trainset.gaussian import GaussianScanRunner, job_status


def test_scan_slurm_script_uses_unix_line_endings(tmp_path):
    runner = GaussianScanRunner(GaussianCfg(mode="slurm"))
    runner._write_slurm(tmp_path)
    script = (tmp_path / "submit.slurm").read_bytes()
    assert script.startswith(b"#!/bin/bash\n")
    assert b"\r" not in script
    assert b"gaussian.exitcode" in script


def test_scan_local_failure_keeps_log_and_exit_status(tmp_path):
    program = tmp_path / "engine with spaces.py"
    program.write_text("import sys\nprint(sys.stdin.read())\nsys.exit(7)\n")
    (tmp_path / "input.gjf").write_text("# failed calculation\n")
    cfg = GaussianCfg(mode="local", g16_cmd=f'"{sys.executable}" "{program}"')
    runner = GaussianScanRunner(cfg)
    runner.submit(tmp_path)
    assert (tmp_path / "gaussian.exitcode").read_text().strip() == "7"
    assert "failed calculation" in (tmp_path / "input.log").read_text()
    assert job_status(tmp_path)[0] == "failed"
