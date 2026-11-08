import json
from pathlib import Path
import sys

import numpy as np

from tsgen.data.xyz import Frame
from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.paths import SensitivityPaths
from tsgen.sensitivity.runner import SensitivityRun
from tsgen.trainset.reaxff_io import write_geo_file
from test_sensitivity_core import minimal_ffield_text


ENGINE = '''from pathlib import Path
import json

for name in ("fort.3", "fort.4", "fort.20", "fort.35", "control", "trainset.in"):
    assert Path(name).is_file(), name
assert Path("fort.20").read_text().strip() == "0"
sample = json.loads(Path("sample.json").read_text())
items = json.loads(Path("training_items.json").read_text())
value = float(sample["physical_values"][0])
loss = (value - 4.0) ** 2 + 1.0
rows = []
for item in items:
    rows.append(f'{item["item_label"]:60s}{0.0:12.4f}{item["reference_value"]:12.4f}'
                f'{item["weight"]:12.4f}{loss:12.4f}{loss:12.4f}')
Path("fort.13").write_text(f"{loss:.4f}\\n")
Path("fort.99").write_text("\\n".join(rows) + "\\n")
'''


def test_local_evaluation_stages_fortran_inputs_and_reuses_results(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "ffield").write_text(minimal_ffield_text(), encoding="utf-8")
    frame = Frame(["C", "C"], np.array([[0., 0., 0.], [1.5, 0., 0.]]), "")
    write_geo_file(inputs / "geo", [{"id": "mol_a", "frame": frame}])
    (inputs / "trainset.in").write_text("ENERGY\n1.0 + mol_a/1 2.0\nENDENERGY\n", encoding="utf-8")
    (inputs / "control").write_text("tabulate_long_range 0\n", encoding="utf-8")
    (inputs / "params-ml").write_text("3 1 4 0.1 1.0 9.0\n", encoding="utf-8")
    engine = inputs / "engine.py"
    engine.write_text(ENGINE, encoding="utf-8")
    cfg = SensitivityConfig.model_validate({
        "run": {"out_dir": tmp_path / "runs", "name": "sensitivity", "seed": 17},
        "inputs": {"ffield": inputs / "ffield", "geo": inputs / "geo",
                   "trainset_in": inputs / "trainset.in", "control": inputs / "control"},
        "parameter_space": {"mode": "from_params_ml", "params_ml": inputs / "params-ml"},
        "sampling": {"samples": 8},
        "reaxff": {"command": [sys.executable, "engine.py"], "executable_source": engine,
                   "timeout_seconds": 10, "stage_mode": "copy"},
        "execution": {"mode": "local", "workers": 2},
        "analysis": {"targets": ["TOTAL", "ENERGY"], "repeats": 2, "subsample_size": 4},
    })
    paths = SensitivityPaths.next(cfg.run.out_dir, cfg.run.name)
    run = SensitivityRun(cfg, paths)
    run.prepare()
    assert json.loads((paths.baseline_dir / "result.json").read_text())["status"] == "success"
    run.execute_local()
    results = [paths.sample_dir(record.sample_id) / "result.json" for record in run.samples]
    before = {path: path.stat().st_mtime_ns for path in results}
    run.resume()
    assert {path: path.stat().st_mtime_ns for path in results} == before
    losses, items = run.collect()
    assert len(losses) == len(items) == 8
    assert set(losses["run_status"]) == {"success"}
