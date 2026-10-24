from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.params_ml import parse_params_ml, validate_parameter_bounds
from tsgen.sensitivity.paths import SensitivityPaths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_indeedopt_bundle(
    paths: SensitivityPaths,
    cfg: SensitivityConfig,
) -> Path:
    selected = paths.selected_params_ml
    if not selected.is_file():
        raise FileNotFoundError(
            f"Missing selected parameters: {selected}; run sensitivity analyze first"
        )
    document = FFieldDocument.from_path(paths.inputs_dir / "ffield")
    bounds = validate_parameter_bounds(
        parse_params_ml(selected),
        document,
        cfg.parameter_space.require_initial_inside_bounds,
    )
    bundle = paths.indeedopt_bundle
    if bundle.exists():
        shutil.rmtree(bundle)
    folder = bundle / "folder"
    folder.mkdir(parents=True)
    shutil.copy2(selected, bundle / "params-ml")
    shutil.copy2(selected, folder / "params-ml")
    for name in ("ffield", "geo", "trainset.in", "control"):
        shutil.copy2(paths.inputs_dir / name, folder / name)
    executable_name = None
    if cfg.reaxff.executable_source is not None:
        executable_name = cfg.reaxff.executable_source.name
        source = paths.inputs_dir / executable_name
        shutil.copy2(source, folder / executable_name)
        try:
            (folder / executable_name).chmod(0o755)
        except OSError:
            pass
    copied_indeedopt = False
    if cfg.outputs.indeedopt_root is not None:
        root = cfg.outputs.indeedopt_root
        main = root / "main-parallel.py"
        if not main.is_file():
            raise FileNotFoundError(f"INDEEDopt root has no main-parallel.py: {root}")
        shutil.copy2(main, bundle / "main-parallel.py")
        copied_indeedopt = True
    readme = (
        "TSGen sensitivity-selected INDEEDopt input bundle\n"
        f"Source run: {paths.run_dir.resolve()}\n"
        f"Selected parameters: {len(bounds)}\n"
        f"params-ml SHA-256: {_sha256(selected)}\n"
        f"ReaxFF wrapper: {executable_name or 'not copied'}\n"
        f"INDEEDopt source copied: {copied_indeedopt}\n\n"
        "Place a compatible main-parallel.py beside this README when it is not copied,\n"
        "then run:\n"
        "python -m scoop main-parallel.py\n\n"
        "TSGen does not create dl-data-l.csv; INDEEDopt performs its own optimization sampling.\n"
    )
    (bundle / "README.txt").write_text(readme, encoding="utf-8")
    return bundle
