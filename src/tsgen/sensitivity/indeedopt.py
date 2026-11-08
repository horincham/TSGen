from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
import shutil

from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.params_ml import parse_params_ml, validate_parameter_bounds
from tsgen.sensitivity.paths import SensitivityPaths
from tsgen.sensitivity.parameter_validity import exclusion_reason
from tsgen.sensitivity.trainset_map import parse_geo_inventory,parse_trainset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_indeedopt_bundle(
    paths: SensitivityPaths,
    cfg: SensitivityConfig,
) -> Path:
    selected = paths.selected_params_ml
    quality_path=paths.tables_dir/'analysis_quality.json'
    method_path=paths.tables_dir/'analysis_method.json'
    if not quality_path.exists() or not method_path.exists():
        raise ValueError('A verified current analysis is required before optimizer export')
    quality=json.loads(quality_path.read_text(encoding='utf-8'))
    method=json.loads(method_path.read_text(encoding='utf-8'))
    if not quality.get('production_ready') or not method.get('production_ready'):
        raise ValueError('Diagnostic/incomplete analysis cannot export production optimizer parameters')
    if not method.get('artifacts_sha256'):
        raise ValueError('Analysis artifact signatures are missing; rerun analysis')
    for relative,digest in method['artifacts_sha256'].items():
        source=paths.run_dir/relative
        if not source.resolve().is_relative_to(paths.run_dir.resolve()) or not source.is_file() or _sha256(source)!=digest:
            raise ValueError(f'Analysis artifact hash changed: {relative}; rerun analysis')
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
    inventory=parse_geo_inventory(paths.inputs_dir/'geo');items=parse_trainset(paths.inputs_dir/'trainset.in')
    definitions={p.coordinate:p for p in document.parameters}
    for bound in bounds:
        reason=exclusion_reason(definitions[bound.coordinate],document,inventory,items,cfg.parameter_space.exclude_inactive_parameters)
        if reason:raise ValueError(f'Unsupported selected parameter {bound.coordinate.parameter_id}: {reason}')
    root=cfg.outputs.indeedopt_root
    package=None
    if root is not None:
        main=root/'main-parallel.py'
        if not main.is_file():raise FileNotFoundError(f'INDEEDopt root has no main-parallel.py: {root}')
        package=next((p for p in [root/'ffield_generator',root/'src/ffield_generator',root.parent/'src/ffield_generator'] if p.is_dir()),None)
        if package is None:raise FileNotFoundError('The complete INDEEDopt ffield_generator package is required')
    bundle = paths.indeedopt_bundle
    if bundle.exists():
        previous=bundle.with_name('indeedopt_bundle.previous');n=1
        while previous.exists():n+=1;previous=bundle.with_name(f'indeedopt_bundle.previous_{n}')
        if not all(p.resolve().is_relative_to(paths.exports_dir.resolve()) for p in [bundle,previous]):
            raise ValueError('Optimizer bundle backup must remain inside exports')
        bundle.rename(previous)
    folder = bundle / "folder"
    folder.mkdir(parents=True)
    shutil.copy2(selected, bundle / "params-ml")
    shutil.copy2(selected, folder / "params-ml")
    for name in ("ffield", "geo", "trainset.in", "control"):
        shutil.copy2(paths.inputs_dir / name, folder / name)
    executable_name = None
    wrapper_ready=False
    if cfg.reaxff.executable_source is not None:
        executable_name = cfg.reaxff.executable_source.name
        source = paths.inputs_dir / executable_name
        shutil.copy2(source, folder / executable_name)
        try:
            (folder / executable_name).chmod(0o755)
        except OSError:
            pass
        command=cfg.reaxff.command.split() if isinstance(cfg.reaxff.command,str) else cfg.reaxff.command
        if command==['./'+executable_name]:
            if executable_name!='exe':
                # Valid only for the configured direct standalone executable.
                # Every optimization evaluation still has its own copied folder.
                wrapper=("#!/bin/sh\nset -eu\ncp geo fort.3\ncp ffield fort.4\n"
                    "printf '0\\n' > fort.20\n"
                    +f"printf '{int(cfg.run.seed)}\\n' > fort.35\n"
                    +"rm -f fort.13 fort.99\nexec "+shlex.quote('./'+executable_name)+" > run.log\n")
                (folder/'exe').write_text(wrapper,encoding='ascii')
                (folder/'exe').chmod(0o755)
            wrapper_ready=True
    (folder/'iopt').write_text('  0\n',encoding='ascii')
    copied_indeedopt = False
    gpf_preserved=None
    if root is not None:
        source_text=(root/'main-parallel.py').read_text(encoding='utf-8')
        copied_text,count=re.subn(r'(?m)^gpf\s*=\s*[-+]?\d+(?:\.\d+)?\s*$',
            'gpf = 0',source_text)
        (bundle/'main-parallel.py').write_text(copied_text,encoding='utf-8')
        shutil.copytree(package,bundle/'ffield_generator',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        gpf_preserved=bool(count)
        copied_indeedopt = True
    for name in ['params-ml.selected.full_precision.json','params-ml.rounding_audit.csv']:
        if (paths.exports_dir/name).is_file():shutil.copy2(paths.exports_dir/name,bundle/name)
    (bundle/'compatibility.json').write_text(json.dumps(dict(
        data_only=not copied_indeedopt,optimizer_source_copied=copied_indeedopt,
        executable_wrapper_ready=wrapper_ready,copied_gpf_set_to_zero=gpf_preserved,
        runtime_execution_tested=False,parameter_file_format='six whitespace-separated columns; adaptive decimal precision; lower then upper',
        delta_semantics='absolute local-search half-width in this INDEEDopt implementation'),indent=2),encoding='utf-8')
    readme = (
        "TSGen sensitivity-selected INDEEDopt input bundle\n"
        f"Source run: {paths.run_dir.resolve()}\n"
        f"Selected parameters: {len(bounds)}\n"
        f"params-ml SHA-256: {_sha256(selected)}\n"
        f"ReaxFF wrapper: {executable_name or 'not copied'}\n"
        f"INDEEDopt source copied: {copied_indeedopt}\n\n"
        +("This is a data-only handoff. Use a complete INDEEDopt installation, including ffield_generator,\n"
          "its Python dependencies and a compatible folder/exe wrapper; a main script alone is insufficient.\n"
          if not copied_indeedopt else
          "Optimizer main script and ffield_generator are copied. Install its dependencies before running.\n")
        +f"Compatible direct executable wrapper ready: {wrapper_ready}\n"
        +f"Copied main gpf reset to 0 to preserve declared ranges: {gpf_preserved}\n"
        +"Review compatibility.json, optimizer sample counts and property counts before starting optimization.\n"
        +"The bundle was not executed by TSGen. params-ml is not the legacy fixed-width params file.\n"
        +"Bounds normally use two decimals, with adaptive precision for small/narrow ranges and initial endpoints.\n"
        +"Bounds are rounded inward; original precision is retained in the audit files.\n\n"
        "TSGen does not create dl-data-l.csv; INDEEDopt performs its own optimization sampling.\n"
    )
    (bundle / "README.txt").write_text(readme, encoding="utf-8")
    return bundle
