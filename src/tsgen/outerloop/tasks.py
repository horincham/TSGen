"""Engine-independent request/receipt protocol; argv commands never run in a shell."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from tsgen.outerloop.config import AdapterConfig
from tsgen.outerloop.io import atomic_json, read_json, sha256


def file_spec(path: Path) -> dict:
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def verify_files(files: dict) -> None:
    for key, spec in files.items():
        p = Path(spec['path'])
        if not p.is_file() or sha256(p) != spec['sha256']:
            raise ValueError(f'Artifact changed or missing: {key}: {p}')


def prepare_task(task: Path, payload: dict) -> dict:
    task = Path(task)
    task.mkdir(parents=True, exist_ok=True)
    request = dict(payload)
    request['schema_version'] = 1
    request['files'] = {k: file_spec(p) for k, p in payload.get('files', {}).items()}
    target = task / 'request.json'
    if target.exists():
        if read_json(target) != request:
            raise ValueError(f'Task inputs changed: {target}')
    else:
        atomic_json(target, request)
    return request


def import_result(task: Path, outputs: dict[str, Path], metadata: dict) -> Path:
    task = Path(task).resolve()
    request = read_json(task / 'request.json')
    verify_files(request['files'])
    if (task / 'result.json').exists():
        raise ValueError('Task already has an immutable result; do not overwrite')
    directory = task / 'artifacts'
    directory.mkdir(exist_ok=True)
    specs = {}
    for key, raw in outputs.items():
        if not key.replace('_', '').isalnum():
            raise ValueError(f'Invalid artifact name: {key}')
        source = Path(raw).resolve()
        dest = directory / (key + source.suffix)
        if source != dest:
            shutil.copyfile(source, dest)
        specs[key] = file_spec(dest)
    result = dict(schema_version=1, request_sha256=sha256(task/'request.json'), outputs=specs, metadata=metadata)
    atomic_json(task/'result.json', result)
    return task/'result.json'


def collect_task(task: Path) -> dict | None:
    task = Path(task)
    request = read_json(task/'request.json')
    verify_files(request['files'])
    if not (task/'result.json').exists():
        return None
    result = read_json(task/'result.json')
    if result.get('schema_version') != 1 or result.get('request_sha256') != sha256(task/'request.json'):
        raise ValueError('Result belongs to a different request')
    verify_files(result['outputs'])
    if not isinstance(result.get('metadata'), dict):
        raise ValueError('Result metadata must be a mapping')
    return result


def execute_once(task: Path, cfg: AdapterConfig) -> None:
    if not cfg.command or (task/'result.json').exists():
        return
    marker = task/'execution.json'
    if marker.exists():
        state = read_json(marker)
        if state.get('returncode', 0) != 0:
            raise RuntimeError(f'External task failed with code {state["returncode"]}; see {task}')
        return  # Submission or interrupted execution: never silently duplicate it.
    request=read_json(task/'request.json')
    argv=[]
    for arg in cfg.command:
        arg=arg.replace('{request}',str((task/'request.json').resolve())).replace('{task_dir}',str(task.resolve()))
        for key,spec in request['files'].items():arg=arg.replace('{input:'+key+'}',spec['path'])
        argv.append(arg)
    atomic_json(marker, dict(argv=argv, status='STARTED'))
    try:
        with (task/'stdout.log').open('wb') as out, (task/'stderr.log').open('wb') as err:
            result = subprocess.run(argv, cwd=task, stdout=out, stderr=err, timeout=cfg.timeout_seconds, check=False)
        atomic_json(marker, dict(argv=argv, status='RETURNED', returncode=result.returncode))
    except Exception as exc:
        atomic_json(marker, dict(argv=argv, status='UNCERTAIN', returncode=-1, error=str(exc)))
        raise
    if result.returncode:
        raise RuntimeError(f'External task failed with code {result.returncode}; see {task}')
