"""Translate a site command plan into a bound result receipt.

Use synchronous engine commands or scheduler commands that wait for completion.
No engine installation, scientific output values or installation-specific syntax
is assumed here. Called by the parent adapter; the parent owns the run lock.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
import yaml
from tsgen.outerloop.io import read_json, atomic_json
from tsgen.outerloop.tasks import collect_task, import_result, verify_files


def run_worker(request_path: Path, site_path: Path) -> None:
    request_path=Path(request_path).resolve();site_path=Path(site_path).resolve()
    task=request_path.parent
    if collect_task(task) is not None:return
    request=read_json(request_path);verify_files(request['files'])
    plan=yaml.safe_load(site_path.read_text(encoding='utf8'))[request['stage']]
    work=task/'work';work.mkdir(exist_ok=True)
    for key,dest in plan.get('stage_inputs',{}).items():
        target=(work/dest).resolve()
        if work.resolve() not in target.parents:
            raise ValueError('Staged input must stay within task/work')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(request['files'][key]['path'],target)
    for i,command in enumerate(plan.get('commands',[])):
        marker=work/f'command_{i:03d}.json'
        if marker.exists():
            if read_json(marker).get('returncode')==0:continue
            raise RuntimeError(f'Command {i} failed/interrupted; inspect {marker} before a deliberate retry')
        argv=[]
        for arg in command:
            value=str(arg).replace('{request}',str(request_path)).replace('{work_dir}',str(work)).replace('{task_dir}',str(task))
            for key,spec in request['files'].items():value=value.replace('{input:'+key+'}',spec['path'])
            argv.append(value)
        atomic_json(marker,dict(argv=argv,status='STARTED'))
        with (work/f'command_{i:03d}.stdout').open('wb') as out,(work/f'command_{i:03d}.stderr').open('wb') as err:
            result=subprocess.run(argv,cwd=work,check=False,stdout=out,stderr=err,timeout=plan.get('timeout_seconds',3600))
        atomic_json(marker,dict(argv=argv,status='RETURNED',returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f'Engine command failed with code {result.returncode}')
    outputs={key:(work/path).resolve() for key,path in plan['outputs'].items()}
    metadata=plan.get('metadata',{})
    if plan.get('metadata_file'):
        metadata=read_json(work/plan['metadata_file'])
    import_result(task,outputs,metadata)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--site',type=Path,required=True)
    args=parser.parse_args()
    run_worker(args.request,args.site)
