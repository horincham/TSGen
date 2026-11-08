from __future__ import annotations
import json
import shutil
from pathlib import Path
import typer
from tsgen.outerloop.config import load_config
from tsgen.outerloop.runner import OuterRunner
from tsgen.outerloop.tasks import import_result as complete_task, collect_task, verify_files
from tsgen.outerloop.io import read_json, run_lock, atomic_json, sha256

app=typer.Typer(help='Refine force fields and expand training data with external simulation engines.')


@app.command()
def run(config: Path=typer.Option(...,exists=True,readable=True),wait: bool | None=typer.Option(None,'--wait/--no-wait')):
    """Freeze initial inputs, then advance until waiting or a terminal condition."""
    runner=OuterRunner.create(load_config(config))
    result = runner.run_until_pause(wait)
    typer.echo(json.dumps({"run_dir": str(runner.root), **result}, ensure_ascii=False, indent=2))


@app.command()
def resume(run_dir: Path=typer.Option(...,exists=True,file_okay=False),wait: bool | None=typer.Option(None,'--wait/--no-wait')):
    """Collect pending results and continue without duplicating submitted jobs."""
    typer.echo(json.dumps(OuterRunner(run_dir).run_until_pause(wait),ensure_ascii=False,indent=2))


@app.command()
def status(run_dir: Path=typer.Option(...,exists=True,file_okay=False)):
    """Read status without submitting or modifying tasks."""
    typer.echo(json.dumps(OuterRunner(run_dir).status(),ensure_ascii=False,indent=2))


@app.command('import-result')
def import_result(task_dir: Path=typer.Option(...,exists=True,file_okay=False),
                  manifest: Path=typer.Option(...,exists=True,readable=True)):
    """Import {outputs: {name: path}, metadata: {...}}; paths relative to manifest."""
    manifest=manifest.resolve();data=read_json(manifest)
    outputs={key:(manifest.parent/str(value)).resolve() for key,value in data['outputs'].items()}
    request=read_json(task_dir/'request.json')
    root=Path(request['run_dir'])
    with run_lock(root):
        receipt=complete_task(task_dir,outputs,data.get('metadata',{}))
    typer.echo(str(receipt))


@app.command()
def retry(run_dir: Path=typer.Option(...,exists=True,file_okay=False)):
    """Explicitly retry a failed/uncertain command after checking external job state."""
    root=run_dir.resolve()
    with run_lock(root):
        state=read_json(root/'state.json')
        if state['status']!='FAILED' or not state.get('pending_task'):
            raise typer.BadParameter('Only a FAILED pending external task can be retried')
        task=Path(state['pending_task']).resolve()
        if root not in task.parents or Path(read_json(task/'request.json')['run_dir']).resolve()!=root:
            raise typer.BadParameter('Pending task must belong to this run')
        marker=task/'execution.json'
        native_marker=task/'native_execution.json'
        if native_marker.exists():
            if (task/'result.json').exists():
                raise typer.BadParameter('Reject the invalid result before retrying')
            prior=task/'native_attempts';prior.mkdir(exist_ok=True)
            attempt=prior/f'{len(list(prior.iterdir()))+1:03d}';attempt.mkdir()
            shutil.move(str(native_marker),str(attempt/native_marker.name))
            if (task/'native').exists():shutil.move(str(task/'native'),str(attempt/'native'))
            state['status']='READY';state.pop('error',None);atomic_json(root/'state.json',state)
            typer.echo('Native attempt archived. Resume only after verifying the earlier process has terminated.')
            return
        if (task/'result.json').exists() or not marker.exists():
            raise typer.BadParameter('Task has a result or has no command execution to retry')
        prior=task/'executions';prior.mkdir(exist_ok=True)
        n=len(list(prior.glob('*.json')))+1
        shutil.move(str(marker),str(prior/f'attempt_{n:03d}.json'))
        state['status']='READY';state.pop('error',None);atomic_json(root/'state.json',state)
    typer.echo('Retry enabled. Run resume; verify no earlier job is still running before doing so.')


@app.command('reject-result')
def reject_result(run_dir: Path=typer.Option(...,exists=True,file_okay=False),reason: str=typer.Option(...)):
    """Archive an invalid result with a reason, permitting a corrected import."""
    root=run_dir.resolve()
    with run_lock(root):
        state=read_json(root/'state.json')
        if state['status']!='FAILED' or not state.get('pending_task'):
            raise typer.BadParameter('Reject requires a FAILED pending task')
        task=Path(state['pending_task']).resolve()
        if root not in task.parents or not (task/'result.json').exists():
            raise typer.BadParameter('Invalid task or no result to reject')
        rejected=task/'rejected';rejected.mkdir(exist_ok=True)
        dest=rejected/f'{len(list(rejected.iterdir()))+1:03d}';dest.mkdir()
        # Preserve old evidence and update tracked paths, rather than erasing it.
        moved=[task/'result.json']
        if (task/'artifacts').exists():moved.append(task/'artifacts')
        affected=[p for p in state['tracked'] if Path(p)==task/'result.json' or task/'artifacts' in Path(p).parents]
        for p in affected:state['tracked'].pop(p)
        for p in moved:shutil.move(str(p),str(dest/p.name))
        for p in dest.rglob('*'):
            if p.is_file():state['tracked'][str(p)]={'path':str(p),'sha256':sha256(p)}
        atomic_json(dest/'rejection.json',dict(reason=reason,stage=state['stage'],cycle=state['cycle']))
        state['status']='READY';state.pop('error',None);atomic_json(root/'state.json',state)
    typer.echo(str(dest))


@app.command()
def export(run_dir: Path=typer.Option(...,exists=True,file_okay=False),
           output: Path=typer.Option(...)):
    """Export audited metrics, parameters and fixed-test predictions without rendering figures."""
    root=run_dir.resolve();output=output.resolve()
    if output==root or root in output.parents or output in root.parents or output.exists():
        raise typer.BadParameter('Output must be a new directory outside the run')
    with run_lock(root):
        state=read_json(root/'state.json');verify_files(state['tracked'])
        output.mkdir(parents=True)
        for p in root.glob('cycle_*/*.csv'):
            dest=output/p.parent.name/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
        for p in root.glob('cycle_*/*.json'):
            dest=output/p.parent.name/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
        for name in ['summary.csv','final.json','config.json']:
            if (root/name).exists():shutil.copyfile(root/name,output/name)
        if (root/'surrogate_reports').exists():
            shutil.copytree(root/'surrogate_reports',output/'surrogate_reports')
        atomic_json(output/'export_manifest.json',{str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()})
    typer.echo(str(output))
