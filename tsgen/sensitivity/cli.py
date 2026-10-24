from __future__ import annotations

import json
from pathlib import Path

import typer


app = typer.Typer(add_completion=False, help="Screen ReaxFF parameters and export HSIC sensitivity data.")


@app.command()
def candidates(
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(Path("params-ml.candidates")),
):
    """Validate or generate candidate params-ml without running ReaxFF."""
    from tsgen.sensitivity.config import load_sensitivity_config
    from tsgen.sensitivity.params_ml import generate_candidate_file
    bounds = generate_candidate_file(load_sensitivity_config(config), output)
    typer.echo(f"params_ml={output.resolve()}")
    typer.echo(f"n_candidates={len(bounds)}")


@app.command()
def doctor(config: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    report = SensitivityRun.from_config(config).doctor(run_baseline=False)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@app.command()
def prepare(config: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    run = SensitivityRun.from_config(config)
    run.prepare()
    typer.echo(f"run_dir={run.paths.run_dir}")


@app.command()
def run(config: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    workflow = SensitivityRun.from_config(config)
    workflow.run()
    typer.echo(f"run_dir={workflow.paths.run_dir}")


@app.command(hidden=True)
def worker(
    run_dir: Path = typer.Option(..., exists=True, readable=True),
    sample_id: str = typer.Option(...),
):
    from tsgen.sensitivity.runner import SensitivityRun
    result = SensitivityRun.from_run_dir(run_dir).evaluate_sample(sample_id)
    typer.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


@app.command()
def resume(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    SensitivityRun.from_run_dir(run_dir).resume_and_finalize()


@app.command()
def status(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    typer.echo(json.dumps(SensitivityRun.from_run_dir(run_dir).status(), indent=2, ensure_ascii=False))


@app.command()
def collect(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    SensitivityRun.from_run_dir(run_dir).collect()


@app.command()
def analyze(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    SensitivityRun.from_run_dir(run_dir).analyze()


@app.command()
def plot(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    SensitivityRun.from_run_dir(run_dir).plot()


@app.command("export-indeedopt")
def export_indeedopt(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    from tsgen.sensitivity.runner import SensitivityRun
    SensitivityRun.from_run_dir(run_dir).export_indeedopt()
