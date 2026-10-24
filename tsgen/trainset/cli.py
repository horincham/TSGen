from __future__ import annotations

import json
from pathlib import Path

import typer

from tsgen.trainset.builder import TrainsetBuilder

app = typer.Typer(add_completion=False, help="Build ReaxFF geo/trainset.in from optimized structures and DFT scan PES.")


@app.command()
def plan(config: Path = typer.Option(..., exists=True, readable=True)):
    """Create a new trainset run and write scan structures/Gaussian inputs without submitting."""
    b = TrainsetBuilder.from_config(config)
    b.plan()
    typer.echo(f"run_dir={b.paths.run_dir}")


@app.command()
def run(config: Path = typer.Option(..., exists=True, readable=True)):
    """Create a new trainset run, write jobs, optionally submit/wait, then write products."""
    b = TrainsetBuilder.from_config(config)
    b.run()
    typer.echo(f"run_dir={b.paths.run_dir}")


@app.command()
def resume(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Resume an existing trainset run: collect logs, optionally submit missing, rewrite products."""
    b = TrainsetBuilder.from_run_dir(run_dir)
    b.resume()
    typer.echo(f"run_dir={b.paths.run_dir}")


@app.command()
def status(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Show Gaussian job and output status for a trainset run."""
    b = TrainsetBuilder.from_run_dir(run_dir)
    s = b.status()
    typer.echo(json.dumps(s, indent=2, ensure_ascii=False))
