from __future__ import annotations

import typer
from pathlib import Path

from tsgen.config import load_config
from tsgen.logging import setup_logging
from tsgen.workflow import run_workflow, resume_workflow
from tsgen.state import load_state
from tsgen.paths import RunPaths

# Explore (uncertainty-aware active learning, independent pipeline)
from tsgen.explore.cli import app as explore_app
from tsgen.trainset.cli import app as trainset_app
from tsgen.sensitivity.cli import app as sensitivity_app

app = typer.Typer(add_completion=False)

# Sub-apps
app.add_typer(explore_app, name="explore")
app.add_typer(trainset_app, name="trainset")
app.add_typer(sensitivity_app, name="sensitivity")

@app.command()
def run(config: Path = typer.Option(..., exists=True, readable=True)):
    """Start a new run (or resume if configured)."""
    cfg = load_config(config)
    rp = RunPaths(root=Path(cfg.run.output_root), run_id=cfg.run.run_id)
    rp.ensure_base()

    logger = setup_logging(rp.logs_dir / "run.log.jsonl", console=True)
    logger.info("tsgen.run.start", extra={"extra": {"run_id": cfg.run.run_id, "config": str(config)}})

    run_workflow(cfg, rp, logger)

@app.command()
def resume(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Resume an existing run directory, continuing from state.json."""
    rp = RunPaths.from_run_dir(run_dir)
    state = load_state(rp.state_file)
    logger = setup_logging(rp.logs_dir / "run.log.jsonl", console=True)
    logger.info("tsgen.resume.start", extra={"extra": {"run_dir": str(run_dir), "stage": state.stage, "iter": state.iteration}})
    resume_workflow(rp, logger)

@app.command()
def status(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Show minimal status from state.json."""
    rp = RunPaths.from_run_dir(run_dir)
    state = load_state(rp.state_file)
    typer.echo(f"run_dir={run_dir}")
    typer.echo(f"stage={state.stage}")
    typer.echo(f"iter={state.iteration}")
    typer.echo(f"dft_used={state.dft_used}")
    typer.echo(f"selected_total={len(state.selected_ids_total)}")
    typer.echo(f"labeled_total={len(state.labeled_ids_total)}")

if __name__ == "__main__":
    app()
