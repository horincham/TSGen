from __future__ import annotations

from pathlib import Path

import typer

from tsgen.explore.config import load_explore_config
from tsgen.explore.io import ExplorePaths, init_run_dir, load_state
from tsgen.explore.runner import ExploreRunner

app = typer.Typer(
    add_completion=False,
    help="Dataset exploration and oracle-in-the-loop active learning (Gaussian + GP).",
)

@app.command("run")
def run_cmd(config: Path = typer.Option(..., exists=True, readable=True)):
    """Start a new explore run."""
    cfg = load_explore_config(config)
    rp = init_run_dir(Path(cfg.run.out_dir), cfg.run.name)

    # Snapshot config into run directory for reproducibility
    (rp.run_dir / "config.yaml").write_text(
        config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    runner = ExploreRunner(cfg, rp)
    runner.run()

@app.command("resume")
def resume_cmd(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Resume an existing explore run directory."""
    rp = ExplorePaths(run_dir=run_dir)
    cfg = load_explore_config(run_dir / "config.yaml")
    runner = ExploreRunner(cfg, rp)
    runner.run()

@app.command("status")
def status_cmd(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Show minimal explore run status."""
    rp = ExplorePaths(run_dir=run_dir)
    st = load_state(rp.state_file)

    typer.echo(f"run_dir={run_dir}")
    typer.echo(f"iter={st.iter}")
    typer.echo(f"dft_used={getattr(st, 'dft_used', 0)}")
    typer.echo(f"selected_total={getattr(st, 'selected', 0)}")
    typer.echo(f"labeled_total={getattr(st, 'labeled', 0)}")
