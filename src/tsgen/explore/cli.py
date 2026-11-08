from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from tsgen.explore.config import load_explore_config, save_explore_config
from tsgen.explore.io import ExplorePaths, init_run_dir, load_state
from tsgen.explore.runner import ExploreRunner

app = typer.Typer(
    add_completion=False,
    help="Dataset exploration and oracle-in-the-loop active learning (Gaussian + GP).",
)


@app.command('reference-bonds')
def reference_bonds_cmd(
    xyz: Path = typer.Option(..., exists=True, readable=True, help='Atom-mapped reference parents, not candidates'),
    output: Path = typer.Option(..., help='Output reference connectivity JSON; review before screening'),
    key: Optional[str] = typer.Option(None, help='Reference id for a single-frame XYZ'),
    bond_scale: float = typer.Option(1.25, min=.01),
    max_bond_angstrom: float = typer.Option(2.25, min=.01),
):
    from tsgen.filters.reference_bonds import write_reference_bonds_from_xyz
    data = write_reference_bonds_from_xyz(xyz, output, key, bond_scale, max_bond_angstrom)
    typer.echo(f'reference_bonds={output.resolve()} n_references={len(data)}')

@app.command("prepare")
def prepare_cmd(config: Path = typer.Option(..., exists=True, readable=True)):
    """Build and screen the candidate pool without submitting Gaussian jobs."""
    cfg = load_explore_config(config)
    rp = init_run_dir(Path(cfg.run.out_dir), cfg.run.name)
    save_explore_config(cfg, rp.run_dir / "config.yaml")
    runner = ExploreRunner(cfg, rp)
    runner._ensure_pool(create_holdout=False)
    typer.echo(str(rp.pool_dir / "pool_index.csv"))


@app.command("run")
def run_cmd(config: Path = typer.Option(..., exists=True, readable=True)):
    """Start a new explore run."""
    cfg = load_explore_config(config)
    rp = init_run_dir(Path(cfg.run.out_dir), cfg.run.name)

    # Snapshot config into run directory for reproducibility
    save_explore_config(cfg, rp.run_dir / 'config.yaml')

    runner = ExploreRunner(cfg, rp)
    runner.run()

@app.command("resume")
def resume_cmd(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Resume an existing explore run directory."""
    rp = ExplorePaths(run_dir=run_dir)
    cfg = load_explore_config(run_dir / "config.yaml")
    runner = ExploreRunner(cfg, rp)
    runner.run()


@app.command("export-references")
def export_references_cmd(run_dir: Path = typer.Option(..., exists=True, readable=True, file_okay=False)):
    """Export saved optimized species geometries without resuming calculations."""
    from tsgen.explore.label_store import LabelStore
    from tsgen.explore.references import export_optimized_references

    paths = ExplorePaths(run_dir)
    labels = LabelStore(paths.labels_jsonl, reference_path=run_dir / "energy_references.json")
    count = export_optimized_references(labels, paths.optimized_references_dir)
    if count:
        typer.echo(f"Exported {count} optimized references to {paths.optimized_references_dir}")
    else:
        typer.echo("No optimized species references are available in this run.")


@app.command("label-holdout")
def label_holdout_cmd(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Prepare or collect fixed test labels in a separate evaluation directory."""
    from tsgen.explore.holdout import label_reserved_set

    cfg = load_explore_config(run_dir / "config.yaml")
    labeled, expected, labels_csv = label_reserved_set(run_dir, cfg)
    typer.echo(f"Collected {labeled}/{expected} test labels. CSV: {labels_csv}")


@app.command("evaluate")
def evaluate_cmd(run_dir: Path = typer.Option(..., exists=True, readable=True)):
    """Evaluate the final acquisition surrogate using independently labeled test IDs."""
    from tsgen.explore.evaluation import evaluate_acquisition

    cfg = load_explore_config(run_dir / "config.yaml")
    typer.echo(str(evaluate_acquisition(run_dir, cfg)))


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
    typer.echo(f"failed_labels={st.failed}")
