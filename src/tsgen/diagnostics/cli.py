"""Inspect fitting data and adjust scheduler execution policy."""

from pathlib import Path
import json

import typer

from tsgen.diagnostics.audit_geo import audit_geo
from tsgen.diagnostics.sensitivity import collect
from tsgen.diagnostics.slurm_limits import update


app = typer.Typer(
    add_completion=False, help="Audit fitting files and diagnose sensitivity runs."
)


@app.command("audit-geo")
def audit_geo_cmd(
    geo: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(..., file_okay=False),
    tolerance: float = typer.Option(0.0001, min=0.0),
):
    """Report repeated geometries and coordinate labels in a ReaxFF GEO file."""
    typer.echo(json.dumps(audit_geo(geo, out, tolerance), indent=2, ensure_ascii=False))


@app.command("collect-sensitivity")
def collect_sensitivity_cmd(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    per_reason: int = typer.Option(3, min=1),
    max_groups: int = typer.Option(20, min=1),
):
    """Collect status tables and a bounded set of failed-sample logs."""
    try:
        result = collect(run_dir, output, per_reason, max_groups)
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("slurm-limits")
def slurm_limits_cmd(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False),
    max_array_size: int = typer.Option(10, min=1),
    array_parallelism: int = typer.Option(10, min=1),
    apply: bool = typer.Option(
        False, help="Save the scheduler policy after creating a config backup."
    ),
):
    """Preview or apply Slurm array limits to a prepared sensitivity run."""
    try:
        result = update(run_dir, max_array_size, array_parallelism, apply)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
