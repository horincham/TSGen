import typer

from tsgen.benchmark.cli import app as benchmark_app
from tsgen.diagnostics.cli import app as diagnostics_app
from tsgen.explore.cli import app as explore_app
from tsgen.outerloop.cli import app as outerloop_app
from tsgen.sensitivity.cli import app as sensitivity_app
from tsgen.trainset.cli import app as trainset_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Training Set Generator for adaptive ReaxFF parameterization.",
)
app.add_typer(explore_app, name="explore")
app.add_typer(trainset_app, name="trainset")
app.add_typer(sensitivity_app, name="sensitivity")
app.add_typer(outerloop_app, name="outerloop")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(diagnostics_app, name="diagnostics")


if __name__ == "__main__":
    app()
