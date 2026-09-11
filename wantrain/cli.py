from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from wantrain.config import ConfigError, load_reference_config
from wantrain.reference import run_reference

app = typer.Typer(help="WANTrain experiment tooling.")
reference_app = typer.Typer(help="Single-process reference runs.")
app.add_typer(reference_app, name="reference")
console = Console()


@reference_app.command("run")
def reference_run(config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False)) -> None:
    """Run a single-process reference experiment."""
    try:
        reference_config = load_reference_config(config)
        run_dir = run_reference(reference_config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    app()

