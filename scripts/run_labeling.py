"""Phase 3: aggregate labels, generate goldens, dedup, populate eval_dataset."""
from __future__ import annotations

import sys

import typer
from rich.console import Console

from evalforge.labeling.aggregate import aggregate
from evalforge.labeling.curate import curate_phase, golden_phase

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_id: int = typer.Argument(..., help="pipeline_runs.id from the classifier phase"),
    skip_golden: bool = typer.Option(False, help="Skip Sonnet golden-answer generation"),
):
    console.rule(f"[bold]labeling for run #{run_id}")

    console.rule("[bold]1. aggregate label_runs → labels")
    console.print(aggregate(run_id))

    if not skip_golden:
        console.rule("[bold]2. golden answers + assertions (Sonnet)")
        console.print(golden_phase(run_id))
    else:
        console.print("[yellow]   --skip-golden set, no Sonnet calls[/yellow]")

    console.rule("[bold]3. dedup + curate → eval_dataset")
    console.print(curate_phase(run_id))

    console.rule(f"[bold green]labeling for run #{run_id} done")
    return 0


if __name__ == "__main__":
    sys.exit(app() or 0)
