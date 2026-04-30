"""Phase 2: classifier — embed, cluster, name, sample, judge."""
from __future__ import annotations

import sys

import typer
from rich.console import Console

from evalforge.config import SAMPLE_SIZE, VOTING_PASSES
from evalforge.pipeline import (
    cluster_phase,
    finish_run,
    label_phase,
    sample_phase,
    start_run,
)

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    sample_size: int = typer.Option(SAMPLE_SIZE, help="How many logs to sample for labeling"),
    strategy: str = typer.Option("signal_boosted", help="uniform | stratified | signal_boosted"),
    voting_passes: int = typer.Option(VOTING_PASSES, help="Number of LLM-judge passes per sample"),
):
    config = {"sample_size": sample_size, "strategy": strategy, "voting_passes": voting_passes}
    run_id = start_run(config, notes="Phase 2 classifier run")
    console.rule(f"[bold]pipeline run #{run_id}")
    console.print(f"   config = {config}")

    try:
        console.rule("[bold]1. cluster")
        console.print(cluster_phase(run_id))

        console.rule("[bold]2. sample")
        sample_ids = sample_phase(run_id, strategy, sample_size)

        console.rule("[bold]3. label (LLM-as-judge)")
        console.print(label_phase(run_id, sample_ids, voting_passes))
    except Exception as e:
        console.print(f"[red]pipeline failed: {e}[/red]")
        finish_run(run_id, status="failed")
        raise

    finish_run(run_id, status="completed")
    console.rule(f"[bold green]run #{run_id} completed")
    return 0


if __name__ == "__main__":
    sys.exit(app() or 0)
