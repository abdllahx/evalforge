"""End-to-end pipeline runner. Executes phases 2 → 4 in sequence.

Assumes logs have already been ingested
(`scripts/ingest_wildchat.py`). Phase 5 is the dashboard;
phase 6 (fixture export) is a separate step.
"""
from __future__ import annotations

import sys

import typer
from rich.console import Console

from evalforge.config import SAMPLE_SIZE, VOTING_PASSES
from evalforge.eval_runner.regression import compare
from evalforge.eval_runner.runner import run_eval
from evalforge.labeling.aggregate import aggregate
from evalforge.labeling.curate import curate_phase, golden_phase
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
    sample_size: int = typer.Option(SAMPLE_SIZE),
    strategy: str = typer.Option("signal_boosted"),
    voting_passes: int = typer.Option(VOTING_PASSES),
    eval_limit: int = typer.Option(25, help="Cap on eval cases per candidate"),
    skip_eval: bool = typer.Option(False, help="Skip Phase 4"),
):
    config = {
        "sample_size": sample_size,
        "strategy": strategy,
        "voting_passes": voting_passes,
    }

    run_id = start_run(config, notes="end-to-end pipeline")
    console.rule(f"[bold]pipeline run #{run_id} — full")

    try:
        console.rule("[bold]Phase 2.1 cluster")
        console.print(cluster_phase(run_id))

        console.rule("[bold]Phase 2.2 sample")
        sample_ids = sample_phase(run_id, strategy, sample_size)

        console.rule("[bold]Phase 2.3 label (judge)")
        console.print(label_phase(run_id, sample_ids, voting_passes))

        console.rule("[bold]Phase 3.1 aggregate")
        console.print(aggregate(run_id))

        console.rule("[bold]Phase 3.2 golden answers")
        console.print(golden_phase(run_id))

        console.rule("[bold]Phase 3.3 curate → eval_dataset")
        console.print(curate_phase(run_id))
    except Exception as e:
        console.print(f"[red]pipeline failed: {e}[/red]")
        finish_run(run_id, status="failed")
        raise

    finish_run(run_id, status="completed")
    console.rule(f"[bold green]pipeline run #{run_id} done")

    if skip_eval:
        return 0

    console.rule("[bold]Phase 4 eval")
    eval_run_ids: list[int] = []
    for label in ("baseline-haiku", "strong-sonnet", "regressed-broken-prompt"):
        result = run_eval(label, limit=eval_limit)
        eval_run_ids.append(result["eval_run_id"])

    console.rule("[bold]Phase 4.1 regression diff")
    if len(eval_run_ids) >= 2:
        diff = compare(eval_run_ids[0], eval_run_ids[-1])
        console.print(f"   compared      = {diff['compared_cases']}")
        console.print(f"   new_failures  = {len(diff['new_failures'])}")
        console.print(f"   new_passes    = {len(diff['new_passes'])}")

    return 0


if __name__ == "__main__":
    sys.exit(app() or 0)
