"""Run additional voting passes on the existing pipeline_run's sample.

Phase 3.3 of the original plan: real confidence-based routing requires multi-pass
agreement. Pass 0 already exists; this fills passes 1..N-1 for each sampled log,
then re-aggregates `labels` so confidence reflects actual cross-pass agreement.

Each pass uses a discriminator prefix so cache keys differ; pass 0 stays
cache-valid for callers that only run a single pass.
"""
from __future__ import annotations

import sys

import typer
from psycopg.types.json import Jsonb
from rich.console import Console

from evalforge import db
from evalforge.classifier.judge import judge as judge_one
from evalforge.labeling.aggregate import aggregate

console = Console()
app = typer.Typer(add_completion=False)


def _existing_passes(run_id: int) -> dict[int, set[int]]:
    """log_id → set of pass_idx already in label_runs."""
    with db.cursor() as cur:
        cur.execute(
            "SELECT log_id, pass_idx FROM label_runs WHERE run_id = %s",
            (run_id,),
        )
        out: dict[int, set[int]] = {}
        for r in cur.fetchall():
            out.setdefault(r["log_id"], set()).add(r["pass_idx"])
    return out


def _sampled_logs(run_id: int) -> list[dict]:
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT s.log_id, l.user_prompt, l.response
              FROM samples s JOIN logs l ON l.id = s.log_id
             WHERE s.run_id = %s
             ORDER BY s.log_id
            """,
            (run_id,),
        )
        return cur.fetchall()


@app.command()
def main(
    run_id: int = typer.Option(0, help="0 = most recent pipeline_run"),
    voting_passes: int = typer.Option(3, help="Total passes (existing + new)"),
):
    if run_id == 0:
        with db.cursor() as cur:
            cur.execute("SELECT MAX(id) AS id FROM pipeline_runs")
            run_id = int(cur.fetchone()["id"])
    console.rule(f"[bold]voting passes = {voting_passes} on run #{run_id}")

    have = _existing_passes(run_id)
    samples = _sampled_logs(run_id)
    console.print(f"   {len(samples)} sampled logs, current passes per log: "
                  f"min={min((len(v) for v in have.values()), default=0)} "
                  f"max={max((len(v) for v in have.values()), default=0)}")

    written, failed, skipped = 0, 0, 0
    for s in samples:
        log_id = s["log_id"]
        existing = have.get(log_id, set())
        for pass_idx in range(voting_passes):
            if pass_idx in existing:
                skipped += 1
                continue
            try:
                result = judge_one(
                    s["user_prompt"], s["response"], pass_idx=pass_idx, run_id=run_id
                )
            except Exception as e:
                console.print(f"   [yellow]judge failed log {log_id} pass {pass_idx}: {e}[/yellow]")
                failed += 1
                continue
            with db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO label_runs
                      (run_id, log_id, pass_idx, quality_score, difficulty,
                       expected_behavior, raw_response)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, log_id, pass_idx,
                        result.get("quality_score"),
                        result.get("difficulty"),
                        result.get("expected_behavior"),
                        Jsonb(result),
                    ),
                )
            written += 1

    console.print(f"   written={written} skipped(existing)={skipped} failed={failed}")
    console.rule("[bold]re-aggregating labels with new confidence")
    console.print(aggregate(run_id))
    console.rule("[bold green]voting done")
    return 0


if __name__ == "__main__":
    sys.exit(app() or 0)
