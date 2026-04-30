"""Resume an interrupted pipeline run.

After hitting the Claude rate-limit window, run this once the limit resets.
It picks up the most recent pipeline_runs row and finishes whichever steps
are incomplete:

  * judge any sampled logs that are missing label_runs
  * aggregate label_runs → labels
  * golden-answer + curate (Phase 3)
  * eval baseline + regressed (Phase 4)

Disk cache covers everything already done, so only the unfinished work
makes new Claude calls.
"""
from __future__ import annotations

import sys

import typer
from rich.console import Console

from evalforge import db
from evalforge.config import VOTING_PASSES
from evalforge.eval_runner.runner import run_eval
from evalforge.labeling.aggregate import aggregate
from evalforge.labeling.curate import curate_phase, golden_phase
from evalforge.pipeline import finish_run, label_phase

console = Console()
app = typer.Typer(add_completion=False)


def _latest_run_id() -> int:
    with db.cursor() as cur:
        cur.execute("SELECT id, status FROM pipeline_runs ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    if not row:
        raise RuntimeError("no pipeline_runs found — start with run_classifier.py")
    return int(row["id"])


def _missing_judges(run_id: int) -> list[int]:
    """Sample log_ids that have no label_runs row yet."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT s.log_id
              FROM samples s
         LEFT JOIN label_runs lr
                ON lr.run_id = s.run_id AND lr.log_id = s.log_id
             WHERE s.run_id = %s AND lr.id IS NULL
             ORDER BY s.log_id
            """,
            (run_id,),
        )
        return [r["log_id"] for r in cur.fetchall()]


def _has_aggregated(run_id: int) -> int:
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM labels WHERE run_id = %s", (run_id,))
        return int(cur.fetchone()["n"])


def _golden_progress(run_id: int) -> tuple[int, int]:
    """(have_golden, total_labeled). golden_phase is a no-op when have == total."""
    with db.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FILTER (WHERE golden_answer IS NOT NULL) AS done, "
            "COUNT(*) AS total FROM labels WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
        return int(row["done"]), int(row["total"])


def _eval_runs_count() -> int:
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM eval_runs WHERE completed_at IS NOT NULL")
        return int(cur.fetchone()["n"])


@app.command()
def main(
    run_id: int = typer.Option(0, help="0 = most recent pipeline run"),
    skip_eval: bool = typer.Option(False, help="Skip Phase 4"),
    eval_limit: int = typer.Option(25),
):
    rid = run_id or _latest_run_id()
    console.rule(f"[bold]resuming pipeline run #{rid}")

    missing = _missing_judges(rid)
    if missing:
        console.rule(f"[bold]Phase 2.3 judge — {len(missing)} logs left")
        console.print(label_phase(rid, missing, VOTING_PASSES))
    else:
        console.print("[green]✓ all sampled logs already judged[/green]")

    if _has_aggregated(rid) == 0:
        console.rule("[bold]Phase 3.1 aggregate")
        console.print(aggregate(rid))
    else:
        console.print(f"[green]✓ labels already aggregated ({_has_aggregated(rid)} rows)[/green]")

    have, total = _golden_progress(rid)
    if have < total:
        console.rule(f"[bold]Phase 3.2 goldens — {total - have} missing ({have}/{total} done)")
        console.print(golden_phase(rid))
    else:
        console.print(f"[green]✓ all {total} goldens already generated[/green]")

    console.rule("[bold]Phase 3.3 curate")
    console.print(curate_phase(rid))

    finish_run(rid, status="completed")

    if skip_eval:
        return 0

    if _eval_runs_count() < 3:
        console.rule("[bold]Phase 4 eval")
        for label in ("baseline-haiku", "strong-sonnet", "regressed-broken-prompt"):
            console.print(run_eval(label, limit=eval_limit))
    else:
        console.print(f"[green]✓ eval already has {_eval_runs_count()} completed runs[/green]")

    console.rule(f"[bold green]resume of run #{rid} complete")
    return 0


if __name__ == "__main__":
    sys.exit(app() or 0)
