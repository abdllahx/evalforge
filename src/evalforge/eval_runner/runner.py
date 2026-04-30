"""Run the full eval_dataset against a candidate. Writes eval_runs + eval_results."""
from __future__ import annotations

from psycopg.types.json import Jsonb
from rich.console import Console

from .. import db
from .candidate import CANDIDATES, Candidate, run_candidate
from .scorer import score

console = Console()


def _load_dataset() -> list[dict]:
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_prompt, golden_answer, must_contain, must_not_contain,
                   expected_behavior, category, difficulty
            FROM eval_dataset
            ORDER BY id
            """
        )
        return cur.fetchall()


def _start_eval_run(candidate: Candidate, total: int) -> int:
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_runs (candidate_model, candidate_label, total_cases, config)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (candidate.model, candidate.label, total, Jsonb({"system_prompt": candidate.system_prompt})),
        )
        return cur.fetchone()["id"]


def _finish_eval_run(eval_run_id: int, passed: int, failed: int) -> None:
    with db.cursor() as cur:
        cur.execute(
            """
            UPDATE eval_runs
               SET completed_at = NOW(), passed = %s, failed = %s
             WHERE id = %s
            """,
            (passed, failed, eval_run_id),
        )


def run_eval(candidate_label: str, *, limit: int | None = None) -> dict:
    candidate = CANDIDATES[candidate_label]
    cases = _load_dataset()
    if limit:
        cases = cases[:limit]
    if not cases:
        raise RuntimeError("eval_dataset is empty — run labeling first")

    eval_run_id = _start_eval_run(candidate, len(cases))
    console.rule(f"[bold]eval run #{eval_run_id} — {candidate.label} ({len(cases)} cases)")
    passed_n, failed_n = 0, 0

    for tc in cases:
        try:
            response = run_candidate(candidate, tc["user_prompt"], run_id=eval_run_id)
            r = score(
                user_prompt=tc["user_prompt"],
                golden_answer=tc["golden_answer"] or "",
                candidate_response=response,
                must_contain=list(tc["must_contain"] or []),
                must_not_contain=list(tc["must_not_contain"] or []),
                expected_behavior=tc["expected_behavior"] or "answer",
            )
        except Exception as e:
            console.print(f"[yellow]   case {tc['id']} errored: {e}[/yellow]")
            failed_n += 1
            with db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval_results
                      (eval_run_id, test_case_id, candidate_response, passed, score, judge_reasoning, failure_reasons)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (eval_run_id, tc["id"], None, False, 0.0, f"error: {e}", ["runner_error"]),
                )
            continue

        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_results
                  (eval_run_id, test_case_id, candidate_response, passed, score, judge_reasoning, failure_reasons)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    eval_run_id,
                    tc["id"],
                    response,
                    r["passed"],
                    r["score"],
                    r["judge_reasoning"],
                    r["failure_reasons"],
                ),
            )
        if r["passed"]:
            passed_n += 1
        else:
            failed_n += 1

    _finish_eval_run(eval_run_id, passed_n, failed_n)
    console.print(f"   [green]passed={passed_n}[/green] [red]failed={failed_n}[/red]")
    return {"eval_run_id": eval_run_id, "passed": passed_n, "failed": failed_n}
