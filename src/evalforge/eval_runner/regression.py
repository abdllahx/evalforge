"""Compare two eval_runs to surface regressions and improvements."""
from __future__ import annotations

from .. import db


def compare(run_a_id: int, run_b_id: int) -> dict:
    """Compare run_b against run_a. Run A is the baseline, run B is the candidate."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT a.test_case_id, a.passed AS a_pass, b.passed AS b_pass,
                   a.score AS a_score, b.score AS b_score,
                   d.user_prompt, d.category, d.difficulty
              FROM eval_results a
              JOIN eval_results b ON a.test_case_id = b.test_case_id
              JOIN eval_dataset d ON d.id = a.test_case_id
             WHERE a.eval_run_id = %s AND b.eval_run_id = %s
            """,
            (run_a_id, run_b_id),
        )
        rows = cur.fetchall()

    new_failures: list[dict] = []
    new_passes: list[dict] = []
    score_deltas: list[dict] = []

    for r in rows:
        if r["a_pass"] and not r["b_pass"]:
            new_failures.append(r)
        elif not r["a_pass"] and r["b_pass"]:
            new_passes.append(r)
        delta = (r["b_score"] or 0) - (r["a_score"] or 0)
        if abs(delta) >= 1:
            score_deltas.append({**r, "delta": delta})

    return {
        "compared_cases": len(rows),
        "new_failures": new_failures,
        "new_passes": new_passes,
        "score_deltas": score_deltas,
    }
