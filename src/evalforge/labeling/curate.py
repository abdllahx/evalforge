"""Phase 3 orchestration: aggregate labels, generate goldens, dedup, populate eval_dataset."""
from __future__ import annotations

from psycopg.types.json import Jsonb
from rich.console import Console

from .. import db
from .aggregate import attach_golden
from .dedup import filter_duplicates
from .golden import generate_golden

console = Console()


def _load_judged_logs(run_id: int) -> list[dict]:
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT
              l.id AS log_id,
              l.user_prompt,
              l.response AS original_response,
              lab.quality_score,
              lab.difficulty,
              lab.expected_behavior,
              lab.confidence,
              lab.needs_review,
              c.name AS cluster_name,
              -- pull the most recent judge reasoning for context
              (SELECT raw_response->>'reasoning'
                 FROM label_runs lr
                WHERE lr.log_id = l.id AND lr.run_id = %s
                ORDER BY pass_idx DESC LIMIT 1) AS judge_reasoning
            FROM labels lab
            JOIN logs l ON l.id = lab.log_id
            LEFT JOIN log_cluster_assignment lca
                   ON lca.log_id = l.id AND lca.run_id = %s
            LEFT JOIN clusters c ON c.id = lca.cluster_id
            WHERE lab.run_id = %s
            """,
            (run_id, run_id, run_id),
        )
        return cur.fetchall()


def golden_phase(run_id: int) -> dict:
    rows = _load_judged_logs(run_id)
    succeeded, failed, skipped = 0, 0, 0
    with db.cursor() as cur:
        cur.execute(
            "SELECT log_id FROM labels WHERE run_id = %s AND golden_answer IS NOT NULL",
            (run_id,),
        )
        already_done = {r["log_id"] for r in cur.fetchall()}
    for r in rows:
        if r["log_id"] in already_done:
            skipped += 1
            continue
        try:
            golden = generate_golden(
                user_prompt=r["user_prompt"],
                original_response=r["original_response"],
                judge_quality=r["quality_score"],
                judge_difficulty=r["difficulty"],
                judge_behavior=r["expected_behavior"],
                judge_reasoning=r["judge_reasoning"] or "",
                run_id=run_id,
            )
            attach_golden(r["log_id"], golden)
            succeeded += 1
        except Exception as e:
            console.print(f"   [yellow]golden failed for log {r['log_id']}: {e}[/yellow]")
            failed += 1
    return {"goldens_written": succeeded, "skipped_existing": skipped, "failed": failed}


def curate_phase(run_id: int) -> dict:
    """Dedup against existing eval_dataset, route by confidence, insert kept rows."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT
              lab.log_id,
              l.user_prompt,
              lab.golden_answer,
              lab.must_contain,
              lab.must_not_contain,
              lab.quality_score,
              lab.difficulty,
              lab.expected_behavior,
              lab.confidence,
              lab.needs_review,
              c.name AS category
            FROM labels lab
            JOIN logs l ON l.id = lab.log_id
            LEFT JOIN log_cluster_assignment lca
                   ON lca.log_id = l.id AND lca.run_id = %s
            LEFT JOIN clusters c ON c.id = lca.cluster_id
            WHERE lab.run_id = %s
              AND lab.golden_answer IS NOT NULL
            """,
            (run_id, run_id),
        )
        candidates = cur.fetchall()

    kept, rejected = filter_duplicates(candidates)

    auto_added = 0
    review_queue = 0
    with db.cursor() as cur:
        for c in kept:
            if c["needs_review"]:
                review_queue += 1
                continue
            rubric = {
                "must_contain": list(c["must_contain"] or []),
                "must_not_contain": list(c["must_not_contain"] or []),
            }
            cur.execute(
                """
                INSERT INTO eval_dataset
                  (log_id, category, difficulty, quality_score, expected_behavior,
                   user_prompt, golden_answer, rubric, must_contain, must_not_contain)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (log_id) DO NOTHING
                """,
                (
                    c["log_id"],
                    c["category"],
                    c["difficulty"],
                    c["quality_score"],
                    c["expected_behavior"],
                    c["user_prompt"],
                    c["golden_answer"],
                    Jsonb(rubric),
                    list(c["must_contain"] or []),
                    list(c["must_not_contain"] or []),
                ),
            )
            if cur.rowcount:
                auto_added += 1

    return {
        "candidates": len(candidates),
        "duplicates_rejected": len(rejected),
        "auto_added_to_dataset": auto_added,
        "in_review_queue": review_queue,
    }
