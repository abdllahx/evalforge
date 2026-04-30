"""Aggregate label_runs (per-pass) into labels (one row per log).

With a single voting pass, confidence collapses to 1.0 if the call succeeded.
With N passes, we measure agreement: fraction of passes that agree on each
field; the minimum across fields is the confidence score.
"""
from __future__ import annotations

from collections import Counter

from .. import db


def aggregate(run_id: int, *, low_confidence_threshold: float = 0.66) -> dict:
    """Read label_runs for this pipeline_run, write a single labels row per log."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT log_id, pass_idx, quality_score, difficulty, expected_behavior
            FROM label_runs WHERE run_id = %s
            ORDER BY log_id, pass_idx
            """,
            (run_id,),
        )
        rows = cur.fetchall()

    by_log: dict[int, list[dict]] = {}
    for r in rows:
        by_log.setdefault(r["log_id"], []).append(r)

    written = 0
    needing_review = 0
    for log_id, passes in by_log.items():
        qs = [p["quality_score"] for p in passes if p["quality_score"] is not None]
        diffs = [p["difficulty"] for p in passes if p["difficulty"]]
        behs = [p["expected_behavior"] for p in passes if p["expected_behavior"]]
        if not qs:
            continue

        # Aggregations: mode for categoricals, rounded mean for quality
        diff_mode, diff_count = Counter(diffs).most_common(1)[0]
        beh_mode, beh_count = Counter(behs).most_common(1)[0]
        qs_mean = sum(qs) / len(qs)
        qs_final = int(round(qs_mean))

        # Confidence: agreement fraction across the categorical fields. With
        # a single pass this is always 1.0 — multi-pass voting brings real
        # signal here. Border quality scores (== 3) flag for review even at
        # 1.0 confidence because the judge itself was ambiguous.
        n = len(passes)
        confidence = min(diff_count / n, beh_count / n)
        needs_review = confidence < low_confidence_threshold or qs_final == 3

        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO labels
                  (log_id, run_id, quality_score, difficulty, expected_behavior,
                   confidence, needs_review)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (log_id) DO UPDATE SET
                  run_id = EXCLUDED.run_id,
                  quality_score = EXCLUDED.quality_score,
                  difficulty = EXCLUDED.difficulty,
                  expected_behavior = EXCLUDED.expected_behavior,
                  confidence = EXCLUDED.confidence,
                  needs_review = EXCLUDED.needs_review,
                  labeled_at = NOW()
                """,
                (log_id, run_id, qs_final, diff_mode, beh_mode, float(confidence), needs_review),
            )
        if needs_review:
            needing_review += 1
        written += 1
    return {"labels_aggregated": written, "needing_review": needing_review}


def attach_golden(log_id: int, golden: dict) -> None:
    with db.cursor() as cur:
        cur.execute(
            """
            UPDATE labels SET
              golden_answer = %s,
              must_contain = %s,
              must_not_contain = %s
            WHERE log_id = %s
            """,
            (
                golden.get("golden_answer"),
                golden.get("must_contain", []),
                golden.get("must_not_contain", []),
                log_id,
            ),
        )
