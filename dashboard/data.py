"""Read-only DB queries for the dashboard.

In live mode, points at Postgres via DATABASE_URL.
In fixture mode (Streamlit Cloud), points at a committed SQLite file.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "snapshot.sqlite"
USE_FIXTURE = os.getenv("EVALFORGE_FIXTURE_MODE") == "1" or (
    not os.getenv("DATABASE_URL") and FIXTURE_PATH.exists()
)


@contextmanager
def _conn():
    if USE_FIXTURE:
        conn = sqlite3.connect(FIXTURE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    else:
        url = os.getenv("DATABASE_URL", "postgresql://evalforge:evalforge@localhost:5432/evalforge")
        with psycopg.connect(url) as conn:
            yield conn


@st.cache_data(ttl=15)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def latest_run_id() -> int | None:
    df = query("SELECT id FROM pipeline_runs ORDER BY id DESC LIMIT 1")
    return int(df.iloc[0]["id"]) if len(df) else None


def overview_stats() -> dict:
    df = query(
        """
        SELECT
          (SELECT COUNT(*) FROM logs) AS total_logs,
          (SELECT COUNT(*) FROM eval_dataset) AS total_test_cases,
          (SELECT COUNT(*) FROM labels WHERE needs_review) AS in_review,
          (SELECT COUNT(*) FROM eval_runs WHERE completed_at IS NOT NULL) AS eval_runs,
          (SELECT COUNT(*) FROM pipeline_runs) AS pipeline_runs
        """
    )
    return df.iloc[0].to_dict()


def clusters_for_run(run_id: int) -> pd.DataFrame:
    return query(
        "SELECT cluster_idx, name, description, size FROM clusters WHERE run_id = %(r)s ORDER BY size DESC",
        {"r": run_id} if not USE_FIXTURE else (run_id,),
    ) if not USE_FIXTURE else query(
        "SELECT cluster_idx, name, description, size FROM clusters WHERE run_id = ? ORDER BY size DESC",
        (run_id,),
    )


def cluster_scatter(run_id: int) -> pd.DataFrame:
    sql = """
        SELECT lca.embedding_2d_x AS x, lca.embedding_2d_y AS y,
               c.name AS cluster_name, l.feature, l.user_prompt,
               lca.is_outlier, lca.outlier_reasons
        FROM log_cluster_assignment lca
        JOIN clusters c ON c.id = lca.cluster_id
        JOIN logs l ON l.id = lca.log_id
        WHERE lca.run_id = {ph}
    """
    return query(
        sql.format(ph="%(r)s" if not USE_FIXTURE else "?"),
        ({"r": run_id} if not USE_FIXTURE else (run_id,)),
    )


def eval_dataset_table() -> pd.DataFrame:
    return query(
        """
        SELECT id, category, difficulty, quality_score, expected_behavior,
               user_prompt,
               golden_answer,
               array_length(must_contain, 1) AS n_must_contain,
               array_length(must_not_contain, 1) AS n_must_not_contain,
               added_at
        FROM eval_dataset
        ORDER BY added_at DESC
        """
        if not USE_FIXTURE
        else """
        SELECT id, category, difficulty, quality_score, expected_behavior,
               user_prompt,
               golden_answer,
               json_array_length(must_contain) AS n_must_contain,
               json_array_length(must_not_contain) AS n_must_not_contain,
               added_at
        FROM eval_dataset
        ORDER BY added_at DESC
        """
    )


def review_queue() -> pd.DataFrame:
    return query(
        """
        SELECT lab.log_id, lab.quality_score, lab.difficulty, lab.expected_behavior,
               lab.confidence, l.feature, left(l.user_prompt, 120) AS user_prompt,
               left(l.response, 120) AS response_preview, lab.review_status
        FROM labels lab
        JOIN logs l ON l.id = lab.log_id
        WHERE lab.needs_review
        ORDER BY lab.labeled_at DESC
        """
        if not USE_FIXTURE
        else """
        SELECT lab.log_id, lab.quality_score, lab.difficulty, lab.expected_behavior,
               lab.confidence, l.feature, substr(l.user_prompt, 1, 120) AS user_prompt,
               substr(l.response, 1, 120) AS response_preview, lab.review_status
        FROM labels lab
        JOIN logs l ON l.id = lab.log_id
        WHERE lab.needs_review = 1
        ORDER BY lab.labeled_at DESC
        """
    )


def eval_runs_table() -> pd.DataFrame:
    return query(
        """
        SELECT id, candidate_label, candidate_model, total_cases, passed, failed,
               completed_at,
               ROUND(100.0 * passed / NULLIF(total_cases, 0), 1) AS pass_rate
        FROM eval_runs
        WHERE completed_at IS NOT NULL
        ORDER BY id DESC
        """
    )


def eval_results_for_run(eval_run_id: int) -> pd.DataFrame:
    response_expr = (
        "left(er.candidate_response, 200)"
        if not USE_FIXTURE
        else "substr(er.candidate_response, 1, 200)"
    )
    ph = "%(r)s" if not USE_FIXTURE else "?"
    sql = f"""
        SELECT er.test_case_id, er.passed, er.score, er.failure_reasons,
               {response_expr} AS response_preview,
               d.category, d.difficulty, d.user_prompt
        FROM eval_results er
        JOIN eval_dataset d ON d.id = er.test_case_id
        WHERE er.eval_run_id = {ph}
        ORDER BY er.passed, er.test_case_id
    """
    return query(sql, {"r": eval_run_id} if not USE_FIXTURE else (eval_run_id,))


def eval_diff(run_a: int, run_b: int) -> pd.DataFrame:
    sql = """
        SELECT a.test_case_id,
               a.passed AS baseline_passed, b.passed AS candidate_passed,
               a.score AS baseline_score, b.score AS candidate_score,
               d.category, d.difficulty, d.user_prompt,
               b.failure_reasons AS candidate_failure_reasons
        FROM eval_results a
        JOIN eval_results b ON a.test_case_id = b.test_case_id
        JOIN eval_dataset d ON d.id = a.test_case_id
        WHERE a.eval_run_id = {pa} AND b.eval_run_id = {pb}
    """
    if not USE_FIXTURE:
        return query(sql.format(pa="%(a)s", pb="%(b)s"), {"a": run_a, "b": run_b})
    return query(sql.format(pa="?", pb="?"), (run_a, run_b))


def growth_over_time() -> pd.DataFrame:
    """Cumulative test cases added per day."""
    if not USE_FIXTURE:
        return query(
            """
            SELECT date_trunc('hour', added_at) AS bucket, COUNT(*) AS added,
                   SUM(COUNT(*)) OVER (ORDER BY date_trunc('hour', added_at)) AS cumulative
            FROM eval_dataset
            GROUP BY bucket ORDER BY bucket
            """
        )
    return query(
        """
        SELECT strftime('%Y-%m-%d %H:00', added_at) AS bucket, COUNT(*) AS added
        FROM eval_dataset GROUP BY bucket ORDER BY bucket
        """
    )


def coverage_heatmap() -> pd.DataFrame:
    """category × difficulty grid with counts."""
    return query(
        """
        SELECT COALESCE(category, '(none)') AS category,
               COALESCE(difficulty, '(none)') AS difficulty,
               COUNT(*) AS n
        FROM eval_dataset
        GROUP BY category, difficulty
        ORDER BY category, difficulty
        """
    )


def freshness_stats() -> dict:
    """% of dataset added in last 30 days + recency cohorts."""
    if not USE_FIXTURE:
        df = query(
            """
            SELECT
              COUNT(*) AS total,
              COUNT(*) FILTER (WHERE added_at > NOW() - INTERVAL '7 days') AS last_7d,
              COUNT(*) FILTER (WHERE added_at > NOW() - INTERVAL '30 days') AS last_30d
            FROM eval_dataset
            """
        )
    else:
        df = query(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN added_at > datetime('now','-7 days') THEN 1 ELSE 0 END) AS last_7d,
              SUM(CASE WHEN added_at > datetime('now','-30 days') THEN 1 ELSE 0 END) AS last_30d
            FROM eval_dataset
            """
        )
    row = df.iloc[0].to_dict()
    total = max(1, int(row["total"]))
    row["pct_last_30d"] = round(100.0 * (int(row["last_30d"]) / total), 1)
    return row


def similar_eval_cases(prompt: str, k: int = 3) -> pd.DataFrame:
    """Return k existing eval_dataset rows most similar to `prompt` (cosine sim)."""
    if not prompt:
        return pd.DataFrame()
    df = query("SELECT id, user_prompt, category, difficulty FROM eval_dataset")
    if not len(df):
        return df
    # Lazy import — sentence-transformers is heavy

    model = _get_st_model()
    cand = model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
    existing = model.encode(df["user_prompt"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    sims = (cand @ existing.T)[0]
    df = df.assign(similarity=sims).sort_values("similarity", ascending=False).head(k)
    df["similarity"] = df["similarity"].round(3)
    return df


@st.cache_resource
def _get_st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def coverage_gaps() -> pd.DataFrame:
    """Clusters present in logs but under-represented in eval_dataset."""
    if not USE_FIXTURE:
        return query(
            """
            WITH cluster_pop AS (
              SELECT c.name, c.size AS log_count
                FROM clusters c
               WHERE c.run_id = (SELECT MAX(id) FROM pipeline_runs)
                 AND c.cluster_idx >= 0
            ),
            cluster_in_dataset AS (
              SELECT category AS name, COUNT(*) AS dataset_count
                FROM eval_dataset
               WHERE category IS NOT NULL
               GROUP BY category
            )
            SELECT cp.name, cp.log_count, COALESCE(cd.dataset_count, 0) AS dataset_count,
                   ROUND(100.0 * COALESCE(cd.dataset_count, 0) / NULLIF(cp.log_count, 0), 1) AS coverage_pct
              FROM cluster_pop cp
              LEFT JOIN cluster_in_dataset cd ON cd.name = cp.name
              ORDER BY coverage_pct ASC NULLS FIRST, cp.log_count DESC
            """
        )
    return query(
        """
        SELECT c.name, c.size AS log_count,
               COALESCE((SELECT COUNT(*) FROM eval_dataset d WHERE d.category = c.name), 0) AS dataset_count
          FROM clusters c
         WHERE c.run_id = (SELECT MAX(id) FROM pipeline_runs) AND c.cluster_idx >= 0
         ORDER BY dataset_count ASC, log_count DESC
        """
    )


# ─────────────────────────── write actions for review UI ───────────────────────────

def approve_label(log_id: int) -> bool:
    """Mark a review-queue label approved AND insert into eval_dataset."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT lab.log_id, l.user_prompt, lab.golden_answer, lab.must_contain,
                       lab.must_not_contain, lab.quality_score, lab.difficulty,
                       lab.expected_behavior, c.name AS category
                  FROM labels lab
                  JOIN logs l ON l.id = lab.log_id
             LEFT JOIN log_cluster_assignment lca
                    ON lca.log_id = l.id AND lca.run_id = lab.run_id
             LEFT JOIN clusters c ON c.id = lca.cluster_id
                 WHERE lab.log_id = %s
                """ if not USE_FIXTURE else
                """
                SELECT lab.log_id, l.user_prompt, lab.golden_answer, lab.must_contain,
                       lab.must_not_contain, lab.quality_score, lab.difficulty,
                       lab.expected_behavior, c.name AS category
                  FROM labels lab
                  JOIN logs l ON l.id = lab.log_id
             LEFT JOIN log_cluster_assignment lca
                    ON lca.log_id = l.id AND lca.run_id = lab.run_id
             LEFT JOIN clusters c ON c.id = lca.cluster_id
                 WHERE lab.log_id = ?
                """,
                (log_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            if not USE_FIXTURE:
                from psycopg.types.json import Jsonb
                rd = dict(row)
                cur.execute(
                    """
                    INSERT INTO eval_dataset
                      (log_id, category, difficulty, quality_score, expected_behavior,
                       user_prompt, golden_answer, rubric, must_contain, must_not_contain)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (log_id) DO NOTHING
                    """,
                    (
                        rd["log_id"], rd["category"], rd["difficulty"], rd["quality_score"],
                        rd["expected_behavior"], rd["user_prompt"], rd["golden_answer"],
                        Jsonb({"must_contain": list(rd["must_contain"] or []),
                               "must_not_contain": list(rd["must_not_contain"] or [])}),
                        list(rd["must_contain"] or []),
                        list(rd["must_not_contain"] or []),
                    ),
                )
                cur.execute(
                    "UPDATE labels SET review_status='approved', needs_review=false WHERE log_id=%s",
                    (log_id,),
                )
            else:
                # fixture (read-only in practice; SQLite path kept for parity)
                return False
        if not USE_FIXTURE:
            conn.commit()
    query.clear()  # bust streamlit cache
    return True


def reject_label(log_id: int, notes: str | None = None) -> bool:
    if USE_FIXTURE:
        return False
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE labels SET review_status='rejected', needs_review=false, reviewer_notes=%s WHERE log_id=%s",
                (notes, log_id),
            )
        conn.commit()
    query.clear()
    return True


def edit_label_notes(log_id: int, notes: str) -> bool:
    if USE_FIXTURE:
        return False
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE labels SET reviewer_notes=%s WHERE log_id=%s",
                (notes, log_id),
            )
        conn.commit()
    query.clear()
    return True


def call_log_summary() -> pd.DataFrame:
    return query(
        """
        SELECT model, purpose, COUNT(*) AS calls,
               SUM(CASE WHEN cached THEN 1 ELSE 0 END) AS cache_hits,
               ROUND(AVG(duration_ms))::int AS avg_ms
        FROM claude_call_log
        GROUP BY model, purpose
        ORDER BY model, purpose
        """
        if not USE_FIXTURE
        else """
        SELECT model, purpose, COUNT(*) AS calls,
               SUM(CASE WHEN cached THEN 1 ELSE 0 END) AS cache_hits,
               CAST(AVG(duration_ms) AS INTEGER) AS avg_ms
        FROM claude_call_log
        GROUP BY model, purpose
        ORDER BY model, purpose
        """
    )
