"""Three sampling strategies over the logs table.

- uniform: random N (baseline diversity)
- stratified: equal counts per feature (ensures all segments represented)
- signal_boosted: oversample interactions with negative signals
  (thumbs_down, retry, high latency) — these are the most valuable eval candidates.
"""
from __future__ import annotations

from psycopg import Connection


def sample_uniform(conn: Connection, n: int, *, seed: int | None = None) -> list[int]:
    with conn.cursor() as cur:
        if seed is not None:
            cur.execute("SELECT setseed(%s)", (seed / 1_000_000,))
        cur.execute(
            "SELECT id FROM logs ORDER BY random() LIMIT %s",
            (n,),
        )
        return [row["id"] for row in cur.fetchall()]


def sample_stratified(
    conn: Connection,
    n: int,
    *,
    by: str = "feature",
    seed: int | None = None,
) -> list[int]:
    if by not in {"feature", "model"}:
        raise ValueError(f"unsupported strata column: {by}")
    with conn.cursor() as cur:
        if seed is not None:
            cur.execute("SELECT setseed(%s)", (seed / 1_000_000,))
        cur.execute(f"SELECT DISTINCT {by} FROM logs WHERE {by} IS NOT NULL")
        strata = [r[by] for r in cur.fetchall()]
        if not strata:
            return []
        per_stratum = max(1, n // len(strata))
        ids: list[int] = []
        for value in strata:
            cur.execute(
                f"SELECT id FROM logs WHERE {by} = %s ORDER BY random() LIMIT %s",
                (value, per_stratum),
            )
            ids.extend(r["id"] for r in cur.fetchall())
        return ids[:n]


def sample_signal_boosted(
    conn: Connection,
    n: int,
    *,
    boost_ratio: float = 0.7,
    seed: int | None = None,
) -> list[int]:
    """Oversample logs with negative signals.

    `boost_ratio` of the budget goes to signal-bearing logs; the rest is uniform.
    """
    n_boost = int(n * boost_ratio)
    n_random = n - n_boost
    with conn.cursor() as cur:
        if seed is not None:
            cur.execute("SELECT setseed(%s)", (seed / 1_000_000,))
        cur.execute(
            """
            SELECT id FROM logs
            WHERE user_feedback IN ('thumbs_down', 'retry', 'moderation_flag')
               OR latency_ms > 5000
               OR COALESCE((metadata->>'moderation_flagged')::boolean, false)
               OR COALESCE((metadata->>'toxic')::boolean, false)
               OR COALESCE((metadata->>'prompt_injection')::boolean, false)
            ORDER BY random() LIMIT %s
            """,
            (n_boost,),
        )
        ids = [r["id"] for r in cur.fetchall()]
        if len(ids) < n_boost:
            shortfall = n_boost - len(ids)
            n_random += shortfall
        cur.execute(
            "SELECT id FROM logs WHERE id != ALL(%s) ORDER BY random() LIMIT %s",
            (ids or [-1], n_random),
        )
        ids.extend(r["id"] for r in cur.fetchall())
    return ids[:n]


def sample_coverage_aware(
    conn: Connection,
    n: int,
    *,
    seed: int | None = None,
    fallback_ratio: float = 0.4,
) -> list[int]:
    """Bias toward logs in clusters under-represented in eval_dataset.

    Reads the latest pipeline_run's cluster assignments + current eval_dataset
    coverage. Spends `(1 - fallback_ratio) * n` on under-represented clusters,
    rest on signal_boosted to preserve edge-case discovery.

    First run (eval_dataset empty) → identical to signal_boosted.
    """
    n_cov = int(n * (1 - fallback_ratio))
    n_fb = n - n_cov
    with conn.cursor() as cur:
        if seed is not None:
            cur.execute("SELECT setseed(%s)", (seed / 1_000_000,))
        cur.execute(
            """
            WITH latest AS (SELECT MAX(id) AS id FROM pipeline_runs),
            cov AS (
              SELECT c.id AS cluster_id, c.name,
                     c.size AS log_count,
                     COALESCE(
                       (SELECT COUNT(*) FROM eval_dataset d WHERE d.category = c.name),
                       0
                     ) AS dataset_count
                FROM clusters c, latest
               WHERE c.run_id = latest.id AND c.cluster_idx >= 0
            ),
            ranked AS (
              SELECT cluster_id,
                     1.0 * dataset_count / NULLIF(log_count, 0) AS coverage
                FROM cov
            )
            SELECT lca.log_id
              FROM log_cluster_assignment lca
              JOIN ranked r ON r.cluster_id = lca.cluster_id
              JOIN latest ON latest.id = lca.run_id
              LEFT JOIN samples s
                     ON s.run_id = lca.run_id AND s.log_id = lca.log_id
             WHERE s.log_id IS NULL
             ORDER BY COALESCE(r.coverage, 0) ASC, random()
             LIMIT %s
            """,
            (n_cov,),
        )
        ids = [r["log_id"] for r in cur.fetchall()]

    if n_fb > 0:
        ids.extend(sample_signal_boosted(conn, n_fb, seed=seed))
    # de-dup while preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    return deduped[:n]


STRATEGIES = {
    "uniform": sample_uniform,
    "stratified": sample_stratified,
    "signal_boosted": sample_signal_boosted,
    "coverage_aware": sample_coverage_aware,
}
