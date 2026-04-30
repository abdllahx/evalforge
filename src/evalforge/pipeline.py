"""End-to-end pipeline orchestration.

Each phase reads from / writes to Postgres. They can be run independently for
debugging, or chained via run_pipeline.py.
"""
from __future__ import annotations

from psycopg.types.json import Jsonb
from rich.console import Console

from . import db
from .classifier.clustering import cluster, project_2d, representative_indices
from .classifier.embeddings import embed
from .classifier.judge import judge as judge_one
from .classifier.naming import name_cluster
from .ingestion.sampling import STRATEGIES

console = Console()


def start_run(config: dict, notes: str | None = None) -> int:
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline_runs (config, notes) VALUES (%s, %s) RETURNING id",
            (Jsonb(config), notes),
        )
        return cur.fetchone()["id"]


def finish_run(run_id: int, status: str = "completed") -> None:
    with db.cursor() as cur:
        cur.execute(
            "UPDATE pipeline_runs SET completed_at = NOW(), status = %s WHERE id = %s",
            (status, run_id),
        )


def cluster_phase(run_id: int) -> dict:
    """Embed all logs, run HDBSCAN, name clusters, store assignments."""
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, feature, user_prompt, response, user_feedback, latency_ms, metadata FROM logs ORDER BY id"
        )
        logs = cur.fetchall()
    if not logs:
        raise RuntimeError("no logs in DB — run scripts/ingest_wildchat.py first")

    console.print(f"[cluster] embedding {len(logs)} prompts…")
    emb = embed([row["user_prompt"] for row in logs])

    console.print("[cluster] running HDBSCAN…")
    labels, _ = cluster(emb)

    console.print("[cluster] projecting to 2D (UMAP)…")
    proj = project_2d(emb)

    unique_labels = sorted({int(x) for x in labels})
    cluster_db_id_by_label: dict[int, int] = {}

    with db.cursor() as cur:
        for lbl in unique_labels:
            members = [i for i, x in enumerate(labels) if int(x) == lbl]
            if lbl == -1:
                name, desc = "Unclustered / noise", "Outlier prompts that didn't fit any major cluster."
            else:
                rep_idx = representative_indices(emb, members, k=5)
                rep_prompts = [logs[i]["user_prompt"] for i in rep_idx]
                try:
                    name, desc = name_cluster(rep_prompts, run_id=run_id, cluster_idx=lbl)
                except Exception as e:
                    console.print(f"   [yellow]cluster {lbl} naming failed: {e}[/yellow]")
                    name, desc = f"Cluster {lbl}", "(naming failed)"
            rep_log_ids = [logs[i]["id"] for i in members[:5]]
            cur.execute(
                """
                INSERT INTO clusters (run_id, cluster_idx, name, description, size, representative_log_ids)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (run_id, lbl, name, desc, len(members), rep_log_ids),
            )
            cluster_db_id_by_label[lbl] = cur.fetchone()["id"]
            console.print(f"   cluster {lbl:>2}  size={len(members):>3}  → {name}")

        for i, log in enumerate(logs):
            lbl = int(labels[i])
            reasons: list[str] = []
            if lbl == -1:
                reasons.append("hdbscan_noise")
            if log["user_feedback"] == "thumbs_down":
                reasons.append("thumbs_down")
            if log["user_feedback"] == "retry":
                reasons.append("user_retry")
            if log["user_feedback"] == "moderation_flag":
                reasons.append("moderation_flag")
            if (log["latency_ms"] or 0) > 5000:
                reasons.append("latency_outlier")
            md = log["metadata"] or {}
            if md.get("prompt_injection") or md.get("moderation_flagged") or md.get("toxic"):
                reasons.append("flagged_content")
            user_prompt_len = len(log["user_prompt"] or "")
            response_len = len(log["response"] or "")
            if user_prompt_len > 0 and (response_len < 50 or response_len > 5000):
                reasons.append("response_length_outlier")
            cur.execute(
                """
                INSERT INTO log_cluster_assignment
                  (run_id, log_id, cluster_id, is_outlier, outlier_reasons, embedding_2d_x, embedding_2d_y)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, log_id) DO UPDATE SET
                  cluster_id = EXCLUDED.cluster_id,
                  is_outlier = EXCLUDED.is_outlier,
                  outlier_reasons = EXCLUDED.outlier_reasons,
                  embedding_2d_x = EXCLUDED.embedding_2d_x,
                  embedding_2d_y = EXCLUDED.embedding_2d_y
                """,
                (
                    run_id,
                    log["id"],
                    cluster_db_id_by_label[lbl],
                    bool(reasons),
                    reasons,
                    float(proj[i][0]),
                    float(proj[i][1]),
                ),
            )
    return {"clusters": len(unique_labels), "logs_clustered": len(logs)}


def sample_phase(run_id: int, strategy: str, n: int) -> list[int]:
    fn = STRATEGIES[strategy]
    with db.connect() as conn:
        sample_ids = fn(conn, n)
        with conn.cursor() as cur:
            for log_id in sample_ids:
                cur.execute(
                    "INSERT INTO samples (run_id, log_id, sample_strategy) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (run_id, log_id, strategy),
                )
        conn.commit()
    console.print(f"[sample] {strategy}: {len(sample_ids)} logs sampled")
    return sample_ids


def label_phase(run_id: int, sample_ids: list[int], voting_passes: int = 1) -> dict:
    """Run LLM-as-judge on each sampled log, voting_passes times."""
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, user_prompt, response FROM logs WHERE id = ANY(%s) ORDER BY id",
            (sample_ids,),
        )
        rows = cur.fetchall()

    judged, failed = 0, 0
    for row in rows:
        for pass_idx in range(voting_passes):
            try:
                result = judge_one(
                    row["user_prompt"], row["response"], pass_idx=pass_idx, run_id=run_id
                )
            except Exception as e:
                console.print(f"   [yellow]judge failed for log {row['id']} pass {pass_idx}: {e}[/yellow]")
                failed += 1
                continue
            with db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO label_runs
                      (run_id, log_id, pass_idx, quality_score, difficulty, expected_behavior, raw_response)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        row["id"],
                        pass_idx,
                        result.get("quality_score"),
                        result.get("difficulty"),
                        result.get("expected_behavior"),
                        Jsonb(result),
                    ),
                )
            judged += 1
    console.print(f"[label] judged={judged} failed={failed}")
    return {"label_runs_written": judged, "failed": failed}
