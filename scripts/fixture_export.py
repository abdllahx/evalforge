"""Snapshot Postgres state into a SQLite fixture for the Streamlit Cloud demo.

The fixture is committed to the repo so reviewers can poke around the dashboard
without running the pipeline.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "dashboard" / "fixture_schema.sql"
DEFAULT_OUT = ROOT / "fixtures" / "snapshot.sqlite"

TABLES_IN_ORDER = [
    "logs",
    "pipeline_runs",
    "samples",
    "clusters",
    "log_cluster_assignment",
    "label_runs",
    "labels",
    "eval_dataset",
    "eval_runs",
    "eval_results",
    "claude_call_log",
]

JSON_COLUMNS = {
    "logs": ["metadata"],
    "pipeline_runs": ["config"],
    "clusters": ["representative_log_ids"],
    "log_cluster_assignment": ["outlier_reasons"],
    "label_runs": ["raw_response"],
    "labels": ["must_contain", "must_not_contain"],
    "eval_dataset": ["rubric", "must_contain", "must_not_contain"],
    "eval_runs": ["config"],
    "eval_results": ["failure_reasons"],
}

BOOL_COLUMNS = {
    "log_cluster_assignment": ["is_outlier"],
    "labels": ["needs_review"],
    "eval_results": ["passed"],
    "claude_call_log": ["cached", "success"],
}

console = Console()


def _convert(table: str, row: dict) -> tuple:
    out = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif k in JSON_COLUMNS.get(table, []):
            out[k] = json.dumps(v)
        elif k in BOOL_COLUMNS.get(table, []):
            out[k] = 1 if v else 0
        elif isinstance(v, datetime):
            out[k] = v.astimezone(UTC).isoformat()
        else:
            out[k] = v
    return out


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    sqlite_conn = sqlite3.connect(out_path)
    sqlite_conn.executescript(SCHEMA.read_text())

    import os
    pg_url = os.getenv("DATABASE_URL", "postgresql://evalforge:evalforge@localhost:5433/evalforge")
    with psycopg.connect(pg_url, row_factory=dict_row) as pg:
        for table in TABLES_IN_ORDER:
            with pg.cursor() as cur:
                cur.execute(f"SELECT * FROM {table}")
                rows = cur.fetchall()
            if not rows:
                console.print(f"   {table:<28} 0 rows")
                continue
            converted = [_convert(table, r) for r in rows]
            cols = list(converted[0].keys())
            placeholders = ",".join(["?"] * len(cols))
            sqlite_conn.executemany(
                f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
                [tuple(r[c] for c in cols) for r in converted],
            )
            console.print(f"   {table:<28} {len(rows)} rows")

    sqlite_conn.commit()
    sqlite_conn.close()
    console.print(f"\n[green]wrote {out_path} ({out_path.stat().st_size // 1024} KB)[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
