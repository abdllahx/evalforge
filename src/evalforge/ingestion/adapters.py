"""Adapters for ingesting logs from different sources into the unified schema.

JSON-file is the canonical adapter (and the one we actually use). The OTel
adapter is a stub showing how the same shape would map from a trace export —
enough to demonstrate the design without a real OTel pipeline.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from psycopg import Connection
from psycopg.types.json import Jsonb

from ..schemas import LogEntry
from .redaction import redact


def _content_hash(entry: LogEntry) -> str:
    h = hashlib.sha256()
    h.update(entry.feature.encode())
    h.update(b"\x00")
    h.update(entry.user_prompt.encode())
    h.update(b"\x00")
    h.update(entry.response.encode())
    h.update(b"\x00")
    h.update(entry.occurred_at.isoformat().encode())
    return h.hexdigest()


def read_json_logs(path: str | Path) -> Iterator[LogEntry]:
    """Read a JSON file containing a list of log entries."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data).__name__}")
    for raw in data:
        yield LogEntry.model_validate(raw)


def read_otel_traces(path: str | Path) -> Iterator[LogEntry]:
    """Stub adapter: map OTel-shaped JSON spans to our LogEntry schema.

    Real-world: parse OTLP spans, find LLM-call spans by attributes
    (e.g. `gen_ai.system`), pull user/assistant from events. Here we accept a
    minimal shape to demonstrate the contract.
    """
    with open(path) as f:
        data = json.load(f)
    for span in data.get("spans", []):
        attrs = span.get("attributes", {})
        yield LogEntry(
            occurred_at=span["start_time"],
            feature=attrs.get("app.feature", "unknown"),
            user_prompt=attrs.get("gen_ai.prompt", ""),
            system_prompt=attrs.get("gen_ai.system_prompt"),
            model=attrs.get("gen_ai.model", "unknown"),
            response=attrs.get("gen_ai.response", ""),
            latency_ms=int((span["end_time"] - span["start_time"]) * 1000)
            if "end_time" in span
            else None,
            prompt_tokens=attrs.get("gen_ai.usage.prompt_tokens"),
            completion_tokens=attrs.get("gen_ai.usage.completion_tokens"),
            user_feedback=attrs.get("user.feedback"),
            metadata={"trace_id": span.get("trace_id"), "span_id": span.get("span_id")},
        )


def ingest(
    conn: Connection,
    entries: Iterable[LogEntry],
    *,
    redact_pii: bool = True,
) -> dict[str, int]:
    """Write entries into the logs table. Dedups via content_hash. Returns stats."""
    inserted = 0
    skipped = 0
    redacted_total: dict[str, int] = {}

    with conn.cursor() as cur:
        for entry in entries:
            if redact_pii:
                rp = redact(entry.user_prompt)
                rr = redact(entry.response)
                user_prompt, response = rp.text, rr.text
                for k, v in {**rp.counts, **rr.counts}.items():
                    redacted_total[k] = redacted_total.get(k, 0) + v
            else:
                user_prompt, response = entry.user_prompt, entry.response

            stamped = entry.model_copy(update={"user_prompt": user_prompt, "response": response})
            chash = _content_hash(stamped)

            cur.execute(
                """
                INSERT INTO logs
                  (occurred_at, feature, user_prompt, system_prompt, model, response,
                   latency_ms, prompt_tokens, completion_tokens, user_feedback,
                   metadata, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                (
                    stamped.occurred_at,
                    stamped.feature,
                    stamped.user_prompt,
                    stamped.system_prompt,
                    stamped.model,
                    stamped.response,
                    stamped.latency_ms,
                    stamped.prompt_tokens,
                    stamped.completion_tokens,
                    stamped.user_feedback,
                    Jsonb(stamped.metadata),
                    chash,
                ),
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped += 1
    conn.commit()
    return {"inserted": inserted, "skipped_duplicates": skipped, **{f"redacted_{k}": v for k, v in redacted_total.items()}}
