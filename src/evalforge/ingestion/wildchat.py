"""WildChat-4.8M (non-toxic) ingestion adapter.

Streams from `allenai/WildChat-4.8M` (no full download), filters to English,
maps to LogEntry. Real signals:

  * `user_feedback = 'moderation_flag'` when `toxic=True` or per-turn
    OpenAI moderation flagged the user prompt (rare in the non-toxic split).
  * `user_feedback = 'retry'` for multi-turn convos (turn > 1).
  * `feature = model name` (e.g. "gpt-4-0314", "gpt-3.5-turbo-0301").
  * Real `occurred_at` from row.timestamp (no synthesis needed).

The non-toxic 4.8M split is public (no gating). Set `DATASET_ID =
"allenai/WildChat-4.8M-Full"` for the gated full version with toxic content.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

from ..schemas import LogEntry

DATASET_ID = "allenai/WildChat-4.8M"


def _first_pair(conversation: list[dict]) -> tuple[str, str] | None:
    user_msg, assistant_msg = None, None
    for msg in conversation:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if user_msg is None and role == "user":
            user_msg = content
            continue
        if user_msg is not None and assistant_msg is None and role == "assistant":
            assistant_msg = content
            break
    if user_msg and assistant_msg:
        return user_msg, assistant_msg
    return None


def _is_flagged(row: dict, conversation: list[dict]) -> tuple[bool, list[str]]:
    """Top-level toxic flag + per-turn moderation. Either trips the flag."""
    if row.get("toxic"):
        return True, ["toxic"]
    # WildChat stores openai_moderation per message inside conversation
    cats: list[str] = []
    flagged = False
    for msg in conversation[:1]:  # only the first user message
        mod = msg.get("openai_moderation")
        if isinstance(mod, dict):
            if mod.get("flagged"):
                flagged = True
            c = mod.get("categories") or {}
            cats.extend(k for k, v in (c.items() if isinstance(c, dict) else []) if v)
    return flagged, cats


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


def stream_wildchat(n: int, *, language: str = "English") -> Iterator[LogEntry]:
    """Stream up to `n` LogEntry objects from WildChat-4.8M-Full."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    yielded = 0
    for row in ds:
        if yielded >= n:
            break
        if row.get("language") != language:
            continue
        conversation = row.get("conversation") or []
        pair = _first_pair(conversation)
        if pair is None:
            continue
        user_prompt, response = pair

        flagged, triggered = _is_flagged(row, conversation)
        turn_count = int(row.get("turn") or 0)

        if flagged:
            feedback = "moderation_flag"
        elif turn_count > 1:
            feedback = "retry"
        else:
            feedback = None

        model = (row.get("model") or "unknown").strip() or "unknown"
        ts = _parse_ts(row.get("timestamp"))

        yield LogEntry(
            occurred_at=ts,
            feature=model,
            user_prompt=user_prompt,
            system_prompt=None,
            model=model,
            response=response,
            latency_ms=None,
            prompt_tokens=max(1, len(user_prompt) // 4),
            completion_tokens=max(1, len(response) // 4),
            user_feedback=feedback,
            metadata={
                "source": "wildchat-4.8m",
                "conversation_id": row.get("conversation_id"),
                "language": row.get("language"),
                "turn_count": turn_count,
                "redacted": bool(row.get("redacted")),
                "toxic": bool(row.get("toxic")),
                "moderation_flagged": flagged,
                "moderation_categories": triggered,
                "country": row.get("country"),
            },
        )
        yielded += 1
