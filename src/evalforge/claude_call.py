"""Single chokepoint for every Claude call in the pipeline.

- Subprocess to `claude -p` (uses local Max-subscription auth).
- Disk cache keyed on (model, system, prompt) — re-runs of the pipeline cost
  nothing once the cache is warm.
- Global concurrency semaphore (default 2) so parallel callers don't melt the
  rate limit window.
- Retry-with-backoff on transient failures.
- Optional Postgres call log via `record_call()` for the dashboard.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass

from . import db
from .config import CACHE_DIR, HAIKU, MAX_CONCURRENCY, TIMEOUT_SECONDS

_semaphore = threading.Semaphore(MAX_CONCURRENCY)


@dataclass
class ClaudeResult:
    text: str
    model: str
    cached: bool
    duration_ms: int
    prompt_hash: str


def _hash(model: str, system: str | None, prompt: str, schema: str | None) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update((system or "").encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    h.update(b"\x00")
    h.update((schema or "").encode())
    return h.hexdigest()


def _cache_path(key: str):
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _write_cache(key: str, data: dict) -> None:
    _cache_path(key).write_text(json.dumps(data))


def _invoke_claude(
    model: str,
    system: str | None,
    prompt: str,
    schema: str | None,
    timeout: int,
) -> str:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--tools",
        "",
    ]
    if system:
        cmd.extend(["--system-prompt", system])
    if schema:
        cmd.extend(["--json-schema", schema])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited {proc.returncode}: {proc.stderr.strip()[:500] or proc.stdout.strip()[:500]}"
        )
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"failed to parse claude output: {e}: {proc.stdout[:500]}") from e

    if envelope.get("is_error"):
        raise RuntimeError(f"claude reported error: {envelope.get('result') or envelope}")

    if schema:
        so = envelope.get("structured_output")
        if isinstance(so, (dict, list)):
            return json.dumps(so)

    for key in ("result", "text", "content", "message"):
        v = envelope.get(key)
        if isinstance(v, str) and v:
            return v
    raise RuntimeError(f"no usable output in claude envelope: keys={list(envelope.keys())} sample={proc.stdout[:300]}")


def claude_call(
    prompt: str,
    *,
    model: str = HAIKU,
    system: str | None = None,
    schema: str | None = None,
    use_cache: bool = True,
    max_retries: int = 3,
    timeout: int = TIMEOUT_SECONDS,
    purpose: str = "unspecified",
    run_id: int | None = None,
) -> ClaudeResult:
    key = _hash(model, system, prompt, schema)

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            record_call(
                run_id=run_id,
                purpose=purpose,
                model=model,
                prompt_hash=key,
                cached=True,
                duration_ms=0,
                success=True,
            )
            return ClaudeResult(text=cached["text"], model=model, cached=True, duration_ms=0, prompt_hash=key)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        start = time.time()
        try:
            with _semaphore:
                text = _invoke_claude(model, system, prompt, schema, timeout)
            duration_ms = int((time.time() - start) * 1000)
            if use_cache:
                _write_cache(key, {"text": text, "model": model})
            record_call(
                run_id=run_id,
                purpose=purpose,
                model=model,
                prompt_hash=key,
                cached=False,
                duration_ms=duration_ms,
                success=True,
            )
            return ClaudeResult(
                text=text, model=model, cached=False, duration_ms=duration_ms, prompt_hash=key
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2**attempt + 1)
            else:
                record_call(
                    run_id=run_id,
                    purpose=purpose,
                    model=model,
                    prompt_hash=key,
                    cached=False,
                    duration_ms=int((time.time() - start) * 1000),
                    success=False,
                    error=str(e)[:500],
                )
    raise RuntimeError(f"claude_call failed after {max_retries} attempts: {last_err}")


def claude_call_json(prompt: str, **kwargs) -> dict:
    """claude_call + parse JSON response. Strips fenced code blocks."""
    result = claude_call(prompt, **kwargs)
    text = result.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def record_call(
    *,
    run_id: int | None,
    purpose: str,
    model: str,
    prompt_hash: str,
    cached: bool,
    duration_ms: int,
    success: bool,
    error: str | None = None,
) -> None:
    """Best-effort log to claude_call_log. Never raises (DB might be down during smoke tests)."""
    try:
        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO claude_call_log
                  (run_id, purpose, model, prompt_hash, cached, duration_ms, success, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, purpose, model, prompt_hash, cached, duration_ms, success, error),
            )
    except Exception:
        pass
