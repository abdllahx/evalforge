"""Phase 0 smoke test: DB reachable + claude_call works + caching kicks in."""
from __future__ import annotations

import sys
import time

from rich.console import Console

from evalforge import db
from evalforge.claude_call import claude_call

console = Console()


def main() -> int:
    console.rule("[bold]evalforge smoke test")

    console.print("\n[bold]1. DB ping[/bold]")
    try:
        ok = db.ping()
        console.print(f"   ✓ Postgres reachable: {ok}")
    except Exception as e:
        console.print(f"   ✗ DB unreachable: {e}", style="red")
        console.print("     → run: docker compose up -d", style="yellow")
        return 1

    console.print("\n[bold]2. Claude call (uncached)[/bold]")
    t0 = time.time()
    r = claude_call(
        "Reply with the single word: PONG",
        system="You are a test responder. Reply with exactly one word, no punctuation.",
        purpose="smoke_test",
    )
    console.print(f"   text     = {r.text!r}")
    console.print(f"   model    = {r.model}")
    console.print(f"   cached   = {r.cached}")
    console.print(f"   duration = {r.duration_ms} ms (wall: {int((time.time()-t0)*1000)} ms)")

    console.print("\n[bold]3. Claude call (cached re-run)[/bold]")
    t0 = time.time()
    r2 = claude_call(
        "Reply with the single word: PONG",
        system="You are a test responder. Reply with exactly one word, no punctuation.",
        purpose="smoke_test",
    )
    console.print(f"   cached   = {r2.cached}")
    console.print(f"   duration = {r2.duration_ms} ms (wall: {int((time.time()-t0)*1000)} ms)")
    if not r2.cached:
        console.print("   ✗ cache miss on second call", style="red")
        return 1

    console.print("\n[bold]4. claude_call_log row written[/bold]")
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM claude_call_log WHERE purpose = 'smoke_test'"
        )
        n = cur.fetchone()["n"]
    console.print(f"   call log rows for smoke_test: {n}")

    console.rule("[bold green]all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
