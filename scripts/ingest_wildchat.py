"""Stream WildChat-4.8M-Full and ingest into Postgres."""
from __future__ import annotations

import sys

import typer
from rich.console import Console

from evalforge import db
from evalforge.ingestion.adapters import ingest
from evalforge.ingestion.wildchat import stream_wildchat

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    n: int = typer.Option(200),
    language: str = typer.Option("English"),
    no_redact: bool = typer.Option(False),
):
    console.rule(f"[bold]WildChat ingest — n={n} language={language!r}")
    entries = list(stream_wildchat(n, language=language))
    console.print(f"   streamed: {len(entries)} entries")

    with db.connect() as conn:
        stats = ingest(conn, entries, redact_pii=not no_redact)
    for k, v in stats.items():
        console.print(f"   {k:30s} = {v}")
    console.rule("[bold green]done")
    return 0


if __name__ == "__main__":
    sys.exit(app() or 0)
