"""Phase 4: run eval_dataset against one or more candidates, optionally diff."""
from __future__ import annotations

import sys

import typer
from rich.console import Console

from evalforge.eval_runner.candidate import CANDIDATES
from evalforge.eval_runner.regression import compare
from evalforge.eval_runner.runner import run_eval

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    candidates: list[str] = typer.Option(
        None,
        help="Candidate label(s). Defaults to baseline-haiku + strong-sonnet + regressed.",
    ),
    limit: int = typer.Option(0, help="Cap on test cases (0 = all)"),
    diff: bool = typer.Option(True, help="Diff the two most-recent runs to detect regressions"),
):
    if not candidates:
        candidates = ["baseline-haiku", "strong-sonnet", "regressed-broken-prompt"]

    for c in candidates:
        if c not in CANDIDATES:
            raise typer.BadParameter(f"unknown candidate '{c}'. choose from {list(CANDIDATES)}")

    run_ids: list[int] = []
    for c in candidates:
        result = run_eval(c, limit=limit or None)
        run_ids.append(result["eval_run_id"])

    if diff and len(run_ids) >= 2:
        console.rule("[bold]regression diff (last vs first)")
        diff_result = compare(run_ids[0], run_ids[-1])
        console.print(f"   compared_cases  = {diff_result['compared_cases']}")
        console.print(f"   new_failures    = {len(diff_result['new_failures'])}")
        console.print(f"   new_passes      = {len(diff_result['new_passes'])}")
        console.print(f"   score_deltas≥1  = {len(diff_result['score_deltas'])}")
        for nf in diff_result["new_failures"][:5]:
            console.print(f"     [red]– {nf['user_prompt'][:80]}…[/red]")

    return 0


if __name__ == "__main__":
    sys.exit(app() or 0)
