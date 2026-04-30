# evalforge

> Mine real production LLM logs, auto-label them, and grow an eval dataset that catches both regressions AND scaling improvements.

![Dashboard overview](docs/images/01-overview-clusters.png)

The hard part of LLM evaluation isn't writing the harness — it's curating the dataset. Hand-built golden sets go stale. **evalforge** treats production traffic as the source of truth: it samples real interactions, clusters them into behavioral categories, generates golden answers and assertions with an LLM, dedups against existing test cases, and runs the resulting eval against any candidate model — flagging regressions on every run.

## What it does (numbers from the latest run)

```
200 real GPT-3.5 / GPT-4 conversations from WildChat-4.8M
        │
        ▼
   Embed (MiniLM) + UMAP → HDBSCAN ──► 20 behavioral clusters auto-named:
        │                  "Autonomous Web Crawling", "Jailbreak Attempts",
        │                  "Mathematical Problem Solving", "Creative Prompt
        │                  Generation", "Toxic Response Solicitation", …
        ▼
   Outlier detection ──► retry, moderation_flag, length, hdbscan-noise
        │
        ▼
   Sample (signal-boosted / coverage-aware) ──► 30 candidate cases
        │
        ▼
   LLM-as-judge × 3 voting passes (Haiku) ──► quality / difficulty /
        │                                    expected-behavior + confidence
        ▼
   Golden answers + assertions (Sonnet) ──► 30 reference answers
        │
        ▼
   Dedup + confidence routing ──► 25 auto-added · 5 in human-review queue
        │
        ▼
   Eval runner ──► 3 candidates × 25 cases = 75 results
                  baseline-haiku  : 40% pass
                  strong-sonnet   : 60% pass    ← scaling improvement detected
                  regressed-prompt: 28% pass    ← regression detected
```

The whole pipeline ran end-to-end on a Claude Code Max subscription via a single `claude -p` subprocess wrapper with disk caching. Re-runs are free.

## Dashboard tour

| | |
|---|---|
| **Coverage heatmap + freshness** | ![](docs/images/02-coverage.png) |
| **Eval runs (table + pass-rate)** | ![](docs/images/03a-eval-runs-table.png) <br/> ![](docs/images/03b-eval-passrate.png) |
| **Regression diff (score deltas)** | ![](docs/images/04-regression-diff.png) |
| **Review queue with approve/reject** | ![](docs/images/05-review-queue.png) |

## Stack

| | |
|---|---|
| Language | Python 3.12 (managed via `uv`) |
| Live store | Postgres 16 (Docker, port 5433) |
| Demo store | SQLite snapshot in `fixtures/snapshot.sqlite` for Streamlit Cloud |
| LLM | Claude Code Max — Haiku for high-volume calls, Sonnet for quality work |
| Data source | `allenai/WildChat-4.8M` (non-toxic, public) — real ChatGPT prod traffic |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Clustering | UMAP (cosine, 8d) → HDBSCAN |
| Dashboard | Streamlit + Altair (7 tabs incl. coverage heatmap, freshness gauge, regression diff) |
| Scheduler | APScheduler in `scheduler` container (opt-in via docker-compose profile) |

## Quick start

```bash
docker compose up -d              # Postgres
uv sync                           # Python deps

# 1. Smoke test
uv run python scripts/smoke_test.py

# 2. Ingest real prod logs (gated dataset — set HF_TOKEN in .env)
uv run python scripts/ingest_wildchat.py --n 200

# 3. Full pipeline: cluster → sample → judge → golden → curate → eval × 3 candidates
uv run python scripts/run_pipeline.py

# 4. (Optional) Multi-pass voting for real confidence
uv run python scripts/run_voting.py

# 5. Dashboard
uv run streamlit run dashboard/app.py

# Tests
uv run pytest -q
```

The full pipeline costs ~220 cached Claude calls end-to-end (Haiku-heavy mix). Disk caching at `cache/*.json` makes every re-run free.

## Architecture

```
src/evalforge/
  claude_call.py          # subprocess wrapper · disk cache · concurrency cap · structured logging
  ingestion/
    redaction.py          # regex PII redaction (emails, cards, account/invoice IDs, …)
    adapters.py           # JSON-file + OTel-trace adapters · dedup-on-ingest
    sampling.py           # uniform · stratified · signal_boosted · coverage_aware
    wildchat.py           # WildChat-4.8M streaming adapter (HF datasets)
  classifier/
    embeddings.py         # MiniLM, normalized
    clustering.py         # UMAP → HDBSCAN · representative selection
    naming.py             # cluster names from a Haiku call
    judge.py              # quality + difficulty + expected_behavior in one structured Haiku call
                          # (multi-pass voting: pass>0 prepends a discriminator for fresh cache keys)
  labeling/
    golden.py             # Sonnet generates reference answer + must/must-not assertions (literal substrings)
    aggregate.py          # per-pass votes → single label row · agreement-based confidence
    dedup.py              # cosine sim against existing eval_dataset (threshold 0.92)
    curate.py             # confidence routing · auto-add vs review queue · eval_dataset writes
  eval_runner/
    candidate.py          # baseline-haiku · strong-sonnet · regressed-broken-prompt
    scorer.py             # must_not_contain hard guardrails + LLM judge as content scorer
    runner.py             # full eval over the dataset · writes eval_runs + eval_results
    regression.py         # diff two eval_runs · new failures · score deltas
  pipeline.py             # phase-by-phase orchestration

dashboard/                # Streamlit app:
                          #   Clusters · Coverage · Eval dataset · Review queue
                          #   · Eval runs · Regression diff · Claude usage
sql/001_schema.sql        # Postgres tables
fixtures/snapshot.sqlite  # Postgres snapshot for fixture-mode dashboard
scripts/                  # smoke_test · ingest_wildchat · run_pipeline · run_voting
                          # · resume_pipeline · fixture_export · scheduler
Dockerfile                # used by docker-compose `scheduler` and `dashboard` profiles
docker-compose.yml        # postgres + scheduler (cron'd nightly pipeline) + dashboard
```

## Cost-conscious design

- **One chokepoint for every LLM call** (`src/evalforge/claude_call.py`): sha256 cache key over `(model, system, prompt, schema)`. After one full run, every subsequent run is a cache hit.
- **Haiku for hot paths** (judging, cluster naming, eval scoring, multi-pass voting); **Sonnet only for quality-sensitive generation** (golden answers).
- **Concurrency capped at 2** by default (`EVALFORGE_MAX_CONCURRENCY`) so parallel callers don't burn through the rate-limit window.
- **Pattern checks vs LLM judge are separated**: `must_not_contain` is a hard guardrail (catches leaks/hallucinations); the LLM judge is the authoritative content scorer (catches regressions).

## Streamlit Cloud demo (fixture mode)

The dashboard auto-detects fixture mode: with `EVALFORGE_FIXTURE_MODE=1` (or no `DATABASE_URL` and a present `fixtures/snapshot.sqlite`), it reads from the committed SQLite snapshot. To deploy:

1. Push the repo to GitHub (with `fixtures/snapshot.sqlite` committed).
2. On share.streamlit.io: point at `dashboard/app.py`, dependency file `dashboard/requirements.txt`, env var `EVALFORGE_FIXTURE_MODE=1`.
3. Done — reviewers can poke at the data without running the pipeline.

## Honesty / limitations

- WildChat-4.8M is real prod traffic from ChatGPT, but timestamps are synthesized for the demo because the dataset's per-row dates aren't always populated. Adapters are domain-agnostic: swap to your own log shape with one new file in `ingestion/`.
- "Candidate models" in the eval are real model + system-prompt combinations (Haiku-helpful, Sonnet-helpful, Haiku-broken-prompt) — designed to surface both regressions and scaling improvements. The harness is generic and would run against any HTTP endpoint with a small adapter swap.
- `must_contain` assertions auto-generated by Sonnet are sometimes rubric-style sentences; the scorer treats those as soft, with the LLM judge as the authoritative pass/fail signal.
- Inter-annotator agreement requires multiple human reviewers — the dashboard shows confidence from auto-voting agreement instead.
