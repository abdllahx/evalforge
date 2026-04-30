"""Nightly pipeline scheduler.

Runs `run_pipeline.py` on a cron schedule via APScheduler. Designed to live
in the `scheduler` service of docker-compose.yml so the eval suite literally
"grows while you sleep."

Auth: the in-container `claude` CLI needs `ANTHROPIC_API_KEY`. The local-machine
OAuth/Max subscription does NOT propagate into containers. Set the env in
docker-compose.yml or .env. Without it, candidate / judge calls will fail and
the pipeline will exit nonzero (logged, but loop continues for the next tick).
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
PIPELINE_SCRIPT = ROOT / "scripts" / "run_pipeline.py"

CRON = os.getenv("EVALFORGE_CRON", "0 3 * * *")  # 03:00 UTC daily
TIMEZONE = os.getenv("EVALFORGE_TZ", "UTC")
ARGS = os.getenv("EVALFORGE_PIPELINE_ARGS", "").split() if os.getenv("EVALFORGE_PIPELINE_ARGS") else []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("evalforge.scheduler")


def run_pipeline() -> None:
    started = datetime.utcnow().isoformat()
    log.info("scheduled pipeline run starting at %s", started)
    cmd = [PYTHON, str(PIPELINE_SCRIPT), *ARGS]
    proc = subprocess.run(cmd, cwd=ROOT)
    log.info("pipeline finished rc=%d", proc.returncode)


def main() -> None:
    sched = BlockingScheduler(timezone=TIMEZONE)
    trigger = CronTrigger.from_crontab(CRON, timezone=TIMEZONE)
    sched.add_job(run_pipeline, trigger=trigger, id="evalforge_nightly", coalesce=True, max_instances=1)
    log.info("scheduler armed: %s (%s)", CRON, TIMEZONE)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: sys.exit(0))

    sched.start()


if __name__ == "__main__":
    main()
