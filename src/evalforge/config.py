import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://evalforge:evalforge@localhost:5432/evalforge")

CACHE_DIR = Path(os.getenv("EVALFORGE_CACHE_DIR", str(ROOT / "cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENCY = int(os.getenv("EVALFORGE_MAX_CONCURRENCY", "2"))
TIMEOUT_SECONDS = int(os.getenv("EVALFORGE_TIMEOUT", "120"))

PRODUCT_NAME = os.getenv("EVALFORGE_PRODUCT_NAME", "WildChat")
SAMPLE_SIZE = int(os.getenv("EVALFORGE_SAMPLE_SIZE", "30"))
VOTING_PASSES = int(os.getenv("EVALFORGE_VOTING_PASSES", "1"))
DEDUP_THRESHOLD = float(os.getenv("EVALFORGE_DEDUP_THRESHOLD", "0.92"))

HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-4-6"
