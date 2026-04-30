"""Dedup new candidates against the existing eval_dataset.

Embeds candidate prompts and existing dataset prompts; rejects candidates with
cosine similarity above the threshold to anything already in the dataset.
"""
from __future__ import annotations

import numpy as np

from .. import db
from ..classifier.embeddings import embed
from ..config import DEDUP_THRESHOLD


def filter_duplicates(
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Returns (kept, rejected). Each rejected dict gets a 'duplicate_of' key."""
    if not candidates:
        return [], []

    with db.cursor() as cur:
        cur.execute("SELECT id, user_prompt FROM eval_dataset")
        existing = cur.fetchall()

    if not existing:
        return candidates, []

    cand_emb = embed([c["user_prompt"] for c in candidates])
    exist_emb = embed([e["user_prompt"] for e in existing])

    sims = cand_emb @ exist_emb.T
    kept: list[dict] = []
    rejected: list[dict] = []
    for i, c in enumerate(candidates):
        max_idx = int(np.argmax(sims[i]))
        max_sim = float(sims[i][max_idx])
        if max_sim >= DEDUP_THRESHOLD:
            rejected.append({**c, "duplicate_of": existing[max_idx]["id"], "similarity": max_sim})
        else:
            kept.append(c)
    return kept, rejected
