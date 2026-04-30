"""Score a candidate response against a test case.

Two layers:
1. Pattern matching on must_contain / must_not_contain (deterministic, free).
2. LLM-as-judge comparing candidate_response vs golden_answer (Haiku, structured).

A test case passes iff:
  - all must_contain assertions match, AND
  - none of must_not_contain match, AND
  - the judge's overall_score >= 4.
"""
from __future__ import annotations

import json
from typing import Any

from ..claude_call import claude_call_json
from ..config import HAIKU

JUDGE_SYSTEM = (
    "You compare a candidate AI assistant response to a golden reference. "
    "Score honestly across any domain. Output JSON only."
)

JUDGE_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "matches_intent": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["overall_score", "matches_intent", "reasoning"],
    }
)


def _contains(haystack: str, needle: str) -> bool:
    return needle.strip().lower() in haystack.lower()


def pattern_check(
    response: str, must_contain: list[str], must_not_contain: list[str]
) -> tuple[bool, list[str]]:
    """Returns (passed, list of failure reasons)."""
    reasons: list[str] = []
    for phrase in must_contain or []:
        if not _contains(response, phrase):
            reasons.append(f"missing must_contain: {phrase!r}")
    for phrase in must_not_contain or []:
        if _contains(response, phrase):
            reasons.append(f"matched must_not_contain: {phrase!r}")
    return (not reasons), reasons


def judge_response(
    *,
    user_prompt: str,
    golden_answer: str,
    candidate_response: str,
    expected_behavior: str,
    run_id: int | None = None,
) -> dict[str, Any]:
    body = f"""User prompt:
\"\"\"{user_prompt}\"\"\"

Golden reference answer:
\"\"\"{golden_answer}\"\"\"

Candidate response:
\"\"\"{candidate_response}\"\"\"

Expected behavior: {expected_behavior}

Score the candidate response on:
- overall_score (1-5): how well does it match the golden's helpfulness, correctness, and tone?
- matches_intent: does it follow the expected_behavior (answer/refuse/clarify)?
- reasoning: 1-2 sentences explaining the score.

Output JSON only."""
    return claude_call_json(
        body,
        model=HAIKU,
        system=JUDGE_SYSTEM,
        schema=JUDGE_SCHEMA,
        purpose="eval_judge",
        run_id=run_id,
    )


def score(
    *,
    user_prompt: str,
    golden_answer: str,
    candidate_response: str,
    must_contain: list[str],
    must_not_contain: list[str],
    expected_behavior: str,
) -> dict[str, Any]:
    """Score = hybrid of literal pattern checks (hard guardrails) + LLM judge.

    must_not_contain hits are HARD fails — these catch leaked secrets, wrong
    facts, banned phrases. The auto-generated must_contain phrases are often
    rubric-style sentences, so we treat misses as soft (informational only).
    The LLM judge is the authoritative content scorer.
    """
    _, pat_reasons = pattern_check(candidate_response, must_contain, must_not_contain)
    must_not_hits = [r for r in pat_reasons if r.startswith("matched must_not_contain")]
    must_contain_misses = [r for r in pat_reasons if r.startswith("missing must_contain")]

    judge = judge_response(
        user_prompt=user_prompt,
        golden_answer=golden_answer,
        candidate_response=candidate_response,
        expected_behavior=expected_behavior,
    )
    overall = int(judge.get("overall_score", 0))
    intent_ok = bool(judge.get("matches_intent"))

    passed = not must_not_hits and overall >= 4 and intent_ok

    failure_reasons: list[str] = list(must_not_hits)
    if overall < 4:
        failure_reasons.append(f"judge_overall_score={overall} (<4)")
    if not intent_ok:
        failure_reasons.append("judge: did not match expected behavior")
    if must_contain_misses:
        failure_reasons.extend(f"[soft] {r}" for r in must_contain_misses[:3])

    return {
        "passed": passed,
        "score": float(overall),
        "judge_reasoning": judge.get("reasoning", ""),
        "failure_reasons": failure_reasons,
    }
