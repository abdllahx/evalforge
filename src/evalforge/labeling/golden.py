from __future__ import annotations

import json
from typing import Any

from ..claude_call import claude_call_json
from ..config import SONNET

GOLDEN_SYSTEM = (
    "You generate reference answers and assertions for an AI evaluation dataset, "
    "based on a real prior interaction. Be concrete and specific. Output JSON only."
)

GOLDEN_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "golden_answer": {"type": "string"},
            "must_contain": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "must_not_contain": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rubric_notes": {"type": "string"},
        },
        "required": ["golden_answer", "must_contain", "must_not_contain", "rubric_notes"],
    }
)


def generate_golden(
    *,
    user_prompt: str,
    original_response: str,
    judge_quality: int,
    judge_difficulty: str,
    judge_behavior: str,
    judge_reasoning: str,
    run_id: int | None = None,
) -> dict[str, Any]:
    """Generate a golden answer + must/must-not assertions for one test case.

    The judge's labels feed in so the generator knows whether the original
    response was good (use it as a baseline) or bad (write a clearly better
    one). Always returns the same schema, regardless.
    """
    body = f"""You are constructing a single test case for a general-purpose AI assistant eval dataset.

User prompt:
\"\"\"{user_prompt}\"\"\"

Original assistant response (for reference):
\"\"\"{original_response}\"\"\"

Prior judgement:
  quality_score = {judge_quality}/5
  difficulty = {judge_difficulty}
  expected_behavior = {judge_behavior}
  reasoning = {judge_reasoning}

Produce:
- golden_answer: an ideal reference response. If the original was clearly good (quality 4-5), you may stay close to it; if poor (1-3), write a clearly better version. Match the expected_behavior:
    * 'answer': directly help; concrete, accurate, on-topic.
    * 'refuse': decline appropriately, briefly explain why, offer safe alternative if any.
    * 'clarify': ask exactly the question(s) needed to disambiguate.
- must_contain: 2-4 SHORT, LITERAL substrings a correct response MUST contain.
  Prefer concrete facts, numbers, key terms (e.g. "10,201", "O(n log n)", "boiling point",
  "I can't help with that"). NOT rubric sentences. Each item should be a string a
  reviewer can grep for verbatim in the candidate response.
- must_not_contain: 1-3 LITERAL substrings a response should NEVER include —
  hallucinations, wrong facts, banned content. Pull verbatim from the original
  response if the judge marked it bad and you can identify a wrong claim.
- rubric_notes: 1 sentence on what makes this test case meaningful.

Output JSON only. No prose."""
    return claude_call_json(
        body,
        model=SONNET,
        system=GOLDEN_SYSTEM,
        schema=GOLDEN_SCHEMA,
        purpose="golden_answer",
        run_id=run_id,
    )
