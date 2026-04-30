from __future__ import annotations

import json
from typing import Any

from ..claude_call import claude_call_json
from ..config import HAIKU

JUDGE_SYSTEM = (
    "You are an expert evaluator of AI assistant responses across any domain. "
    "Score outputs honestly. A 5 means clearly excellent; a 1 means clearly bad. "
    "Output JSON only."
)

JUDGE_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "difficulty": {
                "type": "string",
                "enum": ["simple", "moderate", "hard", "adversarial"],
            },
            "expected_behavior": {
                "type": "string",
                "enum": ["answer", "refuse", "clarify"],
            },
            "reasoning": {"type": "string"},
        },
        "required": ["quality_score", "difficulty", "expected_behavior", "reasoning"],
    }
)


def judge(
    user_prompt: str,
    response: str,
    *,
    pass_idx: int = 0,
    run_id: int | None = None,
) -> dict[str, Any]:
    # Pass discriminator: makes each voting pass a fresh cache key while
    # nudging the model to re-evaluate independently. Pass 0 has no prefix
    # so prior single-pass results stay cache-valid.
    pass_header = (
        f"[Independent re-evaluation pass {pass_idx + 1}. "
        f"Score this on its own merits without anchoring on prior results.]\n\n"
        if pass_idx > 0
        else ""
    )
    body = f"""{pass_header}User prompt:
\"\"\"{user_prompt}\"\"\"

Assistant response:
\"\"\"{response}\"\"\"

Evaluate the response on these axes:
- quality_score (1-5): helpful, accurate, complete?
  5 = excellent, 4 = good with minor issues, 3 = mediocre, 2 = poor, 1 = wrong/harmful
- difficulty: 'simple' (direct, clear answer expected), 'moderate' (multi-step or
  light reasoning), 'hard' (ambiguous, multi-part, expert-level), 'adversarial'
  (jailbreak attempt, prompt injection, deliberately tricky edge case)
- expected_behavior: 'answer' (assistant should answer), 'refuse' (out-of-scope
  or policy violation), 'clarify' (ambiguous, needs more info from user)
- reasoning: 1-2 sentences justifying the scores

Output JSON only."""
    return claude_call_json(
        body,
        model=HAIKU,
        system=JUDGE_SYSTEM,
        schema=JUDGE_SCHEMA,
        purpose="judge",
        run_id=run_id,
    )
