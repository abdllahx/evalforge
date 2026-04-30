from __future__ import annotations

import json

from ..claude_call import claude_call_json
from ..config import HAIKU

NAMING_SYSTEM = (
    "You name clusters of user prompts for an AI eval dataset. "
    "Be specific and concrete. Output JSON only."
)

NAMING_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["name", "description"],
    }
)


def name_cluster(prompts: list[str], *, run_id: int | None = None, cluster_idx: int = 0) -> tuple[str, str]:
    examples = "\n".join(f"- {p[:200]}" for p in prompts[:5])
    user_prompt = f"""These user prompts were grouped into one cluster:

{examples}

Give:
- name: 2-4 words, Title Case (e.g. "Code debugging help" or "Travel planning").
- description: ONE sentence describing what unifies them.

Output JSON only."""
    result = claude_call_json(
        user_prompt,
        model=HAIKU,
        system=NAMING_SYSTEM,
        schema=NAMING_SCHEMA,
        purpose=f"cluster_naming:{cluster_idx}",
        run_id=run_id,
    )
    return result["name"], result["description"]
