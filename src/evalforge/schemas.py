from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    occurred_at: datetime
    feature: str
    user_prompt: str
    system_prompt: str | None = None
    model: str
    response: str
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    user_feedback: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampledLog(BaseModel):
    log_id: int
    feature: str
    user_prompt: str
    response: str
    user_feedback: str | None
    latency_ms: int | None
    metadata: dict[str, Any]
