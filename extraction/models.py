from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LEVELS = ("introductory", "intermediate", "advanced")
KINDS = ("concept", "technique", "tool", "language")


class ExtractedTopic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="lowercase snake_case English name of the concept, without the technology")
    name: str = Field(description="short human-readable name")
    description: str = Field(description="one sentence on what the topic covers")
    evidence: list[str] = Field(description="ids of the syllabus items the topic comes from")
    share: float = Field(description="fraction of the course's teaching time spent on the topic")
    level: Literal["introductory", "intermediate", "advanced"]
    kind: Literal["concept", "technique", "tool", "language"]
    technology: str | None = Field(description="language or tool the topic is tied to, or null")


class NotTopic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    reason: str


class Extraction(BaseModel):
    """The model's reply for one course."""

    model_config = ConfigDict(extra="forbid")

    topics: list[ExtractedTopic]
    not_topics: list[NotTopic]


class EndpointSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # openai: an OpenAI-compatible server at `url`; claude-cli: headless Claude
    # Code (`claude -p`) on this machine, billed to its claude.ai login.
    provider: Literal["openai", "claude-cli"] = "openai"
    url: str | None = Field(default=None, description="server base URL, e.g. http://localhost:8080 (openai only)")
    workers: int = Field(default=1, ge=1, description="requests sent to this endpoint at once")
    model: str | None = Field(default=None, description="model name the server expects, if different")
    api_key: str | None = None
    command: str = Field(default="claude", description="Claude Code executable (claude-cli only)")

    @model_validator(mode="after")
    def _check_provider(self) -> "EndpointSpec":
        if self.provider == "openai" and not self.url:
            raise ValueError("an openai endpoint needs a url")
        if self.provider == "claude-cli" and self.api_key:
            raise ValueError("a claude-cli endpoint uses the Claude Code login, not an api_key")
        return self


class EndpointsFile(BaseModel):
    """Which model to use and where it is served."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="logical model name, recorded in outputs and cache keys")
    endpoints: list[EndpointSpec] = Field(min_length=1)
    # Fields merged into every request body (temperature, seed, max_tokens,
    # chat_template_kwargs, ...). They are part of the cache key.
    request: dict[str, Any] = Field(default_factory=dict)
    timeout: float = Field(default=900.0, gt=0, description="seconds per request")


@dataclass(frozen=True)
class RunConfig:
    max_attempts: int = 3
    # Connection failures tolerated per endpoint before its workers stop.
    max_endpoint_failures: int = 3
    force: bool = False


@dataclass
class Completion:
    content: str
    usage: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
