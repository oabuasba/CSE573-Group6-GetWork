from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from getwork.webapp.models import Direction, DocumentView, ExplainerInfo, SystemId

FactKind = Literal[
    "shared_skills",
    "related_skills",
    "gap_skills",
    "level",
    "title_overlap",
    "document_skills",
    "rank_score",
    "comparison",
    "cluster_members",
]
FailureReason = Literal[
    "timeout",
    "budget_exhausted",
    "llm_unavailable",
    "model_rejected",
    "invalid_reply",
]
QUERY_DOC: str = "query"
PROMPT_VERSION: str = "agent_v3"
MAX_MESSAGE_CHARS: int = 1000
MAX_REPLY_CHARS: int = 4000
MAX_HISTORY: int = 20
MAX_HISTORY_CHARS: int = 2000
MAX_MATCH_REFS: int = 50
MAX_TOOL_STEPS: int = 4
MAX_LLM_CALLS: int = 6
DEFAULT_TIMEOUT_S: float = 60.0
AGENT_ERROR_CODE: str = "assistant_failed"
AGENT_ERROR_STATUS: int = 502


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Fact(BaseModel):
    id: str  # "f1", "f2", ... in creation order within a turn
    kind: FactKind
    doc_ids: list[str]  # QUERY_DOC or corpus ids
    skills: list[str]
    cluster: str | None = None
    text: str  # code-generated sentence


class ToolTrace(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None
    fact_ids: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    llm_calls: int
    tool_calls: int
    elapsed_ms: int


class ChatMessage(_Strict):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_HISTORY_CHARS)


class ChatMatchRef(_Strict):
    id: str = Field(min_length=1, max_length=64)
    rank: int = Field(ge=1)
    score: float


class ChatRequest(_Strict):
    direction: Direction
    query: DocumentView
    focus_id: str = Field(min_length=1, max_length=64)
    matches: list[ChatMatchRef] = Field(max_length=MAX_MATCH_REFS)
    system: SystemId = "Full"
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)  # stripped; blank is invalid

    @field_validator("message", mode="before")
    @classmethod
    def _strip_message(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ChatResponse(BaseModel):
    reply: str  # the model's text, at most MAX_REPLY_CHARS characters
    trace: list[ToolTrace]
    usage: Usage
    explainer: ExplainerInfo  # kind "agent"


@dataclass(frozen=True)
class AgentResult:
    status: Literal["ok", "failed"]
    reply: str  # "" when failed
    facts: list[Fact]
    trace: list[ToolTrace]
    usage: Usage
    model: str
    backend: str
    prompt_version: str
    failure_reason: FailureReason | None = None  # set exactly when status == "failed"
    failure_detail: str | None = None  # redacted cause for the log; deliberately not in as_record
    ungrounded_skills: list[str] = field(default_factory=list)  # vocabulary skills named but in no document

    def as_record(self) -> dict:
        """JSON-serializable dict with exactly the keys: status, failure_reason, reply, facts, trace, usage, model,
        backend, prompt_version, ungrounded_skills (facts, trace, usage via model_dump)."""
        return {
            "status": self.status,
            "failure_reason": self.failure_reason,
            "reply": self.reply,
            "facts": [fact.model_dump() for fact in self.facts],
            "trace": [step.model_dump() for step in self.trace],
            "usage": self.usage.model_dump(),
            "model": self.model,
            "backend": self.backend,
            "prompt_version": self.prompt_version,
            "ungrounded_skills": list(self.ungrounded_skills),
        }
