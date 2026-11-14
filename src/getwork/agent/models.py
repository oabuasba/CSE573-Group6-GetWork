from dataclasses import dataclass
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
    "unverified",
]
QUERY_DOC: str = "query"
PROMPT_VERSION: str = "agent_v1"
MAX_MESSAGE_CHARS: int = 1000
MAX_HISTORY: int = 20
MAX_HISTORY_CHARS: int = 2000
MAX_MATCH_REFS: int = 50
MAX_CLAIMS: int = 8
MAX_CLAIM_CHARS: int = 400
MAX_TOOL_STEPS: int = 4
MAX_LLM_CALLS: int = 6
MAX_VERIFY_RETRIES: int = 2
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


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CLAIM_CHARS)
    doc_ids: list[str]
    skills: list[str]  # every skill the text names, as spelled in the documents
    facts: list[str]  # fact ids


class ToolTrace(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None
    fact_ids: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    llm_calls: int
    tool_calls: int
    verify_retries: int
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


class ChatResponse(BaseModel):  # only for a verified answer or an off-topic reply
    reply: str  # claim texts joined by a space, or the fixed off-topic text
    claims: list[Claim]
    facts: list[Fact]
    suggested_questions: list[str]  # at most 3, rules-generated
    off_topic: bool = False
    trace: list[ToolTrace]
    usage: Usage
    explainer: ExplainerInfo  # kind "agent"


@dataclass(frozen=True)
class AgentResult:
    status: Literal["ok", "failed"]
    claims: list[Claim]  # empty when failed or off_topic
    facts: list[Fact]
    trace: list[ToolTrace]
    usage: Usage
    model: str
    backend: str
    prompt_version: str
    failure_reason: FailureReason | None = None  # set exactly when status == "failed"
    off_topic: bool = False

    def as_record(self) -> dict:
        """JSON-serializable dict with exactly the keys: status, failure_reason, off_topic, claims, facts, trace,
        usage, model, backend, prompt_version (claims, facts, trace, usage via model_dump)."""
        return {
            "status": self.status,
            "failure_reason": self.failure_reason,
            "off_topic": self.off_topic,
            "claims": [claim.model_dump() for claim in self.claims],
            "facts": [fact.model_dump() for fact in self.facts],
            "trace": [step.model_dump() for step in self.trace],
            "usage": self.usage.model_dump(),
            "model": self.model,
            "backend": self.backend,
            "prompt_version": self.prompt_version,
        }
