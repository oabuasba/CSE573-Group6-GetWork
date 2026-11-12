"""Pydantic types of the web API (CC-18). Must not import FastAPI."""

from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

Side = Literal["resume", "posting"]
Direction = Literal["resume_to_posting", "posting_to_resume"]
SystemId = Literal["B1+", "B2", "Full"]  # CC-6 ids
Via = Literal["taxonomy", "noise", "llm", "default"]

DEFAULT_K: int = 10
MAX_K: int = 50
MAX_SKILLS: int = 100
MAX_SKILL_CHARS: int = 80
MAX_EXAMPLES: int = 20
DEFAULT_MAX_QUERY_CHARS: int = 20000

ERROR_CODES: dict[str, int] = {
    "invalid_request": 422,
    "empty_query": 400,
    "text_too_long": 400,
    "side_mismatch": 400,
    "not_found": 404,
    "embedding_mismatch": 409,
    "structuring_failed": 502,
    "embedding_failed": 502,
    "capability_unavailable": 503,
    "internal": 500,
}


class _Req(BaseModel):  # request models: unknown fields are rejected
    model_config = ConfigDict(extra="forbid")


class CorpusQuery(_Req):
    kind: Literal["corpus"]
    id: str = Field(min_length=1, max_length=64)


class TextQuery(_Req):
    kind: Literal["text"]
    text: str  # length checked by the engine (text_too_long), not here
    title: str | None = Field(default=None, max_length=200)  # postings only; ignored for resumes


class FormQuery(_Req):
    kind: Literal["form"]
    title: str = Field(default="", max_length=200)
    level: str | None = Field(default=None, max_length=40)  # postings only (CC-16 DE-3)
    years_experience: float | None = Field(default=None, ge=0, le=60)
    skills: list[str] = Field(default_factory=list, max_length=MAX_SKILLS)

    @field_validator("skills")
    @classmethod
    def _skills_not_too_long(cls, skills: list[str]) -> list[str]:
        if any(len(skill) > MAX_SKILL_CHARS for skill in skills):
            raise ValueError(f"each skill must be at most {MAX_SKILL_CHARS} characters")
        return skills


Query = Annotated[Union[CorpusQuery, TextQuery, FormQuery], Field(discriminator="kind")]


class MatchRequest(_Req):
    direction: Direction
    query: Query
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)
    system: SystemId = "Full"


class DocSkill(BaseModel):
    skill: str
    cluster: str  # a cluster name or "Other"
    via: Via


class DocumentView(BaseModel):
    id: str | None  # None for a document built from the query
    side: Side
    title: str | None
    level: str  # a CC-16 level or "Not specified"
    level_source: str | None  # "listed" | "years" | "missing" | None
    years_experience: float | None
    skills: list[DocSkill] = Field(max_length=200)
    incomplete_reason: str | None = None  # "no_skills" | "few_skills" | "low_coverage" | None


class SharedSkill(BaseModel):
    skill: str
    cluster: str


class RelatedGroup(BaseModel):
    cluster: str  # never "Other"
    resume_skills: list[str]
    posting_skills: list[str]


class GapSkill(BaseModel):
    skill: str
    cluster: str


class LevelComparison(BaseModel):
    resume: str
    posting: str
    relation: Literal["same", "adjacent", "different", "unknown"]


class Evidence(BaseModel):
    shared: list[SharedSkill]
    related: list[RelatedGroup]
    gaps: list[GapSkill]  # posting skills with no shared or related counterpart
    level: LevelComparison
    title_overlap: list[str]  # sorted common title tokens


class Match(BaseModel):
    rank: int  # 1-based
    id: str
    score: float
    document: DocumentView
    evidence: Evidence


class Note(BaseModel):
    code: Literal["no_skills", "sparse_query", "unknown_skills", "relabel_mode_differs"]
    message: str


class EmbeddingIdentity(BaseModel):
    model: str
    backend: str
    dim: int


class Provenance(BaseModel):
    alias_table_version: str | None
    taxonomy_version: str | None
    serialization_version: str | None
    relabel_mode: str | None
    embedding: EmbeddingIdentity | None
    demo: bool = False


class MatchResponse(BaseModel):
    direction: Direction
    system: SystemId
    query_kind: Literal["corpus", "text", "form"]
    k: int  # requested
    k_effective: int  # min(k, number of documents on the other side)
    query: DocumentView
    matches: list[Match]
    notes: list[Note]
    provenance: Provenance


class Capability(BaseModel):
    available: bool
    reason: str | None = None  # why not, naming the missing setting (never its value)
    detail: str | None = None  # e.g. "voyager, model-id" for the privacy notice


class EngineHealth(BaseModel):
    mode: Literal["demo", "artifacts"]
    artifacts_dir: str | None
    counts: dict[str, int]  # {"resumes": n, "postings": n}
    systems: list[SystemId]  # systems this server can rank with
    levels: list[str]  # LEVELS plus "Not specified"
    versions: Provenance
    capabilities: dict[str, Capability]  # keys "structure" and "embed"


class ExplainerInfo(BaseModel):
    kind: Literal["template", "agent"]
    model: str | None = None
    backend: str | None = None
    prompt_version: str | None = None

    # Keeps the old JSON shape when the new fields are None.
    @model_serializer(mode="wrap")
    def _omit_unset_agent_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        for key in ("backend", "prompt_version"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class ChatInfo(BaseModel):
    """Whether the per-match chat agent is on (explanations are always rules-based); reason says why not."""

    enabled: bool = False
    model: str | None = None
    backend: str | None = None
    prompt_version: str | None = None
    reason: str | None = None


class HealthResponse(EngineHealth):
    status: Literal["ok"] = "ok"
    explainer: ExplainerInfo
    chat: ChatInfo = Field(default_factory=ChatInfo)


class ExamplesResponse(BaseModel):
    items: list[DocumentView]


class ExplainRequest(_Req):
    direction: Direction
    query: DocumentView
    match_id: str = Field(min_length=1, max_length=64)


class Explanation(BaseModel):
    summary: str
    bullets: list[str]
    evidence: Evidence
    explainer: ExplainerInfo
    claims: list[dict] = Field(default_factory=list)  # Claim.model_dump() items; [] for template explanations
    facts: list[dict] = Field(default_factory=list)  # Fact.model_dump() items


class ErrorBody(BaseModel):
    code: str
    message: str
    hint: str | None = None
    field: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
