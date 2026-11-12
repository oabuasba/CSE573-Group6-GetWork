"""The agent behind the Explainer seam and the chat service; failures raise AgentFailure, never template text."""

from collections.abc import Sequence
from typing import Any

from getwork.agent import facts as fact_builders
from getwork.agent import graph as agent_graph
from getwork.agent.failure import AgentFailure
from getwork.agent.models import (
    QUERY_DOC,
    AgentResult,
    ChatMatchRef,
    ChatRequest,
    ChatResponse,
)
from getwork.agent.tools import TurnContext
from getwork.agent.turn import AgentRunner
from getwork.webapp.engine import Engine, EngineError
from getwork.webapp.explain import build_evidence
from getwork.webapp.models import (
    Direction,
    DocumentView,
    Evidence,
    Explanation,
    ExplainerInfo,
    SystemId,
)

EXPLAIN_QUESTION: str = "Explain why this match is a good or weak fit, using only the facts."
OFF_TOPIC_REPLY: str = "I can only answer questions about this match and the skills in these documents."
MAX_SUGGESTIONS = 3


def _agent_info(runner: AgentRunner) -> ExplainerInfo:
    cfg = runner.cfg
    return ExplainerInfo(
        kind="agent", model=cfg.model, backend=cfg.backend, prompt_version=cfg.prompt_version
    )


def _seeded_book(
    direction: Direction, query: DocumentView, match: DocumentView, evidence: Evidence
) -> fact_builders.FactBook:
    book = fact_builders.FactBook()
    fact_builders.evidence_facts(book, direction, query, match, evidence)
    return book


class AgentExplainer:
    def __init__(self, graph: Any, runner: AgentRunner, engine: Engine, *, system: SystemId = "Full") -> None:
        self._graph = graph
        self._runner = runner
        self._engine = engine
        self._system = system

    @property
    def info(self) -> ExplainerInfo:
        return _agent_info(self._runner)

    def explain_detailed(self, direction: Direction, query: DocumentView, match: DocumentView) -> AgentResult:
        """Never raises for agent failures. Seeds facts with evidence_facts(build_evidence(...)), scope {QUERY_DOC:
        query, match ref: match}, no refs, runs run_agent with max_tool_steps=0 (MA-9)."""
        match_ref = fact_builders.doc_ref(match)
        evidence = build_evidence(direction, query, match)
        ctx = TurnContext(
            engine=self._engine,
            direction=direction,
            system=self._system,
            query=query,
            focus_id=match_ref,
            refs={},
            docs={QUERY_DOC: query, match_ref: match},
            book=_seeded_book(direction, query, match, evidence),
        )
        return agent_graph.run_agent(
            self._graph, self._runner, ctx=ctx, question=EXPLAIN_QUESTION, history=[], max_tool_steps=0
        )

    def explain(self, direction: Direction, query: DocumentView, match: DocumentView) -> Explanation:
        """status ok -> summary = first claim text, bullets = the other claim texts, evidence = build_evidence(...),
        claims and facts as model_dump dicts, explainer = self.info. status failed -> raise AgentFailure(reason).
        An off_topic result (not expected for the fixed question) raises AgentFailure("invalid_reply")."""
        result = self.explain_detailed(direction, query, match)
        if result.status == "failed" and result.failure_reason is not None:
            raise AgentFailure(result.failure_reason, result.failure_detail)
        if result.off_topic or not result.claims:
            raise AgentFailure("invalid_reply")
        texts = [claim.text for claim in result.claims]
        return Explanation(
            summary=texts[0],
            bullets=texts[1:],
            evidence=build_evidence(direction, query, match),
            explainer=self.info,
            claims=[claim.model_dump() for claim in result.claims],
            facts=[fact.model_dump() for fact in result.facts],
        )


class ChatService:
    def __init__(self, graph: Any, runner: AgentRunner, engine: Engine) -> None:
        self._graph = graph
        self._runner = runner
        self._engine = engine

    @property
    def info(self) -> ExplainerInfo:
        return _agent_info(self._runner)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """EngineError not_found for an unknown focus_id or result id (Engine.get_document), side_mismatch if the
        focus document's side equals the query's side. Builds the context (refs from request.matches, docs
        {QUERY_DOC, focus}), seeds facts, runs the graph with request.message and history. status ok: reply = claim
        texts joined by a space, or OFF_TOPIC_REPLY when off_topic; suggested_questions per suggest(). status
        failed: raise AgentFailure(reason)."""
        focus = self._load_focus(request)
        evidence = build_evidence(request.direction, request.query, focus)
        ctx = TurnContext(
            engine=self._engine,
            direction=request.direction,
            system=request.system,
            query=request.query,
            focus_id=request.focus_id,
            refs={ref.id: ref for ref in request.matches},
            docs={QUERY_DOC: request.query, request.focus_id: focus},
            book=_seeded_book(request.direction, request.query, focus, evidence),
        )
        result = agent_graph.run_agent(
            self._graph, self._runner, ctx=ctx, question=request.message, history=request.history
        )
        if result.status == "failed" and result.failure_reason is not None:
            raise AgentFailure(result.failure_reason, result.failure_detail)
        return ChatResponse(
            reply=OFF_TOPIC_REPLY if result.off_topic else " ".join(c.text for c in result.claims),
            claims=result.claims,
            facts=result.facts,
            suggested_questions=suggest(request.direction, request.matches, request.focus_id, evidence),
            off_topic=result.off_topic,
            trace=result.trace,
            usage=result.usage,
            explainer=self.info,
        )

    def _load_focus(self, request: ChatRequest) -> DocumentView:
        focus = self._engine.get_document(request.focus_id)
        for ref in request.matches:
            self._engine.get_document(ref.id)
        if focus.side == request.query.side:
            raise EngineError(
                "side_mismatch",
                f"Document {request.focus_id!r} is on the same side as the query.",
                field="focus_id",
            )
        return focus


def suggest(direction: Direction, refs: Sequence[ChatMatchRef], focus_id: str, evidence: Evidence) -> list[str]:
    """At most 3, in order, each only when applicable: "Which skills am I missing?" (resume_to_posting) or "Which
    required skills does this candidate lack?" (posting_to_resume) when evidence.gaps is non-empty; f"Why is {a}
    ranked above {b}?" for the focus and the next lower-ranked result when both are in refs; "Which skills count
    as related here?" when evidence.related is non-empty."""
    questions: list[str] = []
    if evidence.gaps:
        questions.append(
            "Which skills am I missing?"
            if direction == "resume_to_posting"
            else "Which required skills does this candidate lack?"
        )
    ranked = sorted(refs, key=lambda ref: ref.rank)
    focus = next((ref for ref in ranked if ref.id == focus_id), None)
    lower = next((ref for ref in ranked if focus is not None and ref.rank > focus.rank), None)
    if focus is not None and lower is not None:
        questions.append(f"Why is {focus.id} ranked above {lower.id}?")
    if evidence.related:
        questions.append("Which skills count as related here?")
    return questions[:MAX_SUGGESTIONS]
