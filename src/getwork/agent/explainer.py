"""The chat service over the agent graph; failures raise AgentFailure, never template text."""

import logging
from typing import Any

from getwork.agent import facts as fact_builders
from getwork.agent import graph as agent_graph
from getwork.agent.failure import AgentFailure, redact_detail
from getwork.agent.models import QUERY_DOC, ChatRequest, ChatResponse
from getwork.agent.tools import TurnContext
from getwork.agent.turn import AgentRunner
from getwork.webapp.engine import Engine, EngineError
from getwork.webapp.explain import build_evidence
from getwork.webapp.models import Direction, DocumentView, Evidence, ExplainerInfo

logger = logging.getLogger("getwork.agent")
UNGROUNDED_LOG = "agent reply names skills outside these documents: %s"


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


class ChatService:
    def __init__(self, graph: Any, runner: AgentRunner, engine: Engine) -> None:
        self._graph = graph
        self._runner = runner
        self._engine = engine

    @property
    def info(self) -> ExplainerInfo:
        return _agent_info(self._runner)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """EngineError not_found for an unknown focus_id or result id, side_mismatch if the focus document's side
        equals the query's side; AgentFailure when the turn failed. Skills named outside the two documents are
        logged (never blocking, never the message or reply text)."""
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
        if result.ungrounded_skills:
            joined = redact_detail(", ".join(result.ungrounded_skills), self._runner.secrets)
            logger.info(UNGROUNDED_LOG, joined)
        return ChatResponse(reply=result.reply, trace=result.trace, usage=result.usage, explainer=self.info)

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
