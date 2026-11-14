"""Node logic of one agent turn as plain methods; graph.py only wires them (no LangGraph import)."""

import time
from collections.abc import Callable, Sequence
from typing import Literal

from getwork.agent import grounding, prompt, tools
from getwork.agent.failure import redact_detail
from getwork.agent.grounding import Vocabulary
from getwork.agent.models import AgentResult, ChatMessage, FailureReason, ToolTrace, Usage
from getwork.agent.state import AgentConfig, Turn
from getwork.llm.client import FatalLLMError, LLMClient, LLMError, LLMRequest, PermanentLLMError
from getwork.llm.retry import RetriesExhausted, Throttle
from getwork.llm.validated import ValidationExhausted, complete_validated

MAX_PARSE_ATTEMPTS = 3
REPLY_EXCERPT_CHARS = 200
NO_MORE_TOOLS = "No more tool calls are allowed; answer now."


class AgentRunner:
    def __init__(
        self,
        client: LLMClient,
        cfg: AgentConfig,
        vocab: Vocabulary,
        *,
        secrets: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.vocab = vocab
        self.secrets = tuple(secrets)
        self._clock = clock
        self._sleep = sleep

    def new_turn(
        self,
        *,
        ctx: tools.TurnContext,
        question: str,
        history: Sequence[ChatMessage],
        max_tool_steps: int | None = None,
    ) -> Turn:
        """Facts for the turn are already in ctx.book. started = clock(); max_tool_steps defaults to cfg's."""
        limit = self.cfg.max_tool_steps if max_tool_steps is None else max_tool_steps
        return Turn(
            ctx=ctx, question=question, history=list(history), max_tool_steps=limit, started=self._clock()
        )

    def decide(self, turn: Turn) -> Turn:
        if self._clock() - turn.started > self.cfg.timeout_s:
            return self._route_to_fail(turn=turn, reason="timeout")
        if turn.stats.calls >= self.cfg.max_llm_calls:
            return self._route_to_fail(turn=turn, reason="budget_exhausted")
        request = self._build_request(turn)
        turn.feedback = None
        replies: list[str] = []

        def parse(text: str) -> prompt.Action:
            replies.append(text)
            return prompt.parse_action(text)

        try:
            action = complete_validated(
                self.client,
                request,
                parse,
                cfg=self.cfg.retry,
                throttle=Throttle(clock=self._clock, sleep=self._sleep),
                sleep=self._sleep,
                stats=turn.stats,
                max_attempts=min(MAX_PARSE_ATTEMPTS, self.cfg.max_llm_calls - turn.stats.calls),
            )
        except (FatalLLMError, PermanentLLMError) as e:
            return self._route_to_fail(turn=turn, reason="model_rejected", detail=_describe_llm_error(e))
        except ValidationExhausted as e:
            excerpt = replies[-1][:REPLY_EXCERPT_CHARS] if replies else ""
            detail = f"{e.last_error}; last reply starts: {excerpt!r}"
            return self._route_to_fail(turn=turn, reason="invalid_reply", detail=detail)
        except RetriesExhausted as e:
            detail = f"{e.kind} retries exhausted after {e.attempts} calls: {_describe_llm_error(e.last_error)}"
            return self._route_to_fail(turn=turn, reason="llm_unavailable", detail=detail)
        except LLMError as e:
            return self._route_to_fail(turn=turn, reason="llm_unavailable", detail=_describe_llm_error(e))
        return self._route_action(turn=turn, action=action)

    def run_tool(self, turn: Turn) -> Turn:
        action = turn.pending
        if action is None or action.tool is None:
            raise ValueError("run_tool needs a pending tool action")
        args = dict(action.args or {})
        turn.tool_steps += 1
        try:
            created = tools.run_tool(ctx=turn.ctx, name=action.tool, args=args)
            step = ToolTrace(tool=action.tool, args=args, ok=True, fact_ids=[f.id for f in created])
        except tools.ToolError as e:
            step = ToolTrace(tool=action.tool, args=args, ok=False, error=str(e))
            turn.tool_notes.append(f"{action.tool}: {e}")
        turn.trace.append(step)
        turn.pending = None
        turn.route = "decide"
        return turn

    def finish(self, turn: Turn) -> Turn:
        answer = turn.answer
        if answer is None:
            raise ValueError("finish needs an answer")
        reply = answer.text or ""
        ungrounded = grounding.ungrounded_skills(reply, list(turn.ctx.docs.values()), self.vocab)
        turn.result = self._build_result(turn, status="ok", reply=reply, ungrounded=ungrounded)
        turn.route = "end"
        return turn

    def fail(self, turn: Turn) -> Turn:
        if turn.failure_reason is None:
            raise ValueError("fail needs turn.failure_reason")
        turn.result = self._build_result(
            turn, status="failed", reason=turn.failure_reason, detail=turn.failure_detail
        )
        turn.route = "end"
        return turn

    def _build_request(self, turn: Turn) -> LLMRequest:
        ctx = turn.ctx
        user = prompt.build_user_message(
            question=turn.question,
            history=turn.history,
            facts=list(ctx.book.facts.values()),
            refs=ctx.refs,
            docs=ctx.docs,
            tool_notes=turn.tool_notes,
            feedback=turn.feedback,
        )
        return LLMRequest(
            system=prompt.load_system_prompt(self.cfg.prompt_version),
            user=user,
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            reasoning_effort=self.cfg.reasoning_effort,
            response_schema=None,
            extra_body=self.cfg.extra_body,
        )

    def _route_action(self, *, turn: Turn, action: prompt.Action) -> Turn:
        if action.kind == "answer":
            turn.answer = action
            turn.route = "finish"
        elif turn.tool_steps >= turn.max_tool_steps:
            turn.feedback = NO_MORE_TOOLS
            turn.route = "decide"
        else:
            turn.pending = action
            turn.route = "run_tool"
        return turn

    def _route_to_fail(self, *, turn: Turn, reason: FailureReason, detail: str | None = None) -> Turn:
        turn.failure_reason = reason
        turn.failure_detail = redact_detail(detail, self.secrets) if detail else None
        turn.route = "fail"
        return turn

    def _build_result(
        self,
        turn: Turn,
        *,
        status: Literal["ok", "failed"],
        reply: str = "",
        reason: FailureReason | None = None,
        detail: str | None = None,
        ungrounded: list[str] | None = None,
    ) -> AgentResult:
        usage = Usage(
            llm_calls=turn.stats.calls,
            tool_calls=len(turn.trace),
            elapsed_ms=int((self._clock() - turn.started) * 1000),
        )
        return AgentResult(
            status=status,
            reply=reply,
            facts=list(turn.ctx.book.facts.values()),
            trace=list(turn.trace),
            usage=usage,
            model=self.cfg.model,
            backend=self.cfg.backend,
            prompt_version=self.cfg.prompt_version,
            failure_reason=reason,
            failure_detail=detail,
            ungrounded_skills=list(ungrounded or []),
        )


def drive(runner: AgentRunner, turn: Turn) -> AgentResult:
    """Plain driver, the oracle for the graph: while turn.route != "end", call the node named by turn.route
    (decide, run_tool, finish, fail) and return turn.result (never None)."""
    while turn.route != "end":
        getattr(runner, turn.route)(turn)
    if turn.result is None:
        raise RuntimeError("the turn ended without a result")
    return turn.result


def _describe_llm_error(error: LLMError) -> str:
    return f"{type(error).__name__} status={error.status}: {error}"
