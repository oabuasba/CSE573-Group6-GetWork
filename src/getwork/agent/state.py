"""Configuration and mutable state of one agent turn (no LangGraph import)."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from getwork.agent.models import (
    DEFAULT_TIMEOUT_S,
    MAX_LLM_CALLS,
    MAX_TOOL_STEPS,
    MAX_VERIFY_RETRIES,
    PROMPT_VERSION,
    AgentResult,
    ChatMessage,
    FailureReason,
    ToolTrace,
)
from getwork.agent.prompt import Action
from getwork.agent.tools import TurnContext
from getwork.llm.retry import CallStats, RetryConfig


@dataclass(frozen=True)
class AgentConfig:
    model: str
    backend: str
    prompt_version: str = PROMPT_VERSION
    temperature: float | None = 0.0
    max_tokens: int = 1024
    reasoning_effort: str | None = "none"
    extra_body: Mapping[str, object] | None = None
    max_tool_steps: int = MAX_TOOL_STEPS
    max_llm_calls: int = MAX_LLM_CALLS
    max_verify_retries: int = MAX_VERIFY_RETRIES
    timeout_s: float = DEFAULT_TIMEOUT_S
    retry: RetryConfig = RetryConfig()

    def __post_init__(self) -> None:
        """ValueError for an empty model, max_tool_steps < 0, max_llm_calls < 1, max_verify_retries < 0,
        timeout_s <= 0."""
        problems = {
            "model": not self.model.strip(),
            "max_tool_steps": self.max_tool_steps < 0,
            "max_llm_calls": self.max_llm_calls < 1,
            "max_verify_retries": self.max_verify_retries < 0,
            "timeout_s": self.timeout_s <= 0,
        }
        for name, is_bad in problems.items():
            if is_bad:
                raise ValueError(f"invalid agent config: {name} = {getattr(self, name)!r}")


@dataclass
class Turn:
    ctx: TurnContext
    question: str
    history: list[ChatMessage]
    max_tool_steps: int  # from the config unless new_turn was given another value
    started: float
    tool_notes: list[str] = field(default_factory=list)
    trace: list[ToolTrace] = field(default_factory=list)
    stats: CallStats = field(default_factory=CallStats)
    tool_steps: int = 0
    verify_retries: int = 0
    feedback: str | None = None  # text for the next prompt (tool-limit or verifier violations)
    pending: Action | None = None  # the tool request decide parsed
    answer: Action | None = None  # the answer decide parsed
    route: str = "decide"  # "decide" | "run_tool" | "verify" | "finish" | "fail" | "end"
    failure_reason: FailureReason | None = None  # set by a node before it routes to "fail"
    result: AgentResult | None = None
