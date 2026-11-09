"""Name skill clusters: an offline heuristic and an optional LLM question per cluster."""

from __future__ import annotations

import functools
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from getwork.llm.client import LLMClient, LLMRequest, PermanentLLMError
from getwork.llm.retry import CallStats, RetriesExhausted, RetryConfig, Throttle
from getwork.llm.validated import ValidationExhausted, complete_validated, parse_json_object
from getwork.taxonomy import model

NAME_PROMPT_VERSION: str = "name_v1"
HEURISTIC_MEMBERS = 3
HEURISTIC_SEPARATOR = " / "
_WHITESPACE = re.compile(r"\s+")


@functools.cache
def load_name_prompt(version: str = NAME_PROMPT_VERSION) -> str:
    path = Path(__file__).parent / "prompts" / f"{version}.md"
    return path.read_text(encoding="utf-8")


def heuristic_name(members: Sequence[str]) -> str:
    if not members:
        raise ValueError("cannot name a cluster with no members")
    text = HEURISTIC_SEPARATOR.join(members[:HEURISTIC_MEMBERS])
    for ch in model.FORBIDDEN_NAME_CHARS:
        text = text.replace(ch, " ")
    text = _WHITESPACE.sub(" ", text).strip()
    return model.check_name(text[: model.MAX_NAME_CHARS].strip())


def build_name_message(members: Sequence[str]) -> str:
    return "MEMBERS:\n" + "\n".join("- " + m for m in members)


def parse_name(text: str) -> str:
    value = parse_json_object(text).get("name")
    if not isinstance(value, str):
        raise ValueError('"name" must be a string')
    return model.check_name(value)


@dataclass(frozen=True)
class NameConfig:
    model: str
    backend: str = "voyager"
    top_members: int = 25
    max_tokens: int = 256
    max_validation_attempts: int = 3
    extra_body: Mapping[str, object] | None = None
    retry: RetryConfig = RetryConfig()


@dataclass
class NameResult:
    names: dict[int, str]
    failed: list[int]
    stats: CallStats = field(default_factory=CallStats)


def _request(cfg: NameConfig, members: Sequence[str]) -> LLMRequest:
    return LLMRequest(
        system=load_name_prompt(),
        user=build_name_message(members[: cfg.top_members]),
        model=cfg.model,
        temperature=0,
        max_tokens=cfg.max_tokens,
        reasoning_effort=None,
        response_schema=None,
        extra_body=cfg.extra_body,
    )


def name_clusters(
    client: LLMClient,
    cfg: NameConfig,
    members: Mapping[int, Sequence[str]],
    *,
    secrets: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> NameResult:
    # secrets is accepted for a uniform step signature; results hold only cluster text.
    result = NameResult(names={}, failed=[])
    throttle = Throttle()
    for cluster_id in sorted(members):
        try:
            result.names[cluster_id] = complete_validated(
                client,
                _request(cfg, members[cluster_id]),
                parse_name,
                cfg=cfg.retry,
                throttle=throttle,
                sleep=sleep,
                stats=result.stats,
                max_attempts=cfg.max_validation_attempts,
            )
        except (ValidationExhausted, RetriesExhausted, PermanentLLMError):
            result.failed.append(cluster_id)
    return result
