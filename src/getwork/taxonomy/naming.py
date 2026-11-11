"""Name skill clusters: an offline heuristic and an optional LLM question per cluster."""

from __future__ import annotations

import functools
import re
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from getwork.alias import normalize
from getwork.llm.client import LLMClient, LLMRequest, PermanentLLMError
from getwork.llm.retry import CallStats, RetriesExhausted, RetryConfig, Throttle
from getwork.llm.validated import ValidationExhausted, complete_validated, parse_json_object
from getwork.taxonomy import model

NAME_PROMPT_VERSION: str = "name_v2"
MAX_NAME_WORDS: int = 5
GENERIC_NAMES: frozenset[str] = frozenset(
    {
        "other",
        "others",
        "misc",
        "miscellaneous",
        "general",
        "various",
        "mixed",
        "tools",
        "technologies",
        "technology",
        "skills",
        "software",
        "technical skills",
        "it",
        "unknown",
    }
)
_CLUSTER_PLACEHOLDER = re.compile(r"cluster\s*\d*", re.IGNORECASE)
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


def build_name_message(members: Sequence[str], used: Sequence[str] = ()) -> str:
    text = "MEMBERS:\n" + "\n".join("- " + m for m in members)
    if used:
        text += "\n\nUSED NAMES:\n" + "\n".join("- " + u for u in used)
    return text


def check_concept_name(
    name: str, members: Sequence[str] = (), used: Collection[str] = ()
) -> str:
    """Rule N-2: model.check_name plus the concept-label rules; ValueError names the broken rule."""
    name = model.check_name(name)
    if name.casefold() in GENERIC_NAMES or _CLUSTER_PLACEHOLDER.fullmatch(name):
        raise ValueError(f"{name!r} is too vague; name the concept the members share")
    if len(name.split()) > MAX_NAME_WORDS:
        raise ValueError(f"{name!r} has more than {MAX_NAME_WORDS} words; use one to four words")
    reason = model.member_list_reason(name, members)
    if reason is not None:
        raise ValueError(reason)
    if normalize.normalize_key(name) in {normalize.normalize_key(m) for m in members}:
        raise ValueError(
            f"{name!r} is one of the member skills; give the broader concept they share"
        )
    if name.casefold() in {u.casefold() for u in used}:
        raise ValueError(f"{name!r} is already the name of another cluster; make it more specific")
    return name


def parse_name(text: str, members: Sequence[str] = (), used: Collection[str] = ()) -> str:
    value = parse_json_object(text).get("name")
    if not isinstance(value, str):
        raise ValueError('"name" must be a string')
    return check_concept_name(value, members, used)


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


def _request(cfg: NameConfig, shown: Sequence[str], used: Sequence[str]) -> LLMRequest:
    return LLMRequest(
        system=load_name_prompt(),
        user=build_name_message(shown, used),
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
    accepted: list[str] = []
    for cluster_id in sorted(members):
        shown = tuple(members[cluster_id][: cfg.top_members])
        used = tuple(accepted)
        try:
            name = complete_validated(
                client,
                _request(cfg, shown, used),
                lambda text, shown=shown, used=used: parse_name(text, shown, used),
                cfg=cfg.retry,
                throttle=throttle,
                sleep=sleep,
                stats=result.stats,
                max_attempts=cfg.max_validation_attempts,
            )
            result.names[cluster_id] = name
            accepted.append(name)
        except (ValidationExhausted, RetriesExhausted, PermanentLLMError):
            result.failed.append(cluster_id)
    return result
