"""Name skill clusters: an offline heuristic and an optional LLM question per cluster."""

from __future__ import annotations

import functools
import json
import re
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Generic, TypeVar

from getwork.alias import normalize
from getwork.llm.client import LLMClient, LLMRequest, PermanentLLMError
from getwork.llm.retry import CallStats, RetriesExhausted, RetryConfig, Throttle
from getwork.llm.validated import ValidationExhausted, complete_validated, parse_json_object
from getwork.structure import pool
from getwork.taxonomy import model

T = TypeVar("T")
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
FALLBACK_REASONS: dict[type[Exception], str] = {
    ValidationExhausted: "replies rejected",
    RetriesExhausted: "retries exhausted",
    PermanentLLMError: "request refused",
}
_CLUSTER_FAILURES = tuple(FALLBACK_REASONS)
MAX_REASON_CHARS = 200


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
    max_tokens: int = 1024
    reasoning_effort: str | None = "none"
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
        reasoning_effort=cfg.reasoning_effort,
        response_schema=None,
        extra_body=cfg.extra_body,
    )


def named_message(done: int, total: int, cluster_id: int, name: str, attempts: int) -> str:
    return f"named {done}/{total}: cluster {cluster_id} -> {json.dumps(name)} (attempts {attempts})"


def fallback_message(done: int, total: int, cluster_id: int, reason: str, attempts: int) -> str:
    return f"fallback {done}/{total}: cluster {cluster_id} ({reason}, attempts {attempts})"


def rejected_message(cluster_id: int, attempt: int, reason: str) -> str:
    return f"rejected cluster {cluster_id} attempt {attempt}: {reason}"


def short_reason(error: ValueError) -> str:
    return " ".join(str(error).split())[:MAX_REASON_CHARS]


@dataclass(frozen=True)
class _Outcome:
    """One cluster's answer: a name, or the fixed fallback reason; stats cover only this cluster."""

    name: str | None
    reason: str
    stats: CallStats
    rejections: tuple[str, ...] = ()


@dataclass(frozen=True)
class Env:
    """What every model call needs; the Throttle is the only shared mutable part."""

    client: LLMClient
    cfg: NameConfig
    throttle: Throttle
    sleep: Callable[[float], None]


def add_stats(first: CallStats, second: CallStats) -> CallStats:
    return CallStats(
        calls=first.calls + second.calls,
        retries=first.retries + second.retries,
        rate_limited=first.rate_limited + second.rate_limited,
    )


def _fallback_reason(exc: Exception) -> str:
    return next(reason for kind, reason in FALLBACK_REASONS.items() if isinstance(exc, kind))


@dataclass(frozen=True)
class Asked(Generic[T]):
    """One validated request: the parsed value, or None with a fixed FALLBACK_REASONS text."""

    value: T | None
    reason: str
    stats: CallStats
    rejections: tuple[str, ...] = ()


def ask_validated(env: Env, request: LLMRequest, parse: Callable[[str], T]) -> Asked[T]:
    """Send request and validate with parse, recording a short reason for every rejected reply."""
    stats = CallStats()
    rejections: list[str] = []

    def recording_parse(text: str) -> T:
        # parse_json_object messages are fixed text; check_concept_name echoes only the short name
        try:
            return parse(text)
        except ValueError as exc:
            rejections.append(short_reason(exc))
            raise

    try:
        value = complete_validated(
            env.client,
            request,
            recording_parse,
            cfg=env.cfg.retry,
            throttle=env.throttle,
            sleep=env.sleep,
            stats=stats,
            max_attempts=env.cfg.max_validation_attempts,
        )
    except _CLUSTER_FAILURES as exc:
        return Asked(None, _fallback_reason(exc), stats, tuple(rejections))
    return Asked(value, "", stats, tuple(rejections))


def _name_one(env: Env, shown: Sequence[str], used: Sequence[str]) -> _Outcome:
    asked = ask_validated(
        env, _request(env.cfg, shown, used), lambda text: parse_name(text, shown, used)
    )
    return _Outcome(
        name=asked.value, reason=asked.reason, stats=asked.stats, rejections=asked.rejections
    )


class _Tally:
    """Collects final outcomes in id order: result, accepted names and progress lines."""

    def __init__(self, total: int, log: Callable[[str], None] | None) -> None:
        self.result = NameResult(names={}, failed=[])
        self.accepted: list[str] = []
        self._total = total
        self._log = log

    def finish(self, cluster_id: int, outcome: _Outcome) -> None:
        self.result.stats = add_stats(self.result.stats, outcome.stats)
        done = len(self.result.names) + len(self.result.failed) + 1
        attempts = outcome.stats.calls
        if self._log is not None:
            for number, reason in enumerate(outcome.rejections, start=1):
                self._log(rejected_message(cluster_id, number, reason))
        if outcome.name is None:
            self.result.failed.append(cluster_id)
            line = fallback_message(done, self._total, cluster_id, outcome.reason, attempts)
        else:
            self.result.names[cluster_id] = outcome.name
            self.accepted.append(outcome.name)
            line = named_message(done, self._total, cluster_id, outcome.name, attempts)
        if self._log is not None:
            self._log(line)


def _shown(
    cfg: NameConfig, members: Mapping[int, Sequence[str]], cluster_id: int
) -> tuple[str, ...]:
    return tuple(members[cluster_id][: cfg.top_members])


def _resolve(env: Env, shown: Sequence[str], draft: _Outcome, tally: _Tally) -> _Outcome:
    taken = {name.casefold() for name in tally.accepted}
    if draft.name is None or draft.name.casefold() not in taken:
        return draft
    duplicate = f"{draft.name!r} is already the name of another cluster; make it more specific"
    again = _name_one(env, shown, tuple(tally.accepted))
    return replace(
        again,
        stats=add_stats(draft.stats, again.stats),
        rejections=(*draft.rejections, short_reason(ValueError(duplicate)), *again.rejections),
    )


def _name_in_parallel(
    env: Env, members: Mapping[int, Sequence[str]], workers: int, tally: _Tally
) -> None:
    def draft_one(cluster_id: int) -> _Outcome:
        return _name_one(env, _shown(env.cfg, members, cluster_id), ())

    ids = sorted(members)
    drafts: dict[int, _Outcome] = {}
    next_index = 0
    stop, abort = threading.Event(), threading.Event()
    try:
        for cluster_id, outcome in pool.run_pool(ids, draft_one, workers, stop, abort):
            if isinstance(outcome, BaseException):
                raise outcome
            drafts[cluster_id] = outcome
            while next_index < len(ids) and ids[next_index] in drafts:
                current = ids[next_index]
                shown = _shown(env.cfg, members, current)
                tally.finish(current, _resolve(env, shown, drafts.pop(current), tally))
                next_index += 1
    finally:
        abort.set()


def name_clusters(
    client: LLMClient,
    cfg: NameConfig,
    members: Mapping[int, Sequence[str]],
    *,
    secrets: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
    workers: int = 1,
    log: Callable[[str], None] | None = None,
) -> NameResult:
    """workers == 1 asks ids ascending with the used names; 2 or more drafts concurrently, then
    re-asks repeated names in id order. log gets one line per cluster, in id order. FatalLLMError
    propagates."""
    # secrets is accepted for a uniform step signature; results hold only cluster text.
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    tally = _Tally(total=len(members), log=log)
    env = Env(client=client, cfg=cfg, throttle=Throttle(), sleep=sleep)
    if workers >= 2:
        _name_in_parallel(env, members, workers, tally)
        return tally.result
    for cluster_id in sorted(members):
        shown = _shown(cfg, members, cluster_id)
        tally.finish(cluster_id, _name_one(env, shown, tuple(tally.accepted)))
    return tally.result
