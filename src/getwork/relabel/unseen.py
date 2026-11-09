"""Assign skills the taxonomy does not know to an existing cluster or Other with an LLM."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from getwork.alias.normalize import normalize_key
from getwork.llm.client import FatalLLMError, LLMClient, LLMRequest, PermanentLLMError
from getwork.llm.retry import CallStats, RetriesExhausted, RetryConfig, Throttle
from getwork.llm.validated import ValidationExhausted, complete_validated, parse_json_object
from getwork.relabel.cache import StampedCache
from getwork.taxonomy.model import OTHER, Taxonomy

UNSEEN_PROMPT_VERSION: str = "unseen_v1"
EXAMPLES_PER_CLUSTER: int = 5
MAX_SKILL_CHARS_IN_ERROR = 60


@functools.cache
def load_unseen_prompt(version: str = UNSEEN_PROMPT_VERSION) -> str:
    path = Path(__file__).parent / "prompts" / f"{version}.md"
    return path.read_text(encoding="utf-8")


def build_unseen_message(batch: Sequence[str], tax: Taxonomy) -> str:
    lines = [
        f"- {c.name}: {', '.join(c.skills[:EXAMPLES_PER_CLUSTER])}" for c in tax.clusters
    ]
    return (
        "CLUSTERS:\n" + "\n".join(lines) + "\n\nSKILLS:\n" + "\n".join("- " + s for s in batch)
    )


def _short(text: str) -> str:
    if len(text) <= MAX_SKILL_CHARS_IN_ERROR:
        return repr(text)
    return repr(text[:MAX_SKILL_CHARS_IN_ERROR] + "...")


def _parse_item(item: object) -> tuple[str, str]:
    if not isinstance(item, dict):
        raise ValueError("each assignment must be an object")
    skill, cluster = item.get("skill"), item.get("cluster")
    if not isinstance(skill, str):
        raise ValueError("assignment skill must be a string")
    if not isinstance(cluster, str):
        raise ValueError(f"cluster of {_short(skill)} must be a string")
    return skill, cluster


def parse_assignments(
    text: str, batch: Sequence[str], valid_names: Collection[str]
) -> dict[str, str]:
    data = parse_json_object(text)
    items = data.get("assignments")
    if not isinstance(items, list):
        raise ValueError('"assignments" must be a list')
    allowed = set(valid_names) | {OTHER}
    wanted = set(batch)
    result: dict[str, str] = {}
    for item in items:
        skill, cluster = _parse_item(item)
        if skill not in wanted:
            raise ValueError(f"skill {_short(skill)} is not in SKILLS")
        if skill in result:
            raise ValueError(f"skill {_short(skill)} is answered more than once")
        if cluster not in allowed:
            raise ValueError(f"cluster {_short(cluster)} for {_short(skill)} is not allowed")
        result[skill] = cluster
    missing = [s for s in batch if s not in result]
    if missing:
        raise ValueError(f"{len(missing)} skills are missing, first: {_short(missing[0])}")
    return result


@dataclass(frozen=True)
class UnseenConfig:
    model: str
    backend: str = "voyager"
    batch_size: int = 50
    max_tokens: int = 2048
    max_validation_attempts: int = 3
    extra_body: Mapping[str, object] | None = None
    retry: RetryConfig = RetryConfig()


@dataclass
class UnseenResult:
    assignments: dict[str, str]  # skill (as given) -> cluster, for every skill that was resolved
    failed: list[str]  # skills of failed batches
    stats: CallStats


def _request(cfg: UnseenConfig, batch: Sequence[str], tax: Taxonomy) -> LLMRequest:
    return LLMRequest(
        system=load_unseen_prompt(),
        user=build_unseen_message(batch, tax),
        model=cfg.model,
        temperature=0,
        max_tokens=cfg.max_tokens,
        reasoning_effort=None,
        response_schema=None,
        extra_body=cfg.extra_body,
    )


def _distinct_by_key(skills: Sequence[str]) -> dict[str, str]:
    """key -> first spelling, sorted by key."""
    first: dict[str, str] = {}
    for skill in skills:
        first.setdefault(normalize_key(skill), skill)
    return dict(sorted(first.items()))


def assign_unseen(
    client: LLMClient,
    cfg: UnseenConfig,
    skills: Sequence[str],
    tax: Taxonomy,
    cache: StampedCache,
    *,
    secrets: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> UnseenResult:
    # secrets is accepted for a uniform step signature; cache and results hold only skill text.
    distinct = _distinct_by_key(skills)
    answers = {k: v for k in distinct if isinstance(v := cache.get(k), str)}
    pending = [k for k in distinct if k not in answers]
    result = UnseenResult(assignments={}, failed=[], stats=CallStats())
    throttle = Throttle()
    names = tax.cluster_names()
    for start in range(0, len(pending), cfg.batch_size):
        keys = pending[start : start + cfg.batch_size]
        batch = [distinct[k] for k in keys]
        try:
            parsed = complete_validated(
                client,
                _request(cfg, batch, tax),
                lambda t, b=batch: parse_assignments(t, b, names),
                cfg=cfg.retry,
                throttle=throttle,
                sleep=sleep,
                stats=result.stats,
                max_attempts=cfg.max_validation_attempts,
            )
        except (ValidationExhausted, RetriesExhausted, PermanentLLMError):
            result.failed.extend(batch)
            continue
        except FatalLLMError:
            cache.save()
            raise
        for key, spelling in zip(keys, batch):
            answers[key] = parsed[spelling]
            cache.put(key, parsed[spelling])
        cache.save()
    for skill in skills:
        if normalize_key(skill) in answers:
            result.assignments[skill] = answers[normalize_key(skill)]
    return result
