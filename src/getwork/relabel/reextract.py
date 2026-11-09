"""Re-read a document's text with an LLM to add skills the first extraction missed."""

from __future__ import annotations

import functools
import hashlib
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from getwork.llm.client import FatalLLMError, LLMClient, LLMRequest, PermanentLLMError
from getwork.llm.retry import CallStats, RetriesExhausted, RetryConfig, Throttle
from getwork.llm.validated import ValidationExhausted, complete_validated, parse_json_object
from getwork.relabel.cache import StampedCache
from getwork.taxonomy.model import OTHER, Taxonomy

REEXTRACT_PROMPT_VERSION: str = "reextract_v1"
MAX_ADDED_SKILLS: int = 50
MAX_SKILL_CHARS = 100
MAX_SHOWN_CHARS = 60
CACHE_KEY_HASH_CHARS = 12


@functools.cache
def load_reextract_prompt(version: str = REEXTRACT_PROMPT_VERSION) -> str:
    path = Path(__file__).parent / "prompts" / f"{version}.md"
    return path.read_text(encoding="utf-8")


def build_reextract_message(
    text: str, skills: Sequence[str], tax: Taxonomy, max_chars: int
) -> str:
    clusters = "\n".join(f"- {name}" for name in tax.cluster_names())
    known = "\n".join(f"- {s}" for s in skills) if skills else "- (none)"
    return f"CLUSTERS:\n{clusters}\n\nKNOWN SKILLS:\n{known}\n\nTEXT:\n{text[:max_chars]}"


def _shown(text: str) -> str:
    return repr(text if len(text) <= MAX_SHOWN_CHARS else text[:MAX_SHOWN_CHARS] + "...")


def _parse_item(item: object, allowed: set[str]) -> tuple[str, str]:
    if not isinstance(item, dict):
        raise ValueError("each additional skill must be an object")
    skill, cluster = item.get("skill"), item.get("cluster")
    if not isinstance(skill, str) or not isinstance(cluster, str):
        raise ValueError("skill and cluster must be strings")
    skill = skill.strip()
    if not skill:
        raise ValueError("a skill is blank")
    if len(skill) > MAX_SKILL_CHARS:
        raise ValueError(f"skill {_shown(skill)} is over {MAX_SKILL_CHARS} characters")
    if cluster not in allowed:
        raise ValueError(f"cluster {_shown(cluster)} for {_shown(skill)} is not allowed")
    return skill, cluster


def parse_reextract(text: str, valid_names: Collection[str]) -> list[tuple[str, str]]:
    items = parse_json_object(text).get("additional_skills")
    if not isinstance(items, list):
        raise ValueError('"additional_skills" must be a list')
    if len(items) > MAX_ADDED_SKILLS:
        raise ValueError(f"at most {MAX_ADDED_SKILLS} additional skills are allowed")
    allowed = set(valid_names) | {OTHER}
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        skill, cluster = _parse_item(item, allowed)
        if skill.lower() in seen:
            continue
        seen.add(skill.lower())
        result.append((skill, cluster))
    return result


@dataclass(frozen=True)
class ReextractConfig:
    model: str
    backend: str = "voyager"
    max_input_chars: int = 48000
    max_tokens: int = 2048
    max_validation_attempts: int = 3
    extra_body: Mapping[str, object] | None = None
    retry: RetryConfig = RetryConfig()


def _cache_key(doc_id: str, text: str, max_chars: int) -> str:
    digest = hashlib.sha256(text[:max_chars].encode("utf-8")).hexdigest()
    return f"{doc_id}:{digest[:CACHE_KEY_HASH_CHARS]}"


def reextract_document(
    client: LLMClient,
    cfg: ReextractConfig,
    doc_id: str,
    text: str,
    skills: Sequence[str],
    tax: Taxonomy,
    cache: StampedCache,
    stats: CallStats,
    *,
    secrets: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> list[tuple[str, str]] | None:
    # secrets is accepted for a uniform step signature; cache and results hold only skill text.
    key = _cache_key(doc_id, text, cfg.max_input_chars)
    cached = cache.get(key)
    if isinstance(cached, list):
        return [(skill, cluster) for skill, cluster in cached]
    request = LLMRequest(
        system=load_reextract_prompt(),
        user=build_reextract_message(text, skills, tax, cfg.max_input_chars),
        model=cfg.model,
        temperature=0,
        max_tokens=cfg.max_tokens,
        reasoning_effort=None,
        response_schema=None,
        extra_body=cfg.extra_body,
    )
    names = tax.cluster_names()
    try:
        added = complete_validated(
            client,
            request,
            lambda t: parse_reextract(t, names),
            cfg=cfg.retry,
            throttle=Throttle(),
            sleep=sleep,
            stats=stats,
            max_attempts=cfg.max_validation_attempts,
        )
    except (ValidationExhausted, RetriesExhausted, PermanentLLMError):
        return None
    except FatalLLMError:
        cache.save()
        raise
    cache.put(key, [[skill, cluster] for skill, cluster in added])
    cache.save()
    return added
