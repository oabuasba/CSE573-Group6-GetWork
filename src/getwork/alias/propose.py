"""Draft alias pairs with an LLM for human review (never edits the curated file)."""

import functools
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from getwork.alias.table import AliasTable
from getwork.llm.client import LLMClient, LLMRequest, PermanentLLMError
from getwork.llm.retry import CallStats, RetriesExhausted, RetryConfig, Throttle
from getwork.llm.validated import ValidationExhausted, complete_validated, parse_json_object

ALIAS_PROMPT_VERSION: str = "alias_v1"
MAX_SKILL_CHARS_IN_ERROR = 60


@functools.cache
def load_alias_prompt(version: str = ALIAS_PROMPT_VERSION) -> str:
    path = Path(__file__).parent / "prompts" / f"{version}.md"
    return path.read_text(encoding="utf-8")


def build_alias_message(batch: Sequence[str], reference: Sequence[str]) -> str:
    return (
        "BATCH:\n"
        + "\n".join("- " + s for s in batch)
        + "\n\nREFERENCE:\n"
        + "\n".join("- " + s for s in reference)
    )


@dataclass(frozen=True)
class AliasGroup:
    canonical: str
    aliases: tuple[str, ...]


def _short(text: str) -> str:
    if len(text) <= MAX_SKILL_CHARS_IN_ERROR:
        return repr(text)
    return repr(text[:MAX_SKILL_CHARS_IN_ERROR] + "...")


def _parse_group_shape(item: object) -> AliasGroup:
    if not isinstance(item, dict):
        raise ValueError("each group must be an object")
    canonical, aliases = item.get("canonical"), item.get("aliases")
    if not isinstance(canonical, str):
        raise ValueError("group canonical must be a string")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"aliases of {_short(canonical)} must be a non-empty list")
    if not all(isinstance(a, str) for a in aliases):
        raise ValueError(f"aliases of {_short(canonical)} must all be strings")
    return AliasGroup(canonical=canonical, aliases=tuple(aliases))


def _check_group(group: AliasGroup, batch: Collection[str], reference: Collection[str]) -> None:
    if group.canonical not in batch and group.canonical not in reference:
        raise ValueError(f"canonical {_short(group.canonical)} is not in the batch or reference")
    for alias in group.aliases:
        if alias not in batch:
            raise ValueError(f"alias {_short(alias)} is not in the batch")
        if alias == group.canonical:
            raise ValueError(f"canonical {_short(alias)} is listed among its aliases")


def parse_groups(
    text: str, batch: Collection[str], reference: Collection[str]
) -> list[AliasGroup]:
    data = parse_json_object(text)
    items = data.get("groups")
    if not isinstance(items, list):
        raise ValueError('"groups" must be a list')
    groups = [_parse_group_shape(item) for item in items]
    seen_aliases: set[str] = set()
    for group in groups:
        _check_group(group, batch, reference)
        for alias in group.aliases:
            if alias in seen_aliases:
                raise ValueError(f"alias {_short(alias)} appears in more than one group")
            seen_aliases.add(alias)
    for group in groups:
        if group.canonical in seen_aliases:
            raise ValueError(f"canonical {_short(group.canonical)} is also an alias elsewhere")
    return groups


@dataclass(frozen=True)
class ProposeConfig:
    model: str
    backend: str = "voyager"
    batch_size: int = 100
    reference_size: int = 150
    max_skills: int = 600
    max_tokens: int = 4096
    max_validation_attempts: int = 3
    extra_body: Mapping[str, object] | None = None
    retry: RetryConfig = RetryConfig()


@dataclass
class ProposeResult:
    rows: list[dict]
    failed_batches: list[int]
    stats: CallStats = field(default_factory=CallStats)


def _request(cfg: ProposeConfig, batch: Sequence[str], reference: Sequence[str]) -> LLMRequest:
    return LLMRequest(
        system=load_alias_prompt(),
        user=build_alias_message(batch, reference),
        model=cfg.model,
        temperature=0,
        max_tokens=cfg.max_tokens,
        reasoning_effort=None,
        response_schema=None,
        extra_body=cfg.extra_body,
    )


def _batch_rows(
    groups: Sequence[AliasGroup],
    *,
    batch_no: int,
    counts: Mapping[str, int],
    table: AliasTable,
    cfg: ProposeConfig,
) -> list[dict]:
    rows = []
    for group in groups:
        for alias in group.aliases:
            if table.canonical(alias) == table.canonical(group.canonical):
                continue
            rows.append({
                "alias": alias,
                "canonical": group.canonical,
                "alias_count": counts.get(alias, 0),
                "canonical_count": counts.get(group.canonical, 0),
                "batch": batch_no,
                "status": "new",
                "model": cfg.model,
                "backend": cfg.backend,
                "prompt_version": ALIAS_PROMPT_VERSION,
            })
    return rows


def _mark_conflicts(rows: list[dict]) -> None:
    first_canonical: dict[str, str] = {}
    for row in sorted(rows, key=lambda r: r["batch"]):
        earlier = first_canonical.setdefault(row["alias"], row["canonical"])
        if earlier != row["canonical"]:
            row["status"] = "conflict"


def propose_aliases(
    client: LLMClient,
    cfg: ProposeConfig,
    skill_counts: Sequence[tuple[str, int]],
    table: AliasTable,
    *,
    secrets: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> ProposeResult:
    # secrets is accepted for a uniform step signature; rows hold only skill text.
    used = list(skill_counts[: cfg.max_skills])
    counts = dict(used)
    skills = [name for name, _ in used]
    reference = skills[: cfg.reference_size]
    result = ProposeResult(rows=[], failed_batches=[])
    throttle = Throttle()
    for batch_no, start in enumerate(range(0, len(skills), cfg.batch_size)):
        batch = skills[start : start + cfg.batch_size]
        try:
            groups = complete_validated(
                client,
                _request(cfg, batch, reference),
                lambda t, b=batch: parse_groups(t, b, reference),
                cfg=cfg.retry,
                throttle=throttle,
                sleep=sleep,
                stats=result.stats,
                max_attempts=cfg.max_validation_attempts,
            )
        except (ValidationExhausted, RetriesExhausted, PermanentLLMError):
            result.failed_batches.append(batch_no)
            continue
        result.rows.extend(
            _batch_rows(groups, batch_no=batch_no, counts=counts, table=table, cfg=cfg)
        )
    _mark_conflicts(result.rows)
    result.rows.sort(key=lambda r: (r["canonical"].casefold(), r["alias"].casefold()))
    return result
