"""Overlap review of cluster names: word-overlap rules, an LLM judge, and renaming (ADR-25)."""

from __future__ import annotations

import functools
import json
import re
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from getwork.llm.client import LLMClient, LLMRequest
from getwork.llm.retry import CallStats, Throttle
from getwork.llm.validated import parse_json_object
from getwork.structure import pool
from getwork.taxonomy import naming

OVERLAP_PROMPT_VERSION: str = "overlap_v1"
RENAME_PROMPT_VERSION: str = "rename_v1"
STOPWORDS: frozenset[str] = frozenset(
    {"and", "or", "of", "for", "the", "a", "an", "in", "with", "to"}
)
JACCARD_THRESHOLD: float = 0.5
MAX_GROUP_SIZE: int = 5
JUDGE_MEMBERS: int = 5
OVERLAP_MEMBERS: int = 8
MAX_REASON_CHARS: int = 400
MAX_PAIR_REASONS = 3
MIN_PLURAL_WORD_CHARS = 4
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class OverlapGroup:
    ids: tuple[int, ...]
    reason: str
    source: str  # "rules", "judge" or "rules+judge"


def _singular(word: str) -> str:
    if len(word) >= MIN_PLURAL_WORD_CHARS and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def name_words(name: str) -> frozenset[str]:
    words = (w for w in _WORD.findall(name.casefold()) if w not in STOPWORDS)
    return frozenset(_singular(w) for w in words)


def rule_overlap(first: str, second: str) -> str | None:
    one, two = name_words(first), name_words(second)
    if not one or not two:
        return None
    if one == two:
        return "the names use the same words"
    shared = ", ".join(sorted(one & two))
    if one < two or two < one:
        return f"one name contains the other (shared words: {shared})"
    if len(one) >= 2 and len(two) >= 2 and len(one & two) / len(one | two) >= JACCARD_THRESHOLD:
        return f"the names share most of their words (shared words: {shared})"
    return None


def _components(ids: Sequence[int], links: Sequence[tuple[int, int]]) -> list[tuple[int, ...]]:
    parent = {i: i for i in ids}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for first, second in links:
        parent[max(find(first), find(second))] = min(find(first), find(second))
    members: dict[int, list[int]] = {}
    for i in sorted(ids):
        members.setdefault(find(i), []).append(i)
    return sorted((tuple(v) for v in members.values()), key=lambda ids_: ids_[0])


def rule_groups(names: Mapping[int, str]) -> list[OverlapGroup]:
    ids = sorted(names)
    pairs = [
        (i, j, reason)
        for n, i in enumerate(ids)
        for j in ids[n + 1 :]
        if (reason := rule_overlap(names[i], names[j])) is not None
    ]
    groups = []
    for component in _components(ids, [(i, j) for i, j, _ in pairs]):
        if len(component) < 2:
            continue
        texts = [f"{names[i]!r} and {names[j]!r}: {r}" for i, j, r in pairs if i in component]
        more = ["..."] if len(texts) > MAX_PAIR_REASONS else []
        reason = "; ".join([*texts[:MAX_PAIR_REASONS], *more])
        groups.append(OverlapGroup(ids=component, reason=reason[:MAX_REASON_CHARS], source="rules"))
    return groups


def merge_groups(*group_lists: Sequence[OverlapGroup]) -> list[OverlapGroup]:
    groups = [g for group_list in group_lists for g in group_list]
    ids = sorted({i for g in groups for i in g.ids})
    links = [(g.ids[0], i) for g in groups for i in g.ids[1:]]
    merged = []
    for component in _components(ids, links):
        parts = [g for g in groups if g.ids[0] in component]
        reasons = list(dict.fromkeys(g.reason for g in parts))
        kinds = {kind for g in parts for kind in g.source.split("+")}
        source = "rules+judge" if kinds == {"rules", "judge"} else next(iter(kinds))
        merged.append(
            OverlapGroup(
                ids=component, reason=" | ".join(reasons)[:MAX_REASON_CHARS], source=source
            )
        )
    return merged


@functools.cache
def load_overlap_prompt(version: str = OVERLAP_PROMPT_VERSION) -> str:
    return (Path(__file__).parent / "prompts" / f"{version}.md").read_text(encoding="utf-8")


def build_judge_message(names: Mapping[int, str], members: Mapping[int, Sequence[str]]) -> str:
    lines = (
        f"- {i}: {names[i]} ({'; '.join(members[i][:JUDGE_MEMBERS])})" for i in sorted(names)
    )
    return "CLUSTERS:\n" + "\n".join(lines)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_judge_group(item: object, ids: Collection[int]) -> OverlapGroup:
    if not isinstance(item, dict):
        raise ValueError("each group must be an object with \"clusters\" and \"reason\"")
    clusters = item.get("clusters")
    if not isinstance(clusters, list) or not all(_is_int(c) for c in clusters):
        raise ValueError('"clusters" must be a list of cluster ids (integers)')
    if not 2 <= len(clusters) <= MAX_GROUP_SIZE:
        raise ValueError(f'a group needs 2 to {MAX_GROUP_SIZE} clusters; split larger groups')
    if len(set(clusters)) != len(clusters):
        raise ValueError("a cluster id is repeated inside a group")
    unknown = sorted(set(clusters) - set(ids))
    if unknown:
        raise ValueError(f"cluster id {unknown[0]} is not in the CLUSTERS list")
    reason = item.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('every group needs a non-empty "reason" string')
    text = " ".join(reason.split())[:MAX_REASON_CHARS].rstrip()
    return OverlapGroup(ids=tuple(sorted(clusters)), reason=text, source="judge")


def parse_judge_groups(text: str, ids: Collection[int]) -> list[OverlapGroup]:
    items = parse_json_object(text).get("groups")
    if not isinstance(items, list):
        raise ValueError('"groups" must be a list')
    groups = [_parse_judge_group(item, ids) for item in items]
    seen: set[int] = set()
    for group in groups:
        repeated = seen & set(group.ids)
        if repeated:
            raise ValueError(f"cluster id {min(repeated)} appears in more than one group")
        seen.update(group.ids)
    return groups


def _request(env: naming.Env, system: str, user: str) -> LLMRequest:
    return LLMRequest(
        system=system,
        user=user,
        model=env.cfg.model,
        temperature=0,
        max_tokens=env.cfg.max_tokens,
        reasoning_effort=env.cfg.reasoning_effort,
        response_schema=None,
        extra_body=env.cfg.extra_body,
    )


def judge_groups(
    env: naming.Env, names: Mapping[int, str], members: Mapping[int, Sequence[str]]
) -> naming.Asked[list[OverlapGroup]]:
    request = _request(env, load_overlap_prompt(), build_judge_message(names, members))
    return naming.ask_validated(
        env=env, request=request, parse=lambda text: parse_judge_groups(text, names.keys())
    )


@functools.cache
def load_rename_prompt(version: str = RENAME_PROMPT_VERSION) -> str:
    return (Path(__file__).parent / "prompts" / f"{version}.md").read_text(encoding="utf-8")


def _member_lines(members: Sequence[str]) -> str:
    return "\n".join(f"- {m}" for m in members)


def build_rename_message(
    cluster_id: int,
    names: Mapping[int, str],
    members: Mapping[int, Sequence[str]],
    group: OverlapGroup,
    *,
    top_members: int,
) -> str:
    own = f"CLUSTER:\nName: {names[cluster_id]}\nMEMBERS:\n" + _member_lines(
        members[cluster_id][:top_members]
    )
    others = [x for x in group.ids if x != cluster_id]
    overlaps = "OVERLAPS WITH:\n" + "\n".join(
        f"- {names[x]} ({'; '.join(members[x][:OVERLAP_MEMBERS])})" for x in others
    )
    sections = [own, overlaps, f"WHY: {group.reason}"]
    rest = [names[x] for x in sorted(names) if x not in group.ids]
    if rest:
        sections.append("OTHER NAMES:\n" + "\n".join(f"- {name}" for name in rest))
    return "\n\n".join(sections)


def check_distinct_name(name: str, cluster_id: int, names: Mapping[int, str]) -> None:
    for other in sorted(names):
        if other == cluster_id:
            continue
        reason = rule_overlap(name, names[other])
        if reason is not None:
            raise ValueError(
                f"{name!r} still overlaps {names[other]!r} ({reason}); make it more distinct"
            )


def parse_rename(
    text: str, shown: Sequence[str], cluster_id: int, names: Mapping[int, str]
) -> str:
    value = parse_json_object(text).get("name")
    if not isinstance(value, str):
        raise ValueError('"name" must be a string')
    name = naming.check_concept_name(name=value, members=shown, used=list(names.values()))
    check_distinct_name(name, cluster_id, names)
    return name


def rename_one(
    env: naming.Env,
    cluster_id: int,
    names: Mapping[int, str],
    members: Mapping[int, Sequence[str]],
    group: OverlapGroup,
) -> naming.Asked[str]:
    top = env.cfg.top_members
    shown = members[cluster_id][:top]
    user = build_rename_message(cluster_id, names, members, group, top_members=top)
    request = _request(env, load_rename_prompt(), user)
    return naming.ask_validated(
        env=env,
        request=request,
        parse=lambda t: parse_rename(t, shown, cluster_id, names),
    )


def judge_failed_message(reason: str) -> str:
    return f"overlap judge failed ({reason}); using rules only"


def group_message(index: int, total: int, group: OverlapGroup) -> str:
    ids = ", ".join(map(str, group.ids))
    return f"overlap group {index}/{total}: clusters {ids} ({group.source})"


def round_message(round_no: int, rounds: int, n_groups: int, n_todo: int) -> str:
    return (
        f"overlap review round {round_no}/{rounds}: {n_groups} group(s), "
        f"{n_todo} cluster(s) to rename"
    )


def renamed_message(
    done: int, total: int, cluster_id: int, old: str, new: str, attempts: int
) -> str:
    return (
        f"renamed {done}/{total}: cluster {cluster_id} {json.dumps(old)} -> {json.dumps(new)} "
        f"(attempts {attempts})"
    )


def kept_message(
    done: int, total: int, cluster_id: int, old: str, reason: str, attempts: int
) -> str:
    return (
        f"kept {done}/{total}: cluster {cluster_id} {json.dumps(old)} "
        f"({reason}, attempts {attempts})"
    )


def done_message(unresolved: Sequence[int]) -> str:
    if not unresolved:
        return "overlap review done: no overlapping names"
    ids = ", ".join(map(str, unresolved))
    return f"overlap review done: {len(unresolved)} cluster(s) still overlapping: {ids}"


@dataclass(frozen=True)
class ReviewConfig:
    rounds: int = 2
    judge: bool = True


@dataclass
class ReviewResult:
    names: dict[int, str]
    renamed: dict[int, int]
    unresolved: list[int]
    rounds_run: int
    judge_failures: int
    stats: CallStats


Ask = Callable[[int, Mapping[int, str]], naming.Asked[str]]


@dataclass
class PassResult:
    names: dict[int, str]
    renamed: list[int]
    kept: list[int]
    stats: CallStats


@dataclass
class _Pass:
    """State of one rename pass; only the consuming thread touches it."""

    env: naming.Env
    members: Mapping[int, Sequence[str]]
    ask: Ask
    log: Callable[[str], None] | None
    total: int
    names: dict[int, str]
    renamed: list[int] = field(default_factory=list)
    kept: list[int] = field(default_factory=list)
    stats: CallStats = field(default_factory=CallStats)
    done: int = 0

    def emit(self, line: str) -> None:
        if self.log is not None:
            self.log(line)

    def finish(self, cluster_id: int, asked: naming.Asked[str]) -> None:
        self.done += 1
        self.stats = naming.add_stats(first=self.stats, second=asked.stats)
        for number, reason in enumerate(asked.rejections, start=1):
            self.emit(naming.rejected_message(cluster_id, number, reason))
        old, attempts = self.names[cluster_id], asked.stats.calls
        if asked.value is None:
            self.kept.append(cluster_id)
            line = kept_message(self.done, self.total, cluster_id, old, asked.reason, attempts)
        else:
            self.names[cluster_id] = asked.value
            self.renamed.append(cluster_id)
            line = renamed_message(
                self.done, self.total, cluster_id, old, asked.value, attempts
            )
        self.emit(line)

    def resolve(self, cluster_id: int, draft: naming.Asked[str]) -> naming.Asked[str]:
        """Accept a concurrent draft if it still passes against the names so far, else ask again."""
        if draft.value is None:
            return draft
        shown = self.members[cluster_id][: self.env.cfg.top_members]
        try:
            naming.check_concept_name(
                name=draft.value, members=shown, used=list(self.names.values())
            )
            check_distinct_name(draft.value, cluster_id, self.names)
            return draft
        except ValueError as exc:
            stale = naming.short_reason(exc)
        again = self.ask(cluster_id, self.names)
        return replace(
            again,
            stats=naming.add_stats(first=draft.stats, second=again.stats),
            rejections=(*draft.rejections, stale, *again.rejections),
        )


def _draft_concurrently(
    state: _Pass, ids: Sequence[int], snapshot: dict[int, str], workers: int
) -> None:
    drafts: dict[int, naming.Asked[str]] = {}
    released = 0
    stop, abort = threading.Event(), threading.Event()
    try:
        for cluster_id, drafted in pool.run_pool(
            list(ids), lambda i: state.ask(i, snapshot), workers, stop, abort
        ):
            if isinstance(drafted, BaseException):
                raise drafted
            drafts[cluster_id] = drafted
            while released < len(ids) and ids[released] in drafts:
                current_id = ids[released]
                released += 1
                state.finish(current_id, state.resolve(current_id, drafts.pop(current_id)))
    finally:
        abort.set()


def run_rename_pass(
    env: naming.Env,
    todo: Sequence[int],
    current: Mapping[int, str],
    members: Mapping[int, Sequence[str]],
    ask: Ask,
    *,
    workers: int,
    log: Callable[[str], None] | None = None,
) -> PassResult:
    """Rename the todo clusters in ascending id order, sequentially from the names so far
    (workers == 1) or from concurrent drafts resolved in id order (workers >= 2)."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    ids = sorted(todo)
    state = _Pass(
        env=env, members=members, ask=ask, log=log, total=len(ids), names=dict(current)
    )
    if workers == 1:
        for cluster_id in ids:
            state.finish(cluster_id, ask(cluster_id, state.names))
    else:
        _draft_concurrently(state, ids, dict(current), workers)
    return PassResult(
        names=state.names, renamed=state.renamed, kept=state.kept, stats=state.stats
    )


@dataclass
class _Review:
    """Mutable state of one review run."""

    env: naming.Env
    members: Mapping[int, Sequence[str]]
    current: dict[int, str]
    log: Callable[[str], None] | None
    judge: bool
    kept: set[int] = field(default_factory=set)
    renamed: dict[int, int] = field(default_factory=dict)
    stats: CallStats = field(default_factory=CallStats)
    judge_failures: int = 0

    def emit(self, line: str) -> None:
        if self.log is not None:
            self.log(line)

    def detect(self) -> list[OverlapGroup]:
        found = rule_groups(self.current)
        if not self.judge:
            return found
        asked = judge_groups(env=self.env, names=self.current, members=self.members)
        self.stats = naming.add_stats(first=self.stats, second=asked.stats)
        if asked.value is None:
            self.judge_failures += 1
            self.emit(judge_failed_message(asked.reason))
            return found
        return merge_groups(found, asked.value)

    def rename(
        self, todo: Sequence[int], group_of: Mapping[int, OverlapGroup], round_no: int, workers: int
    ) -> None:
        def ask(cluster_id: int, names_seen: Mapping[int, str]) -> naming.Asked[str]:
            return rename_one(
                env=self.env,
                cluster_id=cluster_id,
                names=names_seen,
                members=self.members,
                group=group_of[cluster_id],
            )

        result = run_rename_pass(
            env=self.env,
            todo=todo,
            current=self.current,
            members=self.members,
            ask=ask,
            workers=workers,
            log=self.log,
        )
        self.current = result.names
        self.kept.update(result.kept)
        self.renamed.update({i: round_no for i in result.renamed})
        self.stats = naming.add_stats(first=self.stats, second=result.stats)


def _announce(
    review: _Review, groups: Sequence[OverlapGroup], round_no: int, rounds: int
) -> tuple[list[int], dict[int, OverlapGroup]]:
    for index, group in enumerate(groups, start=1):
        review.emit(group_message(index, len(groups), group))
    group_of = {i: group for group in groups for i in group.ids}
    todo = sorted(i for i in group_of if i not in review.kept)
    review.emit(round_message(round_no, rounds, len(groups), len(todo)))
    return todo, group_of


def _unresolved(review: _Review) -> list[int]:
    still_grouped = {i for group in rule_groups(review.current) for i in group.ids}
    return sorted(review.kept | still_grouped)


def review_overlaps(
    client: LLMClient,
    cfg: naming.NameConfig,
    review: ReviewConfig,
    names: Mapping[int, str],
    members: Mapping[int, Sequence[str]],
    *,
    sleep: Callable[[float], None] = time.sleep,
    workers: int = 1,
    log: Callable[[str], None] | None = None,
) -> ReviewResult:
    if review.rounds < 0:
        raise ValueError(f"rounds must not be negative, got {review.rounds}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    env = naming.Env(client=client, cfg=cfg, throttle=Throttle(), sleep=sleep)
    state = _Review(
        env=env, members=members, current=dict(names), log=log, judge=review.judge
    )
    if review.rounds == 0 or len(names) < 2:
        return ReviewResult(
            names=state.current, renamed={}, unresolved=[], rounds_run=0, judge_failures=0,
            stats=state.stats,
        )
    converged, rounds_run = False, 0
    for round_no in range(1, review.rounds + 1):
        groups = state.detect()
        if not groups:
            converged = True
            break
        rounds_run = round_no
        todo, group_of = _announce(state, groups, round_no, review.rounds)
        if not todo:
            break
        state.rename(todo, group_of, round_no, workers)
    unresolved = [] if converged else _unresolved(state)
    state.emit(done_message(unresolved))
    return ReviewResult(
        names=state.current,
        renamed=state.renamed,
        unresolved=unresolved,
        rounds_run=rounds_run,
        judge_failures=state.judge_failures,
        stats=state.stats,
    )
