"""Fit review of cluster names: a judge lists names that do not describe their members, then renames (ADR-26)."""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from getwork.llm.client import LLMClient, LLMRequest
from getwork.llm.retry import CallStats, Throttle
from getwork.llm.validated import parse_json_object
from getwork.taxonomy import naming, overlap

FIT_PROMPT_VERSION: str = "fit_v1"
REFIT_PROMPT_VERSION: str = "refit_v1"
FIT_MEMBERS: int = 10
MAX_WEAK_FRACTION: float = 0.5
MAX_FIT_REASON_CHARS: int = 300


@dataclass(frozen=True)
class WeakFit:
    cluster_id: int
    reason: str


@functools.cache
def _read_prompt(version: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{version}.md").read_text(encoding="utf-8")


def load_fit_prompt(version: str = FIT_PROMPT_VERSION) -> str:
    return _read_prompt(version)


def load_refit_prompt(version: str = REFIT_PROMPT_VERSION) -> str:
    return _read_prompt(version)


def build_fit_message(names: Mapping[int, str], members: Mapping[int, Sequence[str]]) -> str:
    lines = (
        f"- {i}: {names[i]} ({'; '.join(members[i][:FIT_MEMBERS])})" for i in sorted(names)
    )
    return "NAMES TO CHECK:\n" + "\n".join(lines)


def _parse_weak(item: object, ids: Collection[int]) -> WeakFit:
    if not isinstance(item, dict):
        raise ValueError('each entry must be an object with "cluster" and "reason"')
    cluster = item.get("cluster")
    if not isinstance(cluster, int) or isinstance(cluster, bool):
        raise ValueError('"cluster" must be a cluster id (integer)')
    if cluster not in ids:
        raise ValueError(f"cluster id {cluster} is not in the NAMES TO CHECK list")
    reason = item.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('every entry needs a non-empty "reason" string')
    text = " ".join(reason.split())[:MAX_FIT_REASON_CHARS].rstrip()
    return WeakFit(cluster_id=cluster, reason=text)


def parse_fit(text: str, ids: Collection[int]) -> list[WeakFit]:
    items = parse_json_object(text).get("weak")
    if not isinstance(items, list):
        raise ValueError('"weak" must be a list')
    limit = max(1, int(len(ids) * MAX_WEAK_FRACTION))
    if len(items) > limit:
        raise ValueError(f"list at most {limit} clusters: only the clearest misfits")
    weak = [_parse_weak(item, ids) for item in items]
    listed = [w.cluster_id for w in weak]
    if len(set(listed)) != len(listed):
        raise ValueError("a cluster id is listed more than once")
    return sorted(weak, key=lambda w: w.cluster_id)


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


def fit_judge(
    env: naming.Env, names: Mapping[int, str], members: Mapping[int, Sequence[str]]
) -> naming.Asked[list[WeakFit]]:
    request = _request(env, load_fit_prompt(), build_fit_message(names, members))
    return naming.ask_validated(
        env=env, request=request, parse=lambda text: parse_fit(text, names.keys())
    )


def build_refit_message(
    cluster_id: int,
    names: Mapping[int, str],
    members: Mapping[int, Sequence[str]],
    weak: WeakFit,
    *,
    top_members: int,
) -> str:
    shown = "\n".join(f"- {m}" for m in members[cluster_id][:top_members])
    sections = [
        f"CLUSTER:\nName: {names[cluster_id]}\nMEMBERS:\n{shown}",
        f"WHY: {weak.reason}",
    ]
    others = [names[i] for i in sorted(names) if i != cluster_id]
    if others:
        sections.append("OTHER NAMES:\n" + "\n".join(f"- {name}" for name in others))
    return "\n\n".join(sections)


def refit_one(
    env: naming.Env,
    cluster_id: int,
    names: Mapping[int, str],
    members: Mapping[int, Sequence[str]],
    weak: WeakFit,
) -> naming.Asked[str]:
    top = env.cfg.top_members
    shown = members[cluster_id][:top]
    user = build_refit_message(cluster_id, names, members, weak, top_members=top)
    request = _request(env, load_refit_prompt(), user)
    return naming.ask_validated(
        env=env,
        request=request,
        parse=lambda text: overlap.parse_rename(text, shown, cluster_id, names),
    )


def fit_judge_failed_message(reason: str) -> str:
    return f"fit judge failed ({reason}); skipping the fit review"


def fit_round_message(round_no: int, rounds: int, n_weak: int, n_total: int) -> str:
    return f"fit review round {round_no}/{rounds}: {n_weak} of {n_total} name(s) judged weak"


def weak_message(cluster_id: int, name: str) -> str:
    return f"weak fit: cluster {cluster_id} {json.dumps(name)}"


def fit_done_message(unresolved: Sequence[int]) -> str:
    if not unresolved:
        return "fit review done: no weak names left"
    ids = ", ".join(map(str, unresolved))
    return f"fit review done: {len(unresolved)} cluster(s) still weak: {ids}"


@dataclass(frozen=True)
class FitConfig:
    rounds: int = 1


@dataclass
class FitResult:
    names: dict[int, str]
    renamed: dict[int, int]
    unresolved: list[int]
    rounds_run: int
    judge_failures: int
    stats: CallStats


@dataclass
class _Review:
    """Mutable state of one review run."""

    env: naming.Env
    members: Mapping[int, Sequence[str]]
    current: dict[int, str]
    log: Callable[[str], None] | None
    kept: set[int] = field(default_factory=set)
    renamed: dict[int, int] = field(default_factory=dict)
    stats: CallStats = field(default_factory=CallStats)
    rounds_run: int = 0
    judge_failures: int = 0

    def emit(self, line: str) -> None:
        if self.log is not None:
            self.log(line)

    def add_stats(self, stats: CallStats) -> None:
        self.stats = naming.add_stats(first=self.stats, second=stats)

    def judge(self, round_no: int, rounds: int) -> list[WeakFit] | None:
        """The weak clusters of this round, or None when the judge failed."""
        asked = fit_judge(env=self.env, names=self.current, members=self.members)
        self.add_stats(asked.stats)
        if asked.value is None:
            self.judge_failures += 1
            self.emit(fit_judge_failed_message(asked.reason))
            return None
        self.rounds_run = round_no
        self.emit(fit_round_message(round_no, rounds, len(asked.value), len(self.current)))
        for weak in asked.value:
            self.emit(weak_message(weak.cluster_id, self.current[weak.cluster_id]))
        return asked.value

    def rename(self, weak: Sequence[WeakFit], round_no: int, workers: int) -> None:
        by_id = {w.cluster_id: w for w in weak}

        def ask(cluster_id: int, names_seen: Mapping[int, str]) -> naming.Asked[str]:
            return refit_one(
                env=self.env,
                cluster_id=cluster_id,
                names=names_seen,
                members=self.members,
                weak=by_id[cluster_id],
            )

        todo = sorted(i for i in by_id if i not in self.kept)
        result = overlap.run_rename_pass(
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
        self.add_stats(result.stats)

    def has_work(self, weak: Sequence[WeakFit]) -> bool:
        return any(w.cluster_id not in self.kept for w in weak)


def review_fit(
    client: LLMClient,
    cfg: naming.NameConfig,
    fit: FitConfig,
    names: Mapping[int, str],
    members: Mapping[int, Sequence[str]],
    *,
    sleep: Callable[[float], None] = time.sleep,
    workers: int = 1,
    log: Callable[[str], None] | None = None,
) -> FitResult:
    if fit.rounds < 0:
        raise ValueError(f"rounds must not be negative, got {fit.rounds}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    env = naming.Env(client=client, cfg=cfg, throttle=Throttle(), sleep=sleep)
    state = _Review(env=env, members=members, current=dict(names), log=log)
    if fit.rounds == 0 or not names:
        return _result(state)
    for round_no in range(1, fit.rounds + 1):
        weak = state.judge(round_no, fit.rounds)
        if weak is None or not state.has_work(weak):
            break
        state.rename(weak, round_no, workers)
    judge_failed_first = state.judge_failures > 0 and state.rounds_run == 0
    if not judge_failed_first:
        state.emit(fit_done_message(sorted(state.kept)))
    return _result(state)


def _result(state: _Review) -> FitResult:
    return FitResult(
        names=state.current,
        renamed=state.renamed,
        unresolved=sorted(state.kept),
        rounds_run=state.rounds_run,
        judge_failures=state.judge_failures,
        stats=state.stats,
    )
