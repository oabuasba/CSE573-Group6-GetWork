import json
import threading
import time
from collections.abc import Callable

import pytest

from getwork.llm.client import (
    FatalLLMError,
    LLMRequest,
    PermanentLLMError,
    RateLimitError,
    TransientLLMError,
)
from getwork.llm.fake import FakeLLMClient
from getwork.llm.retry import RetryConfig
from getwork.taxonomy import naming, overlap
from getwork.taxonomy.overlap import ReviewConfig

pytestmark = pytest.mark.usefixtures("offline")

V4 = {
    0: "Front-end Development", 1: "Desktop Applications", 2: "Relational Databases",
    3: "Development Tools", 4: "IT Service Management", 5: "Software Development Tools",
    6: "Server-Side Programming", 7: "Cloud Infrastructure", 8: "Network Infrastructure",
    9: "Linux and UNIX Administration",
}
MEMBERS = {i: [f"s{i}a", f"s{i}b", f"s{i}c"] for i in V4}
GOOD = {3: "Build and Code Tools", 5: "Source Control and CI Tools"}
EMPTY_JUDGE = '{"groups": []}'
CFG = naming.NameConfig(model="m", retry=RetryConfig(max_attempts=2, rate_limit_wait_s=0.001))


def _no_sleep(_: float) -> None:
    return None


def _cluster_of(request: LLMRequest) -> int:
    first = request.user.split("MEMBERS:\n- ")[1].split("\n")[0]
    return int(first[1:-1])


class Model:
    """Handler keyed on request text; per-cluster call counters live under one lock."""

    def __init__(
        self,
        rename: Callable[[int, int], str] | None = None,
        judge: Callable[[int], str] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.rename = rename or (lambda cid, n: GOOD[cid])
        self.judge = judge or (lambda n: EMPTY_JUDGE)
        self.delay = delay
        self.lock = threading.Lock()
        self.requests: list[LLMRequest] = []
        self.counts: dict[int, int] = {}
        self.judge_calls = 0

    def __call__(self, request: LLMRequest) -> str:
        with self.lock:
            self.requests.append(request)
            if request.user.startswith("CLUSTERS:"):
                self.judge_calls += 1
                return self.judge(self.judge_calls)
            cid = _cluster_of(request)
            self.counts[cid] = self.counts.get(cid, 0) + 1
            number = self.counts[cid]
        if self.delay:
            time.sleep(self.delay * (10 - cid))
        return json.dumps({"name": self.rename(cid, number)})

    def renames(self) -> list[LLMRequest]:
        return [r for r in self.requests if "OVERLAPS WITH:" in r.user]


def _run(  # type: ignore[no-untyped-def]
    model: Model, names=V4, *, rounds=2, judge=False, workers=1, members=MEMBERS
):
    lines: list[str] = []
    client = FakeLLMClient(handler=model)
    result = overlap.review_overlaps(
        client, CFG, ReviewConfig(rounds=rounds, judge=judge), names, members,
        sleep=_no_sleep, workers=workers, log=lines.append,
    )
    return result, lines, client


def test_rounds_zero_and_invalid_arguments() -> None:
    model = Model()
    result, lines, client = _run(model, rounds=0)
    assert result.names == V4 and lines == [] and client.calls == [] and result.rounds_run == 0
    for kwargs in ({"rounds": -1}, {"workers": 0}):
        with pytest.raises(ValueError):
            _run(model, **kwargs)
    assert client.calls == []


def test_no_overlap_makes_no_rename_calls() -> None:
    clean = {i: n for i, n in V4.items() if i not in (3, 5)}
    model = Model()
    result, lines, client = _run(model, clean, judge=True)
    assert model.judge_calls == 1 and model.renames() == [] and len(client.calls) == 1
    assert (result.renamed, result.unresolved) == ({}, [])
    assert lines == ["overlap review done: no overlapping names"]
    _, _, off = _run(Model(), clean, judge=False)
    assert off.calls == []
    _, none_lines, single = _run(Model(), {3: "Development Tools"}, judge=True)
    assert single.calls == [] and none_lines == []


def test_sequential_review_renames_both_clusters() -> None:
    model = Model()
    result, lines, client = _run(model)
    group = overlap.rule_groups(V4)[0]
    first, second = model.renames()
    assert [_cluster_of(r) for r in client.calls] == [3, 5]
    assert first.user == overlap.build_rename_message(3, V4, MEMBERS, group, top_members=25)
    updated = {**V4, 3: "Build and Code Tools"}
    assert second.user == overlap.build_rename_message(5, updated, MEMBERS, group, top_members=25)
    assert "- Build and Code Tools (" in second.user and "Rules" not in second.user
    assert result.names == {**V4, **GOOD}
    assert (result.renamed, result.unresolved, result.rounds_run) == ({3: 1, 5: 1}, [], 1)
    assert lines[-1] == "overlap review done: no overlapping names"
    assert result.stats.calls == 2


def _first_wave(cid: int, n: int) -> str:
    if cid == 3:
        return "Source Control"
    return "Source Control Tools" if n == 1 else "Build Pipelines"


def test_parallel_review_independent_of_workers_and_finish_order() -> None:
    runs = []
    for workers in (2, 3, 8):
        model = Model(delay=0.002)
        result, lines, _ = _run(model, workers=workers)
        runs.append((result.names, result.renamed, result.unresolved, lines))
        group = overlap.rule_groups(V4)[0]
        for request in model.renames():
            cid = _cluster_of(request)
            assert request.user == overlap.build_rename_message(
                cid, V4, MEMBERS, group, top_members=25
            )
    assert runs[0] == runs[1] == runs[2]
    assert runs[0][0] == {**V4, **GOOD} and runs[0][1] == {3: 1, 5: 1}


def test_parallel_draft_collision_is_reasked() -> None:
    results = []
    for workers in (2, 8):
        model = Model(rename=_first_wave, delay=0.002)
        result, lines, _ = _run(model, workers=workers)
        results.append((result.names, lines))
        again = model.renames()[-1]
        assert _cluster_of(again) == 5 and "- Source Control (" in again.user
    assert results[0] == results[1]
    names, lines = results[0]
    assert names[3] == "Source Control" and names[5] == "Build Pipelines"
    stale = "rejected cluster 5 attempt 1: 'Source Control Tools' still"
    assert any(line.startswith(stale) for line in lines)
    assert (
        'renamed 2/2: cluster 5 "Software Development Tools" -> "Build Pipelines" (attempts 2)'
        in lines
    )


STUBBORN = {3: "Development Tool Kit", 5: "Development Tools Suite"}


def test_stubborn_model_keeps_names_and_stops() -> None:
    model = Model(rename=lambda cid, n: STUBBORN[cid])
    result, lines, client = _run(model, judge=True)
    assert result.names == V4 and result.unresolved == [3, 5] and result.renamed == {}
    assert model.counts == {3: 3, 5: 3} and model.judge_calls == 2
    assert len(client.calls) == 2 + 6 and result.stats.calls == 8
    assert lines[-1] == "overlap review done: 2 cluster(s) still overlapping: 3, 5"
    assert sum(line.startswith("kept ") for line in lines) == 2
    assert 'kept 1/2: cluster 3 "Development Tools" (replies rejected, attempts 3)' in lines


def test_rounds_limit_and_second_round_fix() -> None:
    def judge(n: int) -> str:
        pair = [0, 1] if n == 1 else [0, 7]
        return json.dumps({"groups": [{"clusters": pair, "reason": "close"}]})

    def rename(cid: int, n: int) -> str:
        ui = "UI Engineering" if n == 1 else "Web UI Engineering"
        return {3: GOOD[3], 5: GOOD[5], 1: "Native Clients", 0: ui, 7: "Hosting Platforms"}[cid]

    result, _, _ = _run(Model(rename=rename, judge=judge), judge=True, rounds=2)
    assert result.renamed == {0: 2, 1: 1, 3: 1, 5: 1, 7: 2}
    assert result.unresolved == [] and result.rounds_run == 2
    assert result.names[0] == "Web UI Engineering"

    # A cluster that cannot be renamed stays in the report through the kept route; a renamed
    # cluster can never remain in a rule group because every rename is checked against the rest.
    stuck = lambda cid, n: "Cloud Infrastructure Tools" if cid == 5 else GOOD[cid]  # noqa: E731
    limited, lines, _ = _run(Model(rename=stuck), rounds=1)
    assert limited.rounds_run == 1 and limited.unresolved == [5] and limited.renamed == {3: 1}
    assert lines[-1] == "overlap review done: 1 cluster(s) still overlapping: 5"


SEMANTIC = {0: "Web Development", 1: "Front-end Development", 2: "Cloud"}
SEM_MEMBERS = {i: [f"s{i}a", f"s{i}b"] for i in SEMANTIC}


def test_judge_finds_semantic_overlap() -> None:
    reply = json.dumps({"groups": [{"clusters": [0, 1], "reason": "both are web work"}]})
    model = Model(
        rename=lambda cid, n: {0: "Web Services", 1: "User Interface Design"}[cid],
        judge=lambda n: reply if n == 1 else EMPTY_JUDGE,
    )
    result, lines, _ = _run(model, SEMANTIC, judge=True, members=SEM_MEMBERS)
    assert result.names == {0: "Web Services", 1: "User Interface Design", 2: "Cloud"}
    assert all("WHY: both are web work" in r.user for r in model.renames())
    assert lines[0] == "overlap group 1/1: clusters 0, 1 (judge)"

    both = json.dumps({"groups": [{"clusters": [3, 5], "reason": "same tools"}]})
    merged = Model(judge=lambda n: both if n == 1 else EMPTY_JUDGE)
    _, merged_lines, _ = _run(merged, judge=True)
    assert merged_lines[0] == "overlap group 1/1: clusters 3, 5 (rules+judge)"
    assert all(r.user.split("WHY: ")[1].split("\n")[0].endswith(" | same tools")
               for r in merged.renames())


def test_judge_failure_falls_back_to_rules() -> None:
    model = Model(judge=lambda n: "nope")
    result, lines, _ = _run(model, judge=True)
    assert result.judge_failures == 2 and result.names == {**V4, **GOOD}
    assert lines[0] == "overlap judge failed (replies rejected); using rules only"
    refused = FakeLLMClient(handler=lambda r: (_ for _ in ()).throw(PermanentLLMError("no")))
    refusing_lines: list[str] = []
    refusing = overlap.review_overlaps(
        refused, CFG, ReviewConfig(rounds=1), V4, MEMBERS, sleep=_no_sleep,
        log=refusing_lines.append,
    )
    assert refusing.judge_failures == 1
    assert refusing_lines[0] == "overlap judge failed (request refused); using rules only"

    unknown = Model(judge=lambda n: '{"groups": [{"clusters": [0, 99], "reason": "x"}]}'
                    if n == 1 else EMPTY_JUDGE)
    clean = {i: n for i, n in V4.items() if i not in (3, 5)}
    ok, _, _ = _run(unknown, clean, judge=True)
    judge_requests = [r for r in unknown.requests if r.user.startswith("CLUSTERS:")]
    assert "cluster id 99 is not in the CLUSTERS list" in judge_requests[1].user
    assert ok.judge_failures == 0


def test_progress_lines_formats_and_order() -> None:
    assert overlap.judge_failed_message("x") == "overlap judge failed (x); using rules only"
    group = overlap.OverlapGroup((3, 5), "r", "rules")
    assert overlap.group_message(1, 2, group) == "overlap group 1/2: clusters 3, 5 (rules)"
    assert overlap.round_message(1, 2, 3, 4) == (
        "overlap review round 1/2: 3 group(s), 4 cluster(s) to rename"
    )
    assert overlap.renamed_message(1, 2, 3, 'a"b', "é", 2) == (
        'renamed 1/2: cluster 3 "a\\"b" -> "\\u00e9" (attempts 2)'
    )
    assert overlap.kept_message(2, 2, 5, "Old", "replies rejected", 3) == (
        'kept 2/2: cluster 5 "Old" (replies rejected, attempts 3)'
    )
    assert overlap.done_message([]) == "overlap review done: no overlapping names"
    assert overlap.done_message([3]) == "overlap review done: 1 cluster(s) still overlapping: 3"

    model = Model(rename=lambda cid, n: "Development Tool Kit" if (cid, n) == (3, 1) else GOOD[cid])
    _, lines, _ = _run(model)
    assert lines == [
        "overlap group 1/1: clusters 3, 5 (rules)",
        "overlap review round 1/2: 1 group(s), 2 cluster(s) to rename",
        "rejected cluster 3 attempt 1: 'Development Tool Kit' still overlaps 'Software Development "
        "Tools' (the names share most of their words (shared words: development, tool)); "
        "make it more distinct",
        'renamed 1/2: cluster 3 "Development Tools" -> "Build and Code Tools" (attempts 2)',
        'renamed 2/2: cluster 5 "Software Development Tools" -> '
        '"Source Control and CI Tools" (attempts 1)',
        "overlap review done: no overlapping names",
    ]
    quiet = overlap.review_overlaps(
        FakeLLMClient(handler=Model()), CFG, ReviewConfig(judge=False), V4, MEMBERS, sleep=_no_sleep
    )
    assert quiet.names == {**V4, **GOOD}


def test_progress_lines_carry_no_reply_or_reason_text() -> None:
    marker = "LEAKMARKER"
    reply = json.dumps({"groups": [{"clusters": [3, 5], "reason": marker}], "note": marker})
    model = Model(
        rename=lambda cid, n: GOOD[cid] if n > 1 else "Other",
        judge=lambda n: reply,
    )
    _, lines, _ = _run(model, judge=True)
    assert lines and all(marker not in line for line in lines)
    assert all(line.startswith("rejected") for line in lines if "Other" in line)


def _assert_work_stopped(client: FakeLLMClient) -> None:
    settled = len(client.calls)
    time.sleep(0.15)
    assert len(client.calls) == settled


def test_fatal_and_unexpected_errors_propagate() -> None:
    fatal = FatalLLMError("denied")

    def failing(
        when: Callable[[LLMRequest], bool], error: BaseException
    ) -> Callable[[LLMRequest], str]:
        inner = Model(rename=_first_wave)

        def handler(request: LLMRequest) -> str:
            if when(request):
                raise error
            return inner(request)

        return handler

    cases = [
        (lambda r: r.user.startswith("CLUSTERS:"), 1, True),
        (lambda r: "OVERLAPS WITH:" in r.user and _cluster_of(r) == 3, 2, False),
        (lambda r: "OVERLAPS WITH:" in r.user and "- Source Control (" in r.user, 2, False),
    ]
    for when, workers, judge in cases:
        client = FakeLLMClient(handler=failing(when, fatal))
        lines: list[str] = []
        with pytest.raises(FatalLLMError) as raised:
            overlap.review_overlaps(client, CFG, ReviewConfig(judge=judge), V4, MEMBERS,
                                    sleep=_no_sleep, workers=workers, log=lines.append)
        assert raised.value is fatal
        _assert_work_stopped(client)
        count = len(lines)
        time.sleep(0.05)
        assert len(lines) == count
        assert not any(line.startswith("overlap review done") for line in lines)
    boom = failing(
        lambda r: "OVERLAPS WITH:" in r.user and _cluster_of(r) == 5, RuntimeError("boom")
    )
    client = FakeLLMClient(handler=boom)
    with pytest.raises(RuntimeError, match="boom"):
        overlap.review_overlaps(client, CFG, ReviewConfig(judge=False), V4, MEMBERS,
                                sleep=_no_sleep, workers=2)
    _assert_work_stopped(client)


def test_review_stats_sum() -> None:
    seen: set[str] = set()
    lock = threading.Lock()
    inner = Model(judge=lambda n: "bad" if n == 1 else EMPTY_JUDGE)

    def handler(request: LLMRequest) -> str:
        key = request.user[:40] if "OVERLAPS WITH:" in request.user else ""
        with lock:
            first = bool(key) and key not in seen
            seen.add(key)
        if first and _cluster_of(request) == 3:
            raise RateLimitError("429")
        if first and _cluster_of(request) == 5:
            raise TransientLLMError("503")
        return inner(request)

    client = FakeLLMClient(handler=handler)
    result = overlap.review_overlaps(client, CFG, ReviewConfig(rounds=1, judge=True), V4, MEMBERS,
                                     sleep=_no_sleep, workers=2)
    assert result.stats.calls == len(client.calls)
    assert (result.stats.rate_limited, result.stats.retries) == (1, 2)
    assert result.names == {**V4, **GOOD}


def test_run_rename_pass_is_public_and_worker_independent() -> None:
    outcomes = []
    for workers in (1, 2, 8):
        model = Model(rename=_first_wave, delay=0.002)
        env = naming.Env(FakeLLMClient(handler=model), CFG, naming.Throttle(), _no_sleep)
        group = overlap.rule_groups(V4)[0]

        def ask(cid: int, seen: dict[int, str]) -> naming.Asked[str]:
            return overlap.rename_one(env, cid, seen, MEMBERS, group)

        lines: list[str] = []
        result = overlap.run_rename_pass(env, [5, 3], V4, MEMBERS, ask, workers=workers,
                                         log=lines.append)
        outcomes.append((result.names, result.renamed, result.kept, result.stats.calls, lines))
    assert outcomes[1] == outcomes[2]
    names, renamed, kept, _, lines = outcomes[1]
    assert (renamed, kept) == ([3, 5], []) and names[5] == "Build Pipelines"
    assert lines[-1].startswith("renamed 2/2: cluster 5")
    with pytest.raises(ValueError):
        overlap.run_rename_pass(env, [3], V4, MEMBERS, ask, workers=0)
