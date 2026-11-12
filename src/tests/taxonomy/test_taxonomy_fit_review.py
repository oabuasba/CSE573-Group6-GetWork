import json
import threading
import time
from collections.abc import Callable

import pytest

from getwork.llm.client import FatalLLMError, LLMRequest, PermanentLLMError
from getwork.llm.fake import FakeLLMClient
from getwork.llm.retry import RetryConfig
from getwork.taxonomy import fit, naming

pytestmark = pytest.mark.usefixtures("offline")

V4 = {
    0: "Front-end Development", 1: "Desktop Applications", 2: "Relational Databases",
    3: "Development Tools", 4: "IT Service Management", 5: "Software Development Tools",
    6: "Server-Side Programming", 7: "Cloud Infrastructure", 8: "Network Infrastructure",
    9: "Linux and UNIX Administration",
}
MEMBERS = {i: [f"s{i}a", f"s{i}b", f"s{i}c"] for i in V4}
MEMBERS[1] = ["Windows", "Microsoft Office", "Active Directory", "Excel", "SharePoint"]
MEMBERS[3] = ["Eclipse", "Waterfall", "ANT", "Cassandra", "TOAD"]
MEMBERS[7] = ["AWS", "Azure", "Docker", "MongoDB", "JMS"]
GOOD = {1: "Windows and Office Administration", 3: "Build and Code Tools"}
REASONS = {1: "SECRET-REASON one", 3: "SECRET-REASON three"}
CFG = naming.NameConfig(model="m", retry=RetryConfig(max_attempts=2, rate_limit_wait_s=0.001))


def _no_sleep(_: float) -> None:
    return None


def _weak_reply(ids: list[int]) -> str:
    return json.dumps({"weak": [{"cluster": i, "reason": REASONS.get(i, "bad")} for i in ids]})


def _cluster_of(request: LLMRequest) -> int:
    first = request.user.split("MEMBERS:\n- ")[1].split("\n")[0]
    return next(i for i, ms in MEMBERS.items() if ms[0] == first)


class Model:
    """Handler keyed on request text; counters live under one lock."""

    def __init__(
        self,
        judge: Callable[[int], str] | None = None,
        refit: Callable[[int, int], str] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.judge = judge or (lambda n: _weak_reply([1, 3]))
        self.refit = refit or (lambda cid, n: GOOD[cid])
        self.delay = delay
        self.lock = threading.Lock()
        self.requests: list[LLMRequest] = []
        self.counts: dict[int, int] = {}
        self.judge_calls = 0

    def __call__(self, request: LLMRequest) -> str:
        with self.lock:
            self.requests.append(request)
            if request.user.startswith("NAMES TO CHECK:"):
                self.judge_calls += 1
                return self.judge(self.judge_calls)
            cid = _cluster_of(request)
            self.counts[cid] = self.counts.get(cid, 0) + 1
            number = self.counts[cid]
        if self.delay:
            time.sleep(self.delay * (10 - cid))
        return json.dumps({"name": self.refit(cid, number)})

    def refits(self) -> list[LLMRequest]:
        return [r for r in self.requests if "WHY:" in r.user]


def _run(model: Model, names=V4, *, rounds=1, workers=1):  # type: ignore[no-untyped-def]
    lines: list[str] = []
    client = FakeLLMClient(handler=model)
    result = fit.review_fit(
        client, CFG, fit.FitConfig(rounds=rounds), names, MEMBERS,
        sleep=_no_sleep, workers=workers, log=lines.append,
    )
    return result, lines, client


def test_rounds_zero_and_invalid_arguments() -> None:
    model = Model()
    for names in (V4, {}):
        result, lines, client = _run(model, names, rounds=0 if names else 1)
        assert result.names == names and lines == [] and client.calls == []
        assert (result.renamed, result.unresolved, result.rounds_run) == ({}, [], 0)
    for kwargs in ({"rounds": -1}, {"workers": 0}):
        with pytest.raises(ValueError):
            _run(model, **kwargs)
    assert model.requests == []


def test_no_weak_names_makes_no_rename_calls() -> None:
    model = Model(judge=lambda n: '{"weak": []}')
    result, lines, client = _run(model)
    assert model.judge_calls == 1 and model.refits() == [] and len(client.calls) == 1
    assert (result.renamed, result.unresolved, result.rounds_run) == ({}, [], 1)
    assert lines == [
        "fit review round 1/1: 0 of 10 name(s) judged weak",
        "fit review done: no weak names left",
    ]


def test_review_renames_weak_clusters() -> None:
    model = Model()
    result, lines, client = _run(model)
    first, second = model.refits()
    assert [_cluster_of(r) for r in model.refits()] == [1, 3]
    weak1, weak3 = fit.WeakFit(1, REASONS[1]), fit.WeakFit(3, REASONS[3])
    assert first.user == fit.build_refit_message(1, V4, MEMBERS, weak1, top_members=25)
    updated = {**V4, 1: GOOD[1]}
    assert second.user == fit.build_refit_message(3, updated, MEMBERS, weak3, top_members=25)
    assert f"- {GOOD[1]}" in second.user and "WHY: SECRET-REASON three" in second.user
    assert result.names == {**V4, **GOOD}
    assert (result.renamed, result.unresolved, result.rounds_run) == ({1: 1, 3: 1}, [], 1)
    assert len(client.calls) == 3 and result.judge_failures == 0


def test_review_independent_of_workers_and_finish_order() -> None:
    outcomes = []
    for workers in (2, 3, 8):
        model = Model(delay=0.01)
        result, lines, _ = _run(model, workers=workers)
        for request in model.refits():
            cid = _cluster_of(request)
            weak = fit.WeakFit(cid, REASONS[cid])
            assert request.user == fit.build_refit_message(cid, V4, MEMBERS, weak, top_members=25)
        outcomes.append((result.names, result.renamed, result.unresolved, lines))
    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert outcomes[0][1] == {1: 1, 3: 1}


def test_stubborn_model_keeps_name_and_is_not_reasked() -> None:
    for stubborn in (lambda cid, n: V4[cid], lambda cid, n: "Relational Database Systems"):
        model = Model(judge=lambda n: _weak_reply([1]), refit=stubborn)
        result, lines, client = _run(model, rounds=2)
        assert result.names == V4 and result.unresolved == [1] and result.renamed == {}
        assert model.counts == {1: 3} and model.judge_calls == 2
        assert len(client.calls) == 5
        assert lines[-1] == "fit review done: 1 cluster(s) still weak: 1"


def test_second_round_rejudges_and_renames_again() -> None:
    second = "Office Suite Support"
    model = Model(
        judge=lambda n: _weak_reply([1]) if n == 1 else _weak_reply([1, 3]),
        refit=lambda cid, n: {1: [GOOD[1], second], 3: [GOOD[3]]}[cid][n - 1],
    )
    result, lines, client = _run(model, rounds=2)
    assert result.names[1] == second and result.names[3] == GOOD[3]
    assert result.renamed == {1: 2, 3: 2} and result.rounds_run == 2
    assert model.judge_calls == 2 and lines[-1] == "fit review done: no weak names left"

    one = Model(judge=lambda n: _weak_reply([1]), refit=lambda cid, n: GOOD[cid])
    _, _, last = _run(one, rounds=1)
    assert one.judge_calls == 1 and len(last.calls) == 2


def test_second_round_stops_early() -> None:
    empty = Model(judge=lambda n: _weak_reply([1]) if n == 1 else '{"weak": []}')
    result, _, _ = _run(empty, rounds=3)
    assert empty.judge_calls == 2 and result.renamed == {1: 1} and result.rounds_run == 2
    stuck = Model(
        judge=lambda n: _weak_reply([1]), refit=lambda cid, n: V4[cid]
    )
    result, _, _ = _run(stuck, rounds=3)
    assert stuck.judge_calls == 2 and result.unresolved == [1] and result.rounds_run == 2


def test_judge_failure_stops_review() -> None:
    model = Model(judge=lambda n: '{"weak": 1}')
    result, lines, client = _run(model)
    assert result.names == V4 and result.judge_failures == 1 and len(client.calls) == 3
    assert lines == ["fit judge failed (replies rejected); skipping the fit review"]
    refused = FakeLLMClient(script=[PermanentLLMError("no")])
    gone_lines: list[str] = []
    gone = fit.review_fit(
        refused, CFG, fit.FitConfig(), V4, MEMBERS, sleep=_no_sleep, log=gone_lines.append
    )
    assert gone.judge_failures == 1 and gone.names == V4
    assert gone_lines == ["fit judge failed (request refused); skipping the fit review"]

    later = Model(judge=lambda n: _weak_reply([1]) if n == 1 else "nope")
    result, lines, _ = _run(later, rounds=2)
    assert result.names[1] == GOOD[1] and result.renamed == {1: 1} and result.judge_failures == 1
    assert lines[-2:] == [
        "fit judge failed (replies rejected); skipping the fit review",
        "fit review done: no weak names left",
    ]

    many = Model(judge=lambda n: _weak_reply(list(range(6))) if n == 1 else _weak_reply([1]))
    result, _, _ = _run(many)
    assert many.judge_calls == 2 and result.renamed == {1: 1}


def test_fatal_and_unexpected_errors_propagate() -> None:
    for error in (FatalLLMError("x"), RuntimeError("boom")):
        judge_fail = FakeLLMClient(script=[error])
        with pytest.raises(type(error)):
            fit.review_fit(judge_fail, CFG, fit.FitConfig(), V4, MEMBERS, sleep=_no_sleep)
        assert len(judge_fail.calls) == 1

        def handler(request: LLMRequest, error: BaseException = error) -> str:
            if request.user.startswith("NAMES TO CHECK:"):
                return _weak_reply([1, 3])
            raise error

        for workers in (1, 3):
            client = FakeLLMClient(handler=handler)
            with pytest.raises(type(error)):
                fit.review_fit(
                    client, CFG, fit.FitConfig(), V4, MEMBERS, sleep=_no_sleep, workers=workers
                )
            seen = len(client.calls)
            time.sleep(0.05)
            assert len(client.calls) == seen


def test_review_stats_sum() -> None:
    model = Model(refit=lambda cid, n: V4[cid] if n == 1 else GOOD[cid])
    result, _, client = _run(model, rounds=2)
    assert result.stats.calls == len(client.calls)
    assert result.stats.calls == model.judge_calls + sum(model.counts.values())


def test_progress_lines_formats_and_order() -> None:
    model = Model(refit=lambda cid, n: V4[cid] if cid == 3 else GOOD[cid])
    _, lines, _ = _run(model)
    assert lines[:3] == [
        "fit review round 1/1: 2 of 10 name(s) judged weak",
        'weak fit: cluster 1 "Desktop Applications"',
        'weak fit: cluster 3 "Development Tools"',
    ]
    assert lines[3] == (
        'renamed 1/2: cluster 1 "Desktop Applications" -> "Windows and Office Administration" '
        "(attempts 1)"
    )
    assert [line.split(":")[0] for line in lines[4:7]] == [
        f"rejected cluster 3 attempt {k}" for k in (1, 2, 3)
    ]
    assert lines[7] == 'kept 2/2: cluster 3 "Development Tools" (replies rejected, attempts 3)'
    assert lines[8] == "fit review done: 1 cluster(s) still weak: 3"
    assert len(lines) == 9
    assert fit.fit_done_message([]) == "fit review done: no weak names left"


def test_progress_lines_carry_no_reason_or_reply_text() -> None:
    def handler(request: LLMRequest) -> str:
        if request.user.startswith("NAMES TO CHECK:"):
            return _weak_reply([1, 3])
        return "LEAK-TEXT not json" if _cluster_of(request) == 3 else json.dumps({"name": GOOD[1]})

    lines: list[str] = []
    fit.review_fit(
        FakeLLMClient(handler=handler), CFG, fit.FitConfig(), V4, MEMBERS,
        sleep=_no_sleep, log=lines.append,
    )
    text = "\n".join(lines)
    assert "SECRET-REASON" not in text and "LEAK-TEXT" not in text
    assert GOOD[1] in text and "rejected cluster 3 attempt 1: not valid JSON" in text
    quiet = fit.review_fit(
        FakeLLMClient(handler=Model()), CFG, fit.FitConfig(), V4, MEMBERS,
        sleep=_no_sleep, log=None,
    )
    assert quiet.renamed == {1: 1, 3: 1}
