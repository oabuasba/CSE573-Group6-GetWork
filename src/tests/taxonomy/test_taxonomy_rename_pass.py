import threading
import time
from collections.abc import Callable, Mapping

import pytest

from getwork.llm.client import FatalLLMError
from getwork.llm.fake import FakeLLMClient
from getwork.llm.retry import CallStats, RetryConfig, Throttle
from getwork.taxonomy import naming, overlap

pytestmark = pytest.mark.usefixtures("offline")

CURRENT = {
    0: "Front-end Development", 3: "Development Tools", 5: "Software Development Tools",
    7: "Cloud Infrastructure",
}
MEMBERS = {i: [f"m{i}a", f"m{i}b", f"m{i}c"] for i in CURRENT}
ENV = naming.Env(
    client=FakeLLMClient(script=[]),
    cfg=naming.NameConfig(model="m", retry=RetryConfig(max_attempts=2)),
    throttle=Throttle(),
    sleep=lambda _: None,
)


def _asked(value: str | None, calls: int = 1, rejections: tuple[str, ...] = ()) -> naming.Asked[str]:
    reason = "" if value is not None else "replies rejected"
    return naming.Asked(value, reason, CallStats(calls=calls), rejections)


class Recorder:
    """An ask that answers from a table of lists and records what it saw (thread-safe)."""

    def __init__(self, answers: Mapping[int, list[naming.Asked[str]]], delay: float = 0.0) -> None:
        self.answers = {i: list(v) for i, v in answers.items()}
        self.delay = delay
        self.lock = threading.Lock()
        self.seen: list[tuple[int, dict[int, str]]] = []

    def __call__(self, cluster_id: int, names_seen: Mapping[int, str]) -> naming.Asked[str]:
        with self.lock:
            self.seen.append((cluster_id, dict(names_seen)))
            answer = self.answers[cluster_id].pop(0)
        if self.delay:
            time.sleep(self.delay * (10 - cluster_id))
        return answer


def _pass(ask: Callable[..., naming.Asked[str]], *, workers: int, todo=(5, 3), log=True):  # type: ignore[no-untyped-def]
    lines: list[str] = []
    result = overlap.run_rename_pass(
        ENV, todo, CURRENT, MEMBERS, ask, workers=workers, log=lines.append if log else None
    )
    return result, lines


def test_pass_sequential_sees_updated_names() -> None:
    ask = Recorder({3: [_asked("Build Tools")], 5: [_asked("Source Control")]})
    result, lines = _pass(ask, workers=1)
    assert [i for i, _ in ask.seen] == [3, 5]
    assert ask.seen[0][1] == CURRENT
    assert ask.seen[1][1] == {**CURRENT, 3: "Build Tools"}
    assert result.names == {**CURRENT, 3: "Build Tools", 5: "Source Control"}
    assert (result.renamed, result.kept) == ([3, 5], [])
    assert lines == [
        'renamed 1/2: cluster 3 "Development Tools" -> "Build Tools" (attempts 1)',
        'renamed 2/2: cluster 5 "Software Development Tools" -> "Source Control" (attempts 1)',
    ]
    assert CURRENT[3] == "Development Tools"


def test_pass_parallel_independent_of_workers() -> None:
    outcomes = []
    for workers in (2, 3, 8):
        ask = Recorder(
            {3: [_asked("Build Tools")], 5: [_asked("Source Control")]}, delay=0.01
        )
        result, lines = _pass(ask, workers=workers)
        assert all(seen == CURRENT for _, seen in ask.seen)
        assert sorted(i for i, _ in ask.seen) == [3, 5]
        outcomes.append((result.names, result.renamed, result.kept, lines))
    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert outcomes[0][1] == [3, 5]


def test_pass_drafts_after_first_rename_still_see_snapshot() -> None:
    lines: list[str] = []
    late_seen: list[dict[int, str]] = []

    def ask(cluster_id: int, names_seen: Mapping[int, str]) -> naming.Asked[str]:
        if cluster_id == 7:
            deadline = time.monotonic() + 5
            while not lines and time.monotonic() < deadline:
                time.sleep(0.005)
            assert lines, "the first rename was never accepted"
            late_seen.append(dict(names_seen))
        elif cluster_id == 5:
            time.sleep(0.05)
        return _asked({0: "A Tool", 3: "Build Tools", 5: "Source Control", 7: "Hosting"}[cluster_id])

    todo = (3, 5, 7)
    result = overlap.run_rename_pass(
        ENV, todo, CURRENT, MEMBERS, ask, workers=2, log=lines.append
    )
    assert late_seen == [CURRENT]
    assert result.renamed == [3, 5, 7]


def test_pass_collision_reasked_or_kept() -> None:
    ask = Recorder({
        3: [_asked("Build Tools")],
        5: [_asked("Build Tools"), _asked("Source Control", calls=2, rejections=("bad",))],
    })
    result, lines = _pass(ask, workers=2)
    again = [seen for i, seen in ask.seen if i == 5][1]
    assert again[3] == "Build Tools"
    assert result.names[3] == "Build Tools" and result.names[5] == "Source Control"
    assert lines[1].startswith("rejected cluster 5 attempt 1: ")
    assert lines[2] == "rejected cluster 5 attempt 2: bad"
    assert lines[3] == (
        'renamed 2/2: cluster 5 "Software Development Tools" -> "Source Control" (attempts 3)'
    )

    overlapping = Recorder({
        3: [_asked("Build Tools")],
        5: [_asked("Build Tools Suite"), _asked(None, calls=3, rejections=("a", "b", "c"))],
    })
    result, lines = _pass(overlapping, workers=3)
    assert (result.renamed, result.kept) == ([3], [5])
    assert result.names[5] == CURRENT[5]
    assert lines[-1] == (
        'kept 2/2: cluster 5 "Software Development Tools" (replies rejected, attempts 4)'
    )


def test_pass_failed_request_kept_and_logged() -> None:
    for workers in (1, 2):
        ask = Recorder({
            3: [_asked(None, calls=3, rejections=("x", "y", "z"))],
            5: [_asked("Source Control")],
        })
        result, lines = _pass(ask, workers=workers)
        assert (result.renamed, result.kept) == ([5], [3])
        assert result.names[3] == CURRENT[3]
        assert lines[:3] == [f"rejected cluster 3 attempt {k}: {t}" for k, t in enumerate("xyz", 1)]
        assert lines[3] == 'kept 1/2: cluster 3 "Development Tools" (replies rejected, attempts 3)'
        assert lines[4].startswith("renamed 2/2: cluster 5 ")


def test_pass_errors_and_stats() -> None:
    ask = Recorder({3: [_asked("Build Tools", calls=2)], 5: [_asked("Source Control", calls=3)]})
    result, lines = _pass(ask, workers=1, log=False)
    assert lines == [] and result.stats.calls == 5
    for workers in (1, 2):
        for error in (FatalLLMError("x"), RuntimeError("boom")):

            def failing(cluster_id: int, names_seen: Mapping[int, str], error=error):  # type: ignore[no-untyped-def]
                raise error

            with pytest.raises(type(error)):
                _pass(failing, workers=workers)
    with pytest.raises(ValueError):
        _pass(ask, workers=0)
