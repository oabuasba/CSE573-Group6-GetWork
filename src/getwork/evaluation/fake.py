"""Offline judge test double: mapped or computed verdicts, recorded calls."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping

from getwork.evaluation.judges import JudgeError, JudgeInfo, PairText, Verdict


class FakeJudge:
    def __init__(
        self,
        info: JudgeInfo,
        verdicts: Mapping[str, Verdict] | Callable[[PairText], Verdict],
        *,
        fail: Mapping[str, JudgeError] | None = None,
        truncated: bool = False,
    ) -> None:
        self._info = info
        self._verdicts = verdicts
        self._fail = dict(fail or {})
        self._truncated = truncated
        self._lock = threading.Lock()
        self.calls: list[str] = []

    @property
    def info(self) -> JudgeInfo:
        return self._info

    def judge(self, pair: PairText) -> tuple[Verdict, bool]:
        with self._lock:
            self.calls.append(pair.pair_id)
        if pair.pair_id in self._fail:
            raise self._fail[pair.pair_id]
        if callable(self._verdicts):
            return self._verdicts(pair), self._truncated
        return self._verdicts[pair.pair_id], self._truncated
