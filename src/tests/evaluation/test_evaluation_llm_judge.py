from __future__ import annotations

import json
import threading

import pytest

from getwork.evaluation import judges as ev_judges
from getwork.evaluation.fake import FakeJudge
from getwork.evaluation.judges import JudgeError, JudgeFatalError, JudgeInfo, PairText, Verdict
from getwork.evaluation.llm_judge import LLMJudge
from getwork.llm.client import (
    FatalLLMError,
    LLMRequest,
    PermanentLLMError,
    RateLimitError,
    TransientLLMError,
)
from getwork.llm.fake import FakeLLMClient
from getwork.llm.retry import RetryConfig

pytestmark = pytest.mark.usefixtures("offline")

INFO = JudgeInfo(id="judge-a", model="m-a", family="fam", backend="voyager")
GOOD = json.dumps({"label": 1, "seniority_mismatch": True, "rationale": "Some gaps."})
FAST = RetryConfig(max_attempts=2, backoff_base_s=0)


def _pair(pair_id: str = "p-1") -> PairText:
    return PairText(pair_id=pair_id, resume_text="my resume", posting_text="the job")


def _judge(client: FakeLLMClient, **kw: object) -> LLMJudge:
    kw.setdefault("retry", FAST)
    return LLMJudge(client, INFO, ev_judges.load_rubric(), sleep=lambda s: None, **kw)


def test_request_shape_and_verdict() -> None:
    client = FakeLLMClient(script=[GOOD])
    verdict, truncated = _judge(client).judge(_pair())
    assert verdict == Verdict(label=1, seniority_mismatch=True, rationale="Some gaps.")
    assert not truncated
    (request,) = client.calls
    assert request.model == "m-a" and request.temperature == 0
    assert request.response_schema == ev_judges.VERDICT_SCHEMA
    assert request.system == ev_judges.load_rubric().text
    assert "RESUME:" in request.user and "p-1" not in request.user
    plain = FakeLLMClient(script=[GOOD])
    _judge(plain, use_schema=False).judge(_pair())
    assert plain.calls[0].response_schema is None


def test_invalid_reply_retried_then_error() -> None:
    client = FakeLLMClient(script=["nope", "still nope", "{}"])
    with pytest.raises(JudgeError) as info:
        _judge(client).judge(_pair())
    assert info.value.reason == "invalid_reply"
    assert len(client.calls) == 3
    assert "rejected" in client.calls[1].user
    recovered = FakeLLMClient(script=["nope", GOOD])
    assert _judge(recovered).judge(_pair())[0].label == 1


def test_permanent_transient_fatal_mapping() -> None:
    cases = [
        (PermanentLLMError("bad", status=400), "rejected"),
        (TransientLLMError("down", status=503), "unavailable"),
        (RateLimitError("slow", status=429), "unavailable"),
    ]
    for error, reason in cases:
        client = FakeLLMClient(handler=_raise(error))
        retry = RetryConfig(max_attempts=2, backoff_base_s=0, max_rate_limit_waits=0)
        with pytest.raises(JudgeError) as info:
            _judge(client, retry=retry).judge(_pair())
        assert info.value.reason == reason
    transient = FakeLLMClient(handler=_raise(TransientLLMError("down", status=503)))
    with pytest.raises(JudgeError):
        _judge(transient).judge(_pair())
    assert len(transient.calls) == 2
    fatal = FakeLLMClient(handler=_raise(FatalLLMError("key", status=401)))
    with pytest.raises(JudgeFatalError):
        _judge(fatal).judge(_pair())


def _raise(error: Exception):
    def handler(request: LLMRequest) -> str:
        raise error

    return handler


def test_reasoning_none_vs_omitted() -> None:
    for effort in (None, "none"):
        client = FakeLLMClient(script=[GOOD])
        _judge(client, reasoning_effort=effort).judge(_pair())
        assert client.calls[0].reasoning_effort == effort
        assert client.calls[0].response_schema is not None


def _schema_rejecting(request: LLMRequest) -> str:
    if request.response_schema is not None:
        raise PermanentLLMError("schema", status=400)
    return GOOD


def test_schema_rejected_falls_back_once_and_sticks() -> None:
    client = FakeLLMClient(handler=_schema_rejecting)
    judge = _judge(client)
    assert not judge.schema_fallback
    assert judge.judge(_pair("p-1"))[0].label == 1
    assert [c.response_schema is not None for c in client.calls] == [True, False]
    assert judge.schema_fallback
    judge.judge(_pair("p-2"))
    assert len(client.calls) == 3 and client.calls[2].response_schema is None


def test_unrelated_permanent_error_does_not_disable_schema() -> None:
    client = FakeLLMClient(handler=_raise(PermanentLLMError("too long", status=413)))
    judge = _judge(client)
    for pair_id in ("p-1", "p-2"):
        with pytest.raises(JudgeError) as info:
            judge.judge(_pair(pair_id))
        assert info.value.reason == "rejected"
    assert not judge.schema_fallback
    assert [c.response_schema is not None for c in client.calls] == [True, False, True, False]


def test_malformed_schema_reply_regenerates_before_fallback() -> None:
    client = FakeLLMClient(script=["bad", "bad", GOOD])
    judge = _judge(client)
    assert judge.judge(_pair())[0].label == 1
    assert all(c.response_schema is not None for c in client.calls)
    assert not judge.schema_fallback


def test_fake_judge() -> None:
    verdict = Verdict(label=2, seniority_mismatch=False, rationale="ok")
    boom = JudgeError("rejected")
    ids = [f"p-{i}" for i in range(40)]
    fake = FakeJudge(INFO, {i: verdict for i in ids} | {"p-bad": verdict}, fail={"p-bad": boom},
                     truncated=True)
    assert fake.info == INFO
    assert fake.judge(_pair("p-3")) == (verdict, True)
    with pytest.raises(JudgeError) as info:
        fake.judge(_pair("p-bad"))
    assert info.value is boom
    computed = FakeJudge(INFO, lambda pair: Verdict(len(pair.pair_id) % 3, False, pair.pair_id))
    assert computed.judge(_pair("p-1"))[0].rationale == "p-1" and not computed.judge(_pair())[1]
    threaded = FakeJudge(INFO, lambda pair: verdict)
    workers = [
        threading.Thread(target=lambda: [threaded.judge(_pair(i)) for i in ids]) for _ in range(8)
    ]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    assert len(threaded.calls) == 8 * len(ids)
