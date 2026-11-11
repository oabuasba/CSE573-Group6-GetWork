import threading
import time

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
from getwork.taxonomy import naming

pytestmark = pytest.mark.usefixtures("offline")

MEMBERS = {
    0: ["alpha1", "alpha2"],
    1: ["beta1", "beta2"],
    2: ["gamma1", "gamma2"],
    3: ["delta1", "delta2"],
    4: ["eps1", "eps2"],
    5: ["zeta1", "zeta2"],
}


def _cfg() -> naming.NameConfig:
    return naming.NameConfig(model="m", retry=RetryConfig(max_attempts=2))


def _no_sleep(_: float) -> None:
    return None


def _first_member(request: LLMRequest) -> str:
    return request.user.split("- ")[1].split("\n")[0]


def _own_name(request: LLMRequest) -> str:
    return '{"name": "Name ' + _first_member(request) + '"}'


def test_workers_must_be_positive() -> None:
    client = FakeLLMClient(handler=_own_name)
    with pytest.raises(ValueError):
        naming.name_clusters(client, _cfg(), MEMBERS, workers=0)
    assert client.calls == []


def test_draft_requests_have_no_used_names() -> None:
    client = FakeLLMClient(handler=_own_name)
    result = naming.name_clusters(client, _cfg(), MEMBERS, sleep=_no_sleep, workers=3)
    assert sorted(c.user for c in client.calls) == sorted(
        naming.build_name_message(shown, ()) for shown in MEMBERS.values()
    )
    assert result.names == {i: "Name " + m[0] for i, m in MEMBERS.items()}
    assert result.failed == [] and result.stats.calls == len(MEMBERS)


def test_result_and_log_independent_of_workers_and_finish_order() -> None:
    def handler(request: LLMRequest) -> str:
        member = _first_member(request)
        cluster_id = next(i for i, m in MEMBERS.items() if m[0] == member)
        time.sleep(0.002 * (len(MEMBERS) - cluster_id))
        if cluster_id == 2:
            return '{"name": "Other"}'
        name = "Shared" if cluster_id in (1, 4) and "USED" not in request.user else member
        return '{"name": "' + name.title() + ' Group"}'

    runs = []
    for workers in (2, 3, 8):
        lines: list[str] = []
        result = naming.name_clusters(
            FakeLLMClient(handler=handler), _cfg(), MEMBERS,
            sleep=_no_sleep, workers=workers, log=lines.append,
        )
        runs.append((result.names, result.failed, lines))
    assert runs[0] == runs[1] == runs[2]
    names, failed, lines = runs[0]
    assert failed == [2] and names[1] == "Shared Group" and names[4] == "Eps1 Group"
    final = [line for line in lines if not line.startswith("rejected")]
    counters = [line.split(":")[0].split()[-1] for line in final]
    assert counters == [f"{i}/6" for i in range(1, 7)]
    assert [line.split("cluster ")[1].split()[0] for line in final] == list("012345")
    other = "rejected cluster 2 attempt {}: cluster name must not be 'Other'"
    assert lines[2:6] == [
        other.format(1), other.format(2), other.format(3),
        "fallback 3/6: cluster 2 (replies rejected, attempts 3)",
    ]
    assert "rejected cluster 4 attempt 1: 'Shared Group' is already the name of another cluster; make it more specific" in lines


def _same_name_handler(request: LLMRequest) -> str:
    member = _first_member(request)
    if member in ("beta1", "delta1", "eps1"):
        if "USED NAMES" not in request.user:
            return '{"name": "Same Name"}' if member != "eps1" else '{"name": "same name"}'
        return '{"name": "Fresh ' + member + '"}'
    return _own_name(request)


def test_duplicate_draft_is_reasked_with_used_names() -> None:
    members = {i: MEMBERS[i] for i in range(5)}
    client = FakeLLMClient(handler=_same_name_handler)
    lines: list[str] = []
    result = naming.name_clusters(
        client, _cfg(), members, sleep=_no_sleep, workers=3, log=lines.append
    )
    assert result.names == {
        0: "Name alpha1", 1: "Same Name", 2: "Name gamma1", 3: "Fresh delta1", 4: "Fresh eps1",
    }
    reasks = {_first_member(c): c.user for c in client.calls if "USED NAMES" in c.user}
    assert set(reasks) == {"delta1", "eps1"}
    assert reasks["delta1"].endswith(
        "USED NAMES:\n- Name alpha1\n- Same Name\n- Name gamma1"
    )
    assert reasks["eps1"].endswith(
        "USED NAMES:\n- Name alpha1\n- Same Name\n- Name gamma1\n- Fresh delta1"
    )
    dup = "'Same Name' is already the name of another cluster; make it more specific"
    assert lines[lines.index(f"rejected cluster 3 attempt 1: {dup}") + 1] == (
        'named 4/5: cluster 3 -> "Fresh delta1" (attempts 2)'
    )
    assert result.stats.calls == len(client.calls) == 7


def test_duplicate_reask_exhausted_falls_back() -> None:
    def handler(request: LLMRequest) -> str:
        if _first_member(request) == "beta1":
            return '{"name": "Name alpha1"}'
        return _own_name(request)

    client = FakeLLMClient(handler=handler)
    lines: list[str] = []
    members = {0: MEMBERS[0], 1: MEMBERS[1], 2: MEMBERS[2]}
    result = naming.name_clusters(
        client, _cfg(), members, sleep=_no_sleep, workers=2, log=lines.append
    )
    assert result.failed == [1] and set(result.names) == {0, 2}
    beta = [c.user for c in client.calls if "beta1" in c.user]
    assert len(beta) == 4  # one draft plus three re-ask attempts
    assert "USED NAMES" not in beta[0] and "USED NAMES" in beta[1]
    assert "already the name of another cluster" in beta[3]
    dup = "'Name alpha1' is already the name of another cluster; make it more specific"
    assert "rejected cluster 1 attempt 1: " + dup in lines
    assert "fallback 2/3: cluster 1 (replies rejected, attempts 4)" in lines
    assert len([l for l in lines if l.startswith("rejected cluster 1")]) == 4


def test_member_list_draft_retried_in_worker() -> None:
    members = {0: ["JUnit", "Selenium"], 1: ["a1", "a2"], 2: ["b1", "b2"]}

    def handler(request: LLMRequest) -> str:
        member = _first_member(request)
        if member == "JUnit" and "rejected" not in request.user:
            return '{"name": "JUnit / Selenium"}'
        if member == "a1":
            return '{"name": "a1 / a2"}'
        return '{"name": "Fine ' + member + '"}'

    client = FakeLLMClient(handler=handler)
    result = naming.name_clusters(client, _cfg(), members, sleep=_no_sleep, workers=2)
    assert result.names == {0: "Fine JUnit", 2: "Fine b1"} and result.failed == [1]
    junit = [c.user for c in client.calls if c.user.startswith("MEMBERS:\n- JUnit")]
    first, second = junit
    assert second.startswith(first) and "member skills" in second and "USED NAMES" not in second
    assert sum("a1" in c.user for c in client.calls) == 3  # no re-ask after the third reply


def test_one_cluster_failure_does_not_stop_others() -> None:
    def handler(request: LLMRequest) -> str:
        member = _first_member(request)
        if member == "beta1":
            raise PermanentLLMError("no")
        if member == "gamma1":
            raise TransientLLMError("503")
        return _own_name(request)

    members = {i: MEMBERS[i] for i in (0, 1, 2, 3)}
    lines: list[str] = []
    result = naming.name_clusters(
        FakeLLMClient(handler=handler), _cfg(), members,
        sleep=_no_sleep, workers=3, log=lines.append,
    )
    assert result.failed == [1, 2] and set(result.names) == {0, 3}
    assert lines[1] == "fallback 2/4: cluster 1 (request refused, attempts 1)"
    assert lines[2] == "fallback 3/4: cluster 2 (retries exhausted, attempts 2)"


def _assert_work_stopped(client: FakeLLMClient) -> None:
    settled = len(client.calls)
    time.sleep(0.15)
    assert len(client.calls) == settled


@pytest.mark.parametrize("phase", ["draft", "reask"])
def test_fatal_error_propagates_and_stops_work(phase: str) -> None:
    members = {i: [f"m{i}a", f"m{i}b"] for i in range(40)}
    fatal = FatalLLMError("denied")

    def handler(request: LLMRequest) -> str:
        member = _first_member(request)
        if phase == "draft" and member == "m3a":
            raise fatal
        if phase == "reask" and "USED NAMES" in request.user:
            raise fatal
        time.sleep(0.002)
        return '{"name": "Same Name"}' if phase == "reask" else _own_name(request)

    client = FakeLLMClient(handler=handler)
    lines: list[str] = []
    with pytest.raises(FatalLLMError) as raised:
        naming.name_clusters(client, _cfg(), members, sleep=_no_sleep, workers=2, log=lines.append)
    assert raised.value is fatal
    _assert_work_stopped(client)
    logged = len(lines)
    time.sleep(0.05)
    assert len(lines) == logged and len(client.calls) < 2 * len(members)


def test_unexpected_exception_propagates() -> None:
    def handler(request: LLMRequest) -> str:
        if _first_member(request) == "beta1":
            raise RuntimeError("boom")
        time.sleep(0.002)
        return _own_name(request)

    client = FakeLLMClient(handler=handler)
    with pytest.raises(RuntimeError, match="boom"):
        naming.name_clusters(client, _cfg(), MEMBERS, sleep=_no_sleep, workers=2)
    _assert_work_stopped(client)


def test_rate_limit_handled_and_stats_summed() -> None:
    lock = threading.Lock()
    seen: set[str] = set()
    responses = {"rate_limited": 0}

    def handler(request: LLMRequest) -> str:
        member = _first_member(request)
        with lock:
            first = member not in seen
            seen.add(member)
            if first:
                responses["rate_limited"] += 1
        if first:
            raise RateLimitError("429")
        return _own_name(request)

    client = FakeLLMClient(handler=handler)
    cfg = naming.NameConfig(
        model="m", retry=RetryConfig(max_attempts=2, rate_limit_wait_s=0.001)
    )
    result = naming.name_clusters(client, cfg, MEMBERS, sleep=_no_sleep, workers=3)
    assert result.failed == [] and len(result.names) == len(MEMBERS)
    assert result.stats.rate_limited == responses["rate_limited"] == len(MEMBERS)
    assert result.stats.calls == len(client.calls) == 2 * len(MEMBERS)
