from pathlib import Path
from typing import Any

import pytest

from getwork.llm.retry import CallStats
from getwork.structure import runstate, schema
from getwork.structure.extract import ExtractConfig, Outcome
from getwork.structure.schema import Failure

pytestmark = pytest.mark.usefixtures("offline")

TOP_KEYS = {
    "source", "model", "prompt_version", "schema_version", "status", "abort_reason",
    "total_in_source", "selected", "already_done", "attempted", "succeeded", "failed",
    "not_attempted", "done_total", "failures_total", "failures_by_stage", "calls",
    "retries", "rate_limited", "truncated_inputs", "config",
}  # fmt: skip
CONFIG_KEYS = {
    "limit", "sample", "seed", "temperature", "max_tokens", "reasoning_effort",
    "use_response_schema", "max_validation_attempts", "max_input_chars", "max_attempts",
    "max_rate_limit_waits", "max_consecutive_failures",
}  # fmt: skip


def _cfg(**kw: Any) -> runstate.RunConfig:
    return runstate.RunConfig(
        source="postings", in_dir=Path("in"), out_dir=Path("out"),
        extract=ExtractConfig(model="m1"), **kw,
    )  # fmt: skip


def _ok(id_: str, stats: CallStats, truncated: bool = False) -> Outcome:
    return Outcome(id=id_, record={"id": id_}, failure=None, stats=stats, truncated=truncated)


def _bad(id_: str, stage: str, stats: CallStats) -> Outcome:
    f = Failure(id=id_, stage=stage, error_type="E", message="m", attempts=1)
    return Outcome(id=id_, record=None, failure=f, stats=stats, truncated=False)


def test_build_summary_key_set_and_values() -> None:
    counters = runstate.Counters(attempted=3, succeeded=2, failed=1, calls=5)
    counters.failures_by_stage["permanent"] = 1
    facts = runstate.RunFacts(
        total_in_source=100, selected=10, already_done=2, todo=8, done_total=4, failures_total=1
    )
    s = runstate.build_summary(
        cfg=_cfg(workers=7, progress_every=9, allow_mixed=True, limit=10),
        counters=counters, facts=facts, status="aborted", abort_reason="boom",
    )  # fmt: skip
    assert set(s) == TOP_KEYS
    assert set(s["config"]) == CONFIG_KEYS
    assert "workers" not in s["config"]
    assert s["not_attempted"] == 5
    assert s["model"] == "m1"
    assert s["schema_version"] == schema.SCHEMA_VERSION
    assert s["status"] == "aborted" and s["abort_reason"] == "boom"
    assert s["config"]["limit"] == 10 and s["config"]["max_consecutive_failures"] == 25
    assert s["failures_by_stage"] == {"validation": 0, "transient": 0, "rate_limit": 0, "permanent": 1}


def test_counters_add_accumulates() -> None:
    c = runstate.Counters()
    assert c.failures_by_stage == dict.fromkeys(schema.FAILURE_STAGES, 0)
    c.add(_ok("a", CallStats(calls=1)))
    c.add(_ok("b", CallStats(calls=3, retries=2, rate_limited=1), truncated=True))
    c.add(_bad("c", "validation", CallStats(calls=3)))
    c.add(_bad("d", "validation", CallStats(calls=1, retries=1)))
    assert (c.attempted, c.succeeded, c.failed) == (4, 2, 2)
    assert (c.calls, c.retries, c.rate_limited, c.truncated_inputs) == (8, 3, 1, 1)
    assert c.failures_by_stage["validation"] == 2
    assert c.failures_by_stage["permanent"] == 0


def test_counters_add_unknown_stage_raises_and_changes_nothing() -> None:
    c = runstate.Counters()
    with pytest.raises(ValueError, match="bogus"):
        c.add(_bad("a", "bogus", CallStats(calls=2)))
    assert c == runstate.Counters()


def test_exit_code_for_statuses() -> None:
    clean, dirty = runstate.Counters(), runstate.Counters(failed=2)
    assert runstate.exit_code_for(status="complete", counters=clean) == 0
    assert runstate.exit_code_for(status="complete", counters=dirty) == 1
    assert runstate.exit_code_for(status="aborted", counters=clean) == 3
    assert runstate.exit_code_for(status="interrupted", counters=dirty) == 130
