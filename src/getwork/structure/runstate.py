"""Run configuration, counters, summary builder and exit codes for a structuring run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from getwork.structure import schema
from getwork.structure.extract import ExtractConfig, Outcome

EXIT_COMPLETE = 0
EXIT_WITH_FAILURES = 1
EXIT_ABORTED = 3
EXIT_INTERRUPTED = 130


@dataclass(frozen=True)
class RunConfig:
    source: str
    in_dir: Path
    out_dir: Path
    extract: ExtractConfig
    workers: int = 4
    limit: int | None = None
    sample: int | None = None
    seed: int = 0
    max_consecutive_failures: int = 25
    allow_mixed: bool = False
    progress_every: int = 50


@dataclass(frozen=True)
class RunResult:
    exit_code: int  # 0 complete and no failures; 1 complete with failures; 3 aborted; 130 interrupted
    summary: dict  # the dict written to paths.summary


def _zero_stages() -> dict[str, int]:
    return {stage: 0 for stage in schema.FAILURE_STAGES}


@dataclass
class Counters:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    calls: int = 0
    retries: int = 0
    rate_limited: int = 0
    truncated_inputs: int = 0
    failures_by_stage: dict[str, int] = field(default_factory=_zero_stages)

    def add(self, outcome: Outcome) -> None:
        """Raises ValueError, before changing any count, if the failure stage is unknown."""
        if outcome.failure is not None and outcome.failure.stage not in self.failures_by_stage:
            raise ValueError(f"unknown failure stage: {outcome.failure.stage!r}")
        self.attempted += 1
        if outcome.failure is None:
            self.succeeded += 1
        else:
            self.failed += 1
            stage = outcome.failure.stage
            self.failures_by_stage[stage] += 1
        self.calls += outcome.stats.calls
        self.retries += outcome.stats.retries
        self.rate_limited += outcome.stats.rate_limited
        if outcome.truncated:
            self.truncated_inputs += 1


@dataclass(frozen=True)
class RunFacts:
    total_in_source: int
    selected: int
    already_done: int
    todo: int
    done_total: int
    failures_total: int


def build_summary(
    cfg: RunConfig, counters: Counters, facts: RunFacts, status: str, abort_reason: str | None
) -> dict:
    """Summary dict per the Output contract; workers, progress_every, allow_mixed stay out of config."""
    ex = cfg.extract
    return {
        "source": cfg.source,
        "model": ex.model,
        "prompt_version": ex.prompt_version,
        "schema_version": schema.SCHEMA_VERSION,
        "status": status,
        "abort_reason": abort_reason,
        "total_in_source": facts.total_in_source,
        "selected": facts.selected,
        "already_done": facts.already_done,
        "attempted": counters.attempted,
        "succeeded": counters.succeeded,
        "failed": counters.failed,
        "not_attempted": facts.todo - counters.attempted,
        "done_total": facts.done_total,
        "failures_total": facts.failures_total,
        "failures_by_stage": {s: counters.failures_by_stage.get(s, 0) for s in schema.FAILURE_STAGES},
        "calls": counters.calls,
        "retries": counters.retries,
        "rate_limited": counters.rate_limited,
        "truncated_inputs": counters.truncated_inputs,
        "config": {
            "limit": cfg.limit,
            "sample": cfg.sample,
            "seed": cfg.seed,
            "temperature": ex.temperature,
            "max_tokens": ex.max_tokens,
            "reasoning_effort": ex.reasoning_effort,
            "use_response_schema": ex.use_response_schema,
            "max_validation_attempts": ex.max_validation_attempts,
            "max_input_chars": ex.max_input_chars,
            "max_attempts": ex.retry.max_attempts,
            "max_rate_limit_waits": ex.retry.max_rate_limit_waits,
            "max_consecutive_failures": cfg.max_consecutive_failures,
        },
    }


def exit_code_for(status: str, counters: Counters) -> int:
    if status == "aborted":
        return EXIT_ABORTED
    if status == "interrupted":
        return EXIT_INTERRUPTED
    return EXIT_COMPLETE if counters.failed == 0 else EXIT_WITH_FAILURES
