"""Run loop: resume from the checkpoint, process documents in parallel, write all outputs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace

import pandas as pd

from getwork.llm.client import FatalLLMError, LLMClient, describe_exception, redact_secrets
from getwork.llm.retry import CallStats, Throttle
from getwork.structure import inputs, pool, runstate, store
from getwork.structure.extract import Outcome, SourceDoc, structure_one
from getwork.structure.runstate import Counters, RunConfig, RunFacts, RunResult
from getwork.structure.schema import SCHEMA_VERSION, Failure


class ProvenanceMismatchError(RuntimeError):
    pass


@dataclass
class _Progress:
    """Mutable results of the consuming loop; the finalizer reads it after the loop ends."""

    done: dict[str, dict]
    counters: Counters = field(default_factory=Counters)
    failures: list[Failure] = field(default_factory=list)
    attempted_ids: set[str] = field(default_factory=set)
    consecutive_failures: int = 0
    status: str = "complete"
    abort_reason: str | None = None  # already redacted


def _load_docs(df: pd.DataFrame, source: str) -> dict[str, SourceDoc]:
    titles = df["title"] if "title" in df.columns else [None] * len(df)
    return {
        id_: SourceDoc(id=id_, source=source, title=title, text=text)
        for id_, title, text in zip(df["id"], titles, df["text_clean"])
    }


def _record_outcome(outcome: Outcome, progress: _Progress, writer: store.CheckpointWriter) -> None:
    progress.counters.add(outcome)
    progress.attempted_ids.add(outcome.id)
    if outcome.record is not None:
        progress.done[outcome.id] = outcome.record
        writer.append(outcome.record)
    else:
        progress.failures.append(outcome.failure)


def _progress_message(progress: _Progress, todo_count: int) -> str:
    c = progress.counters
    return f"processed {c.attempted}/{todo_count} (succeeded {c.succeeded}, failed {c.failed})"


@dataclass(frozen=True)
class _LoopState:
    """Collaborators the consuming loop needs besides the results and progress."""

    writer: store.CheckpointWriter
    progress_every: int
    log: Callable[[str], None]
    todo_count: int
    secrets: Sequence[str]
    max_consecutive_failures: int


def _exception_outcome(doc: SourceDoc, exc: Exception, secrets: Sequence[str]) -> Outcome:
    """A worker exception that is not an LLMError becomes a permanent record failure."""
    failure = Failure(
        id=doc.id,
        stage="permanent",
        error_type=type(exc).__name__,
        message=describe_exception(exc, secrets),
        attempts=0,
    )
    return Outcome(id=doc.id, record=None, failure=failure, stats=CallStats(), truncated=False)


def _consume(
    results: Iterator[tuple[SourceDoc, Outcome | BaseException]],
    progress: _Progress,
    loop: _LoopState,
) -> None:
    """Record outcomes until the results end, a fatal error arrives, or the breaker trips."""
    for doc, result in results:
        if isinstance(result, FatalLLMError):
            progress.abort_reason = f"fatal API error (status {result.status}): {result}"
            return
        if isinstance(result, Exception):
            result = _exception_outcome(doc=doc, exc=result, secrets=loop.secrets)
        elif isinstance(result, BaseException):
            raise result
        _record_outcome(outcome=result, progress=progress, writer=loop.writer)
        if progress.counters.attempted % loop.progress_every == 0:
            message = _progress_message(progress=progress, todo_count=loop.todo_count)
            loop.log(message)
        if _breaker_tripped(outcome=result, progress=progress, limit=loop.max_consecutive_failures):
            progress.abort_reason = f"{progress.consecutive_failures} consecutive record failures"
            return


def _breaker_tripped(outcome: Outcome, progress: _Progress, limit: int) -> bool:
    if outcome.failure is None:
        progress.consecutive_failures = 0
    else:
        progress.consecutive_failures += 1
    return progress.consecutive_failures >= limit


def _finalize(
    cfg: RunConfig, paths: store.OutputPaths, progress: _Progress, facts: RunFacts
) -> dict:
    """Rewrite checkpoint and Parquet sorted, merge failures, and write the summary."""
    store.write_final(paths, progress.done)
    merged = store.merge_failures(
        previous=store.read_failures(paths.failures),
        current=progress.failures,
        attempted=progress.attempted_ids,
        done=progress.done.keys(),
    )
    store.write_failures(paths.failures, merged)
    final_facts = replace(facts, done_total=len(progress.done), failures_total=len(merged))
    summary = runstate.build_summary(
        cfg=cfg,
        counters=progress.counters,
        facts=final_facts,
        status=progress.status,
        abort_reason=progress.abort_reason,
    )
    store.write_summary(paths.summary, summary)
    return summary


@dataclass(frozen=True)
class _Plan:
    """What step 1-4 decided: the documents still to process and the counts around them."""

    todo_docs: list[SourceDoc]
    done: dict[str, dict]
    facts: RunFacts


def _check_provenance(cfg: RunConfig, paths: store.OutputPaths) -> None:
    """Refuse to mix records made with different model, prompt or schema versions."""
    if cfg.allow_mixed:
        return
    expected = {
        "model": cfg.extract.model,
        "prompt_version": cfg.extract.prompt_version,
        "schema_version": SCHEMA_VERSION,
    }
    differing: set[str] = set()
    for record in store.read_checkpoint(paths.jsonl, repair=False).values():
        for key, value in expected.items():
            if record.get(key) != value:
                differing.add(f"{key}={record.get(key)!r} (expected {value!r})")
    if differing:
        raise ProvenanceMismatchError(
            "checkpoint provenance differs: " + "; ".join(sorted(differing))
        )


def _prepare(cfg: RunConfig, paths: store.OutputPaths, docs: dict[str, SourceDoc]) -> _Plan:
    _check_provenance(cfg=cfg, paths=paths)
    done = store.read_checkpoint(paths.jsonl, repair=True)
    universe = inputs.select_universe(
        all_ids=list(docs), limit=cfg.limit, sample=cfg.sample, seed=cfg.seed
    )
    todo_docs = [docs[i] for i in universe if i not in done]
    facts = RunFacts(
        total_in_source=len(docs),
        selected=len(universe),
        already_done=len(universe) - len(todo_docs),
        todo=len(todo_docs),
        done_total=len(done),
        failures_total=0,
    )
    return _Plan(todo_docs=todo_docs, done=done, facts=facts)


def _make_work(
    client: LLMClient,
    cfg: RunConfig,
    throttle: Throttle,
    secrets: Sequence[str],
    sleep: Callable[[float], None],
) -> Callable[[SourceDoc], Outcome]:
    def work(doc: SourceDoc) -> Outcome:
        return structure_one(
            client=client,
            cfg=cfg.extract,
            throttle=throttle,
            doc=doc,
            secrets=secrets,
            sleep=sleep,
        )

    return work


def _settle(progress: _Progress, stopped: bool, secrets: Sequence[str]) -> None:
    """Fix status and the redacted abort_reason before the finalizer reads them."""
    if progress.abort_reason is not None:
        progress.status = "aborted"
        progress.abort_reason = redact_secrets(progress.abort_reason, secrets)
    elif stopped or progress.status == "interrupted":
        progress.status = "interrupted"


def run(
    cfg: RunConfig,
    client: LLMClient,
    *,
    secrets: Sequence[str] = (),
    stop: threading.Event | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = lambda message: None,
) -> RunResult:
    paths = store.output_paths(source=cfg.source, out_dir=cfg.out_dir)
    df = inputs.load_inputs(source=cfg.source, in_dir=cfg.in_dir)
    docs = _load_docs(df=df, source=cfg.source)
    writer = store.CheckpointWriter(paths.jsonl)
    try:
        plan = _prepare(cfg=cfg, paths=paths, docs=docs)
    except BaseException:
        writer.close()
        raise
    progress = _Progress(done=plan.done)
    stop = stop or threading.Event()
    abort = threading.Event()
    work = _make_work(
        client=client,
        cfg=cfg,
        throttle=Throttle(clock, sleep),
        secrets=secrets,
        sleep=sleep,
    )
    loop = _LoopState(
        writer=writer,
        progress_every=cfg.progress_every,
        log=lambda message: log(redact_secrets(message, secrets)),
        todo_count=len(plan.todo_docs),
        secrets=secrets,
        max_consecutive_failures=cfg.max_consecutive_failures,
    )
    try:
        results = pool.run_pool(
            items=plan.todo_docs, fn=work, workers=cfg.workers, stop=stop, abort=abort
        )
        _consume(results=results, progress=progress, loop=loop)
    except KeyboardInterrupt:
        progress.status = "interrupted"
    finally:
        abort.set()
        writer.close()
        _settle(progress=progress, stopped=stop.is_set(), secrets=secrets)
        summary = _finalize(cfg=cfg, paths=paths, progress=progress, facts=plan.facts)
    return RunResult(
        exit_code=runstate.exit_code_for(progress.status, progress.counters), summary=summary
    )
