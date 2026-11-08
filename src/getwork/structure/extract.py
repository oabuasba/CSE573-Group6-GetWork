"""structure_one: build the prompt, call the model, validate, and regenerate with error feedback."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from getwork.llm.client import (
    MAX_ERROR_CHARS,
    FatalLLMError,
    LLMClient,
    LLMRequest,
    PermanentLLMError,
    describe_exception,
    redact_secrets,
)
from getwork.llm.retry import (
    CallStats,
    RetriesExhausted,
    RetryConfig,
    Throttle,
    call_with_retry,
)
from getwork.structure.prompt import (
    PROMPT_VERSION,
    build_retry_message,
    build_user_message,
    load_system_prompt,
)
from getwork.structure.schema import (
    Failure,
    OutputValidationError,
    parse_model_output,
    response_schema,
    to_record,
)


@dataclass(frozen=True)
class SourceDoc:
    id: str
    source: str  # "resumes" | "postings"
    title: str | None  # postings only
    text: str  # text_clean


@dataclass(frozen=True)
class ExtractConfig:
    model: str
    prompt_version: str = PROMPT_VERSION
    temperature: float | None = 0.0
    max_tokens: int = 4096
    reasoning_effort: str | None = "none"
    use_response_schema: bool = True
    max_validation_attempts: int = 3
    max_input_chars: int = 48000
    retry: RetryConfig = RetryConfig()


@dataclass
class Outcome:
    id: str
    record: dict | None  # set on success
    failure: Failure | None  # set on failure; exactly one of record/failure is set
    stats: CallStats
    truncated: bool  # input exceeded max_input_chars and was cut


def _build_request(cfg: ExtractConfig, system: str, user: str, schema: dict | None) -> LLMRequest:
    return LLMRequest(
        system=system,
        user=user,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        reasoning_effort=cfg.reasoning_effort,
        response_schema=schema,
    )


def _failure(
    doc_id: str, stage: str, error_type: str, message: str, secrets: Sequence[str], attempts: int
) -> Failure:
    return Failure(
        id=doc_id,
        stage=stage,
        error_type=error_type,
        message=redact_secrets(message, secrets)[:MAX_ERROR_CHARS],
        attempts=attempts,
    )


def _run_attempts(
    client: LLMClient,
    cfg: ExtractConfig,
    throttle: Throttle,
    doc: SourceDoc,
    user: str,
    stats: CallStats,
    sleep: Callable[[float], None],
) -> dict | str:
    """Return the record, or the last validation error text when attempts run out."""
    system = load_system_prompt(cfg.prompt_version)
    schema = response_schema() if cfg.use_response_schema else None
    last_error = ""
    for attempt in range(1, cfg.max_validation_attempts + 1):
        prompt = user if attempt == 1 else build_retry_message(user, last_error)
        request = _build_request(cfg, system, prompt, schema)
        reply = call_with_retry(client, request, cfg.retry, throttle, sleep, stats)
        try:
            output = parse_model_output(reply)
        except OutputValidationError as exc:
            last_error = str(exc)
            continue
        return to_record(output, doc.id, doc.source, cfg.model, cfg.prompt_version)
    return last_error


def structure_one(
    client: LLMClient,
    cfg: ExtractConfig,
    throttle: Throttle,
    doc: SourceDoc,
    secrets: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> Outcome:
    """FatalLLMError propagates; every other Exception becomes an Outcome (see Instructions)."""
    truncated = len(doc.text) > cfg.max_input_chars
    user = build_user_message(doc.source, doc.title, doc.text[: cfg.max_input_chars])
    stats = CallStats()

    def fail(stage: str, error_type: str, message: str) -> Outcome:
        failure = _failure(doc.id, stage, error_type, message, secrets, stats.calls)
        return Outcome(id=doc.id, record=None, failure=failure, stats=stats, truncated=truncated)

    try:
        result = _run_attempts(client, cfg, throttle, doc, user, stats, sleep)
    except RetriesExhausted as exc:
        return fail(exc.kind, type(exc.last_error).__name__, str(exc.last_error))
    except FatalLLMError:
        raise
    except PermanentLLMError as exc:
        return fail("permanent", type(exc).__name__, str(exc))
    except Exception as exc:
        return fail("permanent", type(exc).__name__, describe_exception(exc, secrets))
    if isinstance(result, str):
        return fail("validation", "OutputValidationError", result)
    return Outcome(id=doc.id, record=result, failure=None, stats=stats, truncated=truncated)
