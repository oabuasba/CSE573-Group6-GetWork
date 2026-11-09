"""The Relabeler protocol and its LLM-backed implementation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from getwork.llm.client import LLMClient
from getwork.llm.retry import CallStats, RetryConfig
from getwork.relabel import reextract, unseen
from getwork.relabel.cache import Stamp, StampedCache
from getwork.taxonomy.model import Taxonomy

UNSEEN_CACHE_FILE = "unseen_assignments.json"
REEXTRACT_CACHE_FILE = "reextract.json"


class Relabeler(Protocol):
    mode: str  # "llm"
    model: str | None
    backend: str | None
    prompt_version: str | None

    def assign_unseen(self, skills: Sequence[str]) -> Mapping[str, str]:
        """skill -> cluster; skills that could not be resolved are absent."""
        ...

    def reextract(
        self, doc_id: str, text: str, skills: Sequence[str]
    ) -> list[tuple[str, str]] | None: ...


class LLMRelabeler:
    mode = "llm"

    def __init__(
        self,
        client: LLMClient,
        tax: Taxonomy,
        *,
        model: str,
        backend: str,
        cache_dir: Path,
        batch_size: int = 50,
        max_input_chars: int = 48000,
        extra_body: Mapping[str, object] | None = None,
        retry: RetryConfig = RetryConfig(),
        secrets: Sequence[str] = (),
        allow_mixed: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._tax = tax
        self._secrets = secrets
        self._sleep = sleep
        self.model = model
        self.backend = backend
        self.prompt_version = f"{unseen.UNSEEN_PROMPT_VERSION}+{reextract.REEXTRACT_PROMPT_VERSION}"
        self._unseen_cfg = unseen.UnseenConfig(
            model=model, backend=backend, batch_size=batch_size, extra_body=extra_body,
            retry=retry,
        )
        self._reextract_cfg = reextract.ReextractConfig(
            model=model, backend=backend, max_input_chars=max_input_chars,
            extra_body=extra_body, retry=retry,
        )
        self._unseen_cache = StampedCache(
            cache_dir / UNSEEN_CACHE_FILE,
            Stamp(tax.version, model, backend, unseen.UNSEEN_PROMPT_VERSION),
            allow_mixed=allow_mixed,
        )
        self._reextract_cache = StampedCache(
            cache_dir / REEXTRACT_CACHE_FILE,
            Stamp(tax.version, model, backend, reextract.REEXTRACT_PROMPT_VERSION),
            allow_mixed=allow_mixed,
        )
        self.stats = CallStats()
        self.failed_skills: list[str] = []

    def assign_unseen(self, skills: Sequence[str]) -> Mapping[str, str]:
        result = unseen.assign_unseen(
            self._client,
            cfg=self._unseen_cfg,
            skills=skills,
            tax=self._tax,
            cache=self._unseen_cache,
            secrets=self._secrets,
            sleep=self._sleep,
        )
        self.stats.calls += result.stats.calls
        self.stats.retries += result.stats.retries
        self.stats.rate_limited += result.stats.rate_limited
        self.failed_skills.extend(result.failed)
        return result.assignments

    def reextract(
        self, doc_id: str, text: str, skills: Sequence[str]
    ) -> list[tuple[str, str]] | None:
        return reextract.reextract_document(
            self._client,
            cfg=self._reextract_cfg,
            doc_id=doc_id,
            text=text,
            skills=skills,
            tax=self._tax,
            cache=self._reextract_cache,
            stats=self.stats,
            secrets=self._secrets,
            sleep=self._sleep,
        )
