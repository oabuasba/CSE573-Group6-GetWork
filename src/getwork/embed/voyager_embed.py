"""Voyager embeddings over the OpenAI-compatible SDK, with the LLM error taxonomy and retries."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import openai

from getwork.embed.client import EmbeddingError, EmbeddingInfo, l2_normalize
from getwork.embed.settings import EmbedSettings
from getwork.llm import client as llm_client
from getwork.llm.client import (
    LLMError,
    PermanentLLMError,
    RateLimitError,
    TransientLLMError,
)
from getwork.llm.retry import RetryConfig, backoff_delay

PROBE_TEXT = "dimension probe"


class VoyagerEmbeddingClient:
    def __init__(
        self,
        settings: EmbedSettings,
        *,
        timeout_s: float = 600.0,
        http_client: object | None = None,
        retry: RetryConfig = RetryConfig(),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._retry = retry
        self._sleep = sleep
        self._sdk = openai.OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )
        probe = self._embed_batch([PROBE_TEXT])
        self._info = EmbeddingInfo(settings.model, "voyager", int(probe.shape[1]))

    @property
    def info(self) -> EmbeddingInfo:
        return self._info

    @contextmanager
    def _mapped_errors(self) -> Iterator[None]:
        secrets = [self._settings.api_key]
        try:
            yield
        except openai.APIStatusError as e:
            raise llm_client.error_from_http(
                e.status_code, dict(e.response.headers), str(e), secrets
            ) from None
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            raise llm_client.error_from_http(None, None, str(e), secrets) from None
        except (LLMError, EmbeddingError):
            raise
        except Exception as e:
            raise PermanentLLMError(llm_client.describe_exception(e, secrets)) from None

    def _request(self, batch: Sequence[str]) -> np.ndarray:
        with self._mapped_errors():
            response = self._sdk.embeddings.create(
                model=self._settings.model, input=list(batch), encoding_format="float"
            )
            items = sorted(response.data, key=lambda item: item.index)
            if len(items) != len(batch):
                raise EmbeddingError(
                    f"expected {len(batch)} embeddings, the server returned {len(items)}"
                )
            return l2_normalize(np.array([item.embedding for item in items], dtype=np.float32))

    def _embed_batch(self, batch: Sequence[str]) -> np.ndarray:
        failures = 0
        rate_limit_waits = 0
        while True:
            try:
                return self._request(batch)
            except TransientLLMError as e:
                failures += 1
                if failures >= self._retry.max_attempts:
                    raise
                self._sleep(backoff_delay(failures, self._retry, e.retry_after))
            except RateLimitError as e:
                rate_limit_waits += 1
                if rate_limit_waits > self._retry.max_rate_limit_waits:
                    raise
                self._sleep(max(self._retry.rate_limit_wait_s, e.retry_after or 0.0))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self._info.dim), dtype=np.float32)
        size = self._settings.batch_size
        parts = [self._embed_batch(texts[i : i + size]) for i in range(0, len(texts), size)]
        return np.concatenate(parts).astype(np.float32)
