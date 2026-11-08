"""Voyager adapter over the OpenAI-compatible SDK: no retry, no logging, no content parsing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import openai

from getwork.llm import client as llm_client
from getwork.llm.client import LLMError, LLMRequest, PermanentLLMError
from getwork.llm.settings import Settings

DEFAULT_TIMEOUT_S: float = 1800.0
SCHEMA_NAME = "structured_document"
MODEL_INFO_FIELDS = (
    "model_type",
    "context_window",
    "supports_response_schema",
    "supports_function_calling",
    "status",
    "replacement",
)


class VoyagerClient:
    def __init__(
        self,
        settings: Settings,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        http_client: object | None = None,
    ) -> None:
        self._api_key = settings.api_key
        self._sdk = openai.OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    @contextmanager
    def _mapped_errors(self) -> Iterator[None]:
        secrets = [self._api_key]
        try:
            yield
        except openai.APIStatusError as e:
            raise llm_client.error_from_http(
                e.status_code, dict(e.response.headers), str(e), secrets
            ) from None
        except openai.APIConnectionError as e:
            raise llm_client.error_from_http(None, None, str(e), secrets) from None
        except LLMError:
            raise
        except Exception as e:
            raise PermanentLLMError(llm_client.describe_exception(e, secrets)) from None

    def complete(self, request: LLMRequest) -> str:
        kwargs: dict = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": request.reasoning_effort}
        if request.response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": SCHEMA_NAME,
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        with self._mapped_errors():
            response = self._sdk.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

    def describe_model(self, model: str) -> dict:
        with self._mapped_errors():
            for entry in self._sdk.models.list():
                if entry.id != model:
                    continue
                data = entry.model_dump()
                info = {k: data[k] for k in MODEL_INFO_FIELDS if k in data}
                return {"id": model, "found": True, **info}
            return {"id": model, "found": False}


def build_client(settings: Settings, timeout_s: float = DEFAULT_TIMEOUT_S) -> VoyagerClient:
    return VoyagerClient(settings, timeout_s=timeout_s)
