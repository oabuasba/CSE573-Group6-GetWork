"""Embedding backend settings (no torch, no openai); messages never contain secret values."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from getwork.embed.client import EMBEDDING_BACKENDS
from getwork.llm import settings as llm_settings
from getwork.llm.settings import ConfigError

ENV_EMBED_BACKEND: str = "EMBED_BACKEND"
ENV_EMBED_MODEL: str = "EMBED_MODEL"
ENV_EMBED_VOYAGER_MODEL: str = "VOYAGER_EMBED_MODEL"
DEFAULT_EMBED_BACKEND: str = "local"
DEFAULT_LOCAL_MODEL: str = "Qwen/Qwen3-Embedding-4B"
DEFAULT_BATCH_SIZE_LOCAL: int = 8
DEFAULT_BATCH_SIZE_VOYAGER: int = 32
DEFAULT_MAX_SEQ_LENGTH: int = 2048
DTYPES: tuple[str, ...] = ("auto", "float16", "bfloat16", "float32")
DEVICE_PATTERN = re.compile(r"auto|cpu|cuda(:\d+)?")


@dataclass(frozen=True)
class EmbedSettings:
    backend: str
    model: str
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = DEFAULT_BATCH_SIZE_LOCAL
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    api_key: str = field(default="", repr=False)
    base_url: str = ""

    def __str__(self) -> str:
        return (
            f"EmbedSettings(backend={self.backend}, model={self.model}, "
            f"device={self.device}, batch_size={self.batch_size})"
        )


def _resolve_backend(merged: Mapping[str, str | None], backend: str | None) -> str:
    raw = llm_settings._clean(backend) or llm_settings._clean(merged.get(ENV_EMBED_BACKEND))
    resolved = raw.lower() if raw else DEFAULT_EMBED_BACKEND
    if resolved not in EMBEDDING_BACKENDS:
        raise ConfigError(
            f"unknown embedding backend; allowed values: {', '.join(EMBEDDING_BACKENDS)}"
        )
    return resolved


def _positive(value: int | None, default: int, name: str) -> int:
    if value is None:
        return default
    if value < 1:
        raise ConfigError(f"{name} must be at least 1")
    return value


def load_embed_settings(
    environ: Mapping[str, str],
    dotenv_path: Path | None,
    *,
    backend: str | None = None,
    model: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    batch_size: int | None = None,
    max_seq_length: int | None = None,
) -> EmbedSettings:
    merged = llm_settings._merged_values(environ, dotenv_path)
    chosen = _resolve_backend(merged, backend)
    max_len = _positive(max_seq_length, DEFAULT_MAX_SEQ_LENGTH, "max sequence length")
    if chosen == "voyager":
        size = _positive(batch_size, DEFAULT_BATCH_SIZE_VOYAGER, "batch size")
        return _voyager_settings(merged, model=model, batch_size=size, max_seq_length=max_len)
    size = _positive(batch_size, DEFAULT_BATCH_SIZE_LOCAL, "batch size")
    return _local_settings(
        merged,
        model=model,
        device=device,
        dtype=dtype,
        batch_size=size,
        max_seq_length=max_len,
    )


def _local_settings(
    merged: Mapping[str, str | None],
    *,
    model: str | None,
    device: str | None,
    dtype: str | None,
    batch_size: int,
    max_seq_length: int,
) -> EmbedSettings:
    chosen_device = llm_settings._clean(device) or "auto"
    chosen_dtype = llm_settings._clean(dtype) or "auto"
    if not DEVICE_PATTERN.fullmatch(chosen_device):
        raise ConfigError("device must be one of: auto, cpu, cuda, cuda:N")
    if chosen_dtype not in DTYPES:
        raise ConfigError(f"dtype must be one of: {', '.join(DTYPES)}")
    chosen_model = (
        llm_settings._clean(model)
        or llm_settings._clean(merged.get(ENV_EMBED_MODEL))
        or DEFAULT_LOCAL_MODEL
    )
    return EmbedSettings(
        backend="local",
        model=chosen_model,
        device=chosen_device,
        dtype=chosen_dtype,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )


def _voyager_settings(
    merged: Mapping[str, str | None],
    *,
    model: str | None,
    batch_size: int,
    max_seq_length: int,
) -> EmbedSettings:
    api_key = llm_settings._clean(merged.get(llm_settings.ENV_API_KEY))
    base_url = llm_settings._clean(merged.get(llm_settings.ENV_BASE_URL))
    chosen_model = llm_settings._clean(model) or llm_settings._clean(
        merged.get(ENV_EMBED_VOYAGER_MODEL)
    )
    missing = [
        name
        for name, value in (
            (llm_settings.ENV_API_KEY, api_key),
            (llm_settings.ENV_BASE_URL, base_url),
            (ENV_EMBED_VOYAGER_MODEL, chosen_model),
        )
        if value is None
    ]
    if missing:
        raise ConfigError(
            f"missing required setting(s): {', '.join(missing)} "
            "(set them in .env or the environment)"
        )
    assert api_key and base_url and chosen_model
    return EmbedSettings(
        backend="voyager",
        model=chosen_model,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        api_key=api_key,
        base_url=llm_settings._validate_base_url(base_url, llm_settings.ENV_BASE_URL),
    )
