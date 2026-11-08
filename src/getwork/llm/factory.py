"""Shared LLM client construction and CLI options for every LLM step."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from getwork.llm.client import LLMClient
from getwork.llm.settings import (
    BACKENDS,
    DEFAULT_BACKEND,
    BackendSettings,
    ConfigError,
    load_backend_settings,
)


@dataclass(frozen=True)
class ClientBundle:
    client: LLMClient
    backend: str  # "voyager" | "local"
    model: str
    secrets: list[str] = field(repr=False)  # [real key] or []; never the placeholder


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--extra-body", default=None, help="JSON object merged into every request")


def parse_extra_body(raw: str | None) -> dict | None:
    if raw is None or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise ConfigError("--extra-body must be a JSON object") from None
    if not isinstance(parsed, dict):
        raise ConfigError("--extra-body must be a JSON object")
    return parsed


def make_client(
    *,
    backend: str | None,
    model: str | None,
    env_file: Path,
    environ: Mapping[str, str],
    client: LLMClient | None = None,
    timeout_s: float = 1800.0,
) -> ClientBundle:
    if client is not None:
        return _injected_bundle(backend=backend, model=model, client=client)
    settings = load_backend_settings(environ, env_file, backend, model)
    secrets = [settings.api_key] if settings.has_real_key else []
    return ClientBundle(
        client=_real_client(settings, timeout_s),
        backend=settings.backend,
        model=settings.model,
        secrets=secrets,
    )


def _injected_bundle(*, backend: str | None, model: str | None, client: LLMClient) -> ClientBundle:
    if not model or not model.strip():
        raise ConfigError("--model is required")
    chosen = backend or DEFAULT_BACKEND
    if chosen not in BACKENDS:
        raise ConfigError(f"backend must be one of: {', '.join(BACKENDS)}")
    return ClientBundle(client=client, backend=chosen, model=model, secrets=[])


def _real_client(settings: BackendSettings, timeout_s: float) -> LLMClient:
    from getwork.llm import voyager

    return voyager.build_llm_client(settings, timeout_s)
