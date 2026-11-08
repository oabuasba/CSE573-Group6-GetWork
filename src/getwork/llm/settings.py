from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ENV_API_KEY: str = "VOYAGER_API_KEY"
ENV_BASE_URL: str = "VOYAGER_BASE_URL"
ENV_MODEL: str = "VOYAGER_MODEL"
ENV_BACKEND: str = "LLM_BACKEND"
ENV_LOCAL_BASE_URL: str = "LOCAL_LLM_BASE_URL"
ENV_LOCAL_MODEL: str = "LOCAL_LLM_MODEL"
ENV_LOCAL_API_KEY: str = "LOCAL_LLM_API_KEY"
BACKENDS: tuple[str, ...] = ("voyager", "local")
DEFAULT_BACKEND: str = "voyager"
LOCAL_PLACEHOLDER_KEY: str = "local-no-key"

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ConfigError(Exception):
    """Raised for missing or invalid LLM settings; messages never contain values."""


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str
    model: str

    def __str__(self) -> str:
        return f"Settings(base_url={self.base_url}, model={self.model})"


def _clean(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _validate_base_url(base_url: str, name: str = ENV_BASE_URL) -> str:
    try:
        parts = urlsplit(base_url)
        host = parts.hostname
    except ValueError:
        raise ConfigError(f"{name} is not a valid URL") from None
    is_secure = parts.scheme == "https" and bool(host)
    is_local = parts.scheme == "http" and host in LOCAL_HOSTS
    if not (is_secure or is_local):
        raise ConfigError(
            f"{name} must be https:// (or http:// for localhost, 127.0.0.1, ::1)"
        )
    return base_url.rstrip("/")


def _merged_values(
    environ: Mapping[str, str], dotenv_path: Path | None
) -> Mapping[str, str | None]:
    file_values: Mapping[str, str | None] = {}
    if dotenv_path is not None and dotenv_path.is_file():
        file_values = dotenv_values(dotenv_path)
    return {**file_values, **{k: v for k, v in environ.items() if _clean(v)}}


def load_settings(
    environ: Mapping[str, str],
    dotenv_path: Path | None,
    model_override: str | None = None,
) -> Settings:
    merged = _merged_values(environ, dotenv_path)

    api_key = _clean(merged.get(ENV_API_KEY))
    base_url = _clean(merged.get(ENV_BASE_URL))
    model = _clean(model_override) or _clean(merged.get(ENV_MODEL))

    missing = [
        name
        for name, value in (
            (ENV_API_KEY, api_key),
            (ENV_BASE_URL, base_url),
            (ENV_MODEL, model),
        )
        if value is None
    ]
    if missing:
        raise ConfigError(
            f"missing required setting(s): {', '.join(missing)} "
            "(set them in .env or the environment)"
        )
    assert api_key and base_url and model
    return Settings(api_key=api_key, base_url=_validate_base_url(base_url), model=model)


@dataclass(frozen=True)
class BackendSettings:
    backend: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    has_real_key: bool = False

    def __str__(self) -> str:
        return (
            f"BackendSettings(backend={self.backend}, base_url={self.base_url}, "
            f"model={self.model})"
        )


def resolve_backend(
    environ: Mapping[str, str],
    dotenv_path: Path | None,
    backend_override: str | None = None,
) -> str:
    raw = _clean(backend_override) or _clean(
        _merged_values(environ, dotenv_path).get(ENV_BACKEND)
    )
    backend = raw.lower() if raw else DEFAULT_BACKEND
    if backend not in BACKENDS:
        raise ConfigError(f"unknown backend; allowed values: {', '.join(BACKENDS)}")
    return backend


def load_backend_settings(
    environ: Mapping[str, str],
    dotenv_path: Path | None,
    backend_override: str | None = None,
    model_override: str | None = None,
) -> BackendSettings:
    backend = resolve_backend(environ, dotenv_path, backend_override)
    if backend == "voyager":
        s = load_settings(environ, dotenv_path, model_override)
        return BackendSettings(
            backend=backend,
            base_url=s.base_url,
            model=s.model,
            api_key=s.api_key,
            has_real_key=True,
        )
    return _load_local_settings(environ, dotenv_path, model_override)


def _load_local_settings(
    environ: Mapping[str, str],
    dotenv_path: Path | None,
    model_override: str | None,
) -> BackendSettings:
    merged = _merged_values(environ, dotenv_path)
    base_url = _clean(merged.get(ENV_LOCAL_BASE_URL))
    model = _clean(model_override) or _clean(merged.get(ENV_LOCAL_MODEL))
    missing = [
        name
        for name, value in ((ENV_LOCAL_BASE_URL, base_url), (ENV_LOCAL_MODEL, model))
        if value is None
    ]
    if missing:
        raise ConfigError(
            f"missing required setting(s): {', '.join(missing)} "
            "(set them in .env or the environment)"
        )
    assert base_url and model
    api_key = _clean(merged.get(ENV_LOCAL_API_KEY))
    return BackendSettings(
        backend="local",
        base_url=_validate_base_url(base_url, ENV_LOCAL_BASE_URL),
        model=model,
        api_key=api_key or LOCAL_PLACEHOLDER_KEY,
        has_real_key=api_key is not None,
    )
