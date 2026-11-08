from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ENV_API_KEY: str = "VOYAGER_API_KEY"
ENV_BASE_URL: str = "VOYAGER_BASE_URL"
ENV_MODEL: str = "VOYAGER_MODEL"

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


def _validate_base_url(base_url: str) -> str:
    try:
        parts = urlsplit(base_url)
        host = parts.hostname
    except ValueError:
        raise ConfigError(f"{ENV_BASE_URL} is not a valid URL") from None
    is_secure = parts.scheme == "https" and bool(host)
    is_local = parts.scheme == "http" and host in LOCAL_HOSTS
    if not (is_secure or is_local):
        raise ConfigError(
            f"{ENV_BASE_URL} must be https:// (or http:// for localhost, 127.0.0.1, ::1)"
        )
    return base_url.rstrip("/")


def load_settings(
    environ: Mapping[str, str],
    dotenv_path: Path | None,
    model_override: str | None = None,
) -> Settings:
    file_values: Mapping[str, str | None] = {}
    if dotenv_path is not None and dotenv_path.is_file():
        file_values = dotenv_values(dotenv_path)
    merged = {**file_values, **{k: v for k, v in environ.items() if _clean(v)}}

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
