import os
from pathlib import Path

import pytest

from getwork.llm import settings
from getwork.llm.settings import (
    BackendSettings,
    ConfigError,
    Settings,
    load_backend_settings,
    load_settings,
    resolve_backend,
)

pytestmark = pytest.mark.usefixtures("offline")

FAKE_KEY = "fake-key-do-not-use"
FAKE_URL = "https://fake.example.test/v1"


def _env(**over: str) -> dict[str, str]:
    base = {
        settings.ENV_API_KEY: FAKE_KEY,
        settings.ENV_BASE_URL: FAKE_URL,
        settings.ENV_MODEL: "fake-model",
    }
    return {**base, **over}


def test_missing_key_names_variable_not_value() -> None:
    environ = _env()
    del environ[settings.ENV_API_KEY]
    del environ[settings.ENV_MODEL]
    with pytest.raises(ConfigError) as exc:
        load_settings(environ, None)
    msg = str(exc.value)
    assert settings.ENV_API_KEY in msg and settings.ENV_MODEL in msg
    assert settings.ENV_BASE_URL not in msg
    assert FAKE_URL not in msg and "fake-model" not in msg


def test_dotenv_values_are_used_and_environ_wins(tmp_path: Path) -> None:
    path = tmp_path / "test.env"
    path.write_text(
        f"{settings.ENV_API_KEY}=file-key\n"
        f"{settings.ENV_BASE_URL}=https://file.example.test/v1\n"
        f"{settings.ENV_MODEL}=file-model\n"
    )
    from_file = load_settings({}, path)
    assert (from_file.api_key, from_file.model) == ("file-key", "file-model")
    merged = load_settings({settings.ENV_MODEL: "env-model", settings.ENV_API_KEY: "  "}, path)
    assert merged.model == "env-model"
    assert merged.api_key == "file-key"


def test_load_settings_does_not_mutate_os_environ(tmp_path: Path) -> None:
    path = tmp_path / "test.env"
    path.write_text(f"{settings.ENV_API_KEY}=file-key\n")
    before = dict(os.environ)
    load_settings(_env(), path)
    assert dict(os.environ) == before


def test_model_has_no_default_and_override_wins() -> None:
    environ = _env()
    del environ[settings.ENV_MODEL]
    with pytest.raises(ConfigError, match=settings.ENV_MODEL):
        load_settings(environ, None)
    with pytest.raises(ConfigError, match=settings.ENV_MODEL):
        load_settings(environ, None, model_override="   ")
    assert load_settings(environ, None, model_override="cli-model").model == "cli-model"
    assert load_settings(_env(), None, model_override="cli-model").model == "cli-model"


@pytest.mark.parametrize(
    "url",
    ["https://a.test/v1", "http://localhost:8000/v1", "http://127.0.0.1/v1", "http://[::1]/v1"],
)
def test_base_url_accepted(url: str) -> None:
    assert load_settings(_env(**{settings.ENV_BASE_URL: url}), None).base_url == url


@pytest.mark.parametrize("url", ["http://example.com/v1", "ftp://x", "not a url"])
def test_base_url_rejected(url: str) -> None:
    with pytest.raises(ConfigError):
        load_settings(_env(**{settings.ENV_BASE_URL: url}), None)


def test_base_url_requires_https_except_localhost() -> None:
    test_base_url_accepted("https://a.test/v1")
    test_base_url_rejected("http://example.com/v1")
    assert load_settings(_env(**{settings.ENV_BASE_URL: "https://a.test/v1/"}), None).base_url == (
        "https://a.test/v1"
    )


def test_settings_repr_and_str_hide_key() -> None:
    s = Settings(api_key=FAKE_KEY, base_url=FAKE_URL, model="m")
    assert FAKE_KEY not in repr(s) and FAKE_KEY not in str(s)
    assert str(s) == f"Settings(base_url={FAKE_URL}, model=m)"


LOCAL_KEY = "local-secret-key"


def _local(**over: str) -> dict[str, str]:
    base = {
        settings.ENV_LOCAL_BASE_URL: "http://localhost:8000/v1",
        settings.ENV_LOCAL_MODEL: "qwen-local",
    }
    return {**base, **over}


def test_resolve_backend_precedence(tmp_path: Path) -> None:
    path = tmp_path / "test.env"
    path.write_text(f"{settings.ENV_BACKEND}=local\n")
    assert resolve_backend({}, None) == "voyager"
    assert resolve_backend({}, path) == "local"
    assert resolve_backend({settings.ENV_BACKEND: " Voyager "}, path) == "voyager"
    assert resolve_backend({settings.ENV_BACKEND: "voyager"}, path, " LOCAL ") == "local"
    assert resolve_backend({settings.ENV_BACKEND: "local"}, None, "  ") == "local"


def test_resolve_backend_rejects_unknown() -> None:
    with pytest.raises(ConfigError) as exc:
        resolve_backend({settings.ENV_BACKEND: "sekret-thing"}, None)
    msg = str(exc.value)
    assert "voyager" in msg and "local" in msg and "sekret-thing" not in msg
    with pytest.raises(ConfigError) as exc2:
        resolve_backend({}, None, "other-thing")
    assert "other-thing" not in str(exc2.value)


def test_voyager_backend_requires_key() -> None:
    environ = _env()
    s = load_backend_settings(environ, None)
    assert s == BackendSettings(
        backend="voyager",
        base_url=FAKE_URL,
        model="fake-model",
        api_key=FAKE_KEY,
        has_real_key=True,
    )
    del environ[settings.ENV_API_KEY]
    with pytest.raises(ConfigError) as exc:
        load_backend_settings(environ, None, "voyager")
    assert settings.ENV_API_KEY in str(exc.value)


def test_local_backend_requires_url_and_model_only() -> None:
    with pytest.raises(ConfigError) as exc:
        load_backend_settings(_env(), None, "local")
    msg = str(exc.value)
    assert settings.ENV_LOCAL_BASE_URL in msg and settings.ENV_LOCAL_MODEL in msg
    assert settings.ENV_API_KEY not in msg and FAKE_URL not in msg
    only_url = {settings.ENV_LOCAL_BASE_URL: "http://localhost:1/v1"}
    with pytest.raises(ConfigError) as exc2:
        load_backend_settings(only_url, None, "local")
    assert settings.ENV_LOCAL_MODEL in str(exc2.value)
    assert settings.ENV_LOCAL_BASE_URL not in str(exc2.value)
    s = load_backend_settings(only_url, None, "local", model_override="m")
    assert s.model == "m"


def test_local_key_optional_and_placeholder() -> None:
    s = load_backend_settings(_local(), None, "local")
    assert s.api_key == settings.LOCAL_PLACEHOLDER_KEY and not s.has_real_key
    s = load_backend_settings(_local(**{settings.ENV_LOCAL_API_KEY: LOCAL_KEY}), None, "local")
    assert s.api_key == LOCAL_KEY and s.has_real_key and s.backend == "local"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1/",
        "http://[::1]:8000/v1",
        "https://llm.example.test/v1",
    ],
)
def test_local_base_url_accepted(url: str) -> None:
    s = load_backend_settings(_local(**{settings.ENV_LOCAL_BASE_URL: url}), None, "local")
    assert s.base_url == url.rstrip("/")


def test_local_base_url_rules() -> None:
    bad = "http://gpu-box.example.test:8000/v1"
    with pytest.raises(ConfigError) as exc:
        load_backend_settings(_local(**{settings.ENV_LOCAL_BASE_URL: bad}), None, "local")
    msg = str(exc.value)
    assert settings.ENV_LOCAL_BASE_URL in msg and "gpu-box" not in msg
    with pytest.raises(ConfigError):
        load_backend_settings(_local(**{settings.ENV_LOCAL_BASE_URL: "ftp://x/v1"}), None, "local")
    with pytest.raises(ConfigError) as exc2:
        load_backend_settings(_env(**{settings.ENV_BASE_URL: bad}), None, "voyager")
    assert settings.ENV_BASE_URL in str(exc2.value)


def test_backends_do_not_share_variables() -> None:
    both = {**_env(), **_local(**{settings.ENV_LOCAL_API_KEY: LOCAL_KEY})}
    v = load_backend_settings(both, None, "voyager")
    assert (v.base_url, v.model, v.api_key) == (FAKE_URL, "fake-model", FAKE_KEY)
    loc = load_backend_settings(both, None, "local")
    assert (loc.base_url, loc.model, loc.api_key) == (
        "http://localhost:8000/v1",
        "qwen-local",
        LOCAL_KEY,
    )
    both[settings.ENV_BACKEND] = "local"
    assert load_backend_settings(both, None).backend == "local"


def test_backend_settings_hide_key_and_leave_environ(tmp_path: Path) -> None:
    path = tmp_path / "test.env"
    path.write_text(f"{settings.ENV_LOCAL_API_KEY}={LOCAL_KEY}\n")
    before = dict(os.environ)
    s = load_backend_settings(_local(), path, "local")
    assert dict(os.environ) == before
    assert LOCAL_KEY not in repr(s) and LOCAL_KEY not in str(s)
    assert "qwen-local" in str(s) and "localhost" in str(s)
    v = load_backend_settings(_env(), None, "voyager")
    assert FAKE_KEY not in repr(v) and FAKE_KEY not in str(v)
