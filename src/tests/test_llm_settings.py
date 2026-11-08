import os
from pathlib import Path

import pytest

from getwork.llm import settings
from getwork.llm.settings import ConfigError, Settings, load_settings

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
