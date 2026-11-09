import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from getwork.llm import factory, settings
from getwork.llm.fake import FakeLLMClient
from getwork.llm.settings import ConfigError

pytestmark = pytest.mark.usefixtures("offline")

KEY = "fake-key-do-not-use"
VOYAGER_ENV = {
    settings.ENV_API_KEY: KEY,
    settings.ENV_BASE_URL: "https://fake.example.test/v1",
    settings.ENV_MODEL: "v-model",
}
LOCAL_ENV = {
    settings.ENV_BACKEND: "local",
    settings.ENV_LOCAL_BASE_URL: "http://localhost:8000/v1",
    settings.ENV_LOCAL_MODEL: "l-model",
}


def test_parse_extra_body() -> None:
    assert factory.parse_extra_body(None) is None
    assert factory.parse_extra_body("   ") is None
    assert factory.parse_extra_body('{"a": {"b": 1}}') == {"a": {"b": 1}}
    for bad in ("{not json", "[1, 2]", "3", '"text"', "null"):
        with pytest.raises(ConfigError) as exc:
            factory.parse_extra_body(bad)
        assert "--extra-body must be a JSON object" in str(exc.value)
        assert bad not in str(exc.value).replace("--extra-body must be a JSON object", "")


def test_make_client_with_injected_client(tmp_path: Path) -> None:
    fake = FakeLLMClient(script=[])
    missing = tmp_path / "nope.env"
    bundle = factory.make_client(backend=None, model="m", env_file=missing, environ={}, client=fake)
    assert (bundle.client, bundle.backend, bundle.model, bundle.secrets) == (fake, "voyager", "m", [])
    local = factory.make_client(backend="local", model="m", env_file=missing, environ={}, client=fake)
    assert local.backend == "local"
    with pytest.raises(ConfigError, match="--model is required"):
        factory.make_client(backend=None, model=None, env_file=missing, environ={}, client=fake)
    with pytest.raises(ConfigError):
        factory.make_client(backend="bogus", model="m", env_file=missing, environ={}, client=fake)


def test_make_client_builds_real_client_and_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeLLMClient(script=[])
    seen: list[settings.BackendSettings] = []

    def fake_real(s: settings.BackendSettings, timeout_s: float) -> FakeLLMClient:
        seen.append(s)
        return fake

    monkeypatch.setattr(factory, "_real_client", fake_real)
    env_file = tmp_path / "test.env"
    cases = [
        (VOYAGER_ENV, None, "voyager", "v-model", KEY, [KEY]),
        ({**LOCAL_ENV, settings.ENV_LOCAL_API_KEY: "local-secret"}, None, "local", "l-model",
         "local-secret", ["local-secret"]),
        (LOCAL_ENV, "override", "local", "override", settings.LOCAL_PLACEHOLDER_KEY, []),
    ]
    for environ, model, backend, want_model, want_key, want_secrets in cases:
        bundle = factory.make_client(backend=None, model=model, env_file=env_file, environ=environ)
        got = seen[-1]
        assert bundle.client is fake
        assert (bundle.backend, bundle.model, bundle.secrets) == (backend, want_model, want_secrets)
        assert (got.backend, got.model, got.api_key) == (backend, want_model, want_key)


def test_make_client_propagates_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(factory, "_real_client", lambda s, t: pytest.fail("should not build"))
    with pytest.raises(ConfigError) as exc:
        factory.make_client(backend="local", model=None, env_file=tmp_path / "x.env", environ={})
    assert settings.ENV_LOCAL_BASE_URL in str(exc.value)
    assert settings.ENV_LOCAL_MODEL in str(exc.value)


def test_add_llm_args_defaults() -> None:
    parser = argparse.ArgumentParser()
    factory.add_llm_args(parser)
    dests = {a.dest for a in parser._actions if a.dest != "help"}
    assert dests == {"backend", "model", "env_file", "extra_body"}
    args = parser.parse_args([])
    assert (args.backend, args.model, args.env_file, args.extra_body) == (None, None, ".env", None)
    assert parser.parse_args(["--backend", "local"]).backend == "local"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--backend", "other"])
    assert exc.value.code == 2


def test_factory_import_does_not_import_openai() -> None:
    code = "import sys; import getwork.llm.factory; assert 'openai' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_client_bundle_repr_hides_secrets(tmp_path: Path) -> None:
    bundle = factory.ClientBundle(
        client=FakeLLMClient(script=[]), backend="voyager", model="m", secrets=[KEY]
    )
    assert KEY not in repr(bundle)
    assert KEY not in str(bundle)
