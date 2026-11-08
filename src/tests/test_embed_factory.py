import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from getwork.embed import factory
from getwork.embed import settings as embed_settings
from getwork.embed.client import EmbeddingInfo
from getwork.embed.fake import FakeEmbeddingClient
from getwork.llm.settings import ConfigError

pytestmark = pytest.mark.usefixtures("offline")

VOYAGER_ENV = {
    "VOYAGER_API_KEY": "sk-secret-key",
    "VOYAGER_BASE_URL": "https://voyager.test/v1",
    "VOYAGER_EMBED_MODEL": "embed-model",
}


def load(environ: dict[str, str] | None = None, path: Path | None = None, **kwargs: object):
    return embed_settings.load_embed_settings(environ or {}, path, **kwargs)


def test_defaults_are_local_4b() -> None:
    s = load()
    assert s.backend == "local"
    assert s.model == "Qwen/Qwen3-Embedding-4B"
    assert s.batch_size == 8
    assert s.api_key == ""
    assert load(backend="voyager", environ=VOYAGER_ENV).batch_size == 32


def test_backend_precedence_and_unknown(tmp_path: Path) -> None:
    env_file = tmp_path / "test.env"
    lines = ["EMBED_BACKEND=voyager", *(f"{k}={v}" for k, v in VOYAGER_ENV.items())]
    env_file.write_text("\n".join(lines) + "\n")
    assert load(path=env_file).backend == "voyager"
    assert load({"EMBED_BACKEND": "local"}, env_file).backend == "local"
    assert load({"EMBED_BACKEND": "local"}, env_file, backend="voyager").backend == "voyager"
    with pytest.raises(ConfigError) as excinfo:
        load({"EMBED_BACKEND": "secret-backend-value"})
    assert "local" in str(excinfo.value) and "voyager" in str(excinfo.value)
    assert "secret-backend-value" not in str(excinfo.value)


def test_model_variables_are_per_backend() -> None:
    assert load({"EMBED_MODEL": "my/local"}).model == "my/local"
    assert load({"EMBED_MODEL": "my/local"}, model="arg/model").model == "arg/model"
    s = load({**VOYAGER_ENV, "EMBED_MODEL": "my/local"}, backend="voyager")
    assert s.model == "embed-model"
    assert load(VOYAGER_ENV, backend="voyager", model="arg").model == "arg"


def test_voyager_requires_key_url_model() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load(backend="voyager")
    for name in ("VOYAGER_API_KEY", "VOYAGER_BASE_URL", "VOYAGER_EMBED_MODEL"):
        assert name in str(excinfo.value)
    partial = {"VOYAGER_API_KEY": "sk-secret-key"}
    with pytest.raises(ConfigError) as excinfo:
        load(partial, backend="voyager")
    assert "VOYAGER_API_KEY" not in str(excinfo.value)
    assert "sk-secret-key" not in str(excinfo.value)


def test_value_validation_and_secrecy() -> None:
    with pytest.raises(ConfigError):
        load({**VOYAGER_ENV, "VOYAGER_BASE_URL": "http://remote.example/v1"}, backend="voyager")
    for bad in ({"device": "gpu"}, {"dtype": "int8"}, {"batch_size": 0}, {"max_seq_length": 0}):
        with pytest.raises(ConfigError):
            load(**bad)
    assert load(device="cuda:1", dtype="bfloat16").device == "cuda:1"
    before = dict(os.environ)
    s = load(VOYAGER_ENV, backend="voyager")
    assert "sk-secret-key" not in repr(s) and "sk-secret-key" not in str(s)
    assert dict(os.environ) == before


def parse(argv: list[str], **kwargs: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    factory.add_embed_args(parser, **kwargs)
    return parser.parse_args(argv)


def test_add_embed_args_defaults() -> None:
    args = parse([])
    assert vars(args) == {
        "embed_backend": None,
        "embed_model": None,
        "device": None,
        "dtype": None,
        "embed_batch_size": None,
        "max_seq_length": None,
        "env_file": ".env",
    }
    assert "env_file" not in vars(parse([], with_env_file=False))
    assert parse(["--embed-batch-size", "4"]).embed_batch_size == 4
    with pytest.raises(SystemExit):
        parse(["--embed-batch-size", "0"])
    with pytest.raises(SystemExit):
        parse(["--embed-backend", "other"])


def test_make_client_with_injected_client(tmp_path: Path) -> None:
    fake = FakeEmbeddingClient()
    args = parse(["--env-file", str(tmp_path / "missing.env")])
    bundle = factory.make_embedding_client(args=args, environ={}, client=fake)
    assert bundle.client is fake
    assert bundle.info == fake.info
    assert bundle.secrets == []


def test_make_client_builds_real_client_and_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[embed_settings.EmbedSettings] = []

    def fake_real(settings: embed_settings.EmbedSettings) -> FakeEmbeddingClient:
        seen.append(settings)
        return FakeEmbeddingClient()

    monkeypatch.setattr(factory, "_real_client", fake_real)
    args = parse(["--env-file", str(tmp_path / "missing.env"), "--embed-backend", "voyager"])
    bundle = factory.make_embedding_client(args=args, environ=VOYAGER_ENV)
    assert bundle.secrets == ["sk-secret-key"]
    assert seen[0].backend == "voyager" and seen[0].model == "embed-model"
    assert isinstance(bundle.info, EmbeddingInfo)
    local = factory.make_embedding_client(
        args=parse(["--env-file", str(tmp_path / "missing.env")]), environ={}
    )
    assert local.secrets == [] and seen[1].backend == "local"
    with pytest.raises(ConfigError):
        factory.make_embedding_client(
            args=parse(["--env-file", str(tmp_path / "missing.env"), "--embed-backend", "voyager"]),
            environ={},
        )


def test_factory_import_is_light() -> None:
    code = (
        "import sys; import getwork.embed.factory; "
        "assert not {'torch','sentence_transformers','openai'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
