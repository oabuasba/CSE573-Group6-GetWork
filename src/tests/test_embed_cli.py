import json
from pathlib import Path

import numpy as np
import pytest

from getwork.embed import cli, factory
from getwork.embed.client import EmbeddingError, EmbeddingInfo
from getwork.embed.fake import FakeEmbeddingClient

pytestmark = pytest.mark.usefixtures("offline")

REPORT_KEYS = {
    "backend", "model", "dim", "max_seq_length", "dtype", "seconds",
    "related_cosine", "unrelated_cosine", "norms_ok",
}  # fmt: skip


class FailingClient:
    info = EmbeddingInfo(model="m", backend="local", dim=4)

    def embed(self, texts: list[str]) -> np.ndarray:
        raise EmbeddingError("boom")


def test_preflight_prints_json_and_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([], client=FakeEmbeddingClient(), environ={}) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == REPORT_KEYS
    assert report["norms_ok"] is True
    assert report["backend"] == "fake"
    assert isinstance(report["related_cosine"], float)
    assert isinstance(report["unrelated_cosine"], float)


def test_preflight_embedding_error_exits_3(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([], client=FailingClient(), environ={}) == 3
    captured = capsys.readouterr()
    assert "boom" in captured.err
    assert captured.out == ""


def test_preflight_config_error_exits_2(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    env = {"VOYAGER_API_KEY": "sk-secret-key"}
    argv = ["--embed-backend", "voyager", "--env-file", str(tmp_path / "missing.env")]
    assert cli.main(argv, environ=env) == 2
    captured = capsys.readouterr()
    assert "VOYAGER_BASE_URL" in captured.err and "VOYAGER_EMBED_MODEL" in captured.err
    assert "sk-secret-key" not in captured.err + captured.out


def test_preflight_local_needs_no_variables(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(factory, "_real_client", lambda settings: FakeEmbeddingClient())
    argv = ["--embed-backend", "local", "--env-file", str(tmp_path / "missing.env")]
    assert cli.main(argv, environ={}) == 0
    assert json.loads(capsys.readouterr().out)["norms_ok"] is True
