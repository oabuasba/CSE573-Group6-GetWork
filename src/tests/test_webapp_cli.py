import json
import logging
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from getwork.webapp import cli, models

pytestmark = pytest.mark.usefixtures("offline")

REPO = Path(__file__).resolve().parents[2]
FAKE_SECRET = "fake-secret-xyz-789"


class RecordingRun:
    def __init__(self) -> None:
        self.calls: list[tuple[FastAPI, dict]] = []

    def __call__(self, app: FastAPI, **kwargs: object) -> None:
        self.calls.append((app, kwargs))


def test_check_demo_and_artifacts(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["check", "--demo"]) == 0
    health = models.HealthResponse.model_validate_json(capsys.readouterr().out)
    assert health.mode == "demo" and health.explainer.kind == "template"
    monkeypatch.chdir(tmp_path)
    assert cli.main(["check", "--artifacts-dir", str(tmp_path / "artifacts")]) == 2
    assert "artifact mode is not built yet; use --demo" in capsys.readouterr().err


def test_check_errors_exit_2(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["check", "--demo", "--systems", "B3"]) == 2
    assert "--systems" in capsys.readouterr().err
    missing = tmp_path / "weights.json"
    assert cli.main(["check", "--demo", "--b1plus-weights", str(missing)]) == 2
    assert "--b1plus-weights" in capsys.readouterr().err
    missing.write_text("{not json", encoding="utf-8")
    assert cli.main(["check", "--demo", "--b1plus-weights", str(missing)]) == 2
    assert cli.main(["check", "--demo", "--systems", "B1+,Full"]) == 0
    assert cli.main(["bogus"]) == 2
    assert cli.main(["serve", "--demo", "--port", "0"]) == 2


def test_default_artifacts_dir_is_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    args = cli._build_parser().parse_args(["check"])
    assert args.artifacts_dir == "artifacts_trial"
    assert cli.main(["check"]) == 2
    assert "use --demo" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_serve_defaults_and_options(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = RecordingRun()
    assert cli.main(["serve", "--demo"], run=run) == 0
    assert len(run.calls) == 1
    app, kwargs = run.calls[0]
    assert isinstance(app, FastAPI)
    assert (kwargs["host"], kwargs["port"]) == ("127.0.0.1", 8000)
    assert "http://127.0.0.1:8000" in capsys.readouterr().out
    (tmp_path / "index.html").write_text("<title>Built</title>", encoding="utf-8")
    custom = RecordingRun()
    argv = ["serve", "--demo", "--port", "9001", "--static-dir", str(tmp_path)]
    assert cli.main(argv, run=custom) == 0
    assert custom.calls[0][1]["port"] == 9001
    assert TestClient(custom.calls[0][0]).get("/").text == "<title>Built</title>"


def test_serve_prints_build_command_without_frontend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "WEB_DIST", tmp_path / "missing")
    assert cli.main(["serve", "--demo"], run=RecordingRun()) == 0
    assert "npm --prefix src/web run build" in capsys.readouterr().out
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.html").write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli, "WEB_DIST", tmp_path / "dist")
    assert cli.main(["serve", "--demo"], run=RecordingRun()) == 0
    assert "npm --prefix" not in capsys.readouterr().out


def test_serve_warns_on_public_host(capsys: pytest.CaptureFixture[str]) -> None:
    run = RecordingRun()
    assert cli.main(["serve", "--demo", "--host", "0.0.0.0"], run=run) == 0
    assert "no authentication" in capsys.readouterr().err
    assert run.calls[0][1]["host"] == "0.0.0.0"
    cli.main(["serve", "--demo"], run=RecordingRun())
    assert "no authentication" not in capsys.readouterr().err


def test_web_dist_location() -> None:
    assert cli.WEB_DIST == REPO / "src" / "web" / "dist"


def capabilities_of(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict:
    assert cli.main(["check", "--demo", *argv]) == 0
    return json.loads(capsys.readouterr().out)["capabilities"]


def test_capabilities_from_environment(capsys: pytest.CaptureFixture[str]) -> None:
    default = capabilities_of([], capsys)
    assert default["structure"]["available"] is False
    assert default["structure"]["reason"] == "Demo mode has no language model."
    assert default["embed"]["available"] is True
    off = capabilities_of(["--no-structure", "--no-embed"], capsys)
    assert off["structure"] == {"available": False, "reason": "Disabled with --no-structure.",
                                "detail": None}
    assert off["embed"]["reason"] == "Disabled with --no-embed."
    assert off["embed"]["available"] is False


def test_no_secret_in_output(capsys: pytest.CaptureFixture[str]) -> None:
    environ = {"VOYAGER_API_KEY": FAKE_SECRET}
    assert cli.main(["check", "--demo"], environ=environ) == 0
    assert cli.main(["serve", "--demo"], environ=environ, run=RecordingRun()) == 0
    assert cli.main(["check"], environ=environ) == 2
    captured = capsys.readouterr()
    assert FAKE_SECRET not in captured.out + captured.err


def test_uvicorn_log_redaction(caplog: pytest.LogCaptureFixture) -> None:
    cli.install_uvicorn_redaction([FAKE_SECRET])
    logger = logging.getLogger("uvicorn.error")
    try:
        with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
            try:
                raise RuntimeError(f"{FAKE_SECRET} leaked")
            except RuntimeError:
                logger.error("Exception in ASGI application %s", FAKE_SECRET, exc_info=True)
    finally:
        cli.install_uvicorn_redaction([])
    assert caplog.records
    assert FAKE_SECRET not in caplog.text
    assert "RuntimeError" in caplog.text and "***" in caplog.text


def test_pyproject_script_and_deps() -> None:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["webapp"] == "getwork.webapp.cli:main"
    assert {"fastapi", "uvicorn"} <= set(data["project"]["dependencies"])
    assert "httpx" in data["dependency-groups"]["dev"]


def test_no_heavy_imports() -> None:
    pattern = re.compile(r"^\s*(?:import|from)\s+(torch|sentence_transformers|openai)\b", re.M)
    for path in (REPO / "src" / "getwork" / "webapp").glob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
    code = (
        "import sys, getwork.webapp.api;"
        "bad = [m for m in ('torch', 'sentence_transformers', 'openai') if m in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
