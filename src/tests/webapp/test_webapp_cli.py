import json
import logging
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from artifact_helpers import build_artifacts
from fastapi import FastAPI
from fastapi.testclient import TestClient

from getwork.embed.fake import FakeEmbeddingClient
from getwork.llm.fake import FakeLLMClient
from getwork.webapp import cli, models

pytestmark = pytest.mark.usefixtures("offline")

REPO = Path(__file__).resolve().parents[3]
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
    root = tmp_path / "art"
    build_artifacts(root)
    llm, embedder = FakeLLMClient(script=[]), FakeEmbeddingClient(dim=16)
    argv = ["check", "--artifacts-dir", str(root), "--model", "m"]
    factory_calls: list[object] = []
    real_factory = cli.embed_factory.make_embedding_client
    monkeypatch.setattr(
        cli.embed_factory, "make_embedding_client",
        lambda **kwargs: factory_calls.append(kwargs) or real_factory(**kwargs),
    )
    assert cli.main(argv, environ={}, llm_client=llm, embedding_client=embedder) == 0
    assert factory_calls == []
    health = models.HealthResponse.model_validate_json(capsys.readouterr().out)
    assert health.mode == "artifacts" and health.artifacts_dir == str(root)
    assert health.counts == {"resumes": 6, "postings": 5}
    assert llm.calls == [] and embedder.calls == []


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
    capsys.readouterr()
    assert cli.main(["check", "--artifacts-dir", str(tmp_path / "nope")], environ={}) == 2
    err = capsys.readouterr().err
    assert str(tmp_path / "nope") in err and "artifacts_trial/v5" in err
    broken = tmp_path / "broken"
    build_artifacts(broken, embed=False)
    assert cli.main(["check", "--artifacts-dir", str(broken)], environ={}) == 2
    assert "doc-embed embed" in capsys.readouterr().err
    assert cli.main(["check", "--artifacts-dir", str(broken), "--systems", "B1+"], environ={}) == 0


def test_default_artifacts_dir_is_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli._build_parser({}).parse_args(["check"]).artifacts_dir == "artifacts_trial"
    assert cli.main(["check"], environ={}) == 2
    assert "artifacts_trial" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_artifacts_dir_flag_beats_env_beats_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from_env, from_flag = tmp_path / "env_run", tmp_path / "flag_run"
    build_artifacts(from_env)
    build_artifacts(from_flag, n_resumes=3)
    environ = {cli.ARTIFACTS_DIR_ENV: str(from_env)}

    def resumes(argv: list[str]) -> tuple[str, int]:
        assert cli.main(["check", *argv], environ=environ) == 0
        health = json.loads(capsys.readouterr().out)
        return health["artifacts_dir"], health["counts"]["resumes"]

    assert resumes([]) == (str(from_env), 6)
    assert resumes(["--artifacts-dir", str(from_flag)]) == (str(from_flag), 3)


def test_versioned_artifacts_root_is_resolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trial = tmp_path / "artifacts_trial"
    build_artifacts(trial / "v1", n_resumes=3)
    build_artifacts(trial / "v2")
    for target in (trial, trial / "v2"):
        assert cli.main(["check", "--artifacts-dir", str(target)], environ={}) == 0
        assert json.loads(capsys.readouterr().out)["artifacts_dir"] == str(trial / "v2")


def test_serve_artifacts_with_agent_uses_resolved_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trial = tmp_path / "artifacts_trial"
    build_artifacts(trial / "v3")
    run = RecordingRun()
    argv = ["serve", "--artifacts-dir", str(trial), "--explainer", "agent", "--model", "m"]
    code = cli.main(argv, environ={}, llm_client=FakeLLMClient(script=[]),
                    embedding_client=FakeEmbeddingClient(dim=16), run=run)
    assert code == 0 and "v3" in capsys.readouterr().out
    client = TestClient(run.calls[0][0])
    body = client.get("/api/health").json()
    assert body["explainer"]["kind"] == "agent" and body["artifacts_dir"] == str(trial / "v3")
    assert client.post("/api/chat", json={}).status_code != 404


def test_serve_artifacts_template(tmp_path: Path) -> None:
    root = tmp_path / "art"
    build_artifacts(root)
    run = RecordingRun()
    assert cli.main(["serve", "--artifacts-dir", str(root)], environ={}, run=run) == 0
    client = TestClient(run.calls[0][0])
    reply = client.post("/api/match", json={
        "direction": "resume_to_posting", "query": {"kind": "corpus", "id": "r-1"},
        "system": "B1+", "k": 3,
    })
    assert reply.status_code == 200 and len(reply.json()["matches"]) == 3
    assert client.post("/api/chat", json={}).status_code == 404


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


def test_artifact_capabilities_from_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    build_artifacts(root)
    base = ["check", "--artifacts-dir", str(root)]

    def capabilities(extra: list[str], **clients: object) -> dict:
        assert cli.main([*base, *extra], environ={"VOYAGER_API_KEY": FAKE_SECRET}, **clients) == 0
        captured = capsys.readouterr()
        assert FAKE_SECRET not in captured.out + captured.err
        return json.loads(captured.out)["capabilities"]

    bare = capabilities([])
    assert bare["structure"]["available"] is False
    assert "VOYAGER_MODEL" in bare["structure"]["reason"]
    assert bare["embed"] == {"available": True, "reason": None, "detail": "configured in .env"}
    wired = capabilities(
        ["--model", "m"], llm_client=FakeLLMClient(script=[]),
        embedding_client=FakeEmbeddingClient(dim=16),
    )
    assert wired["structure"]["available"] is True
    assert wired["structure"]["detail"] == "voyager, m"
    assert FAKE_SECRET not in json.dumps(wired)
    assert wired["embed"]["detail"] == "fake, fake-embedding"
    off = capabilities(["--no-structure", "--no-embed"])
    assert off["structure"]["reason"] == "Disabled with --no-structure."
    assert off["embed"]["reason"] == "Disabled with --no-embed."


def test_artifacts_without_any_key_still_check_and_serve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    build_artifacts(root)
    assert cli.main(["check", "--artifacts-dir", str(root)], environ={}) == 0
    structure = json.loads(capsys.readouterr().out)["capabilities"]["structure"]
    assert structure["available"] is False and "VOYAGER_API_KEY" in structure["reason"]
    run = RecordingRun()
    assert cli.main(["serve", "--artifacts-dir", str(root)], environ={}, run=run) == 0
    served = TestClient(run.calls[0][0]).get("/api/health").json()["capabilities"]["structure"]
    assert served["available"] is False and "VOYAGER_API_KEY" in served["reason"]


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
