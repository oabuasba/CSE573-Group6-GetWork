import sys
from pathlib import Path

import pytest
from agent_helpers import chat_json, good_answer
from fastapi import FastAPI
from fastapi.testclient import TestClient

from getwork import paths
from getwork.alias.cli import TABLE_FILE
from getwork.alias.table import make_table, save_table
from getwork.llm import factory as llm_factory
from getwork.llm.fake import FakeLLMClient
from getwork.taxonomy.model import Cluster, make_taxonomy, save_taxonomy
from getwork.webapp import cli, demo, models

pytestmark = pytest.mark.usefixtures("offline")

FAKE_SECRET = "fake-secret-xyz-789"


class RecordingRun:
    def __init__(self) -> None:
        self.calls: list[FastAPI] = []

    def __call__(self, app: FastAPI, **kwargs: object) -> None:
        self.calls.append(app)


def serve(
    *options: str, client: FakeLLMClient | None = None, environ: dict | None = None
) -> tuple[int, RecordingRun]:
    run = RecordingRun()
    code = cli.main(
        ["serve", "--demo", *options], environ=environ or {}, llm_client=client, run=run
    )
    return code, run


def health_of(run: RecordingRun) -> dict:
    return TestClient(run.calls[0]).get("/api/health").json()["explainer"]


def test_serve_agent_demo() -> None:
    fake = FakeLLMClient(script=[good_answer()])
    code, run = serve("--explainer", "agent", "--model", "demo-model", client=fake)
    assert code == 0 and len(run.calls) == 1
    assert health_of(run) == {
        "kind": "agent", "model": "demo-model", "backend": "voyager", "prompt_version": "agent_v1",
    }  # fmt: skip
    response = TestClient(run.calls[0]).post("/api/chat", json=chat_json())
    assert response.status_code == 200 and response.json()["explainer"]["kind"] == "agent"


def test_agent_defaults_to_structuring_backend() -> None:
    options = ["--explainer", "agent", "--backend", "local", "--model", "struct-model"]
    code, run = serve(*options, client=FakeLLMClient(script=[]))
    assert code == 0
    info = health_of(run)
    assert (info["backend"], info["model"]) == ("local", "struct-model")
    code, run = serve("--explainer", "agent", "--model", "m", client=FakeLLMClient(script=[]))
    assert health_of(run)["backend"] == "voyager"


def test_agent_options_override(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    real = llm_factory.make_client

    def spy(**kwargs: object):  # noqa: ANN202
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(llm_factory, "make_client", spy)
    options = ["--explainer", "agent", "--model", "struct", "--agent-model", "agent-m"]
    options += ["--agent-backend", "local", "--agent-timeout", "12.5"]
    code, run = serve(*options, client=FakeLLMClient(script=[]))
    assert code == 0
    info = health_of(run)
    assert (info["backend"], info["model"]) == ("local", "agent-m")
    assert seen["timeout_s"] == 12.5 and seen["backend"] == "local" and seen["model"] == "agent-m"


def test_explicit_template_has_no_chat() -> None:
    code, run = serve("--explainer", "template", client=FakeLLMClient(script=[]))
    assert code == 0
    client = TestClient(run.calls[0])
    assert client.get("/api/health").json()["explainer"] == {"kind": "template", "model": None}
    assert client.post("/api/chat", json=chat_json()).status_code == 404
    query = demo.DemoEngine().get_document("r-1").model_dump(mode="json")
    body = {"direction": "resume_to_posting", "query": query, "match_id": "j-1"}
    explained = client.post("/api/explain", json=body).json()
    assert explained["explainer"] == {"kind": "template", "model": None} and explained["claims"] == []


def write_artifacts(root: Path) -> None:
    table = make_table({"reactjs": "React"})
    save_table(paths.alias_dir(root) / TABLE_FILE, table, {})
    clusters = [Cluster(id=0, name="Front-end", skills=("React",))]
    save_taxonomy(paths.taxonomy_dir(root) / "taxonomy.json", make_taxonomy(clusters, [], table.version, {}))


def test_agent_config_errors_exit_2(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, _ = serve("--explainer", "agent", environ={"VOYAGER_API_KEY": FAKE_SECRET})
    captured = capsys.readouterr()
    assert code == 2 and "VOYAGER_MODEL" in captured.err
    assert FAKE_SECRET not in captured.out + captured.err
    code, _ = serve("--explainer", "agent", "--model", "m", environ={"VOYAGER_BASE_URL": "https://example.invalid"})
    captured = capsys.readouterr()
    assert code == 2 and "VOYAGER_API_KEY" in captured.err
    assert "example.invalid" not in captured.out + captured.err
    monkeypatch.setitem(sys.modules, "langgraph", None)
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)
    code, _ = serve("--explainer", "agent", "--model", "m", client=FakeLLMClient(script=[]))
    assert code == 2 and "uv sync --extra agent" in capsys.readouterr().err
    monkeypatch.undo()
    root = tmp_path.resolve() / "artifacts"
    args = cli._build_parser().parse_args(
        ["check", "--artifacts-dir", str(root), "--explainer", "agent", "--model", "m"]
    )
    with pytest.raises(FileNotFoundError, match="alias-table build"):
        cli.build_agent(args, engine=demo.DemoEngine(), environ={}, llm_client=FakeLLMClient(script=[]))
    write_artifacts(root)
    agent, chat, secrets = cli.build_agent(
        args, engine=demo.DemoEngine(), environ={}, llm_client=FakeLLMClient(script=[])
    )
    assert agent.info.kind == "agent" and chat.info.model == "m" and list(secrets) == []


def test_check_with_agent_makes_no_call(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeLLMClient(script=[])
    code = cli.main(["check", "--demo", "--explainer", "agent", "--model", "m"], environ={}, llm_client=fake)
    assert code == 0 and fake.calls == []
    health = models.HealthResponse.model_validate_json(capsys.readouterr().out)
    assert health.explainer.kind == "agent" and health.explainer.model == "m"


def test_build_agent_loads_vocabulary_from_the_resolved_root(tmp_path: Path) -> None:
    trial = tmp_path.resolve() / "artifacts_trial"
    write_artifacts(trial / "v5")

    class ResolvedEngine(demo.DemoEngine):
        def health(self) -> models.EngineHealth:
            return super().health().model_copy(update={"artifacts_dir": str(trial / "v5")})

    args = cli._build_parser().parse_args(
        ["check", "--artifacts-dir", str(trial), "--explainer", "agent", "--model", "m"]
    )
    agent, _, _ = cli.build_agent(
        args, engine=ResolvedEngine(), environ={}, llm_client=FakeLLMClient(script=[])
    )
    assert agent.info.kind == "agent"


def test_auto_is_the_default_and_uses_the_agent_when_settings_resolve() -> None:
    fake = FakeLLMClient(script=[good_answer()])
    code, run = serve("--model", "demo-model", client=fake)
    assert code == 0
    assert health_of(run)["kind"] == "agent" and health_of(run)["model"] == "demo-model"
    response = TestClient(run.calls[0]).post("/api/chat", json=chat_json())
    assert response.status_code == 200
    assert cli._build_parser().parse_args(["check"]).explainer == "auto"


def test_auto_without_settings_falls_back_to_template(capsys: pytest.CaptureFixture[str]) -> None:
    code, run = serve(environ={"VOYAGER_BASE_URL": "https://example.invalid"})
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err.count("chat agent disabled") == 1
    assert "VOYAGER_API_KEY" in captured.err and "VOYAGER_MODEL" in captured.err
    assert "example.invalid" not in captured.out + captured.err
    info = health_of(run)
    assert info["kind"] == "template" and "VOYAGER_API_KEY" in info["reason"]
    assert TestClient(run.calls[0]).post("/api/chat", json=chat_json()).status_code == 404


def test_auto_without_langgraph_falls_back_to_template(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "langgraph", None)
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)
    code, run = serve("--model", "m", client=FakeLLMClient(script=[]))
    assert code == 0 and "uv sync --extra agent" in capsys.readouterr().err
    assert health_of(run)["kind"] == "template"
    assert TestClient(run.calls[0]).post("/api/chat", json=chat_json()).status_code == 404


def test_explicit_agent_still_exits_2_without_settings(capsys: pytest.CaptureFixture[str]) -> None:
    code, run = serve("--explainer", "agent")
    assert code == 2 and run.calls == [] and "VOYAGER_API_KEY" in capsys.readouterr().err


def test_auto_model_flag_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLMClient(script=[])
    monkeypatch.setattr(llm_factory, "_real_client", lambda settings, timeout_s: fake)
    env = {"VOYAGER_API_KEY": FAKE_SECRET, "VOYAGER_BASE_URL": "https://x.invalid", "VOYAGER_MODEL": "env-model"}
    _, run = serve(environ=env)
    assert health_of(run)["model"] == "env-model"
    _, run = serve("--model", "flag-model", environ=env)
    assert health_of(run)["model"] == "flag-model"


def test_check_auto_reports_explainer_and_reason_without_a_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeLLMClient(script=[])
    assert cli.main(["check", "--demo", "--model", "m"], environ={}, llm_client=fake) == 0
    assert fake.calls == []
    assert models.HealthResponse.model_validate_json(capsys.readouterr().out).explainer.kind == "agent"
    assert cli.main(["check", "--demo"], environ={}) == 0
    captured = capsys.readouterr()
    health = models.HealthResponse.model_validate_json(captured.out)
    assert health.explainer.kind == "template" and "VOYAGER_API_KEY" in (health.explainer.reason or "")
    assert "chat agent disabled" in captured.err


def test_auto_falls_back_on_missing_artifacts_and_rejects_bad_extra_body(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    run = RecordingRun()
    root = tmp_path.resolve() / "artifacts"
    argv = ["serve", "--demo", "--model", "m", "--artifacts-dir", str(root)]
    assert cli.main(argv, environ={}, llm_client=FakeLLMClient(script=[]), run=run) == 0
    assert health_of(run)["kind"] == "agent"
    monkey_args = cli._build_parser().parse_args(["check", "--model", "m", "--artifacts-dir", str(root)])
    choice = cli._choose_agent(
        monkey_args, demo.DemoEngine(), environ={}, llm_client=FakeLLMClient(script=[])
    )
    assert choice.parts is None and "alias-table build" in (choice.fallback_reason or "")
    assert "chat agent disabled" in capsys.readouterr().err
    code, run = serve("--model", "m", "--extra-body", "[1]", client=FakeLLMClient(script=[]))
    assert code == 2 and run.calls == [] and "--extra-body" in capsys.readouterr().err
