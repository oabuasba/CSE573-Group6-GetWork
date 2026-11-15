from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation_helpers import make_eval_tree

from getwork.evaluation import cli, defaults
from getwork.llm.client import LLMError
from getwork.llm.fake import FakeLLMClient

pytestmark = pytest.mark.usefixtures("offline")

GEMMA, LLAMA = defaults.DEFAULT_JUDGE_MODELS
OK_INFO = {"id": "m", "found": True, "supports_response_schema": True}


def _preflight(tmp_path: Path, client: object, *extra: str) -> int:
    return cli.main(["judge", "--preflight", "--out-dir", str(tmp_path / "out"), *extra],
                    llm_client=client, environ={})


def _fake(info: dict) -> FakeLLMClient:
    return FakeLLMClient(handler=lambda r: "", model_info=info)


def test_preflight_prints_one_line_per_judge(tmp_path, capsys):
    assert _preflight(tmp_path, _fake(OK_INFO)) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["judge"], r["model"], r["family"]) for r in lines] == [
        (j, m, f) for j, m, f in zip(defaults.DEFAULT_JUDGE_IDS, defaults.DEFAULT_JUDGE_MODELS,
                                     defaults.DEFAULT_JUDGE_FAMILIES)
    ]
    assert all(r["info"] == OK_INFO for r in lines)


def test_preflight_does_not_need_stage_or_files(tmp_path):
    assert _preflight(tmp_path, _fake(OK_INFO)) == 0
    assert not (tmp_path / "out" / "judges").exists()


def test_preflight_notes_missing_schema_support(tmp_path, capsys):
    assert _preflight(tmp_path, _fake({"id": "m", "found": True})) == 0
    err = capsys.readouterr().err
    assert f"note: {GEMMA} does not report supports_response_schema; " in err
    assert "the plain-JSON path will be used" in err


def test_preflight_missing_model_exits_3_after_all_judges(tmp_path, capsys):
    assert _preflight(tmp_path, _fake({"id": "m", "found": False})) == 3
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 2
    assert GEMMA in captured.err and LLAMA in captured.err


def test_preflight_with_dry_run_is_usage_error(tmp_path):
    assert _preflight(tmp_path, _fake(OK_INFO), "--dry-run") == 2


def test_preflight_client_without_describe_model_is_usage_error(tmp_path, capsys):
    assert _preflight(tmp_path, object()) == 2
    assert "this client does not support --preflight" in capsys.readouterr().err


def test_preflight_llm_error_is_redacted_exit_3(tmp_path, capsys):
    class Failing:
        def describe_model(self, model: str) -> dict:
            raise LLMError("boom")

    assert _preflight(tmp_path, Failing()) == 3
    assert "error: preflight failed: " in capsys.readouterr().err


def test_preflight_independence_failure_is_usage_error(tmp_path, capsys):
    code = _preflight(tmp_path, _fake(OK_INFO), "--judge-models", f"{GEMMA},qwen38-27b",
                      "--judge-families", "google,alibaba")
    assert code == 2
    assert capsys.readouterr().out == ""


def test_preflight_checks_canonical_models_when_artifacts_resolve(tmp_path):
    make_eval_tree(tmp_path / "tree")
    code = _preflight(tmp_path, _fake(OK_INFO), "--artifacts-dir", str(tmp_path / "tree" / "art"),
                      "--also-used-model", LLAMA)
    assert code == 2


def test_preflight_unresolved_artifacts_dir_is_a_note(tmp_path, capsys):
    assert _preflight(tmp_path, _fake(OK_INFO), "--artifacts-dir", str(tmp_path / "nope")) == 0
    assert "note: canonical models not checked" in capsys.readouterr().err


def test_preflight_found_and_missing(tmp_path, capsys):
    assert _preflight(tmp_path, _fake(OK_INFO)) == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 2
    assert not any((tmp_path / "out").glob("*")) if (tmp_path / "out").exists() else True
    assert _preflight(tmp_path, _fake({"id": "m", "found": False})) == 3
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 2
    assert GEMMA in captured.err and LLAMA in captured.err


@pytest.mark.parametrize("info", [
    {"id": "m", "found": True},
    {"id": "m", "found": True, "supports_response_schema": False},
])
def test_preflight_schema_warning(tmp_path, capsys, info):
    assert _preflight(tmp_path, _fake(info)) == 0
    err = capsys.readouterr().err
    for model in (GEMMA, LLAMA):
        assert (f"note: {model} does not report supports_response_schema; "
                "the plain-JSON path will be used") in err
    assert _preflight(tmp_path, _fake(OK_INFO)) == 0
    assert "supports_response_schema" not in capsys.readouterr().err


def test_preflight_usage_errors(tmp_path, capsys):
    assert _preflight(tmp_path, _fake(OK_INFO), "--dry-run") == 2
    capsys.readouterr()
    assert _preflight(tmp_path, object()) == 2
    assert "this client does not support --preflight" in capsys.readouterr().err
    code = cli.main(["judge", "--out-dir", str(tmp_path / "out")],
                    llm_client=_fake(OK_INFO), environ={})
    assert code == 2
    assert "--stage" in capsys.readouterr().err
