from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from evaluation_helpers import make_eval_tree

from getwork import fileio
from getwork.evaluation import cli, defaults, judges
from getwork.llm.client import FatalLLMError, LLMRequest, PermanentLLMError
from getwork.llm.fake import FakeLLMClient

pytestmark = pytest.mark.usefixtures("offline")

SMALL = [
    "--dev-resumes", "3", "--dev-postings", "3", "--test-resumes", "6", "--test-postings", "6",
    "--human-resumes", "2", "--human-postings", "2", "--pool-cap", "20",
]
GEMMA, LLAMA = defaults.DEFAULT_JUDGE_MODELS


def _verdict(label: int) -> str:
    return json.dumps({"label": label, "seniority_mismatch": False, "rationale": "because"})


def _by_model(request: LLMRequest) -> str:
    return _verdict(2 if request.model == GEMMA else 1)


def _common(tmp_path: Path) -> list[str]:
    return [
        "--artifacts-dir", str(tmp_path / "tree" / "art"),
        "--processed-dir", str(tmp_path / "tree" / "processed"),
        "--out-dir", str(tmp_path / "out"),
    ]


def _prepared(tmp_path: Path, *, stage: str = "dev") -> list[str]:
    make_eval_tree(tmp_path / "tree")
    common = _common(tmp_path)
    assert cli.main(["sample", *common, *SMALL]) == 0
    assert cli.main(["pool", "--stage", stage, *common]) == 0
    return common


def _judge(common: list[str], *extra: str, client: FakeLLMClient | None = None) -> int:
    return cli.main(["judge", "--stage", "dev", *common, *extra], llm_client=client, environ={})


def _rows(tmp_path: Path, judge_id: str) -> list[dict]:
    path = tmp_path / "out" / "judges" / f"{judge_id}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _n_pairs(tmp_path: Path) -> int:
    return pq.read_table(tmp_path / "out" / "pairs_dev.parquet").num_rows


def test_judge_dev_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _prepared(tmp_path)
    capsys.readouterr()
    assert _judge(common, client=FakeLLMClient(handler=_by_model)) == 0
    summary = json.loads(capsys.readouterr().out)
    n = _n_pairs(tmp_path)
    rows_a, rows_b = _rows(tmp_path, "judge-a"), _rows(tmp_path, "judge-b")
    assert len(rows_a) == len(rows_b) == n == summary["labeled_pairs"]
    assert {r["model"] for r in rows_a} == {GEMMA} and {r["family"] for r in rows_a} == {"google"}
    assert {r["model"] for r in rows_b} == {LLAMA} and {r["family"] for r in rows_b} == {"meta"}
    labels = pq.read_table(tmp_path / "out" / "labels_judge.parquet")
    assert set(labels.column("label").to_pylist()) == {1}
    manifest = json.loads((tmp_path / "out" / "run.json").read_text())
    assert [(j["id"], j["model"], j["family"]) for j in manifest["judges"]] == [
        ("judge-a", GEMMA, "google"), ("judge-b", LLAMA, "meta")]
    assert manifest["rubric_hash"] == judges.load_rubric().hash


def test_judge_resume_makes_no_calls(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    assert _judge(common, client=FakeLLMClient(handler=_by_model)) == 0
    again = FakeLLMClient(handler=_by_model)
    assert _judge(common, client=again) == 0
    assert again.calls == []


def test_dry_run_and_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _prepared(tmp_path)
    n = _n_pairs(tmp_path)
    capsys.readouterr()
    assert _judge(common, "--dry-run") == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry == {"stage": "dev", "pairs": n, "calls": 2 * n, "already_done": 0}
    assert not (tmp_path / "out" / "judges").exists()
    assert "judges" not in json.loads((tmp_path / "out" / "run.json").read_text())
    assert _judge(common, "--limit", "2", client=FakeLLMClient(handler=_by_model)) == 0
    first = sorted(pq.read_table(tmp_path / "out" / "pairs_dev.parquet").column("pair_id")
                   .to_pylist())[:2]
    assert sorted(r["pair_id"] for r in _rows(tmp_path, "judge-a")) == first
    capsys.readouterr()
    assert _judge(common, "--dry-run") == 0
    assert json.loads(capsys.readouterr().out)["already_done"] == 2


def test_failed_pairs_exit_1_and_fatal_exit_3(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    broken = {"on": True}

    def handler(request: LLMRequest) -> str:
        if broken["on"] and request.model == LLAMA and "python sql" in request.user:
            return "not json"
        return _by_model(request)

    assert _judge(common, "--limit", "1", client=FakeLLMClient(handler=handler)) == 1
    assert [r["status"] for r in _rows(tmp_path, "judge-b")] == ["failed"]
    assert pq.read_table(tmp_path / "out" / "labels_judge.parquet").num_rows == 0
    broken["on"] = False
    assert _judge(common, "--limit", "1", client=FakeLLMClient(handler=handler)) == 0
    assert pq.read_table(tmp_path / "out" / "labels_judge.parquet").num_rows == 1

    def fatal(request: LLMRequest) -> str:
        raise FatalLLMError("bad key", status=401)

    other = _prepared(tmp_path / "other")
    assert _judge(other, client=FakeLLMClient(handler=fatal)) == 3


def _no_calls(common: list[str], *extra: str) -> None:
    client = FakeLLMClient(handler=_by_model)
    assert _judge(common, *extra, client=client) == 2
    assert client.calls == []


def test_refusals(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    _no_calls(common, "--judge-models", f"{GEMMA},{LLAMA}", "--judge-families", "g,g")
    _no_calls(common, "--judge-models", f"{GEMMA},{LLAMA}", "--judge-families", "g,m",
              "--also-used-model", GEMMA.upper())
    _no_calls(common, "--judge-models", f"{GEMMA},pipeline-model", "--judge-families", "g,m")
    assert _judge(common, client=FakeLLMClient(handler=_by_model)) == 0
    _no_calls(common, "--judge-models", f"{GEMMA},other-model", "--judge-families", "google,x")
    client = FakeLLMClient(handler=_by_model)
    assert cli.main(["judge", "--stage", "tune", *common], llm_client=client, environ={}) == 2
    assert cli.main(["judge", *common], llm_client=client, environ={}) == 2
    assert client.calls == []


def test_refusal_missing_texts_file(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    for path in (tmp_path / "tree" / "processed" / "postings").rglob("*.parquet"):
        path.unlink()
    _no_calls(common)


def test_override_models_and_families(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    client = FakeLLMClient(handler=lambda r: _verdict(2))
    assert _judge(common, "--judge-models", "mA,mB", "--judge-families", "fa,fb",
                  client=client) == 0
    assert {c.model for c in client.calls} == {"mA", "mB"}
    assert {r["family"] for r in _rows(tmp_path, "judge-b")} == {"fb"}


def test_test_stage_requires_frozen_rubric(tmp_path: Path) -> None:
    common = _prepared(tmp_path, stage="test")
    argv = ["judge", "--stage", "test", *common]
    client = FakeLLMClient(handler=_by_model)
    assert cli.main(argv, llm_client=client, environ={}) == 2
    assert cli.main(["freeze-rubric", *common]) == 0
    frozen = tmp_path / "out" / "rubric_frozen.json"
    good = json.loads(frozen.read_text())
    fileio.write_json_atomic(frozen, {**good, "rubric_hash": "sha256:000000000000"})
    assert cli.main(argv, llm_client=client, environ={}) == 2
    assert client.calls == []
    fileio.write_json_atomic(frozen, good)
    assert cli.main(argv, llm_client=client, environ={}) == 0


def test_requests_are_blind(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    client = FakeLLMClient(handler=_by_model)
    assert _judge(common, client=client) == 0
    rubric = judges.load_rubric().text
    pair_ids = pq.read_table(tmp_path / "out" / "pairs_dev.parquet").column("pair_id").to_pylist()
    assert client.calls
    for call in client.calls:
        assert call.system == rubric
        for banned in (*pair_ids, "B1+", "B2", "resume_to_posting", "posting_to_resume"):
            assert banned not in call.user
        assert call.user.startswith("RESUME:\n") and "\n\nPOSTING:\n" in call.user
    blind = (tmp_path / "out" / "pairs_blind_dev.jsonl").read_text()
    assert "B1+" not in blind and "system" not in blind and "rank" not in blind


def test_default_judges_and_request_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = _prepared(tmp_path)
    client = FakeLLMClient(handler=_by_model)
    capsys.readouterr()
    assert _judge(common, client=client) == 0
    summary = json.loads(capsys.readouterr().out)
    effort = {c.model: c.reasoning_effort for c in client.calls}
    assert effort == {GEMMA: "none", LLAMA: None}
    assert all(c.response_schema == judges.VERDICT_SCHEMA for c in client.calls)
    assert "qwen38-27b" in summary["checked_against"] and "pipeline-model" in summary[
        "checked_against"]
    assert summary["checked_against"] == sorted(summary["checked_against"])
    assert {j["schema_fallback"] for j in summary["judges"].values()} == {False}
    manifest = json.loads((tmp_path / "out" / "run.json").read_text())
    assert manifest["judges"][0]["backend"] == "voyager"


def test_schema_fallback_is_sticky_after_plain_success(tmp_path: Path) -> None:
    common = _prepared(tmp_path)

    def handler(request: LLMRequest) -> str:
        if request.response_schema is not None:
            raise PermanentLLMError("no schema", status=400)
        return _by_model(request)

    client = FakeLLMClient(handler=handler)
    assert _judge(common, "--workers", "1", "--limit", "3", client=client) == 0
    schema_calls = [c for c in client.calls if c.response_schema is not None]
    assert len(schema_calls) == 2  # one probe per judge, then plain only


@pytest.mark.parametrize(("option", "effort"), [("low", {GEMMA: "low", LLAMA: "low"}),
                                               ("omit", {GEMMA: None, LLAMA: None})])
def test_reasoning_effort_options(tmp_path: Path, option: str, effort: dict) -> None:
    common = _prepared(tmp_path)
    client = FakeLLMClient(handler=_by_model)
    assert _judge(common, "--reasoning-effort", option, client=client) == 0
    assert {c.model: c.reasoning_effort for c in client.calls} == effort


def test_independence_runs_on_defaults(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    _no_calls(common, "--judge-models", "qwen38-27b,llama4-maverick-17b",
              "--judge-families", "qwen,meta")
    _no_calls(common, "--also-used-model", LLAMA)
    for path in (tmp_path / "tree" / "art").rglob("*_canonical.jsonl"):
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        fileio.write_jsonl_atomic(path, [{**r, "model": GEMMA} for r in rows])
    _no_calls(common)


def test_models_and_families_must_be_given_together(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    _no_calls(common, "--judge-models", "mA,mB")
    _no_calls(common, "--judge-families", "fa,fb")
    _no_calls(common, "--judge-models", f"{GEMMA},{LLAMA}", "--judge-families", "google,google")


def test_progress_every_option(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _prepared(tmp_path)
    total = _n_pairs(tmp_path)
    assert _judge(common, "--progress-every", "1", "--workers", "3",
                  client=FakeLLMClient(handler=_by_model)) == 0
    captured = capsys.readouterr()
    for judge_id in ("judge-a", "judge-b"):
        assert f"[{judge_id}] judged {total}/{total} pairs ({total} ok, 0 failed, 0 skipped" \
            in captured.err
    assert json.loads(captured.out.splitlines()[-1])["stage"] == "dev"
    assert _judge(common, client=FakeLLMClient(handler=_by_model)) == 0
    resumed = capsys.readouterr().err
    assert f"judged {total}/{total} pairs (0 ok, 0 failed, {total} skipped as already done)" \
        in resumed


def test_progress_every_zero_is_silent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _prepared(tmp_path)
    assert _judge(common, "--progress-every", "0", client=FakeLLMClient(handler=_by_model)) == 0
    assert "judged" not in capsys.readouterr().err


def test_offset_takes_a_slice_after_the_sorted_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = _prepared(tmp_path)
    capsys.readouterr()
    ids = sorted(pq.read_table(tmp_path / "out" / "pairs_dev.parquet").column("pair_id").to_pylist())
    assert len(ids) >= 5
    assert _judge(common, "--limit", "2", client=FakeLLMClient(handler=_by_model)) == 0
    assert _judge(common, "--offset", "2", "--limit", "2",
                  client=FakeLLMClient(handler=_by_model)) == 0
    assert sorted(r["pair_id"] for r in _rows(tmp_path, "judge-a")) == ids[:4]
    assert _judge(common, "--offset", "4", client=FakeLLMClient(handler=_by_model)) == 0
    assert sorted(r["pair_id"] for r in _rows(tmp_path, "judge-b")) == ids


def test_offset_defaults_to_zero_and_rejects_negative(tmp_path: Path) -> None:
    common = _prepared(tmp_path)
    assert _judge(common, "--offset", "-1") == 2
