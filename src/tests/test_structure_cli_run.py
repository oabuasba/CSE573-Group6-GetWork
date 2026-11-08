import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from getwork.llm.client import FatalLLMError, LLMRequest
from getwork.llm.fake import FakeLLMClient, make_output_json
from getwork.structure import cli, store

pytestmark = pytest.mark.usefixtures("offline")


def _ok(request: LLMRequest) -> str:
    return make_output_json()


def _fake(handler: Callable[[LLMRequest], str] = _ok, **kw: Any) -> FakeLLMClient:
    return FakeLLMClient(handler=handler, **kw)


def _argv(tmp_path: Path, *extra: str, source: str = "resumes") -> list[str]:
    return [
        "--source", source,
        "--in-dir", str(tmp_path / "in"),
        "--out-dir", str(tmp_path / "out"),
        "--model", "m",
        *extra,
    ]  # fmt: skip


def _rows(count: int) -> list[dict]:
    return [{"id": f"r-{n:05d}", "text_clean": f"doc {n}"} for n in range(1, count + 1)]


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def _jsonl(tmp_path: Path) -> Path:
    return store.output_paths(source="resumes", out_dir=tmp_path / "out").jsonl


def test_missing_input_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(_argv(tmp_path), client=_fake())
    assert code == 2
    assert "resumes_clean.parquet" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_corrupt_checkpoint_exits_2(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_clean_parquet("resumes", _rows(2))
    path = _jsonl(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not json\n")
    before = path.read_bytes()
    assert cli.main(_argv(tmp_path), client=_fake()) == 2
    assert "line 1" in capsys.readouterr().err
    assert path.read_bytes() == before


def test_provenance_mismatch_exits_2_writes_nothing_and_allow_mixed_proceeds(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_clean_parquet("resumes", _rows(2))
    assert cli.main(_argv(tmp_path, "--limit", "1"), client=_fake()) == 0
    before = _snapshot(tmp_path / "out")
    other = _argv(tmp_path)
    other[other.index("--model") + 1] = "other-model"
    capsys.readouterr()
    assert cli.main(other, client=_fake()) == 2
    assert "provenance" in capsys.readouterr().err
    assert _snapshot(tmp_path / "out") == before
    assert cli.main([*other, "--allow-mixed"], client=_fake()) == 0


def test_second_instance_exits_2_writes_nothing(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_clean_parquet("resumes", _rows(2))
    path = _jsonl(tmp_path)
    with store.CheckpointWriter(path):
        before = _snapshot(tmp_path / "out")
        assert cli.main(_argv(tmp_path), client=_fake()) == 2
        assert _snapshot(tmp_path / "out") == before
    assert "another run" in capsys.readouterr().err


def test_main_resumes_end_to_end(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_clean_parquet("resumes", _rows(3))
    assert cli.main(_argv(tmp_path), client=_fake()) == 0
    paths = store.output_paths(source="resumes", out_dir=tmp_path / "out")
    assert paths.jsonl.exists() and paths.parquet.exists() and paths.summary.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete"


def test_main_postings_end_to_end_sends_title(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    write_clean_parquet("postings", [{"id": "j-00001", "title": "Data Wrangler", "text_clean": "t"}])
    fake = _fake()
    assert cli.main(_argv(tmp_path, source="postings"), client=fake) == 0
    assert "TITLE: Data Wrangler" in fake.calls[0].user


def test_preflight_prints_model_info(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    found = _fake(model_info={"id": "m", "found": True})
    assert cli.main(_argv(tmp_path, "--preflight"), client=found) == 0
    assert '"found": true' in capsys.readouterr().out
    missing = _fake(model_info={"id": "m", "found": False})
    assert cli.main(_argv(tmp_path, "--preflight"), client=missing) == 3


def test_preflight_failures_exit_3_or_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    class Failing(FakeLLMClient):
        def describe_model(self, model: str) -> dict:
            raise FatalLLMError("denied", status=401)

    assert cli.main(_argv(tmp_path, "--preflight"), client=Failing(handler=_ok)) == 3
    assert "preflight failed" in capsys.readouterr().err

    class NoDescribe:
        def complete(self, request: LLMRequest) -> str:
            return ""

    assert cli.main(_argv(tmp_path, "--preflight"), client=NoDescribe()) == 2


def test_exit_codes_for_failures_and_fatal(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    write_clean_parquet("resumes", _rows(1))
    assert cli.main(_argv(tmp_path), client=_fake(lambda r: "garbage")) == 1

    def fatal(request: LLMRequest) -> str:
        raise FatalLLMError("denied", status=401)

    assert cli.main(_argv(tmp_path, "--allow-mixed"), client=_fake(fatal)) == 3


def test_options_reach_requests_and_records(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _rows(1)
    rows[0]["text_clean"] = "x" * 100
    write_clean_parquet("resumes", rows)
    fake = _fake()
    code = cli.main(
        _argv(
            tmp_path, "--prompt-version", "extract_v1", "--temperature", "omit",
            "--reasoning-effort", "omit", "--no-response-schema",
            "--max-tokens", "99", "--max-input-chars", "20",
        ),
        client=fake,
    )  # fmt: skip
    assert code == 0
    request = fake.calls[0]
    assert request.temperature is None
    assert request.reasoning_effort is None
    assert request.response_schema is None
    assert request.max_tokens == 99
    summary = json.loads(capsys.readouterr().out)
    assert summary["prompt_version"] == "extract_v1"
    assert summary["truncated_inputs"] == 1
    record = json.loads(_jsonl(tmp_path).read_text().splitlines()[0])
    assert record["prompt_version"] == "extract_v1"


def test_temperature_and_schema_defaults(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    write_clean_parquet("resumes", _rows(1))
    fake = _fake()
    assert cli.main(_argv(tmp_path, "--temperature", "0.3"), client=fake) == 0
    assert fake.calls[0].temperature == 0.3
    assert fake.calls[0].response_schema is not None


def test_sigint_restore_falls_back_to_default_when_previous_handler_is_unknown(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_clean_parquet("resumes", _rows(2))
    installed: list[object] = []

    def fake_signal(signum: int, handler: object) -> None:
        installed.append(handler)  # returns None, like a handler set outside Python

    monkeypatch.setattr(cli.signal, "signal", fake_signal)
    assert cli.main(_argv(tmp_path), client=_fake()) == 0
    assert installed[-1] == cli.signal.SIG_DFL
