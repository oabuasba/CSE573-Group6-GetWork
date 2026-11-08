import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from getwork.llm import factory
from getwork.llm.fake import FakeLLMClient, make_output_json
from getwork.structure import cli, store

pytestmark = pytest.mark.usefixtures("offline")


def _fake() -> FakeLLMClient:
    return FakeLLMClient(handler=lambda req: make_output_json())


def _args(tmp_path: Path, *extra: str, source: str = "resumes") -> list[str]:
    return [
        "--source", source,
        "--in-dir", str(tmp_path / "in"),
        "--out-dir", str(tmp_path / "out"),
        "--model", "m",
        *extra,
    ]  # fmt: skip


def test_missing_key_exits_2_before_any_work(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(_args(tmp_path, "--env-file", str(tmp_path / "nope.env")), environ={})
    assert code == 2
    assert "VOYAGER_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "in").exists()


def test_missing_model_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--source", "resumes", "--in-dir", str(tmp_path / "in"),
            "--out-dir", str(tmp_path / "out")]  # fmt: skip
    assert cli.main(argv, client=_fake()) == 2
    assert "--model" in capsys.readouterr().err


def test_unknown_prompt_version_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(_args(tmp_path, "--prompt-version", "nope_v9"), client=_fake())
    assert code == 2
    assert "nope_v9" in capsys.readouterr().err


def test_limit_and_sample_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(_args(tmp_path, "--limit", "1", "--sample", "1"), client=_fake())
    assert info.value.code == 2


@pytest.mark.parametrize(
    "bad", [["--workers", "0"], ["--limit", "-1"], ["--sample", "0"]]
)
def test_parser_rejects_non_positive_numbers(tmp_path: Path, bad: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(_args(tmp_path, *bad), client=_fake())
    assert info.value.code == 2


def _rows(n: int) -> list[dict]:
    return [{"id": f"r-{i}", "text_clean": f"text {i}"} for i in range(1, n + 1)]


def test_dry_run_writes_nothing_and_needs_no_key(
    tmp_path: Path,
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_clean_parquet("resumes", _rows(3))
    argv = _args(tmp_path, "--dry-run", "--limit", "2")
    argv.remove("--model")
    argv.remove("m")
    assert cli.main(argv, environ={}) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["total_in_source"] == 3
    assert report["selected"] == 2
    assert report["already_done"] == 0
    assert report["to_process"] == 2
    assert report["example_user_message"].endswith("text 1")
    assert not (tmp_path / "out").exists()


def test_dry_run_example_is_null_when_nothing_left(
    tmp_path: Path,
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_clean_parquet("resumes", _rows(2))
    out = tmp_path / "out"
    out.mkdir()
    paths = store.output_paths(source="resumes", out_dir=out)
    paths.jsonl.write_text('{"id": "r-1"}\n{"id": "r-2"}\n')
    before = paths.jsonl.read_bytes()
    assert cli.main(_args(tmp_path, "--dry-run")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["already_done"] == 2
    assert report["to_process"] == 0
    assert report["example_user_message"] is None
    assert paths.jsonl.read_bytes() == before
    assert sorted(p.name for p in out.iterdir()) == [paths.jsonl.name]


def test_cli_import_does_not_import_openai() -> None:
    code = "import sys; import getwork.structure.cli; assert 'openai' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_temperature_option_parsing() -> None:
    parse = cli.build_parser().parse_args
    assert parse(["--source", "resumes"]).temperature == 0.0
    assert parse(["--source", "resumes", "--temperature", "omit"]).temperature is None
    assert parse(["--source", "resumes", "--temperature", "0.3"]).temperature == 0.3
    with pytest.raises(SystemExit) as info:
        parse(["--source", "resumes", "--temperature", "abc"])
    assert info.value.code == 2


def test_reasoning_effort_omit_passes_through_raw() -> None:
    args = cli.build_parser().parse_args(["--source", "resumes", "--reasoning-effort", "omit"])
    assert args.reasoning_effort == "omit"


def test_default_dirs_follow_source() -> None:
    args = cli._parse_args(["--source", "postings"])
    assert args.in_dir == Path("data/processed/postings")
    assert args.out_dir == Path("data/structured/postings")


def test_dry_run_missing_input_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(_args(tmp_path, "--dry-run")) == 2
    assert "input parquet not found" in capsys.readouterr().err


def test_dry_run_bad_checkpoint_exits_2(
    tmp_path: Path,
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_clean_parquet("resumes", _rows(1))
    out = tmp_path / "out"
    out.mkdir()
    (out / "resumes_structured.jsonl").write_text("garbage\n")
    assert cli.main(_args(tmp_path, "--dry-run")) == 2
    assert "error:" in capsys.readouterr().err


def _assert_config_error_before_work(tmp_path: Path, code: int) -> None:
    assert code == 2
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "in").exists()


def test_local_backend_missing_settings_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _args(tmp_path, "--backend", "local", "--env-file", str(tmp_path / "nope.env"))
    argv.remove("--model")
    argv.remove("m")
    _assert_config_error_before_work(tmp_path, cli.main(argv, environ={}))
    err = capsys.readouterr().err
    assert "LOCAL_LLM_BASE_URL" in err
    assert "LOCAL_LLM_MODEL" in err


def test_unknown_backend_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(_args(tmp_path, "--backend", "nope"), client=_fake())
    assert info.value.code == 2
    env_file = tmp_path / "bad.env"
    env_file.write_text("LLM_BACKEND=nope\n")
    code = cli.main(_args(tmp_path, "--env-file", str(env_file)), environ={})
    _assert_config_error_before_work(tmp_path, code)
    assert "local" in capsys.readouterr().err


def test_invalid_extra_body_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for bad in ["{not json", "[1]", "3"]:
        code = cli.main(_args(tmp_path, "--extra-body", bad), client=_fake())
        _assert_config_error_before_work(tmp_path, code)
        assert "--extra-body" in capsys.readouterr().err


def test_backend_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "LLM_BACKEND=local\nLOCAL_LLM_BASE_URL=http://localhost:8000/v1\nLOCAL_LLM_MODEL=qwen\n"
    )
    received = []

    def fake_real_client(settings: object, timeout_s: float) -> FakeLLMClient:
        received.append(settings)
        return _fake()

    monkeypatch.setattr(factory, "_real_client", fake_real_client)
    argv = ["--source", "resumes", "--in-dir", str(tmp_path / "in"),
            "--out-dir", str(tmp_path / "out"), "--env-file", str(env_file)]  # fmt: skip
    assert cli.main(argv, environ={}) == 2  # input file missing, so settings were accepted
    assert received[0].backend == "local"
    assert received[0].model == "qwen"
