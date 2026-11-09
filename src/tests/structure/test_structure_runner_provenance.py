import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from getwork.llm.client import FatalLLMError, LLMRequest, PermanentLLMError
from getwork.llm.fake import FakeLLMClient, make_output_json
from getwork.structure import runner, schema, store
from getwork.structure.extract import ExtractConfig
from getwork.structure.runstate import RunConfig

pytestmark = pytest.mark.usefixtures("offline")

MODEL = "m1"
SECRET = "sk-test-123"
PARTIAL_TAIL = '{"id": "r-0000'


def _rows(ids: list[str]) -> list[dict]:
    return [{"id": i, "text_clean": f"doc {i}"} for i in ids]


def _cfg(in_dir: Path, out_dir: Path, **kw: Any) -> RunConfig:
    extract = kw.pop("extract", ExtractConfig(model=MODEL))
    return RunConfig(source="resumes", in_dir=in_dir, out_dir=out_dir, extract=extract, **kw)


def _paths(out_dir: Path) -> store.OutputPaths:
    return store.output_paths(source="resumes", out_dir=out_dir)


def _seed(
    out_dir: Path, schema_version: str = schema.SCHEMA_VERSION, legacy: bool = False
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = schema.parse_model_output(make_output_json(title="T"))
    record = schema.to_record(output, "r-00001", "resumes", MODEL, "extract_v1")
    record["schema_version"] = schema_version
    if legacy:
        del record["backend"]
    _paths(out_dir).jsonl.write_text(json.dumps(record) + "\n" + PARTIAL_TAIL)


def _snapshot(out_dir: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(out_dir.rglob("*")) if p.is_file()}


def _ok(request: LLMRequest) -> str:
    return make_output_json(title="T")


CASES = [
    pytest.param(
        ExtractConfig(model="other"), schema.SCHEMA_VERSION, ["other", MODEL], id="model"
    ),
    pytest.param(
        ExtractConfig(model=MODEL, prompt_version="extract_v2"),
        schema.SCHEMA_VERSION,
        ["extract_v2", "extract_v1"],
        id="prompt_version",
    ),
    pytest.param(ExtractConfig(model=MODEL), "0", ["'0'", "'1'"], id="schema_version"),
]


@pytest.mark.parametrize(("extract", "schema_version", "named"), CASES)
def test_provenance_mismatch_refuses_and_writes_nothing(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    extract: ExtractConfig,
    schema_version: str,
    named: list[str],
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"
    _seed(out_dir=out_dir, schema_version=schema_version)
    before = _snapshot(out_dir)
    client = FakeLLMClient(handler=_ok)
    with pytest.raises(runner.ProvenanceMismatchError) as info:
        runner.run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir, extract=extract), client=client)
    assert _snapshot(out_dir) == before
    assert client.calls == []
    assert all(value in str(info.value) for value in named)


def test_backend_mismatch_refuses_and_writes_nothing(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"
    _seed(out_dir=out_dir)
    before = _snapshot(out_dir)
    client = FakeLLMClient(handler=_ok)
    extract = ExtractConfig(model=MODEL, backend="local")
    with pytest.raises(runner.ProvenanceMismatchError) as info:
        runner.run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir, extract=extract), client=client)
    assert _snapshot(out_dir) == before
    assert client.calls == []
    assert "backend" in str(info.value)


def test_legacy_records_count_as_voyager(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"
    _seed(out_dir=out_dir, legacy=True)
    local = ExtractConfig(model=MODEL, backend="local")
    before = _snapshot(out_dir)
    refused = FakeLLMClient(handler=_ok)
    with pytest.raises(runner.ProvenanceMismatchError, match="backend"):
        runner.run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir, extract=local), client=refused)
    assert _snapshot(out_dir) == before and refused.calls == []
    result = runner.run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir), client=FakeLLMClient(handler=_ok))
    assert result.exit_code == 0
    records = store.read_checkpoint(_paths(out_dir).jsonl)
    assert len(records) == 2
    assert {r["backend"] for r in records.values()} == {"voyager"}


def test_legacy_records_with_allow_mixed_do_not_crash_write_final(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"
    _seed(out_dir=out_dir, legacy=True)
    extract = ExtractConfig(model=MODEL, backend="local")
    cfg = _cfg(in_dir=in_dir, out_dir=out_dir, extract=extract, allow_mixed=True)
    result = runner.run(cfg=cfg, client=FakeLLMClient(handler=_ok))
    assert result.exit_code == 0
    backends = {r["id"]: r["backend"] for r in pq.read_table(_paths(out_dir).parquet).to_pylist()}
    assert backends == {"r-00001": "voyager", "r-00002": "local"}


def test_final_outputs_contain_backend_for_every_record(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002", "r-00003"]))
    out_dir = tmp_path / "out"
    _seed(out_dir=out_dir, legacy=True)
    runner.run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir), client=FakeLLMClient(handler=_ok))
    paths = _paths(out_dir)
    lines = [json.loads(line) for line in paths.jsonl.read_text().splitlines()]
    rows = pq.read_table(paths.parquet).to_pylist()
    assert len(lines) == len(rows) == 3
    assert all(r["backend"] == "voyager" for r in lines + rows)


def test_allow_mixed_proceeds(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"
    _seed(out_dir=out_dir)
    client = FakeLLMClient(handler=_ok)
    extract = ExtractConfig(model="other")
    result = runner.run(
        cfg=_cfg(in_dir=in_dir, out_dir=out_dir, extract=extract, allow_mixed=True), client=client
    )
    assert result.exit_code == 0
    assert len(client.calls) == 1
    assert len(store.read_checkpoint(_paths(out_dir).jsonl)) == 2


def test_secret_never_reaches_any_output(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ids = ["r-00001", "r-00002", "r-00003"]
    in_dir = write_clean_parquet("resumes", _rows(ids))
    out_dir = tmp_path / "out"

    def handler(request: LLMRequest) -> str:
        if "doc r-00001" in request.user:
            raise PermanentLLMError(f"bad {SECRET}")
        if "doc r-00002" in request.user:
            return f"not json {SECRET}"
        return _ok(request)

    logged: list[str] = []
    cfg = _cfg(in_dir=in_dir, out_dir=out_dir, progress_every=1)
    client = FakeLLMClient(handler=handler)
    runner.run(cfg=cfg, client=client, secrets=[SECRET], log=logged.append, sleep=lambda s: None)
    paths = _paths(out_dir)
    captured = capsys.readouterr()
    outputs = [
        paths.jsonl.read_text(),
        paths.failures.read_text(),
        paths.summary.read_text(),
        str(pq.read_table(paths.parquet).to_pylist()),
        *logged,
        captured.out,
        captured.err,
    ]
    assert logged
    assert all(SECRET not in text for text in outputs)
    failures = paths.failures.read_text()
    assert "r-00001" in failures
    assert "r-00002" in failures
    assert "***" in failures


def test_fatal_result_still_writes_outputs(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"

    def handler(request: LLMRequest) -> str:
        if "doc r-00002" in request.user:
            raise FatalLLMError(status=401, message="denied")
        return _ok(request)

    client = FakeLLMClient(handler=handler)
    result = runner.run(
        cfg=_cfg(in_dir=in_dir, out_dir=out_dir, workers=1), client=client, sleep=lambda s: None
    )
    assert result.exit_code == 3
    paths = _paths(out_dir)
    assert paths.parquet.exists()
    assert paths.failures.exists()
    assert paths.summary.exists()


def test_fatal_error_message_secret_is_redacted_everywhere(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"

    def handler(request: LLMRequest) -> str:
        if "doc r-00002" in request.user:
            raise FatalLLMError(status=401, message=f"bad key {SECRET}")
        return _ok(request)

    logged: list[str] = []
    result = runner.run(
        cfg=_cfg(in_dir=in_dir, out_dir=out_dir, workers=1, progress_every=1),
        client=FakeLLMClient(handler=handler),
        secrets=[SECRET],
        log=logged.append,
        sleep=lambda s: None,
    )
    assert result.exit_code == 3
    assert "***" in result.summary["abort_reason"]
    paths = _paths(out_dir)
    captured = capsys.readouterr()
    texts = [
        json.dumps(result.summary),
        paths.jsonl.read_text(),
        paths.failures.read_text(),
        paths.summary.read_text(),
        str(pq.read_table(paths.parquet).to_pylist()),
        *logged,
        captured.out,
        captured.err,
    ]
    assert all(SECRET not in text for text in texts)
