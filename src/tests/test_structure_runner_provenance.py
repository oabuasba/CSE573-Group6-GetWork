import json
from pathlib import Path

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


def _cfg(in_dir: Path, out_dir: Path, **kw) -> RunConfig:
    extract = kw.pop("extract", ExtractConfig(model=MODEL))
    return RunConfig(source="resumes", in_dir=in_dir, out_dir=out_dir, extract=extract, **kw)


def _paths(out_dir: Path) -> store.OutputPaths:
    return store.output_paths(source="resumes", out_dir=out_dir)


def _seed(out_dir: Path, schema_version: str = schema.SCHEMA_VERSION) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = schema.parse_model_output(make_output_json(title="T"))
    record = schema.to_record(output, "r-00001", "resumes", MODEL, "extract_v1")
    record["schema_version"] = schema_version
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
    write_clean_parquet, tmp_path, extract, schema_version, named
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"
    _seed(out_dir=out_dir, schema_version=schema_version)
    before = _snapshot(out_dir)
    client = FakeLLMClient(handler=_ok)
    with pytest.raises(runner.ProvenanceMismatchError) as info:
        runner.run(_cfg(in_dir, out_dir, extract=extract), client)
    assert _snapshot(out_dir) == before
    assert client.calls == []
    assert all(value in str(info.value) for value in named)


def test_allow_mixed_proceeds(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"
    _seed(out_dir=out_dir)
    client = FakeLLMClient(handler=_ok)
    extract = ExtractConfig(model="other")
    result = runner.run(_cfg(in_dir, out_dir, extract=extract, allow_mixed=True), client)
    assert result.exit_code == 0
    assert len(client.calls) == 1
    assert len(store.read_checkpoint(_paths(out_dir).jsonl)) == 2


def test_secret_never_reaches_any_output(write_clean_parquet, tmp_path, capsys) -> None:
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
    cfg = _cfg(in_dir, out_dir, progress_every=1)
    client = FakeLLMClient(handler=handler)
    runner.run(cfg, client, secrets=[SECRET], log=logged.append, sleep=lambda s: None)
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


def test_fatal_result_still_writes_outputs(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(["r-00001", "r-00002"]))
    out_dir = tmp_path / "out"

    def handler(request: LLMRequest) -> str:
        if "doc r-00002" in request.user:
            raise FatalLLMError(status=401, message="denied")
        return _ok(request)

    client = FakeLLMClient(handler=handler)
    result = runner.run(_cfg(in_dir, out_dir, workers=1), client, sleep=lambda s: None)
    assert result.exit_code == 3
    paths = _paths(out_dir)
    assert paths.parquet.exists()
    assert paths.failures.exists()
    assert paths.summary.exists()
