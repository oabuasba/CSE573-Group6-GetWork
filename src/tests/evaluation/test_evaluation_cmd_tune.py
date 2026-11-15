from __future__ import annotations

import csv
import json
import zlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from evaluation_helpers import make_eval_tree

from getwork import fileio
from getwork.evaluation import cli, defaults, loading, pairs, tune
from getwork.llm.client import LLMRequest
from getwork.llm.fake import FakeLLMClient
from getwork.retrieval import b1plus

pytestmark = pytest.mark.usefixtures("offline")

SMALL = [
    "--dev-resumes", "3", "--dev-postings", "3", "--test-resumes", "6", "--test-postings", "6",
    "--human-resumes", "2", "--human-postings", "2", "--pool-cap", "20",
]
GEMMA = defaults.DEFAULT_JUDGE_MODELS[0]


def _verdict(request: LLMRequest) -> str:
    label = 2 if request.model == GEMMA else 1
    return json.dumps({"label": label, "seniority_mismatch": False, "rationale": "because"})


def _varied(request: LLMRequest) -> str:
    label = zlib.crc32(repr(request).encode()) % 3
    return json.dumps({"label": label, "seniority_mismatch": False, "rationale": "because"})


def _common(tmp_path: Path) -> list[str]:
    return [
        "--artifacts-dir", str(tmp_path / "tree" / "art"),
        "--processed-dir", str(tmp_path / "tree" / "processed"),
        "--out-dir", str(tmp_path / "out"),
    ]


def _pooled(tmp_path: Path) -> list[str]:
    make_eval_tree(tmp_path / "tree")
    common = _common(tmp_path)
    assert cli.main(["sample", *common, *SMALL]) == 0
    assert cli.main(["pool", "--stage", "dev", *common]) == 0
    return common


def _plan(common: list[str], step: str = "0.5") -> int:
    return cli.main(["tune", "plan", "--step", step, *common])


def _expected_new_ids(tmp_path: Path, dev_ids: set[str]) -> set[str]:
    out = tmp_path / "out"
    loaded = loading.load_eval_inputs(tmp_path / "tree" / "art", tmp_path / "tree" / "processed")
    dev = fileio.read_json(out / "sets.json")["stages"]["dev"]
    tables = tune.candidate_rankings(
        loaded.data, resume_queries=dev["resumes"], posting_queries=dev["postings"],
        pool_ids=fileio.read_json(out / "pool_postings.json")["ids"],
        weights=tune.candidate_weights(0.5),
    )
    seed = fileio.read_json(out / "run.json")["seed"]
    every = pairs.pairs_from_rankings(pa.concat_tables(tables), seed)
    return set(every.column("pair_id").to_pylist()) - dev_ids


def test_tune_plan_pairs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _pooled(tmp_path)
    capsys.readouterr()
    assert _plan(common) == 0
    summary = json.loads(capsys.readouterr().out)
    out = tmp_path / "out"
    dev_ids = set(pq.read_table(out / "pairs_dev.parquet").column("pair_id").to_pylist())
    new = pq.read_table(out / "pairs_tune.parquet")
    new_ids = new.column("pair_id").to_pylist()
    blind = [json.loads(x) for x in (out / "pairs_blind_tune.jsonl").read_text().splitlines()]
    assert summary == {
        "candidates": 6, "new_pairs": len(new_ids), "calls_if_judged": 2 * len(new_ids),
    }
    assert new_ids and len(set(new_ids)) == len(new_ids) and not set(new_ids) & dev_ids
    assert set(new_ids) == _expected_new_ids(tmp_path, dev_ids)
    assert sorted(r["pair_id"] for r in blind) == sorted(new_ids)
    assert all(set(r) == {"pair_id", "resume_id", "posting_id"} for r in blind)
    capsys.readouterr()
    assert _plan(common, "0.1") == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == 66


def test_tune_plan_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_eval_tree(tmp_path / "tree")
    common = _common(tmp_path)
    assert cli.main(["sample", *common, *SMALL]) == 0
    capsys.readouterr()
    assert _plan(common) == 2
    assert "evaluate pool --stage dev" in capsys.readouterr().err
    assert cli.main(["pool", "--stage", "dev", *common]) == 0
    capsys.readouterr()
    assert _plan(common, "0.3") == 2
    assert "step" in capsys.readouterr().err


def test_tune_pick_requires_labels(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _pooled(tmp_path)
    assert _plan(common) == 0
    capsys.readouterr()
    assert cli.main(["tune", "pick", "--step", "0.5", *common]) == 2
    first = capsys.readouterr().err
    assert "evaluate judge --stage tune" in first
    judge = ["judge", "--stage", "dev", *common]
    assert cli.main(judge, llm_client=FakeLLMClient(handler=_verdict), environ={}) == 0
    capsys.readouterr()
    assert cli.main(["tune", "pick", "--step", "0.5", *common]) == 2
    err = capsys.readouterr().err
    assert "evaluate judge --stage tune" in err and "no label" in err
    assert err.split("error: ")[1].split(" ")[0].isdigit() and int(err.split()[1]) > 0


def test_tune_pick_writes_weights(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _pooled(tmp_path)
    assert _plan(common) == 0
    client = FakeLLMClient(handler=_varied)
    for stage in ("dev", "tune"):
        assert cli.main(["judge", "--stage", stage, *common], llm_client=client, environ={}) == 0
    capsys.readouterr()
    assert cli.main(["tune", "pick", "--step", "0.5", *common]) == 0
    out = tmp_path / "out"
    weights = b1plus.load_weights(out / "tune" / "b1plus_weights.json")
    with open(out / "tune" / "tune_results.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["skills", "title", "level", "ndcg10"] and len(rows) == 7
    recorded = json.loads((out / "run.json").read_text())["tune"]
    assert recorded["step"] == 0.5 and recorded["weights"] == weights.as_dict()
    body = rows[1:]
    best = max(float(r[3]) for r in body)
    first_best = next(r for r in body if float(r[3]) == best)
    assert recorded["ndcg10"] == best
    assert [recorded["weights"][k] for k in ("skills", "title", "level")] == [
        float(v) for v in first_best[:3]]
    printed = json.loads(capsys.readouterr().out)
    assert printed["weights"] == weights.as_dict() and printed["ndcg10"] == best
