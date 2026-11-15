from __future__ import annotations

import json
import zlib
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from evaluation_helpers import make_eval_tree

from getwork import fileio
from getwork.evaluation import cli, metrics, pairs
from getwork.llm.client import LLMRequest
from getwork.llm.fake import FakeLLMClient
from getwork.retrieval.inputs import DIRECTIONS, SYSTEM_IDS

pytestmark = pytest.mark.usefixtures("offline")

SMALL = [
    "--dev-resumes", "3", "--dev-postings", "3", "--test-resumes", "6", "--test-postings", "6",
    "--human-resumes", "2", "--human-postings", "2", "--pool-cap", "20",
]
FAST = ["--n-boot", "200", "--n-perm", "200"]
SIDES = {"resume_to_posting": "resumes", "posting_to_resume": "postings"}


def _label_by_text(request: LLMRequest) -> str:
    label = zlib.crc32(request.user.encode()) % 3
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
    assert cli.main(["pool", "--stage", "test", *common]) == 0
    return common


def _judged(tmp_path: Path) -> list[str]:
    common = _pooled(tmp_path)
    assert cli.main(["freeze-rubric", *common]) == 0
    client = FakeLLMClient(handler=_label_by_text)
    assert cli.main(["judge", "--stage", "test", *common], llm_client=client, environ={}) == 0
    return common


def _metrics(common: list[str], *extra: str) -> int:
    return cli.main(["metrics", *common, *FAST, *extra])


def _read_metrics(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / "metrics.json").read_text())


def _sets(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / "sets.json").read_text())["stages"]


def _write_author(tmp_path: Path, author: str, *, skip: int = 0) -> None:
    out = tmp_path / "out"
    blind = [json.loads(x) for x in (out / "pairs_blind_human.jsonl").read_text().splitlines()]
    rows = [{"pair_id": r["pair_id"], "author": author, "label": 2 if i % 2 else 1,
             "labeled_at": "t"} for i, r in enumerate(blind)]
    fileio.write_jsonl_atomic(out / "human" / f"{author}.jsonl", rows[skip:])


def _recomputed(tmp_path: Path, direction: str, queries: list[str]) -> dict[str, float]:
    out = tmp_path / "out"
    rankings = pq.read_table(out / "pool_rankings_test.parquet").to_pylist()
    pair_ids = pairs.pair_id_map(pq.read_table(out / "pairs_test.parquet"))
    table = pq.read_table(out / "labels_judge.parquet")
    labels = dict(zip(table.column("pair_id").to_pylist(), table.column("label").to_pylist()))
    top: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for row in rankings:
        if row["direction"] != direction or row["query_id"] not in queries or row["rank"] > 10:
            continue
        pid = pair_ids[pairs.pair_key(direction, row["query_id"], row["doc_id"])]
        top.setdefault(row["system"], {}).setdefault(row["query_id"], []).append(
            (row["rank"], pid))
    means = {}
    for system in SYSTEM_IDS:
        values = []
        for query in sorted(queries):
            ranked = [p for _, p in sorted(top[system][query])]
            pool = {p for s in SYSTEM_IDS for _, p in top[s][query]}
            ideal = metrics.ideal_labels(labels[p] for p in pool)
            values.append(metrics.query_metrics([labels[p] for p in ranked], ideal)["nDCG@10"])
        means[system] = sum(values) / len(values)
    return means


def test_metrics_judge_test_group(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    assert _metrics(common, "--labels", "judge", "--queries", "test") == 0
    group = _read_metrics(tmp_path)["groups"]["judge@test"]
    assert (group["labels"], group["queries"]) == ("judge", "test")
    assert (group["n_boot"], group["n_perm"], group["allow_missing"]) == (200, 200, False)
    assert group["n_missing"] == 0 and set(group["n_no_relevant"]) == set(DIRECTIONS)
    assert group["seed"] == json.loads((tmp_path / "out" / "run.json").read_text())["seed"]
    stages = _sets(tmp_path)["test"]
    for direction in DIRECTIONS:
        block = group["directions"][direction]
        assert set(block["systems"]) == set(SYSTEM_IDS)
        assert set(block["gaps"]) == {"B2-B1+", "Full-B2", "Full-B1+"}
        assert set(block["systems"]["B2"]) == set(metrics.METRICS)
        queries = stages[SIDES[direction]]
        expected = _recomputed(tmp_path, direction, queries)
        for system in SYSTEM_IDS:
            assert block["systems"][system]["nDCG@10"]["mean"] == pytest.approx(expected[system])


def test_metrics_human_queries_subset(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    assert _metrics(common, "--labels", "judge", "--queries", "human") == 0
    group = _read_metrics(tmp_path)["groups"]["judge@human"]
    human = _sets(tmp_path)["human"]
    for direction in DIRECTIONS:
        assert group["directions"][direction]["n_queries"] == len(human[SIDES[direction]])


def test_metrics_merges_groups(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    assert _metrics(common, "--labels", "judge", "--queries", "test") == 0
    first = _read_metrics(tmp_path)["groups"]["judge@test"]
    assert _metrics(common, "--labels", "judge", "--queries", "human") == 0
    groups = _read_metrics(tmp_path)["groups"]
    assert set(groups) == {"judge@test", "judge@human"} and groups["judge@test"] == first


def test_metrics_author_group(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    _write_author(tmp_path, "ann")
    assert _metrics(common, "--labels", "ann", "--queries", "human") == 0
    group = _read_metrics(tmp_path)["groups"]["ann@human"]
    assert group["labels"] == "ann" and group["n_missing"] == 0
    assert set(group["directions"]) == set(DIRECTIONS)


def test_metrics_missing_labels_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = _pooled(tmp_path)
    capsys.readouterr()
    assert _metrics(common, "--labels", "judge", "--queries", "test") == 2
    assert "run evaluate judge --stage test" in capsys.readouterr().err
    assert _metrics(common, "--labels", "ann", "--queries", "human") == 2
    assert _metrics(common, "--labels", "ann", "--queries", "test") == 2
    partial = ["--limit", "5"]
    client = FakeLLMClient(handler=_label_by_text)
    assert cli.main(["freeze-rubric", *common]) == 0
    assert cli.main(["judge", "--stage", "test", *common, *partial], llm_client=client,
                    environ={}) == 0
    capsys.readouterr()
    assert _metrics(common, "--labels", "judge", "--queries", "test") == 2
    assert "run evaluate judge --stage test" in capsys.readouterr().err
    _write_author(tmp_path, "ann", skip=3)
    capsys.readouterr()
    assert _metrics(common, "--labels", "ann", "--queries", "human") == 2
    err = capsys.readouterr().err
    assert "3" in err and "--allow-missing" in err
    assert not (tmp_path / "out" / "metrics.json").exists()


def test_metrics_allow_missing_recorded(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    _write_author(tmp_path, "ann", skip=3)
    assert _metrics(common, "--labels", "ann", "--queries", "human", "--allow-missing") == 0
    group = _read_metrics(tmp_path)["groups"]["ann@human"]
    assert group["allow_missing"] is True and group["n_missing"] > 0
