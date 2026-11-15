from __future__ import annotations

import json
import zlib
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from evaluation_helpers import make_eval_tree

from getwork import fileio
from getwork.evaluation import agreement, cli, labels, pairs
from getwork.llm.client import LLMRequest
from getwork.llm.fake import FakeLLMClient

pytestmark = pytest.mark.usefixtures("offline")

SMALL = [
    "--dev-resumes", "3", "--dev-postings", "3", "--test-resumes", "6", "--test-postings", "6",
    "--human-resumes", "4", "--human-postings", "4", "--pool-cap", "20",
]
FAST = ["--n-boot", "200", "--n-perm", "200"]


def _label_by_text(request: LLMRequest) -> str:
    label = zlib.crc32(request.user.encode()) % 3
    return json.dumps({"label": label, "seniority_mismatch": False, "rationale": "because"})


def _common(tmp_path: Path) -> list[str]:
    return [
        "--artifacts-dir", str(tmp_path / "tree" / "art"),
        "--processed-dir", str(tmp_path / "tree" / "processed"),
        "--out-dir", str(tmp_path / "out"),
    ]


def _judged(tmp_path: Path, stages: tuple[str, ...] = ("test",)) -> list[str]:
    make_eval_tree(tmp_path / "tree")
    common = _common(tmp_path)
    assert cli.main(["sample", *common, *SMALL]) == 0
    assert cli.main(["freeze-rubric", *common]) == 0
    client = FakeLLMClient(handler=_label_by_text)
    for stage in stages:
        assert cli.main(["pool", "--stage", stage, *common]) == 0
        code = cli.main(["judge", "--stage", stage, *common], llm_client=client, environ={})
        assert code == 0
    return common


def _agreement(common: list[str], *extra: str) -> int:
    return cli.main(["agreement", *common, *extra])


def _read(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / "agreement.json").read_text())


def _blind_ids(tmp_path: Path) -> list[str]:
    lines = (tmp_path / "out" / "pairs_blind_human.jsonl").read_text().splitlines()
    return [json.loads(x)["pair_id"] for x in lines]


def _write_author(tmp_path: Path, author: str, label_of: dict[str, int], ids: list[str]) -> None:
    rows = [{"pair_id": i, "author": author, "label": label_of[i], "labeled_at": "t"} for i in ids]
    fileio.write_jsonl_atomic(tmp_path / "out" / "human" / f"{author}.jsonl", rows)


def _final(tmp_path: Path) -> dict[str, int]:
    return labels.read_labels(tmp_path / "out" / "labels_judge.parquet")


def _two_authors(tmp_path: Path) -> list[str]:
    ids, final = _blind_ids(tmp_path), _final(tmp_path)
    differ = set(ids[::3])
    _write_author(tmp_path, "a", final, ids)
    _write_author(tmp_path, "b", {i: (final[i] + 1) % 3 if i in differ else final[i] for i in ids},
                  ids)
    return ids


def test_scope_judges(tmp_path: Path) -> None:
    common = _judged(tmp_path, ("dev", "test"))
    assert _agreement(common, "--scope", "judges", "--stage", "dev") == 0
    block = _read(tmp_path)["judges"]["dev"]
    dev_pairs = pq.read_table(tmp_path / "out" / "pairs_dev.parquet")
    assert block["n"] == len(pairs.pair_id_map(dev_pairs))
    assert block["threshold"] == 0.6
    assert block["meets_threshold"] == (block["kappa"] is not None and block["kappa"] >= 0.6)
    assert _agreement(common, "--scope", "judges", "--stage", "test") == 0
    assert set(_read(tmp_path)["judges"]) == {"dev", "test"}
    (tmp_path / "out" / "judges" / "judge-b.jsonl").unlink()
    assert _agreement(common, "--scope", "judges", "--stage", "dev") == 2


def test_scope_human_two_authors(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    ids = _two_authors(tmp_path)
    assert cli.main(["metrics", *common, *FAST, "--labels", "judge", "--queries", "human"]) == 0
    assert _agreement(common, "--scope", "human") == 0
    block = _read(tmp_path)["human"]
    assert block["authors"] == ["a", "b"] and block["n_pairs"] == len(ids)
    assert block["author_vs_author"]["kappa"] == block["author_kappa"]
    assert set(block["judge_vs_author"]) == {"judge-a", "judge-b", "final"}
    assert block["judge_vs_author"]["final"]["a"]["pct_agree"] == 1.0
    four = [block["judge_vs_author"][j][a]["kappa"] for j in ("judge-a", "judge-b") for a in "ab"]
    present = [k for k in four if k is not None]
    assert block["human_judge_kappa"] == pytest.approx(sum(present) / len(present))
    assert block["fallback"] == {
        "threshold": 0.6,
        "decision": agreement.fallback_decision(block["author_kappa"], block["human_judge_kappa"]),
    }
    assert block["tau"] is None  # no author@human groups in metrics.json yet
    for author in "ab":
        args = ["metrics", *common, *FAST, "--labels", author, "--queries", "human"]
        assert cli.main(args) == 0
    assert _agreement(common, "--scope", "human") == 0
    assert set(_read(tmp_path)["human"]["tau"]) == {"nDCG@10", "gaps"}


def test_scope_human_half_second(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    ids = _two_authors(tmp_path)
    out = tmp_path / "out"
    halves = agreement.split_halves(fileio.read_json(out / "sets.json"))
    pair_ids = pairs.pair_id_map(pq.read_table(out / "pairs_test.parquet"))
    expected = agreement.half_pair_ids(pq.read_table(out / "pool_rankings_test.parquet"),
                                       halves, pair_ids)
    assert _agreement(common, "--scope", "human", "--half", "second") == 0
    block = _read(tmp_path)["human"]
    assert block["half"] == "second" and block["n_pairs"] == len(expected["second"])
    assert block["n_pairs"] < len(ids) and block["tau"] is None


def test_scope_human_one_author_and_errors(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    assert _agreement(common, "--scope", "human") == 2  # no author files
    ids, final = _blind_ids(tmp_path), _final(tmp_path)
    _write_author(tmp_path, "a", final, ids[:-2])
    assert _agreement(common, "--scope", "human") == 2  # two pairs missing
    _write_author(tmp_path, "a", final, ids)
    assert _agreement(common, "--scope", "human") == 0
    block = _read(tmp_path)["human"]
    assert block["author_vs_author"] is None and block["author_kappa"] is None
    assert block["fallback"]["decision"] == "undetermined"
    for extra in ("b", "c"):
        _write_author(tmp_path, extra, final, ids)
    assert _agreement(common, "--scope", "human") == 2  # more than two authors


def test_missing_count_in_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _judged(tmp_path)
    ids, final = _blind_ids(tmp_path), _final(tmp_path)
    _write_author(tmp_path, "a", final, ids[:-2])
    assert _agreement(common, "--scope", "human") == 2
    assert "2 of" in capsys.readouterr().err


def test_blocks_do_not_clobber(tmp_path: Path) -> None:
    common = _judged(tmp_path)
    _two_authors(tmp_path)
    assert _agreement(common, "--scope", "human") == 0
    human_block = _read(tmp_path)["human"]
    assert _agreement(common, "--scope", "judges", "--stage", "test") == 0
    merged = _read(tmp_path)
    assert merged["human"] == human_block and "test" in merged["judges"]
