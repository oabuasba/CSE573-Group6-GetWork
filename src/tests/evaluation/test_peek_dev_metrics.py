"""scripts/python/other/peek_dev_metrics.py on tiny fixtures; offline and read only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from getwork import fileio
from getwork.evaluation import pairs

pytestmark = pytest.mark.usefixtures("offline")

SCRIPT = (Path(__file__).resolve().parents[3] / "scripts" / "python" / "other"
          / "peek_dev_metrics.py")
V1, V3 = "sha256:87ecca1b14a7", "sha256:cccccccccccc"
SEED = 0
R2P, P2R = "resume_to_posting", "posting_to_resume"
ORDERS = {
    R2P: {"B1+": ["j1", "j2", "j3"], "B2": ["j3", "j2", "j1"], "Full": ["j2", "j1", "j3"]},
    P2R: {"B1+": ["r1", "r2"], "B2": ["r2", "r1"], "Full": ["r1", "r2"]},
}
QUERY = {R2P: "r1", P2R: "p1"}


@pytest.fixture(scope="module")
def peek():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("peek_dev_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rankings() -> pa.Table:
    rows = [
        {"stage": "dev", "system": system, "direction": d, "query_id": QUERY[d], "rank": rank,
         "doc_id": doc, "score": 1.0}
        for d, systems in ORDERS.items() for system, docs in systems.items()
        for rank, doc in enumerate(docs, start=1)
    ]
    return pa.Table.from_pylist(rows)


def _pair_id(direction: str, doc: str) -> str:
    key = pairs.pair_key(direction, QUERY[direction], doc)
    return pairs.make_pair_id(SEED, *key)


def _row(judge: str, pair_id: str, label: int, rubric_hash: str = V1, status: str = "ok") -> dict:
    return {"pair_id": pair_id, "judge": judge, "rubric_hash": rubric_hash, "status": status,
            "label": label, "effective_label": label}


def _make_dir(out: Path, judged: dict[str, list[dict]], *, drop_pair: bool = False) -> None:
    rankings = _rankings()
    table = pairs.pairs_from_rankings(rankings, SEED)
    if drop_pair:
        table = table.slice(1)
    out.mkdir(parents=True)
    pq.write_table(rankings, out / "pool_rankings_dev.parquet")
    pq.write_table(table, out / "pairs_dev.parquet")
    fileio.write_json_atomic(out / "sets.json", {"stages": {"dev": {"resumes": ["r1"],
                                                                      "postings": ["p1"]}}})
    fileio.write_json_atomic(out / "run.json", {"seed": SEED})
    for judge_id, rows in judged.items():
        path = out / "judges" / f"{judge_id}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _both(*items: tuple[str, int, int]) -> dict[str, list[dict]]:
    """(pair id, label a, label b) to rows of both judges."""
    return {"judge-a": [_row("judge-a", p, a) for p, a, _ in items],
            "judge-b": [_row("judge-b", p, b) for p, _, b in items]}


def _two_dirs(tmp_path: Path) -> list[Path]:
    first, second = tmp_path / "first", tmp_path / "second"
    noise = [_row("judge-a", _pair_id(R2P, "j3"), 2, rubric_hash=V3),
             _row("judge-a", _pair_id(P2R, "r2"), 2, status="failed")]
    rows = _both((_pair_id(R2P, "j1"), 2, 2), (_pair_id(R2P, "j2"), 1, 2))
    rows["judge-a"] += noise
    _make_dir(first, rows)
    _make_dir(second, _both((_pair_id(P2R, "r1"), 2, 2), (_pair_id(P2R, "r2"), 0, 1)))
    return [first, second]


def test_collect_labels_pools_folders_and_applies_the_harness_rules(peek, tmp_path: Path) -> None:
    labels = peek.collect_labels(_two_dirs(tmp_path), V1)
    assert labels == {
        _pair_id(R2P, "j1"): 2, _pair_id(R2P, "j2"): 1,
        _pair_id(P2R, "r1"): 2, _pair_id(P2R, "r2"): 0,
    }
    assert peek.collect_labels(_two_dirs(tmp_path / "again"), V3) == {}


def test_coverage_lower_bound_and_labeled_only(peek, tmp_path: Path) -> None:
    dirs = _two_dirs(tmp_path)
    labels = peek.collect_labels(dirs, V1)
    rankings, pair_ids, queries = peek._load_dev(dirs[0])
    top = peek._top(rankings, pair_ids, queries)
    cov = peek.coverage(top, queries, labels)
    assert cov[R2P]["pooled"] == 3 and cov[R2P]["labeled"] == 2
    assert cov[R2P]["systems"]["B1+"]["slots"] == pytest.approx(0.2)
    kw = {"n_boot": 50, "n_perm": 50, "seed": SEED}
    lower = peek.lower_bound(rankings, pair_ids, queries, labels, **kw)
    assert lower[R2P]["systems"]["B1+"]["P@10"]["mean"] == pytest.approx(0.1)
    assert lower[R2P]["systems"]["B1+"]["MRR"]["mean"] == 1.0
    assert lower[R2P]["systems"]["B2"]["MRR"]["mean"] == pytest.approx(1 / 3)
    only = peek.labeled_only(top, queries, labels, **kw)
    group = only[R2P]["group"]
    assert only[R2P]["n_used"] == 1 and only[R2P]["n_skipped"] == 0
    assert group["systems"]["B1+"]["P@10"]["mean"] == pytest.approx(0.5)
    assert group["systems"]["B1+"]["MRR"]["mean"] == 1.0
    assert group["systems"]["B2"]["MRR"]["mean"] == pytest.approx(0.5)
    assert group["systems"]["B1+"]["nDCG@10"]["mean"] == pytest.approx(1.0)


def test_labeled_only_skips_queries_without_labels(peek, tmp_path: Path) -> None:
    dirs = _two_dirs(tmp_path)
    rankings, pair_ids, queries = peek._load_dev(dirs[0])
    top = peek._top(rankings, pair_ids, queries)
    only = peek.labeled_only(top, queries, {}, n_boot=10, n_perm=10, seed=SEED)
    assert only[R2P] == {"n_used": 0, "n_skipped": 1, "group": None}


def test_report_text_and_mismatch_abort(
    peek, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dirs = _two_dirs(tmp_path)
    text = peek.build_report(dirs, V1, n_boot=50, n_perm=50)
    for needle in ("dev stage only", "## Coverage", "## Lower bound", "## Labeled slots only",
                   "## nDCG@10 gaps, lower bound", "Full-B2", "resume_to_posting (1 queries"):
        assert needle in text, needle
    before = sorted(p.name for p in tmp_path.rglob("*"))
    assert peek.main(["--dir", str(dirs[0]), "--dir", str(dirs[1]), "--n-boot", "20",
                      "--n-perm", "20"]) == 0
    assert "## Coverage" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.rglob("*")) == before
    odd = tmp_path / "odd"
    _make_dir(odd, _both(), drop_pair=True)
    assert peek.main(["--dir", str(dirs[0]), "--dir", str(odd)]) == 2
    assert "dev pair ids" in capsys.readouterr().err
