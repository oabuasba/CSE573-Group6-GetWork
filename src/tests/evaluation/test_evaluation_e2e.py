from __future__ import annotations

import hashlib
import json
import random
import zlib
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from evaluation_helpers import make_eval_tree

from getwork import fileio
from getwork.evaluation import cli, labels
from getwork.llm.client import LLMRequest
from getwork.llm.fake import FakeLLMClient

pytestmark = pytest.mark.usefixtures("offline")

SMALL = [
    "--dev-resumes", "3", "--dev-postings", "3", "--test-resumes", "6", "--test-postings", "6",
    "--human-resumes", "4", "--human-postings", "4", "--pool-cap", "20",
]
FAST = ["--n-boot", "100", "--n-perm", "100"]
TRUNCATING = ["--max-chars", "20"]
AUTHORS = ("a", "b")


def _handler(request: LLMRequest) -> str:
    label = zlib.crc32(request.user.encode()) % 3
    return json.dumps({"label": label, "seniority_mismatch": False, "rationale": "because"})


def _snapshot(root: Path) -> list[tuple[str, str]]:
    return sorted(
        (str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in root.rglob("*") if p.is_file()
    )


def _write_humans(out: Path) -> None:
    blind = (out / "pairs_blind_human.jsonl").read_text().splitlines()
    ids = [json.loads(line)["pair_id"] for line in blind]
    final = labels.read_labels(out / "labels_judge.parquet")
    shifter = random.Random(1)
    for author in AUTHORS:
        picked = {
            i: final[i] if author == "a" or shifter.random() < 0.7 else (final[i] + 1) % 3
            for i in ids
        }
        rows = [{"pair_id": i, "author": author, "label": picked[i], "labeled_at": "t"}
                for i in ids]
        fileio.write_jsonl_atomic(out / "human" / f"{author}.jsonl", rows)


def _run_chain(tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    tree = tmp_path / "tree"
    common = [
        "--artifacts-dir", str(tree / "art"), "--processed-dir", str(tree / "processed"),
        "--out-dir", str(out),
    ]
    client = FakeLLMClient(handler=_handler)

    def judge(stage: str) -> list[str]:
        return ["judge", "--stage", stage, *TRUNCATING]

    steps: list[tuple[list[str], bool]] = [
        (["sample", *SMALL], False), (["pool", "--stage", "dev"], False),
        (judge("dev"), True), (["agreement", "--scope", "judges", "--stage", "dev"], False),
        (["tune", "plan", "--step", "0.5"], False), (judge("tune"), True),
        (["tune", "pick", "--step", "0.5"], False), (["freeze-rubric"], False),
        (["pool", "--stage", "test", "--weights-file", str(out / "tune" / "b1plus_weights.json")],
         False),
        (judge("test"), True),
    ]
    for argv, needs_client in steps:
        kwargs = {"llm_client": client, "environ": {}} if needs_client else {}
        assert cli.main([*argv, *common], **kwargs) == 0, argv
    _write_humans(out)
    tail = [
        ["metrics", "--labels", "judge", "--queries", "test", *FAST],
        ["metrics", "--labels", "judge", "--queries", "human", *FAST],
        *(["metrics", "--labels", a, "--queries", "human", *FAST] for a in AUTHORS),
        ["agreement", "--scope", "human"], ["ari"], ["report"],
    ]
    for argv in tail:
        assert cli.main([*argv, *common]) == 0, argv
    return out


def test_full_chain_offline(tmp_path: Path) -> None:
    make_eval_tree(tmp_path / "tree")
    before = _snapshot(tmp_path / "tree")
    out = _run_chain(tmp_path, "out1")
    text = (out / "report.md").read_text(encoding="utf-8")
    for group in ("judge@human", "a@human", "b@human"):
        assert f"| {group} |" in text
    assert "| B1+ |" in text and "| Full |" in text
    assert "_not run yet_" not in text
    manifest = fileio.read_json(out / "run.json")
    frozen = fileio.read_json(out / "rubric_frozen.json")
    for judge in manifest["judges"]:
        assert judge["id"] in text
    assert _truncated_counts(text, manifest) and frozen["rubric_hash"] in text
    assert _snapshot(tmp_path / "tree") == before


def _truncated_counts(text: str, manifest: dict) -> bool:
    rows = [line for line in text.splitlines() if line.startswith("| judge-")]
    cells = [[c.strip() for c in line.strip("|").split("|")] for line in rows]
    return len(cells) == len(manifest["judges"]) and all(
        c[3].isdigit() and int(c[3]) > 0 for c in cells
    )


def test_chain_is_reproducible(tmp_path: Path) -> None:
    make_eval_tree(tmp_path / "tree")
    first, second = _run_chain(tmp_path, "out1"), _run_chain(tmp_path, "out2")
    for name in ("sets.json", "metrics.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    name = "pool_rankings_test.parquet"
    assert pq.read_table(first / name).equals(pq.read_table(second / name))
