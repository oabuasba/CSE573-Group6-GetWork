from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation_helpers import make_eval_tree

from getwork.evaluation import cli

pytestmark = pytest.mark.usefixtures("offline")

SMALL = [
    "--dev-resumes", "3", "--dev-postings", "3", "--test-resumes", "6", "--test-postings", "6",
    "--human-resumes", "4", "--human-postings", "4", "--pool-cap", "20",
]


def _common(tmp_path: Path) -> list[str]:
    return [
        "--artifacts-dir", str(tmp_path / "tree" / "art"),
        "--processed-dir", str(tmp_path / "tree" / "processed"),
        "--out-dir", str(tmp_path / "out"),
    ]


def test_report_after_sample_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_eval_tree(tmp_path / "tree")
    common = _common(tmp_path)
    assert cli.main(["sample", *common, *SMALL]) == 0
    capsys.readouterr()
    assert cli.main(["report", *common]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    target = tmp_path / "out" / "report.md"
    assert printed["report"] == str(target)
    assert printed["sections_not_run"] == ["metrics", "agreement", "ari", "frozen"]
    text = target.read_text(encoding="utf-8")
    assert "## Run" in text and "## Caveats" in text
    assert text.count("_not run yet_") >= 4


def test_report_requires_sample(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report", *_common(tmp_path)]) == 2
    assert "evaluate sample" in capsys.readouterr().err
    assert not (tmp_path / "out" / "report.md").exists()
