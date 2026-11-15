from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from getwork.evaluation import cli

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "src" / "getwork" / "evaluation" / "README.md"
SCRIPT = ROOT / "scripts" / "run_evaluation.sh"
END_TO_END = ROOT / "docs" / "running-end-to-end.md"
REQUIRED_STRINGS = (
    "evaluate sample", "evaluate pool", "evaluate judge", "--dry-run", "--limit", "freeze-rubric",
    "tune plan", "tune pick", "evaluate metrics", "evaluate agreement", "evaluate ari",
    "evaluate report", "eval_out", "pairs_blind_human.jsonl", "human/", ".env", "user-run",
    "Voyager", "EV-1", "EV-2", "EV-3", "EV-4", "EV-5", "EV-6",
)
STAGES = (
    "prepare", "dev-judge", "tune-plan", "tune-judge", "tune-pick", "freeze", "test-prepare",
    "test-judge", "metrics", "report",
)


def _command_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_block = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_block = not in_block
        elif in_block and line.startswith("uv run evaluate"):
            lines.append(line)
    return lines


def test_readme_strings_and_commands_parse() -> None:
    text = README.read_text(encoding="utf-8")
    for needle in REQUIRED_STRINGS:
        assert needle in text, needle
    commands = _command_lines(text)
    assert len(commands) >= 10
    parser = cli.build_parser()
    for line in commands:
        tokens = shlex.split(line)[3:]
        argv = ["x" if re.fullmatch(r"<.*>", token) else token for token in tokens]
        parser.parse_args(argv)


def test_script_stages_and_safety() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0
    assert "set -euo pipefail" in text
    assert 'EVAL_DIR="${EVAL_DIR:-eval_out}"' in text
    assert re.search(r'^STAGES="' + " ".join(STAGES) + '"$', text, re.MULTILINE)
    assert "--force" not in text.replace("never passes --force", "")
    assert ".env" not in text
    assert not re.search(r"--out-dir\s+\"?(data|artifacts)", text)
    assert not re.search(r">>?\s*\"?(data|artifacts)", text)
    for stage in STAGES:
        assert re.search(rf"^\s+{re.escape(stage)}\)", text, re.MULTILINE), stage
    result = subprocess.run(
        ["bash", str(SCRIPT), "not-a-stage"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "usage" in result.stderr
    assert "all" in result.stderr and "smoke" in result.stderr


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "USER_RUN_JUDGE"}
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, check=False, env=env
    )


@pytest.mark.parametrize("args", [(), ("all",), ("smoke",)])
def test_script_group_stages_refuse_without_user_run_judge(args: tuple[str, ...]) -> None:
    result = _run_script(*args)
    assert result.returncode == 2
    assert "USER_RUN_JUDGE=1" in result.stderr
    assert "==>" not in result.stdout


def test_script_rejects_extra_arguments() -> None:
    assert _run_script("prepare", "report").returncode == 2


def test_script_documents_all_smoke_and_artifacts_dir() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'STAGE="${1:-all}"' in text
    assert 'JUDGE_LIMIT="${JUDGE_LIMIT:-10}"' in text
    assert "ARTIFACTS_DIR=artifacts_trial/v6 EVAL_DIR=eval_out/v6" in text
    smoke = text[text.index("  smoke)\n    require_user_run"):]
    assert "--preflight" in smoke and "freeze" not in smoke and "test" not in smoke.split("esac")[0]


def test_end_to_end_doc_section() -> None:
    text = END_TO_END.read_text(encoding="utf-8")
    heading = re.search(r"^## \d+\. Evaluation$", text, re.MULTILINE)
    assert heading is not None
    assert "../src/getwork/evaluation/README.md" in text[heading.end():]
