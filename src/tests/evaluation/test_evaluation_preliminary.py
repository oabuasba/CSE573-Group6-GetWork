"""The preliminary script (no tuning, no human labels) and the chain it drives, all offline."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import zlib
from pathlib import Path

import pytest
from evaluation_helpers import make_eval_tree

from getwork import fileio
from getwork.evaluation import cli, report
from getwork.llm.client import LLMRequest
from getwork.llm.fake import FakeLLMClient

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "bash" / "run_evaluation_preliminary.sh"
STAGES = ("prepare", "dev-judge", "freeze", "test-prepare", "test-judge", "metrics", "report")
SMALL = [
    "--dev-resumes", "3", "--dev-postings", "3", "--test-resumes", "6", "--test-postings", "6",
    "--human-resumes", "4", "--human-postings", "4", "--pool-cap", "20",
]
FAST = ["--n-boot", "100", "--n-perm", "100"]


def _install_uv_stub(tmp_path: Path) -> tuple[Path, Path]:
    """A fake uv on PATH that records its arguments and runs nothing."""
    bin_dir, log = tmp_path / "bin", tmp_path / "uv_calls.log"
    bin_dir.mkdir(parents=True)
    stub = bin_dir / "uv"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\n', encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return bin_dir, log


def _run(
    tmp_path: Path, *args: str, **env: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir, log = _install_uv_stub(tmp_path)
    base = {k: v for k, v in os.environ.items() if k not in ("USER_RUN_JUDGE", "EVAL_DIR")}
    base.update(PATH=f"{bin_dir}{os.pathsep}{base['PATH']}", EVAL_DIR=str(tmp_path / "out"))
    result = subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, check=False,
        env={**base, **env},
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, calls


def test_script_static_safety() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0
    assert "set -euo pipefail" in text
    assert re.search(r'^STAGES="' + " ".join(STAGES) + '"$', text, re.MULTILINE)
    assert "--force" not in text.replace("never passes --force", "")
    assert ".env" not in text
    for forbidden in ("tune", "--weights", "--scope human"):
        code = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
        assert forbidden not in code, forbidden


def test_script_rejects_bad_usage(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "tune-plan")
    assert result.returncode == 2 and "usage" in result.stderr and not calls
    assert _run(tmp_path / "x", "prepare", "report")[0].returncode == 2


@pytest.mark.parametrize("args", [(), ("all",), ("smoke",), ("dev-judge",), ("test-judge",)])
def test_script_judge_stages_refuse_without_user_run_judge(
    tmp_path: Path, args: tuple[str, ...]
) -> None:
    result, calls = _run(tmp_path, *args)
    assert result.returncode == 2
    assert "USER_RUN_JUDGE=1" in result.stderr
    assert "==>" not in result.stdout and not calls


def test_script_judge_models_and_families_together(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "prepare", JUDGE_MODELS="a,b")
    assert result.returncode == 2 and "together" in result.stderr and not calls


def test_script_all_runs_stages_in_order_without_tune_or_human(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, USER_RUN_JUDGE="1", ALSO_USED_MODELS="m1 m2", JUDGE_LIMIT="3")
    assert result.returncode == 0, result.stderr
    heads = [" ".join(c.split()[:3]) for c in calls]
    assert heads == [
        "run evaluate sample", "run evaluate pool", "run evaluate judge", "run evaluate judge",
        "run evaluate agreement", "run evaluate freeze-rubric", "run evaluate pool",
        "run evaluate judge", "run evaluate agreement", "run evaluate metrics",
        "run evaluate metrics", "run evaluate ari", "run evaluate report",
    ]
    joined = "\n".join(calls)
    assert "tune" not in joined and "--weights" not in joined and "--force" not in joined
    assert "--scope human" not in joined and "--labels judge --queries human" in joined
    assert "--also-used-model m1 --also-used-model m2" in joined and "--limit 3" in joined
    assert result.stdout.count("==> uv run evaluate") == len(calls)


def test_script_offline_stage_runs_without_user_run_judge(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "test-prepare")
    assert result.returncode == 0
    assert len(calls) == 1 and "pool" in calls[0] and "--stage test" in calls[0]
    assert "--weights" not in calls[0]


def _handler(request: LLMRequest) -> str:
    label = zlib.crc32(request.user.encode()) % 3
    return json.dumps({"label": label, "seniority_mismatch": False, "rationale": "because"})


def _run_preliminary_chain(tmp_path: Path) -> Path:
    make_eval_tree(tmp_path / "tree")
    out, tree = tmp_path / "out", tmp_path / "tree"
    common = [
        "--artifacts-dir", str(tree / "art"), "--processed-dir", str(tree / "processed"),
        "--out-dir", str(out),
    ]
    client = FakeLLMClient(handler=_handler)
    judge = ["--max-chars", "20"]
    steps: list[tuple[list[str], bool]] = [
        (["sample", *SMALL], False), (["pool", "--stage", "dev"], False),
        (["judge", "--stage", "dev", *judge], True),
        (["agreement", "--scope", "judges", "--stage", "dev"], False),
        (["freeze-rubric"], False), (["pool", "--stage", "test"], False),
        (["judge", "--stage", "test", *judge], True),
        (["agreement", "--scope", "judges", "--stage", "test"], False),
        (["metrics", "--labels", "judge", "--queries", "test", *FAST], False),
        (["metrics", "--labels", "judge", "--queries", "human", *FAST], False),
        (["ari"], False), (["report"], False),
    ]
    for argv, needs_client in steps:
        kwargs = {"llm_client": client, "environ": {}} if needs_client else {}
        assert cli.main([*argv, *common], **kwargs) == 0, argv
    return out


def test_preliminary_chain_reports_without_tune_or_human(tmp_path: Path) -> None:
    out = _run_preliminary_chain(tmp_path)
    assert not (out / "human").exists() and not (out / "tune").exists()
    manifest = fileio.read_json(out / "run.json")
    assert manifest["weights"]["test"]["source"] == "default"
    text = (out / "report.md").read_text(encoding="utf-8")
    assert report.UNTUNED_CAVEAT in text and report.NO_HUMAN_CAVEAT in text
    assert "source default" in text and "| B1+ |" in text
    assert "| judge@human |" not in text


def _banners(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if re.match(r"^(\[\d+/\d+\] |done: )", ln)]


def test_script_all_prints_numbered_banners_and_done_lines(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, "all", USER_RUN_JUDGE="1")
    assert result.returncode == 0, result.stderr
    lines = _banners(result.stdout)
    assert [ln for ln in lines if ln.startswith("[")] == [
        f"[{i}/7] {name}" for i, name in enumerate(STAGES, start=1)
    ]
    for name in STAGES:
        banner = lines.index(f"[{STAGES.index(name) + 1}/7] {name}")
        assert re.fullmatch(rf"done: {name} \(\d+s\)", lines[banner + 1])
    assert len(lines) == 14


def test_script_single_stage_is_one_of_one(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "freeze")
    assert result.returncode == 0 and len(calls) == 1
    assert _banners(result.stdout)[0] == "[1/1] freeze"
    assert re.fullmatch(r"done: freeze \(\d+s\)", _banners(result.stdout)[1])
    assert result.stdout.index("[1/1] freeze") < result.stdout.index("==> uv run")


def test_script_smoke_numbers_three_steps(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, "smoke", USER_RUN_JUDGE="1")
    assert result.returncode == 0, result.stderr
    assert [ln for ln in _banners(result.stdout) if ln.startswith("[")] == [
        "[1/3] prepare", "[2/3] preflight", "[3/3] dev-judge",
    ]


def test_script_failure_has_no_done_line(tmp_path: Path) -> None:
    bin_dir, _ = _install_uv_stub(tmp_path)
    (bin_dir / "uv").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "EVAL_DIR": str(tmp_path / "out")}
    result = subprocess.run(["bash", str(SCRIPT), "freeze"], capture_output=True, text=True,
                            check=False, env=env)
    assert result.returncode != 0 and "[1/1] freeze" in result.stdout
    assert "done:" not in result.stdout


def test_script_optional_flags_absent_by_default(tmp_path: Path) -> None:
    _, calls = _run(tmp_path, USER_RUN_JUDGE="1")
    joined = "\n".join(calls)
    assert "--rubric-file" not in joined and "--test-resumes" not in joined
    assert "--test-postings" not in joined


def test_script_passes_rubric_file_and_test_sizes_only_where_needed(tmp_path: Path) -> None:
    rubric = tmp_path / "judge_x.md"
    rubric.write_text("rubric", encoding="utf-8")
    result, calls = _run(
        tmp_path, USER_RUN_JUDGE="1", RUBRIC_FILE=str(rubric), TEST_RESUMES="20",
        TEST_POSTINGS="30",
    )
    assert result.returncode == 0, result.stderr
    flag = f"--rubric-file {rubric}"
    with_rubric = {"judge", "agreement", "freeze-rubric"}
    for call in calls:
        command = call.split()[2]
        assert (flag in call) == (command in with_rubric), call
        assert ("--test-resumes 20 --test-postings 30" in call) == (command == "sample"), call
    assert sum("agreement" in c and flag in c for c in calls) == 2


@pytest.mark.parametrize("name", ["TEST_RESUMES", "TEST_POSTINGS"])
@pytest.mark.parametrize("value", ["abc", "0", "-3", "2.5", "9"])
def test_script_rejects_bad_test_sizes_up_front(tmp_path: Path, name: str, value: str) -> None:
    result, calls = _run(tmp_path, "prepare", **{name: value})
    assert result.returncode == 2 and name in result.stderr and "at least 10" in result.stderr
    assert not calls and "==>" not in result.stdout


def test_script_rejects_missing_rubric_file(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "freeze", RUBRIC_FILE=str(tmp_path / "nope.md"))
    assert result.returncode == 2 and "RUBRIC_FILE" in result.stderr and not calls


def test_sample_with_test_sets_below_human_size_exits_2(tmp_path: Path) -> None:
    make_eval_tree(tmp_path / "tree")
    tree = tmp_path / "tree"
    code = cli.main([
        "sample", "--artifacts-dir", str(tree / "art"), "--processed-dir",
        str(tree / "processed"), "--out-dir", str(tmp_path / "out"), *SMALL[:6],
        "--test-resumes", "3", "--test-postings", "3",
    ])
    assert code == 2 and not (tmp_path / "out" / "sets.json").exists()
