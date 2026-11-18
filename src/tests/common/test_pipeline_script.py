import os
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "bash" / "run_pipeline.sh"


def test_run_pipeline_names_with_llm_by_default() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'NAME_METHOD="${NAME_METHOD:-llm}"' in text
    assert '--method "$NAME_METHOD"' in text
    assert "--method heuristic" not in text
    lines = text.splitlines()
    branch = lines.index('if [[ "$NAME_METHOD" == "heuristic" ]]; then')
    assert lines[branch + 1].strip() == "BUILD_ARGS=(--allow-member-names)"
    assert text.count("--allow-member-names") == 2
    archive = next(i for i, line in enumerate(lines) if "cluster_names.csv" in line and line.startswith("archive"))
    name = next(i for i, line in enumerate(lines) if "taxonomy name" in line and "--force" in line)
    assert archive < name
    build = next(i for i, line in enumerate(lines) if "taxonomy build" in line)
    assert "${BUILD_ARGS[@]}" in lines[build]


def test_run_pipeline_passes_name_workers() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'NAME_WORKERS="${NAME_WORKERS:-4}"' in text
    lines = text.splitlines()
    name_at = next(i for i, l in enumerate(lines) if "taxonomy name" in l and "--force" in l)
    assert '--workers "$NAME_WORKERS"' in lines[name_at]
    assert '--method "$NAME_METHOD"' in lines[name_at]
    archive_at = next(
        i for i, l in enumerate(lines) if "cluster_names.csv" in l and l.startswith("archive")
    )
    assert archive_at < name_at


def test_run_pipeline_always_selects_cluster_config() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'CLUSTER_CONFIG_ID="${1:-${CLUSTER_CONFIG_ID:-kmeans-01}}"' in text
    steps = [l for l in text.splitlines() if l.startswith("step ")]
    at = {k: next(i for i, l in enumerate(steps) if f"taxonomy {k}" in l) for k in ("grid", "select", "name")}
    assert at["grid"] + 1 == at["select"] < at["name"]
    assert '--config-id "$CLUSTER_CONFIG_ID"' in steps[at["select"]]


def test_run_pipeline_plots_after_build() -> None:
    steps = [l for l in SCRIPT.read_text(encoding="utf-8").splitlines() if l.startswith("step ")]
    at = {k: next(i for i, l in enumerate(steps) if k in l) for k in ("taxonomy build", "relabel")}
    plot = next(i for i, l in enumerate(steps) if l == 'step uv run taxonomy plot "${ART[@]}"')
    assert at["taxonomy build"] < plot < at["relabel"]


def test_run_pipeline_passes_overlap_rounds() -> None:
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    assert 'NAME_OVERLAP_ROUNDS="${NAME_OVERLAP_ROUNDS:-2}"' in lines
    name = next(l for l in lines if l.startswith("step ") and "taxonomy name" in l)
    assert '--overlap-rounds "$NAME_OVERLAP_ROUNDS"' in name
    assert name.index('--workers "$NAME_WORKERS"') < name.index("--overlap-rounds")
    assert '--method "$NAME_METHOD"' in name and "--force" in name
    assert any(l.startswith("archive") and "cluster_names.csv" in l for l in lines)


def test_run_pipeline_passes_fit_rounds() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert 'NAME_FIT_ROUNDS="${NAME_FIT_ROUNDS:-1}"' in lines
    header = [line for line in lines if line.startswith("#")]
    assert any("NAME_FIT_ROUNDS" in line and "0 disables" in line for line in header)
    name = next(line for line in lines if line.startswith("step ") and "taxonomy name" in line)
    assert '--fit-rounds "$NAME_FIT_ROUNDS"' in name
    assert name.index("--overlap-rounds") < name.index("--fit-rounds")


def _run_with_uv_stub(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir, log = tmp_path / "bin", tmp_path / "uv_calls.log"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\n', encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    base = {k: v for k, v in os.environ.items() if k not in ("TOP_N", "CLUSTER_CONFIG_ID")}
    base.update(PATH=f"{bin_dir}{os.pathsep}{base['PATH']}", ARTIFACTS_DIR=str(tmp_path / "art"),
                NAME_METHOD="heuristic")
    result = subprocess.run(["bash", str(SCRIPT), *env.pop("ARGS", "").split()],
                            capture_output=True, text=True, check=False, env={**base, **env})
    return result, log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def test_run_pipeline_top_n_unset_keeps_counts_default(tmp_path: Path) -> None:
    result, calls = _run_with_uv_stub(tmp_path)
    assert result.returncode == 0, result.stderr
    counts = next(c for c in calls if c.startswith("run taxonomy counts"))
    assert "--top-n" not in counts
    assert "TOP_N=default CLUSTER_CONFIG_ID=kmeans-01" in result.stdout


def test_run_pipeline_top_n_is_passed_to_counts_only(tmp_path: Path) -> None:
    result, calls = _run_with_uv_stub(tmp_path, TOP_N="3000", ARGS="kmeans-04")
    assert result.returncode == 0, result.stderr
    assert [c for c in calls if "--top-n" in c] == [
        f"run taxonomy counts --artifacts-dir {tmp_path / 'art'} --top-n 3000"
    ]
    assert f"ARTIFACTS_DIR={tmp_path / 'art'} TOP_N=3000 CLUSTER_CONFIG_ID=kmeans-04" in result.stdout


@pytest.mark.parametrize("value", ["abc", "0", "-5", "10.5", "1e3"])
def test_run_pipeline_rejects_bad_top_n_before_any_step(tmp_path: Path, value: str) -> None:
    result, calls = _run_with_uv_stub(tmp_path, TOP_N=value)
    assert result.returncode == 2 and "TOP_N" in result.stderr
    assert not calls and "==>" not in result.stdout
