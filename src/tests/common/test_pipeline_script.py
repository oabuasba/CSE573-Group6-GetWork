from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_pipeline.sh"


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
