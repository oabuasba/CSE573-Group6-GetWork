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
