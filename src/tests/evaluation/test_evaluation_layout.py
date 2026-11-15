from __future__ import annotations

from pathlib import Path

import pytest
from evaluation_helpers import make_eval_tree
from retrieval_helpers import embed_other_model

from getwork import paths
from getwork.evaluation import defaults, layout
from getwork.retrieval import inputs

pytestmark = pytest.mark.usefixtures("offline")

PACKAGE = Path(layout.__file__).parent
PROTECTED_DEFAULTS = (Path("data"), paths.DEFAULT_ARTIFACTS_DIR)


def test_out_dir_guard(tmp_path: Path) -> None:
    root = tmp_path / "art"
    protected = [Path("data"), root, paths.DEFAULT_ARTIFACTS_DIR]
    assert layout.check_out_dir(Path("eval_out"), protected) == Path("eval_out").resolve()
    fresh = tmp_path / "new" / "out"
    assert layout.check_out_dir(fresh, protected) == fresh.resolve()
    for bad in (Path("data"), Path("data/x"), root, root / "v1", Path("artifacts_trial/anything")):
        with pytest.raises(ValueError, match="protected"):
            layout.check_out_dir(bad, protected)


def test_path_helpers_validate_names(tmp_path: Path) -> None:
    assert layout.rankings_path(tmp_path, "dev").name == "pool_rankings_dev.parquet"
    assert layout.pairs_path(tmp_path, "tune").name == "pairs_tune.parquet"
    assert layout.blind_path(tmp_path, "human").name == "pairs_blind_human.jsonl"
    assert layout.judge_path(tmp_path, "judge-a") == tmp_path / "judges" / "judge-a.jsonl"
    assert layout.human_path(tmp_path, "omar_1") == tmp_path / "human" / "omar_1.jsonl"
    assert layout.tune_dir(tmp_path) == tmp_path / "tune"
    with pytest.raises(ValueError, match="human"):
        layout.rankings_path(tmp_path, "human")
    with pytest.raises(ValueError, match="human"):
        layout.pairs_path(tmp_path, "human")
    with pytest.raises(ValueError, match="x"):
        layout.blind_path(tmp_path, "x")
    with pytest.raises(ValueError, match="judge-c"):
        layout.judge_path(tmp_path, "judge-c")
    for bad in ("", "Omar", "a" * 21, "judge", "a b"):
        with pytest.raises(ValueError, match="author"):
            layout.human_path(tmp_path, bad)
    assert list(tmp_path.iterdir()) == []


def test_manifest_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError) as err:
        layout.read_manifest(out)
    assert str(out / "run.json") in str(err.value) and "evaluate sample" in str(err.value)
    layout.update_manifest(out, {"seed": 1, "a": {"x": 1}})
    merged = layout.update_manifest(out, {"a": {"y": 2}, "b": 3})
    assert merged == {"seed": 1, "a": {"y": 2}, "b": 3}
    assert layout.read_manifest(out) == merged
    assert [p.name for p in out.iterdir()] == ["run.json"]


def _stamps(root: Path) -> dict:
    return layout.stamps_of(inputs.load_inputs(root))


def test_stale_stamps_name_the_key(tmp_path: Path) -> None:
    root, _ = make_eval_tree(tmp_path / "a", n_resumes=6, n_postings=5)
    stamps = _stamps(root)
    assert set(stamps) == {"documents", "embedding", "n_resumes", "n_postings"}
    assert {"taxonomy_version", "alias_table_version", "b2_order"} <= set(stamps["documents"])
    assert stamps["n_resumes"] == 6 and stamps["n_postings"] == 5
    layout.check_stamps({"stamps": stamps}, stamps)

    changed = {**stamps, "n_postings": 9}
    with pytest.raises(layout.StaleRunError, match="n_postings"):
        layout.check_stamps({"stamps": stamps}, changed)
    docs = {**stamps["documents"], "taxonomy_version": "sha256:other"}
    with pytest.raises(layout.StaleRunError, match="documents.taxonomy_version"):
        layout.check_stamps({"stamps": stamps}, {**stamps, "documents": docs})

    for source in paths.SOURCES:
        embed_other_model(root, source)
    with pytest.raises(layout.StaleRunError, match="embedding"):
        layout.check_stamps({"stamps": stamps}, _stamps(root))


def test_defaults_constants() -> None:
    assert defaults.DEFAULT_JUDGE_IDS == ("judge-a", "judge-b")
    assert defaults.DEFAULT_JUDGE_MODELS == ("gemma4-31b-it", "llama4-maverick-17b")
    assert defaults.DEFAULT_JUDGE_FAMILIES == ("google", "meta")
    assert defaults.DEFAULT_ALSO_USED_MODELS == ("qwen38-27b",)
    assert defaults.JUDGE_CONTEXT_TOKENS == 131072
    assert defaults.REASONING_CHOICES == ("auto", "none", "low", "medium", "high", "omit")
    for judge_id in defaults.DEFAULT_JUDGE_IDS:
        assert layout.judge_path(Path("o"), judge_id).name == f"{judge_id}.jsonl"


def test_resolve_reasoning() -> None:
    gemma, llama = defaults.DEFAULT_JUDGE_MODELS
    assert defaults.resolve_reasoning(gemma) == "none"
    assert defaults.resolve_reasoning(llama) is None
    assert defaults.resolve_reasoning("unknown-model") is None
    for model in (gemma, llama):
        assert defaults.resolve_reasoning(model, "omit") is None
        assert defaults.resolve_reasoning(model, "low") == "low"
    with pytest.raises(ValueError, match="turbo"):
        defaults.resolve_reasoning(gemma, "turbo")


def test_judge_constants_live_only_in_defaults() -> None:
    ids = (*defaults.DEFAULT_JUDGE_MODELS, *defaults.DEFAULT_ALSO_USED_MODELS)
    offenders = [
        f"{path.name}: {model}"
        for path in PACKAGE.rglob("*.py")
        if path.name != "defaults.py"
        for model in ids
        if model in path.read_text(encoding="utf-8")
    ]
    assert not offenders
