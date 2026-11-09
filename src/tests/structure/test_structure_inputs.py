from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from getwork.structure import inputs

pytestmark = pytest.mark.usefixtures("offline")

WriteParquet = Callable[[str, Sequence[dict]], Path]


def test_sort_ids_numeric_not_lexical() -> None:
    assert inputs.sort_ids(["r-10", "r-2", "r-1"]) == ["r-1", "r-2", "r-10"]


@pytest.mark.parametrize("bad", ["r", "r-x"])
def test_sort_ids_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match=bad):
        inputs.sort_ids(["r-1", bad])


def test_load_inputs_columns_per_source(write_clean_parquet: WriteParquet) -> None:
    d = write_clean_parquet(
        "resumes",
        [
            {"id": "r-10", "text_clean": "b", "extra": 1},
            {"id": "r-2", "text_clean": "a", "extra": 2},
        ],
    )
    df = inputs.load_inputs("resumes", d)
    assert list(df.columns) == ["id", "text_clean"]
    assert list(df["id"]) == ["r-2", "r-10"]
    assert list(df.index) == [0, 1]
    d = write_clean_parquet(
        "postings",
        [{"id": "j-3", "title": "T", "text_clean": "x", "extra": 1}],
    )
    assert list(inputs.load_inputs("postings", d).columns) == ["id", "title", "text_clean"]


def test_load_inputs_missing_file_names_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="resumes_clean.parquet"):
        inputs.load_inputs("resumes", tmp_path)


def test_load_inputs_rejects_duplicate_ids(write_clean_parquet: WriteParquet) -> None:
    d = write_clean_parquet(
        "resumes",
        [{"id": "r-1", "text_clean": "a"}, {"id": "r-1", "text_clean": "b"}],
    )
    with pytest.raises(ValueError, match="r-1"):
        inputs.load_inputs("resumes", d)


def test_load_inputs_null_title_becomes_none(write_clean_parquet: WriteParquet) -> None:
    d = write_clean_parquet(
        "postings", [{"id": "j-1", "title": None, "text_clean": "x"}]
    )
    assert inputs.load_inputs("postings", d)["title"].iloc[0] is None


@pytest.mark.parametrize("blank", [None, "   "])
def test_load_inputs_rejects_blank_text(write_clean_parquet: WriteParquet, blank: str | None) -> None:
    d = write_clean_parquet(
        "resumes",
        [{"id": "r-1", "text_clean": "ok"}, {"id": "r-2", "text_clean": blank}],
    )
    with pytest.raises(ValueError, match="r-2"):
        inputs.load_inputs("resumes", d)


def test_load_inputs_rejects_wrong_id_prefix(write_clean_parquet: WriteParquet) -> None:
    d = write_clean_parquet("resumes", [{"id": "j-1", "text_clean": "a"}])
    with pytest.raises(ValueError, match="j-1"):
        inputs.load_inputs("resumes", d)


def test_load_inputs_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bogus"):
        inputs.load_inputs("bogus", tmp_path)


def test_select_universe_limit_is_prefix() -> None:
    ids = [f"r-{i}" for i in (5, 3, 1, 4, 2, 10)]
    three = inputs.select_universe(ids, limit=3, sample=None, seed=0)
    five = inputs.select_universe(ids, limit=5, sample=None, seed=0)
    assert three == five[:3] == ["r-1", "r-2", "r-3"]


def test_select_universe_sample_is_seeded_and_sorted() -> None:
    ids = [f"r-{i}" for i in range(1, 51)]
    a = inputs.select_universe(ids, limit=None, sample=10, seed=7)
    b = inputs.select_universe(ids, limit=None, sample=10, seed=7)
    assert a == b == inputs.sort_ids(a)
    assert len(set(a)) == 10
    assert inputs.select_universe(ids, limit=None, sample=10, seed=8) != a


def test_select_universe_rejects_both() -> None:
    with pytest.raises(ValueError):
        inputs.select_universe(["r-1"], limit=1, sample=1, seed=0)


def test_select_universe_limit_edges() -> None:
    ids = ["r-2", "r-1"]
    assert inputs.select_universe(ids, limit=99, sample=None, seed=0) == ["r-1", "r-2"]
    assert inputs.select_universe(ids, limit=0, sample=None, seed=0) == []
