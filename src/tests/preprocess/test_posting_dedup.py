import pandas as pd

from getwork.preprocess.posting_dedup import drop_duplicate_postings


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": f"j-{job_id}", "job_id": job_id, "title": title, "text_clean": text}
            for job_id, title, text in rows
        ]
    )


def test_duplicates_keep_smallest_job_id():
    df = _frame(
        [
            ("100", "Data  Engineer", "Build  pipelines.\nNow"),
            ("5", "other", "x"),
            ("99", "data engineer", "BUILD pipelines. now"),
        ]
    )
    kept, dropped = drop_duplicate_postings(df)
    assert list(kept["id"]) == ["j-5", "j-99"]
    assert dropped.to_dict("records") == [
        {"id": "j-100", "rule": "duplicate", "detail": "duplicate_of=j-99"}
    ]


def test_same_description_different_title_kept():
    df = _frame([("1", "Data Engineer", "same"), ("2", "ML Engineer", "same")])
    kept, dropped = drop_duplicate_postings(df)
    assert list(kept["id"]) == ["j-1", "j-2"]
    assert dropped.empty


def test_three_way_group():
    df = _frame([("30", "t", "d"), ("4", "t", "d"), ("200", "T", "D")])
    kept, dropped = drop_duplicate_postings(df)
    assert list(kept["id"]) == ["j-4"]
    assert list(dropped["id"]) == ["j-30", "j-200"]
    assert set(dropped["detail"]) == {"duplicate_of=j-4"}


def test_no_duplicates_returns_empty_dropped_with_columns():
    df = _frame([("1", "a", "x"), ("2", "b", "y")])
    kept, dropped = drop_duplicate_postings(df)
    assert list(kept["id"]) == ["j-1", "j-2"]
    assert dropped.empty
    assert list(dropped.columns) == ["id", "rule", "detail"]
