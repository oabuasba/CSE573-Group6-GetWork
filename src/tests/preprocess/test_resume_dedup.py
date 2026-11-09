import pandas as pd

from getwork.preprocess import dedup


def _frame(rows: list[tuple[str, str, list[str]]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "text_clean": [r[1] for r in rows],
            "labels": [r[2] for r in rows],
            "first_label": [r[2][0] if r[2] else None for r in rows],
            "has_label": [len(r[2]) > 0 for r in rows],
        }
    )


def test_duplicates_merge_labels() -> None:
    df = _frame(
        [
            ("r-2", "Hello  World", ["B", "C"]),
            ("r-1", "hello world", ["A", "B"]),
            ("r-3", "other text", ["Z"]),
        ]
    )
    kept, dropped = dedup.drop_duplicates_merge_labels(df)
    assert list(kept["id"]) == ["r-1", "r-3"]
    row = kept.iloc[0]
    assert row["labels"] == ["A", "B", "C"]
    assert row["first_label"] == "A"
    assert row["has_label"] is True or bool(row["has_label"])
    assert list(dropped.columns) == ["id", "rule", "detail"]
    assert dropped.to_dict("records") == [
        {"id": "r-2", "rule": "duplicate", "detail": "duplicate_of=r-1"}
    ]


def test_unlabeled_keeper_gets_labels() -> None:
    df = _frame([("r-1", "same text", []), ("r-2", "Same  Text", ["X"])])
    kept, _ = dedup.drop_duplicates_merge_labels(df)
    assert len(kept) == 1
    assert kept.iloc[0]["labels"] == ["X"]
    assert kept.iloc[0]["first_label"] == "X"
    assert bool(kept.iloc[0]["has_label"])


def test_three_way_group() -> None:
    df = _frame(
        [
            ("r-3", "a b", ["C"]),
            ("r-1", "A B", ["A"]),
            ("r-2", "a  b", ["B", "A"]),
        ]
    )
    kept, dropped = dedup.drop_duplicates_merge_labels(df)
    assert list(kept["id"]) == ["r-1"]
    assert kept.iloc[0]["labels"] == ["A", "B", "C"]
    assert sorted(dropped["id"]) == ["r-2", "r-3"]
    assert set(dropped["detail"]) == {"duplicate_of=r-1"}
