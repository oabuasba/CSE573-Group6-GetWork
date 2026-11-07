import pandas as pd

DUPLICATE_RULE = "duplicate"
DROPPED_COLUMNS = ["id", "rule", "detail"]


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def drop_duplicate_postings(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the smallest numeric job_id per normalized (title, text_clean) group.

    Returns (kept, dropped); kept preserves input row order, dropped has columns id, rule, detail.
    """
    keyed = df.assign(
        _title_key=df["title"].map(_normalize),
        _text_key=df["text_clean"].map(_normalize),
        _job_num=df["job_id"].astype(int),
    )
    winners = keyed.sort_values("_job_num", kind="stable").groupby(
        ["_title_key", "_text_key"], sort=False
    )["id"].first()
    winner_ids = keyed.join(winners.rename("_kept_id"), on=["_title_key", "_text_key"])["_kept_id"]

    is_dropped = (winner_ids != df["id"]).to_numpy()
    kept = df[~is_dropped]
    dropped = pd.DataFrame(
        {
            "id": df["id"][is_dropped].to_numpy(),
            "rule": DUPLICATE_RULE,
            "detail": ("duplicate_of=" + winner_ids[is_dropped]).to_numpy(),
        },
        columns=DROPPED_COLUMNS,
    )
    return kept, dropped
