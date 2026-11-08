"""Load cleaned parquet inputs and choose which ids to process."""

import random
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from getwork.structure import schema

INPUT_COLUMNS: dict[str, list[str]] = {
    "resumes": ["id", "text_clean"],
    "postings": ["id", "title", "text_clean"],
}


def clean_parquet_path(source: str, in_dir: Path) -> Path:
    return in_dir / f"{source}_clean.parquet"


def _id_number(id_: str) -> int:
    _, sep, suffix = id_.partition("-")
    if not sep:
        raise ValueError(f"malformed id (no '-'): {id_!r}")
    try:
        return int(suffix)
    except ValueError:
        raise ValueError(f"malformed id (non-integer suffix): {id_!r}") from None


def sort_ids(ids: Iterable[str]) -> list[str]:
    return sorted(ids, key=_id_number)


def _validate_inputs(df: pd.DataFrame, source: str) -> None:
    prefix = schema.SOURCE_PREFIX[source]
    duplicated = df["id"][df["id"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"duplicate id: {duplicated.iloc[0]!r}")
    for id_, text in zip(df["id"], df["text_clean"]):
        if not id_.startswith(prefix):
            raise ValueError(f"id {id_!r} does not start with {prefix!r}")
        if text is None or pd.isna(text) or not str(text).strip():
            raise ValueError(f"text_clean is null or blank for id {id_!r}")


def load_inputs(source: str, in_dir: Path) -> pd.DataFrame:
    if source not in INPUT_COLUMNS:
        raise ValueError(f"unknown source: {source!r}")
    path = clean_parquet_path(source=source, in_dir=in_dir)
    if not path.exists():
        raise FileNotFoundError(f"input parquet not found: {path}")
    df = pd.read_parquet(path, columns=INPUT_COLUMNS[source])
    _validate_inputs(df, source)
    order = {id_: i for i, id_ in enumerate(sort_ids(df["id"]))}
    df = df.iloc[df["id"].map(order).argsort(kind="stable")].reset_index(drop=True)
    if "title" in df.columns:
        df["title"] = df["title"].astype(object).where(df["title"].notna(), None)
    return df


def select_universe(
    all_ids: Sequence[str], limit: int | None, sample: int | None, seed: int
) -> list[str]:
    if limit is not None and sample is not None:
        raise ValueError("limit and sample are mutually exclusive")
    universe = sort_ids(all_ids)
    if limit is not None:
        return universe[:limit]
    if sample is not None:
        picked = random.Random(seed).sample(universe, min(sample, len(universe)))
        return sort_ids(picked)
    return universe
