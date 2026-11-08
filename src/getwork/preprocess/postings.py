import html
import re
from pathlib import Path

import pandas as pd

POSTING_COLUMNS = [
    "job_id", "title", "description", "company_name", "location",
    "formatted_work_type", "formatted_experience_level",
]
HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
SPACE_TAB_RUN = re.compile(r"[ \t]+")
TRAILING_SPACES = re.compile(r" +$", re.MULTILINE)
NEWLINE_RUN = re.compile(r"\n{3,}")


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from err


def _industries_by_job(job_industries_csv: Path, industries_csv: Path) -> tuple[dict[str, list[str]], int]:
    links = _read_csv(job_industries_csv, ["job_id", "industry_id"])
    names = _read_csv(industries_csv, ["industry_id", "industry_name"])
    name_by_id = dict(zip(names["industry_id"], names["industry_name"]))
    by_job: dict[str, list[str]] = {}
    unmapped = 0
    for job_id, industry_id in zip(links["job_id"], links["industry_id"]):
        if industry_id not in name_by_id:
            unmapped += 1
            continue
        found = by_job.setdefault(job_id, [])
        if name_by_id[industry_id] not in found:
            found.append(name_by_id[industry_id])
    return by_job, unmapped


def load_postings(postings_csv: Path, job_industries_csv: Path,
                  industries_csv: Path) -> tuple[pd.DataFrame, int]:
    """Read postings.csv (only the columns job_id, title, description, company_name,
    location, formatted_work_type, formatted_experience_level) as strings, without
    NA conversion. Return (df, unmapped_industry_rows).
    df columns: id (str "j-<job_id>"), job_id (str), title (str),
    company_name (str, "" if missing), location (str), work_type (str),
    experience_level (str | None; blank becomes None), industries (list[str]:
    industry names in job_industries file order, repeats removed, [] if none),
    text_raw (str, the description verbatim).
    Sorted ascending by int(job_id).
    unmapped_industry_rows: number of job_industries rows whose industry_id is not
    in industries.csv (those rows are skipped).
    Raise ValueError if job_id is not unique or a needed column is missing."""
    raw = _read_csv(postings_csv, POSTING_COLUMNS)
    if raw["job_id"].duplicated().any():
        duplicate = raw.loc[raw["job_id"].duplicated(), "job_id"].iloc[0]
        raise ValueError(f"duplicate job_id {duplicate!r} in {postings_csv}")
    by_job, unmapped = _industries_by_job(job_industries_csv, industries_csv)

    experience = pd.Series(
        [value if value.strip() else None for value in raw["formatted_experience_level"]],
        index=raw.index, dtype=object,
    )
    df = pd.DataFrame({
        "id": "j-" + raw["job_id"],
        "job_id": raw["job_id"],
        "title": raw["title"],
        "company_name": raw["company_name"],
        "location": raw["location"],
        "work_type": raw["formatted_work_type"],
        "experience_level": experience,
        "industries": raw["job_id"].map(lambda job_id: list(by_job.get(job_id, []))),
        "text_raw": raw["description"],
    })
    order = df["job_id"].astype(int).sort_values(kind="stable").index
    return df.loc[order].reset_index(drop=True), unmapped


def clean_posting_text(raw: str) -> str:
    """html.unescape; remove tags matching </?[A-Za-z][^>]*>; convert \\r\\n and \\r to \\n;
    replace "\\xa0" with " "; collapse runs of spaces and tabs to one space;
    strip trailing spaces on each line; collapse 3 or more consecutive newlines to
    exactly 2; strip the whole string. Idempotent on its own output for ordinary text whose
    entities decode only once; not guaranteed for double-escaped entities such as "&amp;lt;".
    Keeps single and double newlines, "**", and all other characters."""
    text = HTML_TAG.sub("", html.unescape(raw))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = SPACE_TAB_RUN.sub(" ", text)
    text = TRAILING_SPACES.sub("", text)
    return NEWLINE_RUN.sub("\n\n", text).strip()


def add_posting_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with text_clean (str) = clean_posting_text(text_raw),
    n_chars (int) = len(text_clean), n_words (int) = len(text_clean.split())."""
    out = df.copy()
    out["text_clean"] = out["text_raw"].map(clean_posting_text)
    out["n_chars"] = out["text_clean"].map(len)
    out["n_words"] = out["text_clean"].map(lambda value: len(value.split()))
    return out
