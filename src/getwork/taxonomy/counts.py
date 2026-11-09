"""Document-frequency counts of canonical skills and the top-N selection (TX-1)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from getwork import fileio
from getwork.alias import normalize

COUNTS_FILE = "skill_counts.parquet"
SUMMARY_FILE = "counts_summary.json"
RESUMES = "resumes"
POSTINGS = "postings"
VERSION_PREFIX = "sha256:"
VERSION_HEX_CHARS = 12

COUNTS_SCHEMA = pa.schema(
    [
        ("skill", pa.string()),
        ("df_total", pa.int64()),
        ("df_resumes", pa.int64()),
        ("df_postings", pa.int64()),
        ("selected", pa.bool_()),
        ("rank", pa.int32()),
    ]
)


@dataclass(frozen=True)
class SkillCount:
    skill: str
    df_total: int
    df_resumes: int
    df_postings: int


def _sort_key(count: SkillCount) -> tuple[int, str, str]:
    return (-count.df_total, count.skill.casefold(), count.skill)


def count_skills(records_by_source: Mapping[str, Mapping[str, dict]]) -> list[SkillCount]:
    """Count each normalized skill once per document and source; display the commonest spelling."""
    df: dict[str, Counter[str]] = {}
    spellings: dict[str, Counter[str]] = {}
    for source, records in records_by_source.items():
        for record in records.values():
            seen: set[str] = set()
            for skill in record["skills"]:
                key = normalize.normalize_key(skill)
                spellings.setdefault(key, Counter())[skill] += 1
                seen.add(key)
            for key in seen:
                df.setdefault(key, Counter())[source] += 1
    counts = [
        SkillCount(
            skill=min(forms, key=lambda s: (-forms[s], s)),
            df_total=sum(df[key].values()),
            df_resumes=df[key][RESUMES],
            df_postings=df[key][POSTINGS],
        )
        for key, forms in spellings.items()
    ]
    return sorted(counts, key=_sort_key)


def select_top(counts: Sequence[SkillCount], top_n: int, min_df: int) -> list[SkillCount]:
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")
    if min_df < 1:
        raise ValueError(f"min_df must be at least 1, got {min_df}")
    return [c for c in counts if c.df_total >= min_df][:top_n]


def skills_version(skills: Sequence[str]) -> str:
    payload = json.dumps(list(skills), ensure_ascii=False).encode("utf-8")
    return VERSION_PREFIX + hashlib.sha256(payload).hexdigest()[:VERSION_HEX_CHARS]


def counts_summary(
    counts: Sequence[SkillCount],
    selected: Sequence[SkillCount],
    n_docs: Mapping[str, int],
    alias_table_version: str,
    top_n: int,
    min_df: int,
) -> dict:
    total_mentions = sum(c.df_total for c in counts)
    one_corpus = sum(1 for c in selected if c.df_resumes == 0 or c.df_postings == 0)
    return {
        "alias_table_version": alias_table_version,
        "top_n": top_n,
        "min_df": min_df,
        "n_docs": dict(n_docs),
        "n_distinct_skills": len(counts),
        "n_selected": len(selected),
        "mention_coverage": (
            sum(c.df_total for c in selected) / total_mentions if total_mentions else 0.0
        ),
        "one_corpus_only_fraction": one_corpus / len(selected) if selected else 0.0,
        "selected_skills_version": skills_version([c.skill for c in selected]),
    }


def write_counts(
    directory: Path,
    counts: Sequence[SkillCount],
    selected: Sequence[SkillCount],
    summary: Mapping[str, object],
) -> None:
    ranks = {c.skill: i for i, c in enumerate(selected, start=1)}
    rows = [
        {
            "skill": c.skill,
            "df_total": c.df_total,
            "df_resumes": c.df_resumes,
            "df_postings": c.df_postings,
            "selected": c.skill in ranks,
            "rank": ranks.get(c.skill),
        }
        for c in counts
    ]
    table = pa.Table.from_pylist(rows, schema=COUNTS_SCHEMA)
    fileio.write_parquet_atomic(directory / COUNTS_FILE, table)
    fileio.write_json_atomic(directory / SUMMARY_FILE, dict(summary))


def read_selected(directory: Path) -> list[str]:
    path = directory / COUNTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"skill counts not found: {path} (run taxonomy counts)")
    rows = pq.read_table(path, columns=["skill", "rank"]).to_pylist()
    ranked = sorted((r["rank"], r["skill"]) for r in rows if r["rank"] is not None)
    return [skill for _, skill in ranked]
