import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from getwork.preprocess import domain, posting_dedup, postings
from getwork.preprocess.filters import ResumeFilterConfig, apply_quality_filters

DEFAULT_DATA: Path = Path("data/raw/job_postings")
DEFAULT_OUT: Path = Path("data/processed/postings")
REVIEW_SAMPLE_SIZE: int = 100

CLEAN_COLUMNS = [
    "id", "job_id", "title", "company_name", "location", "work_type", "experience_level",
    "industries", "families", "text_clean", "n_chars", "n_words",
]
RAW_COLUMNS = ["id", "text_raw"]
REVIEW_COLUMNS = ["id", "title", "decision", "families", "industries"]
DROP_RULES = ["not_in_domain", "too_short", "corrupted", "non_english", "duplicate"]
FAMILY_NAMES = list(domain.DOMAIN_FAMILIES)


def _filter_domain(loaded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept, dropped = domain.apply_domain_filter(loaded)
    # apply_domain_filter loses the input columns on an empty frame, so restore them
    return kept.reindex(columns=[*loaded.columns, "families"]), dropped


def _sample_review(frame: pd.DataFrame, decision: str) -> pd.DataFrame:
    sample = frame.sample(n=min(REVIEW_SAMPLE_SIZE, len(frame)), random_state=0)
    sample = sample.sort_values(by="job_id", key=lambda col: col.astype(int), kind="stable")
    if "families" not in sample.columns:
        sample = sample.assign(families=[[] for _ in range(len(sample))])
    return pd.DataFrame({
        "id": sample["id"],
        "title": sample["title"],
        "decision": decision,
        "families": sample["families"].map(";".join),
        "industries": sample["industries"].map(";".join),
    }, columns=REVIEW_COLUMNS)


def _build_review_sample(
    loaded: pd.DataFrame, domain_kept: pd.DataFrame, domain_dropped: pd.DataFrame
) -> pd.DataFrame:
    missed = loaded[loaded["id"].isin(domain_dropped["id"])]
    has_tech_industry = missed["industries"].map(
        lambda names: any(name in domain.TECH_INDUSTRIES for name in names)
    ).astype(bool)
    return pd.concat(
        [
            _sample_review(frame=domain_kept, decision="kept"),
            _sample_review(frame=missed[has_tech_industry], decision="dropped"),
        ],
        ignore_index=True,
    )


def _sort_by_job_id(df: pd.DataFrame, column: str = "job_id") -> pd.DataFrame:
    order = df[column].astype(int).sort_values(kind="stable").index
    return df.loc[order].reset_index(drop=True)


def _build_drop_log(*parts: pd.DataFrame) -> pd.DataFrame:
    log = pd.concat(parts, ignore_index=True)
    order = log["id"].str.removeprefix("j-").astype(int).sort_values(kind="stable").index
    return log.loc[order].reset_index(drop=True)


def _build_summary(
    loaded: pd.DataFrame, domain_kept: pd.DataFrame, kept: pd.DataFrame,
    drop_log: pd.DataFrame, domain_dropped: pd.DataFrame, unmapped: int, cfg: ResumeFilterConfig,
) -> dict:
    rule_counts = pd.concat([drop_log["rule"], domain_dropped["rule"]]).value_counts()
    return {
        "loaded": len(loaded),
        "in_domain": len(domain_kept),
        "kept": len(kept),
        "dropped": {rule: int(rule_counts.get(rule, 0)) for rule in DROP_RULES},
        "families": {
            name: int(kept["families"].map(lambda found: name in found).sum())
            for name in FAMILY_NAMES
        },
        "experience_level_null": int(kept["experience_level"].isna().sum()),
        "unmapped_industry_rows": unmapped,
        "config": {**dataclasses.asdict(cfg), "review_sample_size": REVIEW_SAMPLE_SIZE},
    }


def _write_outputs(
    out_dir: Path, kept: pd.DataFrame, drop_log: pd.DataFrame,
    review: pd.DataFrame, summary: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept[CLEAN_COLUMNS].to_parquet(
        out_dir / "postings_clean.parquet", engine="pyarrow", compression="zstd", index=False
    )
    kept[RAW_COLUMNS].to_parquet(
        out_dir / "postings_raw.parquet", engine="pyarrow", compression="zstd", index=False
    )
    drop_log.to_csv(out_dir / "postings_drop_log.csv", index=False, encoding="utf-8")
    review.to_csv(out_dir / "postings_domain_review.csv", index=False, encoding="utf-8")
    (out_dir / "postings_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8"
    )


def run(data_dir: Path, out_dir: Path, cfg: ResumeFilterConfig = ResumeFilterConfig()) -> dict:
    loaded, unmapped = postings.load_postings(
        postings_csv=data_dir / "companies" / "postings.csv",
        job_industries_csv=data_dir / "jobs" / "job_industries.csv",
        industries_csv=data_dir / "mappings" / "industries.csv",
    )
    domain_kept, domain_dropped = _filter_domain(loaded=loaded)
    review = _build_review_sample(
        loaded=loaded, domain_kept=domain_kept, domain_dropped=domain_dropped
    )

    cleaned = postings.add_posting_columns(domain_kept)
    quality_kept, quality_dropped = apply_quality_filters(cleaned, cfg)
    deduped, dup_dropped = posting_dedup.drop_duplicate_postings(quality_kept)

    kept = _sort_by_job_id(deduped)
    drop_log = _build_drop_log(quality_dropped, dup_dropped)
    summary = _build_summary(
        loaded=loaded, domain_kept=domain_kept, kept=kept, drop_log=drop_log,
        domain_dropped=domain_dropped, unmapped=unmapped, cfg=cfg,
    )
    _write_outputs(out_dir=out_dir, kept=kept, drop_log=drop_log, review=review, summary=summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the LinkedIn postings.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, dest="data_dir")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, dest="out_dir")
    args = parser.parse_args(argv)
    summary = run(data_dir=args.data_dir, out_dir=args.out_dir)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
