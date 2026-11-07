import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from getwork.preprocess import dedup, filters, resumes
from getwork.preprocess.filters import ResumeFilterConfig

DEFAULT_ZIP: Path = Path("data/resume_corpus/resumes_corpus.zip")
DEFAULT_OUT: Path = Path("data/processed")

CLEAN_COLUMNS = ["id", "text_clean", "labels", "first_label", "has_label", "n_chars", "n_words", "encoding"]
RAW_COLUMNS = ["id", "text_raw"]
DROP_RULES = ["too_short", "corrupted", "non_english", "duplicate"]
ENCODINGS = ["utf-8", "cp1252"]


def _with_stable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric/bool dtypes so an empty frame still writes the documented schema."""
    return df.astype({"n_chars": "int64", "n_words": "int64", "has_label": "bool"})


def _build_summary(
    loaded: pd.DataFrame, kept: pd.DataFrame, drop_log: pd.DataFrame, cfg: ResumeFilterConfig
) -> dict:
    rule_counts = drop_log["rule"].value_counts()
    encoding_counts = loaded["encoding"].value_counts()
    return {
        "loaded": len(loaded),
        "kept": len(kept),
        "dropped": {rule: int(rule_counts.get(rule, 0)) for rule in DROP_RULES},
        "unlabeled_kept": int((~kept["has_label"]).sum()),
        "encodings": {enc: int(encoding_counts.get(enc, 0)) for enc in ENCODINGS},
        "config": dataclasses.asdict(cfg),
    }


def _write_outputs(out_dir: Path, kept: pd.DataFrame, drop_log: pd.DataFrame, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept[CLEAN_COLUMNS].to_parquet(
        out_dir / "resumes_clean.parquet", engine="pyarrow", compression="zstd", index=False
    )
    kept[RAW_COLUMNS].to_parquet(
        out_dir / "resumes_raw.parquet", engine="pyarrow", compression="zstd", index=False
    )
    drop_log.to_csv(out_dir / "resumes_drop_log.csv", index=False, encoding="utf-8")
    (out_dir / "resumes_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8"
    )


def run(zip_path: Path, out_dir: Path, cfg: ResumeFilterConfig = ResumeFilterConfig()) -> dict:
    loaded = resumes.load_resumes(zip_path)
    cleaned = resumes.add_clean_columns(loaded)
    quality_kept, quality_dropped = filters.apply_quality_filters(cleaned, cfg)
    deduped, dup_dropped = dedup.drop_duplicates_merge_labels(quality_kept)

    kept = _with_stable_dtypes(deduped).sort_values("id").reset_index(drop=True)
    drop_log = (
        pd.concat([quality_dropped, dup_dropped], ignore_index=True)
        .sort_values("id", kind="stable")
        .reset_index(drop=True)
    )
    summary = _build_summary(loaded, kept, drop_log, cfg)
    _write_outputs(out_dir, kept, drop_log, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the resume corpus.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, dest="out_dir")
    args = parser.parse_args(argv)
    summary = run(zip_path=args.zip_path, out_dir=args.out_dir)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
