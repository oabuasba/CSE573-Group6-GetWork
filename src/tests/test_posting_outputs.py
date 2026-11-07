import json
from pathlib import Path

import pandas as pd
import pytest

from getwork.preprocess.domain import DOMAIN_FAMILIES, TECH_INDUSTRIES

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "postings"
INDUSTRIES_CSV = ROOT / "data" / "job_postings" / "mappings" / "industries.csv"
MAX_PARQUET_BYTES = 90 * 1024 * 1024
EXPECTED_LOADED = 123849
MIN_KEPT, MAX_KEPT = 5000, 12000
MIN_FAMILY_KEPT = 100
MAX_QUALITY_DUP_FRACTION = 0.20
QUALITY_DUP_RULES = ("too_short", "corrupted", "non_english", "duplicate")


def test_real_outputs() -> None:
    summary_path = PROCESSED / "postings_summary.json"
    if not summary_path.exists():
        pytest.skip("data/processed/postings outputs not generated")

    summary = json.loads(summary_path.read_text())
    clean = pd.read_parquet(PROCESSED / "postings_clean.parquet")

    assert summary["loaded"] == EXPECTED_LOADED
    assert clean["id"].is_unique
    assert MIN_KEPT <= summary["kept"] <= MAX_KEPT
    for family in DOMAIN_FAMILIES:
        assert summary["families"][family] >= MIN_FAMILY_KEPT, family
    dropped_after_domain = sum(summary["dropped"][rule] for rule in QUALITY_DUP_RULES)
    assert dropped_after_domain <= MAX_QUALITY_DUP_FRACTION * summary["in_domain"]
    for name in ("postings_clean.parquet", "postings_raw.parquet"):
        assert (PROCESSED / name).stat().st_size < MAX_PARQUET_BYTES


def test_tech_industries_exist_in_data() -> None:
    if not INDUSTRIES_CSV.exists():
        pytest.skip("industries.csv not available")
    names = set(pd.read_csv(INDUSTRIES_CSV)["industry_name"])
    assert TECH_INDUSTRIES <= names, sorted(TECH_INDUSTRIES - names)
