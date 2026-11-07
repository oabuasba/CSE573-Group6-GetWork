import json
from pathlib import Path

import pandas as pd
import pytest

from getwork.preprocess.resumes import RESUME_CLASSES

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "resumes"
MAX_PARQUET_BYTES = 90 * 1024 * 1024
MIN_KEPT_FRACTION = 0.90
EXPECTED_LOADED = 29783


def test_real_outputs() -> None:
    summary_path = PROCESSED / "resumes_summary.json"
    if not summary_path.exists():
        pytest.skip("data/processed/resumes outputs not generated")

    summary = json.loads(summary_path.read_text())
    clean = pd.read_parquet(PROCESSED / "resumes_clean.parquet")

    assert summary["loaded"] == EXPECTED_LOADED
    assert clean["id"].is_unique
    assert all(label in RESUME_CLASSES for labels in clean["labels"] for label in labels)
    assert not clean["text_clean"].str.contains("<span", regex=False).any()
    assert summary["kept"] / summary["loaded"] >= MIN_KEPT_FRACTION
    for name in ("resumes_clean.parquet", "resumes_raw.parquet"):
        assert (PROCESSED / name).stat().st_size < MAX_PARQUET_BYTES
