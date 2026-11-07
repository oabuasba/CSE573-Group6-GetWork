import csv
import json
from pathlib import Path

import pandas as pd

from getwork.preprocess import run_postings

ENGLISH = (
    "We are hiring an experienced engineer to build and maintain scalable services for our "
    "enterprise clients. You will design APIs, write automated tests, review code, mentor "
    "junior colleagues, and work closely with product managers to ship new features every "
    "sprint while keeping production systems reliable and secure for all of our customers."
)
SPANISH = (
    "Buscamos un ingeniero con mucha experiencia en el desarrollo de aplicaciones web para "
    "grandes empresas. Liderará un equipo de cinco desarrolladores y revisará el código todos "
    "los días. Sus responsabilidades incluyen diseñar interfaces, escribir pruebas automáticas "
    "y ayudar a los compañeros nuevos con las mejores prácticas de la empresa durante muchos años."
)
SPANISH = f"{SPANISH} {SPANISH}"
HEADER = [
    "job_id", "title", "description", "company_name", "location",
    "formatted_work_type", "formatted_experience_level", "extra",
]
INDUSTRIES = {"1": "Construction", "2": "Software Development", "3": "Retail"}

# job_id, title, description, industry ids
POSTINGS = [
    ("100", "Software Engineer", ENGLISH + "\n\nSecond   paragraph &amp; more.", ["2"]),
    ("101", "Registered Nurse", ENGLISH, ["3"]),
    ("102", "Project Manager", ENGLISH + " Construction site.", ["1"]),
    ("103", "IT Project Manager", ENGLISH + " Office.", ["1"]),
    ("104", "Network Engineer", "Too short.", ["2"]),
    ("99", "Software Engineer", ENGLISH + "\n\nSecond paragraph & more.", ["2"]),
    ("105", "Web Developer", SPANISH, ["2"]),
    ("106", "Sales Manager", ENGLISH, ["2"]),
]


def write_tree(
    root: Path, rows: list[tuple[str, str, str, list[str]]] = POSTINGS,
    industries: dict[str, str] = INDUSTRIES,
) -> Path:
    for sub in ("companies", "jobs", "mappings"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    with open(root / "companies" / "postings.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for job_id, title, desc, _ in rows:
            writer.writerow([job_id, title, desc, "Acme", "NY", "Full-time", "", "x"])
    with open(root / "jobs" / "job_industries.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["job_id", "industry_id"])
        for job_id, _, _, ids in rows:
            writer.writerows([job_id, i] for i in ids)
        writer.writerow(["100", "999"])
    with open(root / "mappings" / "industries.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["industry_id", "industry_name"])
        writer.writerows(industries.items())
    return root


def test_run_outputs_and_reconciliation(tmp_path: Path) -> None:
    data, out = write_tree(tmp_path / "data"), tmp_path / "out"
    summary = run_postings.run(data_dir=data, out_dir=out)

    assert set(summary) == {
        "loaded", "in_domain", "kept", "dropped", "families",
        "experience_level_null", "unmapped_industry_rows", "config",
    }
    assert summary["loaded"] == 8
    assert summary["dropped"] == {
        "not_in_domain": 3, "too_short": 1, "corrupted": 0, "non_english": 1, "duplicate": 1,
    }
    assert summary["loaded"] == summary["kept"] + sum(summary["dropped"].values())
    assert summary["in_domain"] == summary["loaded"] - summary["dropped"]["not_in_domain"]
    assert summary["kept"] == 2
    assert summary["families"]["software"] == 1
    assert summary["families"]["project_manager"] == 1
    assert summary["experience_level_null"] == 2
    assert summary["unmapped_industry_rows"] == 1
    assert summary["config"]["review_sample_size"] == run_postings.REVIEW_SAMPLE_SIZE

    clean = pd.read_parquet(out / "postings_clean.parquet")
    raw = pd.read_parquet(out / "postings_raw.parquet")
    log = pd.read_csv(out / "postings_drop_log.csv")
    assert list(clean.columns) == run_postings.CLEAN_COLUMNS
    assert list(raw.columns) == ["id", "text_raw"]
    assert list(log.columns) == ["id", "rule", "detail"]
    assert len(log) == sum(summary["dropped"].values()) - summary["dropped"]["not_in_domain"]
    assert list(clean["id"]) == ["j-99", "j-103"]
    assert list(clean["id"]) == list(raw["id"])
    assert json.loads((out / "postings_summary.json").read_text()) == summary

    originals = {row[0]: row[2] for row in POSTINGS}
    assert all(originals[i.removeprefix("j-")] == t for i, t in zip(raw["id"], raw["text_raw"]))
    assert "duplicate_of=j-99" in set(log["detail"])


def _review_fixture(tmp_path: Path, n_missed: int, n_kept: int) -> Path:
    rows = [(str(1000 + i), "Retail Clerk", ENGLISH, ["2"]) for i in range(n_missed)]
    rows += [(str(5000 + i), f"Software Engineer {i}", ENGLISH, ["1"]) for i in range(n_kept)]
    rows += [(str(9000 + i), "Retail Clerk", ENGLISH, ["3"]) for i in range(5)]
    return write_tree(tmp_path / "data", rows=rows)


def test_domain_review_sample(tmp_path: Path) -> None:
    data = _review_fixture(tmp_path, n_missed=130, n_kept=120)
    out = tmp_path / "out"
    run_postings.run(data_dir=data, out_dir=out)
    review = pd.read_csv(out / "postings_domain_review.csv")
    limit = run_postings.REVIEW_SAMPLE_SIZE
    assert list(review.columns) == ["id", "title", "decision", "families", "industries"]
    assert (review["decision"] == "kept").sum() == limit
    dropped = review[review["decision"] == "dropped"]
    assert len(dropped) == limit
    assert all("Software Development" in ind for ind in dropped["industries"])
    assert all(i.startswith("j-1") for i in dropped["id"])

    small = _review_fixture(tmp_path / "small", n_missed=3, n_kept=4)
    run_postings.run(data_dir=small, out_dir=tmp_path / "small_out")
    review = pd.read_csv(tmp_path / "small_out" / "postings_domain_review.csv")
    assert (review["decision"] == "dropped").sum() == 3
    assert (review["decision"] == "kept").sum() == 4


def test_run_is_deterministic(tmp_path: Path) -> None:
    data = _review_fixture(tmp_path, n_missed=130, n_kept=120)
    first = run_postings.run(data_dir=data, out_dir=tmp_path / "one")
    second = run_postings.run(data_dir=data, out_dir=tmp_path / "two")
    assert first == second
    for name in ("postings_clean.parquet", "postings_raw.parquet"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "one" / name), pd.read_parquet(tmp_path / "two" / name)
        )
    for name in ("postings_drop_log.csv", "postings_domain_review.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "one" / name), pd.read_csv(tmp_path / "two" / name)
        )


def test_no_in_domain_postings(tmp_path: Path) -> None:
    data = write_tree(tmp_path / "data", rows=[("1", "Registered Nurse", ENGLISH, ["3"])])
    summary = run_postings.run(data_dir=data, out_dir=tmp_path / "out")
    assert summary["kept"] == 0
    assert summary["dropped"]["not_in_domain"] == 1
    clean = pd.read_parquet(tmp_path / "out" / "postings_clean.parquet")
    assert list(clean.columns) == run_postings.CLEAN_COLUMNS


def test_main_cli(tmp_path: Path, capsys) -> None:
    data, out = write_tree(tmp_path / "data"), tmp_path / "out"
    code = run_postings.main(["--data", str(data), "--out", str(out)])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((out / "postings_summary.json").read_text())
    assert (out / "postings_clean.parquet").exists()
