import json
from pathlib import Path
from typing import Callable

import pandas as pd

from getwork.preprocess import resumes, run_resumes, text

ENGLISH_A = (
    "Experienced software engineer who built scalable web services and maintained large "
    "databases for several enterprise clients. I led a team of five developers, reviewed "
    "code daily, and shipped new features every sprint. My responsibilities included "
    "designing APIs, writing automated tests, and mentoring junior colleagues on best practices."
)
ENGLISH_B = (
    "Systems administrator responsible for managing Linux servers and monitoring network "
    "performance across multiple data centers. I automated routine backups, configured "
    "firewalls, and resolved production incidents quickly. I also documented procedures and "
    "trained new staff on security policies and disaster recovery plans for the company."
)
ENGLISH_C = (
    "Project manager with ten years of experience delivering complex technology programs "
    "on time and under budget. I coordinated stakeholders, tracked risks, and prepared weekly "
    "status reports for executives. I also negotiated vendor contracts and introduced agile "
    "practices that improved the delivery speed of every team that I worked with."
)
SPANISH = (
    "Ingeniero de software con mucha experiencia en el desarrollo de aplicaciones web para "
    "grandes empresas. Lideré un equipo de cinco desarrolladores y revisé el código todos los "
    "días. Mis responsabilidades incluían diseñar interfaces, escribir pruebas automáticas y "
    "ayudar a los compañeros nuevos con las mejores prácticas de la empresa durante muchos años."
)
ENGLISH_A, ENGLISH_B, ENGLISH_C, SPANISH = (f"{t} {t}" for t in (ENGLISH_A, ENGLISH_B, ENGLISH_C, SPANISH))
CP1252_TEXT = "• " + ENGLISH_C + " • Managed budgets and schedules."

ENTRIES = {
    "a1.txt": ENGLISH_A.encode(),
    "a1.lab": b"Software_Developer\n",
    "a2.txt": ("<span class='hl'>" + ENGLISH_B + "</span>").encode(),
    "a2.lab": b"Systems_Administrator\r\nNetwork_Administrator\r\n",
    "a3.txt": b"too short to keep",
    "a3.lab": b"Web_Developer\n",
    "a4.txt": ENGLISH_A.lower().encode(),
    "a4.lab": b"Java_Developer\n",
    "a5.txt": (ENGLISH_A + " Unlabeled extra sentence about cloud platforms and containers.").encode(),
    "a5.lab": b"",
    "a6.txt": CP1252_TEXT.encode("cp1252"),
    "a6.lab": b"Project_manager\n",
    "a7.txt": SPANISH.encode(),
    "a7.lab": b"Python_Developer\n",
}


def test_unlabeled_resumes_kept_and_counted(
    make_resume_zip: Callable[[dict[str, bytes]], Path], tmp_path: Path
) -> None:
    summary = run_resumes.run(zip_path=make_resume_zip(ENTRIES), out_dir=tmp_path / "out")
    clean = pd.read_parquet(tmp_path / "out" / "resumes_clean.parquet")
    row = clean[clean["id"] == "r-a5"].iloc[0]
    assert not row["has_label"]
    assert pd.isna(row["first_label"])
    assert list(row["labels"]) == []
    assert summary["unlabeled_kept"] == 1


def test_run_outputs_and_reconciliation(
    make_resume_zip: Callable[[dict[str, bytes]], Path], tmp_path: Path
) -> None:
    zip_path = make_resume_zip(ENTRIES)
    out = tmp_path / "out"
    summary = run_resumes.run(zip_path=zip_path, out_dir=out)

    assert set(summary) == {"loaded", "kept", "dropped", "unlabeled_kept", "encodings", "config"}
    assert summary["loaded"] == 7
    assert summary["dropped"] == {"too_short": 1, "corrupted": 0, "non_english": 1, "duplicate": 1}
    assert summary["loaded"] == summary["kept"] + sum(summary["dropped"].values())
    assert summary["encodings"] == {"utf-8": 6, "cp1252": 1}

    clean = pd.read_parquet(out / "resumes_clean.parquet")
    raw = pd.read_parquet(out / "resumes_raw.parquet")
    log = pd.read_csv(out / "resumes_drop_log.csv")
    assert list(clean.columns) == [
        "id", "text_clean", "labels", "first_label", "has_label", "n_chars", "n_words", "encoding",
    ]
    assert list(raw.columns) == ["id", "text_raw"]
    assert list(log.columns) == ["id", "rule", "detail"]
    assert len(log) == sum(summary["dropped"].values())
    assert list(clean["id"]) == sorted(clean["id"])
    assert list(clean["id"]) == list(raw["id"])
    assert json.loads((out / "resumes_summary.json").read_text()) == summary

    original = resumes.load_resumes(zip_path).set_index("id")["text_raw"]
    assert all(original[i] == t for i, t in zip(raw["id"], raw["text_raw"]))
    merged = clean.set_index("id").loc["r-a1", "labels"]
    assert list(merged) == ["Software_Developer", "Java_Developer"]
    assert text.decode_bytes(CP1252_TEXT.encode("cp1252"))[0] == original["r-a6"]


def test_run_is_deterministic(
    make_resume_zip: Callable[[dict[str, bytes]], Path], tmp_path: Path
) -> None:
    zip_path = make_resume_zip(ENTRIES)
    first = run_resumes.run(zip_path=zip_path, out_dir=tmp_path / "one")
    second = run_resumes.run(zip_path=zip_path, out_dir=tmp_path / "two")
    assert first == second
    for name in ("resumes_clean.parquet", "resumes_raw.parquet"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "one" / name), pd.read_parquet(tmp_path / "two" / name)
        )
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "one" / "resumes_drop_log.csv"),
        pd.read_csv(tmp_path / "two" / "resumes_drop_log.csv"),
    )


def test_main_cli(
    make_resume_zip: Callable[[dict[str, bytes]], Path], tmp_path: Path, capsys
) -> None:
    out = tmp_path / "out"
    code = run_resumes.main(["--zip", str(make_resume_zip(ENTRIES)), "--out", str(out)])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((out / "resumes_summary.json").read_text())
    assert (out / "resumes_clean.parquet").exists()
