from pathlib import Path

import pandas as pd
import pytest

from getwork.preprocess import postings

POSTINGS_HEADER = (
    "job_id,title,description,company_name,location,formatted_work_type,"
    "formatted_experience_level,extra\n"
)


def write_tree(tmp_path: Path, postings_body: str | None = None) -> tuple[Path, Path, Path]:
    body = postings_body if postings_body is not None else (
        '100,Dev,"line one\nline two",Acme,NY,Full-time,Entry level,x\n'
        '99,Admin,plain,,LA,Contract,,x\n'
        '101,Other,text,Beta,SF,Full-time,Senior,x\n'
    )
    postings_csv = tmp_path / "postings.csv"
    postings_csv.write_text(POSTINGS_HEADER + body, newline="")
    links = tmp_path / "job_industries.csv"
    links.write_text("job_id,industry_id\n100,2\n100,1\n100,2\n99,3\n99,777\n")
    industries = tmp_path / "industries.csv"
    industries.write_text("industry_id,industry_name\n1,Software\n2,Banking\n3,Retail\n")
    return postings_csv, links, industries


def test_load_postings_schema(tmp_path: Path) -> None:
    df, _ = postings.load_postings(*write_tree(tmp_path))
    assert list(df.columns) == [
        "id", "job_id", "title", "company_name", "location", "work_type",
        "experience_level", "industries", "text_raw",
    ]
    assert list(df["job_id"]) == ["99", "100", "101"]
    assert list(df["id"]) == ["j-99", "j-100", "j-101"]
    by_id = df.set_index("id")
    assert by_id.loc["j-100", "text_raw"] == "line one\nline two"
    assert by_id.loc["j-99", "experience_level"] is None
    assert by_id.loc["j-100", "experience_level"] == "Entry level"
    assert by_id.loc["j-99", "company_name"] == ""
    assert by_id.loc["j-101", "industries"] == []
    assert by_id.loc["j-100", "industries"] == ["Banking", "Software"]
    assert by_id.loc["j-99", "industries"] == ["Retail"]


def test_unmapped_industry_rows_counted(tmp_path: Path) -> None:
    df, unmapped = postings.load_postings(*write_tree(tmp_path))
    assert unmapped == 1
    assert "777" not in str(df["industries"].tolist())


def test_load_errors(tmp_path: Path) -> None:
    duplicate = '5,A,t,C,L,F,E,x\n5,B,t,C,L,F,E,x\n'
    with pytest.raises(ValueError):
        postings.load_postings(*write_tree(tmp_path, duplicate))
    postings_csv, links, industries = write_tree(tmp_path)
    postings_csv.write_text("job_id,title\n1,A\n")
    with pytest.raises(ValueError):
        postings.load_postings(postings_csv, links, industries)


def test_clean_posting_text() -> None:
    raw = "  Tom &amp; Jerry<br/>\r\n<b>Bold</b>\xa0text\t\t x  \r\r\r\r**keep**\nnext   \n\n\n\nend  "
    cleaned = postings.clean_posting_text(raw)
    assert cleaned == "Tom & Jerry\nBold text x\n\n**keep**\nnext\n\nend"
    assert postings.clean_posting_text("a\nb\n\nc") == "a\nb\n\nc"
    assert postings.clean_posting_text("1 < 2 and 3 > 2") == "1 < 2 and 3 > 2"


def test_clean_posting_text_idempotent() -> None:
    raw = "A &nbsp;&amp; <p>B</p>\r\n\r\n\r\n\r\nC \t D \n  E  "
    once = postings.clean_posting_text(raw)
    assert postings.clean_posting_text(once) == once


def test_add_posting_columns() -> None:
    df = pd.DataFrame({"text_raw": ["Hello  <b>big</b> world\n\n\n\nbye", ""]})
    before = df.copy()
    out = postings.add_posting_columns(df)
    pd.testing.assert_frame_equal(df, before)
    assert out["text_clean"].tolist() == ["Hello big world\n\nbye", ""]
    assert out["n_chars"].tolist() == [len("Hello big world\n\nbye"), 0]
    assert out["n_words"].tolist() == [4, 0]
