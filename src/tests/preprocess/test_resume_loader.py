import pandas as pd
import pytest

from getwork.preprocess import resumes


def test_load_pairs_and_schema(make_resume_zip):
    zip_path = make_resume_zip({
        "corpus/": b"",
        "corpus/readme.md": b"ignore me",
        "corpus/2.txt": b"\x95 Built apps",
        "corpus/2.lab": b"Java_Developer\n",
        "corpus/1.txt": b"Plain ascii",
        "corpus/1.lab": b"",
        "corpus/3.txt": b"Third",
        "corpus/3.lab": b"Web_Developer\r\n\r\nPython_Developer\r\nWeb_Developer\r\n",
        "corpus/4.txt": b"Fourth",
        "corpus/4.lab": b"\r\n\n",
    })
    df = resumes.load_resumes(zip_path)
    assert list(df.columns) == ["id", "text_raw", "labels", "encoding"]
    assert list(df["id"]) == ["r-1", "r-2", "r-3", "r-4"]
    assert list(df["labels"]) == [
        [], ["Java_Developer"], ["Web_Developer", "Python_Developer"], [],
    ]
    assert list(df["text_raw"]) == ["Plain ascii", "• Built apps", "Third", "Fourth"]
    assert list(df["encoding"]) == ["utf-8", "cp1252", "utf-8", "utf-8"]


def test_load_missing_pair_raises(make_resume_zip):
    with pytest.raises(ValueError, match="7.txt"):
        resumes.load_resumes(make_resume_zip({"7.txt": b"x"}))
    with pytest.raises(ValueError, match="8.lab"):
        resumes.load_resumes(make_resume_zip({"8.lab": b"Web_Developer"}))


def test_load_unknown_label_raises(make_resume_zip):
    zip_path = make_resume_zip({"9.txt": b"x", "9.lab": b"Astronaut\n"})
    with pytest.raises(ValueError, match="Astronaut"):
        resumes.load_resumes(zip_path)


def test_add_clean_columns():
    df = pd.DataFrame({
        "id": ["r-1", "r-2"],
        "text_raw": ['<span class="hl">Hello</span>  big&nbsp;world', "x"],
        "labels": [["Web_Developer", "Java_Developer"], []],
        "encoding": ["utf-8", "utf-8"],
    })
    before = df.copy(deep=True)
    out = resumes.add_clean_columns(df)
    assert list(out["text_clean"]) == ["Hello big world", "x"]
    assert list(out["n_chars"]) == [15, 1]
    assert list(out["n_words"]) == [3, 1]
    assert list(out["first_label"]) == ["Web_Developer", None]
    assert list(out["has_label"]) == [True, False]
    assert all(isinstance(v, bool) for v in out["has_label"])
    pd.testing.assert_frame_equal(df, before)
    assert "text_clean" not in df.columns
