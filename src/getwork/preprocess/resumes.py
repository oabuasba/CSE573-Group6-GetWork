import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from getwork.preprocess import text

RESUME_CLASSES: frozenset[str] = frozenset({
    "Software_Developer", "Web_Developer", "Systems_Administrator", "Project_manager",
    "Network_Administrator", "Front_End_Developer", "Database_Administrator",
    "Java_Developer", "Security_Analyst", "Python_Developer",
})


def _parse_labels(name: str, data: bytes) -> list[str]:
    lines = [line.strip() for line in data.decode("utf-8").splitlines()]
    labels = list(dict.fromkeys(line for line in lines if line))
    unknown = [label for label in labels if label not in RESUME_CLASSES]
    if unknown:
        raise ValueError(f"{name}: unknown label {unknown[0]!r}")
    return labels


def _read_members(zf: zipfile.ZipFile) -> tuple[dict[str, bytes], dict[str, bytes]]:
    txt_by_stem: dict[str, bytes] = {}
    lab_by_stem: dict[str, bytes] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.suffix == ".txt":
            txt_by_stem[path.stem] = zf.read(info)
        elif path.suffix == ".lab":
            lab_by_stem[path.stem] = zf.read(info)
    return txt_by_stem, lab_by_stem


def load_resumes(zip_path: Path) -> pd.DataFrame:
    """Read every .txt/.lab pair. Columns: id (str "r-<stem>"), text_raw (str),
    labels (list[str], file order, blank lines skipped, repeats removed keeping first),
    encoding (str). Sorted by id. Raise ValueError if a .txt has no .lab, a .lab has
    no .txt, a label is not in RESUME_CLASSES, or a file cannot be decoded."""
    with zipfile.ZipFile(zip_path) as zf:
        txt_by_stem, lab_by_stem = _read_members(zf)

    missing_lab = sorted(txt_by_stem.keys() - lab_by_stem.keys())
    if missing_lab:
        raise ValueError(f"{missing_lab[0]}.txt has no matching .lab")
    missing_txt = sorted(lab_by_stem.keys() - txt_by_stem.keys())
    if missing_txt:
        raise ValueError(f"{missing_txt[0]}.lab has no matching .txt")

    rows = []
    for stem in sorted(txt_by_stem):
        try:
            text_raw, encoding = text.decode_bytes(txt_by_stem[stem])
        except ValueError as err:
            raise ValueError(f"{stem}.txt: {err}") from err
        rows.append({
            "id": f"r-{stem}",
            "text_raw": text_raw,
            "labels": _parse_labels(f"{stem}.lab", lab_by_stem[stem]),
            "encoding": encoding,
        })
    columns = ["id", "text_raw", "labels", "encoding"]
    return pd.DataFrame(rows, columns=columns).sort_values("id").reset_index(drop=True)


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with text_clean, n_chars, n_words, first_label, has_label added."""
    out = df.copy()
    out["text_clean"] = out["text_raw"].map(text.clean_text)
    out["n_chars"] = out["text_clean"].map(len)
    out["n_words"] = out["text_clean"].map(lambda value: len(value.split()))
    first_labels = [labels[0] if labels else None for labels in out["labels"]]
    out["first_label"] = pd.Series(first_labels, index=out.index, dtype=object)
    out["has_label"] = out["labels"].map(lambda labels: len(labels) > 0)
    return out
