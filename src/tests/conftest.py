import socket
import zipfile
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import pytest


@pytest.fixture
def make_resume_zip(tmp_path: Path) -> Callable[[dict[str, bytes]], Path]:
    def build(entries: dict[str, bytes]) -> Path:
        zip_path = tmp_path / "resumes.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, data in entries.items():
                zf.writestr(name, data)
        return zip_path

    return build


ENV_VARS = ("VOYAGER_API_KEY", "VOYAGER_BASE_URL", "VOYAGER_MODEL")


def apply_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    apply_offline(monkeypatch, tmp_path)


@pytest.fixture
def write_clean_parquet(tmp_path: Path) -> Callable[[str, Sequence[dict]], Path]:
    def build(source: str, rows: Sequence[dict]) -> Path:
        in_dir = tmp_path / "in"
        in_dir.mkdir(exist_ok=True)
        pd.DataFrame(list(rows)).to_parquet(
            in_dir / f"{source}_clean.parquet", engine="pyarrow", index=False
        )
        return in_dir

    return build
