import zipfile
from pathlib import Path
from typing import Callable

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
