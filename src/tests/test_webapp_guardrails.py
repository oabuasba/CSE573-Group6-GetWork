import json
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "src" / "web"
BACKEND = REPO / "src" / "getwork" / "webapp"
SKIPPED_DIRS = {"node_modules", "dist"}
TEXT_SUFFIXES = {".js", ".jsx", ".css", ".html", ".json", ".md", ".txt", ".py"}
ALLOWED_ASSET = WEB / "public" / "favicon.svg"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ASU_REFERENCE = re.compile(
    r"github\.com/[^\s\"']*asu|@asu/|asu-web-standards|unity-(react|bootstrap|design)", re.I
)
FORBIDDEN_NAME = re.compile(r"asu|sparky|devil|sundevil", re.I)
URL = re.compile(r"https?://[^\s\"'`)>]*")
FONT_IMPORT = re.compile(r"@import|<link[^>]+(fonts|font)", re.I)
REGISTRY = "https://registry.npmjs.org/"


def files_under(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if path.is_file() and not SKIPPED_DIRS & set(parts) and "__pycache__" not in parts:
            yield path


def is_test_file(path: Path) -> bool:
    return ".test." in path.name


def test_no_asu_github_references() -> None:
    hits = []
    for path in [*files_under(WEB), *files_under(BACKEND)]:
        if path.suffix in TEXT_SUFFIXES and ASU_REFERENCE.search(path.read_text(encoding="utf-8")):
            hits.append(path)
    assert hits == []


def test_dependencies_are_registry_only() -> None:
    package = json.loads((WEB / "package.json").read_text(encoding="utf-8"))
    specs = {
        **package.get("dependencies", {}), **package.get("devDependencies", {}),
        **package.get("optionalDependencies", {}),
    }
    assert not [name for name in specs if "asu" in name.lower()]
    assert not [
        spec for spec in specs.values()
        if re.match(r"(git|github:|https?:|file:)", spec) or "://" in spec
    ]
    lock = (WEB / "package-lock.json").read_text(encoding="utf-8")
    resolved = re.findall(r'"resolved":\s*"([^"]*)"', lock)
    assert resolved and all(url.startswith(REGISTRY) for url in resolved)
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = [
        *project["dependencies"],
        *(d for group in project.get("optional-dependencies", {}).values() for d in group),
    ]
    assert not [d for d in declared if "git+" in d or "://" in d or " @ " in d]


def test_only_authored_assets() -> None:
    assets = [p for p in files_under(WEB) if p.suffix not in TEXT_SUFFIXES]
    assert assets == [ALLOWED_ASSET]
    assert not [p for p in files_under(WEB) if FORBIDDEN_NAME.search(p.name)]


def test_no_external_loads() -> None:
    scanned = [
        p for p in files_under(WEB)
        if (p.suffix in {".css", ".js", ".jsx"} or p.name == "index.html") and not is_test_file(p)
    ]
    assert WEB / "index.html" in scanned
    offenders = []
    for path in scanned:
        text = path.read_text(encoding="utf-8")
        urls = [u for u in URL.findall(text) if not u.startswith(SVG_NAMESPACE)]
        if urls or FONT_IMPORT.search(text):
            offenders.append((path.relative_to(REPO), urls))
    assert offenders == []
