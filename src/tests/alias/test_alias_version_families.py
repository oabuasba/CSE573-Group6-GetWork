import tomllib
from pathlib import Path

import pytest

from getwork.alias import apply, build, versions
from getwork.structure import schema

pytestmark = pytest.mark.usefixtures("offline")

WINDOWS_VARIANTS = [
    "Windows 7", "Windows 10", "Windows 11", "Windows XP", "Windows Server 2019",
    "Windows Server 2012 R2", "Microsoft Windows 10",
]
NOT_VERSIONS = [
    "S3", "EC2", "Route 53", "ISO 27001", "ISO 9001", "FIPS 199", "Cisco 3750", "Catalyst 2950",
    "Nexus 7K", "802.11", "T1", "Log4j", "Web 2.0", "IPv6", "Windows Phone", "Java ME",
    "JavaScript",
]


def test_shipped_families_load_and_build() -> None:
    families = versions.load_families()
    bases = [f.base.casefold() for f in families]
    assert families
    assert len(bases) == len(set(bases))
    table = build.build_table({}, build.load_curated(), families)
    assert table.canonical("Windows") == "Windows"


def test_shipped_families_in_package_data() -> None:
    root = Path(__file__).resolve().parents[3]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = config["tool"]["setuptools"]["package-data"]["getwork.alias"]
    assert versions.FAMILIES_PATH.name in package_data
    assert versions.FAMILIES_PATH.exists()


@pytest.mark.parametrize("skill", WINDOWS_VARIANTS)
def test_shipped_families_fold_windows_variants(skill: str) -> None:
    assert versions.fold_version(skill, versions.load_families()) == "Windows"


@pytest.mark.parametrize("skill", NOT_VERSIONS)
def test_shipped_families_leave_non_versions_alone(skill: str) -> None:
    assert versions.fold_version(skill, versions.load_families()) is None


def test_windows_variants_collapse_in_a_record() -> None:
    forms = {
        "Windows 7": 1, "Windows 10": 1, "windows 10": 1, "Windows 11": 1,
        "Windows Server 2019": 1, "Windows": 1, "Active Directory": 1,
    }
    table = build.build_table(forms, build.load_curated(), versions.load_families())
    skills = ["Windows 7", "Windows 10", "Windows 11", "Windows Server 2019",
              "Active Directory", "windows 10"]
    output = schema.ModelOutput(
        title="Dev", skills=skills, experience=[], education=[], years_experience=None
    )
    record = schema.to_record(
        output, record_id="r-1", source="resumes", model="m", prompt_version="p"
    )
    record["skills"] = list(skills)
    out = apply.canonicalize_record(record, table)
    assert out["skills"] == ["Windows", "Active Directory"]
    assert out["skills_raw"] == skills
