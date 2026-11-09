import pytest

from getwork.docembed import serialize

pytestmark = pytest.mark.usefixtures("offline")

ORDER = ["Front-end", "Cloud"]


def _items(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"skill": s, "cluster": c, "via": "lookup"} for s, c in pairs]


MIXED = _items(("AWS", "Cloud"), ("Excel", "Other"), ("Vue", "Front-end"), ("CSS", "Front-end"))


def test_full_exact_text_and_ordering() -> None:
    text = serialize.serialize_full("Backend Engineer", "Associate", MIXED, ORDER)
    assert text == (
        "Title: Backend Engineer\nLevel: Associate\n"
        "Skills: Front-end: Vue, CSS; Cloud: AWS; Other: Excel"
    )
    only_cloud = _items(("AWS", "Cloud"))
    assert serialize.serialize_full("T", "Associate", only_cloud, ORDER).endswith(
        "Skills: Cloud: AWS"
    )
    assert serialize.group_skills(MIXED, ORDER) == [
        ("Front-end", ["Vue", "CSS"]), ("Cloud", ["AWS"]), ("Other", ["Excel"]),
    ]


def test_full_empty_and_missing_fields() -> None:
    assert serialize.serialize_full(None, "Not specified", [], ORDER) == (
        "Title: Not specified\nLevel: Not specified\nSkills: none"
    )
    assert serialize.serialize_full("  \n ", "", [], ORDER).startswith(
        "Title: Not specified\nLevel: Not specified\n"
    )
    blank = _items(("  ", "Cloud"), ("", "Nope"))
    assert serialize.serialize_full("T", "L", blank, ORDER).endswith("Skills: none")
    assert serialize.serialize_b2("T", "L", blank, ORDER).endswith("Skills: none")
    assert serialize.serialize_b2(None, "L", [], ORDER) == (
        "Title: Not specified\nLevel: L\nSkills: none"
    )
    assert serialize.clean_field(None) == "Not specified"
    assert serialize.SERIALIZATION_VERSION == "ser_v1"


def test_full_whitespace_and_unknown_cluster() -> None:
    items = _items(("Vue\n  JS", "Front-end"), ("  AWS   S3 ", "Cloud"))
    text = serialize.serialize_full("Back \n  end\tDev", "Mid-Senior level", items, ORDER)
    assert text == (
        "Title: Back end Dev\nLevel: Mid-Senior level\nSkills: Front-end: Vue JS; Cloud: AWS S3"
    )
    with pytest.raises(ValueError, match="Mystery"):
        serialize.serialize_full("T", "L", _items(("x", "Mystery")), ORDER)
    long_name = "C" * 100
    with pytest.raises(ValueError) as err:
        serialize.serialize_full("T", "L", _items(("x", long_name)), ORDER)
    assert "C" * 60 in str(err.value)
    assert "C" * 61 not in str(err.value)


def test_b2_is_full_without_labels() -> None:
    full = serialize.serialize_full("Backend Engineer", "Associate", MIXED, ORDER)
    b2 = serialize.serialize_b2("Backend Engineer", "Associate", MIXED, ORDER)
    assert b2 == "Title: Backend Engineer\nLevel: Associate\nSkills: Vue, CSS, AWS, Excel"
    stripped = full
    for name in [*ORDER, "Other"]:
        stripped = stripped.replace(f"{name}: ", "")
    assert stripped.replace("; ", ", ") == b2
    assert b2.splitlines()[:2] == full.splitlines()[:2]
    assert ";" not in b2


def test_b2_record_order_and_bad_order() -> None:
    b2 = serialize.serialize_b2("T", "L", MIXED, ORDER, order="record")
    assert b2 == "Title: T\nLevel: L\nSkills: AWS, Excel, Vue, CSS"
    with pytest.raises(ValueError, match="sorted"):
        serialize.serialize_b2("T", "L", MIXED, ORDER, order="sorted")
    with pytest.raises(ValueError, match="Mystery"):
        serialize.serialize_b2("T", "L", _items(("x", "Mystery")), ORDER, order="record")
