import pytest

from getwork.preprocess import text

ENGLISH = (
    "Experienced software engineer with eight years of experience building web applications. "
    "Led a team of five developers to deliver a customer portal on time and under budget. "
    "Skilled in Python, databases, and automated testing, and passionate about mentoring "
    "junior colleagues."
)
SPANISH = (
    "Ingeniero de software con ocho años de experiencia en el desarrollo de aplicaciones web. "
    "Dirigí un equipo de cinco desarrolladores para entregar un portal de clientes a tiempo. "
    "Tengo experiencia en bases de datos y pruebas automatizadas, y me gusta ayudar a los "
    "compañeros nuevos."
)


def test_decode_utf8():
    assert text.decode_bytes(b"abc") == ("abc", "utf-8")
    assert text.decode_bytes("• item".encode("utf-8")) == ("• item", "utf-8")


def test_decode_cp1252():
    assert text.decode_bytes(b"\x95 item") == ("• item", "cp1252")


def test_decode_invalid_raises():
    with pytest.raises(ValueError):
        text.decode_bytes(b"\x81\x8d")


def test_clean_text_strips_spans_only():
    raw = 'a <span class="hl">b</span> c </SPAN>d <div>e</div>'
    assert text.clean_text(raw) == "a b c d <div>e</div>"


def test_clean_text_entities_and_whitespace():
    assert text.clean_text("  A &amp; B&nbsp;C\xa0D \n\t E • F  ") == "A & B C D E • F"


def test_clean_text_idempotent():
    raw = '<span x="1">Tom &amp;  Jerry</span>\xa0\n• <div>x</div>'
    once = text.clean_text(raw)
    assert text.clean_text(once) == once


def test_alpha_ratio():
    assert text.alpha_ratio("ab 12") == 0.5
    assert text.alpha_ratio("abc") == 1.0
    assert text.alpha_ratio("12 34") == 0.0
    assert text.alpha_ratio("") == 1.0
    assert text.alpha_ratio("  \n") == 1.0


def test_control_ratio():
    assert text.control_ratio("") == 0.0
    assert text.control_ratio("ab\n\r\tcd") == 0.0
    assert text.control_ratio("a\x00b\x01") == 0.5
    assert text.control_ratio("ab�") == 0.5


def test_detect_language_english():
    lang, prob = text.detect_language(ENGLISH)
    assert lang == "en" and prob >= 0.9


def test_detect_language_spanish():
    lang, prob = text.detect_language(SPANISH)
    assert lang == "es" and prob >= 0.9


def test_detect_language_unknown_and_deterministic():
    assert text.detect_language("1234 5678") == ("unknown", 0.0)
    assert text.detect_language(ENGLISH) == text.detect_language(ENGLISH)
