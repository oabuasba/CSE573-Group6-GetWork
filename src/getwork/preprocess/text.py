import html
import re
import unicodedata

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

SPAN_TAG = re.compile(r"</?span\b[^>]*>", re.IGNORECASE)
WHITESPACE_RUN = re.compile(r"\s+")
ALLOWED_CONTROL = frozenset("\n\r\t")
REPLACEMENT_CHAR = "�"


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Return (text, encoding). Try strict UTF-8, then strict cp1252.
    encoding is "utf-8" or "cp1252". Raise ValueError if both fail."""
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "cp1252"
    except UnicodeDecodeError as err:
        raise ValueError(f"bytes are neither UTF-8 nor cp1252 (offset {err.start})") from err


def clean_text(raw: str) -> str:
    """Remove <span ...> and </span> tags (case-insensitive), apply html.unescape,
    replace "\\xa0" with " ", collapse every whitespace run to one space, strip.
    Idempotent on its own output for ordinary text whose entities decode only once; not
    guaranteed for double-escaped entities such as "&amp;lt;". Leaves other tags and all
    other characters (including "•") alone."""
    text = html.unescape(SPAN_TAG.sub("", raw)).replace("\xa0", " ")
    return WHITESPACE_RUN.sub(" ", text).strip()


def alpha_ratio(text: str) -> float:
    """Letters divided by non-whitespace characters. 1.0 if there are none."""
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 1.0
    return sum(ch.isalpha() for ch in non_space) / len(non_space)


def control_ratio(text: str) -> float:
    """(characters in Unicode category Cc or Co, excluding "\\n", "\\r", "\\t",
    plus U+FFFD) divided by len(text). 0.0 for empty text."""
    if not text:
        return 0.0
    bad = sum(
        ch == REPLACEMENT_CHAR
        or (unicodedata.category(ch) in ("Cc", "Co") and ch not in ALLOWED_CONTROL)
        for ch in text
    )
    return bad / len(text)


def detect_language(text: str, sample_chars: int = 2000) -> tuple[str, float]:
    """langdetect on text[:sample_chars] with DetectorFactory.seed = 0.
    Return (top language code, probability).
    Return ("unknown", 0.0) if langdetect raises LangDetectException."""
    try:
        top = detect_langs(text[:sample_chars])[0]
    except LangDetectException:
        return "unknown", 0.0
    return top.lang, top.prob
