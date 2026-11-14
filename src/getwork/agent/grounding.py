"""Reads skill names from text (ADR-35); pure functions, no I/O."""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from getwork.alias.normalize import normalize_key
from getwork.webapp.models import DocumentView

TOKEN_RE = re.compile(r"[A-Za-z0-9#+.]+")
MIN_SINGLE_TOKEN_CHARS = 3
# Vocabulary entries that are also plain English words and almost never a skill on their own (never add a word that
# can be a standalone resume skill, such as cloud or security). Written in lower case in prose they are not skill mentions;
# a declared skill is still checked against the documents, and a capitalised or multi-word form is still detected.
COMMON_WORD_KEYS = frozenset("lists list test code tools support service services email".split())


@dataclass(frozen=True)
class Vocabulary:
    skills: Mapping[str, str]  # normalize_key -> display spelling
    clusters: frozenset[str]  # normalized cluster names, never counted as skills
    max_tokens: int  # longest skill, in whitespace tokens


def make_vocabulary(skills: Iterable[str], clusters: Iterable[str]) -> Vocabulary:
    """Keys by normalize_key; the first display spelling wins; empty keys dropped; max_tokens at least 1."""
    table: dict[str, str] = {}
    for skill in skills:
        key = normalize_key(skill)
        if key and key not in table:
            table[key] = skill
    cluster_keys = frozenset(key for key in map(normalize_key, clusters) if key)
    longest = max((len(key.split()) for key in table), default=1)
    return Vocabulary(skills=table, clusters=cluster_keys, max_tokens=max(longest, 1))


def _tokens(text: str) -> list[str]:
    return [token.rstrip(".") for token in TOKEN_RE.findall(text) if token.rstrip(".")]


def _match_at(*, tokens: Sequence[str], start: int, vocab: Vocabulary) -> tuple[str, int] | None:
    """Longest vocabulary key starting at start, with its token length."""
    for size in range(min(vocab.max_tokens, len(tokens) - start), 0, -1):
        key = normalize_key(" ".join(tokens[start : start + size]))
        if key in vocab.skills:
            return key, size
    return None


def _is_ignored_match(*, key: str, size: int, written: str, vocab: Vocabulary) -> bool:
    too_short = size == 1 and len(key) < MIN_SINGLE_TOKEN_CHARS
    plain_word = size == 1 and key in COMMON_WORD_KEYS and written.islower()
    return key in vocab.clusters or too_short or plain_word


def skill_mentions(text: str, vocab: Vocabulary) -> list[str]:
    """Display spellings of vocabulary skills named in text, in order of first appearance, one per key.
    Tokens are runs of [A-Za-z0-9#+.] with trailing dots stripped. At each position the longest match of up to
    vocab.max_tokens tokens (compared with normalize_key) wins and the scan continues after it. A match whose
    key is in vocab.clusters is skipped. A single-token match shorter than 3 characters is ignored, and so is a
    lower-case single token in COMMON_WORD_KEYS (a plain English word such as "lists")."""
    tokens = _tokens(text)
    found: dict[str, str] = {}
    position = 0
    while position < len(tokens):
        match = _match_at(tokens=tokens, start=position, vocab=vocab)
        if match is None:
            position += 1
            continue
        key, size = match
        if not _is_ignored_match(key=key, size=size, written=tokens[position], vocab=vocab):
            found.setdefault(key, vocab.skills[key])
        position += size
    return list(found.values())


def doc_skill_keys(doc: DocumentView) -> frozenset[str]:
    """normalize_key of each skill of the document."""
    return frozenset(normalize_key(item.skill) for item in doc.skills)


def ungrounded_skills(text: str, docs: Sequence[DocumentView], vocab: Vocabulary) -> list[str]:
    """Skills named in text that are in no document of docs; cluster names never count (the hallucination
    definition). In order of first appearance, one per key."""
    grounded = frozenset().union(*(doc_skill_keys(doc) for doc in docs))
    return [skill for skill in skill_mentions(text, vocab) if normalize_key(skill) not in grounded]
