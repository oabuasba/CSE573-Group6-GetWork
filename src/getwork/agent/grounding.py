"""Deterministic claim verification and skill naming (ADR-20); pure functions, no I/O."""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from getwork.agent.models import MAX_CLAIMS, Claim, Fact
from getwork.alias.normalize import normalize_key
from getwork.webapp.models import DocumentView

TOKEN_RE = re.compile(r"[A-Za-z0-9#+.]+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
FORBIDDEN_TEXT = ("http://", "https://", "```")
MIN_SINGLE_TOKEN_CHARS = 3


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


def _is_ignored_match(*, key: str, size: int, vocab: Vocabulary) -> bool:
    too_short = size == 1 and len(key) < MIN_SINGLE_TOKEN_CHARS
    return key in vocab.clusters or too_short


def skill_mentions(text: str, vocab: Vocabulary) -> list[str]:
    """Display spellings of vocabulary skills named in text, in order of first appearance, one per key.
    Tokens are runs of [A-Za-z0-9#+.] with trailing dots stripped. At each position the longest match of up to
    vocab.max_tokens tokens (compared with normalize_key) wins and the scan continues after it. A match whose
    key is in vocab.clusters is skipped. A single-token match shorter than 3 characters is ignored."""
    tokens = _tokens(text)
    found: dict[str, str] = {}
    position = 0
    while position < len(tokens):
        match = _match_at(tokens=tokens, start=position, vocab=vocab)
        if match is None:
            position += 1
            continue
        key, size = match
        if not _is_ignored_match(key=key, size=size, vocab=vocab):
            found.setdefault(key, vocab.skills[key])
        position += size
    return list(found.values())


def doc_skill_keys(doc: DocumentView) -> frozenset[str]:
    """normalize_key of each skill of the document."""
    return frozenset(normalize_key(item.skill) for item in doc.skills)


def _keys_of(*, doc_ids: Iterable[str], scope_keys: Mapping[str, frozenset[str]]) -> frozenset[str]:
    return frozenset().union(*(scope_keys[doc_id] for doc_id in doc_ids if doc_id in scope_keys))


def _check_count(claims: Sequence[Claim]) -> list[str]:
    if 1 <= len(claims) <= MAX_CLAIMS:
        return []
    return [f"claims: give between 1 and {MAX_CLAIMS} claims, not {len(claims)}."]


def _check_text(text: str) -> list[str]:
    if any(bad in text for bad in FORBIDDEN_TEXT):
        return ["remove links and code blocks from the text."]
    return []


def _check_fact_ids(*, claim: Claim, facts: Mapping[str, Fact]) -> list[str]:
    if not claim.facts:
        return ["cite at least one fact id."]
    unknown = [fact_id for fact_id in claim.facts if fact_id not in facts]
    return [f'fact "{fact_id}" does not exist; cite only fact ids you were given.' for fact_id in unknown]


def _check_doc_ids(*, claim: Claim, scope: Mapping[str, DocumentView]) -> list[str]:
    if not claim.doc_ids:
        return ["name at least one document in doc_ids."]
    return [f'document "{doc_id}" is not in this conversation.' for doc_id in claim.doc_ids if doc_id not in scope]


def _check_fact_documents(*, claim: Claim, facts: Mapping[str, Fact]) -> list[str]:
    messages: list[str] = []
    if not claim.doc_ids:
        return messages
    for fact_id in claim.facts:
        fact = facts.get(fact_id)
        if fact is not None and not set(fact.doc_ids) & set(claim.doc_ids):
            messages.append(f'fact "{fact_id}" is about other documents than {", ".join(claim.doc_ids)}.')
    return messages


def _check_declared_skills(
    *, claim: Claim, facts: Mapping[str, Fact], scope_keys: Mapping[str, frozenset[str]]
) -> list[str]:
    doc_keys = _keys_of(doc_ids=claim.doc_ids, scope_keys=scope_keys)
    fact_keys = frozenset(normalize_key(s) for f in claim.facts if f in facts for s in facts[f].skills)
    messages = []
    for skill in claim.skills:
        key = normalize_key(skill)
        if claim.doc_ids and key not in doc_keys:
            messages.append(f'skill "{skill}" is not in the skills of {", ".join(claim.doc_ids)}.')
        elif key not in fact_keys:
            messages.append(f'skill "{skill}" is not in any fact you cited.')
    return messages


def _check_mentions(*, claim: Claim, scope_keys: Mapping[str, frozenset[str]], vocab: Vocabulary) -> list[str]:
    doc_keys = _keys_of(doc_ids=claim.doc_ids, scope_keys=scope_keys)
    declared = {normalize_key(skill) for skill in claim.skills}
    messages = []
    for skill in skill_mentions(claim.text, vocab):
        key = normalize_key(skill)
        if claim.doc_ids and key not in doc_keys:
            owners = ", ".join(claim.doc_ids)
            messages.append(f'the text names skill "{skill}", which is not in the skills of {owners}.')
        elif key not in declared:
            messages.append(f'the text names skill "{skill}"; add it to skills.')
    return messages


def _check_numbers(*, claim: Claim, facts: Mapping[str, Fact]) -> list[str]:
    cited_text = " ".join(facts[f].text for f in claim.facts if f in facts)
    cited_numbers = set(NUMBER_RE.findall(cited_text))
    unsupported = [n for n in NUMBER_RE.findall(claim.text) if n not in cited_numbers]
    return [f'the number {n} does not appear in the facts you cited.' for n in dict.fromkeys(unsupported)]


def verify_claims(
    claims: Sequence[Claim],
    *,
    facts: Mapping[str, Fact],
    scope: Mapping[str, DocumentView],
    vocab: Vocabulary,
) -> list[str]:
    """Violation messages, empty when everything holds. Each message starts with f"claim {i}: " (1-based).
    Rules V-1 to V-8: claim count; no links or code; fact ids exist; doc ids in scope; facts concern the claim's
    documents; declared skills are in those documents and cited facts; named skills are declared and in those
    documents; numbers appear in the cited facts. The V-1 message starts with "claims: " since no claim is at fault."""
    messages = _check_count(claims)
    scope_keys = {doc_id: doc_skill_keys(doc) for doc_id, doc in scope.items()}
    for number, claim in enumerate(claims, start=1):
        found = [
            *_check_text(claim.text),
            *_check_fact_ids(claim=claim, facts=facts),
            *_check_doc_ids(claim=claim, scope=scope),
            *_check_fact_documents(claim=claim, facts=facts),
            *_check_declared_skills(claim=claim, facts=facts, scope_keys=scope_keys),
            *_check_mentions(claim=claim, scope_keys=scope_keys, vocab=vocab),
            *_check_numbers(claim=claim, facts=facts),
        ]
        messages.extend(f"claim {number}: {message}" for message in found)
    return messages


def named_skills(claims: Sequence[Claim], vocab: Vocabulary) -> list[str]:
    """Declared skills plus skill_mentions of each claim text, deduplicated by key, first spelling, claim order."""
    named: dict[str, str] = {}
    for claim in claims:
        for skill in [*claim.skills, *skill_mentions(claim.text, vocab)]:
            key = normalize_key(skill)
            if key:
                named.setdefault(key, skill)
    return list(named.values())


def ungrounded_skills(claims: Sequence[Claim], docs: Sequence[DocumentView], vocab: Vocabulary) -> list[str]:
    """named_skills whose key is in no document of docs, cluster names excluded (the hallucination definition)."""
    grounded = frozenset().union(*(doc_skill_keys(doc) for doc in docs)) if docs else frozenset()
    return [
        skill
        for skill in named_skills(claims, vocab)
        if normalize_key(skill) not in grounded and normalize_key(skill) not in vocab.clusters
    ]
