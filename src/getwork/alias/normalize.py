"""Skill-name normalization: the key that merges spelling variants."""

import unicodedata

TRAILING_PUNCTUATION = " .,;:"


def clean_surface(skill: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", skill).split())


def normalize_key(skill: str) -> str:
    # Only trailing punctuation is stripped so .NET, C++ and C# keep what is part of the name.
    return clean_surface(skill).casefold().rstrip(TRAILING_PUNCTUATION).lstrip(" ")
