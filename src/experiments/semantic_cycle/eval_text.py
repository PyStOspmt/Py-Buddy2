"""Evaluation-side text normalisation and query validation (docs/18).

Used ONLY by the evaluation harness to decide whether a generated request
shares words with its target (lexical overlap) -- never by production code.
English stop words come from scikit-learn's published list.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]")
STEM_LEN = 5


def content_tokens(text: str) -> set[str]:
    out = {t.casefold() for t in _TOKEN.findall(text or "") if len(t) >= 3}
    return {t for t in out if t not in ENGLISH_STOP_WORDS and not t.isdigit()}


def stems(tokens: set[str]) -> set[str]:
    return {t[:STEM_LEN] for t in tokens}


def shared_words(query: str, reference_text: str) -> list[str]:
    """Reference tokens whose 5-char stem also occurs in the query."""
    q = stems(content_tokens(query))
    ref = content_tokens(reference_text)
    return sorted(t for t in ref if t[:STEM_LEN] in q)


def has_overlap(query: str, reference_text: str) -> bool:
    return bool(shared_words(query, reference_text))


def is_ukrainian(text: str) -> bool:
    cyr, lat = len(_CYR.findall(text)), len(_LAT.findall(text))
    return cyr >= 10 and lat == 0


def lexical_overlap_ratio(query: str, doc_text: str) -> float:
    """Share of query content stems that occur in the document text."""
    q = stems(content_tokens(query))
    if not q:
        return 0.0
    d = stems(content_tokens(doc_text))
    return len(q & d) / len(q)
