"""Evaluation-side text normalisation and query validation.

Used ONLY by the evaluation harness to decide whether a generated request
shares words with its target (lexical overlap) -- never by production code.
"""

from __future__ import annotations

import re

ENGLISH_STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "an", "and", "another", "any", "anyhow",
    "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at",
    "back", "be", "became", "because", "become", "becomes", "becoming", "been",
    "before", "beforehand", "behind", "being", "below", "beside", "besides",
    "between", "beyond", "both", "bottom", "but", "by", "call", "can",
    "cannot", "could", "did", "do", "does", "doing", "done", "down", "due",
    "during", "each", "either", "else", "elsewhere", "empty", "enough", "even",
    "ever", "every", "everyone", "everything", "everywhere", "except", "few",
    "first", "for", "former", "formerly", "from", "front", "full", "further",
    "get", "give", "go", "had", "has", "have", "having", "he", "hence", "her",
    "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
    "him", "himself", "his", "how", "however", "i", "if", "in", "indeed",
    "into", "is", "it", "its", "itself", "just", "keep", "last", "latter",
    "latterly", "least", "less", "made", "many", "may", "me", "meanwhile",
    "might", "mine", "more", "moreover", "most", "mostly", "move", "much",
    "must", "my", "myself", "name", "namely", "neither", "never",
    "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not",
    "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one",
    "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
    "ourselves", "out", "over", "own", "part", "per", "perhaps", "please",
    "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
    "seems", "serious", "several", "she", "should", "show", "side", "since",
    "so", "some", "somehow", "someone", "something", "sometime", "sometimes",
    "somewhere", "still", "such", "take", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "thence", "there", "thereafter",
    "thereby", "therefore", "therein", "thereupon", "these", "they", "this",
    "those", "though", "through", "throughout", "thru", "thus", "to",
    "together", "too", "top", "toward", "towards", "under", "until", "up",
    "upon", "us", "very", "via", "was", "we", "well", "were", "what",
    "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas",
    "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
    "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "with", "within", "without", "would", "yet", "you", "your", "yours",
    "yourself", "yourselves",
})

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
