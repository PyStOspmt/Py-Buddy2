"""
Post-generation claim verifier.

The grounding gate already protects structured fields: recommended product
names and the prices bound to them. The free prose around those fields is
not covered -- a sentence may assert a spec, a stock level, a dose, or a
compatibility the source never stated. This module closes that gap at the
sentence level: every factual claim a sentence makes must be supported by
authorised source evidence, by a number the user stated, or by arithmetic
over supported numbers.

Scope, stated honestly:

* Numbers are checked against every numeric field the evidence carries
  (price, stock, rating, numeric attribute values), user-stated numbers,
  and additive closures of per-sentence product prices.
* Token claims (spec strings like "LGA1700", "DDR4-3600", quoted values)
  are checked against the mentioned product's name and its attribute
  values, and against any evidence row when no product is named.
* Verdicts are per sentence, never global: an answer is reported as the
  list of verified sentences plus the list of flagged claims, and the
  caller decides whether to drop, mark, or regenerate.

What it does NOT check: reasoning quality, opinionated wording, or claims
that carry no verifiable token (e.g. "це хороший вибір"). Those are marked
`no_claims` and pass -- the verifier only removes what the source refutes
or fails to support, never what it cannot judge.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ]+(?:[-–][0-9A-Za-zА-Яа-яІіЇїЄєҐґ]+)*")
# A number token: not glued to letters (so "i5-13400F" yields no 13400,
# while "DDR4-3600" still yields 3600 as a spec value) and not the start
# of a decimal fraction it would split mid-token.
_NUM_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)(?!\d)(?!\.\d)(?![A-Za-zА-Яа-яІіЇїЄєҐґ])")

# A leading list ordinal ("6.", "7)", "**3.**") is document formatting,
# not a factual claim; it is masked before numeric/spec extraction so a
# bare enumeration digit never needs evidence support. The trailing
# whitespace keeps decimals ("958.99 грн") and bare magnitudes ("45 Вт")
# in claim scope.
_LEADING_ORDINAL_RE = re.compile(
    r"^\s*(?:\*\*|__)?\d{1,3}[.)](?:\*\*|__)?(?:\s+|$)")
_SOLO_ORDINAL_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*|__)?\d{1,3}[.)](?:\*\*|__)?[ \t]*$")
_ABBR_BEFORE_DOT_RE = re.compile(
    r"(?:^|[\s(,/(])(?:тис|млн|млрд|див|напр|ім|вул|стор|рис|табл|"
    r"розд|розп|ред|акад|проф|кв|approx|etc|vs|т\.\s*[дпч]|e\.g|i\.e)\.$",
    re.IGNORECASE | re.UNICODE,
)
_INLINE_LIST_START_RE = re.compile(
    r"([ \t]+)(?=(?:\*\*|__)?(\d{1,2})\.(?:\*\*|__)?(?:\s*(?:\*\*|__|\[)|[ \t]+\S))"
)
_NUMBERED_SEQ_RE = re.compile(r"(?:^|\s)1\.\s+.+?\s2\.\s+", re.DOTALL)
_COUNT_UNIT_AFTER_RE = re.compile(
    r"^\s*(?:грн\b|грн\.|₴|uah\b|usd\b|eur\b|\$|€|вт\b|w\b|"
    r"мгц\b|ггц\b|mhz\b|ghz\b|мб\b|гб\b|тб\b|mb\b|gb\b|tb\b|"
    r"мм\b|см\b|м\b|кг\b|г\b|мг\b|мл\b|л\b|%|об/хв|rpm\b)",
    re.IGNORECASE | re.UNICODE,
)
_COUNT_CONTEXT_RE = re.compile(
    r"\b(?:знайдено|показано|виведено|обмежено|повернуто|отримано|"
    r"відібрано|доступн\w*|товар\w*|позиці\w*|варіант\w*|рядк\w*|"
    r"результат\w*|модел\w*|запис\w*|топ|top|items?|results?|found|shown)\b",
    re.IGNORECASE | re.UNICODE,
)

# Thousands-grouped price written with space / nbsp / thin-space
# separators, e.g. "15 738,12 ₴". Extracted before _NUM_RE so the inner
# groups do not appear as unsupported fragments.
_GROUPED_NUM_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?)"
    r"(?!\d)(?![A-Za-zА-Яа-яІіЇїЄєҐґ])")

# A spec phrase: a short lettered code followed by digits, possibly
# separated by a space or hyphen -- "LGA 1700", "PCIe 4.0", "DDR4 3600",
# "KC 3000". Cyrillic letter parts are prose ("версія 2", "крок 3"), not
# spec claims; the digits inside them are still masked from the bare
# number check because they belong to a composite token.
_SPEC_PHRASE_RE = re.compile(
    r"(?<![\w.])([A-Za-zА-Яа-яІіЇїЄєҐґ]{2,8})[- \u00a0\u202f]?"
    r"(\d+(?:[.,]\d+)?)(?!\.\d)(?!\w)")

_POS_COMPAT_RE = re.compile(
    r"\b(?:можна\s+встановити\s+разом|сумісн\w*|працюватим\w*\s+разом|"
    r"підійде\s+до|підходить\s+до|обидва\s+компоненти)\b",
    re.IGNORECASE | re.UNICODE,
)
_NEG_COMPAT_RE = re.compile(
    r"\b(?:не\s*сумісн\w*|несумісн\w*|не\s+можна|неможлив\w*|"
    r"конфлікт\w*|не\s+підход\w*|не\s+працюватим\w*|різн\w*\s+сокет\w*)\b",
    re.IGNORECASE | re.UNICODE,
)
_JOINT_CLAIM_RE = re.compile(
    r"\b(?:обидва|обидві|обох|разом|сумісн\w*|спільн\w*)\b",
    re.IGNORECASE | re.UNICODE,
)

_NEG_EXISTENCE_RE = re.compile(
    r"\b(?:немає|нема|відсутн\w*|жодн\w*|не\s+знайдено|не\s+існує)\b",
    re.IGNORECASE | re.UNICODE,
)
_PRICE_CAP_RE = re.compile(
    r"\b(?:до|не\s+більше|дешевш\w*\s+(?:ніж|за)|under|below|up\s+to)\s+"
    r"(?:[$€₴]\s*)?(\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?)"
    r"(?:\s*(?:(?:грн|uah|usd|eur)\b|[₴$€]))?",
    re.IGNORECASE | re.UNICODE,
)
_RANGE_FROM_RE = re.compile(
    r"\b(?:від|from|between)\s+(?:[$€₴]\s*)?\d[\d \u00a0\u202f'.,]*"
    r"(?:(?:грн|uah|usd|eur)\b|[₴$€])?[^.!?\n]{0,20}$",
    re.IGNORECASE | re.UNICODE,
)
_NON_CATALOG_NEG_PRE_RE = re.compile(
    r"\b(?:дефіцит\w*|нестач\w*|брак\w*|проблем\w*|помил\w*|конфлікт\w*|"
    r"несумісност\w*|ризик\w*|сумнів\w*|обмежен\w*|різниц\w*|перешкод\w*|затрим\w*)\s+$",
    re.IGNORECASE | re.UNICODE,
)
_NON_CATALOG_NEG_POST_RE = re.compile(
    r"^\s+(?:дефіцит\w*|нестач\w*|брак\w*|проблем\w*|помил\w*|конфлікт\w*|"
    r"несумісност\w*|ризик\w*|сумнів\w*|обмежен\w*|різниц\w*|перешкод\w*|затрим\w*)\b",
    re.IGNORECASE | re.UNICODE,
)
# Labelled language lexicon (generic Ukrainian prose function/comparison
# words across arbitrary domains, NOT a domain-specific catalog dictionary).
_EXCLUSION_STOP = frozenset({
    "поточному", "каталозі", "каталогу", "наявності", "товарів", "товари",
    "товар", "моделі", "моделей", "варіанти", "варіантів", "ціною", "ціна",
    "ціни", "менше", "більше", "дешевше", "жаль", "наразі", "базі", "даних",
    "позицій", "позиції", "вашим", "запитом", "порадь", "підкажи", "знайти",
})

#: Tokens that carry no factual weight on their own: units, currencies,
#: and high-frequency function words. A token not in this list and not
#: purely numeric is a candidate claim token.
_STOP_TOKENS = frozenset({
    "грн", "usd", "eur", "шт", "шт.", "см", "мм", "м", "км", "кг", "г",
    "мг", "л", "мл", "вт", "квт", "год", "хв", "с", "%", "id", "rev",
    "і", "в", "у", "на", "з", "зі", "що", "це", "ця", "цей", "для",
    "як", "або", "та", "до", "від", "по", "за", "є", "не", "ні",
})


@dataclass(frozen=True)
class EvidenceItem:
    """One authorised evidence row: what a claim may lean on."""

    name: str
    price: Optional[float] = None
    numbers: frozenset = frozenset()      # every numeric field the row carries
    tokens: frozenset = frozenset()       # name + attribute-value tokens
    # Name + attribute keys/values with every separator stripped, so a
    # claim "LGA1700" matches an attribute written "LGA 1700" and vice
    # versa. Substring check, token order free.
    claim_text: str = ""


@dataclass
class SentenceVerdict:
    text: str
    start: int
    end: int
    verdict: str                        # "verified" | "unsupported" | "no_claims"
    unsupported: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    sentences: List[SentenceVerdict]
    corrected_text: str
    n_supported: int
    n_unsupported: int
    n_no_claims: int

    @property
    def supported_fraction(self) -> float:
        total = self.n_supported + self.n_unsupported
        return self.n_supported / total if total else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentences": [
                {
                    "text": s.text,
                    "verdict": s.verdict,
                    "unsupported": s.unsupported,
                    "products": s.products,
                }
                for s in self.sentences
            ],
            "n_supported": self.n_supported,
            "n_unsupported": self.n_unsupported,
            "n_no_claims": self.n_no_claims,
            "supported_fraction": round(self.supported_fraction, 4),
        }


def _norm(tok: str) -> str:
    return tok.lower().replace("ё", "е").replace("–", "-").strip(".,;:!?\"'«»()[]{}")

_CLAIM_STRIP = re.compile(r"[^0-9a-zа-яіїєґ]+")


def _claim_norm(text: str) -> str:
    """Lowercased alphanumeric-only form: 'LGA 1700' -> 'lga1700'."""
    return _CLAIM_STRIP.sub("", text.lower().replace("ё", "е"))


def _tokens(text: str) -> Set[str]:
    return {_norm(t) for t in _WORD_RE.findall(text)} - {""}


def _numbers_masked(
        text: str, extra_masked: Sequence[Tuple[int, int]],
) -> Set[float]:
    out: Set[float] = set()
    masked: List[Tuple[int, int]] = list(extra_masked)
    for m in _GROUPED_NUM_RE.finditer(text):
        try:
            out.add(float(re.sub(r"[ \u00a0\u202f]", "", m.group(1))
                          .replace(",", ".")))
            masked.append((m.start(), m.end()))
        except ValueError:
            continue
    for m in _NUM_RE.finditer(text):
        if any(a <= m.start() < b for a, b in masked):
            continue
        try:
            out.add(float(m.group(1).replace(",", ".")))
        except ValueError:
            continue
    return out


def _numbers(text: str) -> Set[float]:
    return _numbers_masked(text, ())


def split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Split text into sentence/item spans without breaking decimals, abbreviations, or list ordinals."""
    if not text:
        return []
    has_seq = bool(_NUMBERED_SEQ_RE.search(text))
    inline_splits: Dict[int, int] = {}
    for m in _INLINE_LIST_START_RE.finditer(text):
        tail = text[m.end():]
        is_bold_or_link = bool(re.match(r"^(?:\*\*|__|\[)", tail))
        if is_bold_or_link or (has_seq and re.match(r"^\S", tail)):
            inline_splits[m.start(1)] = m.end(1)

    spans: List[Tuple[int, int]] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        if i in inline_splits and i > start:
            if text[start:i].strip():
                spans.append((start, i))
            i = inline_splits[i]
            start = i
            continue

        ch = text[i]
        if ch in "\r\n":
            prefix = text[start:i]
            if _SOLO_ORDINAL_PREFIX_RE.match(prefix):
                while i < n and text[i] in "\r\n":
                    i += 1
                continue
            end = i
            while i < n and text[i] in "\r\n":
                i += 1
            if text[start:end].strip():
                spans.append((start, end))
            start = i
            continue

        if ch in ".!?…" and i + 1 < n and text[i + 1].isspace():
            prefix = text[start : i + 1]
            if _SOLO_ORDINAL_PREFIX_RE.match(prefix):
                i += 1
                continue
            if ch == ".":
                seg = text[max(0, i - 16) : i + 1]
                if _ABBR_BEFORE_DOT_RE.search(seg):
                    i += 1
                    continue
                if seg.lower().endswith((" т.", "\nт.", " т")) or seg.lower() == "т.":
                    if re.match(r"^\s*[дпч]\.", text[i + 1 :], re.IGNORECASE):
                        i += 1
                        continue
            end = i + 1
            j = end
            while j < n and text[j] in " \t":
                j += 1
            if text[start:end].strip():
                spans.append((start, end))
            start = j
            i = j
            continue

        i += 1

    if text[start:].strip():
        spans.append((start, n))
    return spans


_sentence_spans = split_sentence_spans


def evidence_item_from_row(row: Dict[str, Any]) -> EvidenceItem:
    """Flatten a catalogue row (+ optional 'attributes' dict) into tokens/numbers."""
    numbers: Set[float] = set()
    tokens: Set[str] = set(_tokens(str(row.get("name", ""))))
    claim_parts: List[str] = []
    # Model numbers live inside product names ("Ryzen 5 5600", "Rev 1");
    # they are carried values, not optional spec text.
    numbers |= _numbers(str(row.get("name", "")))
    claim_parts.append(str(row.get("name", "")))
    for key, value in row.items():
        if key == "name" or value is None:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numbers.add(float(value))
            claim_parts.append(str(key))
        elif isinstance(value, str):
            tokens |= _tokens(value)
            numbers |= _numbers(value)
            claim_parts.append(str(key))
            claim_parts.append(value)
        elif isinstance(value, dict):
            for vk, vv in value.items():
                tokens |= _tokens(str(vk)) | _tokens(str(vv))
                claim_parts.append(str(vk))
                claim_parts.append(str(vv))
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    numbers.add(float(vv))
                elif isinstance(vv, str):
                    numbers |= _numbers(vv)
    price = row.get("price")
    return EvidenceItem(
        name=str(row.get("name", "")),
        price=float(price) if isinstance(price, (int, float)) else None,
        numbers=frozenset(numbers),
        tokens=frozenset(tokens),
        claim_text=_claim_norm(" ".join(claim_parts)),
    )


def _has_catalog_neg_existence(sent: str) -> bool:
    """Return True if sentence denies item existence rather than scarcity/errors."""
    for m in _NEG_EXISTENCE_RE.finditer(sent):
        before = sent[max(0, m.start() - 36):m.start()]
        after = sent[m.end():m.end() + 36]
        if _NON_CATALOG_NEG_PRE_RE.search(before) or _NON_CATALOG_NEG_POST_RE.search(after):
            continue
        return True
    return False


def _find_price_cap(sent: str, user_query: str) -> Optional[re.Match]:
    """Find an upper price cap while ignoring 'від X ... до Y' interval bounds."""
    for m in _PRICE_CAP_RE.finditer(sent):
        prefix = sent[max(0, m.start() - 48):m.start()]
        if _RANGE_FROM_RE.search(prefix):
            continue
        return m
    return _PRICE_CAP_RE.search(user_query)


def _contradicts_existence(
    sent: str,
    user_query: str,
    evidence: Sequence[EvidenceItem],
) -> Optional[str]:
    """Flag false-negative existence claims contradicted by a price-bounded row."""
    if not evidence or not _has_catalog_neg_existence(sent):
        return None
    m = _find_price_cap(sent, user_query)
    if not m:
        return None
    try:
        cap = float(re.sub(r"[ \u00a0\u202f]", "", m.group(1)).replace(",", "."))
    except ValueError:
        return None
    subj_tokens = [
        t for t in (_tokens(sent) | _tokens(user_query))
        if len(t) >= 5 and t not in _STOP_TOKENS and t not in _EXCLUSION_STOP
        and not any(ch.isdigit() for ch in t)
    ]
    for item in evidence:
        if item.price is None or item.price > cap:
            continue
        if not subj_tokens:
            return f"суперечить каталогу (є позиція ≤ {cap:g})"
        stems = [t[:5] for t in subj_tokens]
        if any(any(tok.startswith(st) or st.startswith(tok[:5])
                   for tok in item.tokens if len(tok) >= 5)
               for st in stems):
            return f"суперечить каталогу (є «{item.name}» за {item.price:g} ≤ {cap:g})"
    return None


def _arithmetic_closure(base: Set[float]) -> Set[float]:
    """Pairwise sums/differences of supported numbers (totals, deltas)."""
    out = set(base)
    vals = list(base)
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            out.add(a + b)
            out.add(abs(a - b))
    return out


def _is_grounded_tool_count(
    sent: str,
    num: float,
    tool_count_set: frozenset,
) -> bool:
    """Permit tool row_count cardinalities ONLY in non-unit count context."""
    if not tool_count_set or abs(num - round(num)) > 1e-6:
        return False
    target = int(round(num))
    if target not in tool_count_set:
        return False
    matches = list(re.finditer(rf"(?<![\d.,]){target}(?![\d.,])", sent))
    if not matches:
        return False
    for m in matches:
        after = sent[m.end() :]
        if _COUNT_UNIT_AFTER_RE.match(after):
            return False
        window = sent[max(0, m.start() - 40) : min(len(sent), m.end() + 45)]
        if not _COUNT_CONTEXT_RE.search(window):
            return False
    return True


def verify_answer(
    text: str,
    evidence: Sequence[EvidenceItem],
    user_query: str = "",
    *,
    number_tolerance: float = 0.51,
    tool_conflicts: Optional[Sequence[str]] = None,
    compatibility_verdict: Optional[str] = None,
    tool_counts: Optional[Sequence[int]] = None,
) -> VerificationReport:
    """
    Check each sentence's verifiable claims against `evidence`.

    `user_query` numbers are intent/context the answer may legitimately
    repeat (budget, size the user gave). They never validate a product fact.
    `tool_counts` permits exact integer cardinalities returned by executed tools
    ONLY when cited in explicit result-count wording without price/spec units.
    """
    if not text or not text.strip():
        return VerificationReport([], "", 0, 0, 0)

    tool_count_set = frozenset(
        int(c) for c in (tool_counts or ()) if isinstance(c, (int, float)) and int(c) >= 0
    )

    ev_numbers: Set[float] = set()
    for item in evidence:
        ev_numbers |= set(item.numbers)
        if item.price is not None:
            ev_numbers.add(item.price)

    user_nums = _numbers(user_query)
    # Products the answer talks about. A spec claim about "this CPU" must
    # lean on these rows -- a socket value that merely exists on some
    # unrelated catalogue row is not support, it is coincidence.
    answer_items = [it for it in evidence
                    if it.name and _norm(it.name) in _norm(text)]

    # Base support: values the source actually carries, sorted once so a
    # tolerance check is a bisect, not a scan. The arithmetic closure is
    # deliberately per-sentence (see below) -- a closure over the entire
    # catalogue would let any fabricated number sit near some unrelated
    # pair sum and would not scale.
    base_support = sorted(ev_numbers | user_nums)

    def _num_ok(num: float, extra: Sequence[float] = ()) -> bool:
        support = base_support
        if extra:
            support = sorted(
                base_support
                + list(_arithmetic_closure(set(extra) | user_nums)))
        i = bisect.bisect_left(support, num - number_tolerance)
        return i < len(support) and support[i] <= num + number_tolerance

    verdicts: List[SentenceVerdict] = []
    kept: List[str] = []
    n_ok = n_bad = n_none = 0

    for start, end in _sentence_spans(text):
        sent = text[start:end].strip()
        sent_num = _LEADING_ORDINAL_RE.sub("", sent, count=1)
        scoped_items = [it for it in evidence
                        if it.name and _norm(it.name) in _norm(sent)]
        products = list(dict.fromkeys(it.name for it in scoped_items))

        unsupported: List[str] = []

        # Totals/differences over the items named in this sentence are
        # legitimate derived claims; everything else must be a carried value.
        scoped_numbers: List[float] = []
        for it in scoped_items:
            scoped_numbers.extend(it.numbers)
            if it.price is not None:
                scoped_numbers.append(it.price)

        # Composite spec phrases first: their digit parts are claimed as
        # one unit ("LGA 1700"), never as a bare 1700 that might coincide
        # with a price somewhere else.
        phrase_spans: List[Tuple[int, int]] = []
        phrase_claims: List[str] = []
        for m in _SPEC_PHRASE_RE.finditer(sent_num):
            # Only Latin-script codes are spec phrases ("LGA 1700").
            # Cyrillic letters + digits are prose ("коштує 4500") -- the
            # number stays a normal claim and must stand on its own.
            if re.search(r"[A-Za-z]", m.group(1)):
                phrase_spans.append(m.span())
                phrase_claims.append(
                    _claim_norm(m.group(1) + m.group(2)))

        for num in _numbers_masked(sent_num, phrase_spans):
            if not _num_ok(num, scoped_numbers) and not _is_grounded_tool_count(
                sent_num, num, tool_count_set
            ):
                unsupported.append(f"число {num:g}")

        neg_contra = _contradicts_existence(sent, user_query, evidence)
        if neg_contra:
            unsupported.append(neg_contra)

        if (tool_conflicts or compatibility_verdict in ("incompatible", "candidate_conflict")) \
                and _POS_COMPAT_RE.search(sent) and not _NEG_COMPAT_RE.search(sent):
            unsupported.append("суперечить перевірці сумісності (виявлено конфлікт)")

        # A spec claim leans on the products named in this sentence; if the
        # sentence names none, on the products the whole answer names.
        # For explicit joint claims ("обидва компоненти підтримують AM5"),
        # every referenced item must agree, not just one of them.
        claim_scope = scoped_items or answer_items
        require_all = (
            not scoped_items
            and len(answer_items) >= 2
            and bool(_JOINT_CLAIM_RE.search(sent))
        )
        agg = all if require_all else any
        for claim in phrase_claims:
            if claim_scope:
                ok = agg(claim in it.claim_text for it in claim_scope)
            else:
                ok = any(claim in it.claim_text for it in evidence)
            if not ok and claim not in _claim_norm(user_query):
                unsupported.append(f"«{claim}»")

        # Claim tokens: word tokens outside the stop list that assert a
        # spec value -- containing a digit or a hyphen, or being a proper
        # catalog-style identifier. Plain Ukrainian words are not claims.
        for tok in sorted(_tokens(sent_num)):
            if (tok in _STOP_TOKENS or len(tok) < 3
                    or tok.replace(",", "").replace(".", "").isdigit()):
                # Purely numeric tokens are judged by the number check
                # above (which allows the arithmetic closure); re-checking
                # them here as "spec claims" would flag every true price.
                continue
            # A spec claim must carry a digit (LGA1700, DDR4-3600, RTX-4090)
            # or be a bare uppercase code (AM5, ATX). A hyphen alone does
            # not qualify -- "будь-якого" is prose, not a spec.
            is_spec = (any(ch.isdigit() for ch in tok)
                       or (tok.isupper() and len(tok) >= 2))
            if not is_spec:
                continue
            if _claim_norm(tok) in phrase_claims:
                continue  # already judged as a phrase
            if claim_scope:
                ok = agg(_claim_norm(tok) in it.claim_text
                         for it in claim_scope)
            else:
                ok = any(_claim_norm(tok) in it.claim_text
                         for it in evidence)
            if not ok and tok not in _norm(user_query):
                unsupported.append(f"«{tok}»")

        if (not unsupported and not products and not phrase_claims
                and not _numbers_masked(sent_num, phrase_spans)
                and not any(
                    (any(ch.isdigit() for ch in t) or "-" in t)
                    for t in _tokens(sent_num) - _STOP_TOKENS)):
            verdict = "no_claims"
            n_none += 1
            kept.append(sent)
        elif unsupported:
            verdict = "unsupported"
            n_bad += 1
        else:
            verdict = "verified"
            n_ok += 1
            kept.append(sent)

        verdicts.append(SentenceVerdict(
            text=sent, start=start, end=end,
            verdict=verdict, unsupported=unsupported, products=products,
        ))

    return VerificationReport(
        sentences=verdicts,
        corrected_text="\n".join(kept),
        n_supported=n_ok,
        n_unsupported=n_bad,
        n_no_claims=n_none,
    )


def verify_answer_against_rows(
    text: str,
    rows: Sequence[Dict[str, Any]],
    user_query: str = "",
    **kw: Any,
) -> VerificationReport:
    """Convenience wrapper for raw catalogue dicts."""
    return verify_answer(
        text, [evidence_item_from_row(r) for r in rows], user_query, **kw)
