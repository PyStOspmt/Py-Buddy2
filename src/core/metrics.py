"""
Py-Buddy: Honest evaluation metrics.

Design rules (ADR-015):
  * Every metric is a pure function of (response_text, oracle_ground_truth).
  * No LLM-as-a-judge. No keyword-counting shortcuts.
  * A claim is only "correct" if it is a projection of the oracle.

Reference for the factuality/informativeness trade-off this module is built
to measure: arXiv:2603.16817 ("Is Conformal Factuality for RAG-based LLMs
Robust?"), which shows that raising factuality to ~1.0 by filtering produces
*vacuous* answers. We therefore report factuality AND informativeness together.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# Claim extraction
# --------------------------------------------------------------------------

# Price mentions: "9500 грн", "9 500 грн", "9500грн", "17,200.00 грн", "1 234,56 грн".
# We grab the whole numeric *token* (digits + separators) and decide the
# decimal separator afterwards, because "17,200.00" and "1 234,56" are both
# common in Ukrainian retail and naive regexes silently truncate them.
# NOTE: \\s (not a literal space) — LLMs emit NARROW NO-BREAK SPACE (U+202F)
# or NO-BREAK SPACE (U+00A0) as thousand separators, and a literal " " misses
# them: "15\u202f308.61 UAH" then scores as 308.61 (see live_probe_direct_price
# rescoring 2026-09-30: 0/5 -> 5/5 on identical texts after this fix).
_PRICE_RE = re.compile(
    r"(?<![\w.])(\d[\d\s'.,]*\d|\d)\s*(?:грн\b|грн\.?|UAH|₴)",
    re.IGNORECASE,
)


def parse_number(token: str) -> Optional[float]:
    """
    Parse a human-written numeric token into a float.

    Rules (in order):
      * strip spaces, apostrophes (thousand separators);
      * if both ',' and '.' are present, the *last* one is the decimal mark;
      * if only ',' is present and is followed by exactly 2 digits, treat it as
        a decimal mark, otherwise as a thousand separator;
      * if only '.' is present and is followed by exactly 2 digits, decimal.
    """
    t = re.sub(r"[\s\u00a0']", "", token)
    if not t or not re.match(r"^\d", t):
        return None
    last_comma, last_dot = t.rfind(","), t.rfind(".")
    if last_comma != -1 and last_dot != -1:
        # Whichever appears last is the decimal mark; the other groups by 3.
        dec = max(last_comma, last_dot)
        t = t[:dec].replace(",", "").replace(".", "") + "." + t[dec + 1:]
    else:
        sep = last_comma if last_comma != -1 else last_dot
        if sep != -1:
            # Exactly three trailing digits => thousands separator, else decimal.
            if len(t) - sep - 1 == 3:
                t = t[:sep] + t[sep + 1:]
            else:
                t = t[:sep] + "." + t[sep + 1:]
    try:
        return round(float(t), 2)
    except ValueError:
        return None

# F48: the cue lexicon below is an explicitly labelled HEURISTIC FALLBACK
# layer for classifying free-text answers whose producer did not emit a
# structured verdict. It is Ukrainian-scoped and unmeasured on other
# languages -- an English response it cannot parse classifies "neutral",
# which is honest (no verdict detected) rather than silently wrong. Where
# the evaluated system carries a real verdict in its own structure (step
# records, tool output), score_case prefers that verdict via the
# `structured_verdict` argument and this lexicon never runs.
_ACCEPT_CUES = (
    "сумісн", "підійд", "підход", "можна встановити", "рекомендую",
    "придатн", "валідн", "доступн", "в наявності", "є в наявності",
    "порадь", "підбираю", "оптимальн",
)
_REJECT_CUES = (
    "несумісн", "не підійд", "не підход", "не мож", "немає", "не знайд",
    "відсутн", "бракує", "відхилен", "заборонен", "непідход", "конфлікт",
    "не вистач", "перевищу", "не сумісн", "не вдасться", "недоступн",
    "закінчив", "не рекоменду", "нерекоменду", "дефіцит", "не варто", "недоцін",
)
_NEGATED_CONFLICT_RE = re.compile(
    r"\b(?:без|немає|нема|не\s+виявлен\w*|не\s+знайден\w*|відсутн\w*|жодн\w*)\s+конфлікт\w*"
    r"|\bконфлікт\w*\s+(?:не\s+виявлен\w*|немає|нема|не\s+знайден\w*|відсутн\w*)"
    r"|\bне\s+бракує",
    re.IGNORECASE | re.UNICODE,
)
# A positive cue counts only when NOT immediately preceded by a negation.
# Without this, the cue "сумісн" fires inside "несумісний" and every
# rejection is misread as an acceptance.
_NEGATIONS = ("не", "ні", "неє", "без", "ніколи", "немож")
_NEG_WINDOW = 6

_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)


def _cue_variants(cues: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    """
    Ukrainian writes "не сумісний" and "несумісний" interchangeably, so cue
    matching must work both with and without internal whitespace.
    """
    spaced = {normalize(c) for c in cues}
    compact = {re.sub(r"\s+", "", c) for c in spaced}
    return spaced, compact


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def _norm_key(text: str) -> str:
    """Aggressive key for fuzzy product matching: only word chars, no spaces."""
    return re.sub(r"[\s_\-]+", "", normalize(text))


def extract_prices(text: str) -> List[float]:
    """Return every price figure mentioned in the response, in order."""
    out: List[float] = []
    for raw in _PRICE_RE.findall(text or ""):
        value = parse_number(raw)
        if value is not None:
            out.append(value)
    return out


_STRUCTURED_VERDICTS = {"accept": "accept", "compatible": "accept",
                        "reject": "reject", "incompatible": "reject",
                        "candidate_conflict": "reject",
                        "neutral": "neutral", "unknown": "neutral"}


def extract_verdict(text: str,
                    structured_verdict: Optional[str] = None) -> str:
    """
    Classify the response stance as 'accept' | 'reject' | 'neutral'.

    F48: a structured verdict carried by the evaluated system's own
    output wins over text classification -- the cue lexicon is only the
    fallback for answers that carry no verdict field. Text-side, the
    classifier looks at the sentence that carries the conclusion (last
    non-empty sentence, which is where a well-formed assistant puts its
    verdict) and falls back to the whole text.
    """
    mapped = _STRUCTURED_VERDICTS.get(
        str(structured_verdict).lower()) if structured_verdict else None
    if mapped is not None:
        return mapped
    norm = normalize(text)
    if not norm:
        return "neutral"

    reject_s, reject_c = _cue_variants(_REJECT_CUES)
    accept_s, accept_c = _cue_variants(_ACCEPT_CUES)

    def _mask_negated_rejects(text_win: str) -> Tuple[str, str]:
        cleaned = _NEGATED_CONFLICT_RE.sub(" ", text_win)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned, re.sub(r"\s+", "", cleaned)

    def has_reject(window: str, compact: str) -> bool:
        w_clean, c_clean = _mask_negated_rejects(window)
        return (any(c in w_clean for c in reject_s)
                or any(c in c_clean for c in reject_c))

    def has_accept(window: str, compact: str) -> bool:
        """Positive cue that is not negated, e.g. 'сумісна' yes, 'несумісна' no."""
        for hay, cues in ((window, accept_s), (compact, accept_c)):
            for cue in cues:
                for m in re.finditer(re.escape(cue), hay):
                    prefix = hay[max(0, m.start() - _NEG_WINDOW):m.start()].strip()
                    if not any(prefix.endswith(n) for n in _NEGATIONS):
                        return True
        return False

    # A rejection cue is more specific than a positive one, so it wins when
    # both fire in the same window. Decision is taken on the concluding
    # sentence first, falling back to the whole answer unless an opening
    # rejection is merely followed by alternative suggestions.
    sentences = [s for s in re.split(r"[.!?\n]+", norm) if s.strip()]
    tail = sentences[-1] if sentences else norm
    tail_compact = re.sub(r"\s+", "", tail)
    norm_compact = re.sub(r"\s+", "", norm)
    if has_reject(tail, tail_compact):
        return "reject"
    head = sentences[0] if sentences else ""
    head_compact = re.sub(r"\s+", "", head)
    head_unavail = len(sentences) >= 2 and bool(
        re.search(r"\b(?:відсутн\w*|немає|не\s+знайд\w*|бракує)\b", _mask_negated_rejects(head)[0])
    )
    tail_overrides = bool(re.search(r"\b(?:проте|однак|але\s+насправді)\b", tail)) or (
        bool(re.search(r"\b(?:сумісн\w*|підійд\w*|можна\s+встановити)\b", tail))
        and not re.search(r"\b(?:натомість|замість|альтернатив\w*|схож\w*)\b", tail)
    )
    if head_unavail and has_reject(norm, norm_compact) and not tail_overrides:
        return "reject"
    if has_accept(tail, tail_compact):
        return "accept"
    if has_reject(norm, norm_compact):
        return "reject"
    if has_accept(norm, norm_compact):
        return "accept"
    return "neutral"


# "Model-ish" tokens. Two shapes are recognised:
#   letter-led   : "Ryzen 5", "RTX 4080", "ATX 3"
#   digit-led    : "7800X3D", "9950X", "7600X"
# A candidate is kept only if it contains BOTH a digit and an upper-case
# letter (case-sensitive, on the raw text). Technical identifiers are
# case-bearing ("RTX", "B650", "7600X"); ordinary prose with a number is not
# ("за 9 500 грн", "3 штуки"). That single filter removes the false positives
# without a stop-word list.
# "Model-ish" tokens. Two shapes are recognised:
#   letter-led glued    : "LGA1700", "DDR5", "PF500", "i9-15900KS"
#   spaced ASCII model  : "Ryzen 5", "RTX 4080", "ATX 3" (single space ONLY,
#                         never across newlines and never trailing unit nouns)
#   digit-led           : "7800X3D", "9950X", "7600X"
# Using a literal single space avoids joining Cyrillic intro words ("Ось 5")
# or multiline list numbers ("Ukraine\n3.") with trailing digits.
_ENTITY_RE = re.compile(
    r"(?<![\w])[^\W\d_][\w\-]*\d[\w\-]*"
    r"|(?<![\w])[A-Za-z]*[A-Z][A-Za-z\-]* \d[\w\-]*"
    r"(?! *(?:[wW]|вт|Вт|kW|кВт|GB|TB|MB|ТБ|ГБ|МБ|mm|мм|cm|см|кг|kg|г|мг|mg|грн|UAH)\b)"
    r"|(?<![\w])\d[\w\-]*[A-Z][\w\-]*"
)


def extract_entities(text: str) -> List[str]:
    """Return model-like entity tokens mentioned in the text, deduplicated."""
    out: List[str] = []
    for raw in _ENTITY_RE.findall(text or ""):
        if not any(ch.isdigit() for ch in raw):
            continue
        if not any(ch.isupper() for ch in raw):
            continue
        tok = re.sub(r"[\s_\-]+", "", normalize(raw))
        if len(tok) >= 3 and tok not in out:
            out.append(tok)
    return out


def _extract_ground_tokens(name: str) -> List[str]:
    """Extract catalog entity tokens plus embedded unit-bearing specs ('LPX 32GB' -> '32gb', 'DDR4-3600' -> 'ddr4')."""
    toks = extract_entities(name)
    for part in re.findall(r"[0-9A-Za-z]+", name or ""):
        if any(ch.isdigit() for ch in part) and any(ch.isupper() for ch in part):
            tok = re.sub(r"[\s_\-]+", "", normalize(part))
            if len(tok) >= 3 and tok not in toks:
                toks.append(tok)
    return toks


def grounded_entities(
    text: str,
    catalog_names: Sequence[str],
    extra_grounded_terms: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Entity tokens in `text` with no counterpart anywhere in the catalog.

    A fully invented product ("AMD Ryzen 9 9950X") is invisible to name
    matching, because name matching only ever finds things that *exist*.
    This check closes that hole: every model-like token the answer
    introduces must be attributable to a real catalog entry (or an explicit
    catalog attribute value when `extra_grounded_terms` is supplied).
    """
    ground: Set[str] = set()
    for name in set(catalog_names):
        ground.update(_extract_ground_tokens(name))
    if extra_grounded_terms:
        for term in {str(t) for t in extra_grounded_terms}:
            ground.update(_extract_ground_tokens(term))
    return [t for t in extract_entities(text) if t not in ground]


def match_products(text: str, known_names: Sequence[str]) -> List[str]:
    """
    Find which catalog products are referred to in the response.

    Matching is on a whitespace/punctuation-stripped key, longest matches first
    with span masking so "X (Rev 1)" never also emits unsuffixed "X".
    Returns the *original* catalog names, deduplicated, in mention order.
    """
    haystack = _norm_key(text)
    if not haystack:
        return []
    keys: List[Tuple[str, str]] = sorted(
        ((_norm_key(n), n) for n in dict.fromkeys(known_names) if _norm_key(n)),
        key=lambda kv: len(kv[0]), reverse=True,
    )
    masked = haystack
    hits: List[Tuple[int, str]] = []
    seen: Set[str] = set()
    for key, original in keys:
        if original in seen:
            continue
        pos = masked.find(key)
        if pos != -1:
            hits.append((pos, original))
            seen.add(original)
            masked = masked.replace(key, "\x00" * len(key))
    hits.sort(key=lambda item: item[0])
    return [original for _, original in hits]


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------


@dataclass
class CaseScore:
    """Per-case evaluation. All fields are derived, never asserted."""

    case_id: str
    case_type: str

    # --- factual grounding -------------------------------------------------
    mentioned_products: List[str] = field(default_factory=list)
    grounded_products: int = 0        # mentioned AND present in the catalog
    ungrounded_products: List[str] = field(default_factory=list)
    mention_precision: float = 0.0    # grounded / mentioned
    mention_recall: float = 0.0       # expected products actually mentioned

    # --- entity-level grounding ------------------------------------------
    # Fabricated model identifiers ("Ryzen 9", "9950X") that exist nowhere
    # in the catalog. This is the only check that can see a wholly invented
    # product, because name matching structurally cannot.
    ungrounded_entities: List[str] = field(default_factory=list)

    # --- numeric grounding -------------------------------------------------
    mentioned_prices: List[float] = field(default_factory=list)
    price_errors: List[Tuple[float, float]] = field(default_factory=list)  # (said, truth)
    prices_grounded: bool = True

    # --- decision correctness ---------------------------------------------
    verdict: str = "neutral"
    expected_verdict: str = "neutral"
    verdict_correct: bool = False
    #: Unparsed tool-reported structured verdict (e.g. "unknown", "compatible").
    raw_structured_verdict: Optional[str] = None
    #: True when the system abstained ("neutral") on an accept/reject question.
    verdict_abstained: bool = False
    #: True when the system gave an active decision contradicting the oracle.
    verdict_contradicted: bool = False
    #: False when the question has no compatibility verdict to get right, so
    #: this axis carries no information for this case (see Oracle.checks_verdict).
    verdict_applicable: bool = True
    #: Require full coverage over oracle expected products whenever recall_required is True.
    recall_required: bool = False

    # --- informativeness --------------------------------------------------
    informative: bool = False
    blocked: bool = False             # answered with a hard error, no content
    steps: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0

    # --- violations -------------------------------------------------------
    violations: List[str] = field(default_factory=list)

    @property
    def is_correct(self) -> bool:
        """Execution accuracy: no false facts, the decision matches, AND the
        answer is complete about the products the task demands.

        A case with no compatibility verdict is evaluated on grounded facts alone;
        cases with expected products require both full precision and full recall.
        """
        return (
            not self.violations
            and (self.verdict_correct or not self.verdict_applicable)
            and self.mention_precision >= 1.0
            and (self.mention_recall >= 1.0 or not self.recall_required)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "is_correct": self.is_correct,
            "verdict": self.verdict,
            "expected_verdict": self.expected_verdict,
            "verdict_correct": self.verdict_correct,
            "raw_structured_verdict": self.raw_structured_verdict,
            "verdict_abstained": self.verdict_abstained,
            "verdict_contradicted": self.verdict_contradicted,
            "mention_precision": round(self.mention_precision, 4),
            "mention_recall": round(self.mention_recall, 4),
            "grounded_products": self.grounded_products,
            "ungrounded_products": self.ungrounded_products,
            "ungrounded_entities": self.ungrounded_entities,
            "prices_grounded": self.prices_grounded,
            "price_errors": self.price_errors,
            "informative": self.informative,
            "blocked": self.blocked,
            "steps": self.steps,
            "latency_s": round(self.latency_s, 3),
            "cost_usd": round(self.cost_usd, 6),
            "violations": self.violations,
        }


@dataclass
class Oracle:
    """Deterministic ground truth for one case, computed from the database."""

    case_id: str
    case_type: str
    expected_products: List[str] = field(default_factory=list)
    expected_prices: Dict[str, float] = field(default_factory=dict)
    expected_verdict: str = "neutral"
    feasible: bool = True          # does a hard-constraint solution exist?
    #: Whether a COMPATIBILITY verdict exists for this question at all.
    #:
    #: "Which is the cheapest CPU?" has a right answer and no verdict: a
    #: correct answer names the CPU, and nothing more. Scoring it on
    #: `expected_verdict` measures whether the answer happened to contain
    #: the word "сумісний", which the question never asked for.
    #:
    #: The selftest hid this: `_gold_answer` appended "Ця конфігурація
    #: сумісна і підходить" to any case whose expected verdict was accept,
    #: so the gold answer satisfied the metric by construction and the axis
    #: read 100% for reasons that had nothing to do with the system. The
    #: verdict axis is now scored only where a verdict exists, and the
    #: aggregate reports how many cases that was.
    checks_verdict: bool = True


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_case(
    response: str,
    oracle: Oracle,
    catalog_names: Sequence[str],
    price_index: Dict[str, object],
    *,
    steps: int = 0,
    latency_s: float = 0.0,
    cost_usd: float = 0.0,
    structured_verdict: Optional[str] = None,
    extra_grounded_terms: Optional[Sequence[str]] = None,
    user_query: str = "",
) -> CaseScore:
    """
    Score one response against one oracle.

    The four checks are independent and each one is a real defect if violated:

      1. mention_precision < 1  -> the answer named a product that is not
                                    in the catalog  (a hallucination)
      2. prices_grounded == False -> the answer quoted a figure that does not
                                    match the catalog                  (a lie)
      3. verdict != expected      -> wrong recommendation
      4. blocked                  -> gate refused, zero informative value
    """
    s = CaseScore(
        case_id=oracle.case_id,
        case_type=oracle.case_type,
        expected_verdict=oracle.expected_verdict,
        steps=steps,
        latency_s=latency_s,
        cost_usd=cost_usd,
    )

    norm_resp = normalize(response)
    s.mentioned_products = match_products(response, catalog_names)
    s.grounded_products = len(s.mentioned_products)
    s.mention_precision = (
        s.grounded_products / len(s.mentioned_products)
        if s.mentioned_products else 1.0
    )
    s.ungrounded_entities = grounded_entities(
        response, catalog_names, extra_grounded_terms=extra_grounded_terms)

    # Deduplicated: an availability oracle enumerates LISTINGS, and the same
    # name listed by two sellers appears twice. Recall over a duplicated
    # denominator can never reach 1.0 and would fail every complete answer.
    expected = sorted({n for n in oracle.expected_products if n})
    s.recall_required = bool(expected)
    if expected:
        key_map = {_norm_key(n): n for n in expected}
        hit = sum(1 for n in s.mentioned_products if _norm_key(n) in key_map)
        s.mention_recall = hit / len(expected)
        # F25: a mention that exists in the catalogue but is NOT one of the
        # products this task asks about is a false claim, not a neutral
        # extra. Without populating this field, false_claim_rate_pct was
        # structurally zero -- a dead metric.
        s.ungrounded_products = [
            n for n in s.mentioned_products if _norm_key(n) not in key_map
        ]

    # Price grounding: every quoted figure must equal a catalogue price of a
    # product the answer attributes it to.
    #
    # The catalogue is a MARKETPLACE: one product NAME is listed by up to six
    # sellers at six different prices (10 000 products, 190 distinct names).
    # A name therefore anchors a SET of acceptable prices, not one price. A
    # correct answer quoting any of them is grounded; any other figure is a
    # lie. Entries may be a float or an iterable of floats.
    #
    # F25: the binding is per SENTENCE. The previous check anchored every
    # figure against the union of every mentioned product's prices, so
    # "A коштує 9 500 грн, а B коштує 12 000 грн" passed even with the two
    # figures swapped -- the entity-offer binding did not exist. A sentence
    # that names no product falls back to the union (a bare price line still
    # has to match something the answer talked about).
    s.mentioned_prices = extract_prices(response)

    def anchors_for(names: Sequence[str]) -> Set[float]:
        out: Set[float] = set()
        for name in names:
            for source in (oracle.expected_prices.get(name), price_index.get(name)):
                if source is None:
                    continue
                if isinstance(source, (int, float)):
                    out.add(round(float(source), 2))
                else:
                    out.update(round(float(x), 2) for x in source)
        return out

    global_anchors = anchors_for(s.mentioned_products)

    # Positional entity-offer binding within each sentence:
    # - Preceding product mention in the SAME sentence wins (strict against
    #   "A за 6200 грн та B за 9500 грн" swaps);
    # - Otherwise trailing product mention in the SAME sentence ("8327 грн за A");
    # - Otherwise base-family name mentioned in that sentence ("AMD Ryzen 5 5600"
    #   across revisions);
    # - Otherwise summary sentence falls back to global_anchors (all products
    #   talked about on the page, matching docstring F25 above).
    _METRICS_REV_SUFFIX_RE = re.compile(r"\s*\(rev\s*\d+\)\s*$", re.IGNORECASE)
    family_map: Dict[str, List[str]] = {}
    for cname in dict.fromkeys(catalog_names):
        fam = _METRICS_REV_SUFFIX_RE.sub("", cname).strip()
        fk = _norm_key(fam)
        if fk:
            family_map.setdefault(fk, []).append(cname)

    sent_spans: List[Tuple[int, int]] = []
    pos = 0
    for seg in re.split(r"(?:\n+|(?<!\d)[.!?]+(?=\s|$))", response or ""):
        if seg:
            idx = (response or "").find(seg, pos)
            if idx != -1:
                sent_spans.append((idx, idx + len(seg)))
                pos = idx + len(seg)
    if not sent_spans and response:
        sent_spans = [(0, len(response))]

    # Keep longest non-overlapping product match at each character span so
    # a shorter family substring ("AMD Ryzen 5 7600X") never shadows an exact
    # revision mention ("AMD Ryzen 5 7600X (Rev 2)") at the same offset.
    mention_spans: List[Tuple[int, int, str]] = []
    for name in sorted(s.mentioned_products, key=len, reverse=True):
        for m in re.finditer(re.escape(name), response or "", re.IGNORECASE):
            st_m, en_m = m.start(), m.end()
            if any(st_m >= prev_st and en_m <= prev_en for prev_st, prev_en, _ in mention_spans):
                continue
            mention_spans.append((st_m, en_m, name))
    mention_positions: List[Tuple[int, str]] = sorted((st_m, nm) for st_m, _en_m, nm in mention_spans)
    query_nums = set(extract_prices(user_query)) if user_query else set()
    last_family_fk: Optional[str] = None
    for st, en in sent_spans:
        seg = (response or "")[st:en]
        sent_norm = _norm_key(seg)
        seg_norm_ws = normalize(seg)
        sent_fams = [
            fk for fk in family_map
            if len(fk) >= 6 and fk in sent_norm
        ]
        for m in _PRICE_RE.finditer(seg):
            abs_pos = st + m.start()
            said = parse_number(m.group(1))
            if said is None:
                continue
            if any(abs(said - qn) <= 0.51 for qn in query_nums):
                continue
            in_sent = [(p, n) for p, n in mention_positions if st <= p < en]
            same_prec = [n for p, n in in_sent if p < abs_pos]
            same_trail = [n for p, n in in_sent if p > abs_pos]
            # Per-price revision specifier: immediate trailing "(Rev N)" or nearest preceding "Rev N"
            trail_rev = re.match(r"^\s*\(?\s*rev\s*(\d+)\b", seg[m.end():], re.IGNORECASE)
            prec_revs = list(re.finditer(r"\brev\s*(\d+)\b", seg[:m.start()], re.IGNORECASE))
            rev_num = trail_rev.group(1) if trail_rev else (prec_revs[-1].group(1) if prec_revs else None)
            if same_prec:
                cand_name = same_prec[-1]
                anchors = anchors_for([cand_name])
                cand_fk = _norm_key(_METRICS_REV_SUFFIX_RE.sub("", cand_name))
                if (
                    rev_num is not None
                    and not _METRICS_REV_SUFFIX_RE.search(cand_name)
                    and cand_fk in family_map
                ):
                    rev_re = re.compile(rf"\(rev\s*{rev_num}\)\s*$", re.IGNORECASE)
                    rev_revs = [r for r in family_map[cand_fk] if rev_re.search(r)]
                    if rev_revs:
                        anchors = anchors_for(rev_revs)
            elif same_trail:
                anchors = anchors_for(same_trail[:1])
            else:
                if rev_num is not None and sent_fams:
                    rev_re = re.compile(rf"\(rev\s*{rev_num}\)\s*$", re.IGNORECASE)
                    fam_hits = [r for fk in sent_fams for r in family_map[fk] if rev_re.search(r)]
                else:
                    fam_hits = [rev for fk in sent_fams for rev in family_map[fk]]
                if not fam_hits and last_family_fk and last_family_fk in family_map:
                    if rev_num is not None:
                        rev_re = re.compile(rf"\(rev\s*{rev_num}\)\s*$", re.IGNORECASE)
                        fam_hits = [r for r in family_map[last_family_fk] if rev_re.search(r)]
                    elif re.search(
                        r"ревізі|цієї\s+(?:ж\s+)?модел|цієї\s+(?:материнської|відеокарти|пам'яті|плати)|"
                        r"варіант\w*\s+(?:цієї|даної)|їхн\w*\s+цін|"
                        r"інш\w*(?:\s+перевірен\w*)?\s+варіант\w*\s+кошту|ціни\s+в\s+базі\s+починаються",
                        seg_norm_ws,
                    ):
                        fam_hits = family_map[last_family_fk]
                anchors = anchors_for(fam_hits) if fam_hits else global_anchors
            if not any(abs(said - truth) <= 0.51 for truth in anchors):
                nearest = min((abs(said - t) for t in anchors), default=float("inf"))
                s.price_errors.append((said, round(nearest, 2)))
        if sent_fams:
            last_family_fk = max(sent_fams, key=lambda fk: (sent_norm.rfind(fk), len(fk)))
    s.prices_grounded = not s.price_errors

    s.raw_structured_verdict = structured_verdict
    s.verdict = extract_verdict(response, structured_verdict)
    s.verdict_applicable = oracle.checks_verdict
    s.verdict_correct = s.verdict == oracle.expected_verdict
    s.verdict_abstained = bool(
        s.verdict_applicable
        and oracle.expected_verdict != "neutral"
        and s.verdict == "neutral"
    )
    s.verdict_contradicted = bool(
        s.verdict_applicable and not s.verdict_correct and not s.verdict_abstained
    )

    if (
        s.ungrounded_entities
        and oracle.expected_verdict == "reject"
        and s.verdict == "reject"
        and user_query
    ):
        query_entities = set(extract_entities(user_query))
        sents = [
            seg.strip()
            for seg in re.split(r"(?:[!?]+|\.(?!\d))\s*|\n+", response or "")
            if seg.strip()
        ]
        def _is_negated_citation(seg: str, tok: str) -> bool:
            if extract_verdict(seg) == "reject":
                return True
            matched = False
            for m_ent in _ENTITY_RE.finditer(seg or ""):
                if re.sub(r"[\s_\-]+", "", normalize(m_ent.group(0))) != tok:
                    continue
                matched = True
                pre = normalize(seg[max(0, m_ent.start() - 48):m_ent.start()])
                if not re.search(
                    r"(?:^|\b)(?:а\s+не|замість|ніж|крім|без|відсутн\w*|немає)(?:\s+\S+){0,2}$",
                    pre,
                ):
                    return False
            return matched

        kept_ungrounded: List[str] = []
        for tok in s.ungrounded_entities:
            if tok not in query_entities:
                kept_ungrounded.append(tok)
                continue
            citing = [seg for seg in sents if tok in extract_entities(seg)]
            if citing and all(
                _is_negated_citation(seg, tok) and not extract_prices(seg)
                for seg in citing
            ):
                continue
            kept_ungrounded.append(tok)
        s.ungrounded_entities = kept_ungrounded

    # Informativeness is defined RELATIVE TO THE ORACLE, not absolutely.
    #
    # The first version required the answer to contain a product or a price.
    # That silently punished the most desirable behaviour in the whole
    # system: telling a customer that the product they asked about does not
    # exist. A `price_trap` case whose gold answer is an honest refusal
    # scored 0 informativeness, so the selftest could not reach 100%.
    #
    #   oracle demands products -> the answer must surface at least one
    #   oracle demands nothing   -> an honest, non-technical negative answer
    #                               IS the informative answer
    blocked_markers = (
        "помилка верифікації", "zero-hallucination gate",
        "відповідь заблоковано", "verification error",
    )
    s.blocked = any(m in norm_resp for m in blocked_markers)
    if s.blocked:
        s.informative = False
    elif expected:
        s.informative = s.mention_recall > 0
    else:
        s.informative = bool(norm_resp) and s.mention_precision >= 1.0

    # Collect explicit violations.
    if s.mention_precision < 1.0 or s.ungrounded_entities:
        s.violations.append("hallucinated_product")
    if not s.prices_grounded:
        s.violations.append("ungrounded_price")
    if (
        oracle.case_type == "direct_price"
        and oracle.expected_prices
        and not s.mentioned_prices
        and not s.blocked
    ):
        s.violations.append("missing_expected_price")
    if s.verdict_applicable and not s.verdict_correct:
        s.violations.append("wrong_verdict")
    if s.blocked:
        s.violations.append("blocked_no_content")
    return s


# --------------------------------------------------------------------------
# Aggregate metrics
# --------------------------------------------------------------------------


@dataclass
class Aggregate:
    n: int
    execution_accuracy_pct: float          # no violations AND verdict correct
    verdict_accuracy_pct: float
    hallucination_rate_pct: float           # % responses with >=1 false claim
    false_claim_rate_pct: float            # % of all product mentions that are false
    price_error_rate_pct: float
    nvfg_pct: float                         # Non-Vacuous Factual Guarantee
    informative_rate_pct: float
    blocked_rate_pct: float
    avg_steps: float
    avg_latency_s: float
    avg_cost_usd: float
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: How many cases verdict_accuracy_pct is computed over. Fewer than `n`
    #: means the remaining questions have no compatibility verdict at all,
    #: and averaging over them would dilute a meaningful number.
    verdict_cases: int = 0
    #: Decisions where the system abstained ("neutral", e.g. candidate unknown)
    #: instead of guessing on an accept/reject case, vs active contradictions.
    verdict_abstained_count: int = 0
    verdict_contradiction_count: int = 0


def aggregate(scores: Sequence[CaseScore]) -> Aggregate:
    """
    Aggregate per-case scores into the reported metric set.

    NVFG (Non-Vacuous Factual Guarantee):
        share of cases that are BOTH fully factual AND carry real content.
        This is the metric that exposes the vacuous-output failure mode of
        pure factuality filtering: a system that blocks everything scores
        factuality 100% and NVFG 0%.
    """
    n = len(scores)
    if n == 0:
        return Aggregate(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {})

    def pct(num: int, den: int) -> float:
        return round(100.0 * num / den, 2) if den else 0.0

    correct = sum(1 for s in scores if s.is_correct)
    # Verdict accuracy is reported over the cases that HAVE a verdict.
    # Averaging it over every case would dilute a meaningful number with
    # questions on which it means nothing.
    verdict_cases = [s for s in scores if s.verdict_applicable]
    verdict_ok = sum(1 for s in verdict_cases if s.verdict_correct)
    verdict_abstained = sum(1 for s in verdict_cases if s.verdict_abstained)
    verdict_contradicted = sum(1 for s in verdict_cases if s.verdict_contradicted)
    verdict_n = len(verdict_cases)
    hallucinated = sum(1 for s in scores if "hallucinated_product" in s.violations
                       or "ungrounded_price" in s.violations)
    total_mentions = sum(len(s.mentioned_products) for s in scores)
    false_mentions = sum(len(s.ungrounded_products) for s in scores)
    total_prices = sum(len(s.mentioned_prices) for s in scores)
    bad_prices = sum(len(s.price_errors) for s in scores)
    # F25: NVFG cannot be true while the answer hallucinates an entity or
    # gives the wrong verdict -- "useful AND factual" means both.
    nvfg = sum(1 for s in scores
               if s.informative
               and s.mention_precision >= 1.0
               and s.prices_grounded
               and not s.ungrounded_entities
               and (s.verdict_correct or not s.verdict_applicable))
    informative = sum(1 for s in scores if s.informative)
    blocked = sum(1 for s in scores if s.blocked)

    by_type: Dict[str, Dict[str, float]] = {}
    for case_type in sorted({s.case_type for s in scores}):
        subset = [s for s in scores if s.case_type == case_type]
        m = len(subset)
        with_verdict = [s for s in subset if s.verdict_applicable]
        by_type[case_type] = {
            "n": m,
            "execution_accuracy_pct": pct(sum(1 for s in subset if s.is_correct), m),
            "verdict_accuracy_pct": pct(sum(1 for s in with_verdict
                                            if s.verdict_correct), len(with_verdict)),
            "verdict_cases": len(with_verdict),
            "verdict_abstained_count": sum(1 for s in with_verdict if s.verdict_abstained),
            "verdict_contradiction_count": sum(1 for s in with_verdict if s.verdict_contradicted),
            "nvfg_pct": pct(sum(1 for s in subset
                                if s.informative and s.prices_grounded
                                and not s.ungrounded_entities
                                and (s.verdict_correct or not s.verdict_applicable)), m),
        }

    return Aggregate(
        n=n,
        execution_accuracy_pct=pct(correct, n),
        verdict_accuracy_pct=pct(verdict_ok, verdict_n),
        hallucination_rate_pct=pct(hallucinated, n),
        false_claim_rate_pct=pct(false_mentions, total_mentions),
        price_error_rate_pct=pct(bad_prices, total_prices),
        nvfg_pct=pct(nvfg, n),
        informative_rate_pct=pct(informative, n),
        blocked_rate_pct=pct(blocked, n),
        avg_steps=round(sum(s.steps for s in scores) / n, 2),
        avg_latency_s=round(sum(s.latency_s for s in scores) / n, 3),
        avg_cost_usd=round(sum(s.cost_usd for s in scores) / n, 6),
        by_type=by_type,
        verdict_cases=verdict_n,
        verdict_abstained_count=verdict_abstained,
        verdict_contradiction_count=verdict_contradicted,
    )
