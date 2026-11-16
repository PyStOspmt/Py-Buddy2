"""
Py-Buddy: Value Algebra (⊕) for multi-vendor attribute unification.

Motivation (docs/04_LITERATURE_REVIEW_AND_NOVELTY_GAP.md, Slot 3)
-----------------------------------------------------------------
AutoPKG (arXiv:2604.16950) unifies attribute *keys* across merchants.
Nobody unifies attribute *values* so that the result stays simultaneously
satisfiable for the constraint system. In the live Rozetka catalogue a single
logical attribute is spread over three languages and up to six key spellings,
and its values are written half a dozen ways:

    socket    : AM5 | sAM5 | AM4 | LGA1700 | sLGA1700 | Socket AM4
    pcd       : 5*112 | 5x112 | 5*112 мм | 5*120 | 5x114.3
    power     : 350W | 350 Вт | 750 Вт | 1000W
    diameter  : R19 | 19.0" | R17 | 17.0" | R21

This module supplies the missing algebra:

  * `canonical_form` — a projection from surface strings onto a canonical
    representative of their *shape*. Idempotent and order-free by
    construction.
  * `merge` (⊕) — the equivalence test on top of that projection, over the
    semilattice  None < canonical-strings < CONFLICT. CONFLICT is absorbing,
    which is precisely what makes the fold associative and
    permutation-invariant.
  * `discover_concepts` — which EAV keys carry the same logical attribute,
    inferred structurally (value signature + canonical value overlap +
    product-set overlap), never from a hand-written synonym list.

Modelling decisions that are *assumptions*, not facts, and are logged in
docs/07_RESEARCH_LOG.md:
  A1. A unit-less length written as `R<n>` denotes the nominal n-inch size.
      Wheel rims are specified in inches; the catalogue writes `R19` and
      `19.0"` for the same part. Without this the Value Algebra reports
      R19 and 19" as incommensurable.
  A2. Glued single-character vendor markers (`sAM5`, `sLGA1700`) and key-scoped
      label prefixes (`key="Socket"`, value `"Socket AM4"`) are stripped by
      structural projection (F50: only tokens appearing in the catalogue's
      attribute keys are treated as label prefixes; no baked-in lexicon).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
# Closed table of *units* — measurement, not domain concepts. The catalogue is
# multilingual so the table is too, but it is short and it is about units.

UNIT_ALIASES: Dict[str, str] = {
    "w": "W", "watt": "W", "watts": "W", "wattage": "W",
    "вт": "W", "ватт": "W", "ват": "W",
    "kw": "kW", "kilowatt": "kW", "kilowatts": "kW",
    "квт": "kW", "кіловат": "kW", "кіловатт": "kW",
    "mm": "mm", "мм": "mm", "міліметр": "mm", "міліметрів": "mm", "міліметри": "mm",
    "cm": "cm", "см": "cm", "сантиметр": "cm",
    "in": "in", "inch": "in", "inches": "in",
    "дюйм": "in", "дюймів": "in", "дюйма": "in", "дюйми": "in",
    "gb": "GB", "гб": "GB",
    "mb": "MB", "мб": "MB",
    "tb": "TB", "тб": "TB",
    "kg": "kg", "кг": "kg",
    "ghz": "GHz", "ггц": "GHz", "мгц": "MHz",
    # Mass, volume and energy. Added after the real-catalogue measurement in
    # docs/08_REAL_DATA_VALIDATION.md found that a value like `0.5 g` was being
    # canonicalised to `0.5` -- the trailing token was not in this table, so it
    # was dropped, and `0.5 g` then compared EQUAL to `0.5 mg`. Extending a
    # table of *units* is not the same act as writing a domain synonym list;
    # the algebra stays domain-neutral and this is where the difference shows.
    "g": "g", "г": "g", "gram": "g", "grams": "g", "gramme": "g",
    "mg": "mg", "мг": "mg", "milligram": "mg", "milligrams": "mg",
    "ug": "ug", "µg": "ug", "μg": "ug", "mcg": "ug", "мкг": "ug",
    "l": "l", "л": "l", "liter": "l", "litre": "l", "liters": "l", "litres": "l",
    "ml": "ml", "мл": "ml", "milliliter": "ml", "millilitre": "ml",
    "cl": "cl", "dl": "dl",
    "kcal": "kcal", "ккал": "kcal", "kilocalorie": "kcal",
    "kj": "kJ", "кдж": "kJ", "kilojoule": "kJ", "kilojoules": "kJ",
}

DIMENSION_OF: Dict[str, str] = {
    "W": "P", "kW": "P",
    "mm": "L", "cm": "L", "in": "L",
    "GB": "S", "MB": "S", "TB": "S",
    "kg": "M", "g": "M", "mg": "M", "ug": "M",
    "l": "V", "ml": "V", "cl": "V", "dl": "V",
    "kcal": "E", "kJ": "E",
    "GHz": "F", "MHz": "F",
}

# Conversion factor into the dimension's base unit.
UNIT_TO_BASE: Dict[str, float] = {
    "W": 1.0, "kW": 1000.0,
    "mm": 1.0, "cm": 10.0, "in": 25.4,
    "GB": 1.0, "MB": 0.001, "TB": 1000.0,
    "kg": 1.0, "g": 0.001, "mg": 0.000001, "ug": 0.000000001,
    "l": 1.0, "ml": 0.001, "cl": 0.01, "dl": 0.1,
    "kcal": 4.184, "kJ": 1.0,
    "GHz": 1000.0, "MHz": 1.0,
}

# Readable base symbol per dimension.
BASE_UNIT: Dict[str, str] = {
    "P": "W", "L": "mm", "S": "GB", "M": "kg", "V": "l", "E": "kJ", "F": "MHz",
}

# Assumption A1: the universal R<n> rim/tyre radius notation is the
# nominal n-inch size. It applies ONLY to values matching the R<n>
# marking itself; a bare number attached to a length-shaped value has no
# unit and gets none (F39).
RIM_RADIUS_UNIT_IN = "in"

# Brackets and quotes are decoration; dots, commas and asterisks are NOT,
# they carry numeric meaning ("5*112", "19.0", "1,5").
_BRACKET_RE = re.compile(r"[«»()\[\]{}·•№_`«»]")
# In this catalogue a bare quotation mark after a number is the inch mark.
_INCH_MARKS_RE = re.compile(r"[\"″'′]+")

_NUMERIC_HEAD = r"[\d\.,\*xх×rR]"
# `[^\\W\\d_]` is a Unicode letter that is not a digit — this is what lets
# "дюймів" (with its 'й') and "міліметрів" match, which an explicit
# [а-яіїєґ] class silently does not.
_TRAILING_UNIT_RE = re.compile(
    rf"^(?P<head>{_NUMERIC_HEAD}{{1,24}})\s*"
    rf"(?P<unit>[^\W\d_]{{1,16}})?$"
)


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def _fold(text: str) -> str:
    """
    Canonical comparison form.

    Three things this must NOT do, each of which was a real bug:

      * strip the inch mark — `19.0"` is inches, not a bare number;
      * de-accent Cyrillic — NFKD turns `дюймів` into `дюимів`, which then
        misses the unit table entirely;
      * delete decimal separators — `5*112` and `19.0` carry meaning.

    Latin is de-accented; Cyrillic is left intact.
    """
    s = _nfkc(text).lower().replace("\u00a0", " ")
    s = _INCH_MARKS_RE.sub(" in", s)
    s = s.replace("\u2019", "'").replace("\u02bc", "'")
    s = _BRACKET_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    decomposed = unicodedata.normalize("NFD", s)
    out: List[str] = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            # Drop the accent ONLY on a Latin base. Ukrainian 'й' is
            # и + combining breve and must survive, otherwise "дюймів"
            # de-composes to "дюимів" and misses the unit table.
            if out and _is_latin(out[-1]):
                continue
            out.append(ch)
        else:
            out.append(ch)
    # Recompose: a surviving combining mark is not a word character, so a
    # left-decomposed 'й' (U+0438 U+0306) would silently fail every
    # `[^\W\d_]` unit pattern.
    return unicodedata.normalize("NFC", "".join(out))


def _is_latin(ch: str) -> bool:
    o = ord(ch)
    return (0x0041 <= o <= 0x005A) or (0x0061 <= o <= 0x007A) or (0x00C0 <= o <= 0x024F)


def base_of(dimension: Optional[str], unit: Optional[str] = None) -> str:
    """
    Readable base symbol of a dimension -- total, unlike BASE_UNIT[...].

    A dimension the table has never heard of is a symbolic one (`u:oz`) and its
    own unit is the base. Callers must use this instead of indexing BASE_UNIT:
    doing so raised KeyError on `5 j` (joules) once unknown units stopped
    being silently dropped (docs/08 §4.3).
    """
    if dimension:
        return BASE_UNIT.get(dimension) or unit or dimension
    return unit or ""


def split_trailing_unit(folded: str) -> Tuple[str, str]:
    """
    Split a numeric surface form into (head, unit).

    The unit is looked up in the alias table; when it is NOT there the raw
    token is returned as-is. It used to be replaced by "", which destroyed the
    only trace that a unit was present: `5 oz` then became the bare number 5
    and compared EQUAL to `5 lb` (docs/08 §4.3). Keeping the token lets
    `dimension_for()` give it a dimension of its own, so an unrecognised unit
    is sound *without* being in the dictionary.

    Returns (folded, "") when the value is not numeric at all, which is how
    symbolic values such as `AM5` and `ATX` fall through to the TOKEN path.
    """
    m = _TRAILING_UNIT_RE.match(folded)
    if not m:
        return (folded, "")
    tail = m.group("unit") or ""
    return (m.group("head"), UNIT_ALIASES.get(tail, tail))


def dimension_for(unit: str) -> Optional[str]:
    """
    Dimension of a unit, known or not.

    This is the piece that keeps the system usable out of the box. Soundness
    needs no dictionary at all: a unit the table has never heard of becomes a
    dimension of its own (`5 oz` -> `u:oz`), so it can only ever be equal to a
    value carrying the very same unit. The table is therefore no longer load
    bearing for correctness -- it only decides how much *extra* coverage a
    catalogue gets (`500 ml` = `0.5 l`, which no per-unit dimension could
    know). Shrink the table to nothing and the algebra stays sound, only
    less useful; grow it with a learned alias and coverage grows. That is the
    difference between a dictionary and a cache, and it is what "adapts to any
    store without configuration" has to mean if it is to be true.
    """
    if not unit:
        return None
    return DIMENSION_OF.get(unit) or f"u:{unit}"


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


class Shape(Enum):
    TOKEN = "token"        # AM5, DDR5, ATX, LGA1700
    DUPLEX = "duplex"      # 5*112, 5x112, 5*112 мм -> bolts x diameter
    MEASURE = "measure"    # 350 W, 19 in, 16 GB
    RATIO = "ratio"        # 16:9
    FREE = "free"          # unrecognised: never silently merged


@dataclass(frozen=True)
class ValueSpec:
    """Structural signature of one surface value."""

    shape: Shape
    dimension: Optional[str] = None
    unit: Optional[str] = None
    numbers: Tuple[float, ...] = ()
    tokens: Tuple[str, ...] = ()

    @property
    def key(self) -> Tuple:
        """Signature used to *discover* which EAV keys mean the same thing."""
        return (self.shape, self.dimension, self.unit)


_ALPHA = r"A-Za-zА-Яа-яІіЇїЄєҐґ"
_UNIT_CHARS = r"[a-zа-яіїєґ]{0,16}"
_DUPLEX_RE = re.compile(rf"^(\d{{1,2}})([*xх×])(\d{{2,3}}(?:[,]\d+)?)({_UNIT_CHARS})$")
_RADIUS_RE = re.compile(rf"^r(\d{{1,3}}(?:[,]\d+)?)({_UNIT_CHARS})$")
_RATIO_RE = re.compile(r"^(\d{1,3})[:/](\d{1,3})$")
_MEASURE_RE = re.compile(rf"^(\d{{1,15}}(?:[,]\d+)?)({_UNIT_CHARS})$")
# Tokens may carry at most ONE internal space ("Socket LGA1700", "Micro ATX").
# Capping the word count and length is what stops an ordinary sentence
# ("пофарбуйте це синім") from being accepted as a symbolic value.
_TOKEN_RE = re.compile(rf"^([{_ALPHA}][{_ALPHA}0-9\-]{{0,15}}(?: [{_ALPHA}0-9\-]{{1,15}})?)$")

# F11/F50: the assumptions below are EXPLICIT, SCOPED declarations, not
# silent domain leakage. A1 treats the R<n> rim/tire radius notation (a
# universal marking on wheels and tyres) as the nominal n-inch size. The
# old A2 additionally carried a fixed PC lexicon of "decoration" prefixes
# (socket|сокет|гніздо|cpu|тип|type) that it stripped from ANY value --
# including values where the word is meaningful. Decoration is now
# resolved structurally: a leading word is stripped only when it is a
# token of the attribute KEY the value was stated under ("Socket AM4"
# under key "Socket" -> "am4"; the same value under a key that does not
# say "socket" keeps it verbatim). A value that carries neither a
# recognised format nor a unit stays a TOKEN or FREE and is never
# reinterpreted.
_VENDOR_MARKER_RE = re.compile(r"^s(?=[a-zа-яіїєґ])")


_NUM_RE = re.compile(r"^(\d{1,15}(?:[.,]\d+)?)$")
_RADIUS_NUM_RE = re.compile(r"^[rR](\d{1,3}(?:[.,]\d+)?)$")
_DUPLEX_NUM_RE = re.compile(r"^(\d{1,2})[*xх×](\d{2,3}(?:[.,]\d+)?)$")
_RATIO_NUM_RE = re.compile(r"^(\d{1,3})[:/](\d{1,3})$")


def _key_tokens(key: Optional[str]) -> FrozenSet:
    """Folded word tokens of an attribute key."""
    if not key:
        return frozenset()
    return frozenset(
        t for t in re.split(rf"[^{_ALPHA}0-9]+", _fold(str(key))) if t)


def _strip_key_prefix(folded_value: str, key: Optional[str]) -> str:
    """
    F50: strip leading words that restate the attribute's own key
    ("Сокет AM5" under key "Сокет" -> "am5"). The strip is keyed, not
    lexical: nothing but a verbatim token of the stated key is removed,
    so a cross-language decoration ("Socket AM4" under key "Сокет") and
    a meaningful leading word stay verbatim instead of being mangled by
    a hand-written PC dictionary.
    """
    keytoks = _key_tokens(key)
    if not keytoks:
        return folded_value
    words = folded_value.split(" ")
    while words and words[0] in keytoks:
        words.pop(0)
    return " ".join(words)


def spec_of(value: str, key: Optional[str] = None) -> ValueSpec:
    """
    Total, pure: any input yields a ValueSpec. Unrecognised input becomes
    Shape.FREE, which the algebra refuses to canonicalise or merge.
    `key` is the attribute key the value was stated under; it scopes the
    decoration strip to that key's own tokens.
    """
    folded = _fold(value)
    if not folded:
        return ValueSpec(Shape.FREE)

    head, unit_hint = split_trailing_unit(folded)

    m = _DUPLEX_NUM_RE.match(head)
    if m:
        unit = unit_hint or "mm"
        return ValueSpec(
            Shape.DUPLEX, dimension_for(unit), unit,
            (float(m.group(1)), float(m.group(2).replace(",", "."))),
        )

    m = _RATIO_NUM_RE.match(head)
    if m and not unit_hint:
        return ValueSpec(Shape.RATIO, None, None,
                         (float(m.group(1)), float(m.group(2))))

    m = _RADIUS_NUM_RE.match(head)
    if m and not unit_hint:
        # Assumption A1 (explicit, see the module note above): the universal
        # R<n> rim/tyre radius notation is the nominal n-inch size.
        return ValueSpec(Shape.MEASURE, "L", RIM_RADIUS_UNIT_IN,
                         (float(m.group(1).replace(",", ".")),))

    m = _NUM_RE.match(head)
    if m:
        num = float(m.group(1).replace(",", "."))
        if not unit_hint:
            return ValueSpec(Shape.MEASURE, None, None, (num,))
        return ValueSpec(Shape.MEASURE, dimension_for(unit_hint), unit_hint, (num,))

    if unit_hint:
        return ValueSpec(Shape.FREE)

    if _TOKEN_RE.match(folded):
        tok = _strip_key_prefix(folded, key).strip()
        if tok:
            return ValueSpec(Shape.TOKEN, tokens=(_strip_vendor_marker(tok),))
    return ValueSpec(Shape.FREE)


def _strip_vendor_marker(token: str) -> str:
    """
    `sAM5` -> `am5`; `sLGA1700` -> `lga1700` (assumption A2).

    F20: the strip is disambiguated -- a leading `s` is removed only when
    the remainder does not itself begin with `s` + a letter. `ssd` is a
    word, not `s` + `sd`; the old blanket strip made the projection
    non-idempotent (`ssd` -> `sd` -> `d`), so canonical_form was not a
    projection at all.

    F50: the marker applies to a single compact token only. A multi-word
    token's first word is a word ("socket am4"), not a glued vendor
    marker -- stripping its initial letter would produce "ocket am4".
    """
    if " " in token:
        return token.lower()
    t = token.lower()
    if t.startswith("s") and len(t) > 2 and t[1] == "s" and t[2].isalpha():
        return t
    out = _VENDOR_MARKER_RE.sub("", t)
    return out or t


# --------------------------------------------------------------------------
# Canonical projection
# --------------------------------------------------------------------------


def _fmt(x: float) -> str:
    """Format a magnitude so that the canonical form can be read back.

    Six significant digits, as `%g` did, but never in exponent notation: on
    real data `%g` wrote 5e-09 kg as `5e-09kg`, and neither `spec_of` nor
    `magnitude` can parse an exponent -- so the canonical form of a real
    nutrient value was not a fixed point of the projection.
    """
    if not math.isfinite(x):                                # NaN / inf
        return str(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    text = f"{x:.6g}"
    if "e" in text or "E" in text:
        text = format(Decimal(text), "f")
    return text


def canonical_form(value: str, key: Optional[str] = None) -> Optional[str]:
    """
    Project a surface value onto the canonical representative of its shape.

    Idempotent: canonical_form(canonical_form(x)) == canonical_form(x).
    `key` scopes decoration stripping to the stated attribute key (F50).
    """
    sp = spec_of(value, key=key)
    if sp.shape is Shape.FREE:
        return None
    if sp.shape is Shape.TOKEN:
        return sp.tokens[0]
    if sp.shape is Shape.DUPLEX:
        base = base_of(sp.dimension, sp.unit)
        f = UNIT_TO_BASE.get(sp.unit, 1.0)
        return f"{_fmt(sp.numbers[0])}x{_fmt(sp.numbers[1] * f)}{base}"
    if sp.shape is Shape.MEASURE:
        if not sp.dimension:
            return _fmt(sp.numbers[0])
        # An unknown unit is its own base: `5 oz` stays `5oz`.
        base = base_of(sp.dimension, sp.unit)
        f = UNIT_TO_BASE.get(sp.unit, 1.0)
        return f"{_fmt(sp.numbers[0] * f)}{base}"
    if sp.shape is Shape.RATIO:
        return f"{_fmt(sp.numbers[0])}:{_fmt(sp.numbers[1])}"
    return None


def magnitude(value: str, key: Optional[str] = None) -> Optional[Tuple[float, str]]:
    """
    Comparable magnitude as (number, dimension). Dimension '' for unit-less
    numbers, 'TOKEN' for symbolic values (magnitude 0).
    """
    cf = canonical_form(value, key=key)
    if cf is None:
        return None
    sp = spec_of(value, key=key)
    if sp.shape is Shape.TOKEN:
        return (0.0, "TOKEN")
    if not sp.dimension:
        return (sp.numbers[0], "")
    m = re.match(r"^([-+]?\d*\.?\d+)(.*)$", cf)
    if not m:
        return None
    return (float(m.group(1)), sp.dimension)


# --------------------------------------------------------------------------
# The merge operator
# --------------------------------------------------------------------------


class _Conflict:
    """Absorbing element of the merge semilattice."""

    _inst: Optional["_Conflict"] = None

    def __new__(cls) -> "_Conflict":
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "CONFLICT"


CONFLICT = _Conflict()
MergeResult = object   # str | None (identity) | _Conflict


def _project_or_raw(value: str, key: Optional[str] = None) -> str:
    """
    F20: an unparseable value is carried as the raw token -- visible, never
    silently dropped as if it were absent, and never promoted to a fact.
    The closed canonical domain and the raw surface are distinct states.
    """
    c = canonical_form(value, key=key)
    return c if c is not None else value


def merge(a: MergeResult, b: MergeResult, key: Optional[str] = None) -> MergeResult:
    """
    ⊕ on the semilattice  None < canonical-or-raw strings < CONFLICT.

    `None` is the identity: a product simply has no value for that key.
    `CONFLICT` is absorbing: once two values are incommensurable the result
    stays CONFLICT whatever is merged afterwards, in any order. That is what
    makes `fold_all` associative and permutation-invariant.

    F20: the result is canonical WHEN the value parses; an unparseable value
    is carried through as its raw self instead of vanishing. The old
    `merge(None, unparseable) -> None` destroyed information and broke
    associativity: one grouping of {unparseable, AM5} yielded "am5" while
    another yielded CONFLICT. Now an unparseable value is incommensurable
    with anything it does not literally equal, in every grouping.

    Note what is NOT done here: two *different* values of the same attribute
    (AM5 and AM4) are reported as a conflict, never as a disjunction. A
    disjunction would silently assert that either socket is acceptable, which
    is exactly the fabricated constraint the Value Algebra exists to prevent.
    """
    if a is CONFLICT or b is CONFLICT:
        return CONFLICT
    if a is None and b is None:
        return None
    # The identity case still projects to the canonical form, so that a fold
    # always yields canonical values: merge(None, "AM5") must agree with
    # merge("AM5", "sAM5") == "am5", not "AM5".
    if a is None:
        return _project_or_raw(str(b), key=key)
    if b is None:
        return _project_or_raw(str(a), key=key)

    if a == b:
        return _project_or_raw(str(a), key=key)
    ca, cb = canonical_form(str(a), key=key), canonical_form(str(b), key=key)
    if ca is not None and ca == cb:
        return ca
    return CONFLICT


def fold_all(values: Sequence[MergeResult], key: Optional[str] = None) -> MergeResult:
    """
    Do all these values denote the same thing?

    Returns the canonical form when they do, CONFLICT when they do not.
    Use it to validate one *specific* set — e.g. every socket attribute of
    the items in a candidate cart — never to summarise a whole catalogue
    column, whose value set is legitimately a vocabulary.
    """
    acc: MergeResult = None
    for v in values:
        acc = merge(acc, v, key=key)
    return acc


def equivalent(a: Optional[str], b: Optional[str],
               key: Optional[str] = None) -> bool:
    """True when both values are absent or denote the same thing."""
    return merge(a, b, key=key) is not CONFLICT


# --------------------------------------------------------------------------
# Concept discovery
# --------------------------------------------------------------------------


@dataclass
class Concept:
    """A discovered logical attribute shared by several EAV keys."""

    name: str
    shape: Shape
    dimension: Optional[str]
    unit: Optional[str]
    keys: List[str] = field(default_factory=list)
    vocabulary: List[str] = field(default_factory=list)

    def canonical(self, value: str) -> Optional[str]:
        return canonical_form(value)

    def agrees(self, values: Iterable[str]) -> bool:
        """Do all these values agree? (equivalence-class coherence test)

        The concept's own key surface forms provide the strip context, so
        "Socket AM5" agrees with "AM5" when one of the grouped keys is
        literally "socket" -- and stays verbatim when none is.
        """
        return fold_all(list(values), key=" ".join(self.keys)) is not CONFLICT


def _jaccard(a: FrozenSet, b: FrozenSet) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def discover_concepts(
    key_values: Dict[str, Iterable[str]],
    key_scopes: Optional[Dict[str, Set[str]]] = None,
    *,
    min_keys: int = 2,
    min_value_overlap: float = 0.25,
    min_scope_overlap: float = 0.34,
    allow_merge: bool = True,
) -> List[Concept]:
    """
    Group EAV keys that carry the same logical attribute.

    Three structural criteria, all necessary:

      (1) values share a signature (shape, dimension, unit);
      (2) *canonical* value sets overlap — raw spellings differ wildly
          ("Socket AM5" vs "AM5"), so overlap must be measured after ⊕;
      (3) scopes overlap — the set of entities the key is recorded on, here
          the product category. This separates keys that merely draw on the
          same numeric vocabulary: `Максимальний допуск` (a limit, only on
          wheel hubs) and `Діаметр` (the item's own size, only on rims) share
          a unit but no scope, and merging them would turn a containment
          rule into nonsense.

    Category scope rather than raw product set: two socket keys touch
    different CPUs yet both span {cpu, motherboard}, while a wheel-hub limit
    and a rim diameter share none. The coarser signal separates the concepts
    cleanly and does not punish sparse coverage.

  `min_keys=1` degrades to one concept per key — the honest fallback for a
  catalogue with no redundancy. `allow_merge=False` disables the pairwise
  step entirely, which is what the caller uses to give every still-unassigned
  key its own concept instead of leaving it unanalysable.
  """
    key_values = {k: sorted({v for v in vs if v}) for k, vs in key_values.items()}
    key_scopes = key_scopes or {}

    # F50: the decoration vocabulary is the source schema's own key surface
    # forms. A vendor who writes "Socket LGA1700" under key "гніздо CPU"
    # restates a key name that literally exists elsewhere in the schema --
    # stripping it is justified by the corpus, not by a baked-in lexicon.
    all_keys_ctx = " ".join(key_values)

    canon: Dict[str, Set[str]] = {}
    sig: Dict[str, Tuple] = {}
    for key, values in key_values.items():
        got: Set[str] = set()
        first: Optional[Tuple] = None
        for v in values:
            sp = spec_of(v, key=all_keys_ctx)
            if sp.shape is Shape.FREE:
                continue
            cf = canonical_form(v, key=all_keys_ctx)
            if cf is None:
                continue
            got.add(cf)
            if first is None:
                first = sp.key
        if first is not None:
            sig[key] = first
            canon[key] = got

    groups: Dict[Tuple, List[str]] = {}
    for key, k in sig.items():
        groups.setdefault(k, []).append(key)

    concepts: List[Concept] = []
    n = 0
    for key, keys in sorted(groups.items(), key=lambda kv: str(kv[0])):
        keys = sorted(keys)
        if len(keys) < min_keys:
            continue
        parent = {k: k for k in keys}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(keys) if allow_merge else ():
            for b in (keys[i + 1:] if allow_merge else ()):
                if _jaccard(frozenset(canon[a]), frozenset(canon[b])) < min_value_overlap:
                    continue
                if key_scopes and _jaccard(
                    frozenset(key_scopes.get(a, ())),
                    frozenset(key_scopes.get(b, ())),
                ) < min_scope_overlap:
                    continue
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

        buckets: Dict[str, List[str]] = {}
        for k in keys:
            buckets.setdefault(find(k), []).append(k)

        for gkeys in buckets.values():
            if len(gkeys) < min_keys:
                continue
            n += 1
            concepts.append(Concept(
                name=f"c{n:02d}_{key[0].value}_{key[1] or 'x'}",
                shape=key[0], dimension=key[1], unit=key[2],
                keys=sorted(gkeys),
                vocabulary=sorted({v for g in gkeys for v in key_values[g]}),
            ))
    return sorted(concepts, key=lambda c: (-len(c.keys), c.name))
