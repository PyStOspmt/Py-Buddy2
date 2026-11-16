"""
Tests for src/attribute_algebra.py — the Value Algebra (⊕).

These tests pin the algebra *laws* (idempotence, associativity,
commutativity, absorbing conflict) and the structural discovery that lets a
new vendor language work without a single hand-written synonym.
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from attribute_algebra import (
    CONFLICT,
    Shape,
    base_of,
    canonical_form,
    dimension_for,
    discover_concepts,
    equivalent,
    fold_all,
    magnitude,
    merge,
    spec_of,
)

# --------------------------------------------------------------------------
# Canonical projection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("surface,expected", [
    ("350W", "350W"),
    ("350 Вт", "350W"),
    ("750 Вт", "750W"),
    ("105W", "105W"),
    ("1000GB", "1000GB"),
    ("5*112", "5x112mm"),
    ("5x112", "5x112mm"),
    ("5*112 мм", "5x112mm"),
    ("5*114.3", "5x114.3mm"),
    ("R19", "482.6mm"),          # assumption A1: R<n> is the nominal n-inch size
    ('19.0"', "482.6mm"),
    ("19.0 дюймів", "482.6mm"),
    ("R17", "431.8mm"),
    ('17.0"', "431.8mm"),
    ("AM5", "am5"),
    ("sAM5", "am5"),             # assumption A2: vendor marker is decoration
    # F50: without a key context, a decoration word is part of the value --
    # verbatim, not mangled by a hand-written prefix lexicon.
    ("Socket AM4", "socket am4"),
    ("Гніздо AM5", "гніздо am5"),
    ("sLGA1700", "lga1700"),
    ("LGA1851", "lga1851"),
    ("DDR5", "ddr5"),
    ("ATX", "atx"),
    ("Micro-ATX", "micro-atx"),
])
def test_canonical_form(surface, expected):
    assert canonical_form(surface) == expected


def test_key_scoped_decoration_strip():
    """F50: only a verbatim token of the stated key is stripped."""
    assert canonical_form("Socket AM4", key="Socket") == "am4"
    assert canonical_form("Сокет AM5", key="Сокет") == "am5"
    # A cross-language key does NOT strip -- "socket" is not a token of
    # "Сокет", and no translation dictionary is smuggled in to pretend so.
    assert canonical_form("Socket AM4", key="Сокет") == "socket am4"
    # A meaningful leading word that happens to match the old lexicon is
    # no longer mangled when the key never said it.
    assert canonical_form("тип А", key="Категорія") == "тип а"
    assert canonical_form("тип А", key="Тип") == "а"


def test_canonical_form_rejects_free_text():
    assert canonical_form("") is None
    assert canonical_form("   ") is None
    assert canonical_form("пофарбуйте це синім") is None


@pytest.mark.parametrize("surface,expected", [
    # Mass, volume and energy. These came out of the real-catalogue measurement
    # in docs/08_REAL_DATA_VALIDATION.md, which caught the algebra dropping a
    # trailing unit it did not know.
    ("0.5 g", "0.0005kg"),
    ("500 mg", "0.0005kg"),
    ("350 g", "0.35kg"),
    ("0.35 kg", "0.35kg"),
    ("5 μg", "0.000000005kg"),
    ("1 l", "1l"),
    ("1000 ml", "1l"),
    ("0.5 л", "0.5l"),
    ("500 мл", "0.5l"),
    ("500 kcal", "2092kJ"),
])
def test_units_of_real_catalogues(surface, expected):
    assert canonical_form(surface) == expected


def test_units_are_never_merged_across_magnitudes():
    """A 1000x error the real data exposed, kept out for good.

    Before the unit table carried `g`, the trailing token was dropped, so
    `0.5 g` and `0.5 mg` both canonicalised to `0.5` and folded to equality.
    """
    assert canonical_form("0.5 g") != canonical_form("0.5 mg")
    assert canonical_form("1 l") != canonical_form("1 ml")
    assert fold_all(["0.5 g", "0.5 mg"]) is CONFLICT
    assert fold_all(["1 l", "1 ml"]) is CONFLICT
    # ...and the equalities that SHOULD hold now do, across unit spellings.
    assert equivalent(canonical_form("500 mg"), canonical_form("0.5 g"))
    assert equivalent(canonical_form("1000 ml"), canonical_form("1 l"))
    assert equivalent(canonical_form("500 мл"), canonical_form("0.5 л"))


def test_an_unknown_unit_is_a_dimension_of_its_own():
    """Soundness without a dictionary: `5 oz` may not become `5`.

    The trailing token used to be replaced by "" whenever the unit table did
    not know it, so every unknown unit collapsed onto the bare number and
    `5 oz` compared EQUAL to `5 lb` -- a 10x error with no conflict raised
    (docs/08 §4.3). Fixing it by returning a conflict would have been sound but
    blind; instead the token becomes its own dimension, which keeps values
    comparable within the unit and incomparable across units.
    """
    assert fold_all(["5 oz", "5 lb"]) is CONFLICT
    assert fold_all(["1 cup", "5 oz"]) is CONFLICT
    assert canonical_form("5 oz") == "5oz"
    assert canonical_form("5 lb") == "5lb"
    # The same unknown unit still merges -- coverage is not the price of safety.
    assert fold_all(["5 oz", "5 oz"]) == "5oz"
    assert fold_all(["2 cup", "2 cup"]) == "2cup"
    # Two spellings of one unknown unit do not merge either, and that is the
    # honest answer: nothing but a learned alias can relate `oz` to `ounce`.
    assert fold_all(["2 шт", "2 ст"]) is CONFLICT


def test_dimension_lookup_is_total_and_stable_under_emptying_the_table():
    """The unit table is a cache, not a load-bearing assumption.

    `dimension_for` answers for any string; a symbolic dimension keeps the
    unit as its own base. If the table were emptied the algebra would lose
    coverage (`500 ml` = `0.5 l` would stop holding) but every value would
    still be measured in a distinct dimension, i.e. still sound.
    """
    assert dimension_for("g") == "M"
    assert dimension_for("oz") == "u:oz"
    assert dimension_for("") is None
    assert base_of("M", "g") == "kg"
    assert base_of("u:oz", "oz") == "oz"        # no KeyError on a symbolic one
    assert base_of(None, "oz") == "oz"


def test_shape_detection():
    assert spec_of("5*112").shape is Shape.DUPLEX
    assert spec_of("AM5").shape is Shape.TOKEN
    assert spec_of("350 Вт").shape is Shape.MEASURE
    assert spec_of("16:9").shape is Shape.RATIO
    assert spec_of("???").shape is Shape.FREE


# --------------------------------------------------------------------------
# The merge operator
# --------------------------------------------------------------------------

def test_merge_equivalent_surface_forms():
    assert merge("5*112", "5x112") == "5x112mm"
    assert merge("350W", "350 Вт") == "350W"
    assert merge("R19", '19.0"') == "482.6mm"
    assert merge("AM5", "sAM5") == "am5"


def test_merge_distinct_values_is_conflict_not_disjunction():
    """Different sockets must never be reported as 'either is fine'."""
    assert merge("AM5", "AM4") is CONFLICT
    assert merge("DDR5", "DDR4") is CONFLICT
    assert merge("5*112", "5x120") is CONFLICT


def test_merge_incommensurable_dimensions_is_conflict():
    assert merge("350W", "19.0 дюймів") is CONFLICT
    assert merge("1000GB", "R19") is CONFLICT


def test_kcal_kj_rounding_difference_is_a_conflict_not_a_merge():
    """D4 (07_RESEARCH_LOG) / Q1 (08_REAL_DATA_VALIDATION), pinned.

    539 kcal is exactly 2255.176 kJ; the six-significant-digit canonical
    form writes it as 2255.18 kJ. A separately-rounded 2255 kJ is a
    different string, so the algebra reports a conflict, not a merge. This is
    a consequence of the exact (never approximate) projection: the algebra
    cannot recover the original precision from a rounded value, and merging
    two different magnitudes would be the fabricated constraint the Value
    Algebra exists to prevent. Pinned so the behaviour is not accidentally
    changed; the limitation is documented, not papered over with a tolerance.
    """
    assert canonical_form("539 kcal") == "2255.18kJ"
    assert canonical_form("2255 kJ") == "2255kJ"
    assert merge("539 kcal", "2255 kJ") is CONFLICT
    # The exact converted value still merges across unit spellings.
    assert equivalent(canonical_form("539 kcal"), canonical_form("2255.18 kJ"))


def test_merge_none_is_identity():
    assert merge(None, "AM5") == "am5"
    assert merge("AM5", None) == "am5"
    assert merge(None, None) is None


def test_merge_is_a_fixed_point():
    """
    ⊕ projects onto the canonical representative, so `merge(v, v)` is the
    canonical form and feeding that back in changes nothing.
    """
    for v in ("AM5", "350 Вт", "5*112", "R19", "Socket LGA1700"):
        once = merge(v, v)
        assert once == canonical_form(v)
        assert merge(once, once) == once


def test_conflict_is_absorbing():
    assert merge(CONFLICT, "AM5") is CONFLICT
    assert merge("AM5", CONFLICT) is CONFLICT
    assert merge(CONFLICT, CONFLICT) is CONFLICT
    assert merge(merge("AM5", "AM4"), "DDR5") is CONFLICT
    assert merge("AM5", merge("AM4", "DDR5")) is CONFLICT


@pytest.mark.parametrize("triple", [
    ("AM5", "sAM5", "Socket AM5"),
    ("350W", "350 Вт", "750 Вт"),
    ("5*112", "5x112", "5*112 мм"),
    ("R19", '19.0"', "19.0 дюймів"),
    ("AM5", "AM4", "LGA1700"),
])
def test_merge_is_associative(triple):
    a, b, c = triple
    assert merge(merge(a, b), c) == merge(a, merge(b, c))


def test_merge_is_commutative():
    assert merge("5*112", "5x112") == merge("5x112", "5*112")
    assert merge("AM5", "AM4") == merge("AM4", "AM5")


def test_fold_is_permutation_invariant():
    values = ["AM5", "sAM5", "Socket AM5", "LGA1700"]
    base = fold_all(values)
    rnd = random.Random(3)
    for _ in range(50):
        shuffled = values[:]
        rnd.shuffle(shuffled)
        assert fold_all(shuffled) == base


def test_equivalent_helper():
    assert equivalent("AM5", "sAM5")
    assert not equivalent("AM5", "AM4")
    assert equivalent(None, None)          # absent and absent agree


# --------------------------------------------------------------------------
# Magnitudes
# --------------------------------------------------------------------------

def test_magnitude_is_dimension_aware():
    assert magnitude("350 Вт") == (350.0, "P")
    assert magnitude("1000GB") == (1000.0, "S")
    assert magnitude("AM5") == (0.0, "TOKEN")


def test_magnitude_converts_length_to_base_unit():
    assert magnitude('19.0"') == (482.6, "L")
    assert magnitude("R19") == (482.6, "L")


# --------------------------------------------------------------------------
# Concept discovery
# --------------------------------------------------------------------------

def _fixture():
    """A miniature multilingual catalogue in the shape of the real one."""
    key_values = {
        "Сокет":       ["AM5", "LGA1700", "Socket AM4"],
        "socket":      ["AM5", "LGA1700", "sAM5"],
        "гніздо CPU":  ["AM5", "Socket LGA1700", "AM4"],
        "PCD":         ["5*112", "5x120"],
        "Болти":       ["5*112", "5x120"],
        "Розболтовка": ["5*112", "5x120"],
        "Діаметр":     ['R19', '17.0"'],
        "Розмір":      ['R19', '17.0"'],
        "Максимальний допуск": ["19.0 дюймів"],   # limit, not a size
        "Рекомендований БЖ":  ["850W"],
    }
    key_scopes = {
        "Сокет": {"cpu", "motherboard"}, "socket": {"cpu", "motherboard"},
        "гніздо CPU": {"cpu", "motherboard"},
        "PCD": {"rim"}, "Болти": {"rim"}, "Розболтовка": {"rim", "wheel_hub"},
        "Діаметр": {"rim"}, "Розмір": {"rim"},
        "Максимальний допуск": {"wheel_hub"},
        "Рекомендований БЖ": {"psu", "gpu"},
    }
    return key_values, key_scopes


def test_discovery_groups_the_three_socket_spellings():
    kv, ks = _fixture()
    concepts = discover_concepts(kv, ks, min_keys=2)
    by_keys = {tuple(sorted(c.keys)): c for c in concepts}
    socket = next(c for c in concepts if "Сокет" in c.keys)
    assert set(socket.keys) == {"Сокет", "socket", "гніздо CPU"}
    assert socket.shape is Shape.TOKEN


def test_discovery_groups_all_three_pcd_spellings():
    kv, ks = _fixture()
    concepts = discover_concepts(kv, ks, min_keys=2)
    pcd = next(c for c in concepts if "PCD" in c.keys)
    assert set(pcd.keys) == {"PCD", "Болти", "Розболтовка"}
    assert pcd.shape is Shape.DUPLEX


def test_discovery_separates_size_from_limit_by_scope():
    """`Максимальний допуск` and `Діаметр` share a unit but not a scope."""
    kv, ks = _fixture()
    concepts = discover_concepts(kv, ks, min_keys=2)
    diameter = next(c for c in concepts if "Діаметр" in c.keys)
    assert "Максимальний допуск" not in diameter.keys
    assert not any("Максимальний допуск" in c.keys for c in concepts)


def test_discovery_does_not_merge_across_dimensions():
    kv, ks = _fixture()
    concepts = discover_concepts(kv, ks, min_keys=2)
    for c in concepts:
        assert len({spec_of(v).key for v in c.vocabulary
                    if spec_of(v).shape is not Shape.FREE}) <= 1


def test_discovery_keeps_single_keys_when_min_keys_is_one():
    kv, ks = _fixture()
    concepts = discover_concepts(kv, ks, min_keys=1)
    all_keys = {k for c in concepts for k in c.keys}
    assert set(kv) <= all_keys


def test_concept_agrees_validates_one_specific_set():
    kv, ks = _fixture()
    concepts = discover_concepts(kv, ks, min_keys=2)
    socket = next(c for c in concepts if "Сокет" in c.keys)
    assert socket.agrees(["AM5", "sAM5", "Socket AM5"])
    assert not socket.agrees(["AM5", "AM4"])


def test_discovery_on_empty_input():
    assert discover_concepts({}, {}, min_keys=2) == []
