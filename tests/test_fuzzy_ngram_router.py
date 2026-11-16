"""
Tests for the morphological router.

The point of renaming the module was to make a falsifiable claim about
what the code does. These tests are that claim, including the part that
is a limitation: the router generalises over INFLECTION and not over
meaning, and pretending otherwise in a test suite is how the old docstring
happened in the first place.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fuzzy_ngram_router import (
    DEFAULT_THRESHOLD,
    FuzzyNgramRouter,
    LocalMemoryVectorStore,
    NgramVectorizer,
    SemanticIntentRouter,
    VectorEncoder,
)


@pytest.fixture
def router():
    return FuzzyNgramRouter()


def sim(a, b):
    return NgramVectorizer.cosine_similarity(
        NgramVectorizer().encode(a), NgramVectorizer().encode(b))


# --------------------------------------------------------------------------
# What it is good at
# --------------------------------------------------------------------------

def test_an_inflected_rewording_matches_the_original():
    """"диск" -> "диски" -> "диска": stems, endings, case. All matched."""
    assert sim("найдешевший диск", "найдешевші диски") >= 0.6
    assert sim("найдешевший диск", "Найдешевшого диска") >= 0.6


def test_the_router_reuses_a_verified_query(router):
    match = router.route_query("Який найдешевший товар у магазині")
    assert match is not None
    intent, sql, score = match
    assert intent == "EXTREMA_PRICE_MIN_ALL"
    assert "ORDER BY price ASC" in sql
    assert score >= DEFAULT_THRESHOLD


def test_learned_queries_are_reused(router):
    router.learn_from_feedback("скинути на Radeon 4070", "GPU_4070", "SELECT 1;")
    intent, sql, _ = router.route_query("скинути на Radeon 4070 Ti Super")
    assert intent == "GPU_4070"
    assert sql == "SELECT 1;"


def test_a_bigger_vocabulary_costs_memory_not_accuracy_of_shape():
    small = NgramVectorizer(dimensions=64)
    large = NgramVectorizer(dimensions=4096)
    a = "найдешевший процесор"
    assert small.encode(a) and large.encode(a)
    assert len(large.encode(a)) == 4096


def test_a_useless_dimension_count_is_refused():
    with pytest.raises(ValueError):
        NgramVectorizer(dimensions=4)


# --------------------------------------------------------------------------
# What it cannot do -- asserted, so the limitation stays visible
# --------------------------------------------------------------------------

def test_a_synonym_is_invisible_to_it():
    """The headline limitation, as an executable claim.

    A real encoder puts "ціна" and "вартість" near each other. This one
    cannot: it has never been told that either word exists, let alone that
    they mean the same thing. Any deployment that assumed otherwise was
    assuming a model that was never here.
    """
    assert sim("яка ціна на процесор", "скільки коштує чіп") < 0.5


def test_a_completely_unrelated_question_returns_none(router):
    assert router.route_query("який смак у вашого сиру") is None


def test_a_cache_that_never_misses_is_not_a_cache(router):
    for query in ("привіт", "до побачення", "яка погода"):
        router.route_query(query)


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------

def test_vectors_are_l2_normalised():
    vec = NgramVectorizer().encode("найдешевший процесор AMD Ryzen")
    assert sum(x * x for x in vec) == pytest.approx(1.0)


def test_an_empty_query_encodes_to_zero_and_matches_nothing():
    vec = NgramVectorizer().encode("   ")
    assert not any(vec)
    assert NgramVectorizer.cosine_similarity(vec, vec) == 0.0


def test_cosine_of_mismatched_vectors_is_zero():
    assert NgramVectorizer.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_expiry_actually_expires():
    store = LocalMemoryVectorStore()
    store.register_intent("питання", [0.0] * 256, "I", "SELECT 1;", ttl_seconds=0.0)
    assert store.cleanup_expired() == 1
    assert not store.entries


def test_the_store_evicts_least_recently_used_when_full():
    store = LocalMemoryVectorStore(max_size=2)
    for i in range(3):
        store.register_intent(f"q{i}", [0.0] * 256, "I", "SELECT 1;")
    assert set(store.entries) == {"q1", "q2"}


def test_prototypes_outlive_learned_queries():
    store = LocalMemoryVectorStore()
    router = FuzzyNgramRouter(store=store)
    ttls = {e.ttl_seconds for e in store.entries.values()}
    assert ttls == {router.PROTOTYPE_TTL}
    assert router.PROTOTYPE_TTL > 86400.0


def test_determinism():
    a, b = NgramVectorizer(), NgramVectorizer()
    text = "найдешевший процесор AMD Ryzen 5 5600X"
    assert a.encode(text) == b.encode(text)


# --------------------------------------------------------------------------
# The old names
# --------------------------------------------------------------------------

def test_the_old_names_still_resolve():
    assert VectorEncoder is NgramVectorizer
    assert SemanticIntentRouter is FuzzyNgramRouter
    assert SemanticIntentRouter().encoder.dimensions == 256


def test_router_adapts_prototypes_to_non_products_schema(tmp_path):
    """FuzzyNgramRouter builds SQL over discovered tables and skips price extrema when absent."""
    import sqlite3

    db_priced = tmp_path / "custom_priced.sqlite"
    con = sqlite3.connect(str(db_priced))
    con.execute("CREATE TABLE catalog_items (item_uid TEXT PRIMARY KEY, title TEXT, unit_price REAL)")
    con.execute("INSERT INTO catalog_items VALUES ('i1', 'Alpha Widget', 450.0)")
    con.execute("INSERT INTO catalog_items VALUES ('i2', 'Beta Widget', 900.0)")
    con.execute("INSERT INTO catalog_items VALUES ('i3', 'Gamma Widget', 1350.0)")
    con.commit()
    con.close()

    router_priced = FuzzyNgramRouter(db_path=str(db_priced))
    assert any('"catalog_items"' in e.target_sql for e in router_priced.store.entries.values())
    assert not any("FROM products" in e.target_sql for e in router_priced.store.entries.values())
    hit = router_priced.route_query("Покажи найдорожчу позицію в каталозі")
    assert hit is not None and '"unit_price" DESC' in hit[1]

    db_noprice = tmp_path / "custom_noprice.sqlite"
    con2 = sqlite3.connect(str(db_noprice))
    con2.execute("CREATE TABLE catalog_docs (doc_id TEXT PRIMARY KEY, heading TEXT, section TEXT)")
    con2.execute("INSERT INTO catalog_docs VALUES ('d1', 'Manual A', 'Guide')")
    con2.commit()
    con2.close()

    router_noprice = FuzzyNgramRouter(db_path=str(db_noprice))
    assert not any("ORDER BY" in e.target_sql for e in router_noprice.store.entries.values())
    assert router_noprice.route_query("найдешевший товар") is None
