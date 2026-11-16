"""
Tests for src/benchmark_cases.py — database-derived ground truth.

These tests guard the property the whole honest-evaluation effort depends on:
an oracle must be computable, deterministic, and consistent with the
deterministic verifier. A benchmark whose expected answers are wrong is worse
than no benchmark.
"""
import collections
import os
import sqlite3
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from attribute_algebra import canonical_form
from benchmark_cases import (
    CATEGORY_UA,
    build_cases,
    check_bundle,
    diameter_mm,
    load_snapshot,
    max_diameter_mm,
    pcd_of,
    power_watts,
    sockets_of,
)

LIVE_DB = os.path.join(ROOT, "data", "marketplace_rozetka.db")
has_live = os.path.exists(LIVE_DB)
needs_live = pytest.mark.skipif(not has_live, reason="run data/generate_marketplace_scale.py first")


# --------------------------------------------------------------------------
# A miniature catalog so the tests run without the 4.8 MB generated database
# --------------------------------------------------------------------------

def _mini_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE products (
            id TEXT PRIMARY KEY, name TEXT, category TEXT, brand TEXT,
            price REAL, merchant_id TEXT, stock_quantity INTEGER,
            performance_score REAL);
        CREATE TABLE product_eav_meta (
            meta_id INTEGER PRIMARY KEY AUTOINCREMENT, product_id TEXT,
            meta_key TEXT, meta_value TEXT);
    """)
    # Two CPUs on AM5, two on LGA1700; one board per socket; rim + hub pair.
    rows = [
        ("c1", "AMD Ryzen 5 7600X", "cpu", "AMD", 9500.0, 0.5, 3, 82.0),
        ("c2", "AMD Ryzen 7 7800X3D", "cpu", "AMD", 16500.0, 0.5, 2, 96.0),
        ("c3", "Intel Core i5-14600K", "cpu", "Intel", 11800.0, 0.5, 4, 85.0),
        ("c4", "Intel Core i7-14700K", "cpu", "Intel", 17200.0, 0.5, 1, 94.0),
        ("m1", "MSI B650 Gaming Plus", "motherboard", "MSI", 6800.0, 0.5, 6, 85.0),
        ("m2", "ASUS TUF X670E-Plus", "motherboard", "ASUS", 13200.0, 0.5, 3, 95.0),
        ("m3", "ASUS Prime Z790-P", "motherboard", "ASUS", 8400.0, 0.5, 5, 88.0),
        ("m4", "Gigabyte Z790 Aorus", "motherboard", "Gigabyte", 12500.0, 0.5, 2, 94.0),
        ("r1", "Диск BBS 18", "rim", "BBS", 4300.0, 0.5, 7, 80.0),
        ("r2", "Диск Mansory 21", "rim", "Mansory", 9800.0, 0.5, 1, 92.0),
        ("h1", "Ступиця Golf 19", "wheel_hub", "TRW", 5100.0, 0.5, 3, 79.0),
        ("p1", "DeepCool PF500 500W", "psu", "DeepCool", 1700.0, 0.5, 20, 60.0),
    ]
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", rows)
    eav = [
        ("c1", "Сокет", "Socket AM5"), ("c1", "TDP", "105 Вт"),
        ("c2", "socket", "sAM5"), ("c2", "Тепловиділення", "120W"),
        ("c3", "гніздо CPU", "Socket LGA1700"), ("c3", "tdp_watts", "181W"),
        ("c4", "Роз'єм процесора", "LGA1700"), ("c4", "Потужність", "253 Вт"),
        ("m1", "Сокет", "AM5"), ("m2", "гніздо", "AM5"),
        ("m3", "Socket Type", "Socket LGA1700"), ("m4", "socket", "LGA1700"),
        ("r1", "PCD", "5x112"), ("r1", "Діаметр", "R18"),
        ("r2", "Розболтовка", "5*114.3"), ("r2", "Розмір", "R21"),
        ("h1", "Максимальний допуск", "19.0 дюймів"),
        ("h1", "Болти", "5*112"),
        ("p1", "Wattage", "500W"),
    ]
    conn.executemany("INSERT INTO product_eav_meta (product_id, meta_key, meta_value) "
                     "VALUES (?,?,?)", eav)
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="module")
def mini():
    path = _mini_db()
    snap = load_snapshot(path)
    yield snap
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)


# --------------------------------------------------------------------------
# Snapshot + concept discovery
# --------------------------------------------------------------------------

def test_snapshot_loads_and_indexes(mini):
    assert len(mini.products) == 12
    # price_index maps a NAME to every price any seller lists it at, not to
    # one price: in a marketplace the same name is sold by several vendors.
    assert 9500.0 in mini.price_index["AMD Ryzen 5 7600X"]
    assert mini.id_to_name["c1"] == "AMD Ryzen 5 7600X"


def test_discovery_unifies_socket_spellings(mini):
    sockets = mini.canonical_attrs("c1")
    # Socket AM5, TDP 105W -> one socket concept, one power concept
    assert "am5" in sockets.values()


def test_canonical_attrs_are_memoised(mini):
    a = mini.canonical_attrs("c1")
    b = mini.canonical_attrs("c1")
    assert a is b


# --------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------

def test_sockets_of_resolves_three_languages(mini):
    assert sockets_of(mini, "c1") == "am5"     # "Socket AM5"
    assert sockets_of(mini, "c2") == "am5"     # "sAM5"
    assert sockets_of(mini, "c3") == "lga1700"  # "Socket LGA1700"
    assert sockets_of(mini, "c4") == "lga1700"  # "LGA1700"


def test_power_watts_unifies_units(mini):
    assert power_watts(mini, "c1") == 105.0
    assert power_watts(mini, "c4") == 253.0
    assert power_watts(mini, "p1") == 500.0


def test_pcd_unifies_notation(mini):
    assert pcd_of(mini, "r1") == pcd_of(mini, "h1")   # 5x112 == 5*112


def test_diameter_and_clearance(mini):
    assert diameter_mm(mini, "r1") == pytest.approx(18.0 * 25.4, abs=0.01)
    assert diameter_mm(mini, "r2") == pytest.approx(21.0 * 25.4, abs=0.01)
    assert max_diameter_mm(mini, "h1") == pytest.approx(19.0 * 25.4, abs=0.01)


def test_max_diameter_absent_on_items_that_have_none(mini):
    assert max_diameter_mm(mini, "r1") is None


# --------------------------------------------------------------------------
# Compatibility oracle
# --------------------------------------------------------------------------

def test_same_socket_is_compatible(mini):
    v = check_bundle(mini, ["c1", "m1"])       # AM5 + AM5
    assert v.ok, v.reasons


def test_different_socket_is_rejected_with_a_reason(mini):
    v = check_bundle(mini, ["c1", "m3"])       # AM5 + LGA1700
    assert not v.ok
    assert any("сокет" in r for r in v.reasons)


def test_different_pcd_is_rejected(mini):
    v = check_bundle(mini, ["r1", "r2"])       # 5x112 vs 5x114.3
    assert not v.ok
    assert any("розболтовк" in r for r in v.reasons)


def test_spatial_containment_is_enforced(mini):
    v = check_bundle(mini, ["r2", "h1"])       # 21" rim into a 19" hub
    assert not v.ok
    assert any("допуск" in r for r in v.reasons)


def test_spatial_containment_accepts_when_it_fits(mini):
    v = check_bundle(mini, ["r1", "h1"])       # 18" rim into a 19" hub
    assert v.ok, v.reasons


def test_diameter_notation_unifies(mini):
    """R18 (assumption A1) and 18.0" are the same nominal size."""
    assert diameter_mm(mini, "r1") == pytest.approx(
        float(str(canonical_form('18.0"')).removesuffix("mm")), abs=0.01
    )


def test_unknown_attributes_do_not_cause_rejection(mini):
    """Absence of evidence is not evidence of incompatibility."""
    v = check_bundle(mini, ["p1"])              # PSU alone declares no socket
    assert v.ok


# --------------------------------------------------------------------------
# Case generation
# --------------------------------------------------------------------------

def test_case_ids_are_unique_and_ordered(mini):
    cases = build_cases(mini)
    ids = [c.case_id for c in cases]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_generation_is_deterministic(mini):
    a = [(c.case_id, c.query, c.oracle.expected_verdict, tuple(c.oracle.expected_products))
         for c in build_cases(mini)]
    b = [(c.case_id, c.query, c.oracle.expected_verdict, tuple(c.oracle.expected_products))
         for c in build_cases(mini)]
    assert a == b


def test_seed_changes_the_sampled_cases_but_not_the_families(mini):
    a = build_cases(mini, seed=1)
    b = build_cases(mini, seed=2)
    assert collections.Counter(c.case_type for c in a) == \
           collections.Counter(c.case_type for c in b)


def test_every_case_has_an_oracle_id_matching_its_case(mini):
    for c in build_cases(mini):
        assert c.oracle.case_id == c.case_id
        assert c.query.strip()
        assert c.case_type


def test_price_cases_name_a_real_product_at_its_real_price(mini):
    for c in build_cases(mini):
        if c.case_type != "direct_price":
            continue
        name = c.oracle.expected_products[0]
        assert name in mini.price_index
        # The oracle's price must be one of the prices the catalogue lists
        # for that name — otherwise the "correct" answer is unsayable.
        assert c.oracle.expected_prices[name] in mini.price_index[name]


def test_cheapest_case_really_is_the_cheapest(mini):
    for c in build_cases(mini):
        if c.case_type != "cheapest_category":
            continue
        cat = c.meta["category"]
        pool = [p for p in mini.products if mini.products[p]["category"] == cat]
        floor = min(mini.products[p]["price"] for p in pool)
        assert c.oracle.expected_prices[c.oracle.expected_products[0]] == floor


def test_overconstrained_cases_are_below_the_price_floor(mini):
    for c in build_cases(mini):
        if c.case_type != "overconstrained":
            continue
        cat = c.meta["category"]
        pool = [p for p in mini.products if mini.products[p]["category"] == cat]
        floor = min(mini.products[p]["price"] for p in pool)
        assert c.meta["budget"] < floor
        assert c.oracle.expected_products == []
        assert c.oracle.expected_verdict == "reject"


def test_price_traps_name_absent_products(mini):
    traps = [c for c in build_cases(mini) if c.case_type == "price_trap"]
    assert traps
    for c in traps:
        assert c.meta["fabricated"] not in mini.price_index
        assert c.oracle.expected_products == []


def test_compat_cases_agree_with_the_verifier(mini):
    for c in build_cases(mini):
        if c.case_type == "compat_accept":
            assert check_bundle(mini, c.meta["bundle"]).ok, c.case_id
        elif c.case_type == "compat_reject":
            v = check_bundle(mini, c.meta["bundle"])
            assert not v.ok and v.reasons, c.case_id


def test_conversational_cases_expect_no_product(mini):
    for c in build_cases(mini):
        if c.case_type == "conversational":
            assert c.oracle.expected_products == []
            assert c.oracle.expected_verdict == "neutral"


def test_every_category_has_a_ukrainian_name():
    for cat in ("cpu", "motherboard", "ram", "gpu", "psu", "ssd", "rim", "wheel_hub"):
        assert cat in CATEGORY_UA


# --------------------------------------------------------------------------
# The live catalog
# --------------------------------------------------------------------------

@needs_live
def test_live_catalog_scales_to_a_full_benchmark():
    snap = load_snapshot(LIVE_DB)
    assert len(snap.products) == 10000
    # 5 merged concepts cover 27 of 31 keys; the rest stay as singletons so
    # that every key remains analysable.
    assert len(snap.concepts) >= 5
    assert len(snap.key_to_concept) == 31
    assert len(build_cases(snap)) >= 45


@needs_live
def test_live_catalog_unifies_at_least_25_of_31_keys():
    snap = load_snapshot(LIVE_DB)
    unified = sum(len(c.keys) for c in snap.concepts)
    assert unified >= 25
