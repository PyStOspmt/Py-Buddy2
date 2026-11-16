"""
Unit tests for src/metrics.py — the honest-evaluation layer.

These tests pin down the *failure modes of the metric itself*. A benchmark
that cannot detect a hallucination is worse than no benchmark, so every
extraction rule is tested against a known counter-example.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from metrics import (
    Oracle,
    aggregate,
    extract_prices,
    extract_verdict,
    match_products,
    parse_number,
    score_case,
)

CATALOG = [
    "AMD Ryzen 5 7600X",
    "MSI B650 Gaming Plus WiFi",
    "Corsair RM750e 750W 80+ Gold ATX 3.0",
]
PRICES = {n: p for n, p in zip(CATALOG, [9500.0, 6800.0, 4600.0])}


# --------------------------------------------------------------------------
# Number parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("token,expected", [
    ("9500", 9500.0),
    ("9 500", 9500.0),
    ("9'500", 9500.0),
    ("9\u00a0500", 9500.0),
    ("17,200", 17200.0),        # 3 trailing digits -> thousands separator
    ("17,200.00", 17200.0),     # both marks, last one is decimal
    ("1 234,56", 1234.56),      # 2 trailing digits -> decimal
    ("2,5", 2.5),               # 1 trailing digit -> decimal
    ("2.5", 2.5),
    ("1000", 1000.0),
])
def test_parse_number(token, expected):
    assert parse_number(token) == expected


def test_parse_number_rejects_garbage():
    assert parse_number("") is None
    assert parse_number("abc") is None


def test_extract_prices_variants():
    text = "Ціна 9 500 грн, стара 17,200.00 грн та 1234.56грн"
    assert extract_prices(text) == [9500.0, 17200.0, 1234.56]


def test_extract_prices_ignores_non_price_numbers():
    # A bare number with no currency marker must not be treated as a price.
    assert extract_prices("Release 2024, cores 16") == []


# --------------------------------------------------------------------------
# Verdict classification — the negation trap
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("AM5 не сумісний з LGA1700.", "reject"),
    ("AM5 несумісний з LGA1700.", "reject"),
    ("Це повністю сумісна збірка.", "accept"),
    ("Підійде MSI B650.", "accept"),
    ("Немає жодного варіанта.", "reject"),
    ("Нічого не знайдено.", "reject"),
    ("Рекомендую Corsair RM750e, в наявності.", "accept"),
    ("Не рекомендую, дефіцит потужності.", "reject"),
    ("", "neutral"),
])
def test_extract_verdict(text, expected):
    assert extract_verdict(text) == expected


def test_verdict_uses_concluding_sentence():
    text = "Запит цілком можливий. Проте блок живлення не вистачає."
    assert extract_verdict(text) == "reject"


# --------------------------------------------------------------------------
# Product matching
# --------------------------------------------------------------------------

def test_match_products_case_and_punctuation_insensitive():
    assert match_products("Ось msi b650 gaming plus wifi!", CATALOG) == \
        ["MSI B650 Gaming Plus WiFi"]


def test_match_products_preserves_mention_order_and_dedupes():
    text = "Corsair RM750e 750W 80+ Gold ATX 3.0, потім AMD Ryzen 5 7600X, і знову Corsair RM750e 750W 80+ Gold ATX 3.0"
    found = match_products(text, CATALOG)
    assert found == ["Corsair RM750e 750W 80+ Gold ATX 3.0", "AMD Ryzen 5 7600X"]


def test_match_products_returns_empty_for_foreign_names():
    assert match_products("AMD Ryzen 9 9950X", CATALOG) == []


# --------------------------------------------------------------------------
# Scoring — the core honesty checks
# --------------------------------------------------------------------------

def _oracle(**kw):
    base = dict(
        case_id="t1", case_type="direct_price",
        expected_products=["AMD Ryzen 5 7600X"],
        expected_prices={"AMD Ryzen 5 7600X": 9500.0},
        expected_verdict="accept", feasible=True,
    )
    base.update(kw)
    return Oracle(**base)


def test_perfect_answer_scores_correct():
    o = _oracle()
    s = score_case("Раджу AMD Ryzen 5 7600X за 9 500 грн — він повністю сумісний.", o, CATALOG, PRICES)
    assert s.is_correct
    assert s.mention_precision == 1.0
    assert s.prices_grounded
    assert s.verdict == "accept"
    assert not s.violations


def test_hallucinated_product_is_detected():
    o = _oracle()
    s = score_case("Раджу AMD Ryzen 9 9950X за 60 000 грн.", o, CATALOG, PRICES)
    assert "hallucinated_product" in s.violations
    assert not s.is_correct


def test_wrong_price_is_detected():
    o = _oracle()
    s = score_case("AMD Ryzen 5 7600X коштує 7 999 грн, сумісний.", o, CATALOG, PRICES)
    assert not s.prices_grounded
    assert "ungrounded_price" in s.violations


def test_unanchored_price_is_detected():
    """A price for a product the answer never named is a fabricated figure."""
    o = _oracle()
    s = score_case("У комплекті AMD Ryzen 5 7600X та блок за 12 000 грн.", o, CATALOG, PRICES)
    assert not s.prices_grounded


def test_wrong_verdict_is_detected():
    o = _oracle(expected_verdict="reject")
    s = score_case("Це повністю сумісна конфігурація.", o, CATALOG, PRICES)
    assert not s.verdict_correct
    assert "wrong_verdict" in s.violations


def test_blocked_answer_is_not_informative():
    o = _oracle()
    s = score_case(
        "Помилка верифікації фактів. Відповідь заблоковано Zero-Hallucination Gate.",
        o, CATALOG, PRICES,
    )
    assert s.blocked
    assert not s.informative
    assert "blocked_no_content" in s.violations


# --------------------------------------------------------------------------
# Aggregation — NVFG is the metric that exposes vacuous outputs
# --------------------------------------------------------------------------

def test_aggregate_nvfg_separates_factual_from_useful():
    o = _oracle()
    good = score_case("AMD Ryzen 5 7600X за 9 500 грн, сумісний.", o, CATALOG, PRICES)
    blocked = score_case(
        "Помилка верифікації фактів. Відповідь заблоковано Zero-Hallucination Gate.",
        o, CATALOG, PRICES,
    )
    agg = aggregate([good, blocked])
    # Nothing was hallucinated in either answer, so factuality is clean...
    assert agg.hallucination_rate_pct == 0.0
    # ...but the blocked one carries no information, so NVFG is halved.
    assert agg.nvfg_pct == 50.0
    assert agg.blocked_rate_pct == 50.0


def test_aggregate_empty_is_safe():
    agg = aggregate([])
    assert agg.n == 0 and agg.nvfg_pct == 0.0


def test_aggregate_groups_by_case_type():
    o = _oracle(case_type="compatibility")
    s = score_case("AMD Ryzen 5 7600X за 9 500 грн, сумісний.", o, CATALOG, PRICES)
    agg = aggregate([s])
    assert "compatibility" in agg.by_type
    assert agg.by_type["compatibility"]["n"] == 1


def test_direct_price_missing_all_prices_fails_is_correct():
    o = _oracle(expected_verdict="neutral", checks_verdict=False)
    s = score_case("Знайшов у каталозі:\n- AMD Ryzen 5 7600X", o, CATALOG, PRICES)
    assert s.prices_grounded
    assert "missing_expected_price" in s.violations
    assert not s.is_correct


def test_metrics_separate_verdict_abstention_from_contradiction():
    """F12: unknown/neutral keeps is_correct=False while distinguishing abstention from wrong verdict."""
    o_accept = _oracle(case_type="compat_accept", expected_verdict="accept", checks_verdict=True)
    o_reject = _oracle(case_type="compat_reject", expected_verdict="reject", checks_verdict=True)

    exact = score_case(
        "AMD Ryzen 5 7600X сумісний, 9500 грн.",
        o_accept, CATALOG, PRICES, structured_verdict="compatible",
    )
    abstained = score_case(
        "AMD Ryzen 5 7600X за 9500 грн потребує ручної перевірки.",
        o_accept, CATALOG, PRICES, structured_verdict="unknown",
    )
    contradicted = score_case(
        "AMD Ryzen 5 7600X за 9500 грн сумісний.",
        o_reject, CATALOG, PRICES, structured_verdict="compatible",
    )

    assert exact.verdict_correct and not exact.verdict_abstained and not exact.verdict_contradicted
    assert abstained.raw_structured_verdict == "unknown"
    assert abstained.verdict == "neutral"
    assert not abstained.verdict_correct
    assert not abstained.is_correct
    assert abstained.verdict_abstained and not abstained.verdict_contradicted
    assert not contradicted.verdict_abstained and contradicted.verdict_contradicted

    agg = aggregate([exact, abstained, contradicted])
    assert agg.verdict_cases == 3
    assert agg.verdict_abstained_count == 1
    assert agg.verdict_contradiction_count == 1
    assert agg.by_type["compat_accept"]["verdict_abstained_count"] == 1
    assert agg.by_type["compat_reject"]["verdict_contradiction_count"] == 1
