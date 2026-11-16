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


def test_sentence_scoped_summary_and_trailing_price_binding():
    """Summary & trailing-name sentences anchor cleanly while same-sentence swaps fail."""
    rev_catalog = CATALOG + ["AMD Ryzen 5 7600X (Rev 1)", "AMD Ryzen 5 7600X (Rev 2)"]
    rev_prices = {**PRICES, "AMD Ryzen 5 7600X (Rev 1)": [9500.0], "AMD Ryzen 5 7600X (Rev 2)": [9100.0]}
    o = _oracle(
        expected_products=["AMD Ryzen 5 7600X (Rev 2)"],
        expected_prices={"AMD Ryzen 5 7600X (Rev 2)": 9100.0},
        expected_verdict="neutral",
        checks_verdict=False,
    )
    ans = (
        "Ось 2 найдешевші варіанти:\n"
        "1. AMD Ryzen 5 7600X (Rev 2) — 9 100 грн\n"
        "2. AMD Ryzen 5 7600X (Rev 1) — 9 500 грн\n"
        "Найвигідніший варіант — перший за 9 100 грн."
    )
    sc = score_case(ans, o, rev_catalog, rev_prices)
    assert sc.prices_grounded, sc.price_errors
    assert sc.ungrounded_entities == []

    # Trailing product in the same sentence binds cleanly
    sc2 = score_case("Ціна 9 100 грн за AMD Ryzen 5 7600X (Rev 2).", o, rev_catalog, rev_prices)
    assert sc2.prices_grounded, sc2.price_errors

    # Same-sentence price swap MUST still fail
    sc_bad = score_case(
        "AMD Ryzen 5 7600X за 6200 грн та MSI B650 Gaming Plus WiFi за 9500 грн.",
        _oracle(), CATALOG, PRICES,
    )
    assert not sc_bad.prices_grounded


def test_entity_regex_ignores_cyrillic_intro_and_multiline_lists():
    """Prose count phrases ('Ось 5') and newlines ('Ukraine\\n3.') never emit bogus entities."""
    o = _oracle(expected_verdict="neutral", checks_verdict=False)
    ans = "Ось 5 варіантів, AutoDrive Ukraine\n3. AMD Ryzen 5 7600X — 9 500 грн."
    sc = score_case(ans, o, CATALOG, PRICES)
    assert sc.ungrounded_entities == []
    # Genuine fabricated model token is still caught
    sc_bad = score_case("Ось 5: AMD Ryzen 9 9950X.", o, CATALOG, PRICES)
    assert any("9950x" in tok for tok in sc_bad.ungrounded_entities)


def test_match_products_masks_revision_suffix_and_preserves_mention_order():
    rev_catalog = CATALOG + ["AMD Ryzen 5 7600X (Rev 1)", "MSI B650 Gaming Plus WiFi (Rev 2)"]
    found = match_products(
        "Спочатку MSI B650 Gaming Plus WiFi (Rev 2), потім AMD Ryzen 5 7600X (Rev 1).",
        rev_catalog,
    )
    assert found == ["MSI B650 Gaming Plus WiFi (Rev 2)", "AMD Ryzen 5 7600X (Rev 1)"]


def test_extract_verdict_negated_conflict_and_trap_alternatives():
    assert extract_verdict("Так, ці компоненти сумісні! Конфліктів не виявлено.") == "accept"
    assert extract_verdict("Інформації про ціну AMD Ryzen 9 9950X у каталозі бракує.") == "reject"
    trap_with_alt = (
        "Модель Samsung 990 PRO 4TB наразі відсутня в нашому каталозі. "
        "Натомість можу запропонувати схожі варіанти в наявності."
    )
    assert extract_verdict(trap_with_alt) == "reject"


def test_header_to_bullet_revision_and_family_revision_price_anchoring():
    rev_catalog = CATALOG + [
        "AMD Ryzen 5 7600X (Rev 1)",
        "AMD Ryzen 5 7600X (Rev 2)",
    ]
    rev_prices = {
        **PRICES,
        "AMD Ryzen 5 7600X (Rev 1)": [9500.0],
        "AMD Ryzen 5 7600X (Rev 2)": [9100.0],
    }
    o = _oracle(expected_products=[], expected_prices={}, expected_verdict="reject")
    ans = (
        "Запитаної моделі немає, знайдено варіанти AMD Ryzen 5 7600X від різних продавців:\n"
        "• Seller A (Rev 2): 9 100 грн\n"
        "Інші ревізії цієї ж моделі доступні від 9 500 грн."
    )
    sc = score_case(ans, o, rev_catalog, rev_prices)
    assert sc.prices_grounded, sc.price_errors


def test_embedded_capacity_token_contrast_negation_and_per_price_revs():
    """LPX 32GB grounds '32GB', 'а не RTX 5090 Ti' exempts trap, and per-price Rev anchors cleanly."""
    rev_catalog = CATALOG + [
        "AMD Ryzen 5 7600X (Rev 1)",
        "AMD Ryzen 5 7600X (Rev 2)",
        "Corsair Vengeance LPX 32GB DDR4-3600 (Rev 1)",
        "MSI B650 Gaming Plus WiFi (Rev 2)",
    ]
    rev_prices = {
        **PRICES,
        "AMD Ryzen 5 7600X (Rev 1)": [9500.0],
        "AMD Ryzen 5 7600X (Rev 2)": [9100.0],
        "Corsair Vengeance LPX 32GB DDR4-3600 (Rev 1)": [2510.31],
        "MSI B650 Gaming Plus WiFi (Rev 2)": [6900.0],
    }
    o_trap = _oracle(expected_products=[], expected_prices={}, expected_verdict="reject")
    ans_trap = (
        "NVIDIA GeForce RTX 5090 Ti у каталозі немає. "
        "Серед схожих моделей є AMD Ryzen 5 7600X (Rev 2) — це інша модель, а не RTX 5090 Ti. "
        "Їхні ціни: 9 100 грн або 9 500 грн (Rev 1). Модель на 32GB DDR4 — Corsair Vengeance LPX 32GB DDR4-3600 (Rev 1) за 2 510,31 грн."
    )
    sc_trap = score_case(
        ans_trap, o_trap, rev_catalog, rev_prices,
        user_query="Скільки коштує NVIDIA GeForce RTX 5090 Ti? Чи є 32GB?",
    )
    assert sc_trap.ungrounded_entities == []
    assert sc_trap.prices_grounded, sc_trap.price_errors

    # Multi-item line picks the LAST family ('MSI B650...') for downstream 'Rev 2' continuation
    o_multi = _oracle(expected_products=[], expected_prices={}, expected_verdict="neutral", checks_verdict=False)
    ans_multi = (
        "Порівняння: AMD Ryzen 5 7600X (Rev 1) — 9 500 грн; материнська плата MSI B650 Gaming Plus WiFi — 6 800 грн.\n"
        "Інші пропозиції цієї материнської плати: Rev 2 за 6 900 грн."
    )
    sc_multi = score_case(ans_multi, o_multi, rev_catalog, rev_prices)
    assert sc_multi.prices_grounded, sc_multi.price_errors
