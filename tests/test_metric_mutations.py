"""
Mutation tests for the benchmark scoring metric itself.

These tests validate the METRIC, not a system under test: the catalogue and
oracle below are synthetic fixtures, clearly labelled as evaluation
mechanisms. Every mutation of a valid answer -- a dropped required product, a
price swapped between two products, a flipped verdict, a hallucinated model
token and a blocked non-answer -- must fail `is_correct`.

Because recall and entity-offer binding are enforced jointly, dropped-product
and swapped-price mutations are rejected consistently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "core"))

from metrics import Oracle, aggregate, score_case

# --- synthetic catalogue fixture (evaluation mechanism, not product data) ---
CATALOG = [
    "AMD Ryzen 5 7600X",
    "MSI B650 Gaming Plus WiFi",
    "Corsair RM850x",
    "Samsung 990 Pro 2TB",
]
PRICES = {
    "AMD Ryzen 5 7600X": [9500.0],
    "MSI B650 Gaming Plus WiFi": [6200.0],
    "Corsair RM850x": [4300.0],
    "Samsung 990 Pro 2TB": [5100.0],
}

ORACLE = Oracle(
    case_id="mut-001",
    case_type="compatibility",
    expected_products=["AMD Ryzen 5 7600X", "MSI B650 Gaming Plus WiFi"],
    expected_prices={"AMD Ryzen 5 7600X": 9500.0, "MSI B650 Gaming Plus WiFi": 6200.0},
    expected_verdict="accept",
    checks_verdict=True,
)

VALID = ("Раджу AMD Ryzen 5 7600X за 9500 грн та MSI B650 Gaming Plus WiFi "
         "за 6200 грн — конфігурація сумісна.")


def _score(text: str):
    return score_case(text, ORACLE, CATALOG, PRICES)


def test_valid_answer_is_correct():
    s = _score(VALID)
    assert s.is_correct, (s.violations, s.mention_recall, s.price_errors)


def test_mutation_drop_required_product_is_rejected():
    # "Failed before": precision-only scoring accepted this.
    s = _score("Раджу AMD Ryzen 5 7600X за 9500 грн — конфігурація сумісна.")
    assert not s.is_correct
    assert s.mention_recall < 1.0


def test_mutation_swapped_prices_are_rejected():
    # "Failed before": union-anchoring accepted cross-product price swaps.
    s = _score("Раджу AMD Ryzen 5 7600X за 6200 грн та MSI B650 Gaming Plus WiFi "
               "за 9500 грн — конфігурація сумісна.")
    assert not s.is_correct
    assert not s.prices_grounded


def test_mutation_flipped_verdict_is_rejected():
    s = _score("Раджу AMD Ryzen 5 7600X за 9500 грн та MSI B650 Gaming Plus WiFi "
               "за 6200 грн — конфігурація несумісна, відхилено.")
    assert not s.is_correct
    assert not s.verdict_correct


def test_mutation_hallucinated_model_token_is_rejected():
    s = _score("Раджу AMD Ryzen 5 7600X за 9500 грн, MSI B650 Gaming Plus WiFi "
               "за 6200 грн та Ryzen 9 9950X — конфігурація сумісна.")
    assert not s.is_correct
    assert s.ungrounded_entities


def test_mutation_blocked_nonanswer_is_rejected():
    s = _score("Помилка верифікації фактів. Відповідь заблоковано "
               "Zero-Hallucination Gate.")
    assert not s.is_correct
    assert s.blocked


def test_false_claim_rate_is_live_not_structurally_zero():
    # "Failed before": ungrounded_products was never populated, so
    # false_claim_rate_pct was 0.0 for every input -- a dead metric.
    s = _score("Раджу AMD Ryzen 5 7600X за 9500 грн та Corsair RM850x — "
               "конфігурація сумісна.")
    agg = aggregate([s])
    assert s.ungrounded_products, "unexpected-product mention must be recorded"
    assert agg.false_claim_rate_pct > 0.0


def test_nvfg_cannot_be_true_with_wrong_verdict():
    s = _score("Раджу AMD Ryzen 5 7600X за 9500 грн та MSI B650 Gaming Plus WiFi "
               "за 6200 грн — конфігурація несумісна, відхилено.")
    agg = aggregate([s])
    assert agg.nvfg_pct == 0.0