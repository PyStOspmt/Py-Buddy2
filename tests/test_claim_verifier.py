"""
Regression tests for the post-generation claim verifier
(src/claim_verifier.py).

The grounding gate protects structured fields; this module is the check
the reviewer asked for -- it verifies the free-text prose around them.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from claim_verifier import (
    evidence_item_from_row,
    verify_answer,
    verify_answer_against_rows,
)

ROWS = [
    {"name": "Intel Core i5-13400F", "price": 8999.0, "stock_quantity": 5,
     "attributes": {"socket": "LGA1700", "tdp": "65 W"}},
    {"name": "Gigabyte B650 AORUS Elite AX", "price": 7500.0,
     "stock_quantity": 3,
     "attributes": {"socket": "AM5", "form": "ATX"}},
    {"name": "MSI MAG Z790 Tomahawk", "price": 10864.01, "stock_quantity": 0,
     "attributes": {"socket": "LGA1700"}},
]

EVIDENCE = [evidence_item_from_row(r) for r in ROWS]


def test_grounded_sentence_passes():
    text = ("Intel Core i5-13400F коштує 8999 грн і працює на сокеті LGA1700. "
            "Є 5 штук на складі.")
    rep = verify_answer(text, EVIDENCE, "порадь процесор до 9000 грн")
    assert rep.n_unsupported == 0
    assert rep.supported_fraction == 1.0
    for s in rep.sentences:
        assert s.verdict in ("verified", "no_claims")


def test_list_ordinals_are_not_claims():
    text = ("1. Intel Core i5-13400F коштує 8999 грн.\n"
            "2. Gigabyte B650 AORUS Elite AX коштує 7500 грн.\n"
            "3. MSI MAG Z790 Tomahawk коштує 10864.01 грн.")
    rep = verify_answer(text, EVIDENCE)
    assert rep.n_unsupported == 0
    for s in rep.sentences:
        assert s.verdict in ("verified", "no_claims")


def test_list_ordinal_does_not_shield_fabricated_price():
    text = "4. Intel Core i5-13400F коштує 4500 грн."
    rep = verify_answer(text, EVIDENCE)
    assert rep.n_unsupported == 1
    assert any("4500" in u
               for s in rep.sentences for u in s.unsupported)


def test_fabricated_price_is_flagged():
    text = "Intel Core i5-13400F коштує 4500 грн."
    rep = verify_answer(text, EVIDENCE)
    assert rep.n_unsupported == 1
    flagged = rep.sentences[0].unsupported
    assert any("4500" in u for u in flagged)
    assert "Intel Core i5-13400F коштує" not in rep.corrected_text


def test_fabricated_spec_token_is_flagged():
    text = "Gigabyte B650 AORUS Elite AX має сокет LGA1700."
    rep = verify_answer(text, EVIDENCE)
    assert rep.n_unsupported == 1
    assert any("lga1700" in u.lower() for u in rep.sentences[0].unsupported)


def test_user_stated_numbers_are_context_not_fabrication():
    text = "Ваш бюджет 9000 грн покриває Intel Core i5-13400F за 8999 грн."
    rep = verify_answer(text, EVIDENCE, "процесор до 9000 грн")
    assert rep.n_unsupported == 0


def test_total_price_from_arithmetic_closure_passes():
    text = ("Разом Intel Core i5-13400F і Gigabyte B650 AORUS Elite AX "
            "коштують 16499 грн.")
    rep = verify_answer(text, EVIDENCE)
    assert rep.n_unsupported == 0


def test_wrong_total_is_flagged():
    text = ("Разом Intel Core i5-13400F і Gigabyte B650 AORUS Elite AX "
            "коштують 12000 грн.")
    rep = verify_answer(text, EVIDENCE)
    assert rep.n_unsupported == 1


def test_opinion_sentence_without_claims_passes():
    text = "Це дуже хороший вибір для ігор."
    rep = verify_answer(text, EVIDENCE)
    assert rep.sentences[0].verdict == "no_claims"
    assert rep.supported_fraction == 1.0


def test_mixed_answer_drops_only_bad_sentences():
    text = ("Intel Core i5-13400F коштує 8999 грн. "
            "Він має вбудовану відеокарту RTX-4090. "
            "Підходить для ігор.")
    rep = verify_answer(text, EVIDENCE)
    assert rep.n_unsupported == 1
    assert "8999" in rep.corrected_text
    assert "RTX-4090" not in rep.corrected_text
    assert "Підходить" in rep.corrected_text


def test_empty_text_is_honest_empty():
    rep = verify_answer("", EVIDENCE)
    assert rep.sentences == []
    assert rep.corrected_text == ""


def test_rows_wrapper():
    rep = verify_answer_against_rows("Intel Core i5-13400F — 8999 грн.", ROWS)
    assert rep.n_unsupported == 0


def test_negative_existence_contradiction_caught():
    rows = [
        {"name": "Radeon RX 6500 XT", "category": "Відеокарти", "price": 5843.74},
        {"name": "RTX 4060 Windforce", "category": "Відеокарти", "price": 11743.74},
    ]
    text = (
        "На жаль, у поточному каталогу немає відеокарт з ціною до 5900 грн. "
        "Найдешевший варіант — RTX 4060 Windforce за 11743.74 грн."
    )
    rep = verify_answer_against_rows(
        text, rows, "Порадь відеокарта з ціною до 5900 грн")
    assert rep.n_unsupported == 1
    assert "суперечить каталогу" in rep.sentences[0].unsupported[0]


def test_negative_existence_honest_when_no_match():
    rows = [
        {"name": "RTX 4060 Windforce", "category": "Відеокарти", "price": 11743.74},
    ]
    text = "На жаль, у поточному каталогу немає відеокарт з ціною до 5900 грн."
    rep = verify_answer_against_rows(
        text, rows, "Порадь відеокарта з ціною до 5900 грн")
    assert rep.n_unsupported == 0


def test_empty_evidence_flags_spec_and_price_passes_plain_refusal():
    bad = verify_answer("Процесор має сокет LGA1700 і коштує 4500 грн.", [])
    assert bad.n_unsupported == 1
    plain = verify_answer("За вашим запитом товарів у каталозі не знайдено.", [])
    assert plain.n_unsupported == 0


def test_spec_phrase_before_sentence_period_passes():
    rows = [
        {"name": "MSI MAG Z790 Tomahawk (Rev 4)", "price": 9000.0,
         "attributes": {"socket": "LGA1700"}},
    ]
    rep = verify_answer_against_rows(
        "MSI MAG Z790 Tomahawk (Rev 4) має роз'єм LGA 1700.", rows)
    assert rep.n_unsupported == 0


def test_joint_claim_requires_all_items_and_rejects_tool_conflicts():
    rows = [
        {"name": "AMD Ryzen 5 7600X (Rev 4)", "price": 8892.85,
         "attributes": {"socket": "AM5"}},
        {"name": "MSI MAG Z790 Tomahawk (Rev 4)", "price": 9000.0,
         "attributes": {"socket": "LGA1700"}},
    ]
    prose = (
        "Так, AMD Ryzen 5 7600X (Rev 4) та MSI MAG Z790 Tomahawk (Rev 4) "
        "можна встановити разом. "
        "Обидва компоненти підтримують сокет AM5."
    )
    rep = verify_answer_against_rows(
        prose, rows,
        tool_conflicts=["Еквівалентність socket: AM5 vs LGA1700"],
        compatibility_verdict="candidate_conflict",
    )
    assert rep.n_unsupported == 2


def test_score_case_forbids_fake_price_but_allows_honest_reject_query_echo():
    from metrics import Oracle, score_case
    oracle = Oracle(
        case_id="pt_echo",
        case_type="price_trap",
        expected_verdict="reject",
        expected_products=[],
        expected_prices={},
    )
    catalog = ["AMD Ryzen 5 5600 (Rev 2)"]
    p_index = {"AMD Ryzen 5 5600 (Rev 2)": 4053.14}
    honest = score_case(
        "На жаль, процесора 'AMD Ryzen 9 9950X' немає в наявності. Можемо запропонувати AMD Ryzen 5 5600 (Rev 2) за 4053.14 грн.",
        oracle, catalog, p_index,
        user_query="Скільки коштує AMD Ryzen 9 9950X?",
    )
    assert "hallucinated_product" not in honest.violations
    assert honest.verdict_correct is True

    fake_priced = score_case(
        "На жаль, AMD Ryzen 9 9950X майже немає, але коштує 24000 грн.",
        oracle, catalog, p_index,
        user_query="Скільки коштує AMD Ryzen 9 9950X?",
    )
    assert "hallucinated_product" in fake_priced.violations


def test_ordered_list_markers_and_unknown_brand_not_flagged():
    rows = [
        {"name": "Alpha Pen", "price": 958.99, "brand": None},
        {"name": "Beta NAS", "price": 954.99, "brand": None},
        {"name": "Gamma Server", "price": 944.35, "brand": "HP Enterprise"},
        {"name": "Delta Tablet", "price": 919.00, "brand": "Apple"},
        {"name": "Epsilon Node", "price": 868.95, "brand": "HP Enterprise"},
        {"name": "Zeta Card", "price": 837.99, "brand": None},
        {"name": "Eta Blade", "price": 828.95, "brand": "HP Enterprise"},
    ]
    multiline_answer = (
        "Ось товари з каталогу:\n"
        "1. **Alpha Pen** — ціна: 958.99 | бренд: unknown\n"
        "2.\n**Beta NAS** — ціна: 954.99 | бренд: unknown\n"
        "3. **Gamma Server** — ціна: 944.35 | бренд: HP Enterprise\n"
        "4. **Delta Tablet** — ціна: 919.00 | бренд: Apple\n"
        "5. **Epsilon Node** — ціна: 868.95 | бренд: HP Enterprise\n"
        "6.\n**Zeta Card** — ціна: 837.99 | бренд: unknown\n"
        "7.\n**Eta Blade** — ціна: 828.95 | бренд: HP Enterprise\n"
    )
    rep1 = verify_answer_against_rows(multiline_answer, rows, user_query="Порадь товар до 1000")
    assert rep1.n_unsupported == 0, [s for s in rep1.sentences if s.verdict == "unsupported"]
    assert not any(sv.text.strip() in {"1.", "2.", "6.", "7."} for sv in rep1.sentences)

    inline_answer = (
        "Ось товари: 1. **Alpha Pen** — 958.99 | бренд: unknown "
        "2. **Beta NAS** — 954.99 | бренд: unknown "
        "6. **Zeta Card** — 837.99 | бренд: unknown "
        "7. **Eta Blade** — 828.95 | бренд: HP Enterprise"
    )
    rep2 = verify_answer_against_rows(inline_answer, rows, user_query="Порадь товар до 1000")
    assert rep2.n_unsupported == 0, [s for s in rep2.sentences if s.verdict == "unsupported"]


def test_tool_counts_allowed_only_in_count_context_never_as_price_or_spec():
    rows = [{"name": "3Doodler Create", "price": 958.99}]
    ok = verify_answer_against_rows(
        "У каталозі знайдено 20 товарів з ціною до 1000.",
        rows,
        user_query="Порадь товар до 1000",
        tool_counts=[20],
    )
    assert ok.n_unsupported == 0

    bad_price = verify_answer_against_rows(
        "Товар коштує 20 грн.",
        rows,
        user_query="Порадь товар",
        tool_counts=[20],
    )
    assert bad_price.n_unsupported == 1

    bad_spec = verify_answer_against_rows(
        "Потужність пристрою становить 20 Вт.",
        rows,
        user_query="Порадь товар",
        tool_counts=[20],
    )
    assert bad_spec.n_unsupported == 1

    mixed = verify_answer_against_rows(
        "У каталозі знайдено 20 товарів, один має 20 ГБ пам'яті.",
        rows,
        user_query="Порадь товар",
        tool_counts=[20],
    )
    assert mixed.n_unsupported == 1


def test_ukrainian_abbreviations_and_list_markers_not_split_by_sentence_spans():
    from claim_verifier import _sentence_spans
    from grounding_gate import sentence_spans

    abbr = "Ціна товару становить 9000 тис. грн (див. каталог, т.д. без змін)."
    assert len(_sentence_spans(abbr)) == 1
    assert len(sentence_spans(abbr)) == 1

    list_item = "6. **NVIDIA Quadro M4000** — ціна: 837.99"
    assert len(_sentence_spans(list_item)) == 1
    assert len(sentence_spans(list_item)) == 1


def test_price_cap_re_matches_usd_eur_and_symbols():
    from claim_verifier import _PRICE_CAP_RE

    assert _PRICE_CAP_RE.search("У межах до 1000 USD товарів немає").group(1) == "1000"
    assert _PRICE_CAP_RE.search("Under $950 nothing matched").group(1) == "950"
    assert _PRICE_CAP_RE.search("Немає товарів до 500 EUR").group(1) == "500"
    assert _PRICE_CAP_RE.search("Nothing below €49.99").group(1) == "49.99"


def test_multi_offer_evidence_deduplicates_sentence_products():
    rows = [
        {"id": f"offer_{i}", "name": "ASUS ROG Thor 1200W Platinum II", "price": 15000.0 + i * 50}
        for i in range(6)
    ]
    rep = verify_answer_against_rows(
        "Блок живлення ASUS ROG Thor 1200W Platinum II коштує від 15000 грн.",
        rows,
        user_query="Скільки коштує ASUS ROG Thor 1200W Platinum II?",
    )
    assert rep.n_unsupported == 0
    assert rep.sentences[0].products == ["ASUS ROG Thor 1200W Platinum II"]

