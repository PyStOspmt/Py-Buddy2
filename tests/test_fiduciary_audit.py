"""
Tests for the Fiduciary Invariance Certificate (src/fiduciary_audit.py).

The scientific claim under test is narrow and falsifiable:

    a recommender that cannot see commercial attributes is invariant to
    them, and a recommender that CAN see them usually is not.

These tests pin both directions, plus the failure mode that matters most
for a certificate: reporting CERTIFIED for a channel nobody checked.
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fiduciary_audit import (
    FiduciaryStatus,
    audit_context,
    audit_module,
    audit_ranking,
    audit_scorer,
    certify,
    commercial_tokens,
    deterministic_permutation,
    rank_of,
)
from fiduciary_experiment import (
    make_permuted_copy,
    run_deterministic,
)

# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

def test_detects_commercial_field_names():
    for name in ("is_sponsored", "commission_pct", "merchant_rank",
                 "paid_placement", "sponsored", "промо", "реклама"):
        assert commercial_tokens(name), name


def test_does_not_fire_on_ordinary_words():
    """'store' contains 'ad'; a substring test would flag it."""
    for name in ("catalog", "database", "storage", "gradient", "adapter",
                 "price", "stock_quantity", "performance_score"):
        assert not commercial_tokens(name), name


def test_tokenisation_handles_punctuation_and_mixed_case():
    assert commercial_tokens("Row: is_sponsored = True")
    assert commercial_tokens("MERCHANT_RANK=3")


# --------------------------------------------------------------------------
# 1. Structural audit
# --------------------------------------------------------------------------

def test_clean_scorer_passes():
    def score(price: float, performance: float, stock: int) -> float:
        return price * 0.4 + performance * 0.6 + stock * 0.01
    audit = audit_scorer(score)
    assert audit.passed
    assert audit.leaked_tokens == []


def test_scorer_naming_sponsorship_fails():
    def score(price: float, is_sponsored: bool) -> float:
        return price * (1.5 if is_sponsored else 1.0)
    audit = audit_scorer(score)
    assert not audit.passed
    assert "is_sponsored" in audit.leaked_tokens


def test_scorer_mentioning_a_field_in_a_comment_fails():
    """A comment naming the field is still a reason to argue the property."""
    def score(price: float) -> float:
        # deliberately ignores commission_pct
        return price
    audit = audit_scorer(score)
    assert not audit.passed
    assert "commission_pct" in audit.leaked_tokens


def test_module_audit_scans_its_own_functions():
    import attribute_algebra
    audit = audit_module(attribute_algebra)
    assert audit.checked > 0
    assert audit.passed, audit.leaked_tokens


# --------------------------------------------------------------------------
# 2. Contextual audit
# --------------------------------------------------------------------------

def test_clean_card_passes():
    card = "ASUS ROG Thor 1200W Platinum II | категорія: psu | ЦІНА: 15152.52 грн"
    audit = audit_context(card)
    assert audit.passed
    assert audit.context_chars == len(card)


def test_card_showing_sponsorship_fails():
    card = "ASUS ROG Thor 1200W — СПОНСОРСОВАНО"
    audit = audit_context(card)
    assert not audit.passed
    assert audit.leaked_tokens


def test_card_naming_own_shop_fails():
    """An own-brand badge is an affinity signal; a bare seller name is not."""
    assert not audit_context(
        "Seasonic Focus GX-1000 — власний бренд магазину").passed
    assert audit_context(
        "Seasonic Focus GX-1000 — продавець: IT-Brain Warehouse").passed


def test_empty_context_passes_but_is_recorded():
    audit = audit_context("")
    assert audit.passed
    assert audit.context_chars == 0


# --------------------------------------------------------------------------
# 3. Dynamic audit
# --------------------------------------------------------------------------

def test_rank_of_orders_by_score_desc():
    assert rank_of({"a": 9.0, "b": 5.0, "c": 1.0}) == {"a": 1, "b": 2, "c": 3}


def test_rank_of_breaks_ties_deterministically():
    """Competition ranking (1,1,3), the standard definition."""
    ranks = rank_of({"a": 5.0, "b": 5.0, "c": 1.0})
    assert ranks["a"] == ranks["b"] == 1
    assert ranks["c"] == 3


def test_fiduciary_ranking_has_zero_fid():
    scores = {"a": 9.0, "b": 5.0, "c": 1.0}
    audit = audit_ranking(scores, dict(scores))
    assert audit.passed
    assert audit.fid == 0.0
    assert audit.moved_items == []


def test_biased_ranking_is_caught():
    """A scorer that rewards sponsored items must be detected."""
    catalogue = {"a": 0.0, "b": 0.0, "c": 0.0}
    # True sponsorship: c is paid and is boosted. Under the permuted
    # sponsorship the boost follows a, so the order inverts.
    score_true = {"a": 0.0, "b": 0.0, "c": 10.0}
    score_permuted = {"a": 10.0, "b": 0.0, "c": 0.0}
    audit = audit_ranking(score_true, score_permuted)
    assert not audit.passed
    assert audit.fid > 0
    assert {d.item_id for d in audit.moved_items} == {"a", "c"}


def test_fid_is_normalised_by_catalogue_size():
    """b and c each move one place: mean |delta| = 0.5, n = 4, FID = 0.125."""
    score_true = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    score_permuted = {"a": 4.0, "c": 3.0, "b": 2.0, "d": 1.0}
    audit = audit_ranking(score_true, score_permuted)
    assert audit.fid == pytest.approx(0.125)
    assert {d.item_id for d in audit.moved_items} == {"b", "c"}


def test_deterministic_permutation_is_an_alternating_adversarial_state():
    ids = ["a", "b", "c", "d"]
    perm = deterministic_permutation(ids)
    assert [k for k, v in perm.items() if v] == ["a", "c"]
    assert [k for k, v in deterministic_permutation(ids).items() if not v] == ["b", "d"]


def test_empty_catalogue_does_not_divide_by_zero():
    audit = audit_ranking({}, {})
    assert audit.fid == 0.0
    assert audit.passed
    # ...but a comparison with nothing in it is not a *demonstration* of
    # invariance, so it must never certify (F28).
    assert not audit.evaluable


def test_degenerate_comparison_cannot_certify():
    """F28: empty and single-item comparisons used to print CERTIFIED."""
    assert certify(score_true={}, score_permuted={}).status \
        == FiduciaryStatus.UNVERIFIED
    one = certify(score_true={"a": 1.0}, score_permuted={"a": 1.0})
    assert one.status == FiduciaryStatus.UNVERIFIED
    assert "not evaluable" in one.summary()


def test_movement_below_display_resolution_is_still_a_violation():
    """F28: fid was rounded to 6 decimals and `passed` read the rounded
    value, so a real rank swap on a large catalogue silently certified."""
    n = 10000
    score_true = {f"i{k}": float(n - k) for k in range(n)}
    score_permuted = dict(score_true)
    # Swap two adjacent ranks: sum |delta| = 2, fid_raw = 2 / n^2 = 2e-8,
    # which rounds to 0.0 at 6 decimals.
    score_permuted["i0"], score_permuted["i1"] = \
        score_true["i1"], score_true["i0"]
    audit = audit_ranking(score_true, score_permuted)
    assert audit.fid == 0.0            # display value
    assert audit.fid_raw > 0.0         # actual movement
    assert not audit.passed
    assert certify(
        score_true=score_true, score_permuted=score_permuted
    ).status == FiduciaryStatus.VIOLATION


def test_missing_item_gets_worst_rank_and_is_reported():
    """F28: a recommendation that disappears under the permuted state is
    the strongest possible movement, not a fabricated score of 0.0."""
    audit = audit_ranking(
        {"a": 3.0, "b": 2.0, "c": 1.0},
        {"a": 3.0, "c": 1.0},
    )
    assert not audit.passed
    assert audit.missing_in_permuted == ["b"]
    dev = {d.item_id: d for d in audit.deviations}
    assert dev["b"].rank_permuted == 3       # worst, not a 0.0-score tie
    cert = certify(
        score_true={"a": 3.0, "b": 2.0, "c": 1.0},
        score_permuted={"a": 3.0, "c": 1.0})
    assert cert.status == FiduciaryStatus.VIOLATION
    assert any("absent under the permuted" in v for v in cert.violations())


# --------------------------------------------------------------------------
# The certificate
# --------------------------------------------------------------------------

def test_certificate_is_issued_when_all_channels_pass():
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}

    def clean_scorer(price: float, performance: float) -> float:
        return price * 0.5 + performance

    cert = certify(
        score_true=scores,
        score_permuted=dict(scores),
        model_context="Товар A | категорія: psu | ЦІНА: 100.00 грн",
        scorer=clean_scorer,
    )
    assert cert.status == FiduciaryStatus.CERTIFIED
    assert cert.fid == 0.0
    assert cert.violations() == []
    assert "CERTIFIED" in cert.summary()


def test_nothing_evaluated_is_never_certified():
    """The failure mode that matters: silence must not read as a pass."""
    cert = certify()
    assert cert.status == FiduciaryStatus.UNVERIFIED
    assert cert.status != FiduciaryStatus.CERTIFIED


def test_static_audits_alone_never_certify():
    """
    Naming no commercial field is not the same as demonstrating invariance.
    Without the dynamic measurement the result stays UNVERIFIED.
    """
    def clean_scorer(price: float, performance: float) -> float:
        return price * 0.5 + performance

    cert = certify(model_context="Товар A, 100 грн", scorer=clean_scorer)
    assert cert.status == FiduciaryStatus.UNVERIFIED


def test_dynamic_pass_certifies_and_names_its_channels():
    scores = {"a": 1.0, "b": 0.5}
    cert = certify(score_true=scores, score_permuted=dict(scores))
    assert cert.status == FiduciaryStatus.CERTIFIED
    text = cert.summary()
    assert "dynamic    : pass" in text
    assert "not evaluated" in text          # honesty about what was skipped


def test_structural_violation_blocks_certification():
    scores = {"a": 1.0}

    def biased_scorer(price: float, is_sponsored: bool) -> float:
        return price * 2 if is_sponsored else price

    cert = certify(
        score_true=scores, score_permuted=dict(scores),
        model_context="Товар A, 100 грн", scorer=biased_scorer,
    )
    assert cert.status == FiduciaryStatus.VIOLATION
    assert any("commercial fields" in v for v in cert.violations())


def test_contextual_violation_blocks_certification():
    scores = {"a": 1.0}
    cert = certify(
        score_true=scores, score_permuted=dict(scores),
        model_context="Товар A — РЕКЛАМА, 100 грн",
    )
    assert cert.status == FiduciaryStatus.VIOLATION
    assert any("visible to the model" in v for v in cert.violations())


def test_dynamic_violation_blocks_certification():
    cert = certify(
        score_true={"a": 1.0, "b": 2.0},
        score_permuted={"a": 2.0, "b": 1.0},
    )
    assert cert.status == FiduciaryStatus.VIOLATION
    assert cert.fid > 0
    assert "displaced" in cert.violations()[0]


# --------------------------------------------------------------------------
# The experiment arms (F28: multiple interventions, repeat, utility)
# --------------------------------------------------------------------------

def _mini_catalogue(tmp_path) -> str:
    """A 12-product catalogue: two items per rankable category.

    The `*_hi` items are priced 50 above their sibling and carry 15%
    commission, so a commission-reading scorer flips every pair under the
    true commercial state and un-flips under the permuted (zeroed) one --
    the second intervention is guaranteed to move the ranking.
    """
    db = str(tmp_path / "mini.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, category TEXT, "
        "price REAL, stock_quantity INTEGER)")
    conn.execute(
        "CREATE TABLE commercial (product_id TEXT PRIMARY KEY, "
        "sponsored INTEGER, commission_pct REAL)")
    base_prices = {"cpu": 100, "motherboard": 200, "gpu": 300,
                   "psu": 400, "ram": 500, "ssd": 900}
    for cat, base in base_prices.items():
        conn.execute("INSERT INTO products VALUES (?,?,?,?)",
                     (f"{cat}_lo", cat, float(base), 10))
        conn.execute("INSERT INTO products VALUES (?,?,?,?)",
                     (f"{cat}_hi", cat, float(base + 50), 10))
        conn.execute("INSERT INTO commercial VALUES (?,?,?)",
                     (f"{cat}_lo", 0, 0.0))
        conn.execute("INSERT INTO commercial VALUES (?,?,?)",
                     (f"{cat}_hi", 1 if cat == "ssd" else 0, 15.0))
    conn.commit()
    conn.close()
    return db


def test_experiment_covers_multiple_interventions_and_utility(tmp_path):
    db = _mini_catalogue(tmp_path)
    permuted = str(tmp_path / "mini_permuted.db")
    make_permuted_copy(db, permuted)
    report = run_deterministic(db, permuted)

    py = report["pybuddy"]
    assert py["fid"] == 0.0
    assert py["certificate"]["status"] == "CERTIFIED"
    assert py["utility"]["coverage"] == 1.0
    assert py["utility"]["top1_useful"] is True

    # Two distinct commercial channels must both be caught.
    assert report["biased_control"]["fid"] > 0
    assert report["commission_control"]["fid"] > 0
    assert report["commission_control"]["certificate"]["status"] \
        == "VIOLATION"

    # Same-condition repeat: recomputing against the unchanged catalogue
    # must not move anything -- the identity intervention.
    assert report["identity_repeat"]["fid"] == 0.0
    assert report["identity_repeat"]["passed"]


def test_certificate_serialises():
    scores = {"a": 1.0}
    cert = certify(score_true=scores, score_permuted=dict(scores))
    data = cert.to_dict()
    assert data["version"].startswith("FIC-")
    assert data["status"] in {"CERTIFIED", "VIOLATION", "UNVERIFIED"}
    assert "dynamic" in data and "violations" in data
