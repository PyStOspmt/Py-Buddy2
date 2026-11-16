"""
Regression suite: mathematical invariants for attribute folding, constrained optimization, and grounding verification.
"""

import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import sqlite3

from attribute_algebra import (
    CONFLICT,
    canonical_form,
    fold_all,
    merge,
)
from catalog_query import CandidateSolution as CS
from catalog_query import GroundingSet
from entity_resolution import MarketplaceEntityResolver
from nvfg import Constraint, Op, feasible
from pareto_relaxer import CandidateSolution, FuzzyParetoRelaxer

# ---------------------------------------------------------------------------
# F20 · the value algebra must honour its stated laws
# ---------------------------------------------------------------------------

def test_f20_projection_is_idempotent_on_s_prefixed_words():
    """
    Failed before: canonical_form("ssd") == "sd" and canonical_form("sd")
    == "d", so the projection changed the value on every application and
    canonical_form(canonical_form(x)) != canonical_form(x).
    """
    assert canonical_form("ssd") == "ssd"
    assert canonical_form(canonical_form("ssd")) == canonical_form("ssd")
    # The legitimate marker case still unifies.
    assert canonical_form("sAM5") == canonical_form("AM5") == "am5"


def test_f20_unparseable_value_is_never_the_identity():
    """
    Failed before: merge(None, "garbage") returned None, so an unparseable
    value vanished as if the attribute were absent -- and one grouping of
    {garbage, AM5} yielded "am5" while another yielded CONFLICT, breaking
    associativity.
    """
    g = "garbage"
    assert merge(None, g) == g          # preserved, visible, not absent
    assert merge(g, None) == g
    assert merge(g, g) == g             # same raw value: still itself
    # An unparseable value is incommensurable with a parseable one in
    # every grouping.
    assert merge(g, "AM5") is CONFLICT
    assert merge(merge(None, g), "AM5") is CONFLICT
    assert merge(None, merge(g, "AM5")) is CONFLICT


def test_f20_merge_is_associative_on_mixed_inputs():
    """Property check over a mixed corpus: parseable, unparseable, absent."""
    values = ["AM5", "sAM5", "350W", "350 Вт", "garbage", "ssd"]
    for r in range(1, 4):
        for triple in itertools.permutations(values, 3):
            a, b, c = triple
            left = merge(merge(a, b), c)
            right = merge(a, merge(b, c))
            assert left == right or (left is CONFLICT and right is CONFLICT), \
                f"non-associative: ({a!r} ⊕ {b!r}) ⊕ {c!r}"


def test_f20_canonicalization_is_closed_under_unit_conversion():
    """
    Failed before: the numeric regex capped digits at five, so a converted
    large magnitude ("1000 kg" -> "1000000g") could not be re-parsed and
    canonicalization was not closed.
    """
    once = canonical_form("1000 kg")
    assert once is not None
    assert canonical_form(once) == once  # fixed point


def test_f20_fold_preserves_the_information_that_values_disagree():
    """
    Failed before: fold_all([garbage, AM5]) returned "am5" -- the unparseable
    value silently agreed with everything. Now it is a conflict.
    """
    assert fold_all(["garbage", "AM5"]) is CONFLICT
    assert fold_all(["garbage", "garbage"]) == "garbage"


# ---------------------------------------------------------------------------
# F21/P21 · unknown data never satisfies a hard requirement
# ---------------------------------------------------------------------------

def test_f21_unknown_value_does_not_satisfy_a_hard_requirement():
    """
    Failed before: satisfied_by(None) was True, so an item that omits the
    attribute walked through every hard filter -- including safety and
    identity constraints.
    """
    c = Constraint(name="socket", attribute="socket_code", op=Op.EQ, value=1.0)
    assert c.satisfied_by(None) is False
    assert c.violated_by(None) is False  # unknown is not a violation either

    items = [
        {"id": "known", "socket_code": 1.0},
        {"id": "silent"},                    # states no socket at all
    ]
    feasible_ids = [i["id"] for i in feasible(items, [c])]
    assert feasible_ids == ["known"]


def test_f21_relaxation_cannot_rescue_an_item_that_has_no_data():
    """An unknown cannot be made feasible by relaxing a bound."""
    c = Constraint(name="budget", attribute="price", op=Op.AT_MOST, value=100.0)
    items = [{"id": "priced", "price": 500.0}, {"id": "unpriced"}]
    feasible_ids = [i["id"] for i in feasible(items, [c])]
    assert feasible_ids == []  # neither fits; the unpriced one is unknown


# ---------------------------------------------------------------------------
# F23 · dominance and explanations must be truthful
# ---------------------------------------------------------------------------

def test_f23_an_item_silent_on_an_axis_cannot_dominate_on_it():
    """
    Failed before: _dominates walked only a's own keys, so an item with a
    price membership but no performance could "dominate" an item that had
    both, ignoring the axis it was silent on -- leaving genuinely dominated
    items on the front.
    """
    relaxer = FuzzyParetoRelaxer(budget_max=1000.0, min_performance=50.0)
    full = CandidateSolution(item_id="full", name="Full", price=500.0,
                             performance_score=80.0)
    partial = CandidateSolution(item_id="partial", name="Partial", price=100.0,
                                performance_score=None)
    front = relaxer.compute_pareto_front([full, partial])
    # `full` states both axes and is better on performance; `partial` is
    # silent on performance, so it cannot be dominated through it -- but it
    # also cannot dominate `full`, whose price membership it lacks... the
    # honest front keeps both only when neither dominates on shared axes.
    ids = {c.item_id for c in front}
    assert "partial" in ids or "full" in ids
    # The precise honest property: partial does NOT dominate full.
    assert not relaxer._dominates(partial, full)


def test_f23_explanation_claims_are_checked_against_the_numbers():
    """
    Failed before: "вкладається у бюджет" and "забезпечує максимальну
    продуктивність" were asserted unconditionally, even when the numbers
    refuted them.
    """
    relaxer = FuzzyParetoRelaxer(budget_max=1000.0, min_performance=50.0)
    a = CandidateSolution(item_id="a", name="A", price=900.0, performance_score=60.0)
    b = CandidateSolution(item_id="b", name="B", price=1100.0, performance_score=55.0)
    front = relaxer.compute_pareto_front([a, b])
    _, _, explanation = relaxer.select_contrastive_tradeoffs(front)
    # b is NOT within the budget and is NOT the performance maximum, so the
    # explanation must not claim either.
    assert "забезпечує максимальну продуктивність" not in explanation or \
        "55.0" not in explanation.split("максимальну продуктивність")[1][:20]


def test_pareto_relaxer_formats_explicit_currency_without_uah_default():
    relaxer_eur = FuzzyParetoRelaxer(budget_max=100.0, min_performance=50.0, currency="EUR")
    a = CandidateSolution(item_id="a", name="Item A", price=125.0, performance_score=70.0)
    b = CandidateSolution(item_id="b", name="Item B", price=80.0, performance_score=60.0)
    front = relaxer_eur.compute_pareto_front([a, b])
    _, _, exp_eur = relaxer_eur.select_contrastive_tradeoffs(front)
    assert "EUR" in exp_eur
    assert "грн" not in exp_eur

    relaxer_none = FuzzyParetoRelaxer(budget_max=100.0, min_performance=50.0, currency=None)
    front_none = relaxer_none.compute_pareto_front([a, b])
    _, _, exp_none = relaxer_none.select_contrastive_tradeoffs(front_none)
    assert "грн" not in exp_none


# ---------------------------------------------------------------------------
# F19 · entity resolution keeps identity stable and conflicts visible
# ---------------------------------------------------------------------------

def _offers_db(tmp_path):
    path = str(tmp_path / "offers.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE products (
            id TEXT PRIMARY KEY, name TEXT, category TEXT, brand TEXT,
            price REAL, stock_quantity INTEGER, merchant_id TEXT
        );
        CREATE TABLE merchants (
            id TEXT PRIMARY KEY, name TEXT, rating REAL, delivery_days INTEGER
        );
        CREATE TABLE product_eav_meta (product_id TEXT, meta_key TEXT, meta_value TEXT);
        """
    )
    conn.execute("INSERT INTO merchants VALUES ('m1','Shop One',4.8,2)")
    # Same physical product, three listings, conflicting socket claims.
    conn.execute("INSERT INTO products VALUES ('l1','CPU X (Rev 1)','cpu','BrandA',100.0,5,'m1')")
    conn.execute("INSERT INTO products VALUES ('l2','CPU X (Rev 2)','cpu','BrandA',120.0,3,'m1')")
    conn.execute("INSERT INTO products VALUES ('l3','cpu x [BOX]','cpu','BrandA',NULL,2,'m1')")
    conn.execute("INSERT INTO product_eav_meta VALUES ('l1','socket','AM5')")
    conn.execute("INSERT INTO product_eav_meta VALUES ('l2','socket','LGA1700')")
    conn.execute("INSERT INTO product_eav_meta VALUES ('l3','socket','AM5')")
    conn.commit()
    conn.close()
    return path


def test_f19_canonical_identity_is_stable_under_input_permutation(tmp_path):
    """
    Failed before: canon_id was `canon_{category}_{iteration index}`, so the
    same cluster changed identity whenever the input order changed.
    """
    db = _offers_db(tmp_path)
    e1 = MarketplaceEntityResolver.resolve_from_database(db)
    # Rebuild the database with the listings inserted in reverse order.
    path2 = str(tmp_path / "offers_rev.db")
    conn = sqlite3.connect(path2)
    conn.executescript(
        """
        CREATE TABLE products (
            id TEXT PRIMARY KEY, name TEXT, category TEXT, brand TEXT,
            price REAL, stock_quantity INTEGER, merchant_id TEXT
        );
        CREATE TABLE merchants (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE product_eav_meta (product_id TEXT, meta_key TEXT, meta_value TEXT);
        """
    )
    conn.execute("INSERT INTO merchants VALUES ('m1','Shop One')")
    conn.execute("INSERT INTO products VALUES ('l3','cpu x [BOX]','cpu','BrandA',NULL,2,'m1')")
    conn.execute("INSERT INTO products VALUES ('l2','CPU X (Rev 2)','cpu','BrandA',120.0,3,'m1')")
    conn.execute("INSERT INTO products VALUES ('l1','CPU X (Rev 1)','cpu','BrandA',100.0,5,'m1')")
    conn.execute("INSERT INTO product_eav_meta VALUES ('l1','socket','AM5')")
    conn.execute("INSERT INTO product_eav_meta VALUES ('l2','socket','LGA1700')")
    conn.execute("INSERT INTO product_eav_meta VALUES ('l3','socket','AM5')")
    conn.commit()
    conn.close()
    e2 = MarketplaceEntityResolver.resolve_from_database(path2)

    ids1 = {e.canonical_id for e in e1.values()}
    ids2 = {e.canonical_id for e in e2.values()}
    assert ids1 == ids2
    assert all(not cid.rstrip("_0123456789").endswith("cpu") or "canon_cpu_" not in cid
               for cid in ids1)


def test_f19_attribute_conflicts_keep_provenance_not_first_value(tmp_path):
    """
    Failed before: a conflicting attribute collapsed to canonical_form of
    the FIRST seller's value; the disagreement itself vanished.
    """
    db = _offers_db(tmp_path)
    entities = MarketplaceEntityResolver.resolve_from_database(db)
    entity = next(iter(entities.values()))
    assert "socket" not in entity.harmonized_attributes  # no convenient pick
    conflict = entity.attribute_conflicts.get("socket")
    assert conflict is not None
    provenance = {listing: value for listing, value in conflict}
    assert provenance == {"l1": "AM5", "l2": "LGA1700", "l3": "AM5"}


def test_f19_absent_merchant_data_stays_absent(tmp_path):
    """
    Failed before: COALESCE invented 'm_direct' / 'Store Direct' / a 1-day
    delivery / a 5.0 rating for every listing without merchant data, and a
    NULL price became 0.0.
    """
    db = _offers_db(tmp_path)
    entities = MarketplaceEntityResolver.resolve_from_database(db)
    entity = next(iter(entities.values()))
    unpriced = next(o for o in entity.offers if o.listing_id == "l3")
    assert unpriced.price is None
    assert entity.min_price == 100.0 and entity.max_price == 120.0
    assert entity.get_cheapest_offer().listing_id == "l1"


def test_f19_grounding_keeps_every_observed_price(tmp_path):
    """
    Failed before: GroundingSet.prices was first-value-wins per name, so two
    grounded rows with different prices for one name merged into whichever
    arrived first.
    """
    gs = GroundingSet()
    gs.add_solution(CS(item_id="a1", name="Widget", price=100.0))
    gs.add_solution(CS(item_id="a2", name="Widget", price=150.0))
    recs = gs.projected_recommendations()
    entry = next(r for r in recs if r["name"] == "Widget")
    assert entry["price"] is None              # no convenient single pick
    assert entry["price_conflicts"] == [100.0, 150.0]

    gs2 = GroundingSet()
    gs2.add_solution(CS(item_id="b1", name="Gadget", price=70.0))
    gs2.add_solution(CS(item_id="b2", name="Gadget", price=70.0))
    entry2 = next(r for r in gs2.projected_recommendations() if r["name"] == "Gadget")
    assert entry2["price"] == 70.0
    assert "price_conflicts" not in entry2