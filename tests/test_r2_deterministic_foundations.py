"""
Regression suite: deterministic hypergraph, equality prefiltering, and validation contracts.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hypergraph import (
    EquivalenceConstraint,
    KnowledgeHypergraph,
    ProductNode,
    SpatialBoundingConstraint,
)

# ---------------------------------------------------------------------------
# F13 · the entity resolver's token-overlap branch was dead code
# ---------------------------------------------------------------------------

def _hg_with(*nodes):
    hg = KnowledgeHypergraph()
    for n in nodes:
        hg.add_product(n)
    return hg


def test_f13_token_overlap_branch_resolves_partial_names():
    from server import _resolve_from_hypergraph
    hg = _hg_with(ProductNode("gpu1", "ASUS Dual RTX 4070 Super", "gpu", 1.0, {}))
    # Failed before: the query was tokenised from the space-stripped key, so
    # it collapsed into the single glued token "asusdualrtx4070"; the
    # `len(common) >= 2` overlap requirement was unreachable and only the
    # verbatim full name ever resolved.
    hits = _resolve_from_hypergraph(hg, "asus dual rtx 4070")
    assert [n.id for n in hits] == ["gpu1"]


def test_f13_exact_substring_match_still_works():
    from server import _resolve_from_hypergraph
    hg = _hg_with(ProductNode("p1", "Нітрогліцерин 0.5 мг табл.", "med", 1.0, {}))
    hits = _resolve_from_hypergraph(hg, "Купити Нітрогліцерин 0.5 мг табл.")
    assert [n.id for n in hits] == ["p1"]


def test_f13_unrelated_queries_resolve_nothing():
    from server import _resolve_from_hypergraph
    hg = _hg_with(ProductNode("gpu1", "ASUS Dual RTX 4070 Super", "gpu", 1.0, {}))
    assert _resolve_from_hypergraph(hg, "найдешевша материнська плата") == []


# ---------------------------------------------------------------------------
# F16 · the graph candidate filter was unsound
# ---------------------------------------------------------------------------

def test_f16_spatial_dimension_is_not_an_equality_commitment():
    # P22 counterexample: a 40-unit item must stay reachable after a 30-unit
    # item is selected, because both fit the same 100-unit bound.
    # Failed before: `dimension_attr` was treated as an equality commitment,
    # so every candidate with size != 30 was silently excluded.
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("box", "Коробка", "container", 1.0, {"max_size": 100}))
    kg.add_product(ProductNode("item30", "Річ 30", "item", 1.0, {"size": 30}))
    kg.add_product(ProductNode("item40", "Річ 40", "item", 1.0, {"size": 40}))
    kg.add_constraint(SpatialBoundingConstraint("size", "max_size"))

    additions = {n.id for n in kg.find_compatible_additions({"box", "item30"}, "item")}
    assert "item40" in additions
    # And the slow way agrees:
    assert kg.is_compatible({"box", "item30", "item40"}) is True


def test_f16_constraint_category_scope_is_respected():
    # Failed before: the filter committed values from ANY selected node that
    # carried the key, so an out-of-scope node's attribute vetoed in-scope
    # candidates.
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("cpu1", "CPU AM5", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb1", "MB AM5", "mb", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb2", "MB LGA", "mb", 1.0, {"socket": "LGA1700"}))
    kg.add_product(ProductNode("ram", "RAM", "ram", 1.0, {"socket": "weird"}))
    kg.add_constraint(EquivalenceConstraint("socket", categories=["cpu", "mb"]))

    additions = {n.id for n in kg.find_compatible_additions({"cpu1", "ram"}, "mb")}
    # The RAM node is outside the socket rule's scope: its "weird" socket
    # must not veto motherboard candidates. mb1 (AM5) stays reachable;
    # mb2 (LGA1700) is excluded by the socket rule itself, which genuinely
    # applies to {cpu1, mb2} -- not by the out-of-scope RAM value.
    assert "mb1" in additions
    assert "mb2" not in additions
    # Failed before: the out-of-scope "weird" value was committed globally,
    # so EVERY motherboard candidate was excluded and additions was empty.


def test_f16_candidate_without_the_key_is_not_excluded():
    # Absence of the committed attribute is "unknown", not a violation.
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("cpu1", "CPU AM5", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb1", "MB AM5", "mb", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mystery", "Загадка", "mb", 1.0, {}))
    kg.add_constraint(EquivalenceConstraint("socket"))

    additions = {n.id for n in kg.find_compatible_additions({"cpu1"}, "mb")}
    assert "mystery" in additions


def test_f16_node_replacement_cleans_stale_index_memberships():
    # Failed before: add_product only added, so a re-registered product kept
    # its old category and attribute-value memberships forever.
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("p", "Товар", "old_cat", 1.0, {"x": "1"}))
    kg.add_product(ProductNode("p", "Товар", "new_cat", 1.0, {"x": "2"}))

    assert "p" not in kg._by_category.get("old_cat", set())
    assert "p" in kg._by_category.get("new_cat", set())
    assert "p" not in kg._by_attribute_value.get(("x", "1"), set())
    assert "p" in kg._by_attribute_value.get(("x", "2"), set())


def test_f16_indexed_results_match_exhaustive_validation():
    # Differential check on a small generated set: the indexed path must
    # return exactly what exhaustive per-candidate validation returns.
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("cpu", "CPU", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb_am5", "MB AM5", "mb", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb_lga", "MB LGA", "mb", 1.0, {"socket": "LGA1700"}))
    kg.add_product(ProductNode("mb_bare", "MB bare", "mb", 1.0, {}))
    kg.add_constraint(EquivalenceConstraint("socket"))

    selection = {"cpu"}
    indexed = {n.id for n in kg.find_compatible_additions(selection, "mb")}
    exhaustive = {
        pid for pid in ("mb_am5", "mb_lga", "mb_bare")
        if kg.is_compatible(selection | {pid})
    }
    assert indexed == exhaustive


# ---------------------------------------------------------------------------
# F17 · two resolved product names overrode the user's intent
# ---------------------------------------------------------------------------

def test_f17_compatibility_intent_is_recognised():
    from server import _is_compatibility_question
    assert _is_compatibility_question("Чи сумісні AMD Ryzen 7 7800X3D та MSI B650?")
    assert _is_compatibility_question("Чи можна приймати Аспірин разом з Варфарином?")
    assert _is_compatibility_question("Are these two compatible?")


def test_f17_non_compatibility_intents_do_not_take_the_shortcut():
    from server import _is_compatibility_question
    # Failed before: ANY query with two resolved names entered the
    # deterministic compatibility branch, so a price comparison received a
    # "повністю сумісні" verdict (P25).
    assert not _is_compatibility_question("Порівняй ціни на Ryzen 7 7800X3D та Ryzen 5 7600X")
    assert not _is_compatibility_question("Що краще для ігор, 7800X3D чи 14700K?")
    assert not _is_compatibility_question("Чи є в наявності 7800X3D і 7600X?")


def test_f17_shortcut_requires_evaluative_structure_in_the_graph():
    # A graph with no rules and no hyperedges can only ever answer
    # "unknown", so the shortcut must not fire for it.
    from server import _is_compatibility_question
    hg = _hg_with(
        ProductNode("a", "Альфа", "x", 1.0, {}),
        ProductNode("b", "Бета", "y", 1.0, {}),
    )
    named = len(hg.nodes)
    assert named >= 2
    assert _is_compatibility_question("Чи сумісні Альфа та Бета?")
    # The gate itself: no constraints, no hyperedges -> no shortcut.
    assert not (hg.constraints or hg.hyperedges)


# ---------------------------------------------------------------------------
# F12 · induced correlations are candidates, never physical rules
# ---------------------------------------------------------------------------

def test_f12_candidate_conflict_is_a_distinct_honest_state():
    # Failed before: an induced correlation that conflicted flipped the
    # verdict to "incompatible" as if it were verified physics. Now the
    # conflict is a distinct "candidate_conflict" state: conservative enough
    # to act on (is_compatible is False), never reported as verified.
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("cpu", "CPU", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb", "MB", "mb", 1.0, {"socket": "LGA1700"}))
    c = EquivalenceConstraint("socket")
    c.status = "candidate"
    kg.add_constraint(c)

    res = kg.compatibility_verdict({"cpu", "mb"})
    assert res["verdict"] == "candidate_conflict"
    assert res["candidate_conflicts"], "the candidate rule must still signal"
    assert res["evaluated_rules"] == 0
    assert kg.is_compatible({"cpu", "mb"}) is False


def test_f12_candidate_pass_is_not_evidence_of_compatibility():
    # Failed before: a correlation that happened to hold vouched for
    # compatibility with evaluated_rules >= 1.
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("cpu", "CPU", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb", "MB", "mb", 1.0, {"socket": "AM5"}))
    c = EquivalenceConstraint("socket")
    c.status = "candidate"
    kg.add_constraint(c)

    res = kg.compatibility_verdict({"cpu", "mb"})
    assert res["verdict"] == "unknown"
    assert res["candidate_conflicts"] == []
    assert res["evaluated_rules"] == 0


def test_f12_verified_rules_still_decide():
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("cpu", "CPU", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb", "MB", "mb", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb2", "MB2", "mb", 1.0, {"socket": "LGA1700"}))
    kg.add_constraint(EquivalenceConstraint("socket"))  # default: verified

    ok = kg.compatibility_verdict({"cpu", "mb"})
    assert ok["verdict"] == "compatible"
    assert ok["evaluated_rules"] == 1
    bad = kg.compatibility_verdict({"cpu", "mb2"})
    assert bad["verdict"] == "incompatible"


def test_f12_contract_inductor_marks_its_output_candidate():
    from contract_inductor import ContractInductor
    products = [
        ProductNode("c1", "C1", "cpu", 1.0, {"k_sock": "AM5", "k_draw": "100W"}),
        ProductNode("m1", "M1", "motherboard", 1.0, {"k_sock": "AM5"}),
        ProductNode("c2", "C2", "cpu", 1.0, {"k_sock": "LGA1700", "k_draw": "180W"}),
        ProductNode("p1", "P1", "psu", 1.0, {"k_cap": "700W"}),
    ]
    induced = ContractInductor().induce_contracts_from_products(products)
    assert induced, "the inductor must still discover candidate rules"
    assert all(c.status == "candidate" for c in induced)


def test_f12_candidate_rules_veto_conservatively_but_never_vouch():
    # A candidate rule may conservatively veto (a false "incompatible" is
    # annoying, a false "compatible" is harmful), so the LGA board is not
    # offered as an addition -- but the candidate rule can never make the
    # AM5 pair "compatible": the verdict stays non-committal.
    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("cpu1", "CPU AM5", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb_am5", "MB AM5", "mb", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb_lga", "MB LGA", "mb", 1.0, {"socket": "LGA1700"}))
    c = EquivalenceConstraint("socket")
    c.status = "candidate"
    kg.add_constraint(c)

    additions = {n.id for n in kg.find_compatible_additions({"cpu1"}, "mb")}
    assert additions == {"mb_am5"}
    # And the pass of the candidate rule is not evidence either:
    assert kg.compatibility_verdict({"cpu1", "mb_am5"})["verdict"] == "unknown"


# ---------------------------------------------------------------------------
# F14 · coverage must not depend on browsing history
# ---------------------------------------------------------------------------

def test_f14_resolution_reaches_beyond_the_graph_seed(tmp_path):
    import sqlite3

    from domain_catalog import DomainCatalogRegistry
    from server import _resolve_named_products

    db = tmp_path / "catalog.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL)")
    names = ["Товар Альфа", "Товар Бета", "Товар Гамма", "Товар Дельта",
             "Товар Епсилон"]
    for i, name in enumerate(names, 1):
        conn.execute("INSERT INTO products VALUES (?, ?, ?, ?)",
                     (f"p{i}", name, "misc", float(i * 10)))
    conn.commit()
    conn.close()

    # The graph is seeded with only the first two rows.
    hg = DomainCatalogRegistry.create_database_catalog(str(db), seed_limit=2)
    assert len(hg.nodes) == 2

    # Failed before: resolution ran against graph nodes only, so a product
    # outside the seed sample (and outside whatever the UI had browsed)
    # could not be resolved at all.
    named = _resolve_named_products(hg, str(db), "Розкажи про Товар Епсилон")
    assert [n.id for n in named] == ["p5"]
    # And the lazily loaded node is now a first-class graph member.
    assert "p5" in hg.nodes