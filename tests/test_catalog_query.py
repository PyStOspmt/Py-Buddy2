"""
Tests for ADR-012 and ADR-013 — the two holes the audit found in the
deterministic core.

AUD-05: `pareto_relax` used to build its candidates from the LLM's own
        arguments, so a fabricated product entered the "deterministic"
        layer and the gate could not see it.
AUD-27: when the gate rejected an answer it returned a technical error to
        the customer instead of a corrected, non-vacuous answer.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from catalog_query import GroundingSet, fetch_by_ids, fetch_candidates
from pareto_relaxer import CandidateSolution

DB = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "marketplace_rozetka.db",
))
needs_db = pytest.mark.skipif(
    not os.path.exists(DB),
    reason="marketplace catalog not generated; run data/generate_marketplace_scale.py",
)


# --------------------------------------------------------------------------
# fetch_candidates — entities come from SQL, not from arguments
# --------------------------------------------------------------------------

@needs_db
def test_candidates_are_real_rows():
    cands = fetch_candidates(DB, category="cpu", budget_max=20000)
    assert cands, "no candidates for a populated category"
    for c in cands:
        assert c.item_id and c.name
        assert c.price > 0
        assert c.performance_score > 0
        assert c.delivery_days >= 0


@needs_db
def test_candidate_ids_exist_in_the_catalogue():
    import sqlite3
    cands = fetch_candidates(DB, category="gpu", budget_max=60000, limit=10)
    conn = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
    ids = {r[0] for r in conn.execute("SELECT id FROM products")}
    conn.close()
    assert cands
    assert all(c.item_id in ids for c in cands)


@needs_db
def test_candidates_respect_the_category():
    import sqlite3
    cands = fetch_candidates(DB, category="motherboard", budget_max=40000)
    conn = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
    cats = {r[0] for r in conn.execute(
        "SELECT category FROM products WHERE id IN (%s)"
        % ",".join("?" * len(cands)), [c.item_id for c in cands])}
    conn.close()
    assert cats == {"motherboard"}


@needs_db
def test_budget_relaxation_is_bounded_and_visible():
    """
    A budget that excludes everything is relaxed by exactly one 1.5x step,
    and the relaxation is visible in the returned prices.

    The case that matters is the "empty-set deadlock": a strict budget that
    matches no row at all. There the relaxed query MUST return something,
    and every row it returns must cost MORE than the stated budget — that
    is what makes the relaxation legible to the user instead of a silent
    rewrite. The 1.5x cap is also pinned, so raising the factor to 10 to
    make life easy fails this test.
    """
    cheapest = min(c.price for c in fetch_candidates(DB, category="cpu"))
    budget = round(cheapest * 0.99, 2)          # strict query is now empty

    relaxed = fetch_candidates(DB, category="cpu", budget_max=budget)
    assert relaxed, "1.5x relaxation must rescue an empty strict query"
    assert all(c.price > budget for c in relaxed), \
        "relaxed rows must be visibly above the stated budget"
    cap = round(budget * 1.5, 2) + 0.01
    assert all(c.price <= cap for c in relaxed), "relaxation exceeded its bound"
    assert len(relaxed) <= 40


@needs_db
def test_impossible_budget_returns_nothing_in_the_bounded_step():
    """The escalation is explicit, not hidden inside the same query."""
    assert fetch_candidates(DB, category="cpu", budget_max=1.0) == []
    escalated = fetch_candidates(DB, category="cpu", budget_max=None)
    assert escalated, "the caller must be able to escalate explicitly"


@needs_db
def test_fetch_by_ids_returns_only_requested_rows():
    cands = fetch_candidates(DB, category="ram", budget_max=9000, limit=3)
    ids = [c.item_id for c in cands]
    got = fetch_by_ids(DB, ids)
    assert {c.item_id for c in got} == set(ids)
    assert fetch_by_ids(DB, []) == []


# --------------------------------------------------------------------------
# GroundingSet
# --------------------------------------------------------------------------

def test_grounding_set_from_solutions():
    g = GroundingSet()
    g.extend([
        CandidateSolution("p1", "AMD Ryzen 5 7600X", 9500.0, 82.0, 2),
        CandidateSolution("p2", "MSI B650", 6800.0, 85.0, 1),
    ])
    assert g.names == {"AMD Ryzen 5 7600X", "MSI B650"}
    # F19: every observed price is kept, not just the first one seen.
    assert g.prices["AMD Ryzen 5 7600X"] == [9500.0]
    assert g.ids == {"p1", "p2"}


def test_grounding_set_from_rows():
    g = GroundingSet()
    g.extend_rows([
        {"id": "x1", "name": "DeepCool PF500 500W", "price": 1700.0},
        {"id": "x2", "name": "Seasonic Focus GX-1000", "price": "8200.0"},
        {"no_name": True},
        "not a dict",
    ])
    assert g.names == {"DeepCool PF500 500W", "Seasonic Focus GX-1000"}
    assert g.prices["Seasonic Focus GX-1000"] == [8200.0]


def test_grounding_set_keeps_every_observed_price():
    """
    F19: a conflicting price is recorded, not overwritten. Failed before:
    the first-value-wins dict silently discarded the second observation.
    """
    g = GroundingSet()
    g.extend_rows([{"id": "x1", "name": "A", "price": 100.0}])
    g.extend_rows([{"id": "x2", "name": "A", "price": 999.0}])
    assert g.prices["A"] == [100.0, 999.0]
    projected = g.projected_recommendations()
    entry = next(r for r in projected if r["name"] == "A")
    assert entry["price"] is None
    assert entry["price_conflicts"] == [100.0, 999.0]


def test_projected_recommendations_only_contains_grounded_rows():
    g = GroundingSet()
    g.extend([CandidateSolution("p1", "Real Item", 100.0, 80.0, 1)])
    projected = g.projected_recommendations()
    assert projected == [{"id": "p1", "name": "Real Item", "price": 100.0}]
    assert "price_conflicts" not in projected[0]


def test_empty_grounding_set_projects_nothing():
    assert GroundingSet().projected_recommendations() == []
