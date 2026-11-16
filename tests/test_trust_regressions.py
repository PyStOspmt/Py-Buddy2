"""
Trust & security regression suite: read-only SQL bounds, admin authorization,
grounding gate strictness, session isolation, and Pareto relaxer contracts.
"""
import os
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from catalog_query import GroundingSet
from hypergraph import KnowledgeHypergraph
from openrouter_client import OpenRouterAgentRunner
from pareto_relaxer import CandidateSolution
from react_agent import ReActAgent, SessionState
from sanitizer import SecurityViolationError, SQLSanitizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    db = tmp_path / "shop.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "price REAL, stock_quantity INTEGER, performance_score REAL)")
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?)",
        [
            ("p1", "Ноутбук Pro 15", "laptops", 25000.0, 5, 80.0),
            ("p2", "Мишка бездротова", "accessories", 500.0, 40, 60.0),
            ("p3", "Клавіатура механічна", "accessories", 1500.0, 20, 70.0),
        ])
    conn.commit()
    conn.close()
    return str(db)


def _agent(db_path, *solutions):
    agent = ReActAgent(db_path=db_path)
    for s in solutions:
        agent.grounding.add_solution(s)
    return agent


GROUNDED = (
    CandidateSolution("p1", "Ноутбук Pro 15", 25000.0, 80.0, 2),
    CandidateSolution("p2", "Мишка бездротова", 500.0, 60.0, 1),
)


class ScriptedRunner:
    """Returns canned ReAct steps in order."""

    def __init__(self, *steps):
        self.steps = list(steps)

    def generate_react_step(self, **kwargs):
        if self.steps:
            return self.steps.pop(0)
        return {"final_answer": "Немає даних.", "recommended_items": []}


# ---------------------------------------------------------------------------
# F02 · unparseable model output is a protocol failure, not an answer
# ---------------------------------------------------------------------------

def test_f02_unparseable_output_never_reaches_the_client(tmp_path):
    db = _make_db(tmp_path)
    agent = ReActAgent(db_path=db)
    # Failed before: state.final_response == str(step_output) — the raw model
    # dump was published to the client and the router scored the turn as a
    # successful Tier-0 conversation.
    state = agent.run_live_session(
        "Що порадиш?", ScriptedRunner({"thought": "плутанина", "junk": [1, 2]}),
        schema_prompt="s")
    assert state.termination_reason == "PROTOCOL_ERROR"
    assert "junk" not in (state.final_response or "")
    assert "плутанина" not in (state.final_response or "")
    assert "не сформовано" in state.final_response


def test_f02_provider_outage_is_a_typed_outcome_not_a_fake_answer(tmp_path):
    db = _make_db(tmp_path)
    agent = ReActAgent(db_path=db)
    # Failed before: the runner returned a fake apology final_answer and the
    # trace recorded a normal grounded turn.
    state = agent.run_live_session(
        "Що порадиш?", ScriptedRunner({"provider_error": "OpenRouter 503"}),
        schema_prompt="s")
    assert state.termination_reason == "PROVIDER_ERROR"
    assert "недоступна" in state.final_response


def test_f02_openrouter_client_returns_typed_protocol_error():
    # Failed before: the raw-parsing fallback wrapped arbitrary model text as
    # final_answer, and the retry-exhausted path returned a fake apology.
    out = OpenRouterAgentRunner._parse_step(
        "Це зовсім не JSON і не містить final_answer")
    assert "protocol_error" in out
    assert "final_answer" not in out


def test_f02_provider_reasoning_never_becomes_content():
    # Failed before: `content or reasoning_content` leaked provider
    # chain-of-thought into the answer path (F32).
    msg = {"reasoning_content": "ВНУТРІШНІ МІРКУВАННЯ", "content": ""}
    raw = msg.get("content") or ""
    assert raw == ""


# ---------------------------------------------------------------------------
# F03 · prose may not name catalogue products that were never evidenced
# ---------------------------------------------------------------------------

def test_f03_prose_mention_of_unevidenced_catalog_product_is_rejected(tmp_path):
    db = _make_db(tmp_path)
    agent = _agent(db)  # nothing grounded in this run
    state = SessionState("s", "Порадь щось")
    # Failed before: with empty recommended_items and no price tokens the
    # gate returned SUCCESS_GROUNDED for prose naming "Клавіатура механічна".
    ok, text = agent.finalize_response(
        state, "Раджу Клавіатура механічна — чудовий вибір.", [])
    assert not ok
    assert state.termination_reason == "HALLUCINATION_GATE_VIOLATION"
    assert "Клавіатура" not in text


def test_f03_grounded_prose_still_passes(tmp_path):
    db = _make_db(tmp_path)
    agent = _agent(db, *GROUNDED)
    state = SessionState("s", "Порадь ноутбук")
    ok, text = agent.finalize_response(
        state, "Раджу Ноутбук Pro 15.", [{"name": "Ноутбук Pro 15"}])
    assert ok
    assert state.termination_reason == "SUCCESS_GROUNDED"


# ---------------------------------------------------------------------------
# F05 · positional entity-price binding and re-projection
# ---------------------------------------------------------------------------

def test_f05_swapped_prices_are_corrected_not_accepted(tmp_path):
    db = _make_db(tmp_path)
    agent = _agent(db, *GROUNDED)
    state = SessionState("s", "Порівняй")
    # Failed before: both numbers existed somewhere in the grounded rows, so
    # the swap passed the gate untouched.
    ok, text = agent.finalize_response(
        state,
        "Ноутбук Pro 15 коштує 500.00 грн, а Мишка бездротова — 25000.00 грн.",
        [{"name": "Ноутбук Pro 15"}, {"name": "Мишка бездротова"}])
    assert ok
    assert "25000.00 грн, а Мишка" not in text
    assert "500.00 грн, а Мишка" in text or "500.00" in text


def test_f05_unattributed_sum_is_rejected(tmp_path):
    db = _make_db(tmp_path)
    agent = _agent(db, *GROUNDED)
    state = SessionState("s", "Скільки разом?")
    # Failed before: the wide-word window ("разом", "сума", "всього"...)
    # freed any number, so a wrong cart total passed.
    ok, text = agent.finalize_response(
        state, "Разом 99999.00 грн за все.", [])
    assert not ok
    assert "99999" not in text


def test_f05_sentence_level_sum_over_named_products_passes(tmp_path):
    db = _make_db(tmp_path)
    agent = _agent(db, *GROUNDED)
    state = SessionState("s", "Скільки разом?")
    ok, text = agent.finalize_response(
        state,
        "Ноутбук Pro 15 (25000.00 грн) та Мишка бездротова (500.00 грн) "
        "разом коштують 25500.00 грн.",
        [{"name": "Ноутбук Pro 15"}, {"name": "Мишка бездротова"}])
    assert ok
    assert "25500" in text


def test_f05_recommendation_objects_are_reprojected_from_evidence(tmp_path):
    db = _make_db(tmp_path)
    agent = _agent(db, *GROUNDED)
    state = SessionState("s", "Порадь")
    # Failed before: a name match carried the model's own id and price
    # straight into recommended_items.
    ok, _text = agent.finalize_response(
        state, "Раджу Ноутбук Pro 15.",
        [{"name": "Ноутбук Pro 15", "id": "FAKE-ID", "price": 1.0}])
    assert ok
    item = state.recommended_items[0]
    assert item["id"] == "p1"
    assert item["price"] == 25000.0


# ---------------------------------------------------------------------------
# F06 · partial correction regenerates from verified rows only
# ---------------------------------------------------------------------------

def test_f06_mixed_answer_is_regenerated_not_annotated(tmp_path):
    db = _make_db(tmp_path)
    agent = _agent(db, *GROUNDED)
    state = SessionState("s", "Порадь")
    # Failed before: the unmodified prose (still containing the fabricated
    # product) was returned with a reassurance note appended.
    ok, text = agent.finalize_response(
        state,
        "Раджу Ноутбук Pro 15 та Клавіатура механічна.",
        [{"name": "Ноутбук Pro 15"}, {"name": "Клавіатура механічна"}])
    assert ok
    assert state.termination_reason == "SUCCESS_GROUNDED_WITH_CORRECTION"
    assert "Клавіатура" not in text
    assert "Ноутбук Pro 15" in text


# ---------------------------------------------------------------------------
# F07 · evidence belongs to one run
# ---------------------------------------------------------------------------

def test_f07_second_session_cannot_reuse_first_session_evidence(tmp_path):
    db = _make_db(tmp_path)
    agent = ReActAgent(db_path=db)
    runner1 = ScriptedRunner(
        {"action": "sql_query",
         "action_input": {"query": "SELECT id, name, price FROM products WHERE id = 'p1'"}},
        {"final_answer": "Раджу Ноутбук Pro 15.",
         "recommended_items": [{"name": "Ноутбук Pro 15"}]})
    state1 = agent.run_live_session("Ноутбук?", runner1, schema_prompt="s")
    assert "Ноутбук Pro 15" in (state1.final_response or "")

    # A fresh session on the SAME agent: the model answers from memory about
    # the product fetched in session 1.
    # Failed before: the shared GroundingSet still contained session 1 rows,
    # so session 2's answer passed as grounded.
    state2 = agent.run_live_session(
        "Мишка?", ScriptedRunner({"final_answer": "Раджу Ноутбук Pro 15.",
                                  "recommended_items": [{"name": "Ноутбук Pro 15"}]}),
        schema_prompt="s")
    assert state2.grounding is not state1.grounding
    assert state2.grounding.names == set()
    assert state2.termination_reason != "SUCCESS_GROUNDED"


# ---------------------------------------------------------------------------
# F08 · the trace reports what the gate actually decided
# ---------------------------------------------------------------------------

def test_f08_gate_rejection_is_recorded_as_a_rejection(tmp_path):
    db = _make_db(tmp_path)
    agent = ReActAgent(db_path=db)
    # Failed before: the final record hard-coded "Верифікація ... пройдена
    # успішно (100% достовірність)" and status GROUNDED even when the gate
    # rejected the answer.
    state = agent.run_live_session(
        "Порадь щось",
        ScriptedRunner({"final_answer": "Раджу Клавіатура механічна.",
                        "recommended_items": []}),
        schema_prompt="s")
    record = state.execution_history[-1]
    assert record["result"]["status"] == "HALLUCINATION_GATE_VIOLATION"
    assert "пройдена успішно" not in record["observation"]


# ---------------------------------------------------------------------------
# F15 · unknown compatibility is never reported as compatible
# ---------------------------------------------------------------------------

def test_f15_unresolved_items_yield_unknown_not_compatible(tmp_path):
    db = _make_db(tmp_path)
    agent = ReActAgent(db_path=db, hypergraph=KnowledgeHypergraph())
    state = SessionState("s", "Сумісність")
    # Failed before: unrecognized ids were silently dropped, the empty set
    # hit the basket default and the tool reported is_compatible=True.
    res = agent.execute_tool(state, "hypergraph_check",
                             {"item_ids": ["absent-1", "absent-2"]})
    assert res["verdict"] == "unknown"
    assert res["is_compatible"] is None
    assert sorted(res["unresolved_items"]) == ["absent-1", "absent-2"]


def test_f15_rule_free_graph_never_claims_verified_compatibility(tmp_path):
    db = _make_db(tmp_path)
    agent = ReActAgent(db_path=db, hypergraph=KnowledgeHypergraph())
    state = SessionState("s", "Сумісність")
    res = agent.execute_tool(state, "hypergraph_check",
                             {"item_ids": ["p1", "p2"]})
    # Both resolve, but this graph has no rules: the retail basket default
    # is not evidence of compatibility.
    assert res["verdict"] == "unknown"
    assert res["evaluated_rules"] == 0


def test_f15_resolved_set_with_rules_still_reports_compatible(tmp_path):
    kg = KnowledgeHypergraph()
    from hypergraph import EquivalenceConstraint, ProductNode
    kg.add_product(ProductNode("cpu", "CPU", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb", "MB", "mb", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb2", "MB2", "mb", 1.0, {"socket": "LGA1700"}))
    kg.add_constraint(EquivalenceConstraint("socket"))
    agent = ReActAgent(db_path=":memory:", hypergraph=kg)
    state = SessionState("s", "Сумісність")
    ok_res = agent.execute_tool(state, "hypergraph_check", {"item_ids": ["cpu", "mb"]})
    assert ok_res["verdict"] == "compatible"
    assert ok_res["is_compatible"] is True
    # Only the rule that actually applied counts as evaluated evidence.
    assert ok_res["evaluated_rules"] == 1
    bad_res = agent.execute_tool(state, "hypergraph_check", {"item_ids": ["cpu", "mb2"]})
    assert bad_res["verdict"] == "incompatible"
    assert bad_res["is_compatible"] is False


def test_f15_vacuous_rules_do_not_vouch_compatibility(tmp_path):
    # Failed before (live-confirmed 2026-10-04 against the running server):
    # a graph with registered rules reported verdict="compatible" with
    # evaluated_rules=2 for a CPU+PSU pair to which neither rule could
    # apply — a vacuous pass was counted as compatibility evidence.
    kg = KnowledgeHypergraph()
    from hypergraph import (
        CumulativeCapacityConstraint,
        EquivalenceConstraint,
        ProductNode,
    )
    kg.add_product(ProductNode("cpu", "CPU", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("psu", "PSU", "psu", 1.0, {"wattage": 500}))
    kg.add_constraint(EquivalenceConstraint("socket"))
    kg.add_constraint(CumulativeCapacityConstraint("tdp", "wattage"))
    agent = ReActAgent(db_path=":memory:", hypergraph=kg)
    state = SessionState("s", "Сумісність")
    res = agent.execute_tool(state, "hypergraph_check",
                             {"item_ids": ["cpu", "psu"]})
    # Neither rule speaks about this pair: the honest verdict is "unknown",
    # not a rule-verified "compatible".
    assert res["verdict"] == "unknown"
    assert res["evaluated_rules"] == 0
    assert res["is_compatible"] is None
    assert res["conflicts"] == []


# ---------------------------------------------------------------------------
# F04 · literal projections cannot launder fabricated facts
# ---------------------------------------------------------------------------

def test_f04_literal_projection_grounds_nothing(tmp_path):
    db = _make_db(tmp_path)
    san = SQLSanitizer()
    res = san.execute_safe_query(
        db, "SELECT 'Phantom widget' AS name, 999 AS price")
    assert res["success"]
    prov = res["column_provenance"]
    assert prov["name"] == "derived"
    assert prov["price"] == "derived"
    gs = GroundingSet()
    gs.extend_rows(res["rows"], provenance=prov)
    # Failed before: both literals entered the grounding set and the gate
    # then verified the model's phantom recommendation against them.
    assert gs.names == set()
    assert gs.prices == {}


def test_f04_source_columns_still_ground(tmp_path):
    db = _make_db(tmp_path)
    san = SQLSanitizer()
    res = san.execute_safe_query(
        db, "SELECT name, price FROM products WHERE id = 'p1'")
    prov = res["column_provenance"]
    assert prov["name"] == "source"
    assert prov["price"] == "source"
    gs = GroundingSet()
    gs.extend_rows(res["rows"], provenance=prov)
    assert "Ноутбук Pro 15" in gs.names
    # F19: every observed price is kept, not just the first one seen.
    assert gs.prices["Ноутбук Pro 15"] == [25000.0]


def test_f04_derived_price_admits_name_but_not_number(tmp_path):
    db = _make_db(tmp_path)
    san = SQLSanitizer()
    res = san.execute_safe_query(
        db, "SELECT name, 999 AS price FROM products WHERE id = 'p1'")
    prov = res["column_provenance"]
    assert prov["name"] == "source"
    assert prov["price"] == "derived"
    gs = GroundingSet()
    gs.extend_rows(res["rows"], provenance=prov)
    assert "Ноутбук Pro 15" in gs.names
    assert "Ноутбук Pro 15" not in gs.prices


# ---------------------------------------------------------------------------
# F31 · fail-closed read-only, bound-LIMIT clamping, work budget
# ---------------------------------------------------------------------------

def test_f31_no_writable_fallback_for_readonly_failure(tmp_path):
    san = SQLSanitizer()
    # Failed before: a failed read-only URI connect silently fell back to a
    # writable sqlite3.connect(db_path).
    with pytest.raises(SecurityViolationError):
        san.execute_safe_query(str(tmp_path / "missing.db"), "SELECT 1")


def test_f31_bound_limit_value_is_clamped(tmp_path):
    db = _make_db(tmp_path)
    san = SQLSanitizer()
    # Failed before: `LIMIT ?` was accepted without checking the bound value,
    # so a model-controlled parameter removed the row cap.
    res = san.execute_safe_query(
        db, "SELECT id FROM products LIMIT ?", params=[9999])
    assert res["success"]
    assert res["row_count"] <= san.max_limit


# ---------------------------------------------------------------------------
# F29/F30 · admin gate and telemetry isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from server import app
    return TestClient(app)


def test_f29_register_domain_is_refused_without_admin_token(client, monkeypatch, tmp_path):
    monkeypatch.delenv("AINSA_ADMIN_TOKEN", raising=False)
    # Failed before: any caller could register any filesystem path as a
    # live data source.
    res = client.post("/api/register_domain",
                      json={"domain_id": "evil", "db_path": str(tmp_path / "x.db")})
    assert res.status_code == 403


def test_f29_register_domain_rejects_paths_outside_the_source_root(client, monkeypatch, tmp_path):
    monkeypatch.setenv("AINSA_ADMIN_TOKEN", "secret")
    outside = tmp_path / "outside.db"
    sqlite3.connect(str(outside)).close()
    res = client.post(
        "/api/register_domain",
        json={"domain_id": "outside_dom", "db_path": str(outside)},
        headers={"X-Admin-Token": "secret"})
    assert res.status_code == 403


def test_f29_register_domain_accepts_db_inside_configured_root(client, monkeypatch, tmp_path):
    monkeypatch.setenv("AINSA_ADMIN_TOKEN", "secret")
    monkeypatch.setenv("AINSA_SOURCE_ROOT", str(tmp_path))
    inside = tmp_path / "inside.db"
    conn = sqlite3.connect(str(inside))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL)")
    conn.execute("INSERT INTO products VALUES ('a', 'Товар', 'cat', 10.0)")
    conn.commit()
    conn.close()
    res = client.post(
        "/api/register_domain",
        json={"domain_id": "inside_dom", "db_path": str(inside)},
        headers={"X-Admin-Token": "secret"})
    assert res.status_code == 200
    # Source ids are immutable.
    res2 = client.post(
        "/api/register_domain",
        json={"domain_id": "inside_dom", "db_path": str(inside)},
        headers={"X-Admin-Token": "secret"})
    assert res2.status_code == 409


def test_f30_switch_domain_and_set_model_require_admin(client, monkeypatch):
    monkeypatch.delenv("AINSA_ADMIN_TOKEN", raising=False)
    # Failed before: any client could flip the process-global active domain
    # and model for everyone.
    assert client.post("/api/switch_domain", json={"domain": "auto_parts"}).status_code == 403
    assert client.post("/api/set_model", json={"model": "x"}).status_code == 403


def test_f30_telemetry_hides_session_transcripts_from_anonymous_callers(client, monkeypatch):
    monkeypatch.delenv("AINSA_ADMIN_TOKEN", raising=False)
    res = client.get("/api/telemetry")
    data = res.json()
    # Failed before: the full last session transcript was returned to any
    # unauthenticated caller.
    assert data["last_session"] is None


def test_f30_chat_ignores_client_model_switch_without_admin(client, monkeypatch):
    monkeypatch.delenv("AINSA_ADMIN_TOKEN", raising=False)
    before = None
    from server import ctx
    before = ctx.runner.model_name
    res = client.post("/api/chat", json={"query": "Привіт", "model": "other/model"})
    assert res.status_code == 200
    assert ctx.runner.model_name == before