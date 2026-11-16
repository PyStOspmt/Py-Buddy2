"""
Targeted Verification Tests for Py-Buddy Mathematical Invariants
Validates all 6 core pillars:
1. Knowledge Hypergraph n-ary compatibility
2. Fuzzy Pareto Empty-Set Relaxation
3. Bayesian Active Inference & Shannon EIG
4. Domain-Agnostic Deep Profiler & EAV Flattening
5. SQL Sanitizer, AST validation & Safe LIMIT injection
6. ReAct Agent Stagnation Detection & Zero-Hallucination Gate
7. Cascaded LLM Router Unit Economics & Provider Handover
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from active_inference import ActiveInferenceEngine, DiagnosticQuestion, UserBeliefState
from catalog_query import GroundingSet
from hypergraph import HyperEdge, KnowledgeHypergraph, ProductNode
from pareto_relaxer import CandidateSolution, FuzzyParetoRelaxer
from profiler import DomainAgnosticProfiler
from react_agent import ReActAgent, SessionState
from router import ModelEndpoint, ModelPoolExhausted, ModelRouter
from sanitizer import SecurityViolationError, SQLSanitizer

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'catalog.db'))

# ---------------------------------------------------------------------------
# Test 1: Knowledge Hypergraph n-ary compatibility
# ---------------------------------------------------------------------------
def test_hypergraph_n_ary_compatibility_invariants():
    kg = KnowledgeHypergraph()
    
    cpu = ProductNode("cpu_1", "AMD Ryzen 7 7800X3D", "cpu", 16000)
    mb_am5 = ProductNode("mb_1", "MSI B650 Gaming Plus", "motherboard", 7000)
    mb_intel = ProductNode("mb_2", "ASUS Prime Z790", "motherboard", 8500)
    ram_ddr5 = ProductNode("ram_1", "Kingston Fury 32GB DDR5", "ram", 4500)
    
    kg.add_product(cpu)
    kg.add_product(mb_am5)
    kg.add_product(mb_intel)
    kg.add_product(ram_ddr5)
    
    am5_edge = HyperEdge("edge_am5", "AM5 Gaming Rig", {"cpu_1", "mb_1", "ram_1"})
    kg.add_hyperedge(am5_edge)
    
    assert kg.is_compatible({"cpu_1", "mb_1"}) is True
    assert kg.is_compatible({"cpu_1", "mb_1", "ram_1"}) is True
    assert kg.is_compatible({"cpu_1", "mb_2"}) is False
    assert kg.is_compatible({"cpu_1", "mb_2", "ram_1"}) is False
    
    compatible_rams = kg.find_compatible_additions({"cpu_1", "mb_1"}, target_category="ram")
    assert len(compatible_rams) == 1
    assert compatible_rams[0].id == "ram_1"

# ---------------------------------------------------------------------------
# Test 2: Fuzzy Pareto Empty-Set Relaxation
# ---------------------------------------------------------------------------
def test_fuzzy_pareto_empty_set_relaxation():
    relaxer = FuzzyParetoRelaxer(budget_max=10000, min_performance=90)
    
    candidates = [
        CandidateSolution("item_1", "EcoLaptop 14", price=9500, performance_score=75, delivery_days=2),
        CandidateSolution("item_2", "ProGamer X", price=11500, performance_score=95, delivery_days=1),
        CandidateSolution("item_3", "BadDeal Old", price=13000, performance_score=60, delivery_days=4),
    ]
    
    pareto_front = relaxer.compute_pareto_front(candidates)
    pareto_ids = {item.item_id for item in pareto_front}
    assert "item_1" in pareto_ids
    assert "item_2" in pareto_ids
    assert "item_3" not in pareto_ids
    
    best_a, best_b, explanation = relaxer.select_contrastive_tradeoffs(pareto_front)
    assert best_a != best_b
    assert "Парето-оптимальні альтернативи" in explanation
    assert "EcoLaptop 14" in explanation
    assert "ProGamer X" in explanation

# ---------------------------------------------------------------------------
# Test 3: Bayesian Active Inference & Shannon Entropy
# ---------------------------------------------------------------------------
def test_active_inference_entropy_and_eig():
    belief = UserBeliefState()
    initial_entropy = belief.calculate_entropy()
    # F22: the entropy is the differential entropy of the Beta posterior
    # (epistemic, nats). A flat uniform belief has exactly zero; the old
    # assertion pinned the outcome-entropy semantics instead.
    assert initial_entropy == pytest.approx(0.0)

    questions = [
        DiagnosticQuestion(
            id="q_budget",
            text="Який пріоритет: суворий бюджет чи максимальна продуктивність?",
            target_dimension="price_sensitivity",
            options={"Суворий бюджет": 0.95, "Бюджет не обмежений": 0.05},
            cognitive_cost=0.8
        ),
        DiagnosticQuestion(
            id="q_aesthetic",
            text="Чи важлива наявність RGB підсвітки?",
            target_dimension="aesthetic_brand_focus",
            options={"Так, обов'язково": 0.75, "Ні, байдуже": 0.25},
            cognitive_cost=1.8
        )
    ]

    engine = ActiveInferenceEngine(questions, lambda_cost=0.02)
    best_q = engine.select_best_question(belief)
    assert best_q is not None
    q_chosen, gain = best_q
    assert q_chosen.id == "q_budget"
    assert gain > 0.0

    # User answers: high price sensitivity
    engine.record_answer(belief, q_chosen, "Суворий бюджет")
    assert belief.dimensions["price_sensitivity"] > 0.6
    # Entropy should decrease as belief concentrates
    assert belief.calculate_entropy() < initial_entropy

# ---------------------------------------------------------------------------
# Test 4: Domain-Agnostic Deep Profiler & EAV Detection
# ---------------------------------------------------------------------------
def test_domain_agnostic_profiler_and_eav_flattening():
    profiler = DomainAgnosticProfiler(DB_PATH)
    snapshot = profiler.profile_database()

    assert "products" in snapshot.tables
    assert "product_meta" in snapshot.tables

    # Check products table profile
    prod_tbl = snapshot.tables["products"]
    assert prod_tbl.row_count > 0
    assert "price" in prod_tbl.columns
    assert prod_tbl.columns["price"].semantic_type == "price"
    assert prod_tbl.columns["price"].min_value is not None

    # Check EAV table detection
    meta_tbl = snapshot.tables["product_meta"]
    assert meta_tbl.is_eav is True
    assert meta_tbl.eav_key_col == "meta_key"
    assert meta_tbl.eav_val_col == "meta_value"
    assert "product_meta" in snapshot.eav_projections

    # Check compact schema prompt generation
    prompt_context = DomainAgnosticProfiler.generate_compact_schema_context(snapshot.to_dict())
    assert "Database Schema Snapshot" in prompt_context
    assert "EAV Virtual Projections" in prompt_context
    assert "product_meta" in prompt_context

# ---------------------------------------------------------------------------
# Test 5: SQL Sanitizer & Security Invariants
# ---------------------------------------------------------------------------
def test_sql_sanitizer_security_and_limit_invariants():
    sanitizer = SQLSanitizer(default_limit=15, max_limit=40)

    # Valid query gets default limit
    q1 = "SELECT id, name, price FROM products WHERE price < 15000"
    sanitized1 = sanitizer.sanitize_and_prepare_query(q1)
    assert "LIMIT 15" in sanitized1

    # Overly large limit gets clamped
    q2 = "SELECT * FROM products LIMIT 100"
    sanitized2 = sanitizer.sanitize_and_prepare_query(q2)
    assert "LIMIT 40" in sanitized2

    # Malicious queries MUST be blocked
    with pytest.raises(SecurityViolationError):
        sanitizer.sanitize_and_prepare_query("DROP TABLE products")

    with pytest.raises(SecurityViolationError):
        sanitizer.sanitize_and_prepare_query("UPDATE products SET price = 0")

    with pytest.raises(SecurityViolationError):
        sanitizer.sanitize_and_prepare_query("SELECT * FROM products; DELETE FROM products")

    # Safe execution against DB
    res = sanitizer.execute_safe_query(DB_PATH, "SELECT name, price FROM products WHERE category = 'cpu'")
    assert res["success"] is True
    assert res["row_count"] > 0
    assert len(res["rows"]) <= 15

# ---------------------------------------------------------------------------
# Test 6: ReAct Agent Stagnation & Zero-Hallucination Gate
# ---------------------------------------------------------------------------
def test_react_agent_stagnation_and_zero_hallucination():
    agent = ReActAgent(DB_PATH)
    state = SessionState(session_id="sess_test_01", user_query="Знайди мені процесор")

    # Step 1: Execute SQL tool
    res1 = agent.execute_tool(state, "sql_query", {"query": "SELECT id, name, price FROM products WHERE category = 'cpu'"})
    assert res1["success"] is True
    assert len(state.execution_history) == 1

    # Step 2: Attempt duplicate tool call (identical args) -> Stagnation detection
    res2 = agent.execute_tool(state, "sql_query", {"query": "SELECT id, name, price FROM products WHERE category = 'cpu'"})
    assert state.is_terminated is True
    assert state.termination_reason == "STAGNATION_DETECTED"

    # Test the Zero-Hallucination Gate (ADR-013, non-vacuous).
    # 1. Honest recommendation, grounded in the executed SELECT.
    grounded_claim = [{"id": "cpu_am5_1", "name": "AMD Ryzen 7 7800X3D", "price": 16500.0}]
    ok, text = agent.finalize_response(state, "Рекомендую процесор AMD Ryzen 7 7800X3D.", grounded_claim)
    assert ok is True
    assert state.termination_reason == "SUCCESS_GROUNDED"

    # 2. Fabricated item that was never returned by any query.
    #    The gate must NOT pass it through, and — critically — must not
    #    answer the customer with a technical error. Since real evidence
    #    exists, it answers from the grounded set instead.
    agent.grounding = GroundingSet()
    state_h = SessionState(session_id="sess_test_02", user_query="Порадь процесор")
    state_h.execution_history = state.execution_history
    hallucinated_claim = [{"id": "fake_cpu_99", "name": "SuperNova Quantum 9999",
                           "price": 50000.0}]
    ok_h, text_h = agent.finalize_response(state_h, "Рекомендую квантовий процесор.",
                                          hallucinated_claim)
    assert "SuperNova Quantum 9999" not in text_h, "fabrication must never reach the client"
    assert "Помилка" not in text_h, "no technical error to the customer"
    if agent.grounding.names:
        assert ok_h is True
        assert state_h.termination_reason == "SUCCESS_GROUNDED_AFTER_CORRECTION"
    else:
        # Nothing was ever grounded: a genuine block is correct, and it
        # must still read as an answer rather than a crash.
        assert ok_h is False
        assert state_h.termination_reason == "HALLUCINATION_GATE_VIOLATION"
        assert "Уточніть" in text_h

# ---------------------------------------------------------------------------
# Test 7: Model router -- selection, economics, handover
# ---------------------------------------------------------------------------
def test_model_router_selects_by_tier_and_never_sees_the_query():
    """The model router answers "which model", not "which tier".

    It used to carry its own copy of the intent classifier -- a phrase list
    that disagreed with `fast_router`'s. The split is the point of the
    rewrite, so the test asserts the absence of the old API.
    """
    pool = [
        ModelEndpoint("cheap-1", tier=1, cost_per_1k_input=0.0001, cost_per_1k_output=0.0002),
        ModelEndpoint("reasoner-1", tier=2, cost_per_1k_input=0.003, cost_per_1k_output=0.015),
    ]
    router = ModelRouter(pool)

    fast = router.select(1, reason="routine")
    deep = router.select(2, reason="conflict")
    assert fast.endpoint.model_name == "cheap-1"
    assert deep.endpoint.model_name == "reasoner-1"
    assert fast.estimated_cost_usd < deep.estimated_cost_usd
    assert not hasattr(router, "assess_complexity")


def test_model_router_economics_report_a_stated_baseline():
    pool = [
        ModelEndpoint("cheap-1", tier=1, cost_per_1k_input=0.0, cost_per_1k_output=0.0),
        ModelEndpoint("reasoner-1", tier=2, cost_per_1k_input=0.0, cost_per_1k_output=0.0),
    ]
    router = ModelRouter(pool)
    for _ in range(8):
        router.note_call(router.select(1))
    for _ in range(2):
        router.note_call(router.select(2))

    report = router.economics(baseline_cost_per_call_usd=0.007)
    assert report["total_calls"] == 10
    assert report["calls_by_tier"] == {1: 8, 2: 2}
    assert report["baseline_cost_usd"] == pytest.approx(0.07)
    assert report["savings_percent"] == 100.0        # a free pool beats a paid one
    assert "baseline_cost_per_call_usd" in report    # and the number is visible


def test_model_router_handover_degrades_then_says_it_is_out():
    pool = [ModelEndpoint("a", tier=1), ModelEndpoint("b", tier=1)]
    router = ModelRouter(pool)

    first = router.report_failure("a", 429)
    assert first.model_name == "b"          # one failure: still usable
    assert router.report_failure("a", 503) is not None
    assert not pool[0].is_active             # two: out of rotation

    # b is still serving, so there is nowhere to hand over -- the honest
    # answer is None, not "here is b again".
    assert router.report_failure("b", 500) is None
    assert pool[1].is_active
    assert router.report_failure("b", 429) is None
    with pytest.raises(ModelPoolExhausted):
        router.select(1)

    assert router.recover()                  # explicit, and it says what it did
    assert set(router.recover()) == set()    # and idempotent
    assert router.select(1).endpoint.is_active
