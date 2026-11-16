"""
TDD Suite for Py-Buddy Fast Cascaded Router (src/fast_router.py).
Validates <100ms Tier-1 direct query interception vs Tier-2 Neuro-Symbolic ReAct delegation.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fast_router import ExecutionTier, FastCascadedRouter
from hypergraph import KnowledgeHypergraph
from react_agent import ReActAgent
from sanitizer import SQLSanitizer

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'catalog.db'))

class MockRunner:
    """Mock runner to test Tier-0 conversational vs Tier-2 ReAct delegation without live network overhead."""
    def generate_react_step(self, user_query, dialog_history=None, **kwargs):
        q_l = user_query.lower()
        if "привіт" in q_l:
            return {"action": None, "final_answer": "Вітаю! Я інтелектуальний AI-консультант. Чим можу допомогти?"}
        if "хто ти" in q_l:
            return {"action": None, "final_answer": "Я інтелектуальний консультант Py-Buddy на основі віртуального гіперграфа."}
        if "дякую" in q_l:
            return {"action": None, "final_answer": "Завжди радий допомогти! Звертайтеся."}
        
        # Multi-step tool invocation
        if not dialog_history:
            return {"action": "sql_query", "action_input": {"query": "SELECT * FROM products LIMIT 1;"}}
        return {"action": None, "final_answer": "Сумісність деталей перевірена успішно."}

@pytest.fixture
def router():
    sanitizer = SQLSanitizer()
    kg = KnowledgeHypergraph()
    agent = ReActAgent(db_path=DB_PATH, hypergraph=kg)
    return FastCascadedRouter(
        db_path=DB_PATH,
        sanitizer=sanitizer,
        react_agent=agent,
        llm_runner=MockRunner(),
        schema_prompt="Schema Context"
    )

def test_tier1_price_query_interception(router):
    """F49: the fast path fires only when every content token is
    explainable -- superlative + a catalogue value + closed-class words."""
    query = "Яка точна ціна найдешевшого AMD?"
    t0 = time.time()
    result = router.route_and_execute(query)
    elapsed = time.time() - t0

    assert result["tier"] == ExecutionTier.TIER1_FAST_PATH
    assert elapsed < 0.2  # Must execute in < 200ms
    assert "ціна" in result["answer"].lower() or "9500" in result["answer"]
    assert result["token_cost"] == 0.0


def test_unresolved_selector_declines_the_fast_path(router):
    """F49: 'cheapest AMD processor' -- 'процесора' is a selector the
    extremum SQL cannot express (the old code silently dropped it and
    returned the cheapest AMD item, whatever it was). The query must
    take the deep path instead of a confidently wrong fast answer."""
    query = "Яка точна ціна найдешевшого процесора AMD?"
    result = router.route_and_execute(query)
    assert result["tier"] != ExecutionTier.TIER1_FAST_PATH

def test_tier2_complex_compatibility_delegation(router):
    query = "Чи можна поставити процесор AMD Ryzen 7 7800X3D на материнську плату ASUS Prime Z790?"
    result = router.route_and_execute(query)

    assert result["tier"] == ExecutionTier.TIER2_REACT_REASONING
    assert result["status"] == "success"

def test_gaming_substring_not_hijacked_as_min(router):
    # 'Gaming' contains 'min', but must NOT trigger Tier-1 extremum fast path
    query = "Чи сумісна плата MSI B650 Gaming Plus WiFi з Ryzen 5 7600?"
    result = router.route_and_execute(query)
    assert result["tier"] == ExecutionTier.TIER2_REACT_REASONING

def test_tier2_recommendation_delegation(router):
    query = "Порадь процесор до 5000 грн для важких ігор"
    result = router.route_and_execute(query)

    assert result["tier"] == ExecutionTier.TIER2_REACT_REASONING

def test_tier0_adaptive_greeting(router):
    query = "Привіт, як справи?"
    t0 = time.time()
    result = router.route_and_execute(query)
    elapsed = time.time() - t0

    assert result["tier"] == ExecutionTier.TIER0_CONVERSATIONAL
    assert elapsed < 0.05  # Sub-50ms execution
    assert result["token_cost"] <= 0.0002
    assert len(result["steps"]) == 1
    assert "віта" in result["answer"].lower()

def test_tier0_identity_and_capabilities(router):
    query = "Хто ти і що ти вмієш?"
    result = router.route_and_execute(query)

    assert result["tier"] == ExecutionTier.TIER0_CONVERSATIONAL
    assert "Py-Buddy" in result["answer"]
    assert "гіперграф" in result["answer"].lower()

def test_tier0_gratitude(router):
    query = "Дякую, дуже допомогло!"
    result = router.route_and_execute(query)

    assert result["tier"] == ExecutionTier.TIER0_CONVERSATIONAL
    assert "радий" in result["answer"].lower() or "допомог" in result["answer"].lower()
