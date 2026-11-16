"""
Tests for the non-vacuous Zero-Hallucination Gate (ADR-013).

The previous gate answered a customer with

    "Помилка верифікації фактів: ... Відповідь заблоковано
     Zero-Hallucination Gate для захисту достовірності."

which is the vacuous-output failure mode arXiv:2603.16817 names: factuality
purchased by filtering, at the price of every useful sentence. These tests
pin the new behaviour, because the temptation to "just block it" is strong
and it is the single worst thing this system can do to a shopper.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from catalog_query import GroundingSet
from pareto_relaxer import CandidateSolution
from react_agent import ReActAgent, SessionState

CATALOG_DB = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "catalog.db"))
needs_db = pytest.mark.skipif(
    not os.path.exists(CATALOG_DB),
    reason="catalog.db not generated; run data/init_database.py",
)


def _agent_with_grounding(*solutions):
    agent = ReActAgent.__new__(ReActAgent)      # no DB needed for the gate
    agent.currency = "UAH"
    agent.grounding = GroundingSet()
    for s in solutions:
        agent.grounding.add_solution(s)
    return agent


GROUNDED = (
    CandidateSolution("cpu_am5_2", "AMD Ryzen 5 7600X", 9500.0, 82.0, 2),
    CandidateSolution("mb_am5_msi", "MSI B650 Gaming Plus WiFi", 6800.0, 85.0, 1),
)


def test_fully_grounded_answer_passes_untouched():
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s1", "q")
    ok, text = agent.finalize_response(
        state, "Раджу AMD Ryzen 5 7600X.", [{"name": "AMD Ryzen 5 7600X"}])
    assert ok
    assert state.termination_reason == "SUCCESS_GROUNDED"
    assert "AMD Ryzen 5 7600X" in text
    assert "Частину" not in text


def test_fabricated_item_is_dropped_and_reported():
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s2", "q")
    ok, text = agent.finalize_response(
        state, "Раджу.", [{"name": "AMD Ryzen 5 7600X"}, {"name": "AMD Ryzen 9 9950X"}])
    assert ok
    assert state.termination_reason == "SUCCESS_GROUNDED_WITH_CORRECTION"
    assert "не вдалося підтвердити" in text
    assert {r["name"] for r in state.recommended_items} == {"AMD Ryzen 5 7600X"}


def test_fully_fabricated_answer_falls_back_to_grounded_rows_not_an_error():
    """The core ADR-013 guarantee: never a technical error to the customer."""
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s3", "q")
    ok, text = agent.finalize_response(
        state, "сміття", [{"name": "AMD Ryzen 9 9950X"}])
    assert ok, "must not block when grounded evidence exists"
    assert state.termination_reason == "SUCCESS_GROUNDED_AFTER_CORRECTION"
    assert "AMD Ryzen 5 7600X" in text
    assert "Помилка верифікації" not in text
    assert "Zero-Hallucination Gate" not in text
    assert "9950X" not in text


def test_no_grounded_evidence_still_blocks_honestly():
    agent = _agent_with_grounding()             # nothing was ever queried
    state = SessionState("s4", "q")
    ok, text = agent.finalize_response(
        state, "сміття", [{"name": "AMD Ryzen 9 9950X"}])
    assert not ok
    assert state.termination_reason == "HALLUCINATION_GATE_VIOLATION"
    assert "AMD Ryzen 9 9950X" not in text
    assert "Уточніть" in text


def test_blocked_message_is_not_a_technical_error():
    """Even a genuine block must read as a reply, not as a stack trace."""
    agent = _agent_with_grounding()
    state = SessionState("s5", "q")
    _ok, text = agent.finalize_response(state, "x", [{"name": "Made Up GPU"}])
    for banned in ("Помилка", "Traceback", "NoneType", "Gate", "заблоковано"):
        assert banned not in text


def test_conversation_with_no_items_passes():
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s6", "Привіт!")
    ok, text = agent.finalize_response(state, "Вітаю!", [])
    assert ok
    assert state.termination_reason == "SUCCESS_GROUNDED"
    assert text == "Вітаю!"


@needs_db
def test_grounded_synthesis_only_prints_prices_it_has():
    agent = _agent_with_grounding(*GROUNDED)
    text = agent._grounded_synthesis([
        {"id": "p", "name": "AMD Ryzen 5 7600X", "price": 9500.0},
        {"id": "q", "name": "Без ціни", "price": None},
    ])
    assert "9500.00 грн" in text
    assert "Без ціни" in text
    assert text.count("грн") == 1
