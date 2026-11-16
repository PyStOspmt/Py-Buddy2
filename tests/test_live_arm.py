"""
The live LLM arm: does it exist, and does it refuse to fabricate a measurement?

Nothing here needs a network or a key:

  1. the arm `benchmark.py` has always imported now exists;
  2. the budget guards stop BEFORE paying, not after;
  3. a transport failure or an exhausted budget is recorded as such and is never
     silently replaced by a fallback answer that would score as knowledge.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from benchmark import DEFAULT_DB
from benchmark_cases import load_snapshot
from live_agent import (
    BudgetedRunner,
    BudgetExceeded,
    LivePyBuddyAgent,
    default_schema_prompt,
)


class CannedRunner:
    """Stands in for OpenRouter: answers immediately, counts its own tokens."""

    model_name = "canned/model"

    def __init__(self, answer="Так, я відповідаю.", fail=False):
        self.answer = answer
        self.fail = fail
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def generate_react_step(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("upstream 502")
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 20
        return {"thought": "маю всі факти",
                "final_answer": self.answer,
                "recommended_items": []}


def test_the_arm_benchmark_has_always_imported_now_exists():
    import live_agent

    assert hasattr(live_agent, "LivePyBuddyAgent")
    from inspect import signature
    assert {"self", "query"} <= set(signature(LivePyBuddyAgent.answer).parameters)


def test_a_live_arm_without_a_runner_is_refused():
    """A silent fallback to the deterministic agent would be a fake live run."""
    with pytest.raises(ValueError, match="measures"):
        BudgetedRunner(None)


def test_the_call_cap_is_checked_before_the_call_is_made():
    """A guard that fires after payment is not a guard."""
    inner = CannedRunner()
    runner = BudgetedRunner(inner, max_llm_calls=2)
    runner.generate_react_step(user_query="q")
    runner.generate_react_step(user_query="q")
    with pytest.raises(BudgetExceeded):
        runner.generate_react_step(user_query="q")
    assert inner.calls == 2          # the third request never reached the API
    assert runner.calls == 2


def test_the_usd_cap_is_armed_only_when_a_price_is_configured():
    free = BudgetedRunner(CannedRunner(), max_usd=0.001,
                          price_in_per_mtok=0.0, price_out_per_mtok=0.0)
    free.generate_react_step(user_query="q")
    free.generate_react_step(user_query="q")          # free model: never trips
    assert free.budget_mode == "calls"

    paid = BudgetedRunner(CannedRunner(), max_usd=0.000001,
                          price_in_per_mtok=1.0, price_out_per_mtok=1.0)
    paid.generate_react_step(user_query="q")
    with pytest.raises(BudgetExceeded):
        paid.generate_react_step(user_query="q")
    assert paid.budget_mode == "usd+calls"
    assert paid.cost_usd > 0


def test_the_transcript_records_what_the_model_actually_said():
    runner = BudgetedRunner(CannedRunner(answer="2500.00 грн"))
    runner.generate_react_step(user_query="яка ціна?")
    assert runner.transcript[0]["final_answer"] == "2500.00 грн"
    assert runner.transcript[0]["action"] is None
    assert runner.transcript[0]["latency_s"] >= 0


def test_a_transport_failure_is_recorded_not_invented():
    """An outage must never be scored as an answer."""
    runner = BudgetedRunner(CannedRunner(fail=True))
    with pytest.raises(RuntimeError, match="502"):
        runner.generate_react_step(user_query="q")
    entry = runner.transcript[-1]
    assert "RuntimeError" in entry["error"]
    assert entry.get("final_answer") is None


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(DEFAULT_DB)


def test_the_live_agent_answers_with_a_canned_model(snap):
    agent = LivePyBuddyAgent(snap, CannedRunner(answer="Вітаю!"),
                             schema_prompt="Schema")
    out = agent.answer("привіт, хто ти?")
    assert out["text"]
    assert out["steps"] >= 1
    stats = agent.stats()
    assert stats["llm_calls"] >= 1
    assert stats["model"] == "canned/model"


def test_the_live_report_carries_raw_output_tokens_and_caps(snap):
    agent = LivePyBuddyAgent(snap, CannedRunner(), schema_prompt="Schema",
                             max_llm_calls=7, max_usd=0.25)
    agent.answer("привіт")
    stats = agent.stats()
    assert stats["max_llm_calls"] == 7
    assert stats["max_usd"] == 0.25
    assert stats["budget_mode"] == "calls"     # no price configured: honestly stated
    assert stats["prompt_tokens"] > 0
    assert stats["cases_answered"] == 1 and stats["cases_errored"] == 0
    assert stats["transcript"], "raw steps must be stored"


def test_a_budget_stop_is_recorded_as_a_truncated_case(snap):
    agent = LivePyBuddyAgent(snap, CannedRunner(), schema_prompt="Schema",
                             max_llm_calls=1)
    agent.answer("перший запит")
    out = agent.answer("другий запит")
    stats = agent.stats()
    assert stats["truncated"] is True
    assert stats["case_log"][-1]["truncated_by_budget"] is True
    assert out["text"] == ""                   # nothing answered, nothing invented


def test_the_schema_prompt_comes_from_the_database():
    """Never from a hand-written description of the developer's store."""
    assert isinstance(default_schema_prompt("/nonexistent/nope.db"), str)
    prompt = default_schema_prompt(DEFAULT_DB)
    assert isinstance(prompt, str)
    if prompt:
        assert "products" in prompt.lower() or "marketplace" in prompt.lower()


def test_the_live_choice_is_refused_without_a_runner(monkeypatch):
    """`--system live` is a real choice, and it does not degrade quietly."""
    import benchmark

    monkeypatch.setattr(benchmark, "_make_runner", lambda: None)
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        benchmark._live_agent(None)


def test_the_live_arm_executes_real_openrouter_query_when_configured():
    """Validates real end-to-end execution of live arm against OpenRouter without mocks."""
    env_explicit = "OPENROUTER_API_KEY" in os.environ
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not env_explicit:
        env_file = os.path.join(ROOT, ".env")
        if os.path.exists(env_file):
            for line in open(env_file, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "OPENROUTER_API_KEY":
                        api_key = v.strip()
                        os.environ["OPENROUTER_API_KEY"] = api_key
                        break
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not configured in environment or .env")

    from domain_catalog import DomainCatalogRegistry
    from openrouter_client import OpenRouterAgentRunner
    from react_agent import ReActAgent

    runner = OpenRouterAgentRunner()
    kg = DomainCatalogRegistry.create_pc_catalog()
    agent = ReActAgent(db_path=DEFAULT_DB, hypergraph=kg, max_steps=4)
    prompt = default_schema_prompt(DEFAULT_DB)

    state = agent.run_live_session(
        user_query="Яка ціна на процесор AMD Ryzen 7 7800X3D?",
        llm_runner=runner,
        schema_prompt=prompt,
    )

    assert state.final_response is not None
    assert len(state.final_response) > 0
    # Must be grounded and contain accurate price or item mention
    assert "7800x3d" in state.final_response.lower() or "ryzen" in state.final_response.lower()
    # Gate-passed outcomes: clean grounding or ClaimVerifier/price correction
    # preserving grounded catalog identity.
    assert state.termination_reason in (
        "SUCCESS_GROUNDED",
        "SUCCESS_GROUNDED_PRICE_CORRECTED",
        "SUCCESS_GROUNDED_WITH_CORRECTION",
        "SUCCESS_GROUNDED_AFTER_CORRECTION",
    )

