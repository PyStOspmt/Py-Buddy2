"""Offline unit & regression tests for Step 3 benchmark harness & R3/R13 additions."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src", "core"), os.path.join(ROOT, "src", "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_rag import RetrievalBaseline  # noqa: E402
from benchmark_aic2026 import (  # noqa: E402
    build_expanded_cases,
    build_multiturn_scenarios,
    run_model_evaluation,
    run_non_english_induction_eval,
)
from benchmark_cases import load_snapshot  # noqa: E402
from live_agent import BudgetedRunner, SharedBudget  # noqa: E402
from react_agent import ReActAgent  # noqa: E402


DB_PATH = os.path.join(ROOT, "data", "marketplace_rozetka.db")
ZOO_PATH = os.path.join(ROOT, "data", "jury_zoo.db")


class _StubRunner:
    model_name = "stub-model"

    def __init__(self, steps=None, ans=("Тестова відповідь", 0.0)):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.seen_histories = []
        self._steps = list(steps or [])
        self._ans = ans

    def answer_once(self, system_prompt: str, user_prompt: str):
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 20
        return self._ans

    def generate_react_step(self, **kwargs):
        self.seen_histories.append(list(kwargs.get("dialog_history") or []))
        self.total_prompt_tokens += 150
        self.total_completion_tokens += 30
        return self._steps.pop(0)


@pytest.fixture(scope="module")
def rozetka_fixture():
    snap = load_snapshot(DB_PATH)
    return snap, build_expanded_cases(snap)


def test_expanded_suite_counts_and_disjoint_ids(rozetka_fixture):
    snap, cases = rozetka_fixture
    assert len(cases) == 73
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))
    scenarios = build_multiturn_scenarios(snap)
    assert len(scenarios) == 3
    assert all(len(s["turns"]) == 3 for s in scenarios)


def test_non_english_induction_on_jury_zoo_and_cip():
    res = run_non_english_induction_eval(ZOO_PATH)
    assert res["cip_benchmark"]["proposed"]["f1"] >= 70.0
    zoo = res["jury_zoo_ukrainian"]
    assert zoo["available"] is True
    assert zoo["n_entities"] == 420
    assert zoo["contains_dimensional_containment_in_structural"] is True
    assert zoo["contains_dimensional_containment_in_contract_inductor"] is True


def test_budgeted_rag_runner_forwards_history():
    stub = _StubRunner(ans=("Наявних товарів не знайдено.", 0.0))
    br = BudgetedRunner(stub, max_llm_calls=5, max_usd=0.5,
                        price_in_per_mtok=0.1, price_out_per_mtok=0.4)
    rag = RetrievalBaseline(DB_PATH, runner=br, top_k=4)
    text, calls, cost = rag.answer(
        "А дешевше є?",
        conversation_history=[{"role": "user", "content": "Покажи відеокарту RTX 4060"}],
    )
    assert calls == 1
    assert "Наявних" in text
    assert cost > 0.0


def test_multiturn_history_keeps_grounding_isolated_per_turn():
    stub = _StubRunner(steps=[
        {"thought": "done", "final_answer": "Процесор коштує 9000 грн.", "recommended_items": []},
    ])
    agent = ReActAgent(db_path=DB_PATH)
    state = agent.run_live_session(
        user_query="А скільки коштує він зараз?",
        llm_runner=stub,
        schema_prompt="",
        conversation_history=[
            {"role": "user", "content": "Скільки коштує AMD Ryzen 5 7600X?"},
            {"role": "assistant", "content": "AMD Ryzen 5 7600X — 9500 грн."},
        ],
    )
    assert len(stub.seen_histories[0]) == 2
    assert state.termination_reason == "HALLUCINATION_GATE_VIOLATION"


def test_parallel_run_model_evaluation_and_shared_budget(rozetka_fixture):
    import threading
    snap, expanded = rozetka_fixture
    cases = expanded[:6]

    res = run_model_evaluation(
        snap,
        cases,
        "google/gemini-2.5-flash-lite",
        max_usd=1.0,
        max_llm_calls=50,
        include_multiturn=False,
        workers=4,
        runner_factory=lambda: _StubRunner(
            steps=[{"thought": "done", "final_answer": "Товар відсутній у каталозі.", "recommended_items": []}],
            ans=("На жаль, запитуваного товару немає в наявності.", 0.0),
        ),
    )
    assert res["n_cases"] == 6
    assert res["llm_rag"]["llm_calls"] == 6
    assert res["pybuddy_live"]["stats"]["llm_calls"] == 6

    sb = SharedBudget(max_llm_calls=5, max_usd=1.0, price_in_per_mtok=1.0, price_out_per_mtok=1.0)
    errors = []

    def _hit():
        try:
            sb.check_and_reserve()
            sb.book(100, 50)
        except Exception as exc:
            errors.append(type(exc).__name__)

    threads = [threading.Thread(target=_hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sb.calls == 5
    assert errors == ["BudgetExceeded", "BudgetExceeded", "BudgetExceeded"]


def test_multiturn_oracles_and_chronological_openrouter_messages():
    """Multi-turn non-verdict turns keep checks_verdict=False and prior turns precede current query."""
    from benchmark_cases import VERDICT_FAMILIES
    from openrouter_client import OpenRouterAgentRunner

    snap = load_snapshot(DB_PATH)
    for scen in build_multiturn_scenarios(snap):
        for turn in scen["turns"]:
            o = turn["oracle"]
            expected_cv = o.case_type in VERDICT_FAMILIES or o.case_type == "price_trap"
            assert o.checks_verdict is expected_cv, (o.case_id, o.case_type)

    msgs = OpenRouterAgentRunner._build_react_messages(
        "S",
        "NEW_TURN_3_QUERY",
        [
            {"role": "user", "content": "[Попередній хід діалогу — лише контекст наміру]\nOLD_TURN_2"},
            {"role": "assistant", "content": "[Попередній хід діалогу — лише контекст наміру]\nOLD_ANSWER_2"},
            {"role": "assistant", "content": "Action: sql_query, Input: {}"},
        ],
        1,
    )
    contents = [m["content"] for m in msgs]
    idx_old = next(i for i, c in enumerate(contents) if "OLD_TURN_2" in c)
    idx_new = next(i for i, c in enumerate(contents) if "NEW_TURN_3_QUERY" in c)
    idx_act = next(i for i, c in enumerate(contents) if "Action: sql_query" in c)
    assert idx_old < idx_new < idx_act
