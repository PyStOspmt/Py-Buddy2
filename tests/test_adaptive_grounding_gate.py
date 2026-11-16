"""
TDD Suite for Adaptive Context-Grounded & Entity-Attribute Grounding Gate.
Tests that conversational math, user budgets, and linguistic aggregates
are NOT wiped by the Zero-Hallucination gate.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from catalog_query import GroundingSet
from pareto_relaxer import CandidateSolution
from react_agent import ReActAgent, SessionState


def _agent_with_grounding(*solutions):
    agent = ReActAgent.__new__(ReActAgent)
    agent.currency = "UAH"
    agent.grounding = GroundingSet()
    for s in solutions:
        agent.grounding.add_solution(s)
    return agent

GROUNDED = (
    CandidateSolution("cpu_am5_2", "AMD Ryzen 5 7600X", 9500.0, 82.0, 2),
    CandidateSolution("mb_am5_msi", "MSI B650 Gaming Plus WiFi", 6800.0, 85.0, 1),
    CandidateSolution("ram_ddr5", "Kingston Fury 32GB DDR5", 4200.0, 90.0, 1),
)

def test_user_budget_citation_not_wiped():
    """When user mentions a budget (e.g. 20000 грн), the agent citing it must NOT be wiped."""
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s_budget", "Підкажи збірку на бюджет 20000 грн")
    
    prose = (
        "У межах вашого бюджету 20000 грн найкращим вибором є процесор "
        "AMD Ryzen 5 7600X (9500.00 грн) та плата MSI B650 Gaming Plus WiFi (6800.00 грн)."
    )
    ok, text = agent.finalize_response(state, prose, [
        {"name": "AMD Ryzen 5 7600X"},
        {"name": "MSI B650 Gaming Plus WiFi"}
    ])
    assert ok
    assert "20000" in text
    assert "AMD Ryzen 5 7600X" in text
    assert "MSI B650 Gaming Plus WiFi" in text
    # Crucial: Must NOT be wiped to a raw list
    assert "Знайшов у каталозі за вашим запитом:" not in text

def test_arithmetic_subset_sum_not_wiped():
    """The total sum of recommended parts (9500 + 6800 = 16300 грн) must be preserved."""
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s_sum", "Скільки коштує разом?")
    
    prose = (
        "Вартість комплекту з AMD Ryzen 5 7600X (9500.00 грн) та "
        "MSI B650 Gaming Plus WiFi (6800.00 грн) разом складає 16300.00 грн."
    )
    ok, text = agent.finalize_response(state, prose, [
        {"name": "AMD Ryzen 5 7600X"},
        {"name": "MSI B650 Gaming Plus WiFi"}
    ])
    assert ok
    assert "16300" in text
    assert "Знайшов у каталозі за вашим запитом:" not in text

def test_surgical_entity_price_correction():
    """If the LLM hallucinates an incorrect price for a valid product, it is corrected in-place."""
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s_surg", "Ціна процесора?")
    
    # 12000 is incorrect for Ryzen 5 7600X (true price is 9500.0)
    prose = "Рекомендую процесор AMD Ryzen 5 7600X за 12000 грн як чудовий варіант для ігор."
    ok, text = agent.finalize_response(state, prose, [{"name": "AMD Ryzen 5 7600X"}])
    assert ok
    # The false price must be corrected to 9500.00 грн, preserving the surrounding reasoning
    assert "9500.00 грн" in text
    assert "12000 грн" not in text
    assert "чудовий варіант для ігор" in text

def test_pure_hallucination_without_evidence_blocked():
    """Completely fabricated products with fake prices are blocked when there is zero evidence."""
    agent = _agent_with_grounding()
    state = SessionState("s_halluc", "Які є відеокарти?")
    prose = "Рекомендую Nvidia RTX 9090 за 150000 грн."
    ok, text = agent.finalize_response(state, prose, [{"name": "Nvidia RTX 9090"}])
    assert not ok
    assert state.termination_reason == "HALLUCINATION_GATE_VIOLATION"
    assert "Nvidia RTX 9090" not in text


def test_fabricated_spec_sentence_stripped():
    """F23: a socket/spec claim nobody retrieved is prose hallucination --
    the sentence is dropped, verified sentences survive."""
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s_spec", "Чи сумісний процесор?")
    prose = (
        "AMD Ryzen 5 7600X коштує 9500.00 грн. "
        "Він працює на сокеті B660."
    )
    ok, text = agent.finalize_response(
        state, prose, [{"name": "AMD Ryzen 5 7600X"}])
    assert ok
    assert "9500" in text
    assert "B660" not in text
    assert state.termination_reason == "SUCCESS_GROUNDED_WITH_CORRECTION"
    assert state.verification_report is not None
    assert state.verification_report["n_unsupported"] == 1


def test_retrieved_spec_claim_survives():
    """F23: a spec claim IS allowed when the run actually fetched the
    attribute row -- verification is run-scoped, not knowledge-scoped."""
    agent = ReActAgent.__new__(ReActAgent)
    agent.currency = "UAH"
    agent.grounding = GroundingSet()
    agent.grounding.extend_rows([
        {"id": "cpu1", "name": "AMD Ryzen 5 7600X", "price": 9500.0,
         "socket": "AM5"},
    ])
    state = SessionState("s_spec_ok", "Чи сумісний процесор?")
    prose = ("AMD Ryzen 5 7600X коштує 9500.00 грн "
             "і працює на сокеті AM5.")
    ok, text = agent.finalize_response(
        state, prose, [{"name": "AMD Ryzen 5 7600X"}])
    assert ok
    assert "AM5" in text
    assert "9500" in text
    assert state.termination_reason == "SUCCESS_GROUNDED"


def test_all_unverifiable_prose_regenerates():
    """F23: when every claim fails, fall back to a strictly grounded
    rendering instead of shipping partially invented prose."""
    agent = _agent_with_grounding(*GROUNDED)
    state = SessionState("s_regen", "Розкажи про процесор")
    prose = "AMD Ryzen 5 7600X має сокет B660 і 24 потоки."
    ok, text = agent.finalize_response(
        state, prose, [{"name": "AMD Ryzen 5 7600X"}])
    assert ok
    assert "B660" not in text
    assert "24 потоки" not in text
    assert state.termination_reason in (
        "SUCCESS_GROUNDED_AFTER_CORRECTION",
        "SUCCESS_GROUNDED_WITH_CORRECTION")


def test_empty_evidence_parametric_spec_blocked():
    """P1: even when 0 rows were retrieved, parametric specs/prices are blocked."""
    agent = _agent_with_grounding()
    state = SessionState("s_empty_param", "Розкажи про процесор")
    prose = "Цей процесор має сокет LGA1700 і коштує 4500 грн."
    ok, text = agent.finalize_response(state, prose, [])
    assert not ok
    assert state.termination_reason == "HALLUCINATION_GATE_VIOLATION"
    assert "LGA1700" not in text
    assert "4500" not in text


def test_projected_table_hypergraph_and_protected_pareto(tmp_path):
    """RF10 + P1 + P4: ReActAgent runs on non-`products` schema, admits resolved
    hypergraph nodes into grounding, and enforces protected bounds in pareto_relax."""
    import sqlite3
    db_file = tmp_path / "custom_store.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE catalog_items ("
        "item_id TEXT PRIMARY KEY, title TEXT, category TEXT, "
        "cost REAL, in_stock INTEGER, performance_score REAL)"
    )
    conn.execute(
        "INSERT INTO catalog_items VALUES "
        "('c1', 'Alpha Core 100', 'processors', 5000.0, 4, 70.0), "
        "('c2', 'Beta Ultra 200', 'processors', 12000.0, 2, 95.0), "
        "('g1', 'Gamma Card 300', 'gpus', 8000.0, 3, 85.0)"
    )
    conn.commit()
    conn.close()

    agent = ReActAgent(db_path=str(db_file))
    state = SessionState("s_custom", "processors")
    assert agent._infer_category(state) == "processors"
    assert any(n == "Alpha Core 100" for n, _ in agent._catalog_names())

    hg_res = agent.execute_tool(
        state, "hypergraph_check", {"item_ids": ["Alpha Core 100"]})
    assert hg_res["success"] is True
    assert "c1" in hg_res["checked_items"]
    assert hg_res["unresolved_items"] == []
    assert "Alpha Core 100" in agent._gs(state).names

    pareto_res = agent.execute_tool(
        state,
        "pareto_relax",
        {
            "category": "processors",
            "budget_max": 4000.0,
            "min_performance": 70.0,
            "protected": ["price"],
        },
    )
    assert pareto_res["nvfg_status"] == "UNSATISFIABLE"
    assert pareto_res["pareto_front_count"] == 0


def test_finalize_preserves_negative_stance_and_conflict_on_pruned_prose():
    gpu = {
        "id": "g1",
        "name": "Gigabyte GeForce RTX 4060 Windforce (Rev 4)",
        "category": "gpu",
        "price": 11743.74,
        "in_stock": 2,
        "attributes": {"performance": 73},
    }
    agent = _agent_with_grounding()
    st = SessionState("s_relax", "Порадь відеокарта з ціною до 5900 грн")
    st.grounding = GroundingSet()
    agent._gs(st).extend_rows([gpu])
    st.execution_history.append({
        "tool_name": "pareto_relax",
        "observation": {
            "success": True,
            "relaxed": True,
            "nvfg_status": "RELAXED",
            "relaxation_text": "Точних варіантів до 5900 грн немає; бюджет послаблено до 11743.74 грн.",
        },
    })
    prose = (
        "На жаль, жодна відеокарта з продуктивністю від 57 балів не доступна за ціною до 5900 грн. "
        "Найближчий варіант — Gigabyte GeForce RTX 4060 Windforce (Rev 4) за ціною 11743.74 грн."
    )
    ok, final = agent.finalize_response(
        st, prose, [{"name": "Gigabyte GeForce RTX 4060 Windforce (Rev 4)"}])
    assert ok
    assert "немає" in final.lower()

    cpu = {"id": "c1", "name": "AMD Ryzen 5 7600X (Rev 4)", "price": 8892.85, "attributes": {"socket": "AM5"}}
    mb = {"id": "m1", "name": "MSI MAG Z790 Tomahawk (Rev 4)", "price": 9000.0, "attributes": {"socket": "LGA1700"}}
    agent2 = _agent_with_grounding()
    st2 = SessionState("s_conf", "Чи сумісні?")
    st2.grounding = GroundingSet()
    agent2._gs(st2).extend_rows([cpu, mb])
    st2.execution_history.append({
        "tool_name": "hypergraph_check",
        "observation": {
            "verdict": "candidate_conflict",
            "is_compatible": False,
            "candidate_conflicts": ["socket: AM5 != LGA1700"],
        },
    })
    ok2, final2 = agent2.finalize_response(
        st2,
        "Так, AMD Ryzen 5 7600X (Rev 4) та MSI MAG Z790 Tomahawk (Rev 4) можна встановити разом.\n"
        "Обидва компоненти підтримують сокет AM5.\n"
        "Перевірте також блок живлення.",
        [{"name": cpu["name"]}, {"name": mb["name"]}],
    )
    assert ok2
    assert "несуміс" in final2.lower()


def test_orphaned_price_line_pruned_when_item_sentence_is_unsupported():
    """Dropping an item header sentence must also drop its orphaned trailing price line."""
    from grounding_gate import render_from_rows

    cpu = {"id": "c1", "name": "AMD Ryzen 5 7600X (Rev 2)", "price": 8317.57}
    mb = {"id": "m1", "name": "MSI B650 Gaming Plus WiFi", "price": 6200.0}
    agent = _agent_with_grounding()
    st = SessionState("s_orphan", "Які є комплектуючі та їх ціни?")
    st.grounding = GroundingSet()
    agent._gs(st).extend_rows([mb, cpu])

    prose = (
        "• MSI B650 Gaming Plus WiFi коштує 6200.00 грн.\n"
        "• AMD Ryzen 5 7600X (Rev 2) — TDP 105 Вт, сокет AM5.\n"
        "Ціна: 8 317,57 грн."
    )
    ok, final = agent.finalize_response(
        st,
        prose,
        [{"name": mb["name"]}, {"name": cpu["name"]}],
    )
    assert ok
    assert "MSI B650 Gaming Plus WiFi" in final
    assert "6200" in final
    # Crucially, 8317.57 must not survive orphaned after MSI B650.
    assert "8 317" not in final and "8317" not in final

    rendered = render_from_rows([{"name": "MSI B650 Gaming Plus WiFi", "price": None, "price_conflicts": [6100.0, 6450.0]}])
    assert "6100.00" in rendered and "6450.00" in rendered


def test_claim_verifier_single_retry_repairs_or_falls_back_cleanly():
    """ReActAgent gives the LLM 1 bounded ClaimVerifier repair pass and stops cleanly."""
    from grounding_gate import diagnose_grounding_feedback

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "catalog.db",
    )

    class _RepairRunner:
        def __init__(self, responses):
            self.responses = list(responses)
            self.seen_histories = []

        def generate_react_step(self, **kwargs):
            self.seen_histories.append(list(kwargs.get("dialog_history") or []))
            return self.responses.pop(0)

    sql_step = {
        "thought": "Перевіряю процесор",
        "action": "sql_query",
        "action_input": {
            "query": "SELECT id, name, category, brand, price, performance_score FROM products WHERE id = 'cpu_am5_2'"
        },
    }

    # Case 1: Draft claims nonexistent "9.9 GHz"; retry receives feedback and fixes answer.
    agent1 = ReActAgent(db_path=db_path)
    runner1 = _RepairRunner([
        sql_step,
        {
            "thought": "Відповідаю",
            "final_answer": "AMD Ryzen 5 7600X коштує 9500 грн. Він працює на частоті 9.9 GHz.",
            "recommended_items": [{"name": "AMD Ryzen 5 7600X"}],
        },
        {
            "thought": "Виправляю без непідтвердженої частоти",
            "final_answer": "AMD Ryzen 5 7600X коштує 9500.00 грн.",
            "recommended_items": [{"name": "AMD Ryzen 5 7600X"}],
        },
    ])
    st1 = agent1.run_live_session("Яка ціна Ryzen 5 7600X?", runner1, "schema")
    assert st1.termination_reason == "SUCCESS_GROUNDED"
    assert "9500" in st1.final_response
    assert "9.9" not in st1.final_response
    assert len(runner1.seen_histories) == 3
    last_feedback = runner1.seen_histories[-1][-1]["content"]
    assert "ClaimVerifier" in last_feedback and "9.9" in last_feedback
    assert st1.execution_history[-1]["claim_verifier_repair"]["accepted"] is True
    assert diagnose_grounding_feedback(agent1, st1, st1.final_response, st1.recommended_items) is None

    # Case 2: Retry ALSO hallucinates -> bounded to 1 retry, falls back deterministically.
    agent2 = ReActAgent(db_path=db_path)
    runner2 = _RepairRunner([
        sql_step,
        {
            "thought": "Відповідаю",
            "final_answer": "AMD Ryzen 5 7600X коштує 9500 грн. Він працює на частоті 9.9 GHz.",
            "recommended_items": [{"name": "AMD Ryzen 5 7600X"}],
        },
        {
            "thought": "Повторюю вигадку",
            "final_answer": "AMD Ryzen 5 7600X коштує 9500 грн. Має 64 ядра та 9.9 GHz.",
            "recommended_items": [{"name": "AMD Ryzen 5 7600X"}],
        },
    ])
    st2 = agent2.run_live_session("Яка ціна Ryzen 5 7600X?", runner2, "schema")
    assert len(runner2.seen_histories) == 3  # no extra retry loop!
    assert st2.execution_history[-1]["claim_verifier_repair"]["accepted"] is False
    assert "9.9" not in st2.final_response and "9500" in st2.final_response

    # Case 3: Finite runner exhausts responses on retry -> deterministic gate still succeeds safely.
    agent3 = ReActAgent(db_path=db_path)
    runner3 = _RepairRunner([
        sql_step,
        {
            "thought": "Відповідаю",
            "final_answer": "AMD Ryzen 5 7600X коштує 9500 грн. Він працює на частоті 9.9 GHz.",
            "recommended_items": [{"name": "AMD Ryzen 5 7600X"}],
        },
    ])
    st3 = agent3.run_live_session("Яка ціна Ryzen 5 7600X?", runner3, "schema")
    assert "9.9" not in st3.final_response and "9500" in st3.final_response
    assert "runner_error" in st3.execution_history[-1]["claim_verifier_repair"]["reason"]


def test_recommended_items_follow_prose_mentions_and_preserve_order():
    """Recommended cards must match items named in prose and follow prose order."""
    from grounding_gate import finalize_grounded_response
    from react_agent import SessionState

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "catalog.db",
    )
    agent = ReActAgent(db_path=db_path)

    rows = [
        {"id": "cpu_1", "name": "AMD Ryzen 5 7600X", "price": 9500.0},
        {"id": "cpu_2", "name": "AMD Ryzen 7 7800X3D", "price": 17500.0},
        {"id": "cpu_3", "name": "Intel Core i5-14600K", "price": 13500.0},
    ]
    state = SessionState(
        session_id="test_rec_order",
        user_query="Порадь процесор",
        execution_history=[
            {
                "step": 1,
                "tool_name": "sql_query",
                "result": {"status": "ok", "rows": rows, "row_count": len(rows)},
            }
        ],
    )
    from react_tools import agent_grounding
    agent_grounding(agent, state).extend_rows(rows)
    prose = (
        "1. **AMD Ryzen 7 7800X3D** — 17500.00 грн.\n"
        "2. **AMD Ryzen 5 7600X** — 9500.00 грн."
    )
    ok, _ = finalize_grounded_response(
        agent,
        state,
        prose,
        [
            {"name": "AMD Ryzen 5 7600X"},
            {"name": "Intel Core i5-14600K"},
        ],
    )
    assert ok is True
    assert [it["name"] for it in state.recommended_items] == [
        "AMD Ryzen 7 7800X3D",
        "AMD Ryzen 5 7600X",
    ]


def test_finalize_response_uses_domain_currency_without_hardcoded_uah():
    """Grounding gate feedback and auto-repair must honor USD/EUR/None instead of hardcoding грн."""
    from grounding_gate import diagnose_grounding_feedback, finalize_grounded_response
    from react_agent import SessionState

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "real_wdc_fixture.db",
    )
    if not os.path.exists(db_path):
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests",
            "fixtures",
            "real_wdc_fixture.db",
        )
    agent = ReActAgent(db_path=db_path, currency="USD")
    rows = [{"id": "w1", "name": "NVIDIA Quadro M4000", "price": 837.99}]
    state = SessionState(
        session_id="test_curr_usd",
        user_query="Порадь відеокарту",
        execution_history=[
            {
                "step": 1,
                "tool_name": "sql_query",
                "result": {"status": "ok", "rows": rows, "row_count": 1},
            }
        ],
    )
    from react_tools import agent_grounding
    agent_grounding(agent, state).extend_rows(rows)
    fb = diagnose_grounding_feedback(
        agent,
        state,
        "NVIDIA Quadro M4000 має 64 ядра.",
        [{"name": "NVIDIA Quadro M4000"}],
    )
    assert fb is not None
    assert "837.99 USD" in fb[0]
    assert "грн" not in fb[0]

    ok, fixed = finalize_grounded_response(
        agent,
        state,
        "NVIDIA Quadro M4000 коштує 999.00 USD.",
        [{"name": "NVIDIA Quadro M4000"}],
    )
    assert ok is True
    assert "837.99 USD" in fixed
    assert "грн" not in fixed

    agent_none = ReActAgent(db_path=db_path, currency=None)
    state_none = SessionState(
        session_id="test_curr_none",
        user_query="Покажи товар",
        execution_history=[
            {
                "step": 1,
                "tool_name": "sql_query",
                "result": {"status": "ok", "rows": rows, "row_count": 1},
            }
        ],
    )
    agent_grounding(agent_none, state_none).extend_rows(rows)
    ok_n, fixed_n = finalize_grounded_response(
        agent_none,
        state_none,
        "NVIDIA Quadro M4000 коштує 999.00.",
        [{"name": "NVIDIA Quadro M4000"}],
    )
    assert ok_n is True
    assert "837.99" in fixed_n
    assert "грн" not in fixed_n


def test_evidence_summary_prioritizes_newest_unique_items():
    """Later tool steps and distinct item names must not be crowded out by 50 stale rows."""
    from grounding_gate import _format_evidence_summary

    old_rows = [{"name": "Item Alpha", "price": 8000.0 + i, "stock_quantity": 5} for i in range(50)]
    new_rows = [{"name": "Item Beta", "price": 15940.80, "stock_quantity": 3}]
    summary = _format_evidence_summary(old_rows + new_rows)
    assert "Item Beta" in summary
    assert "Item Alpha" in summary


def test_preserve_tool_status_skips_pareto_tail_on_prose_negation_when_verified():
    """Ordinary prose negation ('дефіциту немає') must not inject a Pareto zero-match notice."""
    from grounding_gate import preserve_tool_status

    prose = "Ні, дефіциту немає: Intel Core i7-14700K є в наявності за 15940.80 грн."
    out = preserve_tool_status(
        prose, [], None, None, prose, has_verified_items=True
    )
    assert "На жаль, точних варіантів" not in out


def test_finalize_dedups_recommended_and_binds_pre_name_or_revision_price():
    """Pre-name prices, sibling revisions, and duplicate recommendations must stay grounded."""
    from grounding_gate import finalize_grounded_response
    from react_tools import agent_grounding

    rows = [
        {"id": "cpu_1", "name": "Chip X (Rev 1)", "price": 15533.50, "category": "cpu"},
        {"id": "cpu_2", "name": "Chip X (Rev 2)", "price": 15940.80, "category": "cpu"},
        {"id": "cpu_3", "name": "Chip X (Rev 4)", "price": 18883.11, "category": "cpu"},
    ]
    agent = ReActAgent.__new__(ReActAgent)
    agent.currency = "₴"
    agent.grounding = GroundingSet()
    state = SessionState(
        session_id="s_dedup_rev",
        user_query="Ціна Chip X?",
        execution_history=[
            {"step": 1, "tool_name": "sql_query", "observation": {"success": True, "rows": rows, "row_count": 3}}
        ],
    )
    agent_grounding(agent, state).extend_rows(rows)

    ok, _ = finalize_grounded_response(
        agent,
        state,
        "Наприклад, за 15940.80 грн є Chip X (Rev 2), а в лінійці Chip X (Rev 1..4) старт від 15533.50 грн до 18883.11 грн.",
        [{"name": "Chip X (Rev 2)"}, {"name": "Chip X (Rev 2)"}],
    )
    assert ok is True
    assert state.termination_reason == "SUCCESS_GROUNDED"
    assert [r["name"] for r in state.recommended_items] == ["Chip X (Rev 2)"]

