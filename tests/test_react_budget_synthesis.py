import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from domain_catalog import DomainCatalogRegistry
from react_agent import ReActAgent, SessionState

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'catalog.db'))

class MockRunnerExhaustingBudget:
    def __init__(self):
        self.step_count = 0

    def generate_react_step(self, schema_prompt, user_query, dialog_history, current_step, tools_description):
        self.step_count += 1
        # If budget exhausted prompt received, return final_answer
        if "БЮДЖЕТ ІНСТРУМЕНТІВ ВИЧЕРПАНО" in tools_description:
            return {
                "thought": "Синтезую фінальну відповідь після вичерпання ліміту",
                "final_answer": "Усі перевірки завершено. Рекомендую процесор AMD Ryzen 7 7800X3D.",
                "recommended_items": [{"id": "cpu_am5_1", "name": "AMD Ryzen 7 7800X3D", "price": 16500.0}]
            }
        # Otherwise keep querying
        return {
            "thought": f"Крок {current_step}: продовжую пошук",
            "action": "sql_query",
            "action_input": {"query": f"SELECT id, name, price FROM products WHERE id = 'cpu_am5_1' -- step {current_step}"}
        }

def test_react_agent_recovers_from_step_budget_exhaustion():
    kg = DomainCatalogRegistry.create_pc_catalog()
    agent = ReActAgent(db_path=DB_PATH, hypergraph=kg, max_steps=4)
    runner = MockRunnerExhaustingBudget()

    state = agent.run_live_session(
        user_query="Порадь процесор",
        llm_runner=runner,
        schema_prompt="Test schema"
    )

    # Must NOT fail with "Не вдалося отримати відповідь" or empty final_response
    assert state.final_response is not None
    assert "AMD Ryzen 7 7800X3D" in state.final_response
    assert len(state.recommended_items) > 0
    assert state.termination_reason == "SUCCESS_GROUNDED"

def test_hypergraph_check_resolves_product_names_and_ids():
    kg = DomainCatalogRegistry.create_pc_catalog()
    agent = ReActAgent(db_path=DB_PATH, hypergraph=kg)
    state = SessionState(session_id="test_hg", user_query="Сумісність")

    # F12: the pc catalog's rules are induced from a sample, so they are
    # candidates. A candidate rule that passes is NOT evidence: the honest
    # verdict for the AM5+AM5 pair is "unknown", not a verified "compatible".
    # Failed before: the candidate pass was reported as is_compatible=True.
    res = agent.execute_tool(state, "hypergraph_check", {"item_ids": ["cpu_am5_1", "mb_am5_msi"]})
    assert res["success"] is True
    assert res["verdict"] == "unknown"
    assert res["is_compatible"] is None
    assert res["candidate_conflicts"] == []

    # Pass incompatible items (CPU AM5 and Motherboard LGA1700 ASUS Z790):
    # the candidate rule conservatively vetoes, as a distinct honest state.
    res2 = agent.execute_tool(state, "hypergraph_check", {"item_ids": ["cpu_am5_1", "mb_intel_asus"]})
    assert res2["success"] is True
    assert res2["verdict"] == "candidate_conflict"
    assert res2["is_compatible"] is False
    assert len(res2["candidate_conflicts"]) > 0

    # Pass by product names
    res3 = agent.execute_tool(state, "hypergraph_check", {"item_ids": ["AMD Ryzen 7 7800X3D", "ASUS Prime Z790-P"]})
    assert res3["success"] is True
    assert res3["verdict"] == "candidate_conflict"
    assert res3["is_compatible"] is False
    assert len(res3["candidate_conflicts"]) > 0
