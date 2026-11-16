"""
Regressions and invariants for SQL execution window awareness, ReAct cycle state,
domain-agnostic FTS/LIKE catalog_search pagination, and profiler histograms.
"""
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))


def _make_renamed_db(path, rows):
    if os.path.exists(path):
        os.unlink(path)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE catalog_entries ("
        "sku TEXT PRIMARY KEY, "
        "label TEXT NOT NULL, "
        "grp TEXT NOT NULL, "
        "cost REAL NOT NULL)"
    )
    conn.executemany("INSERT INTO catalog_entries VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


@pytest.fixture
def renamed_large_db(tmp_path):
    """65 matching 'beta' rows where row 65 (last in table) is the cheapest."""
    db = tmp_path / "renamed_scale.db"
    rows = []
    # 25 alpha rows
    for i in range(25):
        rows.append((f"a_{i:03d}", f"Alpha Item {i}", "alpha", 1000.0 + i))
    # 65 beta rows matching X-4070: first 64 cost 500..563, last row costs 120.50
    for i in range(64):
        rows.append((f"b_{i:03d}", f"Module X-4070 Rev {i}", "beta", 500.0 + i))
    rows.append(("b_064", "Module X-4070 Econom", "beta", 120.50))
    _make_renamed_db(db, rows)
    return str(db)


def test_sql_window_warns_on_unordered_truncation_and_reports_exact_total(renamed_large_db):
    from sanitizer import SQLSanitizer

    san = SQLSanitizer(default_limit=20, max_limit=50)
    res = san.execute_safe_query(
        renamed_large_db,
        "SELECT sku, label, cost FROM catalog_entries WHERE grp = 'beta'",
    )
    assert res["success"] is True
    assert res["row_count"] == 20
    # Unordered first-20 slice misses the 120.50 global minimum at row 65!
    assert min(r["cost"] for r in res["rows"]) >= 500.0

    assert res["truncated"] is True
    assert res["ordered"] is False
    assert res["total"] == 65
    assert res["has_more"] is True
    assert res["offset"] == 0
    assert res["next_offset"] == 20
    assert res["showing"] == "1-20 of 65"
    assert res["warning_code"] == "TRUNCATED_UNORDERED"
    assert "ORDER BY" in res["notice"]


def test_sql_ordered_and_tail_page_and_scalar_aggregate_metadata(renamed_large_db):
    from sanitizer import SQLSanitizer

    san = SQLSanitizer(default_limit=20, max_limit=50)
    ordered_res = san.execute_safe_query(
        renamed_large_db,
        "SELECT sku, label, cost FROM catalog_entries WHERE grp = 'beta' ORDER BY cost ASC",
    )
    assert ordered_res["truncated"] is True
    assert ordered_res["ordered"] is True
    assert ordered_res["total"] == 65
    assert ordered_res["rows"][0]["cost"] == 120.50
    assert ordered_res["showing"] == "1-20 of 65"
    assert ordered_res["warning_code"] == "TRUNCATED_ORDERED"

    tail_res = san.execute_safe_query(
        renamed_large_db,
        "SELECT sku, label, cost FROM catalog_entries WHERE grp = 'beta' ORDER BY cost ASC LIMIT 20 OFFSET 60",
    )
    assert tail_res["row_count"] == 5
    assert tail_res["truncated"] is False
    assert tail_res["has_more"] is False
    assert tail_res["total"] == 65
    assert tail_res["offset"] == 60
    assert tail_res["showing"] == "61-65 of 65"

    agg_res = san.execute_safe_query(
        renamed_large_db,
        "SELECT COUNT(*) AS cnt, MIN(cost) AS min_c FROM catalog_entries WHERE grp = 'beta'",
    )
    assert agg_res["row_count"] == 1
    assert agg_res["truncated"] is False
    assert agg_res["has_more"] is False


def test_sql_bound_limit_offset_and_quoted_keyword_safety(renamed_large_db):
    from sanitizer import SQLSanitizer

    san = SQLSanitizer(default_limit=20, max_limit=50)
    bound_res = san.execute_safe_query(
        renamed_large_db,
        "SELECT sku, label, cost FROM catalog_entries WHERE grp = ? ORDER BY cost ASC LIMIT ? OFFSET ?",
        params=["beta", 10, 5],
    )
    assert bound_res["row_count"] == 10
    assert bound_res["total"] == 65
    assert bound_res["offset"] == 5
    assert bound_res["next_offset"] == 15
    assert bound_res["showing"] == "6-15 of 65"
    assert bound_res["truncated"] is True

    # Literal string mentioning ORDER BY / LIMIT must not fool ordered detection.
    lit_res = san.execute_safe_query(
        renamed_large_db,
        "SELECT sku, label, cost FROM catalog_entries WHERE label != 'ORDER BY cost ASC LIMIT 5'",
    )
    assert lit_res["ordered"] is False
    assert lit_res["truncated"] is True
    assert lit_res["total"] == 90


def test_catalog_search_domain_agnostic_pagination_and_grounding(renamed_large_db):
    from claim_verifier import verify_answer_against_rows
    from grounding_gate import _extract_tool_counts
    from react_agent import ReActAgent, SessionState
    from react_tools import agent_grounding, dispatch_tool

    agent = ReActAgent(db_path=renamed_large_db)
    state = SessionState(session_id="s_search", user_query="Який найдешевший X-4070?")

    res, err = dispatch_tool(
        agent,
        state,
        "catalog_search",
        {"query": "X-4070", "sort": "price_asc", "limit": 5, "offset": 0},
    )
    assert err is None
    assert res["success"] is True
    assert res["total"] == 65
    assert res["returned"] == 5
    assert res["has_more"] is True
    assert res["next_offset"] == 5
    assert res["showing"] == "1-5 of 65"
    assert res["rows"][0]["name"] == "Module X-4070 Econom"
    assert res["rows"][0]["price"] == 120.50
    assert res["category_breakdown"] == [{"category": "beta", "count": 65}]

    # Verify page 2 offset
    res2, err2 = dispatch_tool(
        agent,
        state,
        "catalog_search",
        {"query": "X-4070", "sort": "price_asc", "limit": 5, "offset": 60},
    )
    assert err2 is None
    assert res2["showing"] == "61-65 of 65"
    assert res2["has_more"] is False

    # F04 + ClaimVerifier compatibility: exact count '65' and price 120.50 verify cleanly
    tool_counts = _extract_tool_counts([{"tool_name": "catalog_search", "observation": res}])
    assert 65 in tool_counts
    gs = agent_grounding(agent, state)
    rep = verify_answer_against_rows(
        "У каталозі знайдено 65 варіантів, найдешевший — Module X-4070 Econom за 120,50.",
        gs.verification_rows(),
        "Який найдешевший X-4070?",
        tool_counts=tool_counts,
    )
    assert rep.n_unsupported == 0


def test_catalog_search_special_chars_and_like_escaping(tmp_path):
    from query_awareness import search_catalog_window

    db = tmp_path / "special.db"
    _make_renamed_db(
        db,
        [
            ("s1", "Pure 100% Cotton Towel", "home", 199.0),
            ("s2", "Standard Cotton Towel", "home", 99.0),
            ("s3", "Controller X OR Y NOT Z", "parts", 500.0),
        ],
    )
    res_pct = search_catalog_window(str(db), query="100%", sort="price_asc", limit=10)
    assert res_pct["success"] is True
    assert res_pct["total"] == 1
    assert res_pct["rows"][0]["id"] == "s1"

    # FTS operator tokens ('OR', 'NOT') must be treated as literals, never syntax error
    res_ops = search_catalog_window(str(db), query="X OR Y NOT", sort="price_asc", limit=10)
    assert res_ops["success"] is True
    assert res_ops["total"] == 1
    assert res_ops["rows"][0]["id"] == "s3"


def test_react_cycle_state_propagated_safely_in_observation(renamed_large_db):
    from react_agent import ReActAgent

    captured_histories = []

    class _TwoStepRunner:
        def __init__(self):
            self.n = 0

        def generate_react_step(self, schema_prompt, user_query, dialog_history, current_step, tools_description):
            captured_histories.append((tools_description, list(dialog_history)))
            self.n += 1
            if self.n == 1:
                return {
                    "thought": "Перевіряю без сортування",
                    "action": "sql_query",
                    "action_input": {"query": "SELECT sku AS id, label AS name, cost AS price FROM catalog_entries WHERE grp = 'beta'"},
                }
            if self.n == 2:
                return {
                    "thought": "Вибірка обрізана без ORDER BY, уточнюю через catalog_search",
                    "action": "catalog_search",
                    "action_input": {"query": "X-4070", "sort": "price_asc", "limit": 5},
                }
            return {
                "thought": "Маю точний мінімум",
                "final_answer": "Знайдено 65 моделей, найдешевша — Module X-4070 Econom за 120.50.",
                "recommended_items": [{"id": "b_064", "name": "Module X-4070 Econom", "price": 120.50}],
            }

    agent = ReActAgent(db_path=renamed_large_db, max_steps=6)
    runner = _TwoStepRunner()
    st = agent.run_live_session("Який найдешевший X-4070?", runner, "Test schema")
    assert st.termination_reason == "SUCCESS_GROUNDED"
    assert "catalog_search" in captured_histories[0][0]
    # Step 2 observation carried exact truncation metadata + cycle state without leaking DB strings into tools_description
    obs_step1 = captured_histories[1][1][-1]["content"]
    assert "TRUNCATED_UNORDERED" in obs_step1
    assert '"total": 65' in obs_step1
    assert '"_cycle_state"' in obs_step1
    assert "Module X-4070" not in captured_histories[1][0]  # F32: DB text stays out of system tools_description


def test_profiler_emits_bounded_category_counts(renamed_large_db):
    from profiler import DomainAgnosticProfiler

    prof = DomainAgnosticProfiler(renamed_large_db).profile_database()
    ctx_str = DomainAgnosticProfiler.generate_compact_schema_context(prof.to_dict())
    assert "beta" in ctx_str and "65" in ctx_str
    assert "alpha" in ctx_str and "25" in ctx_str
