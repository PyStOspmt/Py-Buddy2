"""
P1.1 regressions: the NVFG solver / relaxation lattice in the serving path.

Previously `pareto_relax` fetched a scoring pool with a hidden 1.5x/-15
margin, and on a strict miss DROPPED EVERY bound silently (the lattice's
TOP element -- correct and useless). Now the tool must answer an
infeasible query with the cheapest proven relaxation: which bound moved,
from what to what, in the customer's own unit -- or an honest
UNSATISFIABLE/UNKNOWN, never a silent all-bounds-off fallback.

Cases mirror the solver's contracts:
  A  strict set empty, one bound rescues          -> RELAXED, named move
  B  no single bound rescues, two together do     -> RELAXED, both moves
  C  category genuinely empty                     -> honest empty result
  D  strict set non-empty                         -> FEASIBLE, nothing moved
  E  constrained axis absent from the source      -> axis skipped, reported
  F  relaxed answers are grounded, not fabricated -> items enter the GS
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


from react_agent import ReActAgent, SessionState


def _db(tmp_path, rows):
    db = tmp_path / "shop.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "price REAL, stock_quantity INTEGER, performance_score REAL)")
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(db)


def _relax(agent, **args):
    state = SessionState(session_id="s1", user_query="порадь ноутбук")
    res = agent.execute_tool(state, "pareto_relax", args)
    return res, res.get("result", res)


ROWS = [
    ("p1", "Бюджетний", "laptops", 800.0, 5, 60.0),
    ("p2", "Середній", "laptops", 1500.0, 5, 75.0),
    ("p3", "Потужний", "laptops", 3200.0, 5, 95.0),
    ("p4", "Тихий", "laptops", 2600.0, 5, 30.0),
]


def test_A_single_bound_relaxation_is_named(tmp_path):
    # budget<=1000 AND perf>=70 is empty. Cheapest proven rescue is
    # performance 70 -> 60 (share 0.143), which admits Бюджетний (800/60) --
    # vs budget -> 1500 (share 0.5). The move must be named, not hidden.
    agent = ReActAgent(db_path=_db(tmp_path, ROWS))
    _raw, res = _relax(agent, budget_max=1000.0, min_performance=70.0,
                       category="laptops")
    assert res["success"] is True
    assert res.get("nvfg_status") == "RELAXED"
    assert res.get("relaxation_text"), "a relaxation must be reported, not hidden"
    assert "продуктивн" in res["relaxation_text"], res["relaxation_text"]
    assert res.get("relative_regret", 1.0) < 0.5
    assert res["option_a"] is not None


def test_B_cheapest_move_is_chosen_not_all_drop(tmp_path):
    # No item satisfies budget<=1000 AND perf>=90. Two single-move rescues
    # exist: budget -> 3200 (share 2.2) or performance -> 60 (share 0.33).
    # The old code dropped BOTH bounds; the solver must take the cheaper
    # named move -- and only that one.
    agent = ReActAgent(db_path=_db(tmp_path, ROWS))
    _raw, res = _relax(agent, budget_max=1000.0, min_performance=90.0,
                       category="laptops")
    assert res["success"] is True
    assert res.get("nvfg_status") == "RELAXED"
    text = res.get("relaxation_text", "")
    assert "продуктивн" in text, text
    assert res["option_a"] is not None
    assert res["option_a"]["name"] == "Бюджетний"


def test_C_empty_category_reports_honestly(tmp_path):
    agent = ReActAgent(db_path=_db(tmp_path, ROWS))
    _raw, res = _relax(agent, budget_max=100.0, min_performance=10.0,
                       category="nonexistent")
    assert res["success"] is True
    assert res.get("nvfg_status") in ("EMPTY_QUERY", "UNSATISFIABLE")
    assert res["pareto_front_count"] == 0
    assert res["option_a"] is None


def test_D_feasible_query_moves_nothing(tmp_path):
    agent = ReActAgent(db_path=_db(tmp_path, ROWS))
    _raw, res = _relax(agent, budget_max=5000.0, min_performance=50.0,
                       category="laptops")
    assert res.get("nvfg_status") == "FEASIBLE"
    assert not res.get("relaxation")
    assert res.get("relative_regret", 0.0) == 0.0


def test_E_missing_axis_is_skipped_and_reported(tmp_path):
    rows = [("a", "Товар без оцінки", "misc", 500.0, 3, None)]
    agent = ReActAgent(db_path=_db(tmp_path, rows))
    _raw, res = _relax(agent, budget_max=1000.0, min_performance=50.0,
                       category="misc")
    assert res["success"] is True
    # performance cannot be evaluated on this source -- it must be skipped,
    # never silently satisfied or silently violated.
    assert "performance" in str(res.get("axes_skipped", [])) or \
        res.get("axes_evaluated") == ["price"]
    assert res["option_a"] is not None


def test_F_relaxed_items_are_grounded(tmp_path):
    agent = ReActAgent(db_path=_db(tmp_path, ROWS))
    state = SessionState(session_id="s6", user_query="q")
    agent.execute_tool(state, "pareto_relax",
                       {"budget_max": 1000.0, "min_performance": 70.0,
                        "category": "laptops"})
    names = [r["name"] for r in agent._gs(state).projected_recommendations()]
    assert names, "relaxed items must enter the grounding set"
    assert any("Бюджетний" in n for n in names)
