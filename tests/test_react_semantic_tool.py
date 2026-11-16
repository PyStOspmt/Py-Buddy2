"""ReAct `semantic_search` tool tests (Handoff 3, part D).

Real SemanticIndex over a SYNTHETIC fixture, offline HashEmbedder.
Covers: grounded hits, bounded observations, the untrusted-data
boundary, not-ready failure, the prompt line, and stagnation.
"""
from __future__ import annotations

import inspect
import json

import react_agent
from catalog_query import GroundingSet
from react_agent import ReActAgent, SessionState, observation_message
from semantic_fixtures import HashEmbedder, make_flat_db
from semantic_service import SemanticIndex


def _state(q="який запис підходить?"):
    st = SessionState(session_id="t", user_query=q)
    st.grounding = GroundingSet()
    return st


def _ready_index(tmp_path):
    db = make_flat_db(str(tmp_path / "t.db"))
    idx = SemanticIndex("dom_t", db,
                        embedder_factory=lambda: HashEmbedder())
    idx.sync_blocking(reason="test")
    return idx


def _agent(tmp_path):
    idx = _ready_index(tmp_path)
    return ReActAgent(db_path=idx.db_path, semantic_index=idx), idx


def test_tool_returns_grounded_hits(tmp_path):
    agent, idx = _agent(tmp_path)
    st = _state()
    out = agent.execute_tool(st, "semantic_search",
                             {"query": "entry number 3", "k": 5})
    assert out["success"] and out["status"] == "ok"
    assert out["hits"] and out["index_fresh"] is True
    h = out["hits"][0]
    assert set(h) == {"id", "name", "snippet", "score"}
    # hits enter the same grounding set as SQL rows (no price keys)
    assert h["id"] in st.grounding.ids
    assert h["name"] in st.grounding.names
    idx.close()


def test_observation_bounded_and_wrapped(tmp_path):
    agent, idx = _agent(tmp_path)
    st = _state()
    out = agent.execute_tool(st, "semantic_search",
                             {"query": "entry", "k": 10})
    assert out["success"]
    assert len(json.dumps(out, ensure_ascii=False)) <= 3000
    for h in out["hits"]:
        assert len(h["snippet"]) <= 240
    # reaches the model only inside the delimited untrusted block
    msg = observation_message(out)
    assert msg["role"] == "user"
    assert "<<<OBSERVATION_DATA" in msg["content"]
    assert "не інструкції" in msg["content"]
    idx.close()


def test_not_ready_is_typed_failure(tmp_path):
    db = make_flat_db(str(tmp_path / "nr.db"))
    idx = SemanticIndex("dom_t", db,
                        embedder_factory=lambda: HashEmbedder())
    agent = ReActAgent(db_path=db, semantic_index=idx)
    st = _state()
    out = agent.execute_tool(st, "semantic_search",
                             {"query": "entry", "k": 5})
    assert out["success"] is False
    assert "not ready" in out["error"]
    assert out["hits"] == []
    idx.close()


def test_unconfigured_is_typed_failure(tmp_path):
    agent = ReActAgent(db_path=None)
    st = _state()
    out = agent.execute_tool(st, "semantic_search",
                             {"query": "x", "k": 5})
    assert out["success"] is False and out["error"]


def test_prompt_lists_semantic_search():
    src = inspect.getsource(react_agent)
    # Handoff 4: the exact prompt line (hyde arg added)
    assert ('semantic_search: arguments {"query": "...", "k": 5, '
            '"hyde": false}') in src
    assert "Пошук за змістом" in src
    assert "Називай лише записи з результатів" in src


def test_duplicate_call_blocked_by_stagnation(tmp_path):
    agent, idx = _agent(tmp_path)
    st = _state()
    args = {"query": "entry number 3", "k": 5}
    first = agent.execute_tool(st, "semantic_search", args)
    assert first["success"]
    second = agent.execute_tool(st, "semantic_search", args)
    assert "error" in second and "stagnation" in second["error"].lower()
    assert st.is_terminated
    idx.close()


def test_tool_hyde_arg_and_prompt_line(tmp_path):
    idx = _ready_index(tmp_path)
    agent = ReActAgent(db_path=idx.db_path, semantic_index=idx)
    st = _state()
    out = agent.execute_tool(st, "semantic_search",
                             {"query": "entry", "k": 5, "hyde": True})
    # no llm_generate on the handle -> visible hybrid degradation
    assert out["success"]
    assert any("hyde unavailable" in n for n in out["notes"])
    idx.close()


def test_prompt_line_exact():
    src = inspect.getsource(react_agent)
    assert ('- semantic_search: arguments {"query": "...", "k": 5, '
            '"hyde": false}' in src)
    assert "гіпотеза лише спрямовує пошук і не є доказом" in src
