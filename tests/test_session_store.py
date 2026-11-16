"""
TDD Suite for Py-Buddy SQLite Session Persistence (src/session_store.py).
Validates atomic persistence, retrieval, delta updates, and serialization of SessionState.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from active_inference import UserBeliefState
from react_agent import SessionState
from session_store import SQLiteSessionStore


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_session_store_lifecycle(temp_db_path):
    store = SQLiteSessionStore(temp_db_path)
    
    # 1. Non-vacuous initial state
    assert len(store.list_sessions()) == 0

    # 2. Create and save a session
    belief = UserBeliefState(dimensions={"price_sensitivity": 0.8, "performance_focus": 0.9}, entropy=1.2)
    session_v0 = SessionState(
        session_id="tg_12345",
        user_query="Шукаю процесор до 10000 грн",
        belief_state=belief,
        cart=[{"id": "cpu_am5_2", "name": "AMD Ryzen 5 7600X"}],
        execution_history=[{"step": 1, "thought": "Пошук ціни"}],
        current_step=1,
        max_steps=5,
        is_terminated=False,
        final_response="Рекомендовано Ryzen 5 7600X",
        conversation_history=[{"role": "user", "content": "Шукаю процесор до 10000 грн"}],
    )

    store.save_session(session_v0)
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert "tg_12345" in sessions

    # 3. Retrieve and assert exact deserialization
    retrieved = store.get_session("tg_12345")
    assert retrieved is not None
    assert retrieved.session_id == "tg_12345"
    assert retrieved.user_query == "Шукаю процесор до 10000 грн"
    assert retrieved.cart[0]["id"] == "cpu_am5_2"
    assert retrieved.belief_state.dimensions["price_sensitivity"] == 0.8
    assert retrieved.current_step == 1
    assert retrieved.conversation_history == [{"role": "user", "content": "Шукаю процесор до 10000 грн"}]

    # 4. State delta update (V0 != V1)
    retrieved.current_step = 2
    retrieved.cart.append({"id": "mb_am5_msi", "name": "MSI B650"})
    store.save_session(retrieved)

    updated = store.get_session("tg_12345")
    assert updated.current_step == 2
    assert len(updated.cart) == 2
    assert updated.cart[1]["id"] == "mb_am5_msi"

    # 5. Delete session
    deleted = store.delete_session("tg_12345")
    assert deleted is True
    assert store.get_session("tg_12345") is None
    assert len(store.list_sessions()) == 0
