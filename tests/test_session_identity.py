"""
F30: session identity is server-owned. A client may resume a session it
already holds; it cannot create one with a chosen id, collide with an
existing session, or pin the shared default id.

Uses a real SQLite session store in a temporary directory.
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "core")))

from react_agent import SessionState
from session_store import SQLiteSessionStore, resolve_session_id

MINTED = re.compile(r"^s-[0-9a-f]{32}$")


def _store(tmp_path):
    return SQLiteSessionStore(str(tmp_path / "sessions.db"))


def _save(store, sid):
    store.save_session(SessionState(session_id=sid, user_query="q"))


def test_unknown_client_id_is_replaced_by_minted_id(tmp_path):
    store = _store(tmp_path)
    sid = resolve_session_id("web_attacker_chosen", store)
    assert sid != "web_attacker_chosen"
    assert MINTED.fullmatch(sid)


def test_existing_session_is_resumed_by_its_id(tmp_path):
    store = _store(tmp_path)
    _save(store, "s-" + "a" * 32)
    assert resolve_session_id("s-" + "a" * 32, store) == "s-" + "a" * 32


def test_chosen_id_cannot_collide_with_existing_session_of_another_user(tmp_path):
    store = _store(tmp_path)
    victim_sid = "web_victimsession1"
    _save(store, victim_sid)
    assert resolve_session_id(victim_sid, store) == victim_sid
    fresh = resolve_session_id("web_victimsession2", store)
    assert fresh != victim_sid and MINTED.fullmatch(fresh)


def test_shared_default_id_is_never_resumed(tmp_path):
    store = _store(tmp_path)
    _save(store, "web_default")
    sid = resolve_session_id("web_default", store)
    assert sid != "web_default" and MINTED.fullmatch(sid)


def test_missing_or_malformed_ids_get_fresh_minted_ids(tmp_path):
    store = _store(tmp_path)
    ids = {resolve_session_id(None, store), resolve_session_id("", store),
           resolve_session_id("bad id with spaces", store)}
    assert len(ids) == 3
    assert all(MINTED.fullmatch(s) for s in ids)


def test_minted_session_is_persisted_by_caller_and_resumable(tmp_path):
    store = _store(tmp_path)
    sid = resolve_session_id(None, store)
    _save(store, sid)
    assert resolve_session_id(sid, store) == sid
