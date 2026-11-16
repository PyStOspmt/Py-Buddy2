"""
Py-Buddy: SQLite Session State Persistence
Provides thread-safe atomic storage and retrieval of user SessionState objects.
Ensures session continuity across server restarts and multi-worker gateway processes.
"""

import re
import sqlite3
import json
import os
import uuid
from typing import List, Optional, Protocol
from datetime import datetime, timezone

from react_agent import SessionState

SHARED_SESSION_ID = "web_default"
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class _SessionLookup(Protocol):
    def get_session(self, session_id: str) -> Optional[SessionState]: ...


def resolve_session_id(client_id: Optional[str], store: _SessionLookup) -> str:
    """
    F30: a client may resume a session it already holds; it cannot choose a
    new id, collide with another session, or pin the shared default id.
    Anything else gets a server-minted id (uuid4, 122 random bits) that the
    caller must return to the client.
    """
    resumable = (client_id is not None and client_id != SHARED_SESSION_ID
                 and _SESSION_ID_RE.fullmatch(client_id) is not None)
    if resumable and store.get_session(client_id) is not None:
        return client_id
    return f"s-{uuid.uuid4().hex}"

class SQLiteSessionStore:
    """
    Persistent SQLite repository for active consultation sessions.
    Supports atomic commits and WAL mode for high concurrent throughput.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_query TEXT,
                    state_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);")

    def save_session(self, session: SessionState) -> None:
        """Saves or updates a SessionState atomically."""
        state_json = session.to_json()
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, user_query, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_query=excluded.user_query,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at;
            """, (session.session_id, session.user_query, state_json, now))

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieves and deserializes a SessionState by its identifier."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT state_json FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row["state_json"])
            return SessionState.from_dict(data)

    def list_sessions(self) -> List[str]:
        """Returns all registered session identifiers."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC")
            return [row["session_id"] for row in cursor.fetchall()]

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session from persistence. Returns True if deleted."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cursor.rowcount > 0

    def get_recent_sessions(self, limit: int = 5) -> List[SessionState]:
        """Returns the most recent sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT state_json FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,))
            return [SessionState.from_dict(json.loads(row["state_json"])) for row in cursor.fetchall()]

