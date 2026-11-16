"""
Py-Buddy: Catalog full-text search index (FTS5 trigram).

Linear `LIKE '%term%'` scans measured 0.27 s at 200k rows and 0.49 s at
500k rows (claim register T5). An in-memory FTS5 index with the trigram
tokenizer preserves exact substring semantics -- including inside-token
matches ('zen' finds 'Ryzen') and case-insensitive Cyrillic -- while
bounding the scan to indexed candidates.

The index is per-bundle process state: it is built lazily on the first
qualifying search, read-only against the source database, and rebuilt
when the source file changes (mtime/size version stamp, same policy as
the engine's lazy rebuild). Failures degrade visibly to the plain LIKE
path, never silently.
"""

import os
import sqlite3
import threading
import time
from typing import List, Optional, Tuple

# Trigram needs at least 3 characters to express a match.
MIN_TERM_LENGTH = 3


def _fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        conn.close()
        return True
    except sqlite3.Error:
        return False


class CatalogSearchIndex:
    """Lazy in-memory FTS5 index over (name, category) text of entities."""

    def __init__(self) -> None:
        # check_same_thread=False + lock: the index is built lazily on
        # whichever request thread first searches, then read from other
        # threadpool workers (same reason as ChannelIndex).
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._version: Optional[Tuple[float, int]] = None
        self.status: str = "not_built"   # not_built | ready | failed
        self.fail_reason: Optional[str] = None
        self.indexed_rows: int = 0
        self.built_seconds: float = 0.0

    # ------------------------------------------------------------------

    def ensure_built(self, db_path: str, entity_table: str,
                     ident_col: str, text_cols: List[str]) -> str:
        """Build or rebuild the index for this source version.

        Returns "ready" | "failed" | "unsupported"."""
        st = os.stat(db_path)
        version = (st.st_mtime, st.st_size)
        if self.status == "ready" and self._version == version:
            return "ready"
        if self.status == "failed" and self._version == version:
            return "failed"
        if self.status == "unsupported":
            return "unsupported"
        if not _fts5_available():
            self.status = "unsupported"
            self.fail_reason = "sqlite3 build lacks FTS5 trigram"
            return "unsupported"

        t0 = time.perf_counter()
        try:
            from source_projection import quote_ident
            q = quote_ident
            txt_expr = " || ' ' || ".join(
                f"COALESCE({q(c)}, '')" for c in text_cols)
            src = sqlite3.connect(
                "file:%s?mode=ro" % os.path.abspath(db_path).replace("\\", "/"),
                uri=True)
            mem = sqlite3.connect(":memory:", check_same_thread=False)
            mem.execute(
                "CREATE VIRTUAL TABLE ft USING fts5(txt, eid UNINDEXED, "
                "tokenize='trigram')")
            cur = src.execute(
                f"SELECT {q(ident_col)} AS eid, {txt_expr} AS txt "
                f"FROM {q(entity_table)}")
            mem.executemany("INSERT INTO ft(txt, eid) VALUES (?, ?)",
                            ((r[1], str(r[0])) for r in cur))
            src.close()
            self._conn = mem
            self._version = version
            self.indexed_rows = mem.execute(
                "SELECT COUNT(*) FROM ft").fetchone()[0]
            self.built_seconds = round(time.perf_counter() - t0, 3)
            self.status = "ready"
            self.fail_reason = None
            return "ready"
        except Exception as exc:
            self.status = "failed"
            self.fail_reason = f"{type(exc).__name__}: {exc}"
            return "failed"

    # ------------------------------------------------------------------

    def total_matches(self, term: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM ft WHERE ft MATCH ?", (term,)
            ).fetchone()[0]

    def page_ids(self, term: str, limit: int, offset: int) -> List[str]:
        """Entity ids for one page, ordered by id for determinism."""
        with self._lock:
            return [r[0] for r in self._conn.execute(
                "SELECT eid FROM ft WHERE ft MATCH ? "
                "ORDER BY eid LIMIT ? OFFSET ?",
                (term, limit, offset))]

    def usable_for(self, term: str) -> bool:
        return (self.status == "ready"
                and len(term.strip()) >= MIN_TERM_LENGTH)
