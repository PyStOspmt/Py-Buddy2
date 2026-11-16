"""Independent lexical baseline for the semantic-cycle study (docs/18).

A second, evaluator-owned BM25 implementation (in-memory SQLite FTS5)
used (a) as the baseline when ablating document folding and (b) as a
differential oracle for the store's own lexical channel. It re-implements
the query grammar from the written spec instead of importing it, so a bug
in the production query builder cannot hide itself.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def match_expression(query: str, prefix_truncate: bool) -> str | None:
    toks = [t for t in _TOKEN.findall(query.lower()) if len(t) >= 2]
    if not toks:
        return None
    parts = []
    for t in toks:
        if prefix_truncate and len(t) >= 5:
            parts.append('"%s"*' % t[:max(4, len(t) - 2)])
        else:
            parts.append('"%s"' % t)
    return " OR ".join(parts)


class IndependentBM25:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(eid UNINDEXED, body, "
            "tokenize='unicode61 remove_diacritics 2')")

    def add_many(self, rows: Iterable[tuple[str, str]]) -> int:
        n = 0
        with self.conn:
            for eid, text in rows:
                self.conn.execute("INSERT INTO t(eid, body) VALUES (?,?)",
                                  (eid, text))
                n += 1
        return n

    def query(self, q: str, k: int, prefix_truncate: bool = False
              ) -> list[str]:
        expr = match_expression(q, prefix_truncate)
        if expr is None:
            return []
        rows = self.conn.execute(
            "SELECT eid FROM t WHERE t MATCH ? ORDER BY bm25(t), eid "
            "LIMIT ?", (expr, k)).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self.conn.close()
