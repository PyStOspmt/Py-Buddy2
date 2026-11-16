"""TEST DOUBLES + synthetic fixtures for the Retrieval Fabric tests.

Everything here is evaluation machinery, never product evidence:
`HashEmbedder` is a deterministic bag-of-token-hash embedder -- it
proves sync/search LOGIC (which rows are re-embedded, how channels
fuse), not semantic quality. The SQLite sources built by the `make_*`
helpers are SYNTHETIC, explicitly labelled as such, and use
opaque-but-plausible names with no commerce-only assumptions.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Sequence

import numpy as np

_TOK_RE = re.compile(r"[^\W_]+", re.UNICODE)


class HashEmbedder:
    """Deterministic test-double embedder (unit-normed hashed bags).

    `count_tokens` is a TEST DOUBLE too: a regex word count + 2 fake
    special tokens -- enough to prove model-window accounting, never
    a claim about real tokenization."""

    def __init__(self, dim: int = 64, model_id: str | None = None,
                 max_tokens: int = 64):
        self.dim = int(dim)
        self.model_id = model_id or f"test:hash-embedder@dim{self.dim}"
        self.max_tokens = int(max_tokens)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [len(_TOK_RE.findall(t)) + 2 for t in texts]

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in _TOK_RE.findall(text.lower()):
            h = int.from_bytes(
                hashlib.blake2b(tok.encode(), digest_size=8).digest(),
                "little")
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = np.linalg.norm(v)
        if n:
            v /= n
        return v.astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_documents(texts)


def _exec_script(path: str, sql: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()


def make_flat_db(path: str, n: int = 8) -> str:
    """SYNTHETIC flat entity table: text, identifier-like, numeric,
    temporal, and sparse columns in one row."""
    rows = []
    for i in range(n):
        rows.append(
            f"({i + 1}, 'entry number {i + 1} alpha', "
            f"'a longer descriptive passage about entry {i + 1} "
            f"with several words in it', 'CODE-{i + 1:04d}', "
            f"{(i + 1) * 3.5}, '2026-01-{(i % 28) + 1:02d} 10:00', "
            "NULL)")
    _exec_script(path, f"""
    CREATE TABLE entries(
        eid INTEGER PRIMARY KEY,
        title TEXT,
        body TEXT,
        refcode TEXT,
        amount REAL,
        stamp TEXT,
        note TEXT);
    INSERT INTO entries VALUES {", ".join(rows)};
    """)
    return path


def make_normalized_db(path: str) -> str:
    """SYNTHETIC normalised source: entity + declared-FK dimension +
    many-to-many bridge, all with declared foreign keys."""
    _exec_script(path, """
    CREATE TABLE groups(gid INTEGER PRIMARY KEY, gname TEXT);
    INSERT INTO groups VALUES
        (1,'warm weather gear'), (2,'cold weather gear'),
        (3,'wet weather gear'), (4,'dry weather gear');
    CREATE TABLE labels(lid INTEGER PRIMARY KEY, lname TEXT);
    INSERT INTO labels VALUES
        (1,'zeta tag'), (2,'alpha tag'), (3,'mid tag');
    CREATE TABLE entries(
        eid INTEGER PRIMARY KEY,
        title TEXT,
        body TEXT,
        grp INTEGER REFERENCES groups(gid),
        amount REAL,
        code TEXT);
    CREATE TABLE entry_labels(
        entry INTEGER REFERENCES entries(eid),
        lab INTEGER REFERENCES labels(lid),
        PRIMARY KEY(entry, lab));
    """)
    conn = sqlite3.connect(path)
    try:
        conn.executemany(
            "INSERT INTO entries VALUES (?,?,?,?,?,?)",
            [(i + 1, f"item {i + 1} short name",
              f"extended narrative describing item {i + 1} fully",
              (i % 4) + 1, (i + 1) * 2.0, f"REF-{i + 1:03d}")
             for i in range(12)])
        conn.executemany(
            "INSERT INTO entry_labels(entry, lab) VALUES (?,?)",
            [(i + 1, (i % 3) + 1) for i in range(12)]
            + [(i + 1, ((i + 1) % 3) + 1)
               for i in range(12) if i % 2 == 0])
        conn.commit()
    finally:
        conn.close()
    return path


def make_eav_db(path: str, n: int = 8) -> str:
    """SYNTHETIC EAV source: entity table + (owner, axis, reading)
    attribute table detected by the structural EAV signature."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
        CREATE TABLE things(
            tid INTEGER PRIMARY KEY, label TEXT, descr TEXT);
        CREATE TABLE attr_rows(
            owner INTEGER, axis TEXT, reading TEXT);
        """)
        for i in range(n):
            conn.execute(
                "INSERT INTO things VALUES (?,?,?)",
                (i + 1, f"thing {i + 1}",
                 f"free text describing thing number {i + 1}"))
        axes = ["hue", "texture", "origin"]
        for i in range(n):
            for j, ax in enumerate(axes):
                val = (f"{12.5 + i:.1f}" if ax == "hue" and i % 3 == 0
                       else f"{ax} value {i % 4}")
                conn.execute(
                    "INSERT INTO attr_rows VALUES (?,?,?)",
                    (i + 1, ax, val))
        conn.commit()
    finally:
        conn.close()
    return path


def make_null_pk_and_empty_db(path: str) -> str:
    """SYNTHETIC edge source: one entity with a NULL text PK and one
    entity whose only non-key columns are all NULL (no_text)."""
    _exec_script(path, """
    CREATE TABLE rows_x(
        xid TEXT PRIMARY KEY, memo TEXT, filler TEXT);
    INSERT INTO rows_x VALUES ('a1', 'visible memo text', 'f');
    INSERT INTO rows_x VALUES ('b2', NULL, NULL);
    INSERT INTO rows_x VALUES (NULL, 'orphan text', 'g');
    """)
    return path


def projection(db_path: str):
    """Fresh (uncached) projection for a fixture database."""
    from source_projection import build_source_projection
    return build_source_projection(db_path, fresh=True)
