"""Held-out evaluation sources for the semantic-cycle study (docs/18).

Library: Project Gutenberg catalog (public-domain metadata), normalised
         relational schema with declared FKs and many-to-many bridges.
Forum:   20 Newsgroups posts (bodies only, headers/footers/quotes removed),
         one dimension table of boards.
Food:    the existing `data/real_off.db` is used unchanged (not rebuilt).

Everything is deterministic (fixed seeds). The manifest written next to the
docs records input hashes, seeds, counts and output hashes; it is the
provenance of the evaluation corpora, not a result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "eval_semantic"
RAW_PG = Path(os.environ["TEMP"]) / "ainsa_ext" / "pg_catalog.csv"
SEED = 20261112
N_BOOKS = 8000
N_POSTS = 6000
POST_MIN_CHARS, POST_MAX_CHARS = 200, 3000

LIBRARY_DDL = """
CREATE TABLE books(book_id INTEGER PRIMARY KEY, title TEXT NOT NULL,
    issued TEXT, lang TEXT, loc_class TEXT);
CREATE TABLE authors(author_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL UNIQUE);
CREATE TABLE book_authors(
    book_id INTEGER NOT NULL REFERENCES books(book_id),
    author_id INTEGER NOT NULL REFERENCES authors(author_id));
CREATE TABLE subjects(subject_id INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE);
CREATE TABLE book_subjects(
    book_id INTEGER NOT NULL REFERENCES books(book_id),
    subject_id INTEGER NOT NULL REFERENCES subjects(subject_id));
CREATE TABLE shelves(shelf_id INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE);
CREATE TABLE book_shelves(
    book_id INTEGER NOT NULL REFERENCES books(book_id),
    shelf_id INTEGER NOT NULL REFERENCES shelves(shelf_id));
CREATE INDEX ix_ba_book ON book_authors(book_id);
CREATE INDEX ix_bs_book ON book_subjects(book_id);
CREATE INDEX ix_bsh_book ON book_shelves(book_id);
"""

FORUM_DDL = """
CREATE TABLE boards(board_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE posts(post_id INTEGER PRIMARY KEY,
    board_id INTEGER NOT NULL REFERENCES boards(board_id),
    body TEXT NOT NULL);
CREATE INDEX ix_posts_board ON posts(board_id);
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_labels(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(";") if p.strip()]


def intern(conn: sqlite3.Connection, table: str, key: str, col: str,
           cache: dict[str, int], label: str) -> int:
    if label not in cache:
        cur = conn.execute(f"INSERT INTO {table}({col}) VALUES (?)", (label,))
        cache[label] = int(cur.lastrowid)
    return cache[label]


def pick_books() -> list[dict[str, str]]:
    with open(RAW_PG, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["Type"] == "Text" and r["Language"] == "en"
                and r["Subjects"].strip()]
    random.Random(SEED).shuffle(rows)
    return sorted(rows[:N_BOOKS], key=lambda r: int(r["Text#"]))


def build_library(path: Path) -> dict[str, object]:
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(LIBRARY_DDL)
    authors: dict[str, int] = {}
    subjects: dict[str, int] = {}
    shelves: dict[str, int] = {}
    books = pick_books()
    with conn:
        for r in books:
            bid = int(r["Text#"])
            title = " ".join(r["Title"].split())
            conn.execute("INSERT INTO books VALUES (?,?,?,?,?)",
                         (bid, title, r["Issued"], r["Language"], r["LoCC"]))
            for name in [a.strip() for a in r["Authors"].split(";") if a.strip()]:
                aid = intern(conn, "authors", "a", "full_name", authors, name)
                conn.execute("INSERT INTO book_authors VALUES (?,?)", (bid, aid))
            for lab in split_labels(r["Subjects"]):
                sid = intern(conn, "subjects", "s", "label", subjects, lab)
                conn.execute("INSERT INTO book_subjects VALUES (?,?)", (bid, sid))
            for lab in split_labels(r["Bookshelves"]):
                hid = intern(conn, "shelves", "h", "label", shelves, lab)
                conn.execute("INSERT INTO book_shelves VALUES (?,?)", (bid, hid))
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("books", "authors", "subjects", "shelves",
                        "book_authors", "book_subjects", "book_shelves")}
    conn.close()
    return {"counts": counts}


def build_forum(path: Path) -> dict[str, object]:
    import sklearn
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="all",
                              remove=("headers", "footers", "quotes"))
    keep = [(i, int(t), " ".join(b.split())) for i, (t, b)
            in enumerate(zip(data.target, data.data))]
    keep = [(i, t, b) for i, t, b in keep
            if POST_MIN_CHARS <= len(b) <= POST_MAX_CHARS]
    seen: set[str] = set()
    uniq = []
    for i, t, b in keep:
        if b not in seen:
            seen.add(b)
            uniq.append((i, t, b))
    random.Random(SEED).shuffle(uniq)
    chosen = sorted(uniq[:N_POSTS])
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(FORUM_DDL)
    with conn:
        for bid, name in enumerate(data.target_names, start=1):
            conn.execute("INSERT INTO boards VALUES (?,?)", (bid, name))
        for i, t, b in chosen:
            conn.execute("INSERT INTO posts VALUES (?,?,?)", (i + 1, t + 1, b))
    counts = {"boards": len(data.target_names), "posts": len(chosen),
              "candidates_after_filters": len(uniq),
              "raw_posts": len(data.data)}
    conn.close()
    return {"counts": counts, "sklearn": sklearn.__version__,
            "dataset": "fetch_20newsgroups(subset='all', "
                       "remove=('headers','footers','quotes'))"}


def main(argv: Iterable[str]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lib, forum = OUT_DIR / "library.db", OUT_DIR / "forum.db"
    manifest = {
        "purpose": "held-out evaluation sources for docs/18 (semantic cycle)",
        "seed": SEED,
        "library": {
            "origin": "https://gutenberg.org/cache/epub/feeds/pg_catalog.csv",
            "raw_sha256": sha256_file(RAW_PG),
            "filters": "Type=Text, Language=en, Subjects non-empty",
            "sample": N_BOOKS, **build_library(lib)},
        "forum": {
            "origin": "scikit-learn fetch_20newsgroups (public benchmark)",
            "sample": N_POSTS,
            "length_filter_chars": [POST_MIN_CHARS, POST_MAX_CHARS],
            **build_forum(forum)},
        "food": {"origin": "data/real_off.db (Open Food Facts extract, "
                           "project-canonical schema, unchanged)",
                 "sha256": sha256_file(ROOT / "data" / "real_off.db")},
    }
    manifest["library"]["db_sha256"] = sha256_file(lib)
    manifest["forum"]["db_sha256"] = sha256_file(forum)
    out = ROOT / "docs" / "semantic_cycle_sources_20261112.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
