"""Apply the frozen point-update scenario to a library SQLite copy and
independently verify the index sidecar (docs/18, P5).

`apply_scenario` edits ONLY the working copy handed to it. The vector
checks read the sidecar read-only (manifest SQL + raw file reads) so they
do not depend on the system's own statistics.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def _intern(conn: sqlite3.Connection, table: str, key: str, col: str,
            label: str) -> int:
    conn.execute(f"INSERT OR IGNORE INTO {table}({col}) VALUES (?)",
                 (label,))
    return conn.execute(f"SELECT {key} FROM {table} WHERE {col}=?",
                        (label,)).fetchone()[0]


def _write_links(conn, bid: int, rec: dict) -> None:
    for name in rec["authors"]:
        conn.execute("INSERT INTO book_authors VALUES (?,?)",
                     (bid, _intern(conn, "authors", "author_id",
                                   "full_name", name)))
    for lab in rec["subjects"]:
        conn.execute("INSERT INTO book_subjects VALUES (?,?)",
                     (bid, _intern(conn, "subjects", "subject_id",
                                   "label", lab)))
    for lab in rec["shelves"]:
        conn.execute("INSERT INTO book_shelves VALUES (?,?)",
                     (bid, _intern(conn, "shelves", "shelf_id",
                                   "label", lab)))


def apply_scenario(db_path: str, scenario: dict) -> dict:
    """Insert / update / delete exactly as frozen in the scenario file.
    Returns the expected id sets (strings)."""
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            for item in scenario["inserts"]:
                r = item["record"]
                bid = int(r["source_book_id"])
                conn.execute("INSERT INTO books VALUES (?,?,?,?,?)",
                             (bid, r["title"], r["issued"], r["lang"],
                              r["loc_class"]))
                _write_links(conn, bid, r)
            for item in scenario["updates"]:
                r, bid = item["new_content"], int(item["book_id"])
                conn.execute(
                    "UPDATE books SET title=?, issued=?, lang=?, "
                    "loc_class=? WHERE book_id=?",
                    (r["title"], r["issued"], r["lang"], r["loc_class"],
                     bid))
                for t in ("book_authors", "book_subjects", "book_shelves"):
                    conn.execute(f"DELETE FROM {t} WHERE book_id=?", (bid,))
                _write_links(conn, bid, r)
            for item in scenario["deletes"]:
                bid = int(item["book_id"])
                for t in ("book_authors", "book_subjects", "book_shelves"):
                    conn.execute(f"DELETE FROM {t} WHERE book_id=?", (bid,))
                conn.execute("DELETE FROM books WHERE book_id=?", (bid,))
    finally:
        conn.close()
    return {"inserted": [str(i["record"]["source_book_id"])
                         for i in scenario["inserts"]],
            "updated": [str(i["book_id"]) for i in scenario["updates"]],
            "deleted": [str(i["book_id"]) for i in scenario["deletes"]]}


def sidecar_snapshot(db_path: str) -> dict:
    """Read-only view of `<db>.arf`: {entity_id: sha256(vector bytes)},
    file bytes, slot count, dim, generation. Never writes."""
    d = Path(db_path + ".arf")
    uri = "file:" + str(d / "manifest.sqlite").replace("\\", "/") + "?mode=ro"
    c = sqlite3.connect(uri, uri=True)
    try:
        meta = dict(c.execute("SELECT key, value FROM meta"))
        rows = c.execute("SELECT slot, entity_id FROM records "
                         "WHERE live=1").fetchall()
        n_slots = c.execute("SELECT COALESCE(MAX(slot)+1,0) FROM records"
                            ).fetchone()[0]
    finally:
        c.close()
    dim = int(meta["dim"])
    vfile = d / meta["vector_file"]
    raw = vfile.read_bytes()
    row = dim * 2
    hashes = {eid: hashlib.sha256(raw[s * row:(s + 1) * row]).hexdigest()
              for s, eid in rows}
    return {"hashes": hashes, "file_bytes": len(raw), "slots": int(n_slots),
            "dim": dim, "generation": int(meta.get("generation", 0)),
            "live": len(rows)}


def compare_snapshots(before: dict, after: dict, expected: dict) -> dict:
    """Independent P5 checks between two sidecar snapshots."""
    ch, ah = before["hashes"], after["hashes"]
    touched = set(expected["inserted"]) | set(expected["updated"]) \
        | set(expected["deleted"])
    unchanged_ids = [e for e in ch if e not in touched]
    mism = [e for e in unchanged_ids if ah.get(e) != ch[e]]
    appended = (after["file_bytes"] - before["file_bytes"]) // (
        after["dim"] * 2)
    return {
        "unchanged_checked": len(unchanged_ids),
        "unchanged_vector_mismatches": len(mism),
        "appended_vector_slots": appended,
        "expected_embeds": len(expected["inserted"]) + len(
            expected["updated"]),
        "deleted_still_live": [e for e in expected["deleted"] if e in ah],
        "inserted_missing": [e for e in expected["inserted"]
                             if e not in ah],
        "updated_vector_changed": sum(
            1 for e in expected["updated"] if ah.get(e) != ch.get(e)),
        "live_before": before["live"], "live_after": after["live"]}


def load_scenario(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
