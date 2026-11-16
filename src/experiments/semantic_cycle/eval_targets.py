"""Evaluation targets with INDEPENDENT ground truth (docs/18).

* label_set targets: the relevant set is every record carrying a label that
  the data owner assigned (cataloguer subject, newsgroup, food category tag).
  The system under test never sees these sets.
* record targets: one specific record the requester "vaguely remembers".

This module is evaluation-only: it contains source-specific SQL because it
encodes the evaluator's knowledge of each corpus, never production logic.
"""

from __future__ import annotations

import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SEED = 20261112
SET_MIN, SET_MAX = 8, 150
N_LABELS, N_RECORDS = 60, 100


@dataclass(frozen=True)
class Target:
    kind: str
    source: str
    target_id: str
    forbidden_text: str
    relevant_ids: tuple[str, ...]
    brief: str
    examples: tuple[str, ...]


def connect(path: Path) -> sqlite3.Connection:
    uri = "file:" + str(path).replace("\\", "/") + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _clean_tag(tag: str) -> str:
    return re.sub(r"^[a-z]{2}:", "", tag).replace("-", " ").strip()


def _group(rows: list[tuple[str, object]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for label, rid in rows:
        out.setdefault(label, []).append(str(rid))
    return out


def _pick_labels(groups: dict[str, list[str]], rng: random.Random,
                 n: int) -> list[str]:
    ok = sorted(k for k, v in groups.items() if SET_MIN <= len(v) <= SET_MAX)
    rng.shuffle(ok)
    return sorted(ok[:n])


def library_targets(db: Path) -> list[Target]:
    rng = random.Random(SEED + 1)
    c = connect(db)
    titles = dict(c.execute("SELECT book_id, title FROM books"))
    groups = _group(c.execute(
        "SELECT s.label, bs.book_id FROM book_subjects bs "
        "JOIN subjects s ON s.subject_id = bs.subject_id").fetchall())
    out = [Target("label_set", "library", lab, lab, tuple(groups[lab]),
                  f'subject heading "{lab}"',
                  tuple(titles[int(i)] for i in rng.sample(groups[lab], 3)))
           for lab in _pick_labels(groups, rng, N_LABELS)]
    for bid in sorted(rng.sample(sorted(titles), N_RECORDS)):
        au = [r[0] for r in c.execute(
            "SELECT a.full_name FROM book_authors x JOIN authors a "
            "ON a.author_id = x.author_id WHERE x.book_id = ?", (bid,))]
        su = [r[0] for r in c.execute(
            "SELECT s.label FROM book_subjects x JOIN subjects s "
            "ON s.subject_id = x.subject_id WHERE x.book_id = ?", (bid,))]
        text = f"title: {titles[bid]}; authors: {'; '.join(au)}; " \
               f"subjects: {'; '.join(su)}"
        out.append(Target("record", "library", str(bid), text, (str(bid),),
                          text, ()))
    c.close()
    return out


def forum_targets(db: Path) -> list[Target]:
    rng = random.Random(SEED + 2)
    c = connect(db)
    boards = dict(c.execute("SELECT board_id, name FROM boards"))
    bodies = dict(c.execute("SELECT post_id, body FROM posts"))
    by_board = _group(c.execute(
        "SELECT b.name, p.post_id FROM posts p JOIN boards b "
        "ON b.board_id = p.board_id").fetchall())
    out = []
    for name in sorted(boards.values()):
        ids = by_board[name]
        ex = tuple(bodies[int(i)][:160] for i in rng.sample(ids, 3))
        out.append(Target("label_set", "forum", name,
                          name.replace(".", " "), tuple(ids),
                          f'discussion board "{name}"', ex))
    for pid in sorted(rng.sample(sorted(bodies), N_RECORDS)):
        out.append(Target("record", "forum", str(pid), bodies[pid],
                          (str(pid),), bodies[pid][:1200], ()))
    c.close()
    return out


def food_targets(db: Path) -> list[Target]:
    rng = random.Random(SEED + 3)
    c = connect(db)
    names = {r[0]: (r[1], r[2], r[3]) for r in c.execute(
        "SELECT id, name, brand, category FROM products")}
    groups = _group(c.execute(
        "SELECT meta_value, product_id FROM product_eav_meta "
        "WHERE meta_key = 'category_tag'").fetchall())
    out = [Target("label_set", "food", lab, _clean_tag(lab),
                  tuple(groups[lab]),
                  f'food category "{_clean_tag(lab)}"',
                  tuple(names[i][0] for i in rng.sample(groups[lab], 3)))
           for lab in _pick_labels(groups, rng, N_LABELS)]
    usable = sorted(k for k, v in names.items()
                    if v[0] and len(v[0]) >= 6 and not v[0].isdigit())
    for pid in sorted(rng.sample(usable, N_RECORDS)):
        nm, br, cat = names[pid]
        text = f"product: {nm}; brand: {br}; category: {_clean_tag(cat or '')}"
        out.append(Target("record", "food", str(pid), text, (str(pid),),
                          text, ()))
    c.close()
    return out


def all_targets() -> list[Target]:
    d = ROOT / "data"
    return (library_targets(d / "eval_semantic" / "library.db")
            + forum_targets(d / "eval_semantic" / "forum.db")
            + food_targets(d / "real_off.db"))
