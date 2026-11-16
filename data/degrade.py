"""
Py-Buddy: Defect injectors for the robustness polygon.

The Adaptive Data Fabric thesis (docs/09_ADAPTIVE_DATA_FABRIC.md, §6) is only
testable against sources that are *actually* broken. Real catalogues are dirty
in ways nobody planned, so the project keeps a named taxonomy of nine defect
classes and applies them to a copy of any database.

Design rules
------------
* **The taxonomy is data, not prose.** ``DEFECT_CLASSES`` is the single source
  of truth. ``data/generate_marketplace_scale.py`` imports the *value-level*
  helpers below, so the synthetic "dirty by construction" profile and the
  "degraded by injection" profile cannot drift apart.
* **Every injector returns the number of objects it touched.** Those numbers
  are what the P4 reports (``docs/adaptation_*_YYYYMMDD.json``) cite as evidence
  that a defect was injected rather than merely named. A class whose count is
  zero must carry a ``note`` saying why -- "the source has no such values" is a
  result, a bare zero is indistinguishable from a broken injector.
* **``rate`` counts *eligible* objects, not rows.** A class damages
  ``max(1, round(rate * eligible))`` objects, where "eligible" means the ones
  this defect can actually apply to (a value with a unit, a key with Cyrillic).
  A source whose values are 20% unit-bearing would otherwise be rewritten
  completely at ``rate=0.2`` while a 100%-unit-bearing one stayed at 20%, and
  the robustness score would measure value composition, not resilience.
* **An unfamiliar schema is a result, not an error.** ``discover_schema``
  returns ``None`` for any role it cannot find, the class is then reported as
  ``not applicable: source lacks <role>``, and no injector raises -- verified on
  a source with only two TEXT columns and on an empty table.
* **Injection never repairs the source.** It copies, breaks, and reports.
  A catalogue is never edited in place (docs/09 §6: "жодне джерело не
  'виправляється' вручну").
* **Determinism.** All randomness comes from an explicit ``random.Random(seed)``,
  so both the contents and the size of a degraded catalogue are reproducible
  from (source, classes, seed) alone.

Usage
-----
    python data/degrade.py --list
    python data/degrade.py --db data/catalog.db --all \
        --outdir data/degraded --report data/degraded/_report_catalog.json
    python data/degrade.py --db data/real_off.db --classes 3,4 \
        --out data/degraded/real_off__nulls_text.db

``--all`` writes one degraded copy per class (the polygon's inputs);
``--classes all`` writes a single copy with every class applied.
"""

import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import time

#: Nine defect classes (docs/09 §6). Order is part of the contract: the
#: numbers in the paper's robustness table are keyed by these ids.
DEFECT_CLASSES = {
    1: "no_primary_key",
    2: "entity_duplicates_conflicting",
    3: "null_in_key_attributes",
    4: "number_stored_as_text",
    5: "unit_in_key_instead_of_value",
    6: "mixed_units_same_key",
    7: "swapped_columns",
    8: "orphan_rows",
    9: "encoding_latin_for_cyrillic",
}

#: Human-readable description printed by ``--list`` and quoted in the paper.
DEFECT_DESCRIPTIONS = {
    1: "таблиця без PRIMARY KEY (рядок неможливо адресувати)",
    2: "дублікати сутностей з конфліктними значеннями",
    3: "NULL / порожнє у ключових атрибутах",
    4: "число збережене як TEXT (локальна кома, пробіли-розділювачі)",
    5: "одиниця в КЛЮЧІ замість значення (`power_500w`)",
    6: "змішані одиниці на тому самому ключі",
    7: "переплутані колонки (`price` <-> `rating`)",
    8: "сирота-рядок (порушений зв'язок із категорією/продавцем)",
    9: "латиниця замість кирилиці / переплутаний порядок (`Cокет`)",
}

# ---------------------------------------------------------------------------
# Value-level helpers. Pure functions: no DB, no I/O. Reused by the generator
# so that "dirty by construction" and "degraded by injection" share one
# definition of each defect.
# ---------------------------------------------------------------------------

#: Cyrillic letters that have a visually identical Latin twin. This is the real
#: failure mode of scraped catalogues: `Сокет` with U+0421 never equals `Сокет`
#: with ASCII C under any equality test, so the key silently becomes an orphan.
_LATIN_TWINS = {
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y", "І": "I", "Ѕ": "S",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s",
}

#: Tokens some sources use to mean "no value". Readers must treat them as NULL.
NULL_TOKENS = ("NULL", "null", "None", "none", "n/a", "N/A", "-", "—", "", " ")

_DECORATION_PREFIXES = ("~", "≈", "прибл. ", "~ ", "≈ ")
_NUMBER_RE = re.compile(r"^(\d+(?:[.,]\d+)?)")


def latinize_cyrillic(text: str, rng: random.Random, rate: float = 1.0) -> str:
    """Replace look-alike Cyrillic letters with Latin twins (defect class 9).

    ``rate`` is the probability of corrupting any single candidate letter, so
    the same key can appear in both spellings across one catalogue -- which is
    what makes the defect dangerous rather than merely noisy.
    """
    out = []
    for ch in text:
        twin = _LATIN_TWINS.get(ch)
        if twin is not None and rng.random() < rate:
            out.append(twin)
        else:
            out.append(ch)
    return "".join(out)


def fold_unit_into_key(key: str, value: str) -> str:
    """Fold a unit-bearing value into its key (defect class 5).

    ``("Power", "500 W")`` -> ``"power_500w"``. The magnitude is now part of
    the *schema*, so no value-level reader can recover it.

    Named ``fold_unit_into_key``, not ``unit_into_key``: the DB-level injector
    of class 5 carries that name, and a same-named helper would be shadowed by
    it -- the injector would then call *itself* with a string as ``conn``.
    """
    m = _NUMBER_RE.match(str(value).strip().lstrip("~≈ "))
    if not m:
        return key
    tail = str(value).strip()[m.end():].strip().replace(" ", "")
    if not tail:
        return key
    num = m.group(1).replace(",", ".")
    if num.endswith(".0"):
        num = num[:-2]
    return f"{key}_{num}{tail}".lower()


def decorate_number(value: str, rng: random.Random) -> str:
    """Prefix an approximate marker: ``250 г`` -> ``~250 г``."""
    return f"{rng.choice(_DECORATION_PREFIXES)}{value}".strip()


def as_text_number(value: float, rng: random.Random, locale: str = "uk") -> str:
    """Render a number the way a foreign exporter would (defect class 4).

    ``1250.0`` -> ``"1 250,00"`` (uk) or ``"1,250.00"`` (us). It gets stored in
    a column the schema declares REAL, so SQLite now holds TEXT there.
    """
    if locale == "us":
        return f"{value:,.2f}"
    whole, _, frac = f"{value:.2f}".partition(".")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    return " ".join(groups) + "," + frac


#: Conversion table for :func:`make_unit_variant`. The value stays *correct*;
#: only its spelling changes. This is deliberate heterogeneity (class 6), the
#: thing the Value Algebra is supposed to fold -- not corruption.
_UNIT_TABLE = {
    "g": [("oz", 1 / 28.3495, 1), ("kg", 1 / 1000, 3)],
    "г": [("oz", 1 / 28.3495, 1), ("кг", 1 / 1000, 3)],
    "гр": [("oz", 1 / 28.3495, 1)],
    "кг": [("g", 1000.0, 0)], "kg": [("g", 1000.0, 0), ("oz", 35.274, 1)],
    "ml": [("l", 1 / 1000, 3), ("cl", 0.1, 0)], "мл": [("l", 1 / 1000, 3)],
    "l": [("ml", 1000.0, 0)], "л": [("ml", 1000.0, 0)],
    "W": [("kW", 1 / 1000, 3)], "Вт": [("W", 1.0, 0)],
    "мм": [("cm", 0.1, 1)], "mm": [("см", 0.1, 1)],
    "oz": [("g", 28.3495, 0)], "kcal": [("kJ", 4.184, 0)], "ккал": [("кДж", 4.184, 0)],
}


def make_unit_variant(value: str, rng: random.Random) -> str:
    """Restate a measurement in another unit of the same dimension.

    ``142 g`` -> ``5 oz``; ``0.5L`` -> ``500 ml``. Returns the input unchanged
    when the unit is unknown (never invent a conversion).
    """
    m = re.match(r"^([~≈]?\s*)(\d+(?:[.,]\d+)?)\s*([a-zA-Zа-яА-Я°]+)$",
                 str(value).strip())
    if not m:
        return value
    prefix, num, unit = m.group(1), float(m.group(2).replace(",", ".")), m.group(3)
    options = _UNIT_TABLE.get(unit)
    if not options:
        return value
    new_unit, factor, digits = rng.choice(options)
    converted = round(num * factor, digits) if digits else round(num * factor)
    return f"{prefix}{converted} {new_unit}".strip()


#: The five unit spellings the polygon must contain (docs/09 §6, P1 criterion).
UNIT_FORMATS = ("5oz", "142 g", "0.5L", "500 ml", "~250гр")


# ---------------------------------------------------------------------------
# DB-level injectors. `inject()` owns the copy, then calls exactly the requested
# classes. Every injector is tolerant of a schema it does not recognise
# (returns 0) -- because "any database" includes databases that do not look
# like ours, and a crash on an unknown shape would be a bug, not a finding.
# ---------------------------------------------------------------------------

def _columns(conn: sqlite3.Connection, table: str):
    try:
        return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
    except sqlite3.DatabaseError:
        return []


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    except sqlite3.DatabaseError:
        return 0


def _distinct(conn: sqlite3.Connection, table: str, col: str) -> int:
    try:
        return conn.execute(
            f'SELECT COUNT(DISTINCT "{col}") FROM "{table}"').fetchone()[0] or 0
    except sqlite3.DatabaseError:
        return 0


def _table_names(conn: sqlite3.Connection):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'")]


def _info(conn: sqlite3.Connection, table: str):
    """``[(name, declared_type, pk_flag), ...]`` for one table."""
    return [(r[1], (r[2] or "").upper(), r[5])
            for r in conn.execute(f'PRAGMA table_info("{table}")')]


#: Preferred *table* names when structure is ambiguous. These are schema names
#: we invented for our own ingest, used only as a tie-break -- never as the
#: source of truth. Domain words ("сокет", "калорії") must never appear here.
_TABLE_HINTS = ("product_eav_meta", "product_meta", "products")


def _references(conn: sqlite3.Connection, table: str, col: str,
                entity_table: str, anchor: str):
    """``(violations, distinct)`` for a candidate *reference* column.

    The subset test runs in SQL so it stays correct on a 200k-row attribute
    table where materialising the id set in Python would blow up.
    """
    try:
        bad = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NOT NULL '
            f'AND "{col}" NOT IN (SELECT "{anchor}" FROM "{entity_table}")'
        ).fetchone()[0]
        tot = conn.execute(
            f'SELECT COUNT(DISTINCT "{col}") FROM "{table}" '
            f'WHERE "{col}" IS NOT NULL').fetchone()[0] or 0
    except sqlite3.DatabaseError:
        return -1, 0
    return bad, tot


def _entity_candidates(conn: sqlite3.Connection, tables):
    """Tables that *carry measured quantities*, largest first.

    An entity table is a table with a key column and at least one further
    REAL/INTEGER column. That single test separates `products` (price, stock,
    score) from a pure attribute store (`meta_key`/`meta_value`), from a
    two-column `ingest_meta`, and from `sqlite_sequence` -- without naming a
    single domain word. A key/value store has nothing to measure, so it can
    never win here.
    """
    out = []
    for t in tables:
        cols = _info(conn, t)
        if len(cols) < 3:
            continue
        n = _row_count(conn, t)
        if n < 2:
            continue
        pks = [c for c, _, pk in cols if pk]
        pk = pks[0] if pks else None
        measured = [c for c, ty, _ in cols
                    if ty in ("REAL", "FLOAT", "DOUBLE", "INTEGER") and c != pk]
        if not measured:
            continue
        score = n + (50 if t in _TABLE_HINTS else 0) + (10 if len(pks) == 1 else 0)
        out.append((score, t, pk))
    out.sort(key=lambda r: -r[0])
    return out


def _eav_shape(conn: sqlite3.Connection, table: str,
               entity_table: str, entity_pk: str):
    """``(ref_col, key_col, value_col)`` if ``table`` is a key/value store.

    The entity *reference* column is removed *first*: it is usually the highest
    cardinality TEXT column and would otherwise be mistaken for the value, so
    the value would silently become a second copy of the entity id.
    """
    cols = _info(conn, table)
    text_cols = [n for n, ty, _ in cols if ty in ("TEXT", "VARCHAR", "CLOB", "")]
    if len(cols) < 3 or len(text_cols) < 2:
        return None
    if _row_count(conn, table) < 10:
        return None
    ref = None
    if entity_table and entity_pk:
        by_card = sorted(text_cols, key=lambda c: -_distinct(conn, table, c))
        for c in by_card[:4]:
            bad, tot = _references(conn, table, c, entity_table, entity_pk)
            if bad == 0 and tot >= 5:
                ref = c
                break
    cand = [c for c in text_cols if c != ref]
    if len(cand) < 2:
        return None
    n = _row_count(conn, table)
    cards = {c: _distinct(conn, table, c) for c in cand}
    key_c = min(cards, key=lambda c: cards[c])
    val_c = max(cards, key=lambda c: cards[c])
    if key_c == val_c or cards[key_c] == 0:
        return None
    # A *key* is what the table repeats per value and uses *fewer* of than the
    # value. The repeat test is 1.5x -- not 2x -- on purpose: a small dirty
    # catalogue may hold only 34 attribute rows with 8 keys and 15 distinct
    # values, and a 2x rule would silently drop the whole source.
    if n < 1.5 * cards[key_c] or cards[val_c] <= cards[key_c]:
        return None                                   # not a key/value split
    return ref, key_c, val_c


def discover_schema(conn: sqlite3.Connection) -> dict:
    """Detect the roles an injector needs, from *shapes* not vocabulary.

    Returns ``entity_table``, ``entity_pk``, ``eav_table``, ``eav_key_col``,
    ``eav_value_col``, ``numeric_col``, ``swap_pair``, ``fk_column``,
    ``name_column``. Any may be ``None``; that is a valid, reportable answer --
    it means "this defect class cannot be injected into this source", which is
    itself a result for the robustness table.

    Deliberately domain-free:

    * **entity table** = a table that carries *measured quantities* (a key plus
      at least one more REAL/INTEGER column). `products` has price/stock/score;
      a key/value store has nothing to measure.
    * **EAV table** = the table whose remaining TEXT columns split into a
      *low-cardinality key* repeated per entity and a *high-cardinality value*;
      the entity reference is detected and removed first.
    * **fk column** = an entity column whose values are a subset of another
      table's key column.
    * **numeric column / swap pair** = REAL-affinity columns in schema order,
      so the swap is reproducible from the record alone.
    """
    tables = _table_names(conn)
    candidates = _entity_candidates(conn, tables)

    # --- pair every plausible entity with every plausible key/value store ---
    best = None            # (score, entity, pk, eav, ref, key, value)
    for score, ent, pk in candidates[:4]:
        for t in tables:
            if t == ent:
                continue
            shape = _eav_shape(conn, t, ent, pk)
            if not shape:
                continue
            ref, key_c, val_c = shape
            # A reference column is what makes it an attribute store *of* this
            # entity rather than a coincidence, so it outweighs raw size.
            s = score + _row_count(conn, t) + (100 if ref else 0)
            if best is None or s > best[0]:
                best = (s, ent, pk, t, ref, key_c, val_c)

    if best:
        _, entity_table, entity_pk, eav_table, _ref, eav_key, eav_value = best
    elif candidates:
        _, entity_table, entity_pk = candidates[0]
        eav_table = eav_key = eav_value = None
    else:
        entity_table = entity_pk = eav_table = eav_key = eav_value = None

    schema = {"entity_table": entity_table, "entity_pk": entity_pk,
              "eav_table": eav_table, "eav_key_col": eav_key,
              "eav_value_col": eav_value, "numeric_col": None,
              "swap_pair": None, "fk_column": None, "name_column": None}
    if not entity_table:
        return schema

    cols = _info(conn, entity_table)

    # --- measurable columns: numeric column + swap pair ---
    # REAL first, then INTEGER: a schema with one REAL and one INTEGER measure
    # -- the common case in a foreign catalogue -- must still yield a swap pair.
    # The old "only fall back when there are no REAL columns" rule returned
    # `swap_pair = None` for `amount REAL, rank INTEGER`, which silently made
    # defect class 7 impossible to inject there. Schema order within each group,
    # so the choice is reproducible from the DDL alone.
    floats = [n for n, ty, _ in cols
              if ty in ("REAL", "FLOAT", "DOUBLE", "NUMERIC")]
    ints = [n for n, ty, _ in cols
            if ty in ("INTEGER", "INT", "BIGINT", "SMALLINT") and n != entity_pk]
    measures = floats + ints
    if measures:
        schema["numeric_col"] = measures[0]
        if len(measures) >= 2:
            schema["swap_pair"] = (measures[0], measures[1])

    # --- name column ---
    texts = [n for n, ty, _ in cols if ty in ("TEXT", "VARCHAR", "")]
    non_id = [c for c in texts if c != entity_pk]
    if non_id:
        schema["name_column"] = ("name" if "name" in non_id else max(
            non_id, key=lambda c: _distinct(conn, entity_table, c)))

    # --- fk column: entity column whose values all live in another table ---
    for name, _ty, _pk in cols:
        if name == entity_pk:
            continue
        vals = {r[0] for r in conn.execute(
            f'SELECT DISTINCT "{name}" FROM "{entity_table}" '
            f'WHERE "{name}" IS NOT NULL LIMIT 2000')}
        if not vals:
            continue
        for other in tables:
            if other in (entity_table, eav_table):
                continue
            o_pks = [n for n, _, pk in _info(conn, other) if pk]
            if not o_pks:
                continue
            o_vals = {r[0] for r in conn.execute(
                f'SELECT DISTINCT "{o_pks[0]}" FROM "{other}" LIMIT 2000')}
            if o_vals and vals <= o_vals:
                schema["fk_column"] = name
                break
        if schema["fk_column"]:
            break
    return schema


def _target_count(rate: float, n: int) -> int:
    """Objects a class should damage: ``max(1, round(rate * n))``.

    Never zero while eligible objects exist. See :func:`_eligible_targets` for
    why the floor matters more than the rate itself.
    """
    return max(1, int(round(rate * n)))


def _eligible_targets(conn: sqlite3.Connection, table: str, cols, rate: float,
                      rng: random.Random, transform):
    """``[(rowid, column, new_value)]`` for the rows this defect can damage.

    ``transform(*old_values)`` returns ``(column, new_value)`` when the row is
    eligible (a unit to convert, Cyrillic to respell) and any falsy value when
    it is not. Two properties matter:

    * **The rate is a fraction of *eligible* objects, not of all rows.** A
      column that is 20% unit-bearing would otherwise be 100% rewritten at
      ``rate=0.2`` while a 100%-unit-bearing one stays at 20% -- the same class
      would cut two sources to different depths and the robustness score would
      measure value composition instead of resilience.
    * **The count is exact, not a Bernoulli draw per row.** On a small dirty
      catalogue (34 attribute rows, ``rate=0.05``) a draw can legitimately
      return zero, and the file in ``data/degraded/`` would then be
      byte-identical to its source while the report calls it broken -- the
      polygon would "test resilience" against an undamaged database.
    """
    known = _columns(conn, table)
    cols = [c for c in cols if c in known]
    if not cols:
        return []
    collist = ", ".join(f'"{c}"' for c in cols)
    cand = []
    for row in conn.execute(f'SELECT rowid, {collist} FROM "{table}"'):
        got = transform(*row[1:])
        if got:
            cand.append((row[0], got[0], got[1]))
    if not cand:
        return []
    k = min(_target_count(rate, len(cand)), len(cand))
    return rng.sample(cand, k)


def drop_primary_key(conn: sqlite3.Connection, table: str = "products",
                     rng: random.Random = None) -> int:
    """Defect class 1: rebuild ``table`` without its PRIMARY KEY.

    SQLite cannot drop a constraint, so the table is rebuilt. Rows are
    preserved; what is lost is *addressability* -- ``UPDATE``/``DELETE`` by id
    can now hit several rows, and ``INSERT OR REPLACE`` silently stops being
    an upsert.
    """
    info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    if not info:
        return 0
    if not any(row[5] for row in info):           # row[5] = pk flag
        return 0                                  # already has no PK
    cols = [row[1] for row in info]
    defs = ", ".join(f'"{row[1]}" {row[2]}' for row in info)
    n = _row_count(conn, table)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(f'ALTER TABLE "{table}" RENAME TO "{table}__old"')
    conn.execute(f'CREATE TABLE "{table}" ({defs})')
    collist = ", ".join(f'"{c}"' for c in cols)
    conn.execute(f'INSERT INTO "{table}" ({collist}) '
                 f'SELECT {collist} FROM "{table}__old"')
    conn.execute(f'DROP TABLE "{table}__old"')
    conn.execute("PRAGMA foreign_keys = ON")
    return n


def duplicate_entities(conn: sqlite3.Connection, table: str = "products",
                       n: int = 0, rng: random.Random = None) -> int:
    """Defect class 2: copy rows and give the copies a *conflicting* value.

    The copy keeps the same name, so entity resolution pairs the two, but its
    measured quantity is shifted -- a naive dedup can therefore keep the wrong
    number. The new id is chosen by *declared type*: an INTEGER PRIMARY KEY is a
    rowid alias and rejects a `"7_dup0"` string, so it gets a fresh integer
    beyond the current maximum instead. Nothing here raises on an unfamiliar
    schema -- a row that still cannot be inserted is skipped and not counted.
    """
    cols = _columns(conn, table)
    if not cols or n <= 0:
        return 0
    rows = conn.execute(f'SELECT * FROM "{table}" LIMIT 100000').fetchall()
    if not rows:
        return 0
    id_col = cols[0]
    id_idx = cols.index(id_col)
    types = {name: ty for name, ty, _ in _info(conn, table)}
    int_id = types.get(id_col, "") in ("INTEGER", "INT", "BIGINT", "SMALLINT")
    next_id = 1
    if int_id:
        latest = conn.execute(
            f'SELECT MAX("{id_col}") FROM "{table}"').fetchone()[0]
        next_id = int(latest) + 1 if isinstance(latest, (int, float)) else 1
    reals = [c for c in cols
             if types.get(c) in ("REAL", "FLOAT", "DOUBLE", "NUMERIC")]
    # `price` is only a tie-break among measured columns, never a requirement:
    # the defect is "the copy disagrees about a quantity", not "about price".
    measured = "price" if "price" in reals else (reals[0] if reals else None)
    made = 0
    for row in rng.sample(rows, min(n, len(rows))):
        new = list(row)
        if int_id:
            new[id_idx] = next_id
            next_id += 1
        else:
            new[id_idx] = f"{row[id_idx]}_dup{made}"
        if measured is not None:
            try:
                new[cols.index(measured)] = round(
                    float(row[cols.index(measured)]) * rng.uniform(1.05, 1.35), 2)
            except (TypeError, ValueError):
                pass
        marks = ", ".join("?" for _ in cols)
        try:
            conn.execute(f'INSERT INTO "{table}" VALUES ({marks})', new)
        except sqlite3.IntegrityError:
            # A constraint we did not anticipate (an id handed out by hand, a
            # NOT NULL we cannot satisfy): skip the row rather than abort the
            # polygon. The reported count then honestly says how many got in.
            continue
        made += 1
    return made


def null_key_attributes(conn: sqlite3.Connection, table: str = "product_eav_meta",
                        column: str = "meta_value", rate: float = 0.05,
                        rng: random.Random = None, token: str = "NULL") -> int:
    """Defect class 3: put a NULL token where a key attribute was expected."""
    if column not in _columns(conn, table):
        return 0
    hit = 0

    def null_at(v):
        # A cell that is already the token is not eligible: counting it as
        # damaged would inflate the report without changing the data.
        return (column, token) if v is not None and v != token else None

    for pk, col, new in _eligible_targets(conn, table, [column], rate, rng, null_at):
        conn.execute(f'UPDATE "{table}" SET "{col}" = ? WHERE rowid = ?',
                     (new, pk))
        hit += 1
    return hit


def stringify_numbers(conn: sqlite3.Connection, table: str = "products",
                      column: str = "price", rate: float = 0.1,
                      rng: random.Random = None) -> int:
    """Defect class 4: store a numeric column as locale-formatted TEXT.

    ``price = 1250.0`` becomes the string ``"1 250,00"``. ``SUM(price)`` still
    works by accident (SQLite coerces), but ``price > 1000`` and ``ORDER BY``
    start lying -- silently.
    """
    if column not in _columns(conn, table):
        return 0

    def as_text(val):
        # Only a genuinely numeric cell is eligible: a column already holding
        # TEXT cannot be "stored as text by mistake".
        if not isinstance(val, (int, float)):
            return None
        return (column, as_text_number(float(val), rng))

    hit = 0
    for pk, col, new in _eligible_targets(conn, table, [column], rate, rng, as_text):
        conn.execute(f'UPDATE "{table}" SET "{col}" = ? WHERE rowid = ?',
                     (new, pk))
        hit += 1
    return hit


def unit_into_key(conn: sqlite3.Connection, table: str = "product_eav_meta",
                  key_col: str = "meta_key", value_col: str = "meta_value",
                  rate: float = 0.15, rng: random.Random = None) -> int:
    """Defect class 5: move the unit from the value into the key name."""
    if key_col not in _columns(conn, table) or value_col not in _columns(conn, table):
        return 0

    def fold(k, v):
        if v is None:
            return None
        new_key = fold_unit_into_key(k, v)
        return (key_col, new_key) if new_key != k else None

    hit = 0
    for pk, col, new in _eligible_targets(
            conn, table, [key_col, value_col], rate, rng, fold):
        conn.execute(f'UPDATE "{table}" SET "{col}" = ? WHERE rowid = ?',
                     (new, pk))
        hit += 1
    return hit


def mix_units(conn: sqlite3.Connection, table: str = "product_eav_meta",
              value_col: str = "meta_value", rate: float = 0.2,
              rng: random.Random = None) -> int:
    """Defect class 6: same key, values restated in different units."""
    if value_col not in _columns(conn, table):
        return 0

    def restate(v):
        if v is None:
            return None
        variant = make_unit_variant(v, rng)
        return (value_col, variant) if variant != v else None

    hit = 0
    for pk, col, new in _eligible_targets(conn, table, [value_col], rate, rng, restate):
        conn.execute(f'UPDATE "{table}" SET "{col}" = ? WHERE rowid = ?',
                     (new, pk))
        hit += 1
    return hit


def swap_columns(conn: sqlite3.Connection, table: str = "products",
                 col_a: str = "price", col_b: str = "rating",
                 rng: random.Random = None) -> int:
    """Defect class 7: exchange two columns wholesale.

    Both columns keep a plausible *type*, so nothing raises: the schema still
    validates, the numbers still look like numbers, and every ranking is now
    computed on the wrong quantity. SQLite evaluates all right-hand sides
    against the pre-update row, so one UPDATE performs a true swap.
    """
    cols = _columns(conn, table)
    if col_a not in cols or col_b not in cols:
        return 0
    conn.execute(f'UPDATE "{table}" SET "{col_a}" = "{col_b}", '
                 f'"{col_b}" = "{col_a}"')
    return _row_count(conn, table)


def orphan_rows(conn: sqlite3.Connection, table: str = "products",
                fk_col: str = "merchant_id", n: int = 0,
                rng: random.Random = None) -> int:
    """Defect class 8: rows whose foreign key points nowhere.

    The new row is filled by *declared type*, not by column name: a REAL column
    gets a plausible decimal, an INTEGER a small count, a TEXT a placeholder.
    Hard-coding `price`/`delivery_days`/`category` here would make this the one
    injector that only works on our own schema -- exactly the coupling the
    Adaptive Data Fabric is meant to remove.
    """
    cols = _columns(conn, table)
    if fk_col not in cols or n <= 0:
        return 0
    types = {name: ty for name, ty, _ in _info(conn, table)}
    pks = [n for n, _, pk in _info(conn, table) if pk]
    id_col = pks[0] if pks else cols[0]
    int_id = types.get(id_col, "") in ("INTEGER", "INT", "BIGINT", "SMALLINT")
    next_id = 1
    if int_id:
        latest = conn.execute(
            f'SELECT MAX("{id_col}") FROM "{table}"').fetchone()[0]
        next_id = int(latest) + 1 if isinstance(latest, (int, float)) else 1
    inserted = 0
    for i in range(n):
        values = []
        for c in cols:
            if c == id_col:
                # An INTEGER PRIMARY KEY is a rowid alias and rejects any
                # string, so the orphan gets a fresh integer instead.
                values.append(next_id if int_id else f"orphan_{i + 1}")
            elif c == fk_col:
                values.append(f"ref_missing_{i + 1}")
            else:
                ty = types.get(c, "")
                if ty in ("REAL", "FLOAT", "DOUBLE", "NUMERIC"):
                    values.append(round(rng.uniform(10.0, 999.0), 2))
                elif ty in ("INTEGER", "INT", "BIGINT", "SMALLINT"):
                    values.append(rng.randint(0, 20))
                elif ty in ("TEXT", "VARCHAR", "CHAR", "CLOB", ""):
                    values.append(f"orphan value {i + 1}")
                else:
                    values.append(None)
        marks = ", ".join("?" for _ in cols)
        try:
            conn.execute(f'INSERT INTO "{table}" VALUES ({marks})', values)
        except sqlite3.IntegrityError:
            next_id += 1
            continue            # skip rather than abort on an unknown constraint
        next_id += 1
        inserted += 1
    return inserted


def latinize_column(conn: sqlite3.Connection, table: str = "product_eav_meta",
                    column: str = "meta_key", rate: float = 0.5,
                    rng: random.Random = None) -> int:
    """Defect class 9: mixed Latin/Cyrillic spellings of the same key."""
    if column not in _columns(conn, table):
        return 0

    def respell(v):
        if not isinstance(v, str):
            return None
        broken = latinize_cyrillic(v, rng)
        return (column, broken) if broken != v else None

    hit = 0
    for pk, col, new in _eligible_targets(conn, table, [column], rate, rng, respell):
        conn.execute(f'UPDATE "{table}" SET "{col}" = ? WHERE rowid = ?',
                     (new, pk))
        hit += 1
    return hit


#: class id -> (callable, default kwargs). Order = row order in the paper table.
INJECTORS = {
    1: (drop_primary_key, {}),
    2: (duplicate_entities, {"n": 25}),
    3: (null_key_attributes, {}),
    4: (stringify_numbers, {}),
    5: (unit_into_key, {}),
    6: (mix_units, {}),
    7: (swap_columns, {}),
    8: (orphan_rows, {"n": 20}),
    9: (latinize_column, {}),
}


def inject(db_path: str, classes, out_path: str = None, seed: int = 0,
           overrides: dict = None) -> dict:
    """Copy ``db_path``, apply ``classes``, return a machine-readable report.

    Returns a JSON-serialisable dict; the ``classes`` section is what ends up
    as a row in the paper's robustness table, and ``total_affected`` is the
    evidence that a defect was really injected rather than merely named.
    """
    classes = sorted(int(c) for c in classes)
    unknown = [c for c in classes if c not in INJECTORS]
    if unknown:
        raise ValueError(f"unknown defect classes: {unknown}; "
                         f"known: {sorted(INJECTORS)}")
    if out_path is None:
        base, ext = os.path.splitext(db_path)
        out_path = f"{base}__degraded{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shutil.copy(db_path, out_path)

    conn = sqlite3.connect(out_path)
    rng = random.Random(seed)
    schema = discover_schema(conn)
    report = {}
    try:
        for cid in classes:
            fn, kwargs = INJECTORS[cid]
            kw = _role_kwargs(cid, schema)
            kw.update(kwargs)
            kw.update((overrides or {}).get(cid, {}))
            kw.setdefault("rng", rng)
            missing = [k for k, v in kw.items() if v is None]
            if missing:
                report[str(cid)] = {
                    "name": DEFECT_CLASSES[cid], "affected": 0,
                    "note": f"not applicable: source lacks {', '.join(missing)}",
                    "detected": {k: v for k, v in schema.items() if v is not None},
                }
                continue
            affected = fn(conn, **kw)
            conn.commit()
            entry = {"name": DEFECT_CLASSES[cid], "affected": affected}
            if affected == 0:
                # A bare 0 is indistinguishable from a broken injector. Say
                # which role was resolved and that nothing matched it, so the
                # robustness table reads "source had no such values", not
                # "the tool failed".
                target = ", ".join(f"{k}={v}" for k, v in kw.items()
                                   if k != "rng" and v is not None)
                entry["note"] = "role resolved, no object matched: " + target
            report[str(cid)] = entry
    finally:
        conn.close()
    return {
        "tool": "data/degrade.py",
        "source": os.path.abspath(db_path),
        "output": os.path.abspath(out_path),
        "seed": seed,
        "schema": schema,
        "classes": report,
        "total_affected": sum(v["affected"] for v in report.values()),
    }


def _role_kwargs(cid: int, schema: dict) -> dict:
    """Map discovered schema roles onto one injector's keyword arguments."""
    entity, eav = schema["entity_table"], schema["eav_table"]
    swap = schema["swap_pair"]
    mapping = {
        1: {"table": entity},
        2: {"table": entity},
        3: {"table": eav, "column": schema["eav_value_col"]},
        4: {"table": entity, "column": schema["numeric_col"]},
        5: {"table": eav, "key_col": schema["eav_key_col"],
            "value_col": schema["eav_value_col"]},
        6: {"table": eav, "value_col": schema["eav_value_col"]},
        7: {"table": entity,
            "col_a": swap[0] if swap else None,
            "col_b": swap[1] if swap else None},
        8: {"table": entity, "fk_col": schema["fk_column"]},
        9: {"table": eav, "column": schema["eav_key_col"]},
    }
    return mapping.get(cid, {})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Inject the nine named defect classes into a copy of a DB.")
    parser.add_argument("--db", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "marketplace_rozetka.db"))
    parser.add_argument("--classes", default="all",
                        help="'all' or comma-separated ids, e.g. 3,4,9")
    parser.add_argument("--all", action="store_true",
                        help="write one degraded copy per class into --outdir "
                             "(the robustness polygon's inputs)")
    parser.add_argument("--out", default=None,
                        help="single output path (ignored with --all)")
    parser.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "degraded"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=None,
                        help="write the JSON report to this path")
    parser.add_argument("--list", action="store_true",
                        help="print the defect taxonomy and exit")
    args = parser.parse_args(argv)

    if args.list:
        for cid in sorted(DEFECT_CLASSES):
            print(f"{cid}. {DEFECT_CLASSES[cid]:32s} {DEFECT_DESCRIPTIONS[cid]}")
        return 0

    if not os.path.exists(args.db):
        parser.error(f"source database not found: {args.db}")

    if args.all:
        wanted, all_mode = sorted(INJECTORS), True
    elif args.classes.strip().lower() == "all":
        wanted, all_mode = sorted(INJECTORS), False
    else:
        wanted, all_mode = [int(x) for x in args.classes.split(",") if x.strip()], False

    base = os.path.splitext(os.path.basename(args.db))[0]
    t0 = time.time()
    reports = []
    if all_mode:
        for cid in wanted:
            out = os.path.join(args.outdir, f"{base}__{cid:02d}_{DEFECT_CLASSES[cid]}.db")
            reports.append(inject(args.db, [cid], out, args.seed))
            r = reports[-1]["classes"][str(cid)]
            print(f"[degrade] {cid:>2}. {DEFECT_CLASSES[cid]:32s} "
                  f"affected={r['affected']:<6} -> {os.path.basename(out)}"
                  + (f"  [{r['note']}]" if "note" in r else ""))
    else:
        rep = inject(args.db, wanted, args.out, args.seed)
        reports.append(rep)
        print(f"[degrade] classes {wanted} -> {rep['output']}")
        for cid in wanted:
            r = rep["classes"][str(cid)]
            print(f"[degrade] {cid:>2}. {DEFECT_CLASSES[cid]:32s} "
                  f"affected={r['affected']}"
                  + (f"  [{r['note']}]" if "note" in r else ""))

    payload = {"source": os.path.abspath(args.db), "seed": args.seed,
               "mode": "all" if all_mode else "selected", "reports": reports}
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[degrade] report written to {args.report}")
    print(f"[degrade] done in {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())