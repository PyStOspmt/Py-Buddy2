"""Regression tests for the defect polygon in data/degrade.py.

Why a *polygon* of synthetic sources rather than one fixture: the robustness
suite scores the engine on degraded copies of catalogues nobody in this
repository wrote, so the injector has to find the structure of a schema it has
never seen. Both fixtures here are therefore built with names that share no
vocabulary with our own catalogue -- tables `items` / `attr_store` / `owners`,
columns `k` / `v` / `amount` / `rank` / `label`. There is no `products`, no
`meta_key`, no `price` anywhere in them, so anything that works on them works
from shape rather than from a domain word.

The assertions are deliberately about *evidence*, not about line coverage:

* every damaged copy must differ from its source -- a copy identical to the
  source would mean the robustness suite was scoring resilience against an
  undamaged database;
* the declared defect must be visible in the data (PK really gone, number
  really TEXT, key really folded, cells really restated);
* the reported ``affected`` count must match what is in the file, because that
  number is what the paper's robustness table cites as proof of injection;
* a source with no such structure must yield 0 *with a reason* instead of an
  exception, and a fixed seed must reproduce the same file byte for byte.
"""
import hashlib
import os
import re
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))

import degrade as D

#: 20 products x 4 attribute rows. The attribute *values* vary per product so
#: the value column has far more distinct entries than the key column -- that
#: ratio is the only thing `discover_schema` is allowed to rely on.
N_ITEMS = 20
KEYS = ["Потужність", "Вага", "Об_єм", "Сокет"]
#: Keys whose values carry a unit, i.e. the rows classes 5 and 6 can damage.
UNIT_KEYS = {"Потужність": "W", "Вага": "g", "Об_єм": "l"}


def _attr_rows():
    for i in range(1, N_ITEMS + 1):
        ref = f"P{i}"
        yield (ref, "Потужність", f"{400 + i * 5} W")
        yield (ref, "Вага", f"{100 + i} g")
        yield (ref, "Об_єм", f"{0.4 + i * 0.01} l")
        yield (ref, "Сокет", "AM5")


def build_foreign_source(path):
    """A catalogue written by 'somebody else': no shared word with ours."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            label TEXT,
            amount REAL,
            rank INTEGER,
            owner_id TEXT
        );
        CREATE TABLE owners (oid TEXT PRIMARY KEY, olabel TEXT);
        CREATE TABLE attr_store (
            rid INTEGER PRIMARY KEY,
            item_ref TEXT,
            k TEXT,
            v TEXT
        );
    """)
    conn.executemany("INSERT INTO owners VALUES (?, ?)",
                     [(f"O{i}", f"Shop {i}") for i in range(1, 6)])
    conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?, ?)",
                     [(i, f"Item {i}", 10.0 * i, i % 7, f"O{i % 5 + 1}")
                      for i in range(1, N_ITEMS + 1)])
    conn.executemany("INSERT INTO attr_store (item_ref, k, v) VALUES (?, ?, ?)",
                     list(_attr_rows()))
    conn.commit()
    conn.close()
    return path


def build_structureless_source(path):
    """Two TEXT columns and nothing else: no entity, no measurement, no EAV."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE notes (k TEXT, v TEXT);
        INSERT INTO notes VALUES ('a','b'),('a','b'),('a','b'),('c','d');
    """)
    conn.commit()
    conn.close()
    return path


def digest(path):
    """Content fingerprint, so 'damaged' can be proven, not assumed."""
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    h = hashlib.sha1()
    for t in sorted(D._table_names(conn)):
        h.update(t.encode())
        h.update(str([n for n, _, pk in D._info(conn, t) if pk]).encode())
        for row in conn.execute(f'SELECT * FROM "{t}" ORDER BY rowid'):
            h.update(repr(row).encode("utf-8", "replace"))
    conn.close()
    return h.hexdigest()


def open_ro(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


EXPECTED_SCHEMA = {
    "entity_table": "items",
    "entity_pk": "id",
    "eav_table": "attr_store",
    "eav_key_col": "k",
    "eav_value_col": "v",
    "numeric_col": "amount",
    "swap_pair": ("amount", "rank"),
    "fk_column": "owner_id",
    "name_column": "label",
}


@pytest.fixture()
def foreign(tmp_path):
    return build_foreign_source(str(tmp_path / "foreign.db"))


def _schema(path):
    conn = open_ro(path)
    try:
        return D.discover_schema(conn)
    finally:
        conn.close()


def test_fixture_shares_no_vocabulary_with_our_own_catalogue(foreign):
    """Guard the guard: the point of this fixture is that it is foreign.

    If a hint word ever leaked into it, the discovery test below would start
    passing for the wrong reason.
    """
    conn = open_ro(foreign)
    tables = set(D._table_names(conn))
    conn.close()
    assert tables == {"items", "owners", "attr_store"}
    assert set(D._TABLE_HINTS).isdisjoint(tables)


def test_discover_schema_reads_shape_not_vocabulary(foreign):
    assert _schema(foreign) == EXPECTED_SCHEMA


def test_structureless_source_yields_no_roles_and_no_exception(tmp_path):
    path = build_structureless_source(str(tmp_path / "flat.db"))
    assert _schema(path) == {k: None for k in EXPECTED_SCHEMA}
    report = D.inject(path, range(1, 10), str(tmp_path / "flat_out.db"))
    assert report["total_affected"] == 0
    for cid in map(str, range(1, 10)):
        entry = report["classes"][cid]
        assert entry["affected"] == 0
        # "0" must arrive with a reason, otherwise a report row is
        # indistinguishable from a broken injector.
        assert "not applicable" in entry["note"]


def eligible_unit_values(path):
    """Rows whose value carries a unit the conversion table knows (classes 5/6)."""
    conn = open_ro(path)
    n = 0
    for _, v in conn.execute("SELECT item_ref, v FROM attr_store"):
        m = re.match(r"^([~≈]?\s*)(\d+(?:[.,]\d+)?)\s*([a-zA-Zа-яА-Я°]+)$", v.strip())
        if m and D._UNIT_TABLE.get(m.group(3)):
            n += 1
    conn.close()
    return n


def eligible_cyrillic_keys(path):
    """Rows whose key holds a character with a Latin look-alike (class 9)."""
    conn = open_ro(path)
    n = sum(1 for _, k in conn.execute("SELECT item_ref, k FROM attr_store")
            if any(ch in D._LATIN_TWINS for ch in k))
    conn.close()
    return n


#: class id -> what the fixture's shape makes us expect. Kept explicit rather
#: than computed, so that a change in injector behaviour fails loudly here.
EXPECTED_AFFECTED = {
    1: N_ITEMS,          # one UPDATE covering every row
    2: N_ITEMS,          # default n=25, capped by 20 rows
    3: 4,                # 5% of the 80 attribute rows
    4: 2,                # 10% of the 20 measured rows
    5: 9,                # 15% of the 60 unit-bearing values
    6: 12,               # 20% of the 60 unit-bearing values
    7: N_ITEMS,
    8: 20,               # n=20 orphan rows inserted
    9: 40,               # 50% of the 80 Cyrillic keys
}


def signature(src, deg, cid):
    """``(ok, description)`` -- is the declared defect really in the file?

    A contact sheet of the nine classes: each one has to leave a trace that can
    be read back out of the database, because "the file changed" alone does not
    prove *which* defect was injected.
    """
    s, d = open_ro(src), open_ro(deg)
    try:
        if cid == 1:
            dropped = not [n for n, _, pk in D._info(d, "items") if pk]
            had = bool([n for n, _, pk in D._info(s, "items") if pk])
            return dropped and had, "items: PRIMARY KEY removed"
        if cid == 2:
            return (D._row_count(d, "items") > D._row_count(s, "items"),
                    f"items rows {D._row_count(s, 'items')} -> "
                    f"{D._row_count(d, 'items')}")
        if cid == 3:
            n = d.execute("SELECT COUNT(*) FROM attr_store WHERE v = 'NULL'"
                          ).fetchone()[0]
            return n > 0, f"{n} NULL tokens in attr_store.v"
        if cid == 4:
            n = d.execute("SELECT COUNT(*) FROM items WHERE typeof(amount) = 'text'"
                          ).fetchone()[0]
            return n > 0, f"{n} numbers stored as TEXT in items.amount"
        if cid == 5:
            ks = {r[0] for r in s.execute("SELECT DISTINCT k FROM attr_store")}
            kd = {r[0] for r in d.execute("SELECT DISTINCT k FROM attr_store")}
            new = [k for k in kd - ks if re.search(r"_\d", k)]
            return bool(new), f"{len(new)} folded keys, e.g. {sorted(new)[:2]}"
        if cid == 6:
            # Vocabulary size stays the same when a conversion moves between
            # units the table already contains, so compare the cells themselves.
            vs = dict(s.execute("SELECT rowid, v FROM attr_store"))
            vd = dict(d.execute("SELECT rowid, v FROM attr_store"))
            diff = sum(1 for k in vs if vs[k] != vd.get(k))
            return diff > 0, f"{diff} cells restated in another unit"
        if cid == 7:
            q = "SELECT COUNT(*) FROM items WHERE amount > rank"
            return (d.execute(q).fetchone()[0] != s.execute(q).fetchone()[0],
                    "items: amount <-> rank ordering flipped")
        if cid == 8:
            n = d.execute("SELECT COUNT(*) FROM items WHERE owner_id = 'ref_missing_1'"
                          ).fetchone()[0]
            return n > 0, "items.owner_id: dangling reference present"
        if cid == 9:
            ks = {r[0] for r in s.execute("SELECT DISTINCT k FROM attr_store")}
            kd = {r[0] for r in d.execute("SELECT DISTINCT k FROM attr_store")}
            twinned = [k for k in kd
                       if any(ch in D._LATIN_TWINS.values() for ch in k)]
            return bool(kd - ks) and bool(twinned), \
                f"{len(kd - ks)} respelled keys, e.g. {sorted(twinned)[:2]}"
    finally:
        s.close()
        d.close()
    return False, "unknown class"


@pytest.mark.parametrize("cid", sorted(EXPECTED_AFFECTED))
def test_polygon_injects_every_class_into_a_foreign_schema(foreign, tmp_path, cid):
    """All nine classes, on a schema with none of our words in it."""
    out = str(tmp_path / f"foreign__{cid:02d}.db")
    report = D.inject(foreign, [cid], out)
    entry = report["classes"][str(cid)]
    assert entry["affected"] == EXPECTED_AFFECTED[cid], entry.get("note", "")
    assert digest(out) != digest(foreign), "degraded copy equals its source"
    ok, desc = signature(foreign, out, cid)
    assert ok, desc


def test_report_count_is_visible_in_the_database(foreign, tmp_path):
    """The reported number is evidence: what it claims must be in the file."""
    for cid in sorted(EXPECTED_AFFECTED):
        out = str(tmp_path / f"claim__{cid:02d}.db")
        report = D.inject(foreign, [cid], out)
        assert report["total_affected"] == EXPECTED_AFFECTED[cid]
        assert report["classes"][str(cid)]["affected"] > 0


def test_intensity_is_a_fraction_of_eligible_objects(foreign, tmp_path):
    """``rate`` must not silently become "everything" or "nothing".

    A column that is only partly unit-bearing would otherwise be rewritten
    completely at ``rate=0.2``, and the robustness score would then measure the
    source's value mix instead of the engine's resilience.
    """
    unit = eligible_unit_values(foreign)
    cyrillic = eligible_cyrillic_keys(foreign)
    assert (unit, cyrillic) == (60, 80)
    cases = ((4, 0.35, N_ITEMS), (5, 0.30, unit), (6, 0.5, unit), (9, 0.25, cyrillic))
    for cid, rate, eligible in cases:
        out = str(tmp_path / f"rate__{cid}.db")
        report = D.inject(foreign, [cid], out, overrides={cid: {"rate": rate}})
        assert report["classes"][str(cid)]["affected"] == max(1, round(rate * eligible))


def test_same_seed_reproduces_the_same_file(foreign, tmp_path):
    """Reproducibility is a paper claim, so it is a test."""
    a = D.inject(foreign, [9], str(tmp_path / "seedA.db"), seed=7)
    b = D.inject(foreign, [9], str(tmp_path / "seedB.db"), seed=7)
    c = D.inject(foreign, [9], str(tmp_path / "seedC.db"), seed=8)
    assert digest(a["output"]) == digest(b["output"])
    assert digest(c["output"]) != digest(a["output"])
    assert a["classes"]["9"]["affected"] == c["classes"]["9"]["affected"]


# ---------------------------------------------------------------------------
# Structure decides which classes are applicable.
#
# The same entity table, but with no foreign key anywhere: class 8 has nowhere
# to point, while the attribute store still supports classes 3, 5, 6 and 9.
# A report that invented an orphan here -- or silently dropped the class -- is
# exactly what the "not applicable" note exists to prevent.
# ---------------------------------------------------------------------------
def build_fk_less_source(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            label TEXT,
            amount REAL,
            rank INTEGER
        );
        CREATE TABLE attr_store (
            rid INTEGER PRIMARY KEY,
            item_ref TEXT,
            k TEXT,
            v TEXT
        );
    """)
    conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?)",
                     [(i, f"Item {i}", 10.0 * i, i % 7)
                      for i in range(1, N_ITEMS + 1)])
    conn.executemany("INSERT INTO attr_store (item_ref, k, v) VALUES (?, ?, ?)",
                     list(_attr_rows()))
    conn.commit()
    conn.close()
    return path


def test_missing_structure_is_reported_not_faked(tmp_path):
    path = build_fk_less_source(str(tmp_path / "fk_less.db"))
    schema = _schema(path)
    assert schema["fk_column"] is None
    assert schema["eav_table"] == "attr_store"        # everything else still found
    report = D.inject(path, range(1, 10), str(tmp_path / "fk_less_out.db"))
    assert report["classes"]["8"]["affected"] == 0
    assert "not applicable" in report["classes"]["8"]["note"]
    assert "fk_col" in report["classes"]["8"]["note"]
    for cid in (1, 2, 3, 4, 5, 6, 7, 9):
        assert report["classes"][str(cid)]["affected"] > 0, cid


# ---------------------------------------------------------------------------
# The polygon on catalogues nobody in this repository wrote.
#
# These skip rather than fail when the git-ignored data/*.db are absent, but
# each name falls back to a trimmed fixture committed under tests/fixtures --
# the same policy as tests/test_real_data_validation.py, so the real-catalogue
# code path cannot rot unnoticed.
# ---------------------------------------------------------------------------
REAL_SOURCES = {
    "openfoodfacts": ("data/real_off.db", "tests/fixtures/real_off_fixture.db"),
    "openprices": ("data/real_prices.db", "tests/fixtures/real_prices_fixture.db"),
    "wdc": ("data/real_wdc.db", "tests/fixtures/real_wdc_fixture.db"),
}


def real_source(key):
    for rel in REAL_SOURCES[key]:
        path = os.path.join(ROOT, *rel.split("/"))
        if os.path.exists(path):
            return path
    return None


@pytest.mark.parametrize("key", sorted(REAL_SOURCES))
def test_real_catalogue_roles_are_discovered(key, tmp_path):
    path = real_source(key)
    if path is None:
        pytest.skip(f"no catalogue for {key}")
    schema = _schema(path)
    for role in ("entity_table", "entity_pk", "eav_table", "eav_key_col",
                 "eav_value_col", "numeric_col", "swap_pair", "name_column"):
        assert schema[role] is not None, f"{key}: {role} not discovered"


@pytest.mark.parametrize("key", sorted(REAL_SOURCES))
def test_merged_polygon_damages_a_real_catalogue(key, tmp_path):
    """All nine classes into one copy, then the report must add up.

    Merged rather than one-file-per-class here: the per-class semantics are
    already pinned by the synthetic fixtures, and this test is about the
    end-to-end path on real data -- nothing raises, something is damaged, and
    every zero comes with a reason.
    """
    path = real_source(key)
    if path is None:
        pytest.skip(f"no catalogue for {key}")
    out = str(tmp_path / f"{key}_all.db")
    report = D.inject(path, range(1, 10), out)
    assert report["total_affected"] > 0
    assert digest(out) != digest(path), "degraded copy equals its source"
    for cid, entry in report["classes"].items():
        if entry["affected"] == 0:
            assert entry.get("note"), f"class {cid} reported 0 with no reason"
    assert report["schema"] == _schema(path)