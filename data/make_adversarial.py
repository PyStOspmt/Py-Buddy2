"""Adversarial source fixtures for stress-testing the adaptive core.

Each database targets one documented failure hypothesis. These are
labelled evaluation fixtures, not demo sources -- nothing here is
registered in sources.demo.json.

Run:  python data/make_adversarial.py
"""
import os
import random
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "adversarial")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(20261005)


def fresh(name: str) -> sqlite3.Connection:
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        os.remove(path)
    return sqlite3.connect(path)


# ------------------------------------------------ A1: sparse EAV swamp
def a1_sparse_eav():
    """500 items, ~70% attribute sparsity, 3 of 8 categories carry no
    useful keys at all -> induction starvation and coverage gaps."""
    con = fresh("a1_sparse_eav.db")
    con.executescript("""
    CREATE TABLE items (id TEXT PRIMARY KEY, title TEXT, cat TEXT,
                        price REAL, qty INTEGER);
    CREATE TABLE item_attrs (item_id TEXT, attr TEXT, val TEXT);
    """)
    cats = ["gear", "wearable", "consumable", "module",
            "hollow_a", "hollow_b", "hollow_c", "frame"]
    attr_pool = {
        "gear": ["torque_nm", "weight_g", "ip_rating", "rpm", "depth_mm"],
        "wearable": ["size_mm", "battery_h", "strap_mm", "water_m"],
        "consumable": ["volume_ml", "expires_days", "mass_g", "kcal_100"],
        "module": ["bus_mhz", "voltage_v", "pins", "current_ma", "temp_c"],
        "frame": ["span_mm", "thickness_mm", "coating", "mass_g"],
        "hollow_a": [], "hollow_b": [], "hollow_c": [],
    }
    for i in range(500):
        cat = cats[i % len(cats)]
        con.execute("INSERT INTO items VALUES (?,?,?,?,?)",
                    (f"a1-{i:04d}", f"Об'єкт {i} {cat}", cat,
                     round(rng.uniform(10, 5000), 2), rng.randint(0, 50)))
        # 30% per-attribute sparsity: a dense-enough multi-attr EAV that
        # still leaves real gaps (a recognisable EAV needs ~2+ attrs/entity).
        for a in attr_pool[cat]:
            if rng.random() < 0.70:
                v = round(rng.uniform(1, 900), 1)
                con.execute("INSERT INTO item_attrs VALUES (?,?,?)",
                            (f"a1-{i:04d}", a, str(v)))
    con.commit()
    con.close()


# ---------------------------------- A2: obfuscated/foreign column names
def a2_obfuscated_schema():
    """No English column names; entity table not called 'products'."""
    con = fresh("a2_obfuscated.db")
    con.executescript("""
    CREATE TABLE t_dane (rek_id TEXT PRIMARY KEY,
                         nazwa TEXT, grupa TEXT,
                         cena REAL, liczba INTEGER, opis TEXT);
    CREATE TABLE t_cechy (rek TEXT, cecha TEXT, wartosc TEXT);
    """)
    for i in range(300):
        con.execute("INSERT INTO t_dane VALUES (?,?,?,?,?,?)",
                    (f"r{i}", f"Pozycja {i}", f"gr{i % 9}",
                     round(rng.uniform(5, 2000), 2), rng.randint(0, 99),
                     f"Opis {i}"))
        con.execute("INSERT INTO t_cechy VALUES (?,?,?)",
                    (f"r{i}", "waga_g", str(rng.randint(10, 5000))))
    con.commit()
    con.close()


# ------------------------------------------- A3: no price axis at all
def a3_no_price():
    """A library catalogue: nothing monetary. price must stay unresolved;
    budget questions must degrade, not invent a column."""
    con = fresh("a3_no_price.db")
    con.executescript("""
    CREATE TABLE books (isbn TEXT PRIMARY KEY, title TEXT, genre TEXT,
                        pages INTEGER, year INTEGER, shelf INTEGER);
    """)
    genres = ["fiction", "science", "poetry", "history", "tech"]
    for i in range(400):
        con.execute("INSERT INTO books VALUES (?,?,?,?,?,?)",
                    (f"978-0-{i:05d}-{i % 10}", f"Книга {i}",
                     genres[i % len(genres)], rng.randint(80, 900),
                     rng.randint(1950, 2025), rng.randint(1, 40)))
    con.commit()
    con.close()


# ------------------------------- A4: many ambiguous numeric columns
def a4_numeric_swamp():
    """Six numeric columns, none named price/stock. The projection must
    report ambiguity rather than guessing a money axis."""
    con = fresh("a4_numeric_swamp.db")
    con.executescript("""
    CREATE TABLE sensors (uid TEXT PRIMARY KEY, label TEXT, kind TEXT,
                          v_min REAL, v_max REAL, v_nom REAL,
                          err_pct REAL, batch INTEGER, seq INTEGER);
    """)
    for i in range(600):
        lo = rng.uniform(0, 100)
        con.execute("INSERT INTO sensors VALUES (?,?,?,?,?,?,?,?,?)",
                    (f"s{i:04d}", f"Датчик {i}", f"k{i % 5}", lo,
                     lo + rng.uniform(10, 500), lo + rng.uniform(0, 250),
                     rng.uniform(0.1, 5), rng.randint(1, 30), i))
    con.commit()
    con.close()


# --------------------------- A5: hostile strings in the data itself
def a5_hostile_strings():
    """Product names carrying SQL fragments, LIKE wildcards, RTL marks,
    emoji and NUL-ish edge cases. Nothing may escape parameterisation."""
    con = fresh("a5_hostile.db")
    con.executescript("""
    CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT,
                           category TEXT, price REAL, stock_quantity INTEGER,
                           performance_score REAL);
    """)
    nasty = [
        "'; DROP TABLE products;--",
        "%_\\%",
        "Товар\u202e\u202d ELICIT",      # RTL override marks
        "Ноутбук 💻🔥 15\"",
        "x" * 4000,                      # oversized name
        "Café — naïve\nmultiline\tname",
        "' OR '1'='1",
        "UNION SELECT * FROM sqlite_master",
    ]
    for i, n in enumerate(nasty):
        con.execute("INSERT INTO products VALUES (?,?,?,?,?,?)",
                    (f"evil-{i}", n, "misc", 9.99 + i, 1, 0.0))
    for i in range(50):
        con.execute("INSERT INTO products VALUES (?,?,?,?,?,?)",
                    (f"ok-{i}", f"Звичайний товар {i}", "misc",
                     10.0 + i, 5, 1.0))
    con.commit()
    con.close()


# --------------------------------- A6: degenerate shapes
def a6_degenerate():
    """Empty table, single row, single column; and a 'wide' table where
    EAV-like long strings masquerade as columns."""
    con = fresh("a6_degenerate.db")
    con.executescript("""
    CREATE TABLE empty_products (id TEXT PRIMARY KEY, name TEXT,
                                 category TEXT, price REAL);
    CREATE TABLE single (id TEXT PRIMARY KEY, name TEXT, category TEXT,
                         price REAL, stock_quantity INTEGER,
                         performance_score REAL);
    CREATE TABLE one_col (id TEXT PRIMARY KEY, payload TEXT);
    """)
    con.execute("INSERT INTO single VALUES ('one','Єдиний','misc',1.0,1,0.0)")
    for i in range(100):
        con.execute("INSERT INTO one_col VALUES (?,?)",
                    (f"oc{i}", f"blob:{i}"))
    con.commit()
    con.close()


# ------------------------- A7: duplicate/near-duplicate entities
def a7_duplicates():
    """Same product offered under 8 spellings and two ids; a 5%-price
    divergence between offers. Resolution must not collapse or merge."""
    con = fresh("a7_dupes.db")
    con.executescript("""
    CREATE TABLE offers (id TEXT PRIMARY KEY, name TEXT, category TEXT,
                         price REAL, stock_quantity INTEGER,
                         performance_score REAL, shop TEXT);
    """)
    spellings = ["Кава Arabica 250г", "Кава Арабіка 250 г",
                 "Kawa Arabica 250g", "Кава \"Арабіка\" 250гр",
                 "Арабика 250г кава", "Кава АРАБІКА 250Г",
                 "Coffee Arabica 0.25kg", "Кава арабіка 250г."]
    for i, n in enumerate(spellings):
        con.execute("INSERT INTO offers VALUES (?,?,?,?,?,?,?)",
                    (f"dup-{i}", n, "grocery",
                     round(120.0 * (1 + 0.05 * (i % 3)), 2), 3, 0.5,
                     f"shop_{i % 4}"))
    for i in range(60):
        con.execute("INSERT INTO offers VALUES (?,?,?,?,?,?,?)",
                    (f"o-{i}", f"Інший товар {i}", "grocery",
                     50.0 + i, 1, 0.1, "shop_0"))
    con.commit()
    con.close()


# --------------------------- A8: high-cardinality attribute explosion
def a8_attr_explosion():
    """5k products, 1200 distinct attribute keys, 30% unique-per-item
    values -> EAV profiler pressure, induction noise floor."""
    con = fresh("a8_explosion.db")
    con.executescript("""
    CREATE TABLE items (id TEXT PRIMARY KEY, name TEXT, category TEXT,
                        price REAL);
    CREATE TABLE attrs (item_id TEXT, k TEXT, v TEXT);
    CREATE INDEX idx_attrs_item ON attrs(item_id);
    """)
    for i in range(5000):
        cat = f"cat_{i % 17}"
        con.execute("INSERT INTO items VALUES (?,?,?,?)",
                    (f"e{i}", f"Елемент {i}", cat, rng.uniform(1, 9999)))
        # DISTINCT keys per item -- repeated (item, key) pairs would make
        # the table a multiset-EAV, a shape the structural check is right
        # to refuse.
        keys = rng.sample(range(1200), rng.randint(3, 12))
        for j, kn in enumerate(keys):
            k = f"attr_{kn}"
            v = str(rng.uniform(0, 999)) if rng.random() < 0.6 else f"txt{i}_{j}"
            con.execute("INSERT INTO attrs VALUES (?,?,?)", (f"e{i}", k, v))
    con.commit()
    con.close()


if __name__ == "__main__":
    a1_sparse_eav(); a2_obfuscated_schema(); a3_no_price()
    a4_numeric_swamp(); a5_hostile_strings(); a6_degenerate()
    a7_duplicates(); a8_attr_explosion()
    for f in sorted(os.listdir(OUT)):
        print(f, os.path.getsize(os.path.join(OUT, f)), "bytes")
