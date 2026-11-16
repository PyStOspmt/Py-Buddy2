"""
Regression suite: schema-agnostic source projection and role mapping invariants.
"""

import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
sys.path.insert(0, os.path.join(ROOT, "src"))


from catalog_query import fetch_candidates, resolve_mentions
from domain_catalog import DomainCatalogRegistry
from pareto_relaxer import CandidateSolution, FuzzyParetoRelaxer
from source_projection import build_source_projection

# ---------------------------------------------------------------------------
# Fixtures: three deliberately renamed / schema-less databases
# ---------------------------------------------------------------------------

def _make_ambiguous_db(tmp_path):
    """
    A store whose price column is not named like one. `amount_due` holds
    1..5 ratings (its name contains the price hint "amount"), `cena` holds
    the real prices but carries no hint at all.
    """
    path = str(tmp_path / "ambiguous.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE goods (
            gid INTEGER PRIMARY KEY,
            title TEXT,
            amount_due REAL,
            cena REAL,
            topic TEXT
        );
        """
    )
    topics = ["prose", "poetry", "essays"]
    for i in range(12):
        conn.execute(
            "INSERT INTO goods VALUES (?,?,?,?,?)",
            (i + 1, f"Book #{i + 1}", (i % 5) + 1, 500.0 + 250 * i, topics[i % 3]),
        )
    conn.commit()
    conn.close()
    return path


def _make_library_db(tmp_path):
    """A library: renamed entity table plus a fully renamed EAV meta table."""
    path = str(tmp_path / "library.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE biblio (
            bid INTEGER PRIMARY KEY,
            nazva TEXT,
            genre TEXT,
            topic TEXT
        );
        CREATE TABLE item_attr (ref INTEGER, k TEXT, v TEXT);
        """
    )
    genres = ["novel", "drama"]
    topics = ["war", "peace", "sea"]
    for i in range(8):
        conn.execute(
            "INSERT INTO biblio VALUES (?,?,?,?)",
            (i + 1, f"Volume {i + 1}", genres[i % 2], topics[i % 3]),
        )
        conn.execute("INSERT INTO item_attr VALUES (?,?,?)", (i + 1, "author", f"Author-{i % 2}"))
        conn.execute("INSERT INTO item_attr VALUES (?,?,?)", (i + 1, "year", str(1901 + i)))
    conn.commit()
    conn.close()
    return path


def _make_renamed_marketplace_db(tmp_path):
    """A marketplace whose every table and column is renamed."""
    path = str(tmp_path / "renamed_market.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tovary (
            sku INTEGER PRIMARY KEY,
            nazva TEXT,
            kat TEXT,
            vartist REAL,
            kilkist INTEGER,
            prodavec_id INTEGER
        );
        CREATE TABLE prodavtsi (pid INTEGER PRIMARY KEY, imya TEXT);
        CREATE TABLE oznaky (tov INTEGER, klyuch TEXT, znachennya TEXT);
        """
    )
    cats = ["cpu", "gpu", "psu", "ram"]
    for v in range(1, 6):
        conn.execute("INSERT INTO prodavtsi VALUES (?,?)", (v, f"Postach {v}"))
    for i in range(60):
        conn.execute(
            "INSERT INTO tovary VALUES (?,?,?,?,?,?)",
            (i + 1, f"Detail {i + 1} Pro", cats[i % 4], 1000.0 + 50 * i, 5 + i % 3, (i % 5) + 1),
        )
        conn.execute("INSERT INTO oznaky VALUES (?,?,?)", (i + 1, "vyrobnyk", f"Brand-{i % 3}"))
        conn.execute("INSERT INTO oznaky VALUES (?,?,?)", (i + 1, "garantia", f"{12 + i} mis"))
    conn.commit()
    conn.close()
    return path


def _make_null_price_db(tmp_path):
    """Standard marketplace shape, but one row states no price at all."""
    path = str(tmp_path / "null_price.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            stock_quantity INTEGER
        );
        """
    )
    conn.execute("INSERT INTO products VALUES (1,'Item A','cat1',100.0,3)")
    conn.execute("INSERT INTO products VALUES (2,'Item B','cat1',NULL,3)")
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# F09 · introspection hypotheses carry evidence; ambiguity stays visible
# ---------------------------------------------------------------------------

def test_f09_ambiguous_price_columns_stay_unresolved_and_visible(tmp_path):
    """
    Failed before: the profiler's name hint made `amount_due` THE price
    (semantic_type="price" acted as ground truth), so ratings 1..5 would
    have been ranked and filtered as money while the real price column was
    invisible. Now the hint is outvoted by value evidence and the role stays
    unresolved with every candidate visible.
    """
    proj = build_source_projection(_make_ambiguous_db(tmp_path))
    price = proj.resolve("price")
    assert not price.resolved
    hinted = [c for c in price.candidates if c.column == "amount_due"]
    assert hinted, "the name-hint candidate must stay visible"
    assert hinted[0].vetoed
    assert any(e.kind == "value" for e in hinted[0].evidence)
    assert "cena" in proj.unassigned_numeric


def test_f09_renamed_eav_table_is_detected_structurally(tmp_path):
    """
    Failed before: `_detect_eav_structures` matched only hard-coded names
    (meta_key/meta_value/...), so a renamed meta table contributed no
    metadata at all. Now the cardinality signature detects it.
    """
    proj = build_source_projection(_make_library_db(tmp_path))
    assert "item_attr" in proj.eav_tables
    meta = proj.eav_tables["item_attr"]
    assert meta["detection"] == "structural"
    assert meta["key_column"] == "k"
    assert meta["entity_fk"] == "ref"


def test_f09_structural_roles_survive_column_renaming(tmp_path):
    """
    Failed before: identity/name/category were usable only when the columns
    carried the expected English names. Now structural evidence (primary
    key, distinct-ratio patterns) resolves them on a renamed schema.
    """
    proj = build_source_projection(_make_library_db(tmp_path))
    assert proj.entity_table == "biblio"
    assert proj.resolve("identity").column == "bid"
    name = proj.resolve("name")
    assert name.resolved and name.column == "nazva"
    assert any(e.kind == "structural" for c in name.candidates for e in c.evidence)
    # Two low-cardinality text columns: the structural tie is broken only by
    # the explicit category hint on `genre`, and both candidates stay visible.
    category = proj.resolve("category")
    assert category.resolved and category.column == "genre"
    assert {c.column for c in category.candidates} == {"genre", "topic"}


def test_f09_dimension_table_is_not_mistaken_for_eav(tmp_path):
    """
    Failed before (design-level): a purely cardinality-based EAV detector
    would flag a small dimension table (unique text name per row) as EAV
    and then join garbage metadata. The key-repetition signature prevents it.
    """
    path = str(tmp_path / "dims.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE goods (gid INTEGER PRIMARY KEY, title TEXT, cena REAL);
        CREATE TABLE postachalnyky (pid INTEGER PRIMARY KEY, imya TEXT, reytynh REAL);
        """
    )
    for i in range(10):
        conn.execute("INSERT INTO goods VALUES (?,?,?)", (i + 1, f"Item {i}", 100.0 + i))
        conn.execute("INSERT INTO postachalnyky VALUES (?,?,?)", (i + 1, f"Vendor {i}", 4.0 + (i % 3)))
    conn.commit()
    conn.close()

    proj = build_source_projection(path)
    assert "postachalnyky" not in proj.eav_tables
    assert proj.entity_table == "goods"


# ---------------------------------------------------------------------------
# F10 · "any database" paths consume projections, not a fixed product schema
# ---------------------------------------------------------------------------

def test_f10_library_without_price_stays_searchable(tmp_path):
    """
    Failed before: `create_database_catalog` returned an empty graph and
    `fetch_candidates` returned [] for any database without a `products`
    table; a budget filter would also crash on the missing price column.
    """
    db = _make_library_db(tmp_path)
    kg = DomainCatalogRegistry.create_database_catalog(db)
    assert len(kg.nodes) == 8
    assert all(n.price is None for n in kg.nodes.values())

    rows = fetch_candidates(db, category=None, budget_max=100.0)
    assert len(rows) == 8
    assert all(r.price is None for r in rows)

    hits = resolve_mentions(db, "Volume 3")
    assert len(hits) == 1 and hits[0].name == "Volume 3"


def test_f10_renamed_marketplace_projects_without_product_code_changes(tmp_path):
    """
    Failed before: every "any DB" path SELECTed hard-coded `products`,
    `name`, `category`, `price`, `product_eav_meta` columns and crashed or
    returned nothing on a renamed schema. Now the projection drives it.
    """
    db = _make_renamed_marketplace_db(tmp_path)
    kg = DomainCatalogRegistry.create_database_catalog(db)
    assert len(kg.nodes) == 60
    node = kg.nodes["1"]
    assert node.name == "Detail 1 Pro"
    assert node.price is None  # `vartist` carries no price hint: honest absence
    # Renamed EAV metadata is attached through the structural detection.
    assert "vyrobnyk" in node.attributes

    hits = resolve_mentions(db, "Detail 7 Pro")
    assert hits and hits[0].item_id == "7"

    loaded = DomainCatalogRegistry.load_product_dynamically(db, "2")
    assert loaded is not None
    assert loaded.name == "Detail 2 Pro"
    assert loaded.price is None


def test_f10_null_price_is_not_fabricated_as_zero(tmp_path):
    """
    Failed before: `float(price or 0.0)` turned an unstated price into a
    stated 0.0, making the item look free in every ranking.
    """
    db = _make_null_price_db(tmp_path)
    kg = DomainCatalogRegistry.create_database_catalog(db)
    assert kg.nodes["1"].price == 100.0
    assert kg.nodes["2"].price is None

    loaded = DomainCatalogRegistry.load_product_dynamically(db, "2")
    assert loaded.price is None


def test_f10_pareto_without_price_axis_is_honest(tmp_path):
    """
    Failed before: a None price would either crash the membership math or
    silently act as the cheapest possible item (0.0 <= any budget).
    """
    relaxer = FuzzyParetoRelaxer(budget_max=1000.0, min_performance=50.0)
    unpriced = CandidateSolution(item_id="x", name="X", price=None, performance_score=80.0)
    memberships = relaxer.compute_memberships(unpriced)
    assert "price" not in memberships
    assert "performance" in memberships

    front = relaxer.compute_pareto_front([unpriced])
    best, contrast, explanation = relaxer.select_contrastive_tradeoffs(front)
    assert "ціна не вказана" in explanation


def test_f10_marketplace_projection_is_unchanged(tmp_path):
    """
    Guard: on the standard marketplace shape the projection must resolve the
    same columns the hard-coded path used, so nothing regresses for the
    existing domains.
    """
    path = str(tmp_path / "standard.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            stock_quantity INTEGER,
            performance_score REAL,
            merchant_id INTEGER
        );
        CREATE TABLE merchants (id INTEGER PRIMARY KEY, name TEXT, rating REAL, delivery_days INTEGER);
        CREATE TABLE product_eav_meta (product_id INTEGER, meta_key TEXT, meta_value TEXT);
        """
    )
    for m in range(1, 4):
        conn.execute("INSERT INTO merchants VALUES (?,?,?,?)", (m, f"Shop {m}", 4.5, 2))
    for i in range(30):
        conn.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?)",
            (i + 1, f"Prod {i}", "cpu", 2000.0 + i, 10, 70.0 + i, (i % 3) + 1),
        )
        conn.execute("INSERT INTO product_eav_meta VALUES (?,?,?)", (i + 1, "socket", "AM5"))
    conn.commit()
    conn.close()

    proj = build_source_projection(path)
    assert proj.entity_table == "products"
    assert proj.resolve("identity").column == "id"
    assert proj.resolve("name").column == "name"
    assert proj.resolve("category").column == "category"
    assert proj.resolve("price").column == "price"
    assert proj.resolve("stock").column == "stock_quantity"
    assert proj.resolve("performance").column == "performance_score"
    assert "product_eav_meta" in proj.eav_tables

    kg = DomainCatalogRegistry.create_database_catalog(db_path=path)
    node = kg.nodes["1"]
    assert node.price == 2000.0
    assert node.attributes.get("socket") == "AM5"

    rows = fetch_candidates(path, category="cpu", budget_max=5000.0)
    assert rows and rows[0].price == 2000.0