import os
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hypergraph import KnowledgeHypergraph, ProductNode
from sanitizer import SQLSanitizer


def test_ukrainian_apostrophe_escaping():
    sanitizer = SQLSanitizer()
    raw = "SELECT * FROM product_eav_meta WHERE meta_key = 'Пам'ять' AND meta_value LIKE '%60%'"
    sanitized = sanitizer.sanitize_and_prepare_query(raw)
    assert "'Пам''ять'" in sanitized
    assert "LIMIT" in sanitized

def test_fuzzy_fallback_search_when_zero_rows(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE products (
        id TEXT PRIMARY KEY,
        name TEXT,
        category TEXT,
        brand TEXT,
        price REAL
    )
    """)
    cur.execute("INSERT INTO products VALUES ('p1', 'Gigabyte GeForce RTX 4060 Windforce (Rev 2)', 'gpu', 'Gigabyte', 12658.93)")
    cur.execute("INSERT INTO products VALUES ('p2', 'Gigabyte GeForce RTX 4060 Windforce (Rev 1)', 'gpu', 'Gigabyte', 13200.00)")
    cur.execute("INSERT INTO products VALUES ('p3', 'ASUS TUF GeForce RTX 4090', 'gpu', 'ASUS', 89000.00)")
    conn.commit()

    sanitizer = SQLSanitizer()
    # Typo query: 'giga byte' instead of 'Gigabyte'
    q = "SELECT id, name, price FROM products WHERE LOWER(name) LIKE '%giga byte 4060%'"
    # Regression (docs/13 C1): the literal path "ignored" made sqlite create a
    # junk file named `ignored` in the repository root on every test run.
    # Use a throwaway database in the test's temp directory instead.
    # F31: the sanitizer is fail-closed now — a read-only connection to a
    # missing file is refused instead of silently creating one, so the
    # throwaway database is materialised explicitly.
    db_path = str(tmp_path / "empty.db")
    src = sqlite3.connect(db_path)
    conn.backup(src)
    src.close()
    res = sanitizer.execute_safe_query(db_path, q)
    # Zero rows on the real database -> the adaptive fuzzy fallback fires.
    assert res.get("fuzzy_suggestions"), "zero rows must trigger the fuzzy fallback"
    # The helper can be called directly or via fallback. F10: the fallback
    # discovers the entity table through the source projection, which needs
    # the database path -- an anonymous in-memory connection carries none.
    hits = sanitizer._fuzzy_fallback_search(conn, q, limit=5, db_path=db_path)
    assert len(hits) >= 1
    assert "4060" in hits[0]["name"]
    assert hits[0]["price"] == 12658.93  # Best price first!

def test_open_basket_compatibility():
    hg = KnowledgeHypergraph()
    # In general grocery / retail, no constraints are registered
    p1 = ProductNode(id="off:1", name="100% Soja Protein Haselnuss", category="food", price=0.0)
    p2 = ProductNode(id="off:2", name="Best Ginger Snap Cookies", category="food", price=0.0)
    hg.add_product(p1)
    hg.add_product(p2)

    assert hg.is_compatible({"off:1", "off:2"}) is True
    assert hg.find_conflicts({"off:1", "off:2"}) == []

def test_domain_agnostic_signal_extraction():
    from fast_router import FastCascadedRouter
    from react_agent import ReActAgent
    from sanitizer import SQLSanitizer
    router = FastCascadedRouter(
        db_path=":memory:",
        sanitizer=SQLSanitizer(),
        react_agent=ReActAgent(db_path=":memory:"),
        llm_runner=None,
        schema_prompt=""
    )
    # Alphanumeric codes or units in arbitrary domain (food, auto, tech)
    s1 = router.extract_domain_signals("Чи є у вас масло 500ml або сир 200g ?")
    assert s1["has_tech_specs"] is True

    s2 = router.extract_domain_signals("Яка ціна на позицію B650 чи шини R17 ?")
    assert s2["has_tech_specs"] is True

