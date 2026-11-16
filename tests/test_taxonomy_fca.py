"""
Tests for Formal Concept Analysis & Dynamic Taxonomy Engine (src/taxonomy_fca.py)
Validates mathematical concept hierarchy, zero domain hardcoding, and Horizon Truncation prevention.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import sqlite3
import tempfile

from taxonomy_fca import FCATaxonomyEngine


@pytest.fixture
def sample_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
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
    # Clothing domain
    cur.execute("INSERT INTO products VALUES ('c1', 'Silk Evening Dress', 'women_clothing', 'Zara', 2500.0)")
    cur.execute("INSERT INTO products VALUES ('c2', 'Cotton Summer Dress', 'women_clothing', 'Zara', 1200.0)")
    cur.execute("INSERT INTO products VALUES ('c3', 'Casual Denim Jacket', 'women_clothing', 'Mango', 3100.0)")
    cur.execute("INSERT INTO products VALUES ('c4', 'Classic Wool Suit', 'men_clothing', 'Hugo Boss', 15000.0)")
    # Tech domain
    cur.execute("INSERT INTO products VALUES ('t1', 'GeForce RTX 4060', 'gpu', 'Gigabyte', 13200.0)")
    cur.execute("INSERT INTO products VALUES ('t2', 'GeForce RTX 4070', 'gpu', 'Gigabyte', 27000.0)")
    cur.execute("INSERT INTO products VALUES ('t3', 'Radeon RX 7800 XT', 'gpu', 'Sapphire', 24900.0)")
    conn.commit()
    conn.close()

    yield path

    if os.path.exists(path):
        os.remove(path)

def test_fca_lattice_construction(sample_db):
    engine = FCATaxonomyEngine(sample_db)
    summary = engine.get_concept_summary()

    assert summary["root_objects"] == 7
    assert summary["categories_count"] == 3  # women_clothing, men_clothing, gpu
    assert summary["lattice_depth"] == 2      # Root (0) -> Category (1) -> (Category, Brand) (2)

def test_fca_drilldown_navigation(sample_db):
    engine = FCATaxonomyEngine(sample_db)

    # 1. Search for clothing category
    nav_clothing = engine.navigate("жіночий одяг women_clothing")
    assert nav_clothing["matched"] is True
    assert nav_clothing["level"] == 1
    assert nav_clothing["extent_size"] == 3
    assert "c1" in nav_clothing["sample_ids"]

    # 2. Search for specific brand within category
    nav_gigabyte_gpu = engine.navigate("відеокарта gpu від gigabyte")
    assert nav_gigabyte_gpu["matched"] is True
    assert nav_gigabyte_gpu["level"] == 2
    assert nav_gigabyte_gpu["extent_size"] == 2
    assert "t1" in nav_gigabyte_gpu["sample_ids"]
    assert "t2" in nav_gigabyte_gpu["sample_ids"]
    assert "t3" not in nav_gigabyte_gpu["sample_ids"]  # Sapphire is filtered out

def test_fca_horizon_truncation_resolution(sample_db):
    engine = FCATaxonomyEngine(sample_db)
    
    # User asks for GPU Gigabyte: the engine returns exact concept with extent size 2,
    # completely avoiding scanning all 7 products or suffering from flat LIMIT truncation.
    res = engine.navigate("Gigabyte 4060 GPU")
    assert res["matched"] is True
    assert res["extent_size"] == 2
    assert set(res["intent"]) == {"category:gpu", "brand:Gigabyte"}
