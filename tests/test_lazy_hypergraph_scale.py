"""
TDD Suite for 10,000+ Item Marketplace Scaling & Lazy Implicit Hypergraph Projection.
Validates sub-5ms constraint evaluation over chaotic multi-vendor EAV data.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from domain_catalog import DomainCatalogRegistry
from hypergraph import ProductNode

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'marketplace_rozetka.db'))

def test_marketplace_db_scale():
    """Verify that marketplace database has at least 10,000 products and thousands of EAV attributes."""
    import sqlite3
    assert os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM products;")
    p_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM product_eav_meta;")
    meta_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM merchants;")
    m_count = cursor.fetchone()[0]
    conn.close()

    assert p_count >= 10000
    assert meta_count >= 20000
    assert m_count >= 6

def test_lazy_product_loading_and_normalization():
    """Test on-demand projection of individual products from 10k database without seed rewriting."""
    import sqlite3
    from attribute_algebra import canonical_form, spec_of
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM products WHERE category = 'cpu' LIMIT 1;")
    target_id = cursor.fetchone()[0]
    cursor.execute("SELECT DISTINCT meta_key FROM product_eav_meta;")
    all_keys_ctx = " ".join(str(r[0]) for r in cursor.fetchall() if r[0])
    conn.close()

    # First call profiles and caches the schema projection; measure on-demand product lookup.
    DomainCatalogRegistry.load_product_dynamically(DB_PATH, target_id)
    t0 = time.time()
    node = DomainCatalogRegistry.load_product_dynamically(DB_PATH, target_id)
    elapsed = time.time() - t0

    assert elapsed < 0.05  # Sub-50ms lookup
    assert node is not None
    assert node.category == "cpu"
    assert node.attributes
    assert any(
        spec_of(str(v), key="socket").dimension == "P"
        for v in node.attributes.values()
    )
    # F50: cross-language labels such as "Socket AM4" under key "гніздо"
    # canonicalize only using the corpus's EAV key vocabulary, never a baked lexicon.
    assert any(
        canonical_form(str(v), key=all_keys_ctx) in ("am5", "am4", "lga1700", "lga1851")
        for v in node.attributes.values()
    )

def test_cross_vendor_hypergraph_compatibility_check():
    """Test Virtual Hypergraph constraint validation across 2 different vendors with heterogeneous EAV keys."""
    kg = DomainCatalogRegistry.create_marketplace_catalog(DB_PATH)

    cpu_rozetka = ProductNode(
        id="cpu_r1",
        name="AMD Ryzen 7 7800X3D",
        category="cpu",
        price=16200.0,
        attributes={"Сокет": "Socket AM5", "TDP": "120W", "vendor": "m_rozetka"},
    )
    mb_it_brain = ProductNode(
        id="mb_b1",
        name="MSI B650 Gaming Plus",
        category="motherboard",
        price=6700.0,
        attributes={"гніздо": "sAM5", "Пам'ять": "DDR5", "vendor": "m_telemart"},
    )
    mb_incompatible = ProductNode(
        id="mb_intel",
        name="ASUS Prime Z790",
        category="motherboard",
        price=8300.0,
        attributes={"Socket Type": "LGA1700", "vendor": "m_autodrive"},
    )

    kg.add_product(cpu_rozetka)
    kg.add_product(mb_it_brain)
    kg.add_product(mb_incompatible)

    # Different sellers must NOT veto a socket-compatible pair
    t0 = time.time()
    is_valid = kg.is_compatible({"cpu_r1", "mb_b1"})
    conflicts = kg.find_conflicts({"cpu_r1", "mb_b1"})
    elapsed = time.time() - t0

    assert is_valid is True
    assert len(conflicts) == 0
    assert elapsed < 0.005  # < 5ms evaluation time

    # Incompatible cross-vendor socket mismatch IS blocked conservatively
    is_invalid = kg.is_compatible({"cpu_r1", "mb_intel"})
    conflicts_bad = kg.find_conflicts({"cpu_r1", "mb_intel"})
    assert is_invalid is False
    assert len(conflicts_bad) > 0
    assert "AM5" in conflicts_bad[0] and "LGA1700" in conflicts_bad[0]
