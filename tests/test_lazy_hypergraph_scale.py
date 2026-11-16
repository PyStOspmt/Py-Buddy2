"""
TDD Suite for 10,000+ Item Marketplace Scaling & Lazy Implicit Hypergraph Projection.
Validates sub-5ms constraint evaluation over chaotic multi-vendor EAV data.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from domain_catalog import DomainCatalogRegistry
from marketplace_normalizer import MarketplaceEAVNormalizer

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
    """Test on-demand projection of individual products from 10k database."""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM products WHERE category = 'cpu' LIMIT 1;")
    target_id = cursor.fetchone()[0]
    conn.close()

    t0 = time.time()
    node = DomainCatalogRegistry.load_product_dynamically(DB_PATH, target_id)
    elapsed = time.time() - t0

    assert elapsed < 0.05  # Sub-50ms lookup
    assert node is not None
    assert node.category == "cpu"
    assert "socket" in node.attributes
    assert node.attributes["socket"] in ["AM5", "AM4", "LGA1700", "LGA1851"]
    assert "tdp_watts" in node.attributes
    assert node.attributes["tdp_watts"] > 0
    assert "vendor" in node.attributes

def test_cross_vendor_hypergraph_compatibility_check():
    """Test Virtual Hypergraph constraint validation across 2 different vendors."""
    kg = DomainCatalogRegistry.create_marketplace_catalog(DB_PATH)
    
    # Create two nodes with same socket from different vendors
    cpu_rozetka = MarketplaceEAVNormalizer.build_product_node(
        product_id="cpu_r1",
        name="AMD Ryzen 7 7800X3D",
        category="cpu",
        price=16200.0,
        raw_meta={"Сокет": "AM5", "TDP": "120W", "vendor": "Rozetka", "rating": "5.0"}
    )
    mb_it_brain = MarketplaceEAVNormalizer.build_product_node(
        product_id="mb_b1",
        name="MSI B650 Gaming Plus",
        category="motherboard",
        price=6700.0,
        raw_meta={"гніздо": "sAM5", "RAM_type": "DDR-5", "vendor": "IT-Brain", "rating": "4.8"}
    )
    mb_incompatible = MarketplaceEAVNormalizer.build_product_node(
        product_id="mb_intel",
        name="ASUS Prime Z790",
        category="motherboard",
        price=8300.0,
        raw_meta={"Socket Type": "LGA 1700", "vendor": "Telemart", "rating": "4.6"}
    )

    kg.add_product(cpu_rozetka)
    kg.add_product(mb_it_brain)
    kg.add_product(mb_incompatible)

    # Check compatible cross-vendor bundle
    t0 = time.time()
    is_valid = kg.is_compatible({"cpu_r1", "mb_b1"})
    conflicts = kg.find_conflicts({"cpu_r1", "mb_b1"})
    elapsed = time.time() - t0

    assert is_valid is True
    assert len(conflicts) == 0
    assert elapsed < 0.005  # < 5ms evaluation time

    # Check incompatible cross-vendor bundle
    is_invalid = kg.is_compatible({"cpu_r1", "mb_intel"})
    conflicts_bad = kg.find_conflicts({"cpu_r1", "mb_intel"})
    assert is_invalid is False
    assert len(conflicts_bad) > 0
    assert "AM5" in conflicts_bad[0] and "LGA1700" in conflicts_bad[0]
