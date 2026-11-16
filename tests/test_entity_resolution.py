"""
Unit and integration tests for Multi-Vendor Entity Resolution (C6 resolution / Slot 4).
Validates clustering of noisy listings, seller offer extraction, and attribute harmonization.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from entity_resolution import MarketplaceEntityResolver

MARKETPLACE_DB = os.path.join(ROOT, "data", "marketplace_rozetka.db")

def test_normalize_entity_key():
    k1 = MarketplaceEntityResolver.normalize_entity_key("AMD Ryzen 7 7800X3D (Rev 1)")
    k2 = MarketplaceEntityResolver.normalize_entity_key("AMD Ryzen 7 7800X3D (Rev 4)")
    k3 = MarketplaceEntityResolver.normalize_entity_key("amd ryzen 7 7800x3d [BOX]")
    assert k1 == k2 == k3
    assert "7800x3d" in k1
    assert "rev" not in k1

@pytest.mark.skipif(not os.path.exists(MARKETPLACE_DB), reason="marketplace_rozetka.db missing")
def test_resolve_marketplace_clusters():
    entities = MarketplaceEntityResolver.resolve_from_database(MARKETPLACE_DB, limit_listings=2000)
    assert len(entities) > 0
    # In 2000 listings with 6 sellers per item, we expect ~40-100 canonical entities, not 2000!
    assert len(entities) < 500

    # Pick an entity and verify multi-seller offers
    sample_entity = next(e for e in entities.values() if len(e.offers) > 1)
    assert sample_entity.seller_count >= 2
    assert sample_entity.min_price <= sample_entity.max_price
    cheapest = sample_entity.get_cheapest_offer()
    assert cheapest is not None
    assert cheapest.price == sample_entity.min_price

    # Verify harmonized attributes
    assert isinstance(sample_entity.harmonized_attributes, dict)


def test_entity_resolution_and_legacy_meta_handle_custom_table(tmp_path):
    """MarketplaceEntityResolver and DomainCatalogRegistry work cleanly without a `products` table."""
    import sqlite3
    from domain_catalog import DomainCatalogRegistry

    db_file = tmp_path / "custom_store.sqlite"
    con = sqlite3.connect(str(db_file))
    con.execute(
        "CREATE TABLE warehouse_skus ("
        "sku_code TEXT PRIMARY KEY, sku_title TEXT, grp TEXT, maker TEXT, unit_price REAL, qty INTEGER)"
    )
    con.execute("CREATE TABLE product_eav_meta (product_id TEXT, attr_key TEXT, attr_value TEXT)")
    con.execute(
        "INSERT INTO warehouse_skus VALUES "
        "('k1', 'Widget Pro [BOX]', 'gadgets', 'Acme', 1200.0, 5),"
        "('k2', 'Widget Pro (Rev 2)', 'gadgets', 'Acme', 1250.0, 3),"
        "('k3', 'Gadget Mini', 'gadgets', 'Acme', 1800.0, 2)"
    )
    con.execute("INSERT INTO product_eav_meta VALUES ('k1', 'tier', 'gold')")
    con.commit()
    con.close()

    resolved = MarketplaceEntityResolver.resolve_from_database(str(db_file), limit_listings=10)
    assert len(resolved) == 2
    entity = next(e for e in resolved.values() if len(e.offers) == 2)
    assert entity.min_price == 1200.0 and entity.max_price == 1250.0

    con_ro = sqlite3.connect(str(db_file))
    meta = DomainCatalogRegistry._legacy_merchant_meta(
        con_ro.cursor(), ["k1", "k2"], entity_table="warehouse_skus", ident_col="sku_code"
    )
    con_ro.close()
    assert meta == {}
