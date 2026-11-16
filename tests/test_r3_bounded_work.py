"""
Bounded work and pagination safety regressions.

Failing-before counterexamples:

- /api/catalog accepted an unlimited page size and a negative offset
  (SQLite treats a negative LIMIT as unlimited);
- the catalog page hydrated every row with a separate lazy
  load_product_dynamically call (N+1 queries per page);
- /api/validate_bundle accepted an unbounded id list and ran an O(n^2)
  pair check over all of them;
- the literal-mention resolver scanned the entire entity table with no
  bound before filtering names in Python.
"""

import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))

from fastapi.testclient import TestClient


def _make_db(path, rows, table="products"):
    if os.path.exists(path):
        os.unlink(path)
    conn = sqlite3.connect(str(path))
    conn.execute(
        f"CREATE TABLE {table} (id TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL)")
    conn.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


@pytest.fixture
def client():
    from server import app
    return TestClient(app)


@pytest.fixture
def ctx():
    from server import ctx
    return ctx


@pytest.fixture
def registered_db(ctx, tmp_path):
    """A fresh registered source with more rows than the seed limit (120),
    so pages past the seed need lazy hydration."""
    db = tmp_path / "bounded.db"
    # Zero-padded names keep alphabetical order identical to rowid order,
    # so the 120-node seed deterministically covers the first 120 rows only.
    _make_db(db, [(f"id{i}", f"Item {i:03d}", "cat", float(i)) for i in range(150)])
    from server import DomainBundle
    ctx.register_domain_bundle(
        DomainBundle("bounded_src", "bounded", str(db), None, "test"))
    yield str(db)
    ctx.domains.pop("bounded_src", None)


class TestCatalogBounds:
    def test_limit_is_capped(self, client, registered_db):
        res = client.get("/api/catalog?domain=bounded_src&limit=100000")
        assert res.status_code == 200
        data = res.json()
        assert data["limit"] <= 200
        assert len(data["products"]) <= 200

    def test_negative_offset_is_rejected_or_clamped(self, client, registered_db):
        res = client.get("/api/catalog?domain=bounded_src&offset=-5")
        assert res.status_code == 200
        assert res.json()["offset"] == 0

    def test_negative_limit_cannot_select_all(self, client, registered_db):
        res = client.get("/api/catalog?domain=bounded_src&limit=-1")
        assert res.status_code == 200
        # SQLite treats LIMIT -1 as unlimited; the applied bound must prevent that.
        assert len(res.json()["products"]) <= 200

    def test_page_hydration_is_bulk_not_per_row(self, ctx, client, registered_db, monkeypatch):
        import server as server_mod

        calls = {"bulk": 0, "single": 0}
        real_bulk = server_mod.DomainCatalogRegistry.load_products_dynamically
        real_single = server_mod.DomainCatalogRegistry.load_product_dynamically

        def counted_bulk(db_path, ids):
            calls["bulk"] += 1
            return real_bulk(db_path, ids)

        def counted_single(db_path, pid):
            calls["single"] += 1
            return real_single(db_path, pid)

        monkeypatch.setattr(server_mod.DomainCatalogRegistry,
                            "load_products_dynamically", staticmethod(counted_bulk))
        monkeypatch.setattr(server_mod.DomainCatalogRegistry,
                            "load_product_dynamically", staticmethod(counted_single))

        # Rows past the 120-node seed are absent from the hypergraph and
        # must be hydrated -- once, in bulk, not per row.
        res = client.get("/api/catalog?domain=bounded_src&limit=20&offset=130")
        assert res.status_code == 200
        products = res.json()["products"]
        assert len(products) == 20
        assert calls["bulk"] == 1
        assert calls["single"] == 0
        assert all(isinstance(p["attributes"], dict) for p in products)


class TestBundleValidationBounds:
    def test_oversized_id_list_is_rejected(self, client):
        res = client.post("/api/validate_bundle",
                          json={"product_ids": [f"p{i}" for i in range(500)],
                                "domain": "pc_hardware"})
        assert res.status_code == 400

    def test_pairwise_links_are_bounded(self, client):
        res = client.post("/api/validate_bundle",
                          json={"product_ids": [f"p{i}" for i in range(60)],
                                "domain": "pc_hardware"})
        assert res.status_code == 200
        data = res.json()
        assert data["graph_links_truncated"] is True
        assert len(data["graph_data"]["links"]) <= 780  # 40 choose 2


class TestMentionResolverBounds:
    def test_scan_is_bounded(self, tmp_path):
        db = tmp_path / "mentions.db"
        _make_db(db, [(f"id{i}", f"Unrelated {i}", "cat", 1.0) for i in range(20)]
                 + [("target", "The Named Product", "cat", 9.0)])

        from catalog_query import resolve_mentions
        # Full scan finds the named product at the end of the table.
        hits = resolve_mentions(str(db), "do you carry The Named Product", scan_cap=100)
        assert any(h.name == "The Named Product" for h in hits)
        # A capped scan that stops before the row honestly reports limited
        # coverage -- it may miss, it must not fabricate.
        capped = resolve_mentions(str(db), "do you carry The Named Product", scan_cap=5)
        assert all(h.name != "The Named Product" for h in capped)
