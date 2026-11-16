"""
Py-Buddy: Dynamic Multi-Domain Isolation & Universal Schema Ingestion Test.
Proves the removal of C1, C4, and C7:
- C1: Chat routing is strictly isolated per active/requested domain DB.
- C4: Dynamic ZCCS (Zero Code Change Schema ingestion) via register_domain.
- C7: Real SQL pagination over large databases removing the 120-node ceiling.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from server import app, ctx


@pytest.fixture
def client():
    return TestClient(app)

#: F29: domain switching/registration is admin-only and fails closed without
#: a token, so these tests authenticate like a real administrator would.
@pytest.fixture
def admin_headers(monkeypatch):
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "test-token")
    return {"X-Admin-Token": "test-token"}

def test_c1_domain_isolation_in_chat(client, admin_headers):
    """
    C1 Proof: When chatting in domain A (pc_hardware) vs domain B (marketplace_rozetka),
    the agent and fast_router query their respective databases.
    """
    # 1. Query PC Hardware domain
    res_pc = client.post("/api/chat", json={
        "query": "Яка найдешевша деталь у наявності?",
        "domain": "pc_hardware"
    })
    assert res_pc.status_code == 200
    pc_data = res_pc.json()
    assert pc_data["domain"] == "pc_hardware"

    # 2. Query Marketplace Rozetka domain
    res_mk = client.post("/api/chat", json={
        "query": "Яка найдешевша деталь у наявності?",
        "domain": "marketplace_rozetka"
    })
    assert res_mk.status_code == 200
    mk_data = res_mk.json()
    assert mk_data["domain"] == "marketplace_rozetka"

    # 3. Switching active domain updates ctx.active_domain and active router
    client.post("/api/switch_domain", json={"domain": "marketplace_rozetka"},
                headers=admin_headers)
    assert ctx.active_domain == "marketplace_rozetka"
    assert "marketplace_rozetka.db" in ctx.fast_router.db_path

    # Restore to pc_hardware
    client.post("/api/switch_domain", json={"domain": "pc_hardware"},
                headers=admin_headers)
    assert ctx.active_domain == "pc_hardware"
    assert "catalog.db" in ctx.fast_router.db_path

def test_c7_real_database_pagination_over_ceiling(client):
    """
    C7 Proof: In marketplace_rozetka, the catalog endpoint accesses the full
    10,000 product database with pagination, removing the hardcoded 120 limit.
    """
    # Request first page
    res_p1 = client.get("/api/catalog?domain=marketplace_rozetka&limit=50&offset=0")
    assert res_p1.status_code == 200
    data_p1 = res_p1.json()
    assert data_p1["total"] >= 1000  # Full DB scale (10,000 or fixture scale)
    assert len(data_p1["products"]) == 50
    assert data_p1["offset"] == 0

    # Request second page
    res_p2 = client.get("/api/catalog?domain=marketplace_rozetka&limit=50&offset=50")
    assert res_p2.status_code == 200
    data_p2 = res_p2.json()
    assert len(data_p2["products"]) == 50
    assert data_p2["offset"] == 50

    # Ensure items on page 1 and page 2 are different
    ids_p1 = {p["id"] for p in data_p1["products"]}
    ids_p2 = {p["id"] for p in data_p2["products"]}
    assert len(ids_p1.intersection(ids_p2)) == 0

def test_c4_dynamic_domain_registration(client, admin_headers, monkeypatch, tmp_path):
    """
    C4 Proof: Dynamic domain registration (ZCCS = 0).
    A new SQLite database created on the fly can be registered and queried
    without any code changes.
    """
    import sqlite3

    # F29: registration only accepts databases inside the configured source
    # root, so the test points the root at its own temporary directory.
    monkeypatch.setenv("PYBUDDY_SOURCE_ROOT", str(tmp_path))

    db_file = tmp_path / "custom_books.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL, stock_quantity INT);")
    conn.execute("INSERT INTO products VALUES ('b1', 'Kobzar Taras Shevchenko', 'books', 450.0, 10);")
    conn.execute("INSERT INTO products VALUES ('b2', 'Eneida Ivan Kotliarevsky', 'books', 380.0, 5);")
    conn.commit()
    conn.close()

    # Register new domain
    res_reg = client.post("/api/register_domain", json={
        "domain_id": "ukr_books",
        "name": "Ukrainian Classic Literature",
        "db_path": str(db_file),
        "description": "Library and bookstore archive"
    }, headers=admin_headers)
    assert res_reg.status_code == 200
    reg_data = res_reg.json()
    assert reg_data["status"] == "success"
    assert reg_data["domain_id"] == "ukr_books"
    assert reg_data["nodes_count"] >= 2

    # Query catalog of the newly registered domain
    res_cat = client.get("/api/catalog?domain=ukr_books")
    assert res_cat.status_code == 200
    cat_data = res_cat.json()
    assert cat_data["total"] == 2
    assert any("Kobzar" in p["name"] for p in cat_data["products"])

    # Switch domain and execute chat query.
    # TEST DOUBLE: scripts LLM step selection so dynamic DB registration,
    # SQL execution, GroundingSet, and ClaimVerifier run hermetically offline without
    # calling OpenRouter during pytest (live LLM runs are evaluated separately in benchmark_aic2026.py).
    res_sw = client.post("/api/switch_domain", json={"domain": "ukr_books"},
                         headers=admin_headers)
    assert res_sw.status_code == 200

    _react_calls = {"n": 0}

    def _scripted_books_react(*args, **kwargs):
        _react_calls["n"] += 1
        if _react_calls["n"] == 1:
            return {
                "thought": "Query the cheapest available book in ukr_books",
                "action": "sql_query",
                "action_input": {
                    "query": (
                        "SELECT id, name, category, price FROM products "
                        "WHERE stock_quantity > 0 ORDER BY price ASC LIMIT 1"
                    )
                },
            }
        return {
            "thought": "Report the grounded cheapest book from SQL output",
            "final_answer": (
                "Найдешевша книга у наявності — Eneida Ivan Kotliarevsky (380 грн)."
            ),
            "recommended_items": [{"name": "Eneida Ivan Kotliarevsky"}],
        }

    import openrouter_client

    monkeypatch.setattr(
        openrouter_client.OpenRouterAgentRunner,
        "generate_react_step",
        lambda self, *args, **kwargs: _scripted_books_react(*args, **kwargs),
    )

    res_chat = client.post("/api/chat", json={
        "query": "Яка найдешевша книга у наявності?",
        "domain": "ukr_books"
    })
    assert res_chat.status_code == 200
    chat_data = res_chat.json()
    assert chat_data["domain"] == "ukr_books"
    assert "Eneida" in chat_data["final_answer"] or "380" in chat_data["final_answer"]

    # Switch back to pc_hardware
    client.post("/api/switch_domain", json={"domain": "pc_hardware"},
                headers=admin_headers)

def test_upload_domain_source_registers_sqlite_file(client, admin_headers, monkeypatch, tmp_path):
    """Upload path: an arbitrary .db file posted as multipart lands inside
    the source root, is validated as real SQLite, and is registered as a
    runtime domain -- the 'drop any database at it' demo path."""
    import sqlite3

    monkeypatch.setenv("PYBUDDY_SOURCE_ROOT", str(tmp_path))

    src = tmp_path / "seed.db"
    conn = sqlite3.connect(str(src))
    conn.execute("CREATE TABLE items (id TEXT PRIMARY KEY, name TEXT, price REAL)")
    conn.execute("INSERT INTO items VALUES ('i1', 'Uploaded Item', 7.0)")
    conn.commit()
    conn.close()
    payload = src.read_bytes()

    res = client.post(
        "/api/domains/upload",
        files={"file": ("uploaded.db", payload, "application/octet-stream")},
        data={"domain_id": "up_test", "name": "Uploaded Test"},
        headers=admin_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "accepted"
    assert body["domain_id"] == "up_test"
    assert body["bytes"] == len(payload)
    assert body["db_path"].startswith(str(tmp_path))
    assert "up_test" in ctx.domains
    ctx.domains.pop("up_test", None)


def test_upload_converts_csv_to_sqlite(client, admin_headers, monkeypatch, tmp_path):
    """A .csv upload is converted to a SQLite source with provenance in
    ingest_meta, then registered like any other domain."""
    import sqlite3

    monkeypatch.setenv("PYBUDDY_SOURCE_ROOT", str(tmp_path))
    csv_bytes = b"sku,title,cost\ni1,Widget,9.5\ni2,Gadget,12.0\n"

    res = client.post(
        "/api/domains/upload",
        files={"file": ("items.csv", csv_bytes, "text/csv")},
        data={"domain_id": "up_csv", "name": "CSV Test"},
        headers=admin_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "accepted"
    assert body["converted"] is True
    assert "up_csv" in ctx.domains
    dest = tmp_path / "up_csv.db"
    assert dest.exists()
    conn = sqlite3.connect(str(dest))
    rows = conn.execute("SELECT sku, title, cost FROM records ORDER BY sku").fetchall()
    meta = dict(conn.execute("SELECT key, value FROM ingest_meta").fetchall())
    conn.close()
    assert rows == [("i1", "Widget", 9.5), ("i2", "Gadget", 12.0)]
    assert meta.get("origin_file") == "items.csv"
    assert meta.get("converted") == "tabular->sqlite"
    ctx.domains.pop("up_csv", None)


def test_upload_rejects_non_sqlite_and_missing_admin(client, admin_headers, monkeypatch, tmp_path):
    monkeypatch.setenv("PYBUDDY_SOURCE_ROOT", str(tmp_path))
    # no admin token -> fail closed
    res = client.post(
        "/api/domains/upload",
        files={"file": ("x.db", b"SQLite format 3\x00" + b"0" * 100, "application/octet-stream")},
        data={"domain_id": "up_noauth"},
    )
    assert res.status_code == 403
    # bad magic -> explicit rejection, nothing registered
    res = client.post(
        "/api/domains/upload",
        files={"file": ("fake.db", b"not a database", "application/octet-stream")},
        data={"domain_id": "up_fake"},
        headers=admin_headers,
    )
    assert res.status_code == 400
    assert "up_fake" not in ctx.domains
    assert not (tmp_path / "up_fake.db").exists()


def test_conductor_endpoint_verifies_links_on_existing_domain(
        client, admin_headers, monkeypatch, tmp_path):
    """The conductor endpoint lets a domain registered WITHOUT
    llm_assist still run hypothesis->verification: a scripted model
    proposes an undeclared link, the measured overlap admits it, and the
    cached projection exposes it to /inspection immediately."""
    import json
    import sqlite3

    db = tmp_path / "cond_api.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE lookup (id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL, suit TEXT)")
    conn.executemany("INSERT INTO lookup VALUES (?,?)",
                     [("d1", "одяг"), ("d2", "взуття")])
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [
        (f"p{i}", f"Виріб {i}", f"g{i % 3}", 10.0 + i,
         "d2" if i == 4 else "d1") for i in range(20)])
    conn.commit()
    conn.close()

    from hypergraph import KnowledgeHypergraph
    from server import DomainBundle
    bundle = DomainBundle("cond_api", "cond", str(db),
                          KnowledgeHypergraph(), "test")
    bundle.state = bundle.state.__class__.READY
    ctx.domains["cond_api"] = bundle

    class Scripted:
        def __init__(self):
            self.answers = [
                json.dumps({"tool": "propose_link", "args": {
                    "child_table": "products", "child_col": "suit",
                    "parent_table": "lookup", "parent_col": "id"}}),
                json.dumps({"done": True}),
            ]

        def generate(self, prompt, temperature=0.0):
            return self.answers.pop(0) if self.answers else '{"done":true}'

    orig = ctx.runner
    ctx.runner = Scripted()
    try:
        # no admin token -> fail closed
        res = client.post("/api/domains/cond_api/conductor")
        assert res.status_code == 403

        res = client.post("/api/domains/cond_api/conductor",
                          headers=admin_headers)
        assert res.status_code == 200, res.text
        rep = res.json()
        assert rep["conductor"] is True
        assert rep["links_accepted"] == 1

        # the same cached projection now carries the verified link,
        # so inspection reports it without a rebuild or restart
        insp = client.get("/api/domains/cond_api/inspection").json()
        links = insp["projection"]["inferred_links"]
        assert links and links[0]["child_column"] == "suit"
        assert links[0]["overlap"] == 1.0
        # and the trace grew a link_inferred event for the canvas
        kinds = [e["kind"] for e in insp["projection"]["build_trace"]]
        assert "link_inferred" in kinds
    finally:
        ctx.runner = orig
        ctx.domains.pop("cond_api", None)
