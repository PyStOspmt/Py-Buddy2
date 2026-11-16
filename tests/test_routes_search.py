"""Route-level tests for the Retrieval-Fabric search API (Handoff 3).

FastAPI TestClient only -- no live server on :8000, no network, no
paid calls. Embedding runs the offline HashEmbedder via
`ctx.embedder_factory` (overridden before registration, as the
SystemContext docstring requires).
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault(
    "PYBUDDY_SESSIONS_DB",
    str(Path(tempfile.gettempdir())
        / f"pybuddy_test_sessions_{os.getpid()}.db"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from semantic_fixtures import HashEmbedder, make_flat_db
from server import app, ctx


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def reg_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "test-token")
    monkeypatch.setenv("PYBUDDY_SOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("PYBUDDY_RUNTIME_REGISTRY",
                       str(tmp_path / "runtime.json"))
    monkeypatch.setattr(ctx, "embedder_factory",
                        lambda: HashEmbedder())
    return tmp_path


def _register(client, tmp_path, n=8):
    db = make_flat_db(str(tmp_path / "src.db"), n=n)
    domain = f"sem_{uuid.uuid4().hex[:10]}"
    r = client.post("/api/register_domain",
                    json={"domain_id": domain, "db_path": db,
                          "async_init": True},
                    headers={"X-Admin-Token": "test-token"})
    assert r.status_code == 200, r.text
    return domain, db


def _wait_semantic(client, domain, states=("ready",), timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/domains/{domain}/semantic")
        assert r.status_code == 200, r.text
        if r.json()["state"] in states:
            return r.json()
        time.sleep(0.2)
    raise AssertionError(
        f"semantic stayed {client.get(f'/api/domains/{domain}/semantic').json()['state']}")


def _append_row(db):
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO entries(title, body, refcode) "
                 "VALUES ('new row', 'fresh body', 'CODE-7777')")
    conn.commit()
    conn.close()


def test_register_poll_ready_search(client, reg_env):
    domain, _db = _register(client, reg_env)
    deadline = time.time() + 90
    while time.time() < deadline:
        jobs = client.get("/api/init_jobs").json()["jobs"]
        if all(j["status"] in ("done", "failed", "cancelled")
               for j in jobs if j["domain_id"] == domain):
            break
        time.sleep(0.2)
    st = _wait_semantic(client, domain)
    assert st["state"] == "ready"
    assert st["entity"]["table"] == "entries"
    r = client.get(f"/api/domains/{domain}/search",
                   params={"q": "entry number 3 alpha", "k": 3})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "ok" and out["hits"]
    assert out["hits"][0]["name"].startswith("entry number")
    assert out["hits"][0]["provenance"]


def test_search_validation(client, reg_env):
    domain, _db = _register(client, reg_env)
    _wait_semantic(client, domain)
    assert client.get(
        f"/api/domains/{domain}/search", params={"q": ""}
    ).status_code == 422
    assert client.get(
        f"/api/domains/{domain}/search", params={"q": "x" * 501}
    ).status_code == 422
    for bad_k in (0, 51, -3):
        assert client.get(
            f"/api/domains/{domain}/search",
            params={"q": "entry", "k": bad_k}).status_code == 422
    assert client.get(
        f"/api/domains/{domain}/search",
        params={"q": "entry", "mode": "bogus"}).status_code == 422
    for mode in ("dense", "lexical", "hybrid"):
        assert client.get(
            f"/api/domains/{domain}/search",
            params={"q": "entry", "mode": mode}
        ).status_code == 200


def test_unknown_domain_404(client, monkeypatch):
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "test-token")
    assert client.get("/api/domains/no_such_x/search",
                      params={"q": "x"}).status_code == 404
    assert client.get("/api/domains/no_such_x/semantic"
                      ).status_code == 404
    # anonymous is refused (403); an admin gets the honest 404
    assert client.post("/api/domains/no_such_x/refresh"
                       ).status_code == 403
    assert client.post("/api/domains/no_such_x/refresh",
                       json={},
                       headers={"X-Admin-Token": "test-token"}
                       ).status_code == 404


def test_refresh_requires_admin(client, reg_env):
    domain, _db = _register(client, reg_env)
    _wait_semantic(client, domain)
    r = client.post(f"/api/domains/{domain}/refresh", json={})
    assert r.status_code == 403            # fail closed, no token
    r = client.post(f"/api/domains/{domain}/refresh",
                    json={"rebuild": False},
                    headers={"X-Admin-Token": "test-token"})
    assert r.status_code == 202, r.text
    _wait_semantic(client, domain)


def test_stale_fresh_transitions_via_api(client, reg_env):
    domain, db = _register(client, reg_env)
    _wait_semantic(client, domain)
    _append_row(db)
    st = client.get(f"/api/domains/{domain}/semantic").json()
    assert st["state"] == "stale" and st["fresh"] is False
    out = client.get(f"/api/domains/{domain}/search",
                     params={"q": "entry"}).json()
    assert out["status"] == "ok"
    assert out["index"]["fresh"] is False
    _wait_semantic(client, domain)        # auto refresh started


def test_inspection_has_semantic_and_entity_blocks(client, reg_env):
    domain, _db = _register(client, reg_env)
    _wait_semantic(client, domain)
    r = client.get(f"/api/domains/{domain}/inspection")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("semantic"), dict)
    assert body["semantic"]["state"] == "ready"
    assert isinstance(body.get("entity"), dict)
    assert body["entity"]["table"] == "entries"


@pytest.mark.live_model
@pytest.mark.skipif((os.environ.get("PYBUDDY_LIVE_LOCAL_MODEL") or os.environ.get("AINSA_LIVE_LOCAL_MODEL")) != "1",
                    reason="set PYBUDDY_LIVE_LOCAL_MODEL=1 to run")
def test_live_e5_crosslingual_route(client, reg_env, monkeypatch):
    from embedders import default_embedder
    monkeypatch.setattr(ctx, "embedder_factory", default_embedder)
    db = str(reg_env / "live.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE docs(did INTEGER PRIMARY KEY, "
                 "title TEXT, body TEXT)")
    conn.execute("INSERT INTO docs VALUES (1, 'Coffee brewing guide',"
                 " 'How to brew filter coffee with fresh beans')")
    conn.execute("INSERT INTO docs VALUES (2, 'Tax filing notes',"
                 " 'Annual income tax return checklist')")
    conn.commit()
    conn.close()
    domain = f"live_{uuid.uuid4().hex[:10]}"
    r = client.post("/api/register_domain",
                    json={"domain_id": domain, "db_path": db,
                          "async_init": True},
                    headers={"X-Admin-Token": "test-token"})
    assert r.status_code == 200, r.text
    _wait_semantic(client, domain, timeout=300)
    out = client.get(f"/api/domains/{domain}/search",
                     params={"q": "як заварити каву", "k": 2}).json()
    assert out["status"] == "ok"
    assert out["hits"][0]["name"] == "Coffee brewing guide"


def test_search_hyde_param_and_fallback(client, reg_env):
    domain, _db = _register(client, reg_env)
    _wait_semantic(client, domain)
    r = client.get(f"/api/domains/{domain}/search",
                   params={"q": "entry", "hyde": "true", "k": 3})
    assert r.status_code == 200, r.text
    out = r.json()
    # the test runner has no api key -> generator returns "" -> HyDE
    # degrades visibly, never silently
    assert out["mode"] in ("hyde", "hybrid")
    if out["mode"] == "hybrid":
        assert out["requested_mode"] == "hyde"
        assert any("hyde unavailable" in n for n in out["notes"])
    else:
        assert out["hypothesis"]["generated_by"] == "llm"
