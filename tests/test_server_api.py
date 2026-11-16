"""
Unit and Integration Tests for Py-Buddy FastAPI Server & Omnichannel Gateways.
Validates all REST endpoints, Virtual Hypergraph bundle validation, Pareto optimization,
and Telegram/Discord message formatting.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from server import app, ctx

from gateways.discord_adapter import DiscordGateway
from gateways.telegram_adapter import TelegramGateway


@pytest.fixture
def client():
    return TestClient(app)

def test_api_status_endpoint(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "Py-Buddy" in data["architecture"]
    assert "Py-Buddy" in data["lineage"]
    assert data["invariants"]["zero_hallucination_gate"] == "ACTIVE"

def test_api_catalog_endpoint(client):
    res = client.get("/api/catalog?domain=pc_hardware")
    assert res.status_code == 200
    data = res.json()
    assert data["domain"] == "pc_hardware"
    assert data["total"] > 0
    assert any("7800X3D" in p["name"] for p in data["products"])

def test_api_switch_domain(client, monkeypatch):
    # F30: switching the global domain is now an administrative operation.
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "test-token")
    admin = {"X-Admin-Token": "test-token"}
    # Switch to auto parts
    res = client.post("/api/switch_domain", json={"domain": "auto_parts"}, headers=admin)
    assert res.status_code == 200
    assert res.json()["active_domain"] == "auto_parts"

    # Verify catalog contains auto parts
    res_cat = client.get("/api/catalog?domain=auto_parts")
    assert res_cat.status_code == 200
    assert any("Golf" in p["name"] for p in res_cat.json()["products"])

    # Switch back to pc_hardware
    client.post("/api/switch_domain", json={"domain": "pc_hardware"}, headers=admin)

def test_api_validate_bundle_compatibility(client):
    # 1. Compatible PC bundle: AM5 CPU + B650 Board
    res_valid = client.post("/api/validate_bundle", json={
        "product_ids": ["cpu_am5_1", "mb_am5_msi"],
        "domain": "pc_hardware"
    })
    assert res_valid.status_code == 200
    data_v = res_valid.json()
    # The demo catalogue's socket rule is INDUCED (status=candidate), so it
    # may flag a conflict but cannot vouch for compatibility: a passing
    # pair is honestly "unknown", and is_compatible must be null -- the
    # old test asserted the permissive basket answer which contradicted
    # the verdict field on the same response (R6 live finding).
    assert data_v["verdict"] == "unknown"
    assert data_v["is_compatible"] is None
    assert len(data_v["conflicts"]) == 0

    # 2. Incompatible PC bundle: AM5 CPU + Z790 Intel Board
    res_invalid = client.post("/api/validate_bundle", json={
        "product_ids": ["cpu_am5_1", "mb_intel_asus"],
        "domain": "pc_hardware"
    })
    assert res_invalid.status_code == 200
    data_inv = res_invalid.json()
    assert data_inv["verdict"] == "candidate_conflict"
    assert data_inv["is_compatible"] is False
    assert len(data_inv["candidate_conflicts"]) >= 1
    assert any("AM5" in c and "LGA1700" in c for c in data_inv["candidate_conflicts"])

    # 3. Auto parts cross-domain bundle: Golf Hub (5x112) + BMW Rim (5x120)
    res_auto = client.post("/api/validate_bundle", json={
        "product_ids": ["hub_golf", "rim_bmw_m"],
        "domain": "auto_parts"
    })
    assert res_auto.status_code == 200
    data_auto = res_auto.json()
    assert data_auto["is_compatible"] is False
    assert any("5x112" in c and "5x120" in c
               for c in data_auto["candidate_conflicts"] or data_auto["conflicts"])

def test_api_pareto_endpoint(client):
    res = client.post("/api/pareto", json={
        "budget_max": 15000.0,
        "min_performance": 85.0,
        "domain": "pc_hardware"
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["all_candidates"]) > 0
    assert len(data["pareto_front"]) > 0
    assert data["best_a"] is not None
    assert "Парето" in data["explanation"] or "рішення" in data["explanation"]

def test_api_pareto_front_is_exactly_the_non_dominated_set(client):
    res = client.post("/api/pareto", json={
        "budget_max": 12000.0,
        "min_performance": 80.0,
        "domain": "pc_hardware"
    })
    assert res.status_code == 200
    data = res.json()
    points = {c["item_id"]: (c["price"], c["performance_score"])
              for c in data["all_candidates"]}

    def dominates(a, b):
        (pa, sa), (pb, sb) = points[a], points[b]
        return pa <= pb and sa >= sb and (pa < pb or sa > sb)

    expected = {i for i in points if not any(dominates(j, i) for j in points if j != i)}
    assert {c["item_id"] for c in data["pareto_front"]} == expected


def test_api_pareto_feasible_front_is_the_non_dominated_set_of_feasible_items(client):
    res = client.post("/api/pareto", json={
        "budget_max": 12000.0,
        "min_performance": 80.0,
        "domain": "pc_hardware"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["front_scope"] == "feasible"
    points = {c["item_id"]: (c["price"], c["performance_score"])
              for c in data["all_candidates"]
              if c["price"] <= 12000.0 and c["performance_score"] >= 80.0}

    def dominates(a, b):
        (pa, sa), (pb, sb) = points[a], points[b]
        return pa <= pb and sa >= sb and (pa < pb or sa > sb)

    expected = {i for i in points if not any(dominates(j, i) for j in points if j != i)}
    assert {c["item_id"] for c in data["pareto_front_feasible"]} == expected


def test_api_pareto_feasible_front_responds_to_the_budget_slider(client):
    fronts = {}
    for budget in (5000.0, 12000.0):
        res = client.post("/api/pareto", json={
            "budget_max": budget,
            "min_performance": 50.0,
            "domain": "pc_hardware"
        })
        fronts[budget] = frozenset(c["item_id"] for c in res.json()["pareto_front_feasible"])
    assert fronts[5000.0] != fronts[12000.0]

def test_omnichannel_telegram_formatting():
    tg = TelegramGateway(agent=ctx.agent)
    dummy_result = {
        "final_answer": "Рекомендую процесор AMD Ryzen 7 7800X3D за 16500 грн.",
        "steps": [{"step": 1, "thought": "Пошук ціни", "action": {"tool": "QueryDB"}}]
    }
    formatted = tg.format_response(dummy_result)
    assert formatted["parse_mode"] == "HTML"
    assert "<b>Консультант Py-Buddy</b>" in formatted["text"]
    assert "16500 грн" in formatted["text"]
    assert "кроків: 1" in formatted["text"]

def test_omnichannel_discord_formatting():
    dc = DiscordGateway(agent=ctx.agent)
    # F55: colour follows the structured verdict, not prose substrings.
    # A conflict reported through step verdicts -> Crimson color.
    conflict_result = {
        "final_answer": "The requested parts cannot be used together.",
        "steps": [{"step": 1, "verdict": "incompatible"}]
    }
    formatted = dc.format_response(conflict_result)
    embed = formatted["embeds"][0]
    assert embed["color"] == 0xEF4444
    # Prose alone -- even conflict-sounding prose -- evaluates nothing and
    # must not paint the embed red.
    prose_only = {
        "final_answer": "Несумісність: AM5 не підходить до плати LGA1700.",
        "steps": []
    }
    embed2 = dc.format_response(prose_only)["embeds"][0]
    assert embed2["color"] != 0xEF4444

def test_api_telemetry_endpoint(client):
    res = client.get("/api/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "active_invariants" in data
    assert data["active_invariants"]["zero_hallucination_gate"]["status"] == "ACTIVE"
    assert data["active_invariants"]["fast_cascaded_router"]["status"] == "ACTIVE"

def test_api_marketplace_domain_switch_and_catalog(client, monkeypatch):
    # F30: switching the global domain is now an administrative operation.
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "test-token")
    admin = {"X-Admin-Token": "test-token"}
    # Switch to marketplace_rozetka
    res = client.post("/api/switch_domain", json={"domain": "marketplace_rozetka"}, headers=admin)
    assert res.status_code == 200
    assert res.json()["active_domain"] == "marketplace_rozetka"

    # Search in marketplace catalog
    res_cat = client.get("/api/catalog?domain=marketplace_rozetka&search=Ryzen")
    assert res_cat.status_code == 200
    cat_data = res_cat.json()
    assert cat_data["domain"] == "marketplace_rozetka"
    assert len(cat_data["products"]) > 0
    assert any("Ryzen" in p["name"] for p in cat_data["products"])

    # Switch back to pc_hardware
    client.post("/api/switch_domain", json={"domain": "pc_hardware"}, headers=admin)

def test_api_chat_tier0_instant_response(client, monkeypatch):
    class MockServerRunner:
        def generate_react_step(self, user_query, **kwargs):
            return {"action": None, "final_answer": "Вітаю! Я інтелектуальний консультант Py-Buddy."}
    monkeypatch.setattr(ctx.fast_router, "runner", MockServerRunner())
    res = client.post("/api/chat", json={"query": "Привіт! Підкажи, як ти працюєш?"})
    assert res.status_code == 200
    data = res.json()
    assert data["tier"] == "tier0_conversational"
    assert len(data["steps"]) >= 1
    assert "thought" in data["steps"][0]
    assert "action" in data["steps"][0]
    # The console renders the gate verdict as a badge; the API must carry it.
    assert data["termination_reason"] == "SUCCESS_GROUNDED"
    assert "recommended_items" in data
    assert "token_cost" in data


def test_api_chat_persists_multiturn_conversation_history(client, monkeypatch):
    seen = []

    class RecordingRunner:
        def generate_react_step(self, user_query, dialog_history=None, **kwargs):
            seen.append(list(dialog_history or []))
            return {"action": None, "final_answer": "Вітаю! Чим допомогти?"}

    monkeypatch.setattr(ctx.fast_router, "runner", RecordingRunner())
    r1 = client.post("/api/chat", json={"query": "Привіт, підбираю процесор."})
    assert r1.status_code == 200
    sid = r1.json()["session_id"]
    r2 = client.post("/api/chat", json={"query": "Підкажи, як ти працюєш?", "session_id": sid})
    assert r2.status_code == 200
    assert seen[0] == []
    assert len(seen[1]) == 2
    assert "Привіт, підбираю процесор." in seen[1][0]["content"]


def test_api_chat_deterministic_hypergraph_branch(client):
    """Two catalogue names verbatim -> hypergraph answer, not a model call."""
    res = client.post("/api/chat", json={
        "query": "Чи сумісні AMD Ryzen 7 7800X3D та MSI B650 Gaming Plus WiFi?"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["tier"] is None
    assert data["model"] is None
    assert data["termination_reason"] == "DETERMINISTIC_HYPERGRAPH"
    assert len(data["recommended_items"]) >= 2
    assert data["steps"][0]["action"]["tool"] == "HypergraphValidate"


def test_api_models_and_set_model(client, monkeypatch):
    import openrouter_client

    openrouter_client._clear_models_cache()
    monkeypatch.setattr(ctx.runner, "api_key", None)

    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    assert res_status.json()["llm_available"] is False
    assert res_status.json()["model"] is None

    res_empty = client.get("/api/models")
    assert res_empty.status_code == 200
    empty_data = res_empty.json()
    assert empty_data["key_configured"] is False
    assert empty_data["models_count"] == 0
    assert empty_data["models"] == []
    assert empty_data["active_model"] is None

    class _Resp:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}
        def json(self):
            return self._payload

    monkeypatch.setattr(openrouter_client.requests, "get", lambda *a, **kw: _Resp(401, {"error": "invalid"}))
    res_bad = client.get("/api/models", headers={"X-OpenRouter-Key": "sk-or-bad"})
    assert res_bad.status_code == 200
    bad_data = res_bad.json()
    assert bad_data["key_configured"] is True
    assert bad_data["models_count"] == 0
    assert bad_data["error"]

    openrouter_client._clear_models_cache()
    def _mock_good_get(url, **kwargs):
        if url.endswith("/key"):
            return _Resp(200, {"data": {"label": "test"}})
        return _Resp(200, {
            "data": [
                {"id": "nvidia/nemotron-3-super-120b-a12b:free", "name": "Nemotron (free)",
                 "context_length": 131072, "pricing": {"prompt": "0", "completion": "0"}},
                {"id": "openai/gpt-4o-mini", "name": "GPT-4o-mini",
                 "context_length": 128000, "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}},
            ]
        })

    monkeypatch.setattr(openrouter_client.requests, "get", _mock_good_get)
    res = client.get("/api/models", headers={"X-OpenRouter-Key": "sk-or-valid"})
    assert res.status_code == 200
    data = res.json()
    assert data["key_configured"] is True
    assert data["key_source"] == "client"
    assert data["source"] == "openrouter_live"
    assert data["models_count"] == 2
    assert any(m["is_free"] for m in data["models"])

    target = "nvidia/nemotron-3-super-120b-a12b:free"
    # F30: switching the global model is now an administrative operation.
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "test-token")
    res_set = client.post("/api/set_model", json={"model": target},
                          headers={"X-Admin-Token": "test-token"})
    assert res_set.status_code == 200
    assert res_set.json()["active_model"] == target
    assert ctx.runner.model_name == target

def test_api_taxonomy_endpoint(client):
    res = client.get("/api/taxonomy?domain=pc_hardware")
    assert res.status_code == 200
    data = res.json()
    assert data["domain"] == "pc_hardware"
    assert "tree" in data
    tree = data["tree"]
    assert tree["level"] == 0
    assert tree["item_count"] > 0
    assert len(tree["children"]) > 0




def test_api_deregister_domain(client, monkeypatch, tmp_path):
    """DELETE /api/domains/{id} removes runtime sources only: demo- and
    code-registered ids are refused, persisted rows are removed."""
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "test-token")
    admin = {"X-Admin-Token": "test-token"}
    reg = tmp_path / "sources.runtime.json"
    monkeypatch.setenv("PYBUDDY_RUNTIME_REGISTRY", str(reg))
    monkeypatch.setenv("PYBUDDY_SOURCE_ROOT", str(tmp_path))

    db = tmp_path / "del.db"
    import sqlite3 as _sq
    conn = _sq.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL)")
    conn.execute("INSERT INTO products VALUES ('p1','n1','c',1.0)")
    conn.commit()
    conn.close()

    # Unauthenticated -> 403 (fail-closed admin check).
    assert client.delete("/api/domains/whatever", ).status_code == 403

    r = client.post("/api/register_domain",
                    json={"domain_id": "del_src", "db_path": str(db)},
                    headers=admin)
    assert r.status_code in (200, 202), r.text

    r = client.delete("/api/domains/del_src", headers=admin)
    assert r.status_code == 200, r.text
    assert r.json()["persisted_registration_removed"] is True
    assert "del_src" not in ctx.domains
    assert "del_src" not in str(reg.read_text())

    # Demo-registered source cannot be deleted through the API.
    r = client.delete("/api/domains/pc_hardware", headers=admin)
    assert r.status_code == 409
    assert "pc_hardware" in ctx.domains

    assert client.delete("/api/domains/ghost_src", headers=admin).status_code == 404


def test_api_chat_hypergraph_and_tier0_include_verifier_fields(client, monkeypatch):
    """All /api/chat routes expose verification_report and pre_verifier_response keys."""
    class Tier0Mock:
        def generate_react_step(self, user_query, **kwargs):
            return {"action": None, "final_answer": "Вітаю! Я готовий допомогти."}

    monkeypatch.setattr(ctx.fast_router, "runner", Tier0Mock())
    r0 = client.post("/api/chat", json={"query": "Привіт, як ти працюєш?"})
    assert r0.status_code == 200
    d0 = r0.json()
    assert "verification_report" in d0
    assert "pre_verifier_response" in d0
    # Tier-0 prose still runs through ClaimVerifier (no factual claims -> 0 supported/0 unsupported).
    assert d0["verification_report"]["n_unsupported"] == 0

    rh = client.post("/api/chat", json={
        "query": "Чи сумісні AMD Ryzen 7 7800X3D та MSI B650 Gaming Plus WiFi?"
    })
    assert rh.status_code == 200
    dh = rh.json()
    assert dh["termination_reason"] == "DETERMINISTIC_HYPERGRAPH"
    assert dh["verification_report"] is None
    assert dh["pre_verifier_response"] is None


def test_api_chat_exposes_verification_report_and_diff_on_strip(client, monkeypatch):
    """ReAct /api/chat exposes ClaimVerifier report and raw pre-verifier prose when stripped."""
    cat = client.get("/api/catalog?domain=pc_hardware").json()["products"]
    target = next(p for p in cat if "7800X3D" in p["name"])
    real_price = int(round(float(target["price"])))
    calls = {"n": 0}

    class StripProbeRunner:
        model_name = "mock-probe"

        def generate_react_step(self, user_query, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "thought": "Перевіряю точну позицію в SQLite.",
                    "action": "sql_query",
                    "action_input": {
                        "query": (
                            "SELECT id, name, category, brand, price, performance_score "
                            "FROM products WHERE category = 'cpu' LIMIT 1"
                        ),
                    },
                    "final_answer": None,
                }
            return {
                "thought": "Маю факти з БД, відповідаю.",
                "action": None,
                "recommended_items": [target["name"]],
                "final_answer": (
                    f"Процесор {target['name']} коштує {real_price} грн. "
                    "Він також має 64 ядра на частоті 9.9 GHz та кеш 512 MB."
                ),
            }

    probe = StripProbeRunner()
    monkeypatch.setattr(ctx.fast_router, "runner", probe)

    res = client.post("/api/chat", json={
        "query": "Порівняй тепловиділення та енергоспоживання AMD Ryzen 7 7800X3D у робочій станції?"
    })
    assert res.status_code == 200
    data = res.json()
    assert calls["n"] >= 2, data
    vr = data["verification_report"]
    assert isinstance(vr, dict), data
    assert vr["n_supported"] >= 1, vr
    assert vr["n_unsupported"] >= 1, vr
    assert data["pre_verifier_response"] is not None
    assert "9.9 GHz" in data["pre_verifier_response"]
    assert "9.9 GHz" not in data["final_answer"]


def test_nvfg_relaxation_exposes_protected_axes():
    """tool_pareto_relax and summarize_answer_nvfg carry protected axis labels cleanly."""
    from answer_nvfg import summarize_answer_nvfg
    from react_agent import SessionState
    from react_tools import tool_pareto_relax

    state = SessionState(session_id="test-nvfg-prot", user_query="cpu до 500 грн")
    rel = tool_pareto_relax(
        ctx.agent,
        state,
        {
            "category": "cpu",
            "budget_max": 500.0,
            "min_performance": 70.0,
            "protected": ["бюджет"],
        },
    )
    assert "protected" in rel
    assert "бюджет" in rel["protected"]
    summary = summarize_answer_nvfg(
        [{"tool_name": "pareto_relax", "result": rel}],
        rel.get("nvfg_status") or "UNSATISFIABLE",
    )
    assert summary["protected"] == ["бюджет"]
    assert "relaxation" in summary
    assert "regret" in summary


def test_web_ui_wires_verifier_module():
    """index.html loads verifier_ui.js before app.js and app.js wires PyBuddyVerifierUI."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    index_html = (root / "web" / "index.html").read_text(encoding="utf-8")
    app_js = (root / "web" / "app.js").read_text(encoding="utf-8")
    vui_js = (root / "web" / "verifier_ui.js").read_text(encoding="utf-8")

    idx_vui = index_html.find("/static/verifier_ui.js")
    idx_app = index_html.find("/static/app.js")
    assert 0 < idx_vui < idx_app
    assert "PyBuddyVerifierUI" in app_js
    assert "annotateProvenance" in app_js
    assert "renderVerificationPanel" in app_js
    assert "renderRelaxationCard" in app_js
    assert "renderVerifierBadge" in vui_js
    assert "constraints/status" in app_js
    assert "btn-rule-status" in app_js


def test_admin_constraint_promotion_updates_compat_verdict(client, monkeypatch):
    """Adversarial check: candidate rules yield unknown until operator verifies with admin token."""
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "test-secret")
    admin = {"X-Admin-Token": "test-secret"}

    insp0 = client.get("/api/domains/pc_hardware/inspection").json()
    rule_ids = [
        c["rule_id"] for c in insp0["hypergraph"]["constraints"] if c.get("rule_id")
    ]
    assert rule_ids, "expected at least one induced constraint with rule_id"

    # Without token: 403 and verdict remains honest unknown
    r_unauth = client.post(
        "/api/domains/pc_hardware/constraints/status",
        json={"rule_id": rule_ids[0], "status": "verified"},
    )
    assert r_unauth.status_code == 403

    v0 = client.post(
        "/api/validate_bundle",
        json={"domain": "pc_hardware", "product_ids": ["cpu_am5_1", "mb_am5_msi"]},
    ).json()
    assert v0["verdict"] == "unknown"
    assert v0["is_compatible"] is None

    # Unknown rule_id or bad status are rejected cleanly
    assert client.post(
        "/api/domains/pc_hardware/constraints/status",
        headers=admin,
        json={"rule_id": "missing_rule", "status": "verified"},
    ).status_code == 404
    assert client.post(
        "/api/domains/pc_hardware/constraints/status",
        headers=admin,
        json={"rule_id": rule_ids[0], "status": "invented"},
    ).status_code == 422

    # Promote all induced candidate rules to verified -> compatible becomes True
    for rid in rule_ids:
        r_ok = client.post(
            "/api/domains/pc_hardware/constraints/status",
            headers=admin,
            json={"rule_id": rid, "status": "verified"},
        )
        assert r_ok.status_code == 200
        assert r_ok.json()["status"] == "verified"

    v1 = client.post(
        "/api/validate_bundle",
        json={"domain": "pc_hardware", "product_ids": ["cpu_am5_1", "mb_am5_msi"]},
    ).json()
    assert v1["verdict"] == "compatible"
    assert v1["is_compatible"] is True

    # Cleanup: demote back to candidate so shared ctx stays unpolluted
    for rid in rule_ids:
        r_demote = client.post(
            "/api/domains/pc_hardware/constraints/status",
            headers=admin,
            json={"rule_id": rid, "status": "candidate"},
        )
        assert r_demote.status_code == 200
        assert r_demote.json()["status"] == "candidate"

    v2 = client.post(
        "/api/validate_bundle",
        json={"domain": "pc_hardware", "product_ids": ["cpu_am5_1", "mb_am5_msi"]},
    ).json()
    assert v2["verdict"] == "unknown"
    assert v2["is_compatible"] is None


def test_openrouter_runner_defers_missing_key_only_when_requested(monkeypatch):
    """Server context can initialize offline without OPENROUTER_API_KEY while LLM invocations fail closed."""
    from openrouter_client import OpenRouterAgentRunner

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _p: False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterAgentRunner()

    runner = OpenRouterAgentRunner(require_key=False)
    assert runner.api_key is None
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        runner._ensure_api_key()


def test_live_arm_openrouter_stops_immediately_on_auth_error(monkeypatch):
    """Invalid API keys (401/403) must fail on the first attempt without retry storms."""
    import openrouter_client

    calls = []
    sleeps = []

    class _FakeResp:
        status_code = 401
        text = '{"error":"invalid"}'

    monkeypatch.setattr(
        openrouter_client.requests,
        "post",
        lambda *args, **kwargs: (calls.append(kwargs.get("json") or kwargs.get("data")) or _FakeResp()),
    )
    monkeypatch.setattr(openrouter_client.time, "sleep", lambda s: sleeps.append(s))
    runner = openrouter_client.OpenRouterAgentRunner(
        api_key="invalid-test-key", model_name="test/m"
    )
    assert runner.generate("hello") == ""
    assert len(calls) == 1
    assert sleeps == []
    calls.clear()
    res = runner.generate_react_step("schema", "hi", [], 1, "tools")
    assert "provider_error" in res
    assert len(calls) == 1
    assert sleeps == []


def test_domain_currency_and_null_price_honesty(client):
    """Domain metadata exposes real currency (or None), and null DB prices never turn into 0."""
    domains_res = client.get("/api/domains")
    assert domains_res.status_code == 200
    by_id = {d["id"]: d for d in domains_res.json()["domains"]}

    assert by_id["pc_hardware"]["currency"] == "UAH"
    assert by_id["wdc_products"]["currency"] == "USD"
    assert by_id["openprices"]["currency"] == "EUR"
    assert by_id["openfoodfacts"]["currency"] is None

    wdc = client.get("/api/catalog?domain=wdc_products&limit=50").json()
    assert wdc["currency"] == "USD"
    assert any(p["price"] is None for p in wdc["products"])
    assert all(p["price"] is None or p["price"] > 0 for p in wdc["products"])

    off = client.get("/api/catalog?domain=openfoodfacts&limit=20").json()
    assert off["currency"] is None
    assert all(p["price"] is None for p in off["products"])

    pareto_res = client.post(
        "/api/pareto",
        json={"domain": "wdc_products", "budget_max": 2000.0, "min_performance": 0.0},
    )
    assert pareto_res.status_code == 200
    pareto_wdc = pareto_res.json()
    assert pareto_wdc["currency"] == "USD"
    assert "USD" in pareto_wdc["explanation"]
    assert "грн" not in pareto_wdc["explanation"]
    assert pareto_wdc["price_range"]["max"] > pareto_wdc["price_range"]["min"] > 0

    pareto_off = client.post(
        "/api/pareto",
        json={"domain": "openfoodfacts", "budget_max": 100.0, "min_performance": 0.0},
    ).json()
    assert pareto_off["currency"] is None
    assert pareto_off["price_range"] is None

    st_pc = client.get("/api/status?domain=pc_hardware").json()
    st_wdc = client.get("/api/status?domain=wdc_products").json()
    assert st_pc["active_domain"] == "pc_hardware"
    assert st_wdc["active_domain"] == "wdc_products"
    assert st_wdc["active_currency"] == "USD"
    assert st_wdc["nodes_count"] != st_pc["nodes_count"]


def test_fast_router_tier1_uses_domain_currency_and_handles_null_price(tmp_path):
    import sqlite3
    from fast_router import ExecutionTier, FastCascadedRouter
    from sanitizer import SQLSanitizer

    db_usd = tmp_path / "usd_items.db"
    conn = sqlite3.connect(str(db_usd))
    conn.executescript(
        """
        CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, brand TEXT, price REAL, stock_quantity INTEGER);
        CREATE TABLE ingest_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO ingest_meta VALUES ('currency_kept', 'USD');
        INSERT INTO products VALUES
            ('u0', 'Unpriced Placeholder', 'gpu', 'NVIDIA', NULL, 3),
            ('u1', 'Quadro M4000', 'gpu', 'NVIDIA', 837.99, 3),
            ('u2', 'Quadro P5000', 'gpu', 'NVIDIA', 1299.00, 2);
        """
    )
    conn.commit()
    conn.close()

    class _DummyAgent:
        currency = "USD"

    router = FastCascadedRouter(
        db_path=str(db_usd),
        sanitizer=SQLSanitizer(),
        react_agent=_DummyAgent(),
        llm_runner=None,
        schema_prompt="",
    )
    res_usd = router.route_and_execute("Який найдешевший товар?")
    assert res_usd["tier"] == ExecutionTier.TIER1_FAST_PATH
    assert "837.99 USD" in res_usd["answer"]
    assert "грн" not in res_usd["answer"]
    assert "837.99 USD" in res_usd["steps"][0]["observation"]

    class _NullResultSanitizer(SQLSanitizer):
        def execute_safe_query(self, db_path, sql, params=None):
            return {"rows": [{"id": "n1", "name": "Organic Tea", "category": "beverages", "price": None}]}

    class _NullAgent:
        currency = None

    router_null = FastCascadedRouter(
        db_path=str(db_usd),
        sanitizer=_NullResultSanitizer(),
        react_agent=_NullAgent(),
        llm_runner=None,
        schema_prompt="",
        currency=None,
    )
    res_null = router_null.route_and_execute("Який найдешевший товар?")
    assert res_null["tier"] == ExecutionTier.TIER1_FAST_PATH
    assert "ціна не вказана" in res_null["answer"]
    assert "0.00" not in res_null["answer"]
    assert "грн" not in res_null["answer"]


def test_api_pareto_supports_category_filter_and_metadata(client):
    res_all = client.post("/api/pareto", json={
        "budget_max": 50000.0,
        "min_performance": 60.0,
        "domain": "pc_hardware",
    })
    assert res_all.status_code == 200
    d_all = res_all.json()
    assert "gpu" in d_all["categories"]
    assert d_all["selected_category"] is None
    assert all("category" in c for c in d_all["all_candidates"])

    res_gpu = client.post("/api/pareto", json={
        "budget_max": 50000.0,
        "min_performance": 60.0,
        "domain": "pc_hardware",
        "category": "gpu",
    })
    assert res_gpu.status_code == 200
    d_gpu = res_gpu.json()
    assert d_gpu["selected_category"] == "gpu"
    assert len(d_gpu["all_candidates"]) > 0
    assert all(c["category"] == "gpu" for c in d_gpu["all_candidates"])
    assert len(d_gpu["all_candidates"]) < len(d_all["all_candidates"])


def test_demo_auto_parts_and_pharmacy_assortments_and_pareto(client):
    for dom in ("auto_parts", "pharmacy"):
        res_cat = client.get(f"/api/catalog?domain={dom}&limit=50")
        assert res_cat.status_code == 200
        cat = res_cat.json()
        assert cat["total"] >= 30
        res_par = client.post("/api/pareto", json={
            "budget_max": 5000.0,
            "min_performance": 80.0,
            "domain": dom,
        })
        assert res_par.status_code == 200
        par = res_par.json()
        assert len(par["all_candidates"]) >= 30
        assert par["performance_range"]["max"] > par["performance_range"]["min"]


def test_live_arm_missing_api_key_returns_typed_error_not_500(monkeypatch):
    """Greeting/chat without OPENROUTER_API_KEY must return typed API_KEY_MISSING, never HTTP 500."""
    monkeypatch.setattr(ctx.runner, "api_key", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    safe_client = TestClient(app, raise_server_exceptions=False)
    res = safe_client.post("/api/chat", json={"query": "Привіт", "domain": "pharmacy"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "error"
    assert data["termination_reason"] == "API_KEY_MISSING"
    assert data["model"] is None
    assert "OpenRouter" in data["final_answer"]


def test_live_arm_byok_header_isolates_runner_without_mutating_ctx(monkeypatch):
    """Caller X-OpenRouter-Key uses isolated per-request runner without leaking or mutating ctx.runner."""
    import json as _json
    import openrouter_client

    monkeypatch.setattr(ctx.runner, "api_key", None)
    before_model = ctx.runner.model_name
    seen_auth = []
    seen_models = []

    class _MockPostResp:
        status_code = 200
        text = "{}"
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": _json.dumps({
                            "thought": "Привітання",
                            "final_answer": "Вітаю! Чим допомогти у каталозі?",
                            "recommended_items": [],
                        })
                    }
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            }

    def _mock_post(url, headers=None, json=None, timeout=45):
        seen_auth.append((headers or {}).get("Authorization"))
        seen_models.append((json or {}).get("model"))
        return _MockPostResp()

    monkeypatch.setattr(openrouter_client.requests, "post", _mock_post)
    monkeypatch.setenv("PYBUDDY_ADMIN_TOKEN", "adm-secret")
    safe_client = TestClient(app, raise_server_exceptions=False)
    byok_key = "sk-or-byok-session-only-xyz"
    res = safe_client.post(
        "/api/chat",
        json={"query": "Привіт", "domain": "pharmacy", "model": "custom/byok-model"},
        headers={"X-OpenRouter-Key": byok_key},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["model"] == "custom/byok-model"
    assert seen_auth == [f"Bearer {byok_key}"]
    assert seen_models == ["custom/byok-model"]
    assert ctx.runner.api_key is None
    assert ctx.runner.model_name == before_model
    tel = safe_client.get("/api/telemetry", headers={"X-Admin-Token": "adm-secret"}).text
    assert byok_key not in tel

