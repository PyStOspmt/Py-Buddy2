"""Status, domain listing, inspection, attestation, model and telemetry routes."""

from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Request
from typing import Optional
import os
import sqlite3
from openrouter_client import fetch_openrouter_models
from server_config import env_setting
from server_context import ctx
from server_resolution import hypergraph_for
from server_schemas import SetModelRequest
from server_security import _is_admin_request, _require_admin

router = APIRouter()


def _entity_block(proj_dict):
    """Compact entity-table summary for the inspection response."""
    if not isinstance(proj_dict, dict) or "entity_table" not in proj_dict:
        return None
    return {
        "table": proj_dict.get("entity_table"),
        "ambiguous": proj_dict.get("entity_ambiguous"),
        "alternatives": proj_dict.get("entity_alternatives"),
        "candidates": [c.get("table") for c in
                       proj_dict.get("entity_candidates", [])],
    }


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@router.get("/api/status")
def get_system_status(domain: Optional[str] = None):
    target_domain = domain if (domain and domain in ctx.domains) else ctx.active_domain
    nodes_cnt = len(hypergraph_for(target_domain).nodes)
    # F08: the marketplace scale is measured when that domain is initialised,
    # never asserted from a constant.
    mp = ctx.domains.get("marketplace_rozetka")
    mp_nodes = len(mp.hypergraph.nodes) if (mp and mp.hypergraph) else None

    active_bundle = ctx.domains.get(target_domain)
    llm_ready = bool(getattr(ctx.runner, "api_key", None))
    return {
        "status": "online",
        "architecture": "Py-Buddy (Active-Inference Neuro-Symbolic)",
        "lineage": "Py-Buddy V2.0 Omnichannel Evolution",
        "active_domain": target_domain,
        "active_currency": active_bundle.currency if active_bundle else None,
        "llm_available": llm_ready,
        "model": getattr(ctx.runner, "model_name", None) if llm_ready else None,
        "routing": "Adaptive Fast Cascaded Router (<15ms Tier-0/1 + Deep ReAct)",
        "persistence": "SQLite WAL Session Store",
        "invariants": {
            "zero_hallucination_gate": "ACTIVE",
            "virtual_hypergraph": "ACTIVE",
            "fuzzy_pareto_relaxer": "ACTIVE",
            "sql_ast_sanitizer": "ACTIVE"
        },
        "nodes_count": nodes_cnt,
        "marketplace_scale": mp_nodes,
        # F29/F30: the UI can learn whether administrative operations exist.
        "admin_enabled": bool(env_setting("PYBUDDY_ADMIN_TOKEN", "AINSA_ADMIN_TOKEN")),
    }

@router.get("/api/domains")
def get_domains():
    result = []
    for d_id, bundle in ctx.domains.items():
        # Honest counts: `item_count` is the catalogue's row count, not the
        # seeded hypergraph's node count -- on lazy sources the graph holds
        # only a sample while the catalogue holds everything (found live:
        # 120 nodes vs the real row count misled the UI).
        item_count: Optional[int] = None
        if bundle.db_path and os.path.exists(bundle.db_path):
            from source_projection import build_source_projection, quote_ident
            try:
                proj = build_source_projection(bundle.db_path)
            except Exception:
                proj = None  # unprofileable source: honest null
            if proj is not None and proj.entity_table:
                conn = sqlite3.connect(
                    f"file:{os.path.abspath(bundle.db_path).replace(chr(92), '/')}?mode=ro",
                    uri=True)
                try:
                    item_count = conn.execute(
                        f"SELECT COUNT(*) FROM {quote_ident(proj.entity_table)}"
                    ).fetchone()[0]
                except sqlite3.Error:
                    item_count = None
                finally:
                    conn.close()
        result.append({
            "id": d_id,
            "name": bundle.name,
            "description": bundle.description,
            "item_count": item_count,
            "graph_nodes": len(bundle.hypergraph.nodes) if bundle.hypergraph else 0,
            "has_db": bool(bundle.db_path and os.path.exists(bundle.db_path)),
            "currency": bundle.currency,
            "state": bundle.state.value,
            "state_reason": bundle.state_reason or None,
            "fiduciary": (bundle.fiduciary or {}).get("status"),
            "induction": (getattr(bundle.hypergraph, "induction_report", None)
                          if bundle.hypergraph else None),
            "is_active": d_id == ctx.active_domain
        })
    return {
        "active_domain": ctx.active_domain,
        # T9: registrations skipped at restore (missing file, out of
        # root) are surfaced, not silently dropped.
        "skipped_runtime_sources": getattr(
            ctx, "skipped_runtime_sources", []),
        "domains": result
    }


@router.get("/api/domains/{domain_id}/inspection")
def get_domain_inspection(domain_id: str):
    """
    Read-only introspection of what the adaptation layer inferred for one
    registered source: role resolutions with their evidence tiers, EAV
    discoveries, induced contract candidates, taxonomy summary, the
    question bank, and the literal schema prompt the language model
    receives. The point of the endpoint is that the mapping is inspectable
    -- nothing in the adaptation is asserted off-screen.
    """
    bundle = ctx.get_domain_bundle(domain_id)  # triggers lazy init honestly

    proj_dict = None
    if bundle.db_path and os.path.exists(bundle.db_path):
        from source_projection import build_source_projection
        try:
            proj_dict = build_source_projection(bundle.db_path).to_dict()
        except Exception as exc:
            proj_dict = {"error": f"{type(exc).__name__}: {exc}"}

    tables = {}
    if bundle.snapshot is not None:
        for tname, tp in bundle.snapshot.tables.items():
            tables[tname] = {
                "row_count": tp.row_count,
                "columns": {
                    c: {"type": col.data_type,
                        "semantic_type": getattr(col, "semantic_type", None),
                        "distinct_count": col.distinct_count,
                        "sample_values": [
                            str(v)[:60] for v in
                            (col.distinct_sample_values or [])[:5]],
                        "fk_target": (f"{col.fk_target_table}."
                                      f"{col.fk_target_column}")
                                     if col.is_foreign_key else None}
                    for c, col in tp.columns.items()
                },
            }

    constraints = []
    if bundle.hypergraph is not None:
        for c in bundle.hypergraph.constraints:
            entry = {
                "rule_id": c.rule_id() if hasattr(c, "rule_id") else "",
                "type": type(c).__name__,
                "status": getattr(c, "status", None),
                "description": getattr(c, "description", None),
            }
            for k, v in vars(c).items():
                if k.startswith("_") or k in ("status", "description"):
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    entry[k] = v
                elif isinstance(v, (list, tuple, set)):
                    entry[k] = [str(x) for x in v][:20]
            constraints.append(entry)

    taxonomy = None
    if bundle.ontograph is not None and bundle.ontograph.root is not None:
        root = bundle.ontograph.root
        taxonomy = {
            "item_count": root.item_count,
            "top_categories": [
                {"name": c.name, "item_count": c.item_count,
                 "children": len(c.children)}
                for c in root.children[:15]
            ],
            "parameters": root.parameters,
        }

    question_bank = []
    ai = getattr(bundle.agent, "active_inference", None) if bundle.agent else None
    if ai is not None:
        for qd in getattr(ai, "question_bank", []):
            question_bank.append({
                "id": qd.id, "text": qd.text,
                "target_dimension": qd.target_dimension,
                "options": qd.options,
                "adapted": getattr(qd, "adapted", None),
            })

    fts = getattr(bundle, "search_index", None)
    return {
        "domain_id": domain_id,
        "name": bundle.name,
        "description": bundle.description,
        "currency": bundle.currency,
        "state": bundle.state.value,
        "state_reason": bundle.state_reason or None,
        "registered_via": getattr(bundle, "registered_via", "code"),
        "db_path": bundle.db_path,
        "source_version": list(bundle.source_version) if bundle.source_version else None,
        "projection": proj_dict,
        "entity": _entity_block(proj_dict),
        "semantic": (bundle.semantic.status()
                     if bundle.semantic is not None else None),
        "tables": tables,
        "vector": (getattr(bundle, "vector_index", None)
                   and bundle.vector_index.stats) or None,
        "hypergraph": {
            "seeded_nodes": len(bundle.hypergraph.nodes) if bundle.hypergraph else 0,
            "constraint_count": len(constraints),
            "constraints": constraints[:200],
            "constraints_truncated": len(constraints) > 200,
            "induction_report": getattr(
                bundle.hypergraph, "induction_report", None) if bundle.hypergraph else None,
        },
        "taxonomy": taxonomy,
        "question_bank": question_bank,
        # The exact prompt text the model receives -- the visible mapping.
        "schema_prompt": bundle.schema_prompt,
        "search_index": ({
            "status": fts.status,
            "indexed_rows": fts.indexed_rows,
            "built_seconds": fts.built_seconds,
            "fail_reason": fts.fail_reason,
        } if fts is not None else None),
        "fiduciary": bundle.fiduciary,
        "attestation": bundle.attestation,
        "llm_assist": bool(getattr(bundle, "llm_assist", False)),
        "llm_mapping": getattr(bundle, "llm_mapping_report", None),
    }


@router.get("/api/domains/{domain_id}/attestation")
def get_domain_attestation(domain_id: str):
    """
    P1.3: the machine-checkable attestation for a registered source.
    Includes the probe result, the source sha256 it was bound to, and the
    policy digest; a third party replays it via
    `attestation.verify_attestation(att, db_path)` -- the endpoint only
    serves the artefact, it does not vouch for it.
    """
    bundle = ctx.domains.get(domain_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="unknown domain")
    if bundle.attestation is None:
        return {
            "domain_id": domain_id,
            "attestation": None,
            "status": "UNVERIFIED",
            "reason": "attestation not issued yet (init pending or failed)",
        }
    return {
        "domain_id": domain_id,
        "attestation": bundle.attestation,
        "status": (bundle.attestation.get("result") or {}).get("status"),
    }


@router.get("/api/models")
def get_models(request: Request):
    """Return available OpenRouter models ONLY when a valid key is configured
    (server env/.env or per-caller X-OpenRouter-Key). Never lists synthetic/offline
    fallback models as live when no API key exists.
    """
    byok_key = (request.headers.get("X-OpenRouter-Key") or "").strip()
    server_key = getattr(ctx.runner, "api_key", None) or ""
    effective_key = byok_key or server_key
    if not effective_key:
        return {
            "active_model": None,
            "key_configured": False,
            "key_source": "none",
            "source": "none",
            "models_count": 0,
            "models": [],
            "error": None,
        }
    models, err = fetch_openrouter_models(effective_key)
    return {
        "active_model": getattr(ctx.runner, "model_name", None),
        "key_configured": True,
        "key_source": "client" if byok_key else "server",
        "source": "openrouter_live" if not err else "error",
        "models_count": len(models),
        "models": models,
        "error": err,
    }

@router.post("/api/set_model")
def set_model(req: SetModelRequest, request: Request):
    # F30: switching the global model is an administrative operation.
    _require_admin(request)
    model_id = req.model.strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="Model name cannot be empty")
    ctx.runner.model_name = model_id
    ctx.runner.last_model = None
    if hasattr(ctx.runner, "fallback_models"):
        ctx.runner.fallback_models = [m for m in ctx.runner.fallback_models if m != model_id]
    return {
        "status": "success",
        "active_model": ctx.runner.model_name,
        "message": f"Active model set to {model_id}"
    }

@router.get("/api/telemetry")
def get_telemetry(request: Request):
    recent = ctx.session_store.get_recent_sessions(limit=5)
    # F30: session transcripts are not public. Anonymous telemetry carries
    # aggregates only; the full last session needs the admin token.
    is_admin = _is_admin_request(request)
    return {
        "status": "online",
        "active_domain": ctx.active_domain,
        "active_invariants": {
            "zero_hallucination_gate": {"status": "ACTIVE", "desc": "100% перевірка заземлення фактів у БД"},
            "virtual_hypergraph": {"status": "ACTIVE", "desc": "N-арна верифікація 4 універсальних мета-обмежень"},
            "sql_ast_sanitizer": {"status": "ACTIVE", "desc": "AST ReadOnly валідація без ін'єкцій"},
            "fast_cascaded_router": {"status": "ACTIVE", "desc": "Адаптивний Tier-0/1 шорткат (<15мс) + ReAct"},
            "session_persistence": {"status": "ACTIVE", "desc": "SQLite WAL Session Store"}
        },
        "recent_sessions_count": len(recent),
        "last_session": asdict(recent[0]) if (recent and is_admin) else None,
        "last_session_available": bool(recent),
    }
