"""
Py-Buddy: High-Performance FastAPI Backend Server (Modular Route Composition)
Provides REST endpoints for the Interactive Web/Chat Demo, Virtual Hypergraph visualizer,
Pareto Frontier analyzer, Fast Cascaded Router (<50ms), and SQLite Session Persistence.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
# This module now lives in src/core/. Sibling core modules resolve via this
# directory; the `gateways` package resolves via the parent (src/).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routes_admin import build_domain_vector_index, get_embedding_settings, reanalyze_domain, run_domain_conductor, set_embedding_model
from routes_analysis import compute_pareto, validate_bundle
from routes_catalog import get_catalog, get_taxonomy_tree, navigate_ontograph
from routes_chat import _resolve_chat_domain, chat_consult
from routes_ingest import _tabular_to_sqlite, upload_domain_source
from routes_search import domain_search, domain_semantic, refresh_domain
from routes_lifecycle import cancel_init_job, deregister_domain, init_domain_async, list_init_jobs, register_domain, switch_domain
from routes_status import get_domain_attestation, get_domain_inspection, get_domains, get_models, get_system_status, get_telemetry, set_model
from server_config import AUTO_PARTS_DB_PATH, CURATED_FALLBACK_MODELS, DB_PATH, MARKETPLACE_DB_PATH, MAX_BUNDLE_ITEMS, MAX_CATALOG_PAGE, MAX_VIZ_LINK_NODES, MODELS_CACHE_PATH, PROJECT_ROOT, SESSIONS_DB, WEB_DIR, env_setting, load_openrouter_models
from server_context import DomainBundle, SystemContext, _InitCancelled, ctx
from server_resolution import DOMAINS, DOMAIN_NAMES, _COMPAT_CUES, _DomainRegistryProxy, __getattr__, _is_compatibility_question, _resolve_from_hypergraph, _resolve_named_products, hypergraph_for
from server_schemas import BundleValidateRequest, ChatQueryRequest, InitDomainRequest, ParetoRequest, RegisterDomainRequest, SetModelRequest, SwitchDomainRequest
from server_security import _is_admin_request, _require_admin
from routes_status import router as routes_status_router
from routes_admin import router as routes_admin_router
from routes_catalog import router as routes_catalog_router
from routes_lifecycle import router as routes_lifecycle_router
from routes_ingest import router as routes_ingest_router
from routes_chat import router as routes_chat_router
from routes_analysis import router as routes_analysis_router
from routes_search import router as routes_search_router
from domain_catalog import DomainCatalogRegistry

__all__ = [
    "AUTO_PARTS_DB_PATH",
    "BundleValidateRequest",
    "CURATED_FALLBACK_MODELS",
    "ChatQueryRequest",
    "DB_PATH",
    "DOMAINS",
    "DOMAIN_NAMES",
    "DomainBundle",
    "DomainCatalogRegistry",
    "InitDomainRequest",
    "MARKETPLACE_DB_PATH",
    "MAX_BUNDLE_ITEMS",
    "MAX_CATALOG_PAGE",
    "MAX_VIZ_LINK_NODES",
    "MODELS_CACHE_PATH",
    "PROJECT_ROOT",
    "ParetoRequest",
    "RegisterDomainRequest",
    "SESSIONS_DB",
    "SetModelRequest",
    "SwitchDomainRequest",
    "SystemContext",
    "WEB_DIR",
    "_ALLOWED_ORIGINS",
    "_COMPAT_CUES",
    "_DomainRegistryProxy",
    "_InitCancelled",
    "__getattr__",
    "_is_admin_request",
    "_is_compatibility_question",
    "_require_admin",
    "_resolve_chat_domain",
    "_resolve_from_hypergraph",
    "_resolve_named_products",
    "_tabular_to_sqlite",
    "app",
    "build_domain_vector_index",
    "cancel_init_job",
    "chat_consult",
    "compute_pareto",
    "ctx",
    "deregister_domain",
    "domain_search",
    "domain_semantic",
    "get_catalog",
    "get_domain_attestation",
    "get_domain_inspection",
    "get_domains",
    "get_embedding_settings",
    "get_models",
    "get_system_status",
    "get_taxonomy_tree",
    "get_telemetry",
    "hypergraph_for",
    "init_domain_async",
    "list_init_jobs",
    "load_openrouter_models",
    "navigate_ontograph",
    "reanalyze_domain",
    "refresh_domain",
    "register_domain",
    "run_domain_conductor",
    "serve_index",
    "set_embedding_model",
    "set_model",
    "switch_domain",
    "upload_domain_source",
    "validate_bundle",
]


app = FastAPI(
    title="Py-Buddy Interactive Demo",
    version="2.1.0",
    description="Active-Inference Neuro-Symbolic Agent over Knowledge Hypergraphs"
)

app.include_router(routes_status_router)
app.include_router(routes_admin_router)
app.include_router(routes_catalog_router)
app.include_router(routes_lifecycle_router)
app.include_router(routes_ingest_router)
app.include_router(routes_chat_router)
app.include_router(routes_analysis_router)
app.include_router(routes_search_router)

# F30: wildcard origins combined with credentials let any web page call this
# API with cookies. Origins must be allow-listed by deployment; credentials
# stay off for the public demo surface.
_ALLOWED_ORIGINS = [o.strip() for o in env_setting("PYBUDDY_ALLOWED_ORIGINS", "AINSA_ALLOWED_ORIGINS").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def disable_client_cache(request, call_next):
    """Force the browser to revalidate the UI shell and static assets on every
    load. Without this, Chromium heuristically caches legacy responses (e.g. a
    pre-rebuild style.css) and serves them without revalidation, masking a fresh
    deploy behind a stale stylesheet."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(content={"message": "Py-Buddy API is running."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
