"""Domain registration, deregistration, switching and init-job routes."""

from fastapi import HTTPException, Request
import os
import re
from react_agent import SessionState
from session_store import SHARED_SESSION_ID, resolve_session_id
from server_config import env_setting
from server_context import DomainBundle, ctx
from server_schemas import InitDomainRequest, RegisterDomainRequest, SwitchDomainRequest
from server_security import _require_admin
from fastapi import APIRouter

router = APIRouter()


@router.post("/api/register_domain")
def register_domain(req: RegisterDomainRequest, request: Request):
    """
    Zero Code Change Schema ingestion (ZCCS = 0).
    Dynamically registers any SQLite DB as an active authoritative domain.

    F29: administrative operation. The path must resolve inside the
    configured source root (PYBUDDY_SOURCE_ROOT / AINSA_SOURCE_ROOT, default: the project's data
    directory), must be a SQLite file, and source ids are immutable.
    """
    _require_admin(request)
    if not req.domain_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", req.domain_id):
        raise HTTPException(
            status_code=400,
            detail="domain_id must be 1-64 characters of [A-Za-z0-9_-].")
    if req.domain_id in ctx.domains:
        raise HTTPException(
            status_code=409,
            detail=f"Source id '{req.domain_id}' already exists; source ids are immutable.")
    root = env_setting("PYBUDDY_SOURCE_ROOT", "AINSA_SOURCE_ROOT") or os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
    root_real = os.path.realpath(root)
    real = os.path.realpath(os.path.abspath(req.db_path))
    if not real.startswith(root_real + os.sep):
        raise HTTPException(
            status_code=403,
            detail=f"Database path is outside the allowed source root ({root_real}).")
    if os.path.splitext(real)[1].lower() not in (".db", ".sqlite", ".sqlite3"):
        raise HTTPException(
            status_code=400,
            detail="Only SQLite database files (.db/.sqlite/.sqlite3) can be registered.")
    if not os.path.exists(real):
        raise HTTPException(
            status_code=400,
            detail=f"Database file does not exist: {real}"
        )
    bundle = DomainBundle(
        domain_id=req.domain_id,
        name=req.name or req.domain_id,
        db_path=real,
        hypergraph=None,
        description=req.description or f"Custom dynamic database at {real}"
    )
    bundle.registered_via = "runtime"
    bundle.llm_assist = bool(req.llm_assist)
    ctx.register_domain_bundle(bundle)
    ctx.persist_runtime_registration(bundle)
    if req.async_init:
        # F33: heavy builds can run as a background job with observable
        # status instead of blocking the registration response.
        job = ctx.start_init_job(req.domain_id)
        return {
            "status": "accepted",
            "domain_id": req.domain_id,
            "name": bundle.name,
            "db_path": real,
            "state": bundle.state.value,
            "init_job": job,
            "message": (f"Registered '{req.domain_id}'; engine build queued "
                        f"as job {job['job_id']} (poll /api/init_jobs).")
        }
    ctx.get_domain_bundle(req.domain_id)  # trigger lazy profile & engine build
    return {
        "status": "success",
        "domain_id": req.domain_id,
        "name": bundle.name,
        "db_path": real,
        "nodes_count": len(bundle.hypergraph.nodes),
        "state": bundle.state.value,
        "state_reason": bundle.state_reason or None,
        "message": f"Successfully registered and profiled dynamic domain '{req.domain_id}'."
    }


@router.delete("/api/domains/{domain_id}")
def deregister_domain(domain_id: str, request: Request):
    """Remove a runtime-registered source: in-memory entry plus its row
    in `sources.runtime.json`. Demo-registry sources cannot be removed
    here -- deleting them in memory would resurrect them from
    `sources.demo.json` on restart, so they must be edited there."""
    _require_admin(request)
    bundle = ctx.domains.get(domain_id)
    if bundle is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown domain '{domain_id}'.")
    if getattr(bundle, "registered_via", "runtime") != "runtime":
        raise HTTPException(
            status_code=409,
            detail=(f"'{domain_id}' is registered via the demo registry; "
                    "edit data/sources.demo.json to remove it."))
    del ctx.domains[domain_id]
    persisted = ctx.remove_runtime_registration(domain_id)
    return {
        "status": "removed",
        "domain_id": domain_id,
        "persisted_registration_removed": persisted,
    }

@router.post("/api/switch_domain")
def switch_domain(req: SwitchDomainRequest, request: Request):
    if req.domain not in ctx.domains:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain. Choose one of: {', '.join(ctx.domains.keys())}.",
        )
    # R6: session-scoped preference -- one client switching domains must
    # not redirect other sessions. Only the global fallback is admin-only.
    if req.session_id:
        if req.session_id == SHARED_SESSION_ID:
            # The well-known shared id: pinning it is a global change in
            # disguise, and any caller can name it. Require a server- or
            # client-minted unique id for preference writes.
            raise HTTPException(
                status_code=400,
                detail="The shared default session id cannot carry a domain "
                       "preference; use a unique session id.")
        sid = resolve_session_id(req.session_id, ctx.session_store)
        session = ctx.session_store.get_session(sid) or SessionState(
            session_id=sid, user_query="")
        session.domain = req.domain
        ctx.session_store.save_session(session)
        return {"message": f"Session '{sid}' domain set to {req.domain}",
                "domain": req.domain, "session_id": sid}
    # F30: switching the global default domain is administrative.
    _require_admin(request)
    ctx.active_domain = req.domain
    return {"message": f"Switched default domain to {req.domain}", "active_domain": ctx.active_domain}

@router.post("/api/init_domain", status_code=202)
def init_domain_async(req: InitDomainRequest, request: Request):
    """
    F33: queue a background engine build for a registered source instead
    of paying init latency inside an unrelated request. Administrative:
    building an engine is expensive and must not be an open trigger.
    Returns the job record; poll /api/init_jobs for status.
    """
    _require_admin(request)
    return ctx.start_init_job(req.domain)

@router.get("/api/init_jobs")
def list_init_jobs():
    """F33: init lifecycle is observable -- every job's status, stage,
    and error is queryable, including failures and cancellations."""
    return {"jobs": ctx.list_init_jobs()}

@router.post("/api/init_jobs/{job_id}/cancel")
def cancel_init_job(job_id: str, request: Request):
    """F33: cooperative cancel; the running stage finishes and the
    source returns to pending rather than half-initialised."""
    _require_admin(request)
    return ctx.cancel_init_job(job_id)
