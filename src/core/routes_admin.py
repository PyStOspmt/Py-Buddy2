"""Administrative domain build routes: conductor, reanalyze, embedding, vector index."""

from fastapi import APIRouter, HTTPException, Request
from server_context import ctx
from server_schemas import ConstraintStatusRequest
from server_security import _require_admin
from vector_index import EMBEDDING_MODELS

router = APIRouter()


@router.post("/api/domains/{domain_id}/conductor")
def run_domain_conductor(domain_id: str, request: Request,
                         max_rounds: int = 8):
    """Run the conductor loop on an already-registered source.

    For domains registered without `llm_assist` (e.g. the demo registry)
    this is the way to invoke LLM-assisted schema discovery afterwards:
    the model proposes hypotheses, deterministic tools verify them, and
    accepted inferred links immediately become binding channels because
    the projection object is shared through the cache. The full report
    and the trace events land in the cached projection -- /inspection
    and the build canvas pick them up on the next read.
    """
    _require_admin(request)
    bundle = ctx.get_domain_bundle(domain_id)
    if not bundle.db_path:
        raise HTTPException(
            status_code=400,
            detail=f"Domain {domain_id!r} has no SQLite source.")
    from conductor import run_conductor
    from source_projection import build_source_projection
    proj = build_source_projection(bundle.db_path)
    report = run_conductor(bundle.db_path, proj, ctx.runner,
                           max_rounds=max(1, min(int(max_rounds), 12)))
    bundle.llm_mapping_report = {"conductor": True, **report}
    return bundle.llm_mapping_report


@router.post("/api/domains/{domain_id}/reanalyze")
def reanalyze_domain(domain_id: str, request: Request,
                     llm_assist: bool = False,
                     vector: bool = False):
    """Explicit "re-run the analysis with MY parameters".

    The auto-projection that ran at registration stays untouched; this
    call invalidates the cached projection and rebuilds it, then runs
    the optional stages the operator asked for (conductor, vector
    index) so the canvas/inspection reflects exactly the analysis the
    user selected -- not the defaults from boot.
    """
    _require_admin(request)
    bundle = ctx.get_domain_bundle(domain_id)
    if not bundle.db_path:
        raise HTTPException(
            status_code=400,
            detail=f"Domain {domain_id!r} has no SQLite source.")
    # Job-mode re-analyze: invalidates the projection cache, rebuilds
    # the engine, then runs the operator-selected optional stages --
    # streamed through the same init-job trace the UI already polls.
    job = ctx.start_init_job(
        domain_id, reanalyze={"llm_assist": bool(llm_assist),
                              "vector": bool(vector)})
    return {"domain": domain_id, "job": job,
            "message": "Re-analyze queued -- poll /api/init_jobs."}


@router.get("/api/settings/embedding")
def get_embedding_settings():
    """Which dense model the vector channel would use. Public so the UI
    can render the selector; switching stays admin-gated."""
    return {"model": ctx.embedding_model, "choices": EMBEDDING_MODELS}


@router.post("/api/settings/embedding")
def set_embedding_model(request: Request, model: str):
    _require_admin(request)
    if model not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown embedding model {model!r}; "
                   f"choices: {EMBEDDING_MODELS}")
    ctx.embedding_model = model
    return {"model": model}


@router.post("/api/domains/{domain_id}/vector_index")
def build_domain_vector_index(domain_id: str, request: Request,
                              max_cells: int = 1500):
    """Build the dense index for one source -- the opt-in that turns on
    the vector channel for relax_search on this domain. Uses the same
    OpenRouter key as the chat runner and the operator-selected
    embedding model. Honest stats: cells indexed, sampled flag, elapsed
    seconds; failures degrade to a recorded reason."""
    _require_admin(request)
    bundle = ctx.get_domain_bundle(domain_id)
    if not bundle.db_path:
        raise HTTPException(
            status_code=400,
            detail=f"Domain {domain_id!r} has no SQLite source.")
    key = getattr(ctx.runner, "api_key", None)
    if not key:
        raise HTTPException(
            status_code=400,
            detail="No OpenRouter key configured -- the vector channel "
                   "stays unavailable, lexical channels keep working.")

    from source_projection import build_source_projection
    from vector_index import OpenRouterEmbedder, get_vector_index
    proj = build_source_projection(bundle.db_path)
    emb = OpenRouterEmbedder(key, ctx.embedding_model)
    idx = get_vector_index(bundle.db_path, proj, emb,
                           ctx.embedding_model,
                           max(100, min(int(max_cells), 5000)))
    bundle.vector_embedder = emb
    bundle.vector_index = idx if idx.stats.get("status") == "ready" \
        else None
    return {"domain_id": domain_id, **idx.stats}


@router.post("/api/domains/{domain_id}/constraints/status")
def set_domain_constraint_status(
    domain_id: str,
    body: ConstraintStatusRequest,
    request: Request,
):
    """Operator confirmation / demotion of an induced rule (candidate <-> verified)."""
    _require_admin(request)
    bundle = ctx.get_domain_bundle(domain_id)
    if bundle.hypergraph is None:
        raise HTTPException(
            status_code=400,
            detail=f"Domain {domain_id!r} has no hypergraph.",
        )
    updated = bundle.hypergraph.set_constraint_status(body.rule_id, body.status)
    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f"Constraint {body.rule_id!r} not found.",
        )
    if body.status == "verified":
        bundle.verified_rule_ids.add(body.rule_id)
    else:
        bundle.verified_rule_ids.discard(body.rule_id)
    if getattr(bundle, "registered_via", "code") == "runtime":
        ctx.persist_runtime_registration(bundle)
    return {
        "domain_id": domain_id,
        "rule_id": body.rule_id,
        "status": updated.status,
        "type": type(updated).__name__,
    }
