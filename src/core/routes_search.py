"""Public semantic-search routes over the per-domain SemanticIndex.

Read-only surface: search + status are public; refresh is admin-gated.
No endpoint ever substitutes LIKE search or another source's data.
"""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from server_context import ctx
from server_security import _require_admin

router = APIRouter()


class RefreshRequest(BaseModel):
    rebuild: bool = False


def _bundle_or_404(domain_id: str):
    bundle = ctx.domains.get(domain_id)
    if bundle is None:
        raise HTTPException(status_code=404,
                            detail=f"Unknown domain '{domain_id}'.")
    return bundle


def _semantic(bundle):
    """Lazily attach the lifecycle handle for degraded bundles whose
    engine init never ran -- the handle itself reports `unavailable`
    honestly when the source cannot be indexed."""
    if bundle.semantic is None:
        from semantic_service import SemanticIndex
        bundle.semantic = SemanticIndex(
            bundle.domain_id, bundle.db_path,
            embedder_factory=ctx.embedder_factory)
    return bundle.semantic


@router.get("/api/domains/{domain_id}/search")
def domain_search(domain_id: str,
                  q: str = Query(min_length=1, max_length=500),
                  k: int = Query(default=10, ge=1, le=50),
                  mode: Literal["dense", "lexical",
                                "hybrid"] = "hybrid",
                  hyde: bool = False):
    bundle = _bundle_or_404(domain_id)
    return _semantic(bundle).search(q, k=k, mode=mode, hyde=hyde)


@router.get("/api/domains/{domain_id}/semantic")
def domain_semantic(domain_id: str):
    bundle = _bundle_or_404(domain_id)
    return _semantic(bundle).status()


@router.post("/api/domains/{domain_id}/refresh", status_code=202)
def refresh_domain(domain_id: str, request: Request,
                   body: RefreshRequest | None = None):
    _require_admin(request)
    bundle = _bundle_or_404(domain_id)
    res = _semantic(bundle).start_sync(
        reason="manual_refresh",
        rebuild=bool(body and body.rebuild))
    if not res["started"]:
        raise HTTPException(status_code=409,
                            detail="semantic sync already running")
    return bundle.semantic.status()
