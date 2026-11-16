"""
Retrieval Fabric: dense + lexical + hybrid (RRF) search.

Pure function of (store, embedder, query): no freshness logic here --
the sync layer owns freshness. Hybrid fusion is Reciprocal Rank Fusion
with k=60 over the channels that returned each id. Empty query or empty
index yields empty hits plus an explanatory note -- hits are never
fabricated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from vector_store import VectorStore

logger = logging.getLogger(__name__)

_RRF_K = 60
_SNIPPET_CHARS = 240

#: Lexical channel weight of the PRODUCT hybrid mode (SemanticIndex).
#: Chosen on the dev split (docs/18 scope-change 2): lexical noise hurts
#: requests that share no words with the record. `search()` itself keeps
#: the registered 1.0/1.0 default so the registered system is unchanged.
PRODUCT_LEXICAL_WEIGHT = 0.3


@dataclass(frozen=True)
class Hit:
    entity_id: str
    score: float
    dense_rank: int | None
    dense_sim: float | None
    lexical_rank: int | None
    lexical_score: float | None
    snippet: str
    provenance: str


@dataclass(frozen=True)
class SearchResult:
    query: str
    mode: str
    hits: list[Hit]
    top1_z: float | None
    margin: float | None
    indexed: int
    generation: int
    notes: list[str] = field(default_factory=list)
    #: dense top-1 cosine (the dense channel's own top hit, NOT the
    #: fused score); None when the dense channel did not run.
    #: top1_z/margin are kept for the record but are NOT usable
    #: abstention signals; top1_sim separated out-of-domain requests
    #: on the dev split only -- it has no calibrated threshold and
    #: must never become a hard cut-off.
    top1_sim: float | None = None


def _fuse(dense_ids, dense_scores, lex, k,
          dense_weight: float = 1.0, lexical_weight: float = 1.0):
    """Weighted RRF over whichever channels produced ids."""
    fused = {}
    for rank, (eid, sim) in enumerate(zip(dense_ids, dense_scores), 1):
        e = fused.setdefault(eid, {"score": 0.0})
        e["dense_rank"], e["dense_sim"] = rank, sim
        e["score"] += dense_weight / (_RRF_K + rank)
    for rank, (eid, s) in enumerate(lex, 1):
        e = fused.setdefault(eid, {"score": 0.0})
        e["lexical_rank"], e["lexical_score"] = rank, s
        e["score"] += lexical_weight / (_RRF_K + rank)
    order = sorted(fused.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
    return order[:k]


def _hits(store: VectorStore, fused) -> list[Hit]:
    """Hydrate hit texts in ONE get_many query (no vector reads)."""
    texts = store.get_many([eid for eid, _m in fused])
    out = []
    for eid, e in fused:
        rec = texts.get(eid)
        out.append(Hit(
            entity_id=eid, score=e["score"],
            dense_rank=e.get("dense_rank"), dense_sim=e.get("dense_sim"),
            lexical_rank=e.get("lexical_rank"),
            lexical_score=e.get("lexical_score"),
            snippet=(rec.text[:_SNIPPET_CHARS] if rec else ""),
            provenance=(rec.provenance if rec else "")))
    return out


def search(store: VectorStore, embedder, query: str, *, k: int = 10,
           mode: str = "hybrid", depth: int | None = None,
           dense_weight: float = 1.0, lexical_weight: float = 1.0
           ) -> SearchResult:
    """Search the store. mode: "dense" | "lexical" | "hybrid".
    dense_weight/lexical_weight scale each channel's RRF share in
    hybrid mode (evaluation knob; 1.0/1.0 is the production default)."""
    if mode not in ("dense", "lexical", "hybrid"):
        raise ValueError(f"unknown mode {mode!r}")
    depth = depth if depth is not None else max(50, 5 * k)
    notes: list[str] = []
    # live_count is served from the cached snapshot bitmap -- a search
    # must not pay an O(N) count query per request.
    indexed, generation = store.live_count(), store.generation
    if not query or not query.strip():
        return SearchResult(query, mode, [], None, None, indexed,
                            generation, ["empty query"], None)
    if indexed == 0:
        return SearchResult(query, mode, [], None, None, indexed,
                            generation, ["index is empty"], None)

    dense_ids: list[str] = []
    dense_scores: list[float] = []
    dres = None
    if mode in ("dense", "hybrid"):
        qv = embedder.embed_queries([query])[0]
        dres = store.search_dense(np.asarray(qv, dtype=np.float32),
                                  depth)
        dense_ids, dense_scores = dres.ids, dres.scores
    lex = []
    if mode in ("lexical", "hybrid"):
        lex = store.search_lexical(query, depth)

    fused = _select(mode, dense_ids, dense_scores, lex, k,
                    dense_weight, lexical_weight)
    if not fused:
        notes.append("no matches")

    top1_z, margin = _outliers(dres, dense_scores, k)
    top1_sim = dense_scores[0] if dense_scores else None
    return SearchResult(query, mode, _hits(store, fused), top1_z,
                        margin, indexed, generation, notes, top1_sim)


def _select(mode: str, dense_ids, dense_scores, lex, k,
            dense_weight: float = 1.0, lexical_weight: float = 1.0):
    if mode == "dense":
        return [(e, {"dense_rank": i + 1, "dense_sim": s, "score": s})
                for i, (e, s)
                in enumerate(zip(dense_ids, dense_scores))][:k]
    if mode == "lexical":
        return [(e, {"lexical_rank": i + 1, "lexical_score": s,
                     "score": s})
                for i, (e, s) in enumerate(lex)][:k]
    return _fuse(dense_ids, dense_scores, lex, k,
                 dense_weight, lexical_weight)


def _outliers(dres, dense_scores, k):
    """top1_z and margin from the dense scan; None when undefined."""
    top1_z = margin = None
    if dres is None or not dense_scores:
        return top1_z, margin
    s1 = dense_scores[0]
    if not dres.score_std or dres.score_std <= 0:
        return top1_z, margin
    top1_z = (s1 - dres.score_mean) / dres.score_std
    if len(dense_scores) >= 2:
        margin = s1 - dense_scores[min(k, len(dense_scores)) - 1]
    return top1_z, margin
