"""Catalogue, taxonomy and ontology navigation routes."""

from typing import List, Optional
import os
from domain_catalog import DomainCatalogRegistry
from ontograph import OntoGraph
from sanitizer import escape_like, LIKE_ESCAPE_CLAUSE
from server_config import MAX_CATALOG_PAGE
from server_context import ctx
from fastapi import APIRouter

router = APIRouter()


@router.get("/api/catalog")
def get_catalog(domain: Optional[str] = None, search: Optional[str] = None, limit: int = 40, offset: int = 0):
    target_domain = domain or ctx.active_domain
    bundle = ctx.get_domain_bundle(target_domain)
    hg = bundle.hypergraph
    limit = max(1, min(int(limit), MAX_CATALOG_PAGE))
    offset = max(0, int(offset))

    # If the domain is backed by a real SQLite database, query the actual
    # database directly (Zero-Limit / C7 fix). F10: the query is built from
    # the canonical source projection, not from a hard-coded `products` shape.
    if bundle.db_path and os.path.exists(bundle.db_path):
        import sqlite3
        from source_projection import build_source_projection, quote_ident
        try:
            proj = build_source_projection(bundle.db_path)
        except Exception:
            proj = None
        if proj is not None and proj.entity_table is not None:
            ident = proj.resolve("identity")
            name_res = proj.resolve("name")
            cat_res = proj.resolve("category")
            price_res = proj.resolve("price")
            if ident.resolved:
                q = quote_ident
                t = q(proj.entity_table)
                name_sel = f"{q(name_res.column)}" if name_res.column else f"{q(ident.column)}"
                cat_sel = f"{q(cat_res.column)}" if cat_res.column else "NULL"
                price_sel = f"{q(price_res.column)}" if price_res.column else "NULL"

                where_clauses = []
                params: List[object] = []
                search_backend = "none"
                fts = None
                if search:
                    # FTS5 trigram index (T5): exact substring semantics
                    # at index speed; LIKE remains the honest fallback for
                    # short terms or when the index cannot be built.
                    raw = search.strip()
                    from catalog_index import CatalogSearchIndex, MIN_TERM_LENGTH
                    fts = getattr(bundle, "search_index", None)
                    if fts is None:
                        fts = CatalogSearchIndex()
                        bundle.search_index = fts
                    text_cols = [c for c in (name_res.column, cat_res.column)
                                 if c]
                    if (len(raw) >= MIN_TERM_LENGTH and text_cols
                            and fts.ensure_built(
                                bundle.db_path, proj.entity_table,
                                ident.column, text_cols) == "ready"
                            and fts.usable_for(raw)):
                        search_backend = "fts5"
                    else:
                        search_backend = "like"
                if search and search_backend != "fts5":
                    # LIKE metacharacters escaped so `%`/`_` are literal
                    # (T4). SQLite LIKE folds ASCII case natively but
                    # LOWER() cannot fold Cyrillic, so the column stays
                    # raw and the term is bound in common case variants
                    # instead of lowercasing the column.
                    pats = sorted({
                        f"%{escape_like(v)}%"
                        for v in {raw, raw.lower(), raw.upper(),
                                  raw.capitalize()}
                    })
                    cols = [name_sel] + ([cat_sel] if cat_res.column else [])
                    parts = [
                        "(" + " OR ".join(
                            f"{c} LIKE ?{LIKE_ESCAPE_CLAUSE}" for _ in pats
                        ) + ")"
                        for c in cols
                    ]
                    where_clauses.append("(" + " OR ".join(parts) + ")")
                    for _ in cols:
                        params.extend(pats)

                where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

                conn = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(bundle.db_path).replace("\\", "/"), uri=True)
                cursor = conn.cursor()
                if search_backend == "fts5":
                    total_count = fts.total_matches(raw)
                    eids = fts.page_ids(raw, limit, offset)
                    if eids:
                        ph = ",".join("?" * len(eids))
                        cursor.execute(
                            f"SELECT {q(ident.column)}, {name_sel}, "
                            f"{cat_sel}, {price_sel} FROM {t} "
                            f"WHERE {q(ident.column)} IN ({ph})", eids)
                        rows = cursor.fetchall()
                        order = {eid: i for i, eid in enumerate(eids)}
                        rows.sort(key=lambda r: order.get(str(r[0]), 0))
                    else:
                        rows = []
                else:
                    cursor.execute(f"SELECT COUNT(*) FROM {t} {where_sql}", params)
                    total_count = cursor.fetchone()[0]

                    query_sql = (f"SELECT {q(ident.column)}, {name_sel}, {cat_sel}, {price_sel} "
                                 f"FROM {t} {where_sql} LIMIT ? OFFSET ?")
                    cursor.execute(query_sql, params + [limit, offset])
                    rows = cursor.fetchall()
                conn.close()

                # Page products are hydrated into a LOCAL map, not the
                # shared hypergraph: a GET is a read and must not grow
                # hg.nodes unboundedly as the user pages (R6 review).
                page_ids = [str(r[0]) for r in rows]
                page_nodes = dict(hg.nodes)
                missing = [pid for pid in page_ids if pid not in page_nodes]
                if missing:
                    page_nodes.update(
                        DomainCatalogRegistry.load_products_dynamically(
                            bundle.db_path, missing))

                products = []
                for r in rows:
                    pid, name, cat, price = r
                    pid = str(pid)
                    attrs = (dict(page_nodes[pid].attributes)
                             if pid in page_nodes else {})
                    products.append({
                        "id": pid,
                        "name": name if name is not None else pid,
                        "category": cat or "general",
                        # F10: an absent price stays absent (null), never 0.0.
                        "price": float(price) if price is not None else None,
                        "attributes": attrs
                    })

                # EBR-1: a zero-result lexical query may still be
                # answerable through EAV axes -- items whose attribute
                # VALUES mention the term, plus the axis bridge that
                # carried it. Relaxation is reported as typed evidence,
                # never merged into `total` (a relaxed hit is not a
                # direct hit).
                relaxation = None
                if search and total_count == 0 and offset == 0:
                    try:
                        from associative import relax_search
                        # Dense channel rides along only when an index
                        # was explicitly built for this source -- same
                        # opt-in contract as the conductor.
                        vec = None
                        idx = getattr(bundle, "vector_index", None)
                        emb = getattr(bundle, "vector_embedder", None)
                        if idx is not None and emb is not None \
                                and idx.stats.get("status") == "ready":
                            vec = (idx, emb)
                        rel = relax_search(
                            bundle.db_path, proj, raw, vector=vec,
                            channel_index=getattr(
                                bundle, "channel_index", None))
                        if rel is not None:
                            relaxation = rel.to_dict()
                    except Exception:
                        # Associative recall is additive; its failure must
                        # not take down the catalogue read.
                        relaxation = None

                return {
                    "domain": target_domain,
                    "currency": bundle.currency,
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "products": products,
                    "relaxation": relaxation,
                    # Which search path served this request -- the FTS
                    # index or the LIKE fallback -- is part of the
                    # contract, not hidden telemetry.
                    "search_backend": search_backend,
                    "search_index": ({
                        "status": fts.status,
                        "indexed_rows": fts.indexed_rows,
                        "built_seconds": fts.built_seconds,
                        "fail_reason": fts.fail_reason,
                    } if search and fts is not None else None),
                }

    # Fallback for in-memory or purely synthetic catalogs (e.g. auto_parts)
    products = [
        {"id": p.id, "name": p.name, "category": p.category, "price": p.price,
         "attributes": dict(p.attributes)}
        for p in hg.nodes.values()
    ]
    if search:
        s_l = search.lower().strip()
        products = [p for p in products if s_l in p["name"].lower() or s_l in p["category"].lower()]

    total_count = len(products)
    paged_slice = products[offset:offset + limit]
    return {
        "domain": target_domain,
        "currency": bundle.currency,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "products": paged_slice
    }

@router.get("/api/taxonomy")
@router.get("/api/ontograph/tree")
def get_taxonomy_tree(domain: Optional[str] = None):
    """
    Returns the Dynamic Domain Taxonomy Tree (OntoGraph) for the active catalog.
    Provides hierarchical category mindmap navigation.
    """
    dom = domain or ctx.active_domain
    bundle = ctx.get_domain_bundle(dom)
    if not bundle.ontograph and bundle.db_path and os.path.exists(bundle.db_path):
        bundle.ontograph = OntoGraph(bundle.db_path, openrouter_runner=ctx.runner)
    return {
        "domain": dom,
        "tree": bundle.ontograph.to_tree_dict() if bundle.ontograph else {}
    }

@router.get("/api/ontograph/navigate")
def navigate_ontograph(query: str, domain: Optional[str] = None):
    """Drills down to specific category or brand node in the OntoGraph."""
    dom = domain or ctx.active_domain
    bundle = ctx.get_domain_bundle(dom)
    if not bundle.ontograph and bundle.db_path and os.path.exists(bundle.db_path):
        bundle.ontograph = OntoGraph(bundle.db_path, openrouter_runner=ctx.runner)
    if not bundle.ontograph:
        return {"found": False, "message": "Таксономія не знайдена"}
    return bundle.ontograph.navigate(query)
