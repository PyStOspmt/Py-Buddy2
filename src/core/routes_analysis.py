"""Bundle validation and Pareto analysis routes."""

from fastapi import HTTPException
from typing import Dict, Optional
import os
import sqlite3
from domain_catalog import DomainCatalogRegistry
from pareto_relaxer import FuzzyParetoRelaxer, CandidateSolution
from server_config import MAX_BUNDLE_ITEMS, MAX_VIZ_LINK_NODES
from server_context import ctx
from server_resolution import hypergraph_for
from server_schemas import BundleValidateRequest, ParetoRequest
from fastapi import APIRouter

router = APIRouter()


@router.post("/api/validate_bundle")
def validate_bundle(req: BundleValidateRequest):
    domain = req.domain or ctx.active_domain
    hg = hypergraph_for(domain)              # 400 on an unknown domain
    bundle = ctx.get_domain_bundle(domain)
    if len(req.product_ids) > MAX_BUNDLE_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Bundle validation is bounded to {MAX_BUNDLE_ITEMS} items; "
                   f"got {len(req.product_ids)}.")
    # Dynamic on-demand projection from domain database -- bulk hydration,
    # one query for the whole bundle rather than N+1 (F34).
    if bundle.db_path and os.path.exists(bundle.db_path):
        missing = [pid for pid in req.product_ids if pid not in hg.nodes]
        if missing:
            for node in DomainCatalogRegistry.load_products_dynamically(
                    bundle.db_path, missing).values():
                hg.add_product(node)

    product_set = set(req.product_ids)
    # F15: the honest verdict distinguishes verified compatibility from the
    # retail basket default and from unresolved items. `is_compatible` must
    # DERIVE from the verdict -- the legacy is_compatible() has permissive
    # basket semantics that answered True for an "unknown" verdict (R6 live
    # finding): "unknown" is None, not a silent pass.
    verdict_info = hg.compatibility_verdict(product_set)
    verdict = verdict_info["verdict"]
    is_compatible = {"compatible": True,
                     "incompatible": False,
                     "candidate_conflict": False}.get(verdict)
    active_nodes = [hg.nodes[pid] for pid in product_set if pid in hg.nodes]

    graph_nodes = [{"id": n.id, "name": n.name, "category": n.category,
                    "price": n.price, "attributes": dict(n.attributes)}
                   for n in active_nodes]
    # F34: pairwise link evaluation is O(n^2) -- bounded and honestly
    # reported, never silently partial.
    viz_pids = sorted(product_set)[:MAX_VIZ_LINK_NODES]
    graph_links_truncated = len(product_set) > len(viz_pids)
    graph_links = []
    for i in range(len(viz_pids)):
        for j in range(i + 1, len(viz_pids)):
            p1, p2 = viz_pids[i], viz_pids[j]
            # F15 consistency: the edge label must derive from the honest
            # verdict, not the permissive basket check -- "no rule fired"
            # is "немає даних", not "Сумісно".
            pair_verdict = hg.compatibility_verdict({p1, p2})["verdict"]
            pair_conflicts = hg.find_conflicts({p1, p2})
            if pair_verdict == "compatible":
                status, label = "valid", "Сумісно"
            elif pair_verdict == "incompatible":
                status = "conflict"
                label = pair_conflicts[0] if pair_conflicts else "Конфлікт"
            elif pair_verdict == "candidate_conflict":
                status = "warn"
                label = pair_conflicts[0] if pair_conflicts \
                    else "Ймовірний конфлікт"
            else:
                status, label = "unknown", "немає даних"
            graph_links.append({
                "source": p1,
                "target": p2,
                "status": status,
                "label": label,
            })

    return {
        "domain": domain,
        "is_compatible": is_compatible,
        "verdict": verdict,
        "unresolved_ids": verdict_info["unresolved_ids"],
        "evaluated_rules": verdict_info["evaluated_rules"],
        "conflicts": verdict_info["conflicts"],
        "coverage_gaps": verdict_info.get("coverage_gaps", []),
        "candidate_conflicts": verdict_info.get("candidate_conflicts", []),
        # `n.__dict__` would embed the read-only attribute view in the
        # response, which the JSON encoder cannot serialise.
        "products": [{"id": n.id, "name": n.name, "category": n.category,
                      "price": n.price, "attributes": dict(n.attributes)}
                     for n in active_nodes],
        "graph_data": {"nodes": graph_nodes, "links": graph_links},
        "graph_links_truncated": graph_links_truncated,
    }

@router.post("/api/pareto")
def compute_pareto(req: ParetoRequest):
    domain = req.domain or ctx.active_domain
    hg = hypergraph_for(domain)              # 400 on an unknown domain
    bundle = ctx.get_domain_bundle(domain)
    selected_category = req.category.strip() if isinstance(req.category, str) and req.category.strip() else None
    relaxer = FuzzyParetoRelaxer(
        budget_max=req.budget_max,
        min_performance=req.min_performance,
        currency=bundle.currency,
    )

    # F18: performance comes from the source column when the domain has one;
    # a catalogue that does not state performance does not get an invented
    # value, and delivery days are used only when a merchant actually
    # provides them. Products without the attribute are excluded from the
    # front and counted, not silently assigned defaults.
    perf_by_id: Dict[str, Optional[float]] = {}
    deliv_by_id: Dict[str, Optional[int]] = {}
    categories_set = {str(n.category).strip() for n in hg.nodes.values() if getattr(n, "category", None)}
    category_nodes = None
    total_source_items = len(hg.nodes)
    if bundle.db_path and os.path.exists(bundle.db_path):
        try:
            # F10: the performance column is discovered through the canonical
            # source projection, not assumed to be `products.performance_score`.
            from source_projection import build_source_projection, quote_ident
            proj = build_source_projection(bundle.db_path)
            perf_res = proj.resolve("performance")
            ident = proj.resolve("identity")
            cat_res = proj.resolve("category")
            price_res = proj.resolve("price")
            name_res = proj.resolve("name")
            if ident.resolved:
                conn = sqlite3.connect(f"file:{bundle.db_path}?mode=ro", uri=True)
                try:
                    q = quote_ident
                    t = q(proj.entity_table)
                    total_row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                    if total_row:
                        total_source_items = int(total_row[0])
                    if cat_res.resolved:
                        for (cval,) in conn.execute(
                            f"SELECT DISTINCT {q(cat_res.column)} FROM {t} "
                            f"WHERE {q(cat_res.column)} IS NOT NULL AND TRIM({q(cat_res.column)}) != '' "
                            f"ORDER BY {q(cat_res.column)} LIMIT 100"
                        ):
                            categories_set.add(str(cval).strip())
                    if perf_res.resolved:
                        for pid, val in conn.execute(
                            f"SELECT {q(ident.column)}, {q(perf_res.column)} FROM {t}"
                        ):
                            perf_by_id[str(pid)] = None if val is None else float(val)
                    # Optional merchant delivery_days read directly from relational offer join
                    tbl_meta = proj.snapshot.tables.get(proj.entity_table)
                    if tbl_meta and "merchant_id" in tbl_meta.columns and "merchants" in proj.snapshot.tables:
                        m_tbl = proj.snapshot.tables["merchants"]
                        if "delivery_days" in m_tbl.columns:
                            for pid, dval in conn.execute(
                                f"SELECT e.{q(ident.column)}, m.delivery_days "
                                f"FROM {t} e JOIN merchants m ON e.merchant_id = m.id "
                                f"WHERE m.delivery_days IS NOT NULL"
                            ):
                                deliv_by_id[str(pid)] = int(dval)
                    if selected_category and cat_res.resolved:
                        name_col = q(name_res.column) if name_res.resolved else q(ident.column)
                        price_col = q(price_res.column) if price_res.resolved else "NULL"
                        perf_col = q(perf_res.column) if perf_res.resolved else "NULL"
                        category_nodes = []
                        for pid, cname, ccat, cprice, cperf in conn.execute(
                            f"SELECT {q(ident.column)}, {name_col}, {q(cat_res.column)}, {price_col}, {perf_col} "
                            f"FROM {t} WHERE LOWER(TRIM(COALESCE({q(cat_res.column)}, ''))) = LOWER(?) LIMIT 500",
                            (selected_category,),
                        ):
                            attrs = dict(hg.nodes[str(pid)].attributes) if str(pid) in hg.nodes else {}
                            if cperf is not None and "performance" not in attrs:
                                attrs["performance"] = float(cperf)
                            category_nodes.append({
                                "id": str(pid),
                                "name": str(cname) if cname is not None else str(pid),
                                "category": str(ccat).strip() if ccat is not None else selected_category,
                                "price": float(cprice) if cprice is not None else None,
                                "attributes": attrs,
                            })
                finally:
                    conn.close()
        except Exception:
            perf_by_id = {}

    if category_nodes is not None:
        source_pool = category_nodes
    elif selected_category:
        sel_low = selected_category.lower()
        source_pool = [
            {"id": p.id, "name": p.name, "category": p.category, "price": p.price, "attributes": dict(p.attributes)}
            for p in hg.nodes.values()
            if str(getattr(p, "category", "") or "").strip().lower() == sel_low
        ]
    else:
        source_pool = [
            {"id": p.id, "name": p.name, "category": p.category, "price": p.price, "attributes": dict(p.attributes)}
            for p in hg.nodes.values()
        ]

    candidates = []
    category_by_id: Dict[str, Optional[str]] = {}
    excluded_without_performance = 0
    excluded_without_price = 0
    for p in source_pool:
        pid = str(p["id"])
        attrs = p["attributes"]
        raw = attrs.get("performance")
        if raw is None:
            raw = perf_by_id.get(pid)
        if raw is None:
            excluded_without_performance += 1
            continue
        # F10: a source without a price axis cannot support a budget-constrained
        # front; such items are excluded and counted, not priced at 0.0.
        if p["price"] is None:
            excluded_without_price += 1
            continue
        raw_deliv = attrs.get("delivery_days")
        if raw_deliv is None:
            raw_deliv = deliv_by_id.get(pid)
        try:
            deliv = int(raw_deliv) if raw_deliv is not None else None
        except (TypeError, ValueError):
            deliv = None
        category_by_id[pid] = str(p.get("category") or "").strip() or None
        candidates.append(CandidateSolution(
            item_id=pid,
            name=str(p["name"]),
            price=float(p["price"]),
            performance_score=float(raw),
            delivery_days=deliv
        ))

    def _serialize(c: CandidateSolution) -> Dict[str, object]:
        d = dict(c.__dict__)
        d["category"] = category_by_id.get(c.item_id)
        return d

    sorted_categories = sorted(categories_set)

    if not candidates:
        reasons = []
        if excluded_without_performance:
            reasons.append(f"{excluded_without_performance} товарів без заявленого показника performance")
        if excluded_without_price:
            reasons.append(f"{excluded_without_price} товарів без заявленої ціни")
        explanation = ("Парето-фронт побудувати неможливо: " + "; ".join(reasons) + "."
                       if reasons else
                       "У цьому домені немає товарів, придатних для чесного Парето-фронту.")
        return {
            "currency": bundle.currency,
            "categories": sorted_categories,
            "selected_category": selected_category,
            "total_source_items": total_source_items,
            "price_range": None,
            "performance_range": None,
            "budget_max": req.budget_max,
            "min_performance": req.min_performance,
            "all_candidates": [],
            "pareto_front": [],
            "pareto_front_feasible": [],
            "best_a": None,
            "best_b": None,
            "excluded_without_performance": excluded_without_performance,
            "excluded_without_price": excluded_without_price,
            "explanation": explanation,
        }

    pareto_front = relaxer.compute_pareto_front(candidates)
    feasible = [c for c in candidates if relaxer.fits(c)]
    feasible_front = relaxer.compute_pareto_front(feasible)
    scope_front = feasible_front if feasible else pareto_front
    best_a, best_b, explanation = relaxer.select_contrastive_tradeoffs(scope_front)
    prices = [float(c.price) for c in candidates if c.price is not None]
    perfs = [float(c.performance_score) for c in candidates]

    return {
        "currency": bundle.currency,
        "categories": sorted_categories,
        "selected_category": selected_category,
        "total_source_items": total_source_items,
        "price_range": {"min": min(prices), "max": max(prices)} if prices else None,
        "performance_range": {"min": min(perfs), "max": max(perfs)} if perfs else None,
        "budget_max": req.budget_max,
        "min_performance": req.min_performance,
        "front_scope": "feasible" if feasible else "no_feasible_item",
        "all_candidates": [_serialize(c) for c in candidates],
        "pareto_front": [_serialize(c) for c in pareto_front],
        "pareto_front_feasible": [_serialize(c) for c in feasible_front],
        "best_a": _serialize(best_a) if best_a else None,
        "best_b": _serialize(best_b) if best_b else None,
        "excluded_without_performance": excluded_without_performance,
        "excluded_without_price": excluded_without_price,
        "explanation": explanation
    }
