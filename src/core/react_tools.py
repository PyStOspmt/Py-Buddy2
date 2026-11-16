"""
Py-Buddy: Neuro-Symbolic ReAct Tool Executors.
Cohesive domain handlers for SQL querying, semantic search, compatibility checking,
Pareto constraint relaxation, active preference elicitation, and ontology navigation.
"""

from dataclasses import asdict
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from catalog_query import GroundingSet, fetch_candidates
from nvfg import Constraint, Op
from pareto_relaxer import FuzzyParetoRelaxer
from relaxation_lattice import solve as nvfg_lattice_solve
from sanitizer import SecurityViolationError, escape_like
from source_currency import format_currency_label
from source_projection import build_source_projection, quote_ident

#: Upper bound on the NVFG solver candidate pool to keep lattice search O(N*C*B) bounded.
_SOLVER_POOL_CAP = 2000

#: Fixed latent elicitation axes mapped onto Pareto optimization criteria.
PREFERENCE_AXIS_MAP: Dict[str, str] = {
    "price_sensitivity": "price",
    "performance_focus": "performance",
    "urgency_delivery": "delivery",
}


def agent_grounding(agent: Any, state: Any) -> GroundingSet:
    """Return the per-run GroundingSet on state, falling back to agent.grounding."""
    gs = getattr(state, "grounding", None)
    return gs if gs is not None else agent.grounding


def preference_weights(state: Any) -> Optional[Dict[str, float]]:
    """Derive Pareto ranking weights from elicited answers, or None if unelicited."""
    if not getattr(state, "asked_question_ids", None):
        return None
    means = state.belief_state.dimensions
    weights = {
        axis: means[dim]
        for dim, axis in PREFERENCE_AXIS_MAP.items()
        if dim in means
    }
    return weights or None


def infer_category(agent: Any, state: Any) -> Optional[str]:
    """Match user query tokens against live category values projected from db_path."""
    if agent._category_cache is None:
        if not agent.db_path or not os.path.exists(agent.db_path):
            agent._category_cache = []
            return None
        try:
            proj = build_source_projection(agent.db_path)
            table = proj.entity_table
            cat_res = proj.resolve("category")
            if not table or not cat_res.resolved:
                agent._category_cache = []
                return None
            ro_uri = "file:%s?mode=ro" % os.path.abspath(agent.db_path).replace("\\", "/")
            conn = sqlite3.connect(ro_uri, uri=True)
            try:
                agent._category_cache = [
                    str(r[0])
                    for r in conn.execute(
                        f"SELECT DISTINCT {quote_ident(cat_res.column)} "
                        f"FROM {quote_ident(table)} "
                        f"WHERE {quote_ident(cat_res.column)} IS NOT NULL"
                    )
                ]
            finally:
                conn.close()
        except Exception:
            agent._category_cache = []
            return None

    q = set(re.findall(r"[\w]+", (state.user_query or "").lower()))
    best, best_hits = None, 0
    for cat in agent._category_cache:
        hits = len(q & set(re.split(r"[^a-z0-9]+", cat.lower())))
        if hits > best_hits:
            best, best_hits = cat, hits
    return best if best_hits else None


def tool_sql_query(
    agent: Any, state: Any, args: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Execute read-only SQL through SQLSanitizer and ground provenance-verified rows."""
    raw_query = args.get("query", "")
    try:
        result = agent.sanitizer.execute_safe_query(agent.db_path, raw_query)
        rows = result.get("data") or result.get("rows") or []
        if isinstance(rows, list):
            agent_grounding(agent, state).extend_rows(
                rows, provenance=result.get("column_provenance")
            )
        if result.get("fuzzy_suggestions"):
            agent_grounding(agent, state).extend_rows(result["fuzzy_suggestions"])
        return result, None
    except SecurityViolationError as sve:
        err = str(sve)
        return {"success": False, "error": err}, err


def tool_catalog_search(
    agent: Any, state: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Run domain-agnostic FTS5/LIKE paginated search with total counts and sorted slice."""
    from catalog_index import CatalogSearchIndex
    from query_awareness import search_catalog_window

    sem_idx = getattr(agent, "search_index", None)
    if sem_idx is None:
        sem_idx = CatalogSearchIndex()
        agent.search_index = sem_idx

    res = search_catalog_window(
        agent.db_path,
        query=str(args.get("query") or ""),
        category=args.get("category"),
        sort=str(args.get("sort") or "price_asc"),
        limit=int(args.get("limit", 10) or 10),
        offset=int(args.get("offset", 0) or 0),
        search_index=sem_idx,
        max_work_steps=getattr(getattr(agent, "sanitizer", None), "max_work_steps", 5_000_000),
    )
    rows = res.get("rows") or []
    if rows:
        agent_grounding(agent, state).extend_rows(
            rows, provenance=res.get("column_provenance")
        )
    return res


def tool_semantic_search(
    agent: Any, state: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Run hybrid semantic search over the bundle's SemanticIndex and ground hits."""
    sem = agent.semantic_index
    query = str(args.get("query") or "")[:500]
    if sem is None:
        return {"success": False, "error": "semantic index not configured"}
    if not query.strip():
        return {"success": False, "error": "empty query"}
    try:
        k = max(1, min(10, int(args.get("k", 5))))
    except (TypeError, ValueError):
        k = 5
    try:
        res = sem.search(query, k=k, mode="hybrid", hyde=bool(args.get("hyde", False)))
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    hits = [
        {
            "id": h["entity_id"],
            "name": h["name"],
            "snippet": h["snippet"][:240],
            "score": h["score"],
        }
        for h in res.get("hits", [])
    ]
    idx_state = (res.get("index") or {}).get("state") or res.get("status")
    ok = res.get("status") == "ok"
    result: Dict[str, Any] = {
        "success": ok,
        "status": res.get("status"),
        "hits": hits,
        "confidence": res.get("confidence"),
        "index_fresh": (res.get("index") or {}).get("fresh"),
        "notes": res.get("notes", []),
    }
    if not ok:
        result["error"] = f"semantic index not ready: {idx_state}"
    if hits:
        agent_grounding(agent, state).extend_rows(
            [{"id": h["id"], "name": h["name"]} for h in hits]
        )
    while len(json.dumps(result, ensure_ascii=False)) > 3000 and len(hits) > 1:
        hits = hits[: len(hits) - 2]
        result["hits"], result["truncated"] = hits, True
    return result


def tool_hypergraph_check(
    agent: Any, state: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve item IDs/names into the KnowledgeHypergraph and evaluate compatibility."""
    from domain_catalog import DomainCatalogRegistry

    raw_ids = args.get("item_ids", [])
    if not isinstance(raw_ids, (list, tuple)):
        raw_ids = [raw_ids]

    resolved_ids = set()
    unresolved_items: List[str] = []
    for raw_val in raw_ids:
        s_val = str(raw_val).strip()
        if s_val in agent.hypergraph.nodes:
            resolved_ids.add(s_val)
            continue

        node = DomainCatalogRegistry.load_product_dynamically(agent.db_path, s_val)
        if node:
            agent.hypergraph.add_product(node)
            resolved_ids.add(node.id)
            continue

        matched = False
        if agent.db_path and os.path.exists(agent.db_path):
            try:
                proj = build_source_projection(agent.db_path)
                table = proj.entity_table
                ident = proj.resolve("identity")
                name_res = proj.resolve("name")
                if table and ident.resolved:
                    t_id = quote_ident(table)
                    id_c = quote_ident(ident.column)
                    ro_uri = "file:%s?mode=ro" % os.path.abspath(agent.db_path).replace("\\", "/")
                    conn = sqlite3.connect(ro_uri, uri=True)
                    try:
                        if name_res.resolved:
                            n_c = quote_ident(name_res.column)
                            r = conn.execute(
                                f"SELECT {id_c} FROM {t_id} "
                                f"WHERE {id_c} = ? OR {n_c} LIKE ? ESCAPE '\\' LIMIT 1",
                                (s_val, f"%{escape_like(s_val)}%"),
                            ).fetchone()
                        else:
                            r = conn.execute(
                                f"SELECT {id_c} FROM {t_id} WHERE {id_c} = ? LIMIT 1",
                                (s_val,),
                            ).fetchone()
                    finally:
                        conn.close()
                    if r:
                        node = DomainCatalogRegistry.load_product_dynamically(
                            agent.db_path, str(r[0])
                        )
                        if node:
                            agent.hypergraph.add_product(node)
                            resolved_ids.add(node.id)
                            matched = True
            except Exception:
                pass
        if not matched:
            unresolved_items.append(s_val)

    resolved_rows = []
    for nid in sorted(resolved_ids):
        node = agent.hypergraph.nodes.get(nid)
        if node is not None:
            resolved_rows.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "category": node.category,
                    "price": node.price,
                    "attributes": dict(node.attributes),
                }
            )
    if resolved_rows:
        agent_grounding(agent, state).extend_rows(resolved_rows)

    verdict_info = agent.hypergraph.compatibility_verdict(resolved_ids)
    verdict = verdict_info["verdict"]
    conflicts = (
        agent.hypergraph.find_conflicts(resolved_ids)
        if verdict == "incompatible"
        else []
    )
    candidate_rules_passed = 0
    if (
        verdict == "unknown"
        and resolved_ids
        and not unresolved_items
        and agent.hypergraph.constraints
    ):
        active_nodes = [
            agent.hypergraph.nodes[pid]
            for pid in resolved_ids
            if pid in agent.hypergraph.nodes
        ]
        applied_cand = [
            c
            for c in agent.hypergraph.constraints
            if c.status != "verified" and c.applies_to(active_nodes)
        ]
        if applied_cand and not verdict_info.get("candidate_conflicts"):
            candidate_rules_passed = len(applied_cand)

    return {
        "success": True,
        "verdict": verdict,
        "is_compatible": {
            "compatible": True,
            "incompatible": False,
            "candidate_conflict": False,
        }.get(verdict),
        "conflicts": conflicts,
        "candidate_conflicts": verdict_info.get("candidate_conflicts", []),
        "candidate_rules_passed": candidate_rules_passed,
        "checked_items": sorted(resolved_ids),
        "checked_item_details": resolved_rows,
        "unresolved_items": unresolved_items,
        "evaluated_rules": verdict_info["evaluated_rules"],
    }


def tool_pareto_relax(
    agent: Any, state: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Solve hard + soft constraints via NVFG relaxation lattice and Pareto frontier."""
    budget = float(args.get("budget_max", 10000.0))
    min_perf = float(args.get("min_performance", 80.0))
    category = args.get("category") or infer_category(agent, state)

    pool = fetch_candidates(
        agent.db_path,
        category=category,
        budget_max=None,
        min_performance=None,
        limit=_SOLVER_POOL_CAP,
        hard_limit=_SOLVER_POOL_CAP,
    )
    pool_dicts = [
        {
            "id": c.item_id,
            "name": c.name,
            "price": c.price,
            "performance_score": c.performance_score,
        }
        for c in pool
    ]
    by_id = {d["id"]: c for d, c in zip(pool_dicts, pool)}

    raw_prot = args.get("protected") or []
    if isinstance(raw_prot, str):
        raw_prot = [raw_prot]
    protected_set = {str(p).strip().lower() for p in raw_prot}
    budget_prot = bool(args.get("budget_protected")) or bool(
        {"price", "budget", "бюджет"} & protected_set
    )
    perf_prot = bool(args.get("performance_protected")) or bool(
        {"performance", "performance_score", "продуктивність"} & protected_set
    )

    constraints: List[Constraint] = []
    axes_evaluated: List[str] = []
    axes_skipped: List[str] = []
    if any(d["price"] is not None for d in pool_dicts):
        budget_unit = format_currency_label(getattr(agent, "currency", None))
        constraints.append(
            Constraint("бюджет", "price", Op.AT_MOST, budget, budget_unit, protected=budget_prot)
        )
        axes_evaluated.append("price")
    else:
        axes_skipped.append("price")

    if any(d["performance_score"] is not None for d in pool_dicts):
        constraints.append(
            Constraint(
                "продуктивність",
                "performance_score",
                Op.AT_LEAST,
                min_perf,
                protected=perf_prot,
            )
        )
        axes_evaluated.append("performance")
    else:
        axes_skipped.append("performance")

    nvfg_res = nvfg_lattice_solve(pool_dicts, constraints)
    solved = [by_id[d["id"]] for d in nvfg_res.solved_items if d["id"] in by_id]
    search_space = {
        "pool_size": len(pool),
        "pool_capped": len(pool) >= _SOLVER_POOL_CAP,
    }
    protected_axes = [
        label for label, is_prot in (("бюджет", budget_prot), ("продуктивність", perf_prot))
        if is_prot
    ]
    relaxed = nvfg_res.status == "RELAXED"

    if not solved:
        return {
            "success": True,
            "nvfg_status": nvfg_res.status,
            "protected": protected_axes,
            "pareto_front_count": 0,
            "option_a": None,
            "option_b": None,
            "explanation": nvfg_res.explain(),
            "axes_evaluated": axes_evaluated,
            "axes_skipped": axes_skipped,
            "search_space": search_space,
        }

    relaxer = FuzzyParetoRelaxer(
        budget_max=budget,
        min_performance=min_perf,
        currency=getattr(agent, "currency", None),
    )
    front = relaxer.compute_pareto_front(solved)
    pref_weights = preference_weights(state)
    best_a, best_b, explanation = relaxer.select_contrastive_tradeoffs(
        front, weights=pref_weights
    )
    agent_grounding(agent, state).extend(solved)
    if relaxed:
        explanation = (nvfg_res.relaxation_text or nvfg_res.explain()) + " " + explanation

    return {
        "success": True,
        "category": category,
        "budget_max": budget,
        "min_performance": min_perf,
        "nvfg_status": nvfg_res.status,
        "protected": protected_axes,
        "relaxed": relaxed,
        "relaxation": nvfg_res.relaxation.to_dict() if nvfg_res.relaxation else None,
        "relaxation_text": nvfg_res.relaxation_text,
        "regret": nvfg_res.regret,
        "relative_regret": nvfg_res.relative_regret,
        "candidate_count": len(solved),
        "pareto_front_count": len(front),
        "option_a": asdict(best_a) if best_a else None,
        "option_b": asdict(best_b) if best_b else None,
        "explanation": explanation,
        "preference_weights": pref_weights,
        "preference_source": "recorded_answers" if pref_weights is not None else "default",
        "axes_evaluated": axes_evaluated,
        "axes_skipped": axes_skipped,
        "search_space": search_space,
    }


def tool_active_inference(
    agent: Any, state: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Select the next diagnostic question or record an answer into session belief_state."""
    asked = set(state.asked_question_ids)
    if "question_id" in args or "answer" in args:
        qid = args.get("question_id")
        question = next(
            (q for q in agent.active_inference.question_bank if q.id == qid), None
        )
        if question is None:
            return {"success": False, "error": f"Unknown diagnostic question id: {qid!r}."}
        if agent.active_inference.record_answer(
            state.belief_state, question, args.get("answer"), asked=asked
        ):
            state.asked_question_ids = sorted(asked)
            dim = question.target_dimension
            return {
                "success": True,
                "recorded": True,
                "question_id": question.id,
                "target_dimension": dim,
                "posterior_mean": state.belief_state.posterior_mean(dim),
                "posterior_means": state.belief_state.dimensions,
            }
        return {
            "success": False,
            "error": f"Answer is not one of the offered options for {qid!r}; nothing was recorded.",
        }

    best_q = agent.active_inference.select_best_question(
        state.belief_state, exclude=asked
    )
    if best_q:
        q, net_gain = best_q
        return {
            "success": True,
            "question_id": q.id,
            "question_text": q.text,
            "target_dimension": q.target_dimension,
            "options": list(q.options.keys()),
            "net_information_gain": net_gain,
            "adapted": bool(q.adapted),
        }
    if not agent.active_inference.question_bank:
        return {
            "success": True,
            "question_id": None,
            "reason": "no_question_bank",
            "message": "У цьому домені не налаштовано жодного діагностичного запитання.",
        }
    return {
        "success": True,
        "question_id": None,
        "reason": "below_gain_threshold",
        "message": (
            "Очікувана інформативність наступного запитання нижча за поріг — "
            "вподобання достатньо з'ясовано."
        ),
    }


def tool_ontograph_navigate(agent: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Query the hierarchical OntoGraph taxonomy for categories and price ranges."""
    q = args.get("query") or args.get("category") or ""
    if not getattr(agent, "ontograph", None) and agent.db_path and os.path.exists(agent.db_path):
        from ontograph import OntoGraph
        agent.ontograph = OntoGraph(agent.db_path)
    res = agent.ontograph.navigate(q) if getattr(agent, "ontograph", None) else {"found": False}
    return {"success": True, **res}


def dispatch_tool(
    agent: Any, state: Any, tool_name: str, args: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Dispatch a single ReAct tool call and return (observation, error)."""
    if tool_name == "sql_query":
        return tool_sql_query(agent, state, args)
    if tool_name == "catalog_search":
        return tool_catalog_search(agent, state, args), None
    if tool_name == "hypergraph_check":
        return tool_hypergraph_check(agent, state, args), None
    if tool_name == "pareto_relax":
        return tool_pareto_relax(agent, state, args), None
    if tool_name == "active_inference_step":
        return tool_active_inference(agent, state, args), None
    if tool_name == "ontograph_navigate":
        return tool_ontograph_navigate(agent, args), None
    if tool_name == "semantic_search":
        return tool_semantic_search(agent, state, args), None
    error = f"Unknown tool: '{tool_name}'"
    return {"success": False, "error": error}, error
