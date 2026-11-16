"""Chat consultation route and per-session domain resolution."""

from fastapi import Request
from typing import Optional
import time
from answer_nvfg import summarize_answer_nvfg
from react_agent import SessionState
from session_store import resolve_session_id
from server_context import ctx
from server_resolution import _is_compatibility_question, _resolve_named_products, hypergraph_for
from server_schemas import ChatQueryRequest
from server_security import _is_admin_request
from fastapi import APIRouter

router = APIRouter()


def _resolve_chat_domain(req_domain: Optional[str],
                         session: SessionState) -> str:
    """R6: per-request domain > session preference > global default.

    A stored session domain that no longer names a registered domain is
    ignored rather than bricking the session with a 400.
    """
    session_domain = (session.domain
                      if session.domain in ctx.domains else None)
    return req_domain or session_domain or ctx.active_domain


@router.post("/api/chat")
def chat_consult(req: ChatQueryRequest, request: Request):
    t0 = time.perf_counter()
    # F30: switching the global model from a chat request is an
    # administrative operation, not something any caller may do inline.
    if req.model and _is_admin_request(request):
        ctx.runner.model_name = req.model.strip()
        ctx.runner.last_model = None
    # F30: session identity is validated server-side; a missing or malformed
    # id is replaced by a server-minted one instead of trusting the client.
    session_id = resolve_session_id(req.session_id, ctx.session_store)
    # F22/R6: the session is loaded before the domain is resolved, because
    # the session carries its own domain preference. Resolution order:
    # explicit request domain > this session's domain > the admin-set
    # global default. An explicit request domain never leaks into another
    # session's state.
    session = ctx.session_store.get_session(session_id) or SessionState(
        session_id=session_id,
        user_query=req.query
    )
    domain = _resolve_chat_domain(req.domain, session)

    # Two or more products named verbatim in the question: answer with the
    # deterministic hypergraph instead of spending a model call.
    #
    # The previous version hard-coded four product ids behind the words
    # "golf", "bmw", "bbs" and "mansory". It was a demo masquerading as a
    # feature: it could only answer the four questions it was written for
    # and silently did nothing for every other query. Resolution is now by
    # exact name match against the active domain's own catalogue, so it
    # works for every product the store actually carries.
    hg = hypergraph_for(domain)
    # F14: resolution must not depend on what the UI happened to browse.
    # Resolve against the graph first, then against the full source catalogue
    # via the DB index, lazily loading products the graph has not seen yet.
    bundle = ctx.get_domain_bundle(domain)
    named = _resolve_named_products(hg, bundle.db_path, req.query)
    # F17: two resolved names alone do not validate the task type. The
    # deterministic branch answers a compatibility question, so it fires
    # only when (a) the query is phrased as one and (b) the graph actually
    # carries evaluative structure -- otherwise the verdict would always be
    # "unknown" and the general agent path is strictly more useful.
    if len(named) >= 2 and _is_compatibility_question(req.query) \
            and (hg.constraints or hg.hyperedges):
        pids = {n.id for n in named}
        # F15: the verdict distinguishes rule-verified compatibility from the
        # retail basket default and from unresolved items.
        verdict_info = hg.compatibility_verdict(pids)
        verdict = verdict_info["verdict"]
        conflicts = hg.find_conflicts(pids) if verdict == "incompatible" else []
        if verdict == "compatible":
            explanation = ("✅ **Комплект перевірено верифікованими правилами — сумісний!** "
                           + ", ".join(n.name for n in named) + ".")
        elif verdict == "incompatible":
            explanation = ("❌ **Несумісність виявлена Віртуальним гіперграфом:**\n"
                           + "\n".join(f"- {c}" for c in conflicts))
        elif verdict == "candidate_conflict":
            # F12: an induced, unverified rule signals a conflict. Honest
            # framing: a warning, not verified physics.
            explanation = ("⚠️ **Індуковане правило сигналізує про можливу несумісність** "
                           "(правило виведене з кореляцій вибірки і не верифіковане):\n"
                           + "\n".join(f"- {c}" for c in verdict_info["candidate_conflicts"]))
        else:
            # F12: candidate (induced, unverified) rules may signal a
            # conflict, but their pass or failure is not evidence.
            cand_note = ""
            if verdict_info.get("candidate_conflicts"):
                cand_note = (" Кандидат-правила (індуковані з кореляцій, не верифіковані) "
                             "сигналізують про можливий конфлікт: "
                             + "; ".join(verdict_info["candidate_conflicts"]) + ".")
            explanation = ("⚠️ **Сумісність не підтверджена верифікованими правилами.** "
                           "Спільна купівля можлива, але це не доказ сумісності: "
                           + ", ".join(n.name for n in named) + "." + cand_note)
        return {
            "query": req.query,
            "final_answer": explanation,
            "steps": [{
                "step": 1,
                "thought": "Розпізнано сутності каталогу за точним збігом назв",
                "action": {"tool": "HypergraphValidate", "input": sorted(pids)},
                "observation": str(conflicts),
            }],
            # tier=None: this answer came from the deterministic hypergraph,
            # not from any router tier. The old API omitted `tier` here and the
            # old console rendered the omission as "Tier-2 Deep ReAct".
            "tier": None,
            # F08: the latency of this branch is measured, not reported as 0.
            "latency_seconds": round(time.perf_counter() - t0, 4),
            "token_cost": 0.0,
            "model": None,
            "termination_reason": "DETERMINISTIC_HYPERGRAPH",
            "verification_report": None,
            "pre_verifier_response": None,
            "nvfg": None,
            "recommended_items": [
                {"id": n.id, "name": n.name, "category": n.category,
                 "price": n.price}
                for n in named
            ],
            "verdict": verdict,
            "is_compatible": {"compatible": True, "incompatible": False,
                              "candidate_conflict": False}.get(verdict),
            "status": "success",
            "domain": domain,
            "currency": getattr(ctx.domains.get(domain), "currency", None),
        }

    # Execute through Fast Cascaded Router bound to the active/requested domain (C1 fix)
    #
    # F22: the persisted belief travels with the session -- the elicitation
    # dialogue picks up where the last turn ended rather than restarting
    # from a flat prior on every request.
    domain_router = ctx.get_router_for_domain(domain)
    routed = domain_router.route_and_execute(
        req.query,
        session_id=session_id,
        belief_state=session.belief_state,
        asked_question_ids=session.asked_question_ids,
        conversation_history=session.conversation_history,
    )

    # Persist session state into SQLite. The session's domain is NOT
    # rewritten here: `req.domain` is a per-request override, and
    # `session.domain` is an explicit preference set only through
    # /api/switch_domain. Pinning a request-time override would turn the
    # shared default session id into the same leak the global was.
    session.final_response = routed["answer"]
    session.execution_history = routed["steps"]
    if routed.get("status") != "error" and routed.get("answer"):
        session.conversation_history = (
            list(session.conversation_history)
            + [{"role": "user", "content": req.query}, {"role": "assistant", "content": str(routed["answer"])}]
        )[-12:]
    if routed.get("belief_state") is not None:
        session.belief_state = routed["belief_state"]
    if routed.get("asked_question_ids") is not None:
        session.asked_question_ids = list(routed["asked_question_ids"])
    ctx.session_store.save_session(session)

    return {
        "query": req.query,
        "final_answer": routed["answer"],
        "steps": routed["steps"],
        "tier": routed["tier"],
        "latency_seconds": routed.get("latency_seconds", 0.0),
        "token_cost": routed.get("token_cost", 0.0),
        "model": routed.get("model"),
        # Gate verdict for this answer (None on the deterministic SQL path).
        "termination_reason": routed.get("termination_reason"),
        "verification_report": routed.get("verification_report"),
        "pre_verifier_response": routed.get("pre_verifier_response"),
        "nvfg": summarize_answer_nvfg(routed["steps"], routed.get("termination_reason")),
        "recommended_items": routed.get("recommended_items", []),
        "status": routed.get("status", "success"),
        "domain": domain,
        "currency": getattr(ctx.domains.get(domain), "currency", None),
        "session_id": session_id,
        "stage_timings": routed.get("stage_timings")
    }
