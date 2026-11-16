"""
Py-Buddy: Fast Cascaded Router
Provides sub-50ms Tier-1 heuristic routing for direct catalog queries (pricing, availability, specifications)
while transparently delegating multi-constraint reasoning to Tier-2 Neuro-Symbolic ReAct.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
import time

from catalog_query import build_extremum_sql
from router import ModelRouter
from sanitizer import SQLSanitizer
from source_currency import UNSET_CURRENCY, format_price_with_currency, resolve_source_currency
from react_agent import ReActAgent

class ExecutionTier(str, Enum):
    TIER0_CONVERSATIONAL = "tier0_conversational"
    TIER1_FAST_PATH = "tier1_fast_path"
    TIER2_REACT_REASONING = "tier2_react_reasoning"

@dataclass
class RoutingDecision:
    tier: ExecutionTier
    query: str
    target_sql: Optional[str] = None
    reason: str = ""

class FastCascadedRouter:
    """
    Decides the EXECUTION TIER and runs it.

    Which model serves a tier is `router.ModelRouter`'s question, not this
    one's. The two were merged once and split again: a single object that
    both classified intent and picked models grew its own second complexity
    classifier, and the two copies disagreed. Keeping the boundary explicit
    means a change to intent detection cannot silently alter model choice.
    """
    def __init__(
        self,
        db_path: str,
        sanitizer: SQLSanitizer,
        react_agent: ReActAgent,
        llm_runner: Any,
        schema_prompt: str,
        model_router: Optional[ModelRouter] = None,
        currency: Any = UNSET_CURRENCY,
    ):
        self.db_path = db_path
        self.sanitizer = sanitizer
        self._last_fast_params: Optional[list] = None
        self._last_fast_error: Optional[str] = None
        self.agent = react_agent
        self.runner = llm_runner
        self.schema_prompt = schema_prompt
        self._explicit_currency = currency
        #: Optional. When present, every LLM-backed call is booked against
        #: it, so the unit-economics figure covers the traffic that actually
        #: happened rather than the traffic we assumed would happen.
        self.model_router = model_router
        self.last_model: Optional[str] = None

    def _active_currency(self) -> Optional[str]:
        explicit = self._explicit_currency
        if explicit is UNSET_CURRENCY:
            explicit = getattr(self.agent, "currency", UNSET_CURRENCY)
        return resolve_source_currency(self.db_path, configured_currency=explicit)

    def extract_domain_signals(self, query: str) -> Dict[str, Any]:
        """
        Dynamically extracts domain-agnostic signals (alphanumeric model codes, units)
        without relying on static dictionary wordlists.
        """
        q_lower = query.lower()
        has_tech_specs = bool(re.search(
            r"(\b\d+[a-zA-Zа-яА-ЯёЁіІїЇєЄґҐ%]+\b|\b[a-zA-Zа-яА-ЯёЁіІїЇєЄґҐ]+\d+\b|\b\d+x\d+\b)",
            q_lower
        ))
        return {
            "has_tech_specs": has_tech_specs
        }

    def classify_intent(self, query: str) -> Tuple[ExecutionTier, Optional[str]]:
        """
        Determines execution tier using pure schema-grounded introspection:
        - If query matches a closed-class extremum lookup over active catalogue vocabulary -> Tier-1 Fast Path.
        - All other queries -> Tier-2 Neuro-Symbolic ReAct reasoning (intelligent neural agent with dynamic tool dispatch).
        """
        built = build_extremum_sql(self.db_path, query, limit=1)
        if built:
            sql, params = built
            self._last_fast_params = params
            return ExecutionTier.TIER1_FAST_PATH, sql

        self._last_fast_params = None
        return ExecutionTier.TIER2_REACT_REASONING, None

    def _book_model_call(self, tier: int) -> None:
        """
        Attribute one LLM-backed call to the model pool, if one is attached.

        A missing pool is not an error: the deterministic Tier-1 path and any
        deployment that brings its own client both run fine without it. It
        only means the unit-economics figure is unavailable, which is
        reported as unavailable rather than as zero.
        """
        if self.model_router is None:
            return
        try:
            choice = self.model_router.select(tier, reason="FastCascadedRouter")
        except Exception:                    # pool exhausted: count nothing,
            return                          # and do not pretend it was free
        self.model_router.note_call(choice)
        self.last_model = choice.model

    def route_and_execute(
        self,
        query: str,
        session_id: str = "user_default",
        belief_state: Optional[Any] = None,
        asked_question_ids: Optional[list] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Routes and executes the query through the optimal execution tier.
        Sub-20ms deterministic queries execute via Tier-1 Fast SQL.
        All other queries are processed dynamically by the neural network, which intelligently
        decides whether to invoke agent tools (SQL, Hypergraph, Pareto) or answer directly.
        """
        start_time = time.time()
        tier, fast_sql = self.classify_intent(query)
        router_elapsed_ms = round((time.time() - start_time) * 1000, 2)

        if tier == ExecutionTier.TIER1_FAST_PATH and fast_sql:
            # Execute Tier-1 Fast Path
            try:
                # Bind, never interpolate: a catalogue value containing a
                # quote must not be able to become executable SQL.
                res = self.sanitizer.execute_safe_query(
                    self.db_path, fast_sql, self._last_fast_params)
                rows = res.get("rows", [])
                if rows:
                    item = rows[0]
                    name = item.get("name", "Товар")
                    price = item.get("price")
                    currency = self._active_currency()
                    if price is None:
                        answer = f"У каталозі знайдено {name}, але ціна не вказана."
                        obs_text = f"Знайдено: {name} (ціна не вказана)"
                    else:
                        price_text = format_price_with_currency(float(price), currency)
                        answer = f"Точна ціна {name} становить {price_text} (наявний у каталозі)."
                        obs_text = f"Знайдено: {name} ({price_text})"
                    elapsed = time.time() - start_time
                    return {
                        "query": query,
                        "answer": answer,
                        "tier": ExecutionTier.TIER1_FAST_PATH,
                        "latency_seconds": round(elapsed, 4),
                        "token_cost": 0.0,
                        "model": None,       # no model was called
                        "status": "success",
                        # Deterministic SQL: no model free text existed, so
                        # there was nothing for the gate to verify. `None`
                        # (rather than SUCCESS_GROUNDED) keeps "the LLM was
                        # checked and passed" distinct from "no LLM ran".
                        "termination_reason": None,
                        "verification_report": None,
                        "pre_verifier_response": None,
                        "recommended_items": [{"name": name, "price": price}],
                        "stage_timings": {
                            "router_time_ms": router_elapsed_ms,
                            "tool_time_ms": round((elapsed * 1000) - router_elapsed_ms, 2),
                            "llm_time_ms": 0.0,
                            "gate_time_ms": 0.0,
                            "total_time_ms": round(elapsed * 1000, 2)
                        },
                        "steps": [
                            {
                                "step": 1,
                                "thought": "Миттєвий детермінований SQL-пошук (Tier-1 Fast Path)",
                                "tool_name": "DirectSafeSQL",
                                "action": {"tool": "DirectSafeSQL", "query": fast_sql},
                                "observation": obs_text,
                                "result": {"name": name, "price": price},
                                "latency_s": round(elapsed, 4)
                            }
                        ]
                    }
            except Exception as exc:
                # A silent fall-through here once hid a SQL syntax error for
                # a whole debugging session. Falling back to the expensive
                # tier is correct; swallowing the reason is not.
                self._last_fast_error = f"{type(exc).__name__}: {exc}"

        # Intelligent Neural Agent execution. Book the call against the model
        # pool when one is attached, so the economics describe this traffic
        # and not an assumed one.
        self._book_model_call(tier=2)
        state = self.agent.run_live_session(
            user_query=query,
            llm_runner=self.runner,
            schema_prompt=self.schema_prompt,
            session_id=session_id,
            belief_state=belief_state,
            asked_question_ids=asked_question_ids,
            conversation_history=conversation_history,
        )
        elapsed = time.time() - start_time

        # F02: "no tools were invoked" alone never makes a turn a successful
        # conversation — a protocol or provider failure is neither.
        reason = state.termination_reason
        if reason and str(reason).startswith("SUCCESS"):
            status = "success"
        elif reason == "HALLUCINATION_GATE_VIOLATION":
            status = "gate_rejected"
        elif reason in ("PROTOCOL_ERROR", "PROVIDER_ERROR", "UNPARSEABLE_OUTPUT"):
            status = "error"
        else:
            status = "incomplete"
        is_direct_conversational = (len(state.tool_hashes) == 0 and status == "success")
        effective_tier = ExecutionTier.TIER0_CONVERSATIONAL if is_direct_conversational else ExecutionTier.TIER2_REACT_REASONING
        # F08: this figure is an estimate from history length, not a provider
        # bill. It is labelled as such, and the runner's real cumulative
        # counters are surfaced alongside when it exposes them.
        token_cost = round(len(state.execution_history) * 0.0001, 6)
        provider_usage: Optional[Dict[str, Any]] = None
        pt = getattr(self.runner, "total_prompt_tokens", None)
        ct = getattr(self.runner, "total_completion_tokens", None)
        if pt is not None or ct is not None:
            provider_usage = {
                "prompt_tokens": pt, "completion_tokens": ct,
                "basis": "cumulative_runner_lifetime",
            }

        stage_timings = dict(getattr(state, "stage_timings", {}))
        stage_timings["router_time_ms"] = router_elapsed_ms
        stage_timings["total_time_ms"] = round(elapsed * 1000, 2)

        return {
            "query": query,
            "answer": state.final_response or "Не вдалося отримати відповідь від асистента.",
            "tier": effective_tier,
            "latency_seconds": round(elapsed, 4),
            "token_cost": token_cost,
            # The router's own bookkeeping is None when no model pool is
            # attached; fall back to what the runner actually served, so the
            # UI shows the model that produced this answer, not "unknown".
            "model": (self.last_model
                      or getattr(self.runner, "last_model", None)
                      or getattr(self.runner, "model_name", None)),
            "status": status,
            # The gate's verdict on this answer (SUCCESS_GROUNDED, *_CORRECTION,
            # HALLUCINATION_GATE_VIOLATION, ...). The web console renders it as
            # a badge; telemetry consumers must not have to re-derive it.
            "termination_reason": state.termination_reason,
            "verification_report": state.verification_report,
            "pre_verifier_response": state.pre_verifier_response,
            "recommended_items": state.recommended_items,
            "token_cost_basis": "estimated_from_history_length",
            "provider_token_usage": provider_usage,
            "stage_timings": stage_timings,
            "steps": state.execution_history,
            # F22: the persistent belief this turn ended with. The caller
            # (session store) saves it so the next turn does not restart
            # elicitation from a flat prior.
            "belief_state": state.belief_state,
            "asked_question_ids": state.asked_question_ids,
        }
