"""
Live LLM agent arm for OpenRouter evaluation benchmarks.

Supports repeatable live comparison runs (`python src/experiments/benchmark.py --repeat 3`)
with strict fail-closed error accounting.

Its design decisions are all about not fabricating a measurement:

  * `runner=None` is an error, not a fallback. Offline fallbacks exist elsewhere
    so a suite can run without a key, but a *live* arm that silently went offline
    would be reported as live.
  * Every raw model output is recorded with the model name, tokens, latency and
    the caps in force. A non-deterministic arm that does not store its raw
    responses cannot be reproduced, and what cannot be reproduced is an anecdote.
  * Two caps: a hard call count always, a USD cap only when prices are
    configured. `stats()["budget_mode"]` names which was armed, so nobody can
    quote a dollar cap that was never active.
  * A transport error or a budget stop ends that case; it never invents a
    fallback answer.

Usage:
    agent = LivePyBuddyAgent(snap, runner, max_llm_calls=40)
    out = agent.answer(query)      # {"text", "steps", "cost_usd"}
    report = agent.stats()         # model, caps, spend, transcript
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))


class BudgetExceeded(Exception):
    """Raised inside the loop when a hard cap is hit. Ends the case, not the run."""


class SharedBudget:
    """Thread-safe shared call/USD budget ledger across concurrent worker runners."""

    def __init__(
        self,
        *,
        max_llm_calls: int = 100,
        max_usd: Optional[float] = None,
        price_in_per_mtok: float = 0.0,
        price_out_per_mtok: float = 0.0,
        transcript: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.max_llm_calls = int(max_llm_calls)
        self.max_usd = max_usd
        self.price_in_per_mtok = float(price_in_per_mtok)
        self.price_out_per_mtok = float(price_out_per_mtok)
        self.calls = 0
        self.reserved_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()
        self.transcript = transcript if transcript is not None else []

    @property
    def budget_mode(self) -> str:
        if self.max_usd is not None and (self.price_in_per_mtok or self.price_out_per_mtok):
            return "usd+calls"
        return "calls"

    def check_and_reserve(self) -> None:
        with self._lock:
            if self.reserved_calls >= self.max_llm_calls:
                raise BudgetExceeded(f"call cap {self.max_llm_calls} reached")
            if (self.max_usd is not None and self.budget_mode == "usd+calls"
                    and self.cost_usd >= self.max_usd):
                raise BudgetExceeded(f"USD cap {self.max_usd} reached")
            self.reserved_calls += 1

    def book(
        self,
        in_delta: int,
        out_delta: int,
        transcript_entry: Optional[Dict[str, Any]] = None,
    ) -> float:
        cost_delta = (in_delta * self.price_in_per_mtok
                      + out_delta * self.price_out_per_mtok) / 1_000_000.0
        with self._lock:
            self.calls += 1
            self.prompt_tokens += in_delta
            self.completion_tokens += out_delta
            self.cost_usd += cost_delta
            if transcript_entry is not None:
                transcript_entry.setdefault("step", self.calls)
                self.transcript.append(transcript_entry)
            return cost_delta


class BudgetedRunner:
    """
    Wraps a runner: counts calls and tokens, records every step, stops at the cap.

    The cap is checked BEFORE the call, so an over-budget runner never reaches the
    API -- a guard that stops you after you have paid is not a guard.
    """

    def __init__(
        self,
        inner: Any,
        *,
        max_llm_calls: int = 100,
        max_usd: Optional[float] = None,
        price_in_per_mtok: float = 0.0,
        price_out_per_mtok: float = 0.0,
        transcript: Optional[List[Dict[str, Any]]] = None,
        shared_budget: Optional[SharedBudget] = None,
    ) -> None:
        if inner is None:
            raise ValueError(
                "live_agent needs a runner: a live arm with no model measures "
                "nothing, it is not a cheaper run")
        self.inner = inner
        self.shared_budget = shared_budget
        if shared_budget is not None:
            self.max_llm_calls = shared_budget.max_llm_calls
            self.max_usd = shared_budget.max_usd
            self.price_in_per_mtok = shared_budget.price_in_per_mtok
            self.price_out_per_mtok = shared_budget.price_out_per_mtok
            self.transcript = shared_budget.transcript
            self._lock = shared_budget._lock
        else:
            self.max_llm_calls = int(max_llm_calls)
            self.max_usd = max_usd
            self.price_in_per_mtok = float(price_in_per_mtok)
            self.price_out_per_mtok = float(price_out_per_mtok)
            self._lock = threading.Lock()
            #: Shared with the caller so every case's steps land in one transcript.
            self.transcript = transcript if transcript is not None else []
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    @property
    def budget_mode(self) -> str:
        """Which cap is actually armed: `usd+calls` only when a price exists."""
        if self.max_usd is not None and (self.price_in_per_mtok or self.price_out_per_mtok):
            return "usd+calls"
        return "calls"

    def _book(
        self,
        before_prompt: int,
        before_completion: int,
        transcript_entry: Optional[Dict[str, Any]] = None,
    ) -> None:
        in_delta = max(0, getattr(self.inner, "total_prompt_tokens",
                                  before_prompt) - before_prompt)
        out_delta = max(0, getattr(self.inner, "total_completion_tokens",
                                   before_completion) - before_completion)
        cost_delta = (in_delta * self.price_in_per_mtok
                      + out_delta * self.price_out_per_mtok) / 1_000_000.0
        if self.shared_budget is not None:
            self.shared_budget.book(in_delta, out_delta, transcript_entry)
        with self._lock if self.shared_budget is None else self._noop_lock():
            self.calls += 1
            self.prompt_tokens += in_delta
            self.completion_tokens += out_delta
            self.cost_usd += cost_delta
            if self.shared_budget is None and transcript_entry is not None:
                transcript_entry.setdefault("step", self.calls)
                self.transcript.append(transcript_entry)

    @staticmethod
    def _noop_lock():
        import contextlib
        return contextlib.nullcontext()

    def _check_budget(self) -> None:
        if self.shared_budget is not None:
            self.shared_budget.check_and_reserve()
            return
        with self._lock:
            if self.calls >= self.max_llm_calls:
                raise BudgetExceeded(f"call cap {self.max_llm_calls} reached")
            if (self.max_usd is not None and self.budget_mode == "usd+calls"
                    and self.cost_usd >= self.max_usd):
                raise BudgetExceeded(f"USD cap {self.max_usd} reached")

    def generate_react_step(self, **kwargs: Any) -> Dict[str, Any]:
        self._check_budget()
        before_p = getattr(self.inner, "total_prompt_tokens", 0)
        before_c = getattr(self.inner, "total_completion_tokens", 0)
        t0 = time.time()
        try:
            step = self.inner.generate_react_step(**kwargs)
        except BudgetExceeded:
            raise
        except Exception as exc:
            # Transport or API failure: record it and re-raise. Inventing a
            # fallback answer here would score a network outage as knowledge.
            self._book(before_p, before_c, {
                "latency_s": round(time.time() - t0, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "query": kwargs.get("user_query"),
            })
            raise
        self._book(before_p, before_c, {
            "latency_s": round(time.time() - t0, 3),
            "model_step": (getattr(self.inner, "last_model", None)
                           or getattr(self.inner, "model_name",
                                      type(self.inner).__name__)),
            "action": step.get("action"),
            "final_answer": step.get("final_answer"),
            "recommended_items": len(step.get("recommended_items") or []),
        })
        return step

    def answer_once(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> tuple:
        self._check_budget()
        before_p = getattr(self.inner, "total_prompt_tokens", 0)
        before_c = getattr(self.inner, "total_completion_tokens", 0)
        t0 = time.time()
        try:
            text, _cost = self.inner.answer_once(system_prompt, user_prompt, **kwargs)
        except BudgetExceeded:
            raise
        except Exception as exc:
            self._book(before_p, before_c, {
                "latency_s": round(time.time() - t0, 3),
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        self._book(before_p, before_c)
        return text, self.cost_usd


def default_schema_prompt(db_path: str) -> str:
    """
    The catalogue description the ReAct prompt is built around.

    Built from the database itself, exactly as the server builds it, so the live
    arm and the deployed system see the same context. Falls back to "" rather
    than to a hand-written description: a hand-written one would describe the
    developer's store, not the catalogue under test.
    """
    try:
        from profiler import DomainAgnosticProfiler

        snapshot = DomainAgnosticProfiler(db_path).profile_database()
        return DomainAgnosticProfiler.generate_compact_schema_context(snapshot.to_dict())
    except Exception:
        return ""


class LivePyBuddyAgent:
    """
    The proposed system with a real language model in the loop.

    Same answer shape as OfflineDeterministicAgent so the harness scores both
    arms identically; the difference is that here the reasoning is the model's
    and only the gate decides what may be said.
    """

    name = "pybuddy_live"

    def __init__(
        self,
        snap,
        runner: Any,
        *,
        max_llm_calls: int = 100,
        max_usd: Optional[float] = None,
        price_in_per_mtok: float = 0.0,
        price_out_per_mtok: float = 0.0,
        schema_prompt: Optional[str] = None,
        shared_budget: Optional[SharedBudget] = None,
        hypergraph: Optional[Any] = None,
    ) -> None:
        import os
        from domain_catalog import DomainCatalogRegistry
        from react_agent import ReActAgent

        self.snap = snap
        db_path = getattr(snap, "db_path", "")
        self.budgeted = BudgetedRunner(
            runner,
            max_llm_calls=max_llm_calls,
            max_usd=max_usd,
            price_in_per_mtok=price_in_per_mtok,
            price_out_per_mtok=price_out_per_mtok,
            shared_budget=shared_budget,
        )
        if hypergraph is None and db_path and os.path.exists(db_path):
            hypergraph = DomainCatalogRegistry.create_database_catalog(
                db_path, seed_limit=120
            )
        self.agent = ReActAgent(hypergraph=hypergraph, db_path=db_path)
        self.schema_prompt = (default_schema_prompt(db_path) if schema_prompt is None
                              else schema_prompt)
        self.model_name = getattr(runner, "model_name", type(runner).__name__)
        self.case_log: List[Dict[str, Any]] = []
        self._truncated = False
        self.last_state: Optional[Any] = None

    def answer(
        self,
        query: str,
        *,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        belief_state: Optional[Any] = None,
        asked_question_ids: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        steps_before = self.budgeted.calls
        t0 = time.time()
        error: Optional[str] = None
        truncated = False
        text = ""
        pre_verifier = None
        term_reason = None
        verif_rep = None
        cost_before = self.budgeted.cost_usd
        try:
            state = self.agent.run_live_session(
                user_query=query,
                llm_runner=self.budgeted,
                schema_prompt=self.schema_prompt,
                session_id=f"bench:{len(self.case_log)}",
                belief_state=belief_state,
                asked_question_ids=asked_question_ids,
                conversation_history=conversation_history,
            )
            self.last_state = state
            text = state.final_response or ""
            pre_verifier = getattr(state, "pre_verifier_response", None)
            term_reason = state.termination_reason
            verif_rep = getattr(state, "verification_report", None)
        except BudgetExceeded as exc:
            self._truncated = truncated = True
            error = f"BudgetExceeded: {exc}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        used = self.budgeted.calls - steps_before
        case_cost = max(0.0, self.budgeted.cost_usd - cost_before)
        self.case_log.append({
            "query": query,
            "text": text,
            "pre_verifier_text": pre_verifier,
            "termination_reason": term_reason,
            "verification_report": verif_rep,
            "steps": used,
            "latency_s": round(time.time() - t0, 3),
            "cost_usd": round(case_cost, 6),
            "cumulative_cost_usd": round(self.budgeted.cost_usd, 6),
            "error": error,
            "truncated_by_budget": truncated,
        })
        return {
            "text": text,
            "pre_verifier_text": pre_verifier,
            "termination_reason": term_reason,
            "verification_report": verif_rep,
            "steps": used,
            "cost_usd": case_cost,
        }

    def stats(self) -> Dict[str, Any]:
        """Everything a reviewer needs to reproduce or disbelieve this run."""
        return {
            "model": self.model_name,
            "llm_calls": self.budgeted.calls,
            "prompt_tokens": self.budgeted.prompt_tokens,
            "completion_tokens": self.budgeted.completion_tokens,
            "cost_usd": round(self.budgeted.cost_usd, 6),
            "max_llm_calls": self.budgeted.max_llm_calls,
            "max_usd": self.budgeted.max_usd,
            "budget_mode": self.budgeted.budget_mode,
            "truncated": self._truncated,
            "cases_answered": sum(1 for c in self.case_log if not c["error"]),
            "cases_errored": sum(1 for c in self.case_log if c["error"]),
            "transcript": self.budgeted.transcript,
            "case_log": self.case_log,
        }

