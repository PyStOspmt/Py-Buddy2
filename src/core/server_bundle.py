"""Domain bundle record and the cooperative init-cancel signal."""

from typing import Dict, Any, Optional, Set
import threading
from fast_router import FastCascadedRouter
from hypergraph import KnowledgeHypergraph
from ontograph import OntoGraph
from profiler import DomainAgnosticProfiler
from react_agent import ReActAgent
from source_currency import UNSET_CURRENCY, resolve_source_currency
from source_lifecycle import SourceState




class _InitCancelled(Exception):
    """Raised at an init stage boundary when the job's cancel flag was
    set. Distinct from a build failure: nothing was attempted-and-failed,
    the work was deliberately unwound (F33)."""

class DomainBundle:
    def __init__(
        self,
        domain_id: str,
        name: str,
        db_path: Optional[str] = None,
        hypergraph: Optional[KnowledgeHypergraph] = None,
        description: str = "",
        currency: Any = UNSET_CURRENCY,
    ):
        self.domain_id = domain_id
        self.name = name
        self.db_path = db_path
        self.currency: Optional[str] = resolve_source_currency(db_path, currency)
        self.hypergraph = hypergraph or KnowledgeHypergraph()
        self.description = description
        self.profiler: Optional[DomainAgnosticProfiler] = None
        self.snapshot: Optional[Any] = None
        self.schema_prompt: str = ""
        self.agent: Optional[ReActAgent] = None
        self.fast_router: Optional[FastCascadedRouter] = None
        # Materialized FTS5 sidecar for the lexical evidence channels;
        # None until built (or when the source moves past its stamp).
        self.channel_index = None
        self.ontograph: Optional[OntoGraph] = None
        # Registration provenance: "demo_registry" (bootstrap sources),
        # "runtime" (/api/register_domain, persisted), "code" (created
        # in-process -- not removable through the API).
        self.registered_via: str = "code"
        # F33: explicit lifecycle. `source_version` pins the engine to the
        # file it was built from; `_init_lock` makes lazy init single-flight.
        self.state: SourceState = SourceState.PENDING
        self.state_reason: str = ""
        self.source_version: Optional[tuple] = None
        self._init_lock = threading.Lock()
        # F33: cooperative cancellation for async init jobs. Checked at
        # stage boundaries inside _init_bundle_engine; between checks a
        # running stage is allowed to finish.
        self._cancel = threading.Event()
        # P1.2: behavioural fiduciary probe result for this source version
        # (CERTIFIED/VIOLATION/UNVERIFIED/NO_COMMERCIAL_CHANNEL + details),
        # populated during init; None until the probe ran.
        self.fiduciary: Optional[Dict[str, Any]] = None
        # P1.3: machine-checkable attestation binding the probe result to
        # the source bytes (sha256) + policy digest; verifiable offline
        # via attestation.verify_attestation.
        self.attestation: Optional[Dict[str, Any]] = None
        # T5: lazy FTS5 trigram index for catalogue search (built on first
        # qualifying search, version-pinned to the source file).
        self.search_index: Optional[Any] = None
        # Retrieval-Fabric semantic index lifecycle handle
        # (SemanticIndex); attached at engine init, lazily at route
        # level for degraded bundles.
        self.semantic: Optional[Any] = None
        # Opt-in: the model may propose bindings for roles the
        # deterministic projection could not resolve; every proposal is
        # verified against data before it resolves anything.
        self.llm_assist: bool = False
        self.llm_mapping_report: Optional[Dict[str, Any]] = None
        # Operator-verified rule_ids (candidate -> verified across engine rebuilds).
        self.verified_rule_ids: Set[str] = set()
