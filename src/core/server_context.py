"""Process-wide SystemContext: domain registry, runtime registrations, and the shared runner/router state (ctx)."""

from fastapi import HTTPException
from typing import Dict, List, Any
import json
import os
import threading
from domain_catalog import DomainCatalogRegistry
from fast_router import FastCascadedRouter
from gateways.discord_adapter import DiscordGateway
from gateways.telegram_adapter import TelegramGateway
from hypergraph import KnowledgeHypergraph
from openrouter_client import OpenRouterAgentRunner
from react_agent import ReActAgent
from sanitizer import SQLSanitizer
from session_store import SQLiteSessionStore
from embedders import default_embedder
from source_currency import UNSET_CURRENCY
from source_lifecycle import SourceState
from vector_index import EMBEDDING_MODELS
from server_config import MARKETPLACE_DB_PATH, PROJECT_ROOT, SESSIONS_DB, env_setting



from server_bundle import DomainBundle, _InitCancelled  # noqa: F401
from server_init_jobs import InitJobsMixin
from server_bundle_build import BundleBuildMixin


class SystemContext(InitJobsMixin, BundleBuildMixin):
    def __init__(self):
        # Allow web server and deterministic neuro-symbolic routes/tests to boot
        # without an API key; LLM calls still fail closed in OpenRouterAgentRunner.
        self.runner = OpenRouterAgentRunner(require_key=False)
        # Dense channel: operator-selected embedding model (same
        # OpenRouter key as the chat runner). The channel is opt-in per
        # source -- built via POST /api/domains/{id}/vector_index.
        self.embedding_model = env_setting(
            "PYBUDDY_EMBEDDING_MODEL", "AINSA_EMBEDDING_MODEL", EMBEDDING_MODELS[0])
        self.sanitizer = SQLSanitizer()
        self.session_store = SQLiteSessionStore(SESSIONS_DB)
        # Embedder factory for the per-source SemanticIndex handles.
        # Tests MUST override this BEFORE registering a source so the
        # offline HashEmbedder is used instead of the local model.
        self.embedder_factory = default_embedder
        self.domains: Dict[str, DomainBundle] = {}
        # F33: asynchronous source-init jobs. Keyed by job id; a job's
        # life is queued -> running -> done|failed|cancelled and never
        # lies about which it is in.
        self.init_jobs: Dict[str, Dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()

        # Multi-domain catalogs (delegated to DomainCatalogRegistry)
        self.pc_hypergraph = DomainCatalogRegistry.create_pc_catalog()
        self.auto_hypergraph = DomainCatalogRegistry.create_auto_catalog()
        self.marketplace_hypergraph = DomainCatalogRegistry.create_marketplace_catalog(MARKETPLACE_DB_PATH)

        self.active_domain = self._bootstrap_sources()

        # Omnichannel Gateways (bound to active agent); the shared session
        # store is the persistence path that actually executes (F35).
        self.telegram_gateway = TelegramGateway(
            self.agent, runner=self.runner, schema_prompt=self.schema_prompt,
            session_store=self.session_store)
        self.discord_gateway = DiscordGateway(
            self.agent, runner=self.runner, schema_prompt=self.schema_prompt,
            session_store=self.session_store)

    def _bootstrap_sources(self) -> str:
        """
        F45: the default source set is configuration, not code. It lives in
        `data/sources.demo.json`, an explicitly labelled demo registry;
        production sources are registered at runtime through
        /api/register_domain. Descriptions absent from the registry are
        generated from the source's own profiled schema at engine init,
        not from baked-in domain prose. Returns the active domain id.
        """
        registry_path = os.path.abspath(
            os.path.join(PROJECT_ROOT, "data", "sources.demo.json"))
        graphs = {
            "pc_hardware": self.pc_hypergraph,
            "auto_parts": self.auto_hypergraph,
            "marketplace": self.marketplace_hypergraph,
        }
        default_domain = ""
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                registry = json.load(fh)
        except (OSError, ValueError):
            registry = {"sources": []}
        for entry in registry.get("sources", []):
            db_path = ""
            for cand in entry.get("db_candidates", []):
                p = os.path.abspath(os.path.join(PROJECT_ROOT, cand))
                if os.path.exists(p):
                    db_path = p
                    break
            if not db_path:
                continue
            bundle = DomainBundle(
                entry["domain_id"], entry["name"], db_path,
                graphs.get(entry.get("graph") or ""),
                entry.get("description", ""),
                currency=(entry["currency"] if "currency" in entry else UNSET_CURRENCY),
            )
            bundle.registered_via = "demo_registry"
            self.register_domain_bundle(bundle)
            if not default_domain:
                default_domain = entry["domain_id"]
        self._restore_runtime_registrations()
        env_active = env_setting("PYBUDDY_ACTIVE_DOMAIN", "AINSA_ACTIVE_DOMAIN")
        if env_active in self.domains:
            return env_active
        configured = registry.get("default_domain", "")
        if configured in self.domains:
            return configured
        return default_domain or next(iter(self.domains), "")

    def register_domain_bundle(self, bundle: DomainBundle) -> None:
        self.domains[bundle.domain_id] = bundle

    # ---- runtime registration persistence (T9) -----------------------

    def _runtime_registry_path(self) -> str:
        return env_setting("PYBUDDY_RUNTIME_REGISTRY", "AINSA_RUNTIME_REGISTRY") or os.path.join(
            PROJECT_ROOT, "data", "sources.runtime.json")

    def _source_root(self) -> str:
        return env_setting("PYBUDDY_SOURCE_ROOT", "AINSA_SOURCE_ROOT") or os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "data"))

    def _restore_runtime_registrations(self) -> None:
        """Re-register sources persisted by earlier /api/register_domain
        calls. The same source-root and existence checks are re-applied:
        a stale or out-of-root entry is skipped and recorded in
        `skipped_runtime_sources`, never silently served."""
        self.skipped_runtime_sources: List[Dict[str, str]] = []
        try:
            with open(self._runtime_registry_path(), "r",
                      encoding="utf-8") as fh:
                entries = json.load(fh).get("sources", [])
        except (OSError, ValueError):
            return
        root_real = os.path.realpath(self._source_root())
        for entry in entries:
            d_id = entry.get("domain_id", "")
            real = os.path.realpath(os.path.abspath(entry.get("db_path", "")))
            why = None
            if not d_id or d_id in self.domains:
                why = "duplicate_or_invalid_id"
            elif not real.startswith(root_real + os.sep):
                why = "outside_source_root"
            elif not os.path.exists(real):
                why = "file_missing"
            elif os.path.splitext(real)[1].lower() not in (
                    ".db", ".sqlite", ".sqlite3"):
                why = "not_sqlite"
            if why:
                self.skipped_runtime_sources.append(
                    {"domain_id": d_id, "reason": why})
                continue
            bundle = DomainBundle(
                d_id, entry.get("name") or d_id, real, None,
                entry.get("description", ""),
                currency=(entry["currency"] if "currency" in entry else UNSET_CURRENCY),
            )
            bundle.registered_via = "runtime"
            bundle.verified_rule_ids = {
                str(r) for r in (entry.get("verified_rule_ids") or ()) if str(r)
            }
            self.register_domain_bundle(bundle)

    def persist_runtime_registration(self, bundle: DomainBundle) -> None:
        """Upsert a runtime registration atomically (tmp + replace)."""
        path = self._runtime_registry_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {"sources": []}
        data["sources"] = [e for e in data.get("sources", [])
                           if e.get("domain_id") != bundle.domain_id]
        data["sources"].append({
            "domain_id": bundle.domain_id,
            "name": bundle.name,
            "db_path": bundle.db_path,
            "description": bundle.description,
            "currency": bundle.currency,
            "verified_rule_ids": sorted(getattr(bundle, "verified_rule_ids", ()) or ()),
        })
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def remove_runtime_registration(self, domain_id: str) -> bool:
        """Remove a persisted runtime registration. Returns False when
        the registry has no such entry."""
        path = self._runtime_registry_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return False
        kept = [e for e in data.get("sources", [])
                if e.get("domain_id") != domain_id]
        if len(kept) == len(data.get("sources", [])):
            return False
        data["sources"] = kept
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True

    def get_domain_bundle(self, domain_id: str) -> DomainBundle:
        if domain_id not in self.domains:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown domain {domain_id!r}. Available: {', '.join(self.domains.keys())}."
            )
        bundle = self.domains[domain_id]
        self._ensure_bundle_ready(bundle)
        if bundle.state == SourceState.FAILED:
            # F33: fail closed. A request for a missing registered source
            # gets an explicit unavailable error -- never answers served
            # from a different source's database.
            raise HTTPException(
                status_code=503,
                detail=f"Source for domain {domain_id!r} is unavailable: {bundle.state_reason}"
            )
        return bundle

    def _ensure_bundle_ready(self, bundle: DomainBundle) -> None:
        """Single-flight, version-pinned lazy initialization.

        The engine is (re)built when it does not exist or when the source
        file's version moved since the components were built -- a request
        must never serve parts pinned to different source snapshots.
        """
        if self._bundle_needs_init(bundle):
            with bundle._init_lock:
                if self._bundle_needs_init(bundle):
                    self._init_bundle_engine(bundle)

    # ------------------------------------------------------------------
    # F33: asynchronous initialization jobs
    # ------------------------------------------------------------------
    def get_agent_for_domain(self, domain_id: str) -> ReActAgent:
        return self.get_domain_bundle(domain_id).agent

    def get_router_for_domain(self, domain_id: str) -> FastCascadedRouter:
        return self.get_domain_bundle(domain_id).fast_router

    def get_hypergraph_for_domain(self, domain_id: str) -> KnowledgeHypergraph:
        return self.get_domain_bundle(domain_id).hypergraph

    @property
    def agent(self) -> ReActAgent:
        return self.get_agent_for_domain(self.active_domain)

    @property
    def fast_router(self) -> FastCascadedRouter:
        return self.get_router_for_domain(self.active_domain)

    @property
    def snapshot(self) -> Any:
        bundle = self.get_domain_bundle(self.active_domain)
        return bundle.snapshot

    @property
    def schema_prompt(self) -> str:
        bundle = self.get_domain_bundle(self.active_domain)
        return bundle.schema_prompt

ctx = SystemContext()
