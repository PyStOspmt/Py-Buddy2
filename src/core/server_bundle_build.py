"""Engine-build mixin for SystemContext: profiling, catalogue projection and engine assembly."""

from typing import Any, Optional
import os
import sqlite3
from server_bundle import DomainBundle
from active_inference import ActiveInferenceEngine, DomainQuestionBankInductor
from domain_catalog import DomainCatalogRegistry
from fast_router import FastCascadedRouter
from ontograph import OntoGraph
from profiler import DomainAgnosticProfiler
from react_agent import ReActAgent
from source_lifecycle import SourceState, source_version
import attestation
import fiduciary_probe




class BundleBuildMixin:
    def _init_bundle_engine(
        self,
        bundle: DomainBundle,
        progress: Optional[Any] = None,
        trace_listener=None,
    ) -> None:
        report = progress or (lambda _stage: None)
        bundle.source_version = source_version(bundle.db_path)

        if not bundle.db_path:
            # Graph-only registration: honestly degraded, never pointed at
            # another source's database (the old effective_db fallback did).
            bundle.state = SourceState.DEGRADED
            bundle.state_reason = "no database file registered; graph-only bundle"
        elif not os.path.exists(bundle.db_path):
            bundle.state = SourceState.FAILED
            bundle.state_reason = f"registered source file is missing: {bundle.db_path}"
            bundle.fast_router = None
            bundle.agent = None
            return
        else:
            bundle.state = SourceState.READY
            bundle.state_reason = ""

        base_prompt = (
            f"АКТИВНИЙ КАТАЛОГ: {bundle.name} (ID: {bundle.domain_id})\n"
            f"Опис предметної області: {bundle.description or 'не надано; визначається профілем джерела'}\n\n"
        )
        report("profiling")
        self._check_cancel(bundle, "profiling")
        if bundle.db_path and os.path.exists(bundle.db_path):
            try:
                bundle.profiler = DomainAgnosticProfiler(bundle.db_path)
                bundle.snapshot = bundle.profiler.profile_database()
                if not bundle.description:
                    # F45: a source that came without a stated description
                    # gets one generated from its own profile -- source
                    # facts (table/row counts), never invented domain prose.
                    bundle.description = self._describe_from_profile(bundle)
                    base_prompt = (
                        f"АКТИВНИЙ КАТАЛОГ: {bundle.name} (ID: {bundle.domain_id})\n"
                        f"Опис предметної області: {bundle.description}\n\n"
                    )
                schema_ctx = DomainAgnosticProfiler.generate_compact_schema_context(bundle.snapshot.to_dict())
                bundle.schema_prompt = base_prompt + schema_ctx
                # T7: EAV tables detected STRUCTURALLY (renamed/FK-less)
                # are absent from the profiler's name-hinted projection
                # section, so the agent could not query their attribute
                # space. Append them as an explicit queryable axis.
                try:
                    from source_projection import build_source_projection
                    proj = build_source_projection(
                        bundle.db_path, trace_listener=trace_listener)
                    # Materialized channel index: turns the per-term
                    # O(rows) LIKE sweeps into trigram-indexed lookups.
                    # Sidecar file, rebuilt when the source stamp moves.
                    try:
                        report("channel_index")
                        from channel_index import get_index
                        bundle.channel_index = get_index(
                            bundle.db_path, proj)
                    except Exception:
                        bundle.channel_index = None
                    # Canonical role resolution -- the model should not
                    # have to re-derive from column names what the
                    # projection already proved. Unresolved roles are
                    # listed as such so the model does not guess them.
                    role_lines = [
                        f"- entity-таблиця: `{proj.entity_table}`"
                    ] if proj.entity_table else []
                    for role, res in proj.roles.items():
                        if res.resolved:
                            tgt = f"`{res.table}.{res.column}`"
                        elif role in proj.axis_roles:
                            ar = proj.axis_roles[role]
                            tgt = (f"вісь EAV `{ar['table']}` "
                                   f"({ar['key_column']}="
                                   f"'{ar['axis']}', покриття "
                                   f"{ar['coverage']:.0%})")
                        else:
                            tgt = "НЕ РОЗВ'ЯЗАНО"
                        role_lines.append(f"- `{role}` → {tgt}")
                    if role_lines:
                        bundle.schema_prompt += (
                            "\n\n### Канонічні ролі (виявлено адаптацією; "
                            "НЕ РОЗВ'ЯЗАНО означає: колонки немає або "
                            "докази неоднозначні — не вгадувати)\n"
                            + "\n".join(role_lines))
                    # Declared foreign keys are real join edges -- surface
                    # them so the model joins real relations, not guesses.
                    try:
                        from source_projection import quote_ident
                        fk_conn = sqlite3.connect(
                            f"file:{bundle.db_path}?mode=ro", uri=True)
                        fk_lines = []
                        for tname in bundle.snapshot.tables:
                            for fk in fk_conn.execute(
                                    f"PRAGMA foreign_key_list({quote_ident(tname)})"):
                                fk_lines.append(
                                    f"- `{tname}.{fk[3]}` → `{fk[2]}.{fk[4]}`")
                        fk_conn.close()
                        if fk_lines:
                            bundle.schema_prompt += (
                                "\n\n### Оголошені зв'язки (FK)\n"
                                + "\n".join(fk_lines))
                    except Exception:
                        pass  # FK listing is additive; its absence is benign
                    # Optional advisory layer: unresolved roles may be
                    # proposed by the model and are then verified against
                    # the role's own structural predicates. Proposals that
                    # fail verification are recorded as vetoed candidates,
                    # so the assist can never manufacture a mapping.
                    if bundle.llm_assist and getattr(self.runner, "api_key", None):
                        bundle.llm_mapping_report = self._llm_assist_projection(
                            bundle, proj)
                    eav_lines = []
                    for tbl, meta in proj.eav_tables.items():
                        if not meta.get("key_column"):
                            continue
                        keys = meta.get("discovered_attributes") or []
                        total = meta.get("keys_total")
                        shown = ", ".join(str(k) for k in keys[:12])
                        scope = (f"{min(12, len(keys))} з {total}"
                                 if total is not None else "приклади")
                        eav_lines.append(
                            f"- `{tbl}`: entity `{meta['entity_fk']}`, "
                            f"key `{meta['key_column']}`, "
                            f"value `{meta['value_column']}`; "
                            f"приклади ключів ({scope}): {shown}")
                    if eav_lines:
                        bundle.schema_prompt += (
                            "\n\n### Віртуальні атрибути (EAV): запити через "
                            "JOIN/EXISTS на entity-колонку\n"
                            + "\n".join(eav_lines))
                    # Text-containment links: a dimension whose labels
                    # live INSIDE entity text (e.g. size '42' in a
                    # product name) -- a real relation with measured
                    # coverage, not a guess.
                    tl_lines = [
                        f"- `{l['dim_table']}.{l['label_column']}` лейбли "
                        f"всередині тексту `{l['entity_table']}` "
                        f"(покриття {l['coverage']:.0%}, словник "
                        f"{l['vocab']})"
                        for l in getattr(proj, "text_links", []) or []]
                    if tl_lines:
                        bundle.schema_prompt += (
                            "\n\n### Текстові зв'язки (лейбл довідника — "
                            "токен у тексті сутності; FK відсутній)\n"
                            + "\n".join(tl_lines))
                except Exception:
                    # EAV axis listing is additive; failure must not
                    # degrade the rest of the schema prompt.
                    pass
            except Exception as exc:
                # Preparation partially failed: the domain stays usable via
                # SQL tools, but the degradation is recorded, not hidden.
                self._mark_degraded(bundle, f"profiling failed: {type(exc).__name__}")
                bundle.schema_prompt = base_prompt + \
                    f"Schema for {bundle.domain_id} database ({bundle.db_path})"
        else:
            bundle.schema_prompt = base_prompt + \
                f"In-memory catalog for domain {bundle.domain_id}"

        # If hypergraph has 0 nodes and db_path exists, dynamically seed it
        report("catalogue seeding")
        self._check_cancel(bundle, "catalogue seeding")
        if len(bundle.hypergraph.nodes) == 0 and bundle.db_path and os.path.exists(bundle.db_path):
            try:
                bundle.hypergraph = DomainCatalogRegistry.create_database_catalog(bundle.db_path, seed_limit=120)
            except Exception as exc:
                self._mark_degraded(bundle, f"catalogue seeding failed: {type(exc).__name__}")
        if bundle.hypergraph is not None and getattr(bundle, "verified_rule_ids", None):
            for rid in list(bundle.verified_rule_ids):
                bundle.hypergraph.set_constraint_status(rid, "verified")

        report("ontology induction")
        self._check_cancel(bundle, "ontology induction")
        if not bundle.ontograph and bundle.db_path and os.path.exists(bundle.db_path):
            try:
                bundle.ontograph = OntoGraph(bundle.db_path, openrouter_runner=self.runner)
            except Exception as exc:
                self._mark_degraded(bundle, f"ontology induction failed: {type(exc).__name__}")

        report("question bank")
        self._check_cancel(bundle, "question bank")
        # Domain-adapted Active Inference question bank induction
        question_bank = None
        if bundle.ontograph and bundle.ontograph.root:
            top_cats = [c.name for c in bundle.ontograph.root.children[:15]]
            top_params = bundle.ontograph.root.parameters
            question_bank = DomainQuestionBankInductor.induce_question_bank(
                domain_name=bundle.name or bundle.domain_id,
                categories=top_cats,
                parameters=top_params,
                runner=self.runner,
                db_path=bundle.db_path
            )
        report("fiduciary probe")
        self._check_cancel(bundle, "fiduciary probe")
        if bundle.db_path and os.path.exists(bundle.db_path):
            try:
                # P1.2: behavioural counterfactual -- the production
                # ordering is recomputed under a permuted commercial
                # state; any rank movement is a VIOLATION, an absent
                # channel is NO_COMMERCIAL_CHANNEL, never guessed.
                bundle.fiduciary = fiduciary_probe.probe_source(bundle.db_path)
            except Exception as exc:
                bundle.fiduciary = {
                    "status": "UNVERIFIED",
                    "probe": "fiduciary_probe/1.0",
                    "reason": f"probe failed: {type(exc).__name__}",
                }
            try:
                # P1.3: bind the probe result to the source bytes and the
                # policy it was computed under, so an independent verifier
                # can replay the same audit -- not a self-assessment.
                bundle.attestation = attestation.issue_attestation(
                    bundle.db_path, bundle.fiduciary)
            except Exception as exc:
                bundle.attestation = {
                    "attestation_version": attestation.ATTESTATION_VERSION,
                    "error": f"attestation failed: {type(exc).__name__}",
                }

        report("engine build")
        ai_engine = ActiveInferenceEngine(question_bank=question_bank)
        bundle.agent = ReActAgent(
            db_path=bundle.db_path,
            hypergraph=bundle.hypergraph,
            ontograph=bundle.ontograph,
            active_inference=ai_engine,
            currency=bundle.currency,
        )
        bundle.fast_router = FastCascadedRouter(
            db_path=bundle.db_path,
            sanitizer=self.sanitizer,
            react_agent=bundle.agent,
            llm_runner=self.runner,
            schema_prompt=bundle.schema_prompt
        )
        if bundle.semantic is None:
            from semantic_service import SemanticIndex
            gen = getattr(self.runner, "generate", None)
            bundle.semantic = SemanticIndex(
                bundle.domain_id, bundle.db_path,
                embedder_factory=self.embedder_factory,
                llm_generate=(
                    (lambda p: self.runner.generate(p, temperature=0.3))
                    if callable(gen) else None))
        bundle.agent.semantic_index = bundle.semantic

    def _llm_assist_projection(self, bundle: DomainBundle, proj) -> dict:
        """
        Ask the model to propose columns for roles the deterministic
        projection left unresolved, then verify each proposal against the
        role's structural predicate. The model proposes; the data decides.
        Returns a provenance report surfaced verbatim via /inspection.
        """
        # Agentic path: the conductor loop lets the model propose
        # structural hypotheses (undeclared links, EAV tables, role
        # bindings) -- every one verified against the real data by the
        # deterministic tool layer before it touches the projection.
        from conductor import run_conductor
        report: dict = {"conductor": True}
        cond = run_conductor(bundle.db_path, proj, self.runner)
        report.update(cond)
        return report
