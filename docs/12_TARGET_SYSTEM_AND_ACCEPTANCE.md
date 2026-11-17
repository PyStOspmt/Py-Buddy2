# 12 · Target system, product goals, and acceptance contract

> [Українська версія / Ukrainian version](12_TARGET_SYSTEM_AND_ACCEPTANCE.uk.md)

**Date:** 2026-10-04. **Status:** target specification; not a description of completed functionality.

**Companion documents:** [current-state findings and remediation](11_CURRENT_STATE_AUDIT_AND_REMEDIATION.md), [mandatory development instructions](../AGENTS.md).

## 1. The idea we must preserve

Build a hosted, domain-independent consultation system over a customer's authorized data and knowledge sources. The customer connects databases or supplies files, enables a website chat widget, and the service automatically discovers usable structure, maintains an evidence-linked map/index, and helps visitors find, compare, understand, or select relevant information.

The same core must serve a shop, library, forum, documentation collection, or other knowledge environment. It must not contain a shop's brands, a domain vocabulary, fixture product names, or customer-specific schema logic that makes the demonstration appear adaptive.

**The intended customer experience:** connect → automatic preparation with visible progress/quality → ready chat → automatic updates. No per-customer programming, hand-maintained synonym lists, manual category repair, or prewritten answers to evaluation questions.

**The internal meaning of “magic”:** genuine discovery, bounded computation, explicit uncertainty, and independently verifiable results—not invisible guessing. If a source lacks critical knowledge, the system must disclose that limitation. It must not make the customer's missing knowledge look complete.

## 2. Product promise versus impossible guarantees

### 2.1 Promise we can engineer and test

Within declared connector/format support and a measured resource envelope, the system adapts without customer/domain code, accounts for all ingested records, retrieves authorized evidence, and produces useful responses whose factual claims and derivations are bound to that evidence. Unknown, contradictory, stale, unsupported, or incompletely searched information remains explicitly identified.

### 2.2 Promises we must not make

- Literal support for every storage engine/file encoding without a connector/parser.
- Unlimited dataset size, zero preparation cost, or constant latency on arbitrary workloads.
- Recovery of the correct meaning of every ambiguous column from data alone.
- Discovery of all physical, clinical, legal, or semantic rules from co-occurrence.
- Proof that source data is true merely because it was retrieved.
- “No conflicts found” as proof of safety or fitment.
- A finite test run as a universal zero-hallucination theorem.
- A complete materialized graph of every fact, relation, and possible combination as a prerequisite for useful search.

Two identical observed tables can describe different real-world relationships. An algorithm cannot distinguish those worlds without additional evidence. Dirty values may be correct anomalies or errors; numeric ranges cannot tell us which. The right response is evidence-aware uncertainty, not a hidden domain dictionary.

**Automatic onboarding is a goal; unknowable semantics is a limitation to expose, not a justification for fabricated certainty.** A degraded source can still support useful textual/structural retrieval without supporting price, stock, clinical, or compatibility claims.

## 3. Definition of the finished system

The system is ready for its advertised scope only when all of the following are demonstrated together:

1. New supported sources attach without changes to production domain logic.
2. Different schemas, languages, data quality, and noncommerce content are handled through the same adaptation contracts.
3. The complete source is reachable by retrieval; bounded samples for discovery are not falsely presented as complete coverage.
4. Source facts, identities, offers, documents, relationships, uncertainty, conflicts, and derivations retain provenance and versions.
5. The model cannot manufacture a source fact through text, query expressions, tool arguments, or recommendation metadata.
6. Task completion, evidence verification, and useful delivery are separate measured outcomes.
7. Multiple tenants and visitors are isolated in data access, sessions, evidence, model settings, resource usage, and telemetry.
8. Preparation, queries, updates, cancellation, failure recovery, and deletion have explicit observable lifecycles.
9. A real embeddable widget works on an external site with the proper authorization and origin constraints.
10. Independent held-out evaluations and load trials establish the quality/scale envelope; the scientific contribution is narrowly and honestly supported.

No single component, graph picture, or number of passing unit tests establishes this definition.

## 4. Terminology and boundaries

| Concept | Meaning |
|---|---|
| Tenant | Customer security/resource boundary |
| Knowledge space | Authorized collection of source versions available to a particular assistant |
| Source | A connected database/export/document collection with identity, ownership, format, and policy |
| Source version | Immutable logical snapshot used for reproducible retrieval and evidence checks |
| Entity | A source-backed object/topic/person/record; not necessarily a product |
| Offer or observation | A seller-, time-, variant-, or context-specific assertion; distinct from entity identity |
| Assertion | Entity/predicate/value with units, context, source location, and uncertainty |
| Hypothesis | A proposed mapping, entity link, concept, or constraint; not yet a factual rule |
| Verified rule | A rule with documented applicability, trusted evidence, independent validation, and version |
| Claim | A factual proposition proposed for the client answer |
| Derivation | An allowed operation whose exact inputs and result can be checked |
| Task result | Outcome of search/comparison/question/selection, including scope and completeness |
| Evidence map | Retrieval/relationship indexes plus assertion provenance; not necessarily one graph engine |

Storage connectors implement engine-specific access. The generic adaptation core discovers customer structure. Domain policy and authoritative external knowledge, when required, are explicit connected sources—not secret facts in core code.

## 5. End-to-end architecture contract

All items in this section are proposed requirements. Existing implementation coverage is recorded in the [audit](11_CURRENT_STATE_AUDIT_AND_REMEDIATION.md).

### 5.1 Connection and access control

- Authenticate source administration; authorize tenant/knowledge-space ownership.
- Website widget callers receive narrowly scoped public capability tokens, not source administration credentials.
- Use read-only least-privilege database access; separate credentials and data per tenant.
- Constrain filesystem/network connectors, upload sizes, parsing cost, source counts, model spend, and destination providers.
- Apply table/column/document sensitivity policy before sampling, indexing, telemetry, and external-model transmission.
- Persist connection identity/configuration safely; never depend on mutable global active-domain/model settings.

**Output:** registered source and preparation job, not an immediate unconditional “ready” assertion.

### 5.2 Discovery and quality profiling

Discover tables, columns, keys, structural relationships, value types/shapes, unit dimensions, distributions, missingness, duplicate/identity ambiguity, EAV candidates, text content, and source capabilities.

Role discovery uses structural/value evidence and documented probabilistic or learned hypotheses. Names can contribute to semantic interpretation only as uncertain evidence; hardcoded business-role/name dictionaries cannot determine ground truth. The method must survive renamed/anonymized business identifiers on held-out tests.

Sampling is bounded, recorded, and representative of relevant strata. Report which records/attributes were inspected and what uncertainty remains. A high-frequency placeholder does not become a meaningful category merely because it is common.

**Output:** versioned schema/quality profile, mapping hypotheses, available capabilities, and unresolved areas. A source with no prices is not “broken retail”; it simply lacks a price-query capability.

### 5.3 Canonical evidence layer

Represent:

- Source-scoped stable entity/record identities and separate offer/observation identities.
- Typed values with known units/dimensions/currency or explicitly unknown units.
- Source file/table/row/field or document/span locations and logical source version.
- Original raw evidence and transformation/normalization lineage.
- Distinct absent, unparseable, uncertain, contradictory, invalid, and known values.
- Confidence and verification status of mappings, cross-source entity links, and rules.
- Timestamp, freshness, sensitivity, ownership, and access policy.

No silently invented price, performance, rating, delivery time, availability, relationship, or identity. Uniform defaults are still fabrications if rendered as source facts. Default algorithm parameters are allowed only when documented, validated, and not confused with observed data.

**Output:** a common evidence interface consumed by retrieval, graph/constraints, ranking, response verification, and federation—not just a prompt.

### 5.4 Map/index construction

Choose the representation from source/task needs:

- Relational/indexed access for structured predicates and exact aggregates.
- Lexical/dense/hybrid retrieval for names, descriptions, documents, and semantic discovery.
- Relationship graphs for evidence-linked traversal and multi-hop tasks.
- Implicit constraint evaluators for applicable assemblies or other relationships.
- Taxonomy/facets for navigation where supported by data.

All derived edges, labels, clusters, and summaries retain evidence and confidence. LLM-proposed relations are candidates until checked; structural correlation alone is not a physical law. Unsupported relation classes remain unknown.

Do not materialize all product combinations or every formal concept merely to display a graph. Budget construction, index bytes, and candidate expansion; support incremental updates. Sampling may guide discovery but must not make the remainder unreachable.

### 5.5 Query planning and retrieval

Identify the user's task and required evidence: literal/semantic search, lookup, comparison, explanation, aggregation, contextual recommendation, verified relation check, or clarification.

- Preserve entities, predicates, hard requirements, soft preferences, negations, currency/unit context, and safety policy.
- Resolve ambiguous entities to candidates and uncertainty; never silently take the first substring match as unique identity.
- Use only capabilities actually supported by source evidence.
- Compile safe typed query operations through adapters; arbitrary model-generated expressions do not become factual columns.
- Enforce runtime, output-size, join/expansion, and access budgets outside the model.
- Return source-versioned task results, required-field coverage, search scope, ambiguity, and exact/approximate status.
- A negative answer states what was searched. Limited retrieval cannot prove global absence.

Domain-specific aliases may be learned from authorized source evidence with verification and versioning; they may not be secretly inserted into core code to satisfy an evaluation case.

### 5.6 Consultation and response verification

The model may propose a plan, interpretation, explanation, or claim set. It cannot declare its own facts verified.

Before delivery:

1. Bind each factual claim to the correct entity/offer/attribute/value and source version.
2. Verify all derivations against their exact input evidence, operation, units, and result.
3. Check access permissions, freshness, conflicts, required qualifiers, and completeness.
4. Confirm that the output fulfills the task, not merely that some mentioned data is real.
5. Render only the verified claims and explicit uncertainty/status.
6. Make response status, trace, badge, and evaluation agree.

Use controlled rendering or a validated structured realization contract where strong mechanical guarantees are required. A regex over unrestricted prose cannot certify every statement. A declaration of “grounded” must identify exactly which claims and semantics were checked.

The response outcome distinguishes fulfilled, partially fulfilled, needs clarification, evidence conflict, unsupported capability, unknown due to missing evidence, cancelled/timeout, protocol/provider failure, and safety-blocked. Nonfactual greetings are not labelled factual verification. Explicit failure remains visible; it is not reformatted into success.

### 5.7 Preferences and relaxation

- Persist dialogue/context and preference evidence by authorized session, not by process-global state.
- Define the model and update rule precisely; if it is not Bayesian information gain, do not call it that.
- Ask questions with measured utility/information benefit, not because a bank exists.
- Derive ranking axes from actual available data and user goals; price/performance are not universal axes.
- Distinguish immutable safety/identity/access requirements from negotiable preferences.
- Propose soft-constraint relaxation explicitly, including changed bound, reason, tradeoff, and supporting evidence.
- A truncated search does not establish infeasibility or minimality.
- Never relax medical contraindication, physical safety, access permissions, or product identity to make an answer look useful.

### 5.8 Federation

Multiple sources may be queried in one knowledge space, but must retain source identity, access boundaries, source versions, independent offers, currencies, observation times, and conflicting assertions.

Cross-source entity equivalence is a hypothesis until supported. Preserve alternative assertions; conflict resolution policy must be explicit and traceable, not “first source wins.” A source priority may choose a displayed assertion with a qualifier, but cannot erase evidence that another source disagrees.

Derived cross-source answers expose which sources contributed and what was unresolved. Independent source-local querying is a useful staged capability; do not call it semantic federation until cross-source semantics and conflicts work.

### 5.9 Updates, operations, and erasure

Persistent idempotent preparation jobs progress through queued, discovering, validating, indexing, ready, degraded, failed, cancelled, or deleting states. No status is assigned merely because an endpoint returned successfully.

- Atomic snapshot publication and request pinning to a knowledge version.
- Cache keys include content/version, algorithm/configuration, and scope—not only a filesystem path.
- Database/file replacement, insert/update/delete, renamed schema, and changed rules invalidate affected indexes and assertions.
- Restart-safe progress, retry/backoff, cancellation, diagnostics, rollback, and quota enforcement.
- Customer deletion removes credentials, source content, derived indexes, caches, relevant telemetry, and provider-retained artifacts to the extent allowed by the chosen provider policy.
- Backups are controlled and subject to documented retention/erasure policy.

## 6. Hard invariants

These are requirements to enforce, not current measured achievements.

1. **Source preservation:** every received record is represented, explicitly excluded with reason, quarantined, or rejected. Counts and reasons reconcile; silent dropping is forbidden.
2. **Claim soundness:** each delivered factual claim has valid authorized evidence or a verified derivation from such evidence.
3. **Entity binding:** a real number somewhere in a result set does not validate its attribution to a different entity/offer.
4. **Uncertainty preservation:** unknown, unparseable, contradictory, and incomplete are not success states by default.
5. **Scope isolation:** evidence, sessions, caches, jobs, credentials, and costs cannot cross tenant/request boundaries without explicit authorized sharing.
6. **Task integrity:** fulfilling the wrong task, an empty introduction, or a data-free internal monologue is not successful consultation.
7. **Constraint integrity:** no safety/identity/access requirement is relaxed to increase apparent coverage.
8. **Observation integrity:** metrics, usage, statuses, timings, counts, and certificates derive from actual execution or are explicitly unavailable/estimated.
9. **Adaptation integrity:** new source/domain support cannot require secret production dictionaries, fixture exceptions, or manual benchmark data cleaning.
10. **Update integrity:** evidence refers to its exact source version; obsolete assertions cannot silently certify current answers.

The claim-soundness invariant proves correspondence to the authorized source under stated transformation assumptions. It does not prove the source is medically, physically, or legally correct.

## 7. Goals and measurement

**All numeric thresholds below are proposed acceptance targets, not observed results.** They must be approved and frozen before the held-out evaluation. Tightening a target is legitimate; lowering it after seeing failures requires a documented scope change and new evaluation, not quiet rewriting.

| Goal | Measurement | Proposed initial acceptance |
|---|---|---|
| G1 Zero domain-code onboarding | Production diff, connector/config audit, no fixture-specific dictionaries after freeze | Zero per-customer/domain production changes for supported held-out sources |
| G2 Correct adaptation | Independently labelled role/concept/entity/rule predictions; precision plus coverage, per domain | Role/concept precision ≥95%; useful adaptation coverage ≥80% on the declared polygon; unresolved portion explicit |
| G3 Evidence correspondence | Independent claim checker, exact entity/offer/value binding, source version and derivation coverage | Every delivered claim has a checkable evidence path; zero unsupported claims in the release acceptance set |
| G4 Useful consultation | Independently labelled required facts, relevance, correct constraints, appropriate unknowns, fulfilled task | ≥90% task success on supported held-out cases, reported separately from safety/unknown outcomes |
| G5 Honest uncertainty and safety | False affirmative compatibility/safety, missing-evidence and contradiction mutations | Zero unsafe affirmative decisions in the release safety set; no unknown silently presented as verified |
| G6 Dirty-data accounting | Input versus represented/excluded/quarantined counts; defect-class behavior | Full count reconciliation; zero silent losses or fabricated repairs |
| G7 Federation fidelity | Independently labelled links/conflicts and source-local claim provenance | No unsupported cross-source joins/merges; conflict recall and precision both reported |
| G8 Isolation and authorization | Concurrent tenant/session tests, hostile widget/admin calls, sensitive-field policy | No cross-tenant/session unauthorized reads, evidence reuse, model interference, or telemetry leakage |
| G9 Update fidelity | Insert/update/delete/schema-change/restart tests, fixed query set before/after | No stale certification; atomic version transitions; documented lag within the agreed service target |
| G10 Measured scale | Same semantic query workload at growing size/diversity, pinned hardware/config, build/query costs | Pass the published resource/SLO envelope, not an unspecified “any size” claim |
| G11 Reproducibility | Clean checkout, locked dependencies, hashes/seeds/commands/raw evidence | Independent clean-machine reproduction of each released quantitative claim |
| G12 Scientific contribution | Closest prior art, independent labels, fair baselines, preregistered ablations | A specific method with supported advantage or a clearly reported negative result; no unsupported priority claim |

Precision and coverage must be jointly evaluated: answering nothing is not success. Correct abstention is a separate important outcome, not a tool for inflating supported-task success. False conflict merges and missed conflicts require different denominators. A zero-event finite test needs sample count and a confidence bound; it is not proof that future failure probability is zero.

### 7.1 Adaptation measures

Expand earlier [adaptive-fabric goals](09_ADAPTIVE_DATA_FABRIC.md:143) into independently labelled metrics:

- Role/concept/entity-link precision, recall, and abstention, split by domain/schema/source quality.
- Mapping coverage over semantically relevant fields and task coverage—not merely count of any assigned labels.
- Constraint applicability, acquisition precision/recall, false rejection/acceptance, and rule-evidence quality.
- Unknown-unit and invalid-value handling; normalization correctness; conflict detection precision and recall.
- Amount of customer/domain code/config/manual repair, with explicit distinction from engine connectors.
- Retrieval reachability beyond induction samples and recall over the full authorized source.

### 7.2 Answer measures

Report claim precision, evidence/derivation coverage, entity–offer binding, required-fact recall, relevance, constraint satisfaction, supported negative answers, appropriate abstention, and task success. Do not combine them into a flattering single score that masks an unsafe failure.

Latency: request wall clock, preparation queue/build time, retrieval, model, verification, serialization; warm/cold p50/p95/p99. Cost: provider tokens/calls/retries, real or explicitly estimated pricing, CPU/RAM/disk/network, and amortized indexing/update cost. Unavailable measurements remain unavailable.

## 8. Test polygon and anti-overfitting protocol

### 8.1 Required source diversity

1. Multicategory marketplace with genuinely distinct entities/attributes and separately labelled synthetic offers.
2. Public food knowledge with no true retail price capability and multilingual noisy categories.
3. Public price observations with currency/time/quantity semantics and missing categories.
4. Public web products with multiple sellers, variants, missing values, and provenance.
5. Relational library: works, authors, editions, loans/tags; no universal commercial criteria.
6. Forum or support knowledge: threads, messages, authors, documents, permissions, time, and citations.
7. Generic EAV and multi-table sources with anonymized/renamed schema identifiers and no product table convention.
8. Document collections in each explicitly supported format, with span evidence and hostile embedded instructions.
9. Federation of at least three independently versioned sources with labelled matches, nonmatches, and contradictions.
10. Broken-source variants plus evolving source snapshots.

These are evaluation categories, not runtime hardcoded domain names. Development fixtures are permitted; held-out source/domain identifiers and labels are not allowed to leak into production logic.

### 8.2 Defect classes

Missing/composite/unstable keys; duplicate identities and conflicting assertions; null/unknown placeholders; text-encoded numbers; units in keys; mixed/unknown units; swapped/mislabelled columns; orphan relations; Unicode/confusable/encoding issues; malformed documents; missing prices/stock; contradictory merchant observations; stale/deleted facts; adversarial content; huge individual values; source/provider errors.

For each defect specify the expected truthful outcome, independently validate transformations, and retain all loss/conflict counters. Do not make the input easier by silently discarding hard examples.

### 8.3 Freeze, split, and oracle rules

- Freeze requirements, supported capabilities, score definitions, and thresholds before evaluating held-out data.
- Split by unseen source/schema/domain and meaningful entities, not merely different rows in the same authored schema.
- Hold out domain vocabularies; rename/anonymize keys and categories to detect dictionary dependence.
- Independent experts/source-backed annotations define truth; the production checker cannot generate its own answer key.
- Score complete answers, not substrings that may appear in leaked thoughts or negated text.
- Run evaluator mutation tests before using it to support any success rate.
- Retain repeated-run variability for live model paths, model/provider/configuration identity, raw traces, timeouts, and exclusions.
- Keep input/query sets paired for scale comparisons; explain any truth change when corpus content grows.
- Protect private customer data; redact or use authorized published/synthetic fixtures for public research artifacts.

## 9. Performance and scaling plan

The initial engineering polygon should include approximately 10,000, 100,000, and 1,000,000 records, plus growth in distinct identities, EAV keys, categories, documents, and tenants. Those sizes are future trial targets; none is a claim that current software has passed them.

Before running, record hardware, storage engine/version, indexes, RAM/disk limits, average/worst row and document sizes, update rate, query classes, tenant concurrency, top-k and expansion limits, model settings, and preparation state. Set explicit ingestion/query/update SLOs per workload and budget; there is no useful universal latency target without these conditions.

Require:

- Bounded working memory through streaming/chunked ingest and bulk/indexed retrieval.
- Full-source reachability, measured retrieval recall, and explicit approximation boundaries.
- Background discovery/indexing separated from synchronous visitor queries.
- Persistent job coordination, idempotency, backpressure, cancellation, quotas, and resource isolation.
- Incremental update experiments alongside rebuild experiments.
- Exact small-instance differential checks before optimizing large searches.
- Latency/quality/cost curves, not only one “under 50ms” path or an attractive graph screenshot.

The maximum validated envelope is a documented operational result. Expand it through measurement and architecture changes, never by relabelling a repeated synthetic dataset.

## 10. Required client scenarios

| Scenario | Required outcome |
|---|---|
| New unseen database | Automatically profile/adapt supported structure; expose progress and uncertainty; no production edit |
| Library without price | Useful search/references; unsupported price requests explained without fake zeros |
| Forum/document question | Relevant source citations/spans; no unauthorized/private-content leakage |
| Ambiguous entity | Candidate clarification or qualified result; not arbitrary first-match certainty |
| Two-entity price comparison | Correct bound values/offers; not a compatibility shortcut |
| Missing relation/fitment knowledge | Unknown or explicit missing evidence; no affirmative safety/compatibility default |
| Conflicting sources | Both assertions and provenance visible, with explicit policy if selecting one |
| Hard budget versus soft preference | Enforce hard bound; propose consented, traceable soft relaxation if needed |
| Model returns only internal reasoning | Protocol repair/failure, never client-visible reasoning or successful answer |
| Model query manufactures literals | Not admitted as source facts |
| Concurrent visitors/tenants | Independent sessions/evidence/settings, authorized data, actual spend accounting |
| Product/document changed or deleted | New snapshot used correctly; obsolete cached claims not certified current |
| Provider/ingestion/search failure | Explicit failure, cancellation/retry semantics, no fabricated successful fallback |
| External site widget | Real tenant-scoped embed, allowed origin, accessibility, isolated lifecycle |
| Large/dirty source | Bounded processing with accountable degraded states, measured quality and resource limits |

## 11. Scientific objective and finish line

The dissertation application remains intelligent consultation for online-store visitors, while the research core can test domain-independent adaptation as a broader mechanism. The project is not required to invent every building block. It **is** required to identify a specific additional method or verified improvement and not falsify its origin, guarantees, or results.

Recommended primary research question:

> Can a provenance-preserving, uncertainty-calibrated adaptation method improve useful factual consultation on heterogeneous imperfect sources and unseen schemas, without customer-specific code, relative to fair retrieval/SQL/graph baselines within a measured compute budget?

Required research deliverables:

- Closest-prior-art matrix: problem, input assumptions, method, guarantees, evaluation, and precise difference.
- One primary algorithmic contribution with precise definitions, complexity, and limits.
- Proofs for the actual canonical representation and conditional guarantees, not for an unrelated abstract structure.
- Independent labels/oracles, unseen-domain splits, fair baselines, ablations, repeated runs, and uncertainty intervals.
- Positive and negative results, counterexamples, resource costs, failure explanations, and reproducible artifacts.
- A claim register distinguishing proposed, implemented, tested, integrated, and independently evaluated.

**Product finish line:** real zero-domain-code onboarding, evidence-linked useful consultation, tenant-safe embedding, correct updates/conflicts, and a published validated operating envelope.

**Scientific finish line:** a narrowly defined, independently supported contribution. “An ideal consultant over any amount of arbitrary data” remains the direction of the product, not an experimentally proven absolute.
