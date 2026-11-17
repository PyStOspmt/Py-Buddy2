# 11 · Current-state audit and evidence-backed remediation

> [Українська версія / Ukrainian version](11_CURRENT_STATE_AUDIT_AND_REMEDIATION.uk.md)

**Audit date:** 2026-10-04. **Status:** assessment of the inspected snapshot, not a statement that defects have been fixed.

**Read together:** [target system and acceptance contract](12_TARGET_SYSTEM_AND_ACCEPTANCE.md), [mandatory agent instructions](../AGENTS.md), [local evidence](audit_evidence_20261004.json), and [offline test transcript](audit_pytest_20261004.txt).

## 1. Executive conclusion

The intended product is a domain-independent adaptation and consultation service: a customer connects sources, embeds a chat widget, and the service discovers usable structure, retrieves evidence, and answers without customer-written domain code.

**The current implementation is a useful catalogue research prototype, not that product.** Its strongest parts are deterministic database access, explicit constraint classes, experiments on imperfect public data, and regression tests. Its weakest parts are the boundaries between those parts:

1. A generic-looking schema profile becomes a prompt; downstream code still expects a commerce schema.
2. A model-written read-only query result is trusted as a source fact even when the query manufactures constants.
3. Grounding is a reused agent-level name/price cache, not request-scoped, provenance-carrying evidence.
4. Missing rules, unresolved entities, and missing attributes can become affirmative compatibility or feasibility.
5. Free-text answer checking is substantially weaker than the advertised guarantees.
6. Evaluation can reward empty, incomplete, wrong, or unsupported answers; some scientific experiments share their oracle with the method being tested.
7. Multiple algorithmic names and mathematical claims exceed what is implemented or connected to the user-facing path.
8. The service lacks the isolation, onboarding, resource controls, update lifecycle, and embeddable distribution needed for a customer-facing multi-tenant product.

**Do not respond by polishing failed output, adding a few domain words, or increasing timeout limits.** Restore the evidence and evaluation contracts first, then build genuine adaptation. Retain useful components; do not substitute a wholesale rewrite for understanding their failures.

### 1.1 What was actually measured

- **25 selected local counterexamples reproduced 25 defects.** This is not a representative failure-rate estimate and not a live LLM benchmark. Scripted model outputs exercise real agent, SQL, gate, graph, and solver implementations.
- **395 existing offline tests passed; 1 was skipped; runtime 44.40 seconds.** Three modules were excluded to avoid live-server initialization or external-model execution. This is not a full-suite claim.
- No external model inference was performed for this audit. Public literature pages were fetched separately, without credentials.
- Existing application source and existing benchmark/adversarial artifacts were not corrected. The audit adds documentation, instructions, an audit-only reproduction utility, and new evidence artifacts.

The test transcript records the exclusions and arguments. The evidence records environment versions, source checksums, database inventory, actual counterexample outputs, and the historical adversarial answers. **Green tests do not negate counterexamples outside their asserted input domain.**

## 2. Method, coverage, and limits

### 2.1 Evidence labels

- **R — reproduced:** executed against this snapshot using local data or read-only inventory; a probe identifier is supplied.
- **S — source-supported:** control flow or data handling directly supports the finding; an end-to-end exploit or production-scale failure was not executed.
- **A — artifact-supported:** present in a saved historical result; not independently rerun with the live model.
- **U — unverified:** insufficient information or no controlled experiment. Not counted as a demonstrated defect.

Severity: **P0** blocks trustworthy or externally exposed use; **P1** blocks adaptive-product acceptance; **P2** blocks reliable operation or defensible research claims. Severity is an engineering priority, not a CVSS score.

### 2.2 Inspected scope

Runtime and research modules, all existing test modules through direct inspection or the offline run, the web interface, gateway adapters/runners, ingestion/generation/degradation scripts, stored evaluation artifacts, all numbered project documents 01–10, and architecture documents 00–06 were examined across the audit and delegated reviews. The current academic practice assignment was also checked: [approved-form assignment](../../pre-dipl-practick/готові%20доки/3_ЗАВДАННЯ_НА_ПЕРЕДДИПЛОМНУ_ПРАКТИКУ.docx) identifies online-store consultation as the dissertation topic. A generic adaptation core can serve that application without pretending that the academic title proves a universal system.

**Important inventory correction:** some directory/search tools hide ignored files. Documents 03, 06, 07, 09, 10 and the architecture documents **do exist locally**, but are ignored by [project ignore rules](../.gitignore:62). An initial delegated inventory incorrectly called them absent; direct reads and the second review corrected that. Absence from a filtered listing is not evidence of absence from disk.

Limits: no production security penetration test; no concurrent multi-tenant deployment; no million-row experiment; no paid/live model evaluation; no exhaustive novelty search; no full suite with the three excluded modules. No claim is made that this audit finds every possible defect.

### 2.3 Actual data, not filenames or UI labels

Counts come from the new [evidence report](audit_evidence_20261004.json).

| Local source | Product rows | Distinct names | Categories | Important limitation |
|---|---:|---:|---:|---|
| PC demonstration catalogue | 18 | 18 | 5 | Authored fixture, not a heterogeneous customer database |
| Auto-parts catalogue | 10 | 10 | 3 | Authored parts fixture; no authoritative vehicle-fitment table |
| Generated marketplace | 10,000 | 190 | 8 | Repeated listings; not 10,000 independently diverse product identities |
| Pharmacy catalogue | 10 | 10 | 8 | Authored fixture; the word “real” in its filename is not clinical validation |
| Open Food Facts import | 3,000 | 2,622 | 364 | All 3,000 prices are zero placeholders, not free-product evidence |
| Open Prices import | 6,554 | 5,794 | 1 | All category values are “unknown”; different price/observation semantics |
| WDC import | 3,781 | 2,899 | 9 | 3,289 zero prices; heavily concentrated in one category |

Open Food Facts' three largest category values contain **1,736/3,000 rows = 57.87%**: “en:null” 1,010; “en:undefined” 402; “unknown” 324. A top-category strategy that treats these as meaningful categories prioritizes missing semantics.

### 2.4 Real current integration path

Registration or first domain access → schema profiling → capped catalogue projection and heuristic contract induction → synchronous ontology and question-bank preparation → router → SQL or ReAct tools → free-text gate → response and partial session persistence.

Evidence: [`SystemContext._init_bundle_engine()`](../src/server.py:204), [`DomainCatalogRegistry.create_database_catalog()`](../src/domain_catalog.py:116), [`FastCascadedRouter.route_and_execute()`](../src/fast_router.py:110), [`ReActAgent.run_live_session()`](../src/react_agent.py:587).

There is **no demonstrated federation ingest pipeline**, arbitrary-file knowledge pipeline, customer widget lifecycle, or verified end-to-end connection of the research lattice/NVFG/FCA components to this consultation path.

## 3. Reproduced counterexample ledger

All observations are retained in [audit evidence](audit_evidence_20261004.json). “Reproduced” means the defect occurred, not that the product passed acceptance.

| Probe | Observed failure | Main finding |
|---|---|---|
| P01 | A thought-only object becomes the client answer | F02 |
| P02 | Unsupported availability/safety prose without evidence becomes successful grounding | F03 |
| P03 | SQL-selected literal product and literal price become a verified recommendation | F04 |
| P04 | Alpha and Beta's prices can be swapped and accepted, including recommendation objects | F05 |
| P05 | A price preceded by “від” bypasses source checking | F05 |
| P06 | A user's budget can be presented as a product price | F05 |
| P07 | A fabricated entity remains in prose after the gate says it was removed | F06 |
| P08 | A second session reuses the first session's grounding without executing a tool | F07 |
| P09 | A rejected answer gets a successful-grounding trajectory record | F08 |
| P10 | Entirely unresolved tool arguments yield an affirmative compatibility result | F15 |
| P11 | A bound parameter exceeds the sanitizer's advertised row maximum | F31 |
| P12 | Failed tool-loop output is retrospectively labelled conversational success | F02/F08 |
| P13 | A recognized product table without a price column crashes candidate retrieval | F10 |
| P14 | Projection of “ssd” changes from “sd” to “d” when reapplied | F20 |
| P15 | Two groupings of the same merge yield a value versus conflict | F20 |
| P16 | Reordering an unparseable value and a symbolic value changes the fold outcome | F20 |
| P17 | Canonical conversion of “1000 TB” cannot be reparsed | F20 |
| P18 | Kept equality constraints are excluded from optimization, producing false infeasibility | F21 |
| P19 | Budget-truncated search reports infeasibility rather than incomplete search | F21 |
| P20 | A dominated price relaxation remains on the claimed Pareto frontier | F21 |
| P21 | Missing numeric data satisfies a hard requirement | F15/F21 |
| P22 | An equality-style prefilter excludes an actually compatible spatial addition | F16 |
| P23 | Partial names fail; only a verbatim full catalogue name resolves | F13 |
| P24 | Matching PCD yields compatibility although the fixture contains 21 versus maximum 19 | F14 |
| P25 | A price-comparison query naming two products is hijacked into compatibility advice | F17 |

**Reproduction:** from the project directory run [`python scripts/audit_current_state.py --output docs/audit_evidence_20261004.json`](../scripts/audit_current_state.py:233). For historical reproducibility, use a different output name on later runs instead of overwriting this audit's evidence. The utility extracts the exact two server functions without importing server global initialization; that is a local function reproduction, not an HTTP deployment test.

## 4. Findings and required corrections

### 4.1 Evaluation integrity and completion

#### F01 · The adversarial pass rate does not establish task success — P0 · S/A

The evaluator checks nonempty/non-error output, a few domain keywords, and three case-specific keyword conditions; it never uses the declared expected constraint in its verdict. Grounding, required facts, budget satisfaction, supported availability, or a fulfilled recommendation are not verified. See [verdict implementation](../tests/test_adversarial_client_eval.py:195).

Historical false passes include the thought-only prescription response, the no-tool auto recommendation, and answers ending after “available products” or “cheapest offers” without the promised items. See [original artifact](../adversarial_eval_results.json). The prescription keyword appears in a leaked thought and satisfies its test. The original “8/12” is only eight passes through that checker.

**Do not replace it with “4–5/12 real success.”** That estimate also lacks an independently applied task oracle. Record at least four obvious false passes; recompute the rate only after validating every case against source-supported expectations. The historical artifact also omits complete observations, making retrospective grounding analysis incomplete.

**Fix and acceptance:** machine-readable expectations for each case; expected entity/attribute/value binding; required facts and relevance; constraint satisfaction; documented unknown outcomes; separate client timeout, server failure, model/provider failure, format failure, unsafe answer, and task failure. Preserve full traces with privacy controls. Mutate valid answers into empty introductions, thought objects, wrong products, swapped prices, and ungrounded safety claims: the evaluator must reject each mutation. Add a true runnable evaluation entry point; naming a script as a test module does not cause its main evaluation loop to run under pytest.

#### F02 · Invalid model output is terminal output; failed execution becomes conversation — P0 · R/S

The agent publishes an unparseable step by stringifying it: [terminal branch](../src/react_agent.py:674). The client parser accepts arbitrary parsed JSON without enforcing its shape and otherwise wraps raw output as a final answer: [parser fallbacks](../src/openrouter_client.py:262). The router labels an execution conversational solely because there were no tool hashes: [retrospective classification](../src/fast_router.py:186). P01/P12 reproduce both paths.

**Fix and acceptance:** strict mutually exclusive typed outcomes: tool request, supported final response, nonfactual conversation, clarification, explicit failure. Validate argument types and tool names before execution. Reject thought-only, arrays, scalars, mixed malformed outcomes, raw provider reasoning, and arbitrary parse failures. Allow bounded format repair, then explicit protocol failure; never turn internal reasoning into polished client prose. A data-dependent request without required evidence cannot terminate successfully merely because the model stopped asking for tools.

#### F03 · “Grounded” does not mean all claims were checked — P0 · R/S

The live gate largely checks recommendation names and Ukrainian-currency price tokens; unsupported product names in prose, specifications, stock, clinical claims, fitment, citations, negation, and completeness are outside that contract. An empty recommendation list and no price tokens can produce success with no retrieved evidence. P02 demonstrates unsupported availability and safety passing. See [live gate](../src/react_agent.py:438).

The separate [recommendation checker](../src/sanitizer.py:261) is not a universal prose verifier and is not called by this live finalization path. Merely having a stronger-looking class in the repository is not integration.

**Fix and acceptance:** final factual claims must be represented and verified independently before rendering. Each binds an entity, predicate, value, units/currency, source location/version, and any explicit derivation. Nonfactual conversation has a separate label, not a factual certificate. Negative claims require evidence of a sufficiently complete query, not a missing match in a limited sample. Every answer must satisfy its task contract as well as its evidence contract.

#### F04 · Model-written SQL can launder fabricated facts — P0 · R

The sanitizer permits read-only expressions and the grounding set trusts named result columns. A query selecting a literal “Phantom widget” and a literal price, without any source row, produces a successful verified recommendation (P03). [SQL execution](../src/sanitizer.py:163) → [grounding admission](../src/react_agent.py:180) → [name admission](../src/catalog_query.py:355).

**Read-only is not factual provenance.** The same problem applies to fabricated aliases and unverified expressions over real rows.

**Fix and acceptance:** the model proposes a typed query intent, not arbitrary fact-producing projections. Compile permitted access through source adapters. Retrieve stable row IDs/source versions and rehydrate factual fields from trusted projections. Derived computations carry an operator, input evidence, units, and checked result. Reject literal or arbitrary computed fields as catalogue facts unless explicitly classified as a valid derivation. Test constants, synthetic aliases, rewritten prices, malicious source strings, and legitimate verified aggregates.

#### F05 · Entity–value binding and recommendation projection are broken — P0 · R

The gate checks whether a number occurs **anywhere** in its allowed rows before binding it to an entity; then it exempts user numbers, arbitrary permitted arithmetic values, and broad words such as “від”, “до”, and “сума”. Recommendation objects are filtered by name but their ID/price fields are not reprojected from source evidence. See [number exemptions](../src/react_agent.py:469), [recommendation filtering](../src/react_agent.py:513). P04–P06 reproduce three bypasses.

**Fix and acceptance:** bind every asserted value to its entity and semantic role. A budget is not an offer price; a mathematically possible subset sum is not proof that this basket total is correct. Source-project returned recommendation objects; never preserve model-supplied factual values simply because a name matches. Test swapped values, altered IDs, multiple offers per name, user-budget collisions, invented discounts, other currencies, and calculated totals with explicit selected inputs.

#### F06 · “Correction” can retain unsupported prose and invent usefulness — P0 · R/S

When some recommendations are fabricated and some verified, the gate filters the structured list but appends a note to unchanged prose saying the fabrications were removed. P07 still contains the fabricated entity. [Partial correction](../src/react_agent.py:561).

Other correction paths render a few rows but claim they meet the user's criteria, without proving relevance or all requested requirements: [fallback renderer](../src/react_agent.py:575). An answer can be locally factually sourced yet solve the wrong task.

**Fix and acceptance:** reject the unsupported candidate response or regenerate a constrained response solely from verified claims and explicit task results. Record removed/repaired claims, actual verified count, and separate factuality from usefulness. “Grounded fallback” is not permission to return unrelated cheap rows. Test mixed valid/invalid prose, incomplete lists, missing qualifications, and off-intent sourced recommendations.

#### F07 · Grounding is shared across requests and sessions — P0 · R/S

The [agent initializes grounding once](../src/react_agent.py:120), while [each run creates a fresh session](../src/react_agent.py:604) but never resets or request-scopes the evidence. Domain bundles reuse the agent. P08 accepts a second session's answer using the first session's retrieved rows. Concurrent requests can also contaminate each other.

**Fix and acceptance:** evidence belongs to a request-scoped execution context with tenant, knowledge snapshot, session, and request identity. Immutable shared indexes may be reused; request evidence may not. Simply resetting a shared attribute at run start is still unsafe under concurrency. Test interleaved requests, distinct principals, changed source versions, removed products, and identical names across sources.

#### F08 · Telemetry misstates verification, routing, latency, and cost — P0 · R/S

The return value of finalization is discarded and successful-grounding trajectory labels are hardcoded: [normal completion](../src/react_agent.py:655), [budget completion](../src/react_agent.py:698). P09 shows a rejection recorded as successful grounding. API invariant status and catalogue scale are static: [status](../src/server.py:318), [telemetry](../src/server.py:382). Token cost is step-history length times a constant, not actual provider billing: [cost estimate](../src/fast_router.py:189). The deterministic branch reports literal zero latency after initialization work: [deterministic response](../src/server.py:655).

**Fix and acceptance:** distinguish declared capability, actual execution, verified result, unavailable measurement, and estimated cost. Record per-call provider/model/usage; actual or explicitly estimated pricing; retries and induction cost; request-wall-clock and individual stage timings. Never substitute zero for unavailable. Gate rejection must agree in API status, trace, badge, and aggregate metric. Static demonstrations belong in labelled demo displays, not live telemetry.

**Not fabricated:** the agent's LLM/tool/gate timing accumulation and router timing deltas are real measurements. They are incomplete request accounting, not grounds to call every timing invented.

### 4.2 Adaptation, retrieval, and source semantics

#### F09 · Introspection is not yet a canonical adaptation layer — P1 · S

The profiler discovers actual tables, keys, foreign keys, and distributions, which is useful. Semantic-role assignments and EAV detection nevertheless rely on name conventions: [role heuristics](../src/profiler.py:166), [EAV heuristics](../src/profiler.py:208). A snapshot becomes mainly prompt text: [bundle setup](../src/server.py:207). It is not the common source adapter consumed by query, graph, and retrieval engines.

**Fix and acceptance:** schema and role hypotheses with confidence/evidence become a versioned canonical interface used everywhere. Structural/value evidence must continue working after table/column/key renaming. Generic introspection cannot by itself identify whether two numeric columns mean price, rating, dosage, or measurements. Ambiguity must remain visible rather than be guessed from a token. Test unseen schemas, quoted identifiers, multiple tables, missing PKs, composite identities, and ambiguous roles.

#### F10 · “Any database” downstream paths require a product schema — P1 · R/S

Catalogue projection requires a specifically named product table and fixed identity/name/category/price columns: [projection](../src/domain_catalog.py:130). Candidate access optionally detects performance and stock but still selects price: [candidate construction](../src/catalog_query.py:79); P13 fails on an unpriced product table. Ontology construction uses a short table/column-name allowlist: [ontology discovery](../src/ontograph.py:325).

**Fix and acceptance:** consume discovered source projections and capability metadata instead of direct table conventions. A library without price remains searchable; no performance field means that ranking axis is absent. Test a relational library, a forum, a generic EAV schema with renamed keys, and document collections, without domain-specific production edits. A storage-engine adapter is legitimate; a customer/category adapter hardcoded to pass one fixture is not zero configuration.

#### F11 · Domain knowledge remains in dictionaries and parser assumptions — P1 · S

The normalizer retains socket/power/memory/PCD/diameter word lists: [normalizer](../src/marketplace_normalizer.py:85). Runtime induction uses demand/provider/spatial vocabulary: [inductor](../src/contract_inductor.py:69). Even the supposedly generic value algebra strips domain prefixes and assumes an undecorated radius token is inches: [value interpretation](../src/attribute_algebra.py:268), [radius assumption](../src/attribute_algebra.py:306).

Supporting Ukrainian versions of the same physical words is **not** removing hardcoding. The profiler's business-role hints also must not be promoted to ground truth solely by naming.

**Fix and acceptance:** typed parsing for known universal formats; standardized unit conversion with explicit dimension context; structural discovery for domain concepts; source-backed hypotheses with verified confidence. Keep physical claims external to the generic engine and supported by authoritative connected knowledge. Rename all business keys/categories, introduce a new language/domain, and rerun held-out tests. A universal unit registry or dictionary data structure is not automatically an antipattern; domain word lists that determine factual behavior are.

#### F12 · Observed attribute correlations are promoted to physical rules — P1 · S

Shared string attributes across categories become equality rules, numeric names choose demand/provider or bounding roles, and a fixed headroom is introduced. [Runtime induction](../src/contract_inductor.py:69). Same color, material, or vendor across two categories does not establish an assembly rule; numeric ordering does not establish causality or capacity direction.

The structural benchmark inducer is separate from the deployed method: [structural inducer](../src/cip_benchmark.py:252) versus [runtime call site](../src/domain_catalog.py:182). Its benchmark performance cannot be reported as the deployed engine's performance.

**Fix and acceptance:** separate discovered associations, candidate constraints, and verified constraints. Require trusted declared relations or independently validated rule supervision/counterexamples before hard-rule promotion. Carry rule scope, dimension, relation identity, provenance, confidence, and verification status. Include negative-control correlations and category/key anonymization in the induction benchmark. Four universal evaluator classes do not constitute universal rule discovery.

#### F13 · Entity resolution has a dead token-overlap branch — P1 · R

The resolver uses an aggressive no-space key before attempting tokenization, yielding one concatenated token. Its additional requirement for at least two common tokens cannot hold. See [`_resolve_from_hypergraph()`](../src/server.py:504), [key normalization](../src/metrics.py:111). P23 reproduces partial-name failures and the verbatim full-name match.

**Fix and acceptance:** separate exact-key normalization from tokenization, use an indexed resolver over full source identities and aliases, rank/retain ambiguous candidates, and ask for clarification when appropriate. Morphology may be general-language processing, but adding a few fixture brands/suffixes is not a solution. Test unseen names, paraphrases, inflection, Unicode, overlapping variants, and multiple plausible matches.

#### F14 · Seed-limited rule and entity coverage depends on browsing history — P1 · R/S

Graph construction groups by name and caps a seed before inducing rules: [seed query](../src/domain_catalog.py:141). Rules are inferred from that seed only. API catalogue pages dynamically add nodes: [page loading](../src/server.py:434). Chat resolves only currently loaded graph nodes, so opening pages can change its ability to answer. New nodes do not automatically trigger sound incremental rule validation.

P24 finds only a PCD rule in the auto fixture. Mansory's diameter is 21 and the hub's maximum is 19, yet compatibility is true. This verifies missing rule coverage **for the fixture**, not the real geometry of Volkswagen Golf; the fixture is not an authoritative vehicle-fitment model.

**Fix and acceptance:** distinguish complete source coverage, sample coverage, rule coverage, and resolution coverage. Resolve/query the full source through indexes; use stratified, versioned discovery rather than an accidental first seed. Test entities past the seed boundary, before and after UI browsing, and late-arriving constraint-bearing rows. Do not falsely infer vehicle fitment from a hub's name or pretend a brand-priority matcher repairs absent rules.

#### F15 · Unknown or incomplete compatibility is reported as compatible — P0 · R/S

The tool drops unresolved arguments and evaluates only remaining IDs; an empty resolved set is compatible (P10): [tool resolution](../src/react_agent.py:194). The graph's no-rule basket default is suitable only for joint checkout, not verified physical, drug, or semantic compatibility: [default](../src/hypergraph.py:259). Missing matching/capacity/spatial evidence also passes various predicates: [matching](../src/hypergraph.py:66), [capacity](../src/hypergraph.py:101), [bounds](../src/hypergraph.py:158). P21 reproduces missing numeric data satisfying a hard requirement.

**Fix and acceptance:** at least supported-compatible, supported-incompatible, unknown, and partially checked outcomes. Preserve every requested entity and its resolution state. Validate applicability and completeness of every required rule; distinguish absence of a detected violation from proof of compatibility. Clinical/fitment/safety decisions require authoritative evidence and cannot inherit a permissive checkout default. Test unresolved IDs, one resolved ID from two, missing critical attributes, empty rules, contradictory values, and incomplete knowledge.

#### F16 · Graph candidate prefilter is unsound — P1 · R/S

The prefilter treats spatial dimension keys as equality commitments and ignores constraint category scope: [matching-key extraction](../src/hypergraph.py:198), [filter](../src/hypergraph.py:271), [candidate loop](../src/hypergraph.py:295). P22 shows a 40-unit object excluded after selecting a 30-unit object even though both fit within the same 100-unit bound.

Replacing a node also leaves old category/attribute memberships in indexes: [node addition](../src/hypergraph.py:191). This is a source-supported update correctness risk, not a measured large-scale failure.

**Fix and acceptance:** only use pruning predicates proven necessary for the corresponding rule and scope. Differential-test indexed results against exhaustive evaluation on small generated datasets, including updates/removals. No optimization may silently delete a feasible result; approximation requires an explicit recall/coverage contract.

#### F17 · Naming two products overrides the user's intent — P1 · R

Any query with two resolved catalogue names enters the deterministic compatibility branch; there is no compatibility-intent check: [shortcut](../src/server.py:622). P25 asks to compare prices and receives “fully compatible.”

**Fix and acceptance:** shortcuts require a validated task type and enough evidence, not merely entity count. Test price comparison, descriptions, “do not recommend,” availability, compatibility, and unrelated mentions over identical entities. Successful SQL retrieval does not excuse answering a different question.

#### F18 · Source metadata is lost; placeholders become commercial facts — P0 · S/R inventory

Public-data ingestion records useful source limitations, but query/recommendation paths largely ignore them. The OFF import assigns zero price, export-presence stock, and data-completeness performance: [ingest semantics](../data/ingest_real_data.py:362), [metadata](../data/ingest_real_data.py:380). Candidate retrieval treats stock as availability and performance as recommendation quality: [candidate filters](../src/catalog_query.py:103). UI/API projection defaults missing values to zero or assumed merchant rating/delivery: [lazy projection](../src/domain_catalog.py:93), [metadata defaults](../src/domain_catalog.py:110), [Pareto defaults](../src/server.py:757).

Answers also format every monetary value as hryvnia without preserving source currency. Different currencies, observation dates, sellers, variants, or export-presence fields cannot be silently treated as equivalent current offers.

**Fix and acceptance:** source capability/semantic declarations flow through the canonical evidence model. Missing remains missing; distinguish not stated, parsing failure, placeholder, stale observation, and known zero. Preserve currency, quantity basis, timestamp, offer identity, and availability meaning. OFF price queries must explain unavailable price evidence; data completeness must never be rendered as product performance; missing delivery/rating cannot become a factual default.

#### F19 · Multi-offer resolution destroys conflicts and stable identity — P1 · S

Entity clustering folds merchant attributes, but on conflict keeps a canonicalized first value or the raw first value: [resolution](../src/entity_resolution.py:174). Canonical IDs depend on iteration index: [identity assignment](../src/entity_resolution.py:153). Name grouping is not sufficient to establish that two offers represent the same entity/variant.

Raw consumed synonym keys can also reappear beside canonical ones because only a short skip list is used: [normalizer pass-through](../src/marketplace_normalizer.py:145). The live grounding structure further keys prices by name and keeps first values, collapsing distinct offers: [grounding storage](../src/catalog_query.py:339).

**Fix and acceptance:** stable source identities, explicit uncertain cross-source entity links, separate entities from offers/observations, conflict records retaining all conflicting assertions and provenance. Do not pick a convenient first value. Test input permutations, homonyms, revisions/pack sizes, changed offers, concurrent sources, conflicting units, and keys consumed by normalization. Independent evidence labels are necessary for entity-match precision.

### 4.3 Mathematical and research correctness

#### F20 · Implemented value algebra violates its advertised laws — P1/P2 · R

P14–P17 reproduce non-idempotent projection, non-associative merge, permutation-dependent folding, and lack of closure under canonicalization. “ssd” becomes “sd” and then “d”; one grouping produces a value while the other produces conflict; an unparseable first value can be erased as if it were missing; a converted large number cannot be reparsed.

Evidence: [token stripping](../src/attribute_algebra.py:329), [projection](../src/attribute_algebra.py:359), [merge identity cases](../src/attribute_algebra.py:447), [fold](../src/attribute_algebra.py:465). Existing hand-picked algebra tests are not a proof for all admitted inputs.

**Fix and acceptance:** separate raw values from a closed, typed canonical domain; missing, unparseable, uncertain, and contradictory are distinct states. Define merge on canonical values without repeatedly mutating token meaning. Prove laws for the mathematical carrier actually used, then property-test generated valid/invalid Unicode, free text, zero/negative/large numbers, all unit conversions, unknown units, and repeated projection. Include these four counterexamples as permanent failing-before/fixed-after regressions. Remove universal “proven implementation” claims until repaired.

#### F21 · Relaxation can claim false infeasibility or false optimality — P1/P2 · R/S

P18 shows optimization excluding kept equality constraints and later rejecting a relaxation that serves the wrong item, despite another feasible relaxation. P19 shows search exhaustion labelled infeasibility. P20 shows reversed Pareto dominance retaining a dominated bound. P21 shows unknown data satisfying a hard requirement.

Evidence: [solver bridge](../src/relaxation_lattice.py:475), [truncated enumeration](../src/relaxation_lattice.py:372), [dominance](../src/relaxation_lattice.py:445), [missing-value predicate](../src/nvfg.py:131). Additional source-supported issues: tightening infinities admitted, alternate encodings of unchanged bounds, and unvalidated zero/negative regret weights.

**Fix and acceptance:** optimize with all kept constraints enforced; return unknown/incomplete search when exploration is truncated; correct dominance and numeric-domain/weight validation; preserve missing/invalid evidence as unknown. Check small instances against exhaustive enumeration or an independent solver. Hard safety/identity requirements cannot be relaxed into an apparently useful answer. The finite-relaxation existence result is conditional on allowed bounds and a nonempty relevant catalogue; it is neither a universal factuality guarantee nor a new algorithm by itself.

#### F22 · Active inference is not a completed adaptive dialogue — P1/P2 · S

There is a Beta belief representation and question selection, but the quantity called entropy is expected Bernoulli outcome entropy, not posterior parameter entropy. The expected-gain computation combines that quantity, fractional updates, a separate answer predictor, and clipping: [entropy](../src/active_inference.py:147), [gain](../src/active_inference.py:554). Question selection is exposed, but production does not demonstrate answer-driven persistent belief updates feeding ranking: [tool](../src/react_agent.py:285), [fresh session](../src/react_agent.py:604).

**Fix and acceptance:** define a coherent latent preference model, likelihood, posterior update, and expected information/utility objective. If keeping a heuristic, name and evaluate it as a heuristic. Wire customer answers to persistent belief and ranking. Compare fewer-question utility against random/fixed-question and no-elicitation baselines on held-out users. No-question outcomes must distinguish missing bank from actual confidence.

#### F23 · Fuzzy Pareto selection is retail-specific and can misstate tradeoffs — P1/P2 · S

Fixed price/performance/delivery memberships remain: [membership functions](../src/pareto_relaxer.py:26). Heterogeneous candidates can carry different dimension sets; the comparison can assume missing keys exist: [dominance loop](../src/pareto_relaxer.py:82). Explanations can assert budget fit or maximum performance without checking those propositions: [selection/explanation](../src/pareto_relaxer.py:113). The live tool does not feed learned user weights: [call](../src/react_agent.py:266).

**Fix and acceptance:** derive available axes and explicit direction/unit/meaning from source capabilities and user goals. Unknown axes are absent or uncertain, not fabricated performance. Keep hard/soft preferences separate. Emit an actual relaxation ledger; the current bounded candidate retrieval can already relax a request before its “relaxed” flag is set. Test missing dimensions, contradictory criteria, hard budgets, noncommerce ranking, and explanation truth.

#### F24 · Research solvers and “FCA lattice” are not the live capability claimed — P1/P2 · S

The FCA implementation builds root/category/category-brand concepts rather than computing all closed formal concepts, joins, meets, and a proper lattice: [builder](../src/taxonomy_fca.py:101). Navigation scans/sorts and returns capped samples: [navigation](../src/taxonomy_fca.py:159). Production uses the separate [OntoGraph construction](../src/server.py:233). NVFG/lattice and the structural benchmark inducer are likewise not the deployed tools.

**Fix and acceptance:** label the current structure a faceted taxonomy unless formal closure is implemented. Document each algorithm as implemented, independently tested, experimentally evaluated, and production-wired—four separate states. Build integration tests proving that the claimed component executes for the claimed user task. Adopt graph/lattice structures only where their measured benefit exceeds cost; constructing every possible concept or product combination is not required for adaptation.

#### F25 · Benchmark correctness and informativeness metrics are permissive — P0/P2 · S/A

General task correctness does not require expected-product recall or required price delivery: [correctness property](../src/metrics.py:284). Entity detection recognizes corpus-known names and a restricted model-token grammar, not all fabricated entities: [extraction](../src/metrics.py:183). Price checking loses entity–offer binding: [price scoring](../src/metrics.py:398). Non-vacuous factual aggregation can ignore hallucinated entities or wrong verdicts, and one false-claim numerator relies on an unpopulated field: [aggregation](../src/metrics.py:487).

The stored live price-trap family reports 20% execution accuracy alongside 100% NVFG: [live artifact](benchmark_live_20260930.json:69). That alone demonstrates that the metrics are not interchangeable; it must not become “100% useful factual answers.”

**Fix and acceptance:** independently specified task/claim labels; facts as bound assertions; required-answer completeness; meaningful absence/unknown handling; a combined factual-usefulness score that cannot be true on a known hallucination. Verify metric code with adversarial answer mutations and hand-calculated fixtures. Publish raw numerators/denominators and skipped-family reasons.

#### F26 · Evaluation oracle leakage and an underpowered comparison baseline — P2 · S

The offline agent imports the benchmark compatibility checker and labels/answers are derived from the same logic: [agent dependency](../src/offline_agent.py:26), [case oracle](../src/benchmark_cases.py:246). High agreement establishes shared-implementation consistency, not independent physical truth.

The offline BM25 baseline retrieves one product into a fixed template; live cards omit relevant evidence fields: [baseline rendering](../src/baseline_rag.py:117), [offline output](../src/baseline_rag.py:160). A purpose-built SQL planner versus an impoverished answer template does not isolate a new scientific method. Index construction occurs before benchmark timing: [benchmark setup](../src/benchmark.py:138).

**Fix and acceptance:** independent expert/source-backed labels and separate verifier code; fair SQL-only, lexical, dense/hybrid, LLM-plus-SQL, and relevant graph baselines. Same source fields, model budget, query workload, inference constraints, and comparable answer opportunities. Measure ingestion/index build separately from warm/cold retrieval and generation. Add ablations of each proposed contribution; no component gets credit merely because an entire system beats a weaker template.

#### F27 · The reported scale comparison changes the workload denominator — P2 · A/S

The smaller WDC result has 39 cases including ten absent-product cases; the larger result has 29 without them. The systems' direct-price family performance is unchanged. The widening headline difference corresponds to **8/39 versus 8/29**, not demonstrated baseline deterioration with catalogue size. Compare [smaller result](real_wdc_20260929.json), [larger result](real_wdc_scale_20260929.json), and [interpretation](08_REAL_DATA_VALIDATION.md:356).

**Fix and acceptance:** use paired, versioned query sets across sizes; separately report changes in relevance/identity/absence truth caused by corpus growth. Hold model, filters, budget, hardware, and indexing state constant. Report confidence intervals, timeouts, resource usage, and both entity diversity and row count. Do not present repeated listings as semantic diversity.

#### F28 · Fiduciary “certification” exceeds its bounded experiment — P2 · S

A single intervention and source-token inspection do not prove invariance to all commercial influence. Rounded deviation can hide a real change; dynamic-only or empty comparisons can appear certified: [deviation](../src/fiduciary_audit.py:147), [certification](../src/fiduciary_audit.py:274). A deterministic experimental scorer is negative price on a preselected set: [experiment](../src/fiduciary_experiment.py:97).

**Fix and acceptance:** label bounded empirical intervention tests as such; require nonempty meaningful comparisons; compare unrounded deviations; include missing-item penalties and multiple sponsorship/commission/role interventions plus same-condition repeats. Measure recommendation utility alongside invariance. Invariance of empty or useless output is not fiduciary success.

### 4.4 Security, lifecycle, delivery, and scalability

#### F29 · Unauthenticated source registration exposes local SQLite scope — P0 · S

The registration endpoint accepts a client-supplied server filesystem path and checks only existence. It has no authentication, tenant ownership, path confinement, read policy, uniqueness policy, or persistent registration transaction: [registration](../src/server.py:548). The registry can overwrite an existing domain entry. Subsequent query execution can access the chosen SQLite database through the same unrestricted read interface.

**This is not a demonstrated arbitrary-text-file reader or OS privilege escalation.** The supported risk is unauthorized registration/querying of process-readable SQLite databases, registry interference, and costly initialization, if the API is exposed to an attacker.

**Fix and acceptance:** authenticated tenant-scoped upload/connect jobs; source ownership and per-table/column permissions; controlled storage roots and safe path resolution for local connectors; extension/content validation; reject escaping symlinks and unexpected network paths; immutable source IDs; transactional version publication. Public widget callers cannot administer sources or models. Test other-tenant resources, outside-root SQLite, duplicate IDs, source quota exhaustion, and unauthorized sensitive columns.

#### F30 · Global state, caller-selected sessions, and telemetry violate isolation — P0 · S

Active domain and model are shared mutable process state: [domain switch](../src/server.py:578), [model switch](../src/server.py:367), [per-query mutation](../src/server.py:618). Session IDs are caller-selected with a shared default: [request model](../src/server.py:294). Telemetry returns the latest full stored session to unauthenticated callers: [session telemetry](../src/server.py:382). This is sensitive data exposure and cross-client interference; grounding contamination is separately reproduced by F07.

Saved state is loaded only after a new agent run and only a subset is updated: [persistence](../src/server.py:673). The execution does not actually continue that stored conversation/belief. A web caller choosing a gateway-like session ID can collide with its persistence namespace; that does not establish that prior dialogue was fed into the model.

CORS accepts all origins while enabling credentials: [CORS policy](../src/server.py:74). Exact wildcard/mirrored-origin behavior depends on request/cookie handling; do not claim every credentialed browser response is automatically readable. CORS is not authentication, and permissive origins make the unauthenticated surface easier to call from unrelated sites.

**Fix and acceptance:** server-owned session identity bound to tenant/principal/channel, explicit authorized knowledge scope, immutable request model choice, per-session preferences, authenticated/redacted telemetry, and allowed embed origins. Remove or admin-gate global switches. Test parallel tenants, session fixation/collisions, omitted IDs, model races, unauthorized telemetry, and legitimate cross-origin embedding. Persistence and conversational continuity need independent tests.

#### F31 · SQL protection is not AST validation or a resource sandbox — P0/P1 · R/S

The sanitizer is token/regex-based, not an AST validator: [sanitization](../src/sanitizer.py:37). Parameterized limits are accepted without checking the bound value (P11): [limit logic](../src/sanitizer.py:78). Read-only URI failure falls back to a normal writable connection: [connection fallback](../src/sanitizer.py:180). Output limits do not bound joins, sorting, aggregates, recursion, blobs, or CPU work before the first row. No query-time progress handler or total execution budget is enforced in this path.

**Fix and acceptance:** trusted query compilation/AST-aware permitted projections; fail-closed read-only connections; database authorizer and tenant/table/column policy where supported; limit all output encodings including parameters; bound bytes, runtime, recursion, and total work; cancellation and connection cleanup. Correctly handle quoted identifiers and literals rather than searching for the first occurrence of a limit token. Parser validation alone is not security; neither statement completeness nor an explain plan substitutes for an access policy.

**Not demonstrated:** SQL injection in the API's parameterized catalogue search or a successful mutating-query bypass. Preserve its existing parameter binding and the blacklist's useful defense-in-depth while repairing the architecture.

#### F32 · Untrusted source content enters elevated prompt context — P0/P1 · S

Schema descriptions, sampled values, and tool observations are inserted into model context; observations are appended as system-role messages: [tool observations](../src/react_agent.py:640), [schema prompt](../src/openrouter_client.py:116). The direct client may substitute provider reasoning content for ordinary content: [provider content](../src/openrouter_client.py:241). Source-derived text may contain adversarial instructions, personal information, or confidential fields.

**Fix and acceptance:** treat source text as untrusted data, not authority; delimit/structure it with typed tool results and trust labels; minimize fields and redact sensitive data before provider transmission; enforce allowed operations outside the model. Do not expose internal reasoning or raw provider details to public clients. Prompt instructions cannot prove resistance. Test malicious descriptions, EAV values, schema names, and documents attempting policy override, cross-tenant reads, fabricated SQL facts, or unsafe output. Live attack success rates remain U until measured.

#### F33 · Cold start and updates have no robust job/snapshot lifecycle — P1 · S/A

Registration and first domain access build profilers/graphs/ontologies/questions synchronously: [initialization](../src/server.py:193). Ontology construction reads every main-table row and performs grouped/per-column scans: [full read](../src/ontograph.py:347). Cache identity uses only absolute database path: [cache key](../src/ontograph.py:313); category/profile caches are not tied to source versions. Missing source paths can fall back to the unrelated default PC database: [effective database](../src/server.py:232).

The ontology prompt currently caps cleaned category examples at 60: [prompt sampling](../src/ontograph.py:137). Therefore the historical 30-second client timeouts do **not** prove “all 364 categories were sent to the model” in the current code. Provider retries/timeouts, profiling, full reads, and question preparation are distinct possible contributors; saved client timeouts contain insufficient stage evidence to apportion them.

**Fix and acceptance:** separate ingestion/index preparation from request execution through persistent idempotent jobs, source-version snapshots, bounded sampling, progress/cancellation, and explicit ready/degraded/failed states. Publish complete validated snapshots atomically; pin each request to a version. No default-domain fallback for a missing registered source. Test runtime registration, concurrent initialization, corrupt/stale cache, database replacement, deletions, failure/retry, and worker restart. Prewarming the initial domains alone does not solve later registration.

#### F34 · There is no evidence of arbitrary-scale search — P1 · S/U

Some code uses useful category/attribute indexes and paginated SQL. Other paths load/sort all rows: [literal mention resolver](../src/catalog_query.py:137), [ontology read](../src/ontograph.py:356); profiling counts distinct values per column: [profile scans](../src/profiler.py:126). Catalogue listing can trigger one metadata lookup per returned row: [page projection](../src/server.py:434). Page limits are uncapped. Pair-link generation is quadratic: [visualization loop](../src/server.py:719). The lattice enumerates a Cartesian product; its exact complexity is conditional on attributes/candidate bounds.

**Fix and acceptance:** streaming/chunked ingest, indexed full-source retrieval, bulk hydration, bounded result bytes, incremental statistics, workload-aware cache/index design, and proper admission control. Use implicit constraint evaluation rather than materializing all item combinations. Separate approximate search coverage from exact facts. Validate on progressively larger and more diverse corpora with concurrent tenants, realistic updates, hardware/resource disclosure, cold/warm latency, build time, recall, memory, disk, provider usage, and failure/cancellation behavior. No absolute “any size” promise is technically supportable.

#### F35 · Widget, federation, and omnichannel product delivery are incomplete — P1 · S

The web application is a first-party demo with direct relative API calls and hardcoded scenarios/scale text, not a tenant-scoped embeddable SDK: [network wrapper](../web/app.js:376), [scale label](../web/app.js:832), [scenario examples](../web/app.js:1278). Returning illustrative graph links is not evidence of a complete semantic map; pairwise edges do not express every n-ary constraint.

The Discord runner constructs a gateway but does not start a listener: [runner](../src/runners/run_discord.py:25). Gateway runners import the server global context, and the Telegram runner's independently created session store is not the executed persistence path: [Telegram runner](../src/runners/run_telegram.py:15), [gateway dispatch](../src/gateways/base_gateway.py:37). Telegram inserts answer text into HTML mode without a proper escaping boundary: [formatter](../src/gateways/telegram_adapter.py:35).

**Fix and acceptance:** deliver the website widget first with scoped public tokens, origin authorization, lifecycle/versioning, accessibility, rate limits, session isolation, and explicit errors. Add federation as evidence-linked querying of authorized sources, not concatenation under shared names. Implement and integration-test actual gateway listeners, state restoration, platform formatting/limits, retry/idempotency, and cancellation before advertising omnichannel support.

#### F36 · Rebuild scripts and deployment are not reproducible or safe — P1/P2 · S

Initializers/importers remove existing target databases without an explicit safe overwrite workflow: [ingest rebuild](../data/ingest_real_data.py:191), [marketplace rebuild](../data/generate_marketplace_scale.py:14). The PC initializer's main call omits a required argument: [entry point](../data/init_database.py:220). The commercial migration points one directory too deep: [migration path](../data/migrate_commercial.py:19). Fixture creation contains an author-machine absolute path: [fixture root](../data/make_fixtures.py:15). Long EAV values are dropped without matching loss counters: [value filtering](../data/ingest_real_data.py:236).

There is no project dependency/lock manifest in the inspected checkout. The [README](../README.md) has stale endpoint/test-count claims and installation instructions. Ignoring architecture and decision documents means a clean checkout may omit the actual governing specification: [ignored documents](../.gitignore:62).

**Fix and acceptance:** explicit safe target/overwrite flags, atomic build-validate-publish, backups and rollback, portable paths, per-reason loss counters, pinned dependencies and supported interpreter versions, clean-clone CI, actual API examples, and version-controlled architecture/research decisions. Do not silently destroy or manually repair customer evidence to improve a benchmark.

#### F37 · Existing audit closures and scientific positioning are overstated — P0/P2 · S

The older idea audit says all C1–C7 are fully closed and the shell matches the core: [closure claims](10_IDEA_AUDIT_20261001.md:67). Current evidence supports **partial** closure:

| Old closure | Current assessment |
|---|---|
| C1 domain-specific routing | Real improvement; isolation, missing-source fallback, and shared request state remain |
| C2 no domain dictionaries | False: known-key dictionaries remain, even though unknown keys now pass through |
| C3 genuine induction | False as zero-domain logic: multilingual domain keywords are still domain keywords |
| C4 arbitrary schema registration | A new API exists; complete arbitrary-schema adaptation and security do not |
| C5 products moved out of code | Genuine improvement; authored SQLite fixture is not independent real-world fitment truth |
| C6 marketplace diversity fixed by clustering | Clustering cannot manufacture diversity; live use and conflict/identity integrity remain incomplete |
| C7 complete paginated listing | Genuine improvement for listing; not complete graph/rule/resolution coverage |

**Fix and acceptance:** keep old reports as historical artifacts with a visible superseded notice and a current claim register. Reopen affected IDs; mark partial versus fixed versus verified. A fix must close the original requirement, not a narrower surrogate. Update paper and architecture capability claims only from actual integration and independently meaningful measurements.

## 5. Scientific novelty: what can and cannot be claimed

### 5.1 Established building blocks are not new because renamed

Graph-based RAG, constraint satisfaction, formal concepts, lattice/semilattice merging, Pareto optimization, Bayesian preference elicitation, and text-to-SQL are established research areas. Combining them can be engineering value; it is not proof of inventing previously nonexistent algorithms.

Primary-source checks during this audit include:

- [GraphRAG, Edge et al., 2024/2025](https://arxiv.org/abs/2404.16130): automatic graph indexing and community summaries over document collections already exist. This is overlap in graph-based corpus consultation, not proof that the exact proposed method is already published.
- [Spider, Yu et al., EMNLP 2018](https://aclanthology.org/D18-1425/): evaluates new database schemas and domains using separated databases. Reusing one renamed product schema is not that level of generalization.
- [CRDT survey, Preguiça, Baquero, Shapiro, 2018](https://arxiv.org/abs/1805.06358): convergent, mathematically specified merge behavior is established prior art. A new application must identify its actual additional method and assumptions.

These primary metadata/abstract checks are **not** an exhaustive full-text priority search. The broader bibliography in [literature review](04_LITERATURE_REVIEW_AND_NOVELTY_GAP.md) needs an explicit claim-to-source matrix and independent verification. Do not call a citation fabricated merely because it is unfamiliar, or claim “first” because a short search did not find a match.

### 5.2 Recommended research focus

**Candidate contribution, not proven novelty:** confidence-calibrated, provenance-preserving adaptation of heterogeneous imperfect data, with explicit conflicts and selective constraint acquisition, evaluated by factual usefulness on held-out schemas/domains.

Narrow it to one primary method and a few testable propositions:

1. A closed typed value representation and conflict-preserving merge reduce incorrect cross-source equivalence relative to appropriate normalization baselines while preserving coverage.
2. Structural relation hypotheses plus independent validation reduce false compatibility verdicts compared with name-dictionary induction, especially after key/category anonymization and domain shift.
3. A claim/derivation contract improves useful answer delivery without allowing claim fabrication through SQL, prose, or incomplete observations.
4. Incremental bounded graph/index maintenance preserves quality under updates at a measured resource cost.

For each proposition specify the mathematical input/output domain, assumption set, algorithm/pseudocode, complexity, counterexamples/limits, strongest relevant baselines, independent labels, splits, ablations, and confidence intervals **before** test-set tuning. A negative result is valid research; an overstated certificate is not.

The four conditional constraint classes can remain evaluators. Do not require the system to infer unknowable physical laws from correlations or to build a complete graph for every piece of data. The scientific question concerns **when adaptation is justified and what evidence makes it useful**, not a universal magic oracle.

## 6. Ordered remediation plan

This plan is future work. Nothing below is marked implemented by this audit. Each work item must have an owner, requirement, failing-before test, source diff, after result, immutable evidence artifact, and residual-limit statement. Use the mandatory workflow in [agent instructions](../AGENTS.md). File organization, code cleanup, and repository revision are split out into the separate [repository cleanup and revision plan](13_REPO_CLEANUP_AND_REVISION_PLAN.md) (C0–C4), which is subordinate to this order.

| Stage | Deliverable and dependencies | Closes | Release gate |
|---|---|---|---|
| R0 · Freeze and correct measurement | Preserve raw evidence; fix task/claim scoring and independent oracles; reopen false closure claims | F01, F25–F28, F37 | Invalid-answer mutations cannot pass; old/new denominators and versions are explicit |
| R1 · Close P0 trust and exposure defects | Request-scoped evidence; strict protocol; trusted factual projection; bound claims; actual gate outcomes; source/admin/session authorization | F02–F08, F15, F18, F29–F32 | All relevant local counterexamples become regression assertions of safe behavior; isolation/adversarial HTTP tests pass |
| R2 · Repair deterministic foundations | Closed typed algebra; conflict/identity model; sound graph pruning; complete/unknown solver outcomes | F16, F19–F21 | Property/differential tests plus independent numeric/small-instance oracles pass |
| R3 · Build real adaptation and lifecycle | Canonical source capabilities; discover/validate mappings and rules; versioned background ingest; full-source entity indexes; no unrelated-source fallback | F09–F14, F17, F33–F34 | Unseen library/forum/EAV sources work without domain-code changes; updates/dirty data have accounted outcomes |
| R4 · Deliver useful conversation and platform | Task-aware ranking/relaxation; persistent preferences; verified text/document retrieval; widget isolation; real gateway integration if retained | F22–F24, F35–F36 | End-to-end client tasks meet the [target contract](12_TARGET_SYSTEM_AND_ACCEPTANCE.md), including honest unknowns |
| R5 · Demonstrate research result and measured scale | Paired held-out corpora, independent constraints, fair baselines, ablations, live trials, resource/load tests | F12, F24–F28, F34, F37 | Reproducible report supports only the contribution and scale envelope actually measured |

Do not start with brand stemming, JSON cleaning, or top-N category clipping. Those are at most small substeps after the correct contracts exist. Do not wait until R5 to make measurements honest; R0/R1 are prerequisites for meaningful later results.

### 6.1 Required regression families

- Protocol and no-evidence finalization; provider errors; typed tool arguments; explicit failure classification.
- SQL literal/computed fact laundering; entity–attribute–offer binding; negative claims; valid derivations.
- Cross-session/request/tenant grounding and model/domain races; sensitive-data/provider transmission boundaries.
- Unknown rule/entity/attribute coverage, safety vetoes, compatibility versus joint checkout.
- Rename/anonymize schemas and domains; price-less and multi-table sources; dirty/partial/contradictory data.
- Closed projection/merge laws; stable IDs and source-linked conflicts; indexed versus exhaustive graph results.
- Solver completeness/truncation/optimality; preference updates; heterogeneous ranking axes; consented soft relaxation.
- Source replacement/deletion/incremental ingest; cache invalidation; job idempotency; backup/rollback.
- Independent task-oracle and evaluator mutation tests; same-query paired scale trials; fair ablation and baseline tests.
- Widget origin/accessibility/lifecycle; actual gateway listeners; escaping and platform limits.

Every claimed repaired item needs a test that would fail on the snapshot described here. Merely checking response status, existence of a string, or an “ACTIVE” badge is insufficient.

## 7. Strengths and non-findings to retain

1. Real SQL parameter binding on catalogue search and fast-path constructed queries is valuable; no injection was demonstrated there.
2. Optional-column introspection for performance/stock is a genuine improvement; extend it to semantic capabilities rather than deleting it.
3. Attribute-conflict states, unknown-unit dimensions, regression fixtures, deterministic degradation, and ingestion metadata are useful foundations, despite specific correctness gaps.
4. Real timing deltas and stored raw answer artifacts are worth preserving; correct their interpretation and missing coverage.
5. Database-backed auto data, per-domain agents, and direct SQL pagination genuinely improve previous demo-only behavior.
6. Disclosure of prior fabricated measurements in [paper draft](02_RESEARCH_PAPER_DRAFT.md:3) is appropriate. Do not replace that disclosure with a new unsupported success claim.
7. The web application escapes many dynamic strings. An HTML sink alone is not evidence of a demonstrated browser XSS; the Telegram formatting issue is a distinct boundary.

## 8. Final state statement

**Current state:** research prototype with reusable components, substantial tests, demonstrable trust-boundary failures, incomplete zero-config adaptation, and unproven end-to-end scientific guarantees.

**Required next state:** not a prettier demo, but a system in which uncertainty, provenance, coverage, resource limits, and task outcomes are explicit and mechanically testable. The intended automatic experience is retained; claims of universal understanding, unlimited scale, or absolute hallucination immunity are not substituted for evidence.
