# 18 · Semantic adaptation cycle: protocol frozen before scored runs

**Date frozen:** 2026-11-12. **Status:** protocol only. Every number below is a *proposed acceptance target*, not an observed result. Tightening a target is allowed; relaxing one after seeing results requires a dated scope-change entry at the bottom and a new, separately named result file.

Companion documents: [target contract](12_TARGET_SYSTEM_AND_ACCEPTANCE.md), [audit](11_CURRENT_STATE_AUDIT_AND_REMEDIATION.md), [claim register](16_CLAIM_REGISTER.md).

## 1. What this study tests

The product idea (docs/12 §1, docs/09): a consultant that attaches to an *unfamiliar* source, adapts without domain code, does not hold the source in memory, finds records by meaning (including requests that share no words with the record), and absorbs point changes by touching only the changed rows. The study runs that whole cycle end to end through the real HTTP API and measures each link independently.

| ID | Claim under test (scoped) |
|---|---|
| C1 | **Adaptation.** The entity table and folded text structure are chosen correctly on unseen schemas without domain dictionaries. |
| C2 | **Coverage.** Every entity row is either indexed or excluded with a recorded reason; counts reconcile exactly; nothing is sampled away. |
| C3 | **Meaning-based retrieval.** Requests with no shared word stems (and cross-lingual Ukrainian requests) find the relevant records better than lexical baselines on the same evidence. |
| C4 | **Bounded memory.** Serve-time memory does not scale with source text volume or vector count. |
| C5 | **Point updates.** Insert/update/delete of single rows re-embeds exactly the changed rows; unchanged vectors are byte-identical; deletions are never served again. |
| C6 | **Honest uncertainty.** A confidence signal separates answerable from unanswerable requests, or its absence is reported. |
| C7 | **Consultant integration.** The chat agent reaches the index through a tool and its answer is traceable to retrieved records. |

Not claimed: any-database support, multi-tenant isolation, free-text claim verification (F03), production scale beyond measured sizes, or retrieval quality of any other embedding model.

## 2. Sources (held out from development)

Provenance and hashes: `docs/semantic_cycle_sources_20261112.json` (build script `src/experiments/semantic_cycle/build_sources.py`, seed 20261112).

| Source | Content | Schema shape | Records |
|---|---|---|---|
| library | Project Gutenberg catalogue sample (English, real metadata) | normalised: entity + 3 many-to-many bridges, declared FKs | 8,000 books (8,996 subjects, 6,978 authors) |
| forum | 20 Newsgroups posts, bodies only | entity + one FK dimension (board) | 6,000 posts, 20 boards |
| food | Open Food Facts extract (`data/real_off.db`, unchanged) | project-canonical schema + EAV | 3,000 products, 207,260 EAV rows |

Development used only tiny hand-made fixtures. Baseline entity detection on these sources before any change: `docs/semantic_cycle_baseline_entity_20261112.json` (library: wrong, forum: wrong, food: correct).

## 3. Targets, requests and independent ground truth

Request file: `docs/semantic_queries_20261112.json` (sha256 in `.sha256`), written once by `gen_queries.py` and never edited. LLM: `deepseek/deepseek-v4.1-flash-1017`, reasoning disabled, temperature 0.5, hard spend cap USD 0.40 for generation (all calls logged in `docs/semantic_queries_llm_log_20261112.jsonl`).

* **label_set targets (relevant set = every record carrying a data-owner label):** library subject headings (60 with 8–150 members), food `category_tag` values (60 with 8–150 members), forum boards (all 20). Labels were assigned by cataloguers/the newsgroup structure/the Open Food Facts community; the model never judges relevance.
* **record targets (one remembered record):** 100 random records per source.
* **Request types per target:** `en_natural` (any words), `en_strict` (no word whose 5-character stem appears in the target's reference text; programmatically validated, up to 2 regenerations, failures kept and flagged), `uk` (Ukrainian, Cyrillic only; cross-lingual over English/multilingual records).
* **Split:** `dev` / `test` by sha256 parity of `source|kind|target_id`. All tuning (fusion constant, confidence cut-offs, plan options) uses `dev` only. The `test` split is scored once per system version.
* **Unanswerable requests:** 20 anachronistic out-of-domain requests (modern technology absent from 1990s newsgroups, pre-1930 books and food data), English and Ukrainian, authored by the evaluator in `gen_queries.py`.

Known validity limits: requests are LLM-written (possible stylistic alignment with embedding models); label sets are noisy (cross-posted newsgroup posts, Open Food Facts hierarchy overlap, `en:null` style placeholders); no human relevance judgements; the unanswerable set only tests anachronistic absence.

## 4. Systems compared (identical evidence)

All use the same rendered documents from the same index: `bm25` (FTS5 unicode61, OR of tokens), `bm25_prefix` (adds language-agnostic prefix truncation), `dense` (intfloat/multilingual-e5-small, exact scan), `hybrid` (RRF, constant 60). Ablation: `raw_columns_bm25` (BM25 over entity-table text columns only, no folding of EAV/FK/bridge evidence).

## 5. Metrics

Label_set: Recall@k (|top-k ∩ R| / min(|R|, k)) for k = 10, 50; Precision@10; nDCG@10 (binary); MRR of the first relevant. Record: Recall@1/5/10/50, MRR. Macro-averages per source × family × request type × split, with 95% percentile-bootstrap intervals (10,000 resamples, seed 20261112). Failures (no index, error, timeout) count as zero hits and are reported separately; no case is dropped from a denominator. The `en_strict` rows that failed validation after retries are reported as their own group and excluded only from the "strict" headline, with counts.

## 6. Endpoints and proposed acceptance targets

| ID | Endpoint | Proposed target |
|---|---|---|
| P1 | Pooled `label_set` Recall@10 on `en_strict` and `uk`, test split, hybrid vs best lexical baseline | hybrid ≥ best baseline + 0.15 absolute, bootstrap lower bound > 0 |
| P2 | Same on `en_natural` | hybrid not worse than best lexical baseline by more than 0.03 |
| P3 | Pooled `record` Recall@10 on `en_strict` and `uk`, test split, hybrid | ≥ 0.50 (aspirational; a shortfall is reported as measured) |
| P4 | C4: `tracemalloc` peak during 20 searches on a 200k-vector synthetic store; peak USS growth while serving the 8k-book index (model excluded) | < 64 MB; < 150 MB |
| P5 | C5 on a copy of the library source: 18 inserts (held-out real records), 18 updates (content swapped with another real record), 18 deletes (sizes per scope-change entry 1) | embedded texts exactly 36; unchanged vector bytes identical; 0 deleted ids returned over all requests; new rows' Recall@10 within 0.10 of baseline rows; refresh wall time < 10% of the full build |
| P6 | C2 reconciliation on every source | `source_rows == indexed + sum(excluded)` exactly; `truncated_docs` reported |
| P7 | C6: AUROC of dense top-1 z-score separating test answerable requests from unanswerable requests | ≥ 0.80, else "no usable confidence signal" is the reported result |
| P8 | C1: entity-table accuracy on the labelled schema-shape polygon (≥ 10 shapes) plus the 3 held-out sources, after the fix; baseline (before) recorded | 100% on all three held-out sources; polygon accuracy reported per shape |
| P9 | C7: 12 chat requests per source through `/api/chat` (cheap model, budget cap USD 0.30) | report counts only: requests where the answer names at least one retrieved record; answers naming a record not in the retrieved evidence (must be 0 to claim traceability) |

## 7. Procedure and rules

1. Freeze this document and the request file (hash recorded) before any scored run.
2. Implement and tune only against `dev`; keep every intermediate result file under a new name.
3. Run the cycle once on `test` per system version; store raw per-request outputs.
4. Record exact commands, git state, library versions (torch, sentence-transformers, numpy, sqlite), hardware, embedder `model_id`, seeds, and wall-clock timings.
5. A failed endpoint stays failed in the report; narrative may explain it, never replace it.

## 8. Scope-change log

1. **2026-11-12, before any scored run.** P5 scenario sized 18/18/18 instead of 20/20/20 (embedded-text target 36 instead of 40): after the request set was frozen, only 36 of the 40 test-split library record targets had validated strict and Ukrainian requests, and updates/deletes must reuse those requests. No result had been observed.
2. **2026-11-12, after the dev-split runs (`docs/semantic_eval_dev2_dev.json`, `docs/semantic_eval_dev3_dev.json`) and before any test-split run.** Registered endpoints P1-P3 and P7 are UNCHANGED and are evaluated with the registered systems exactly as defined (`bm25`, `bm25_prefix`, `dense`, `hybrid` = RRF 1:1 with explicit weights, `raw_bm25`; P7 on `top1_z`). The changes below were chosen on the dev split only, are declared before the single test run, and are evaluated as ADDITIONAL systems -- they never replace a registered row. (a) Product hybrid default lexical weight 0.3 (`product_default`): dev mean over the four metric sets 0.411 vs 0.400 (RRF 1:1) vs 0.393 (dense); strict-request record hit@10 0.297 vs 0.228. (b) HyDE promoted from exploratory ablation to a product mode (`product_hyde`: model-written hypothetical record, query+record vector mix, RRF 1:0.3 with the lexical channel of the original query), evaluated on all request groups; dev record hit@10 on strict+Ukrainian requests 0.621 vs 0.307 for dense. (c) Embedder window 512 tokens instead of 256: no measurable quality difference on dev; it removes silent model-window truncation for library (3 -> 0 of 8,000) and forum (1,621 -> 19 of 6,000) and halves it for food (2,826 -> 1,525 of 3,000); the remaining truncation is reported, not hidden. (d) The dense top-1 cosine `top1_sim` is exposed as a confidence signal; P7 is still evaluated on the registered `top1_z` (dev AUROC 0.303 / 0.588 / 0.541: expected to FAIL), and `top1_sim`, `margin`, `lex_top` are reported as exploratory signals (dev AUROC of `top1_sim` 0.996 / 0.940 / 0.979, unanswerable requests are anachronistic and therefore easy). (e) The test run indexes and searches through the product service (`SemanticIndex`); registered baselines read the same store. (f) Logged LLM spend before the test run: request generation USD 0.0743, mutation scenario 0.0063, dev HyDE 0.1078 (total 0.188 of the USD 0.50 cap). Dev observations that motivated these changes: registered `hybrid` loses to `dense` on strict/Ukrainian record requests (-0.08 hit@10, CI excludes 0) because the lexical channel adds noise when no word is shared, and dev P1 missed its +0.15 target (+0.133); neither fact changes the registered targets.
