# 19 · top1_sim as a candidate "query-corpus fit" certificate: protocol frozen before the scored run

**Date frozen:** 2026-11-13. **Status:** protocol only. Every number below is a *proposed acceptance target*, not an observed result. Relaxing a target after seeing results requires a dated scope-change entry and a new, separately named result file.

Companion documents: [semantic-cycle protocol](18_SEMANTIC_CYCLE_PREREGISTRATION.md), [claim register](16_CLAIM_REGISTER.md).

## 1. What this study tests

The P7 endpoint in docs/18 asked for a confidence signal that separates answerable from unanswerable requests and fixed `top1_z` (z-score of the top-1 dense similarity within the result set) as the carrier. The registered signal **failed** (test AUROC ~0.32–0.55); the exploratory `top1_sim` (raw top-1 dense cosine) separated the dev/test unanswerable sets at AUROC ~0.93–0.99 — but those unanswerable requests were anachronistic (easy) and `top1_sim` was never a registered endpoint on unseen schemas.

This study promotes `top1_sim` to a **candidate certificate** — "the request fits this corpus" vs "no adequate record exists" — and scores it once, under a frozen protocol, on **two schemas never used in the retrieval evaluation** and never tuned on:

| Source | Why it qualifies | Size |
|---|---|---|
| `jury_zoo` | Built 2026-11-13 (`build_jury_demo.py`, seed 20261112) *after* all retrieval tuning; Ukrainian transliterated schema (`tvaryny`), never present in any dev/test evaluation | 420 entities |
| `eon_universe` | Game-world relic store (3 tables, Ukrainian, `relikvii` entity auto-selected); used in earlier UI/engine work but **never** in retrieval evaluation or tuning | 60 entities |

## 2. Signal and systems

- **Signal under test:** `confidence.top1_sim` returned by `GET /api/domains/{id}/search` (mode `hybrid`, k=10) — the dense cosine of the best hit. One HTTP call per request through the live server; no special evaluator path.
- **Exploratory-only signals** (recorded, never substituted): `margin` (top1−top2 dense sim), `top1_z`, lexical top-1 score.
- Baseline for context: the same queries also run `mode=lexical`; lexical top-1 score AUROC is reported as a reference, not a competitor.

## 3. Query set (frozen before the run)

File: `docs/top1sim_queries_20261113.json`, sha256 in `.sha256`, written once by `gen_top1sim_queries.py` and never edited.

- 12 `present` + 12 `absent` queries per source (48 total), mixed English and Ukrainian.
- `present`: at least one relevant record exists by construction of the corpus (evaluator rationale in `note`).
- `absent`: no relevant record exists. Roughly half are **hard negatives** — concepts adjacent to corpus content (a venomous desert lizard vs. the corpus's venomous desert scorpion; a sprinting big cat vs. actual big cats; firearms vs. a "зброя" category). Hard negatives are what make the certificate meaningful; easy cross-domain queries alone would not.
- Labels are authored against corpus content, never produced by the system under test.

## 4. Endpoint and proposed acceptance target

| ID | Endpoint | Proposed target |
|---|---|---|
| T1 | AUROC of `top1_sim` separating `present` from `absent`, **pooled** over both sources, single scored run | ≥ 0.80 |
| T2 | Same, **per source** (jury_zoo, eon_universe) | ≥ 0.70 each (small-n tolerance; report exact) |
| T3 | **Threshold calibration**: max accuracy of any single `top1_sim` cut-off on pooled labels | reported descriptively; no target — a threshold would need its own held-out study |

Interpretation contract: PASS on T1 means "the candidate certificate separates answerable from unanswerable on these two unseen schemas" — nothing more. It does not prove universality, does not repair the P7 failure (which stays failed on its registered carrier), and does not entitle the UI to any "verified" badge.

## 5. Rules

1. Freeze this document and the query file (hash recorded) before any scored run.
2. One scored run per source; raw per-request outputs stored in `docs/top1sim_eval_<date>.json`.
3. A failed endpoint stays failed in the report; narrative may explain it, never replace it.
4. Record exact commands, index generation, embedder `model_id`, and wall-clock.
5. The 24-per-source sample is small — report the AUROC point estimate plus a 95% percentile-bootstrap interval (10,000 resamples, seed 20261113) rather than implying precision that is not there.

## 6. Known validity limits

- n=48 total; bootstrap intervals will be wide. This is a screening gate, not a proof.
- `absent` labels are evaluator-authored; some hard negatives could legitimately match a broad interpretation (a "healing device" query vs. a medicinal relic) — the label, not the model's opinion, is ground truth.
- Both corpora are small and partly synthetic; absolute `top1_sim` levels depend on the e5-small model and cannot be read as calibrated probabilities.
- `top1_sim` is monotonically related to nothing else claimed; a PASS here coexists with the P1/P3/P7 failures of docs/18 and does not amend them.

## 7. Scope-change log

(none yet)
