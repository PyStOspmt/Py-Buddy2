# Py-Buddy Benchmark & Verification Report

All numbers below come directly from the versioned JSON reports in [`docs/benchmarks/`](benchmarks/) and [`bench/results/`](../bench/results/). They distinguish live LLM runs, offline deterministic core scaling, schema-agnostic retrieval probes, and synthetic controls.

---

## 1. End-to-End Multi-Model Evaluation (Live LLM)

Source: [`docs/benchmarks/multi_model_eval_20261116.json`](benchmarks/multi_model_eval_20261116.json) (superseding [`multi_model_eval_20261113.json`](benchmarks/multi_model_eval_20261113.json) after removing all production domain seed vocabularies, enforcing `F12` unknown compatibility checks, and propagating source currency)  
Catalog: `marketplace_rozetka.db` · Single-turn suite: `N = 73` (`30` verdict cases) · Offline suite ([`benchmark_offline_20261116.json`](benchmarks/benchmark_offline_20261116.json)): **`100.0%`** acc / `100.0%` NVFG (`0.002s` avg latency vs baseline `40.82%` / `0.022s`) · Total multi-model live API spend: `$0.9529`

| Model | Arm | Exec Acc (%) | NVFG (%) | Hallucination Rate (%) | False Claim Rate (%) | Price Error Rate (%) | Verdict Acc (%) | Avg Cost / Query |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `deepseek/deepseek-v4.1-flash-1017` | Baseline RAG | 47.95 | 56.16 | 8.22 | 14.55 | 9.28 | 56.67 | $0.000213 |
| `deepseek/deepseek-v4.1-flash-1017` | Pre-Verifier Ablation | 83.56 | 84.93 | 9.59 | 32.53 | 1.54 | **90.00** | $0.004238 |
| `deepseek/deepseek-v4.1-flash-1017` | **Py-Buddy (Full)** | **83.56** | **84.93** | **6.85** | 31.71 | **0.54** | **83.33** | $0.003766 |
| `openai/gpt-6-luna` | Baseline RAG | 50.68 | 63.01 | **0.00** | 14.58 | **0.00** | 60.00 | $0.000101 |
| `openai/gpt-6-luna` | Pre-Verifier Ablation | 73.97 | 75.34 | 2.74 | 3.61 | 1.37 | 53.33 | $0.001418 |
| `openai/gpt-6-luna` | **Py-Buddy (Full)** | **83.56** | **83.56** | 2.74 | **3.49** | 1.14 | **76.67** | $0.001274 |
| `xiaomi/mimo-v2.6-flash` | Baseline RAG | 53.42 | 63.01 | **5.48** | **11.48** | 14.29 | **73.33** | $0.000132 |
| `xiaomi/mimo-v2.6-flash` | **Py-Buddy (Full)** | **79.45** | **80.82** | **5.48** | 34.86 | **1.18** | 70.00 | $0.002785 |
| `anthropic/claude-haiku-5.5` | Baseline RAG | 49.32 | 61.64 | 12.33 | **11.67** | 19.54 | **80.00** | $0.000233 |
| `anthropic/claude-haiku-5.5` | **Py-Buddy (Full)** | **73.97** | **78.08** | **10.96** | 21.01 | **4.27** | 76.67 | $0.003206 |

### Key Takeaways & Honest Boundaries
- **Substantial execution accuracy gain with sharp price-error reduction:** Py-Buddy improves execution accuracy by **+24.65 to +35.61 percentage points** over live `LLM-RAG` (`73.97%–83.56%` vs `47.95%–53.42%`), scores **`100.0%`** across all four models on `direct_price` (15/15), `cheapest_category` (6/6), and `priciest_category` (4/4), and cuts price errors sharply (`Haiku`: `19.54% -> 4.27%`, `MiMo`: `14.29% -> 1.18%`, `DeepSeek`: `9.28% -> 0.54%`; `Luna` has 1 strict revision-attribution flag at `1.14%` on `bc042`).
- **Epistemic conservatism (`F12`):** On unverified compatibility pairs (`compat_accept`, `n = 12`), Py-Buddy enforces `verdict="unknown"` whenever compatibility constraints are automatically induced rather than operator-verified, refusing to fabricate positive guarantees (`compat_reject` stays `75.0%–91.67%` accurate / `100.0%` verdict on Luna, DeepSeek, MiMo).
- **Sentence-level verifier replay** ([`claim_verifier_replay_20261116.json`](benchmarks/claim_verifier_replay_20261116.json), alongside historical [`20261113`](benchmarks/claim_verifier_replay_20261113.json)): Across `49` replayed cases, `ClaimVerifier` triggered **`0` false flags** on all `33` clean responses (`0.0%` false-positive rate). In the 4-model live suite, `ClaimVerifier` intervened in **77 cases / 149 unsupported sentences**.

Reproduce locally (requires OpenRouter key):
```bash
python src/experiments/benchmark_aic2026.py --suite expanded --models openai/gpt-6-luna --out bench/results/custom_run.json
```

---

## 2. Schema-Agnostic Constraint Induction (CIP)

Source: [`docs/benchmarks/cip_benchmark_20261004.json`](benchmarks/cip_benchmark_20261004.json) and `non_english_induction` in [`multi_model_eval_20261116.json`](benchmarks/multi_model_eval_20261116.json).

Evaluates automatic discovery of compatibility constraints (equivalences `eq(...)` and unit/dimension containment `le(a <= b)`) across `pc_build` (`15` items), `auto_parts` (`9` items), and `pharma` (`5` items), plus full bilingual column/key anonymization:

| Method | Precision (%) | Recall (%) | F1 (%) | TP / FP / FN | Anonymized invariance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Exact Header Matching (Baseline) | 28.57 | 18.18 | 22.22 | 2 / 5 / 9 | Degrades on anonymized keys |
| **Py-Buddy Structural Induction** | **64.71** | **100.0** | **78.57** | **11 / 6 / 0** | **100% invariant** across all 3 domains |

On a Ukrainian non-commerce SQLite catalogue (`jury_zoo`, `420` entities, `1,519` EAV rows), both `ContractInductor` and `induce_structural` discovered **`7` structural rules** (equivalences + dimensional/capacity containment such as `висота <= зріст`, `висота <= розмах`, `розмах <= зріст`, and habitat-scoped weight bounds) zero-shot without any English or domain seed vocabularies.

---

## 3. Public Real-World Catalogues & Retrieval Recall

Sources: [`real_off_20260929.json`](benchmarks/real_off_20260929.json), [`real_prices_20260929.json`](benchmarks/real_prices_20260929.json), [`real_wdc_20260929.json`](benchmarks/real_wdc_20260929.json), [`real_wdc_scale_20260929.json`](benchmarks/real_wdc_scale_20260929.json), [`heldout_bench_1791313872.json`](../bench/results/heldout_bench_1791313872.json), [`recall_bench_1791321685.json`](../bench/results/recall_bench_1791321685.json), [`live_vector_bench_1791311504.json`](../bench/results/live_vector_bench_1791311504.json), [`formats_bench_1791331396.json`](../bench/results/formats_bench_1791331396.json).

### Public Catalog Ingestion & Attribute Algebra

| Corpus | Products | EAV Rows | EAV Keys | FCA Concepts | Attribute Algebra Highlights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Open Food Facts** (`real_off`) | 3,000 | 184,158 | 237 | 161 | `10,999` values examined (`79.6%` canonicalisable); `110` extra equivalence pairs unified over string normalization (`50` substantive e.g. `0.075 g == 75 mg`, `60` zero-magnitude) |
| **Open Prices** (`real_prices`, EUR) | 6,554 | 55,260 | 12 | 11 | `100.0%` price coverage; `0% -> 100%` recall on structured `ONLINE` query vs FTS (`76.34%` coverage on `experiments`, `n=655`) |
| **WDC Products** (`real_wdc`, USD) | 3,781 | 15,896 | 158 | 25 | `13.01%` explicit price coverage preserved without synthetic zeros; `DDR4` recall `35.46% -> 100.0%`, `kingston ssd` `42.86% -> 100.0%` |
| **WDC Scale** (`real_wdc_scale`) | 6,877 | 29,102 | 184 | 27 | `8.38%` canonicalisable attributes; preserves heterogeneous units safely |

### Structural Schema Shapes (`recall_bench` & `live_vector_bench`)
Across 12 controlled relational/EAV schema shapes:
- Pure **FTS5 lexical search** returns **`0.0`** recall whenever target attributes reside in `eav`, `eav_renamed`, `fk_dim`, `multi_hop`, `junction`, `inferred`, `inferred_junc`, or `conj_mix`.
- **Evidence-Bound Relaxation (EBR)** achieves **`1.0` recall** across all reachable schema shapes (including multi-hop joins and renamed columns), while returning `0.0` on ungrounded `no_match` / `eav_ambiguous` controls.
- Adding the hybrid embedding channel boosts `paraphrase` probes from `0.0` (exact lexical) to **`0.5–1.0`** (`live_vector_bench` / `recall_bench`).
- Multi-format ingestion (`formats_bench`): JSON, Markdown tables, and PDF tables all scored **`1.0` recall / `1.0` precision** with `0` false positives.

---

## 4. Deterministic Scale & Memory Envelope

Sources: [`scale_envelope_20261004.json`](benchmarks/scale_envelope_20261004.json), [`scale_envelope_50k_20261004.json`](benchmarks/scale_envelope_50k_20261004.json), [`scale_envelope_wdc_20261004.json`](benchmarks/scale_envelope_wdc_20261004.json), [`scale_probe_1791327290.json`](../bench/results/scale_probe_1791327290.json).

| Corpus / Scale Tier | Products | EAV Rows | Build Time | Median Latency (p50) | p95 Latency | RSS Delta | Accuracy / Parity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Synthetic 200 | 200 | 585 | 0.013 s | 0.4 ms | 0.8 ms | +0.5 MB | 62.5% offline det. |
| Synthetic 2,000 | 2,000 | 5,714 | 0.052 s | 0.6 ms | 1.1 ms | +3.1 MB | 87.5% offline det. |
| Synthetic 10,000 | 10,000 | 28,659 | 0.243 s | 0.6 ms | 3.3 ms | +13.2 MB | 87.5% offline det. |
| Synthetic 25,000 | 25,000 | 71,950 | 0.540 s | 1.0 ms | 8.2 ms | +29.0 MB | 87.5% offline det. |
| Synthetic 50,000 | 50,000 | 144,049 | 1.080 s | 1.0 ms | 10.7 ms | +60.6 MB | 87.5% offline det. |
| Real WDC 3,781 | 3,781 | 15,908 | 0.886 s | 0.9 ms | 1.4 ms | +5.0 MB | 75.0% offline det. |
| **Scale Probe 500,000** | 500,000 | `product_eav_meta` | 5.30 s proj / 54.76 s FTS5 | 0.88 s indexed scan | — | — | `bindings_equal: True` |

---

## 5. Fiduciary Independence Verification (FID)

Sources: [`fiduciary_20261116.json`](benchmarks/fiduciary_20261116.json) (post zero-dictionary update, alongside [`fiduciary_20260930.json`](benchmarks/fiduciary_20260930.json) and [`fiduciary_live_20261004.json`](benchmarks/fiduciary_live_20261004.json)).

Adversarial sponsorship & commission permutation audit over `30` products (with injected `+1550.14` sponsorship control bonuses):

| Ranker | FID Score (lower = cleaner) | Displaced Items (`n=30`) | Status |
| :--- | :---: | :---: | :---: |
| **Py-Buddy Deterministic Core** | **0.000000** | **0** | **`CERTIFIED` (`FIC-1.1`)** |
| Sponsored-Biased Control | 0.148889 | 24 | `VIOLATION` (worst rank delta: 15) |
| Commission-Biased Control | 0.064444 | 20 | `VIOLATION` (worst rank delta: 14) |
