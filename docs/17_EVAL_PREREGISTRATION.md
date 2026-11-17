# 17 · Evaluation preregistration (frozen protocol)

> Frozen BEFORE the expanded held-out benchmark run. Changing a target
> or a denominator after seeing results requires a documented scope
> change and a NEW report id -- old reports stay immutable in
> `bench/results/`.

## Scope of what is measured

`bench/run_recall_bench.py` measures **channel reachability**: given a
term that exists in the source only behind a particular structural
pattern, which retrieval method can bind it to the correct entity?

- It does NOT measure real-world relevance, ranking quality, or model
  hypothesis quality on held-out schemas.
- Ground truth is hand-labelled per fixture and disclosed verbatim in
  the report.

## Methods under test

| Method | Definition |
|---|---|
| `fts` | conjunctive LIKE over all entity-table text columns -- the strongest lexical fixed-schema baseline |
| `ebr` | `relax_search` conjunctive items, regret-0 results only |
| `ebr_vec` | `ebr` + dense vector channel (scripted deterministic embedder -- tests the channel contract, not a paid model's quality) |
| `vec_only` | pure dense baseline: conjunctive intersection of vector bindings alone -- isolates what the vector channel contributes |

## Metrics (frozen)

- `recall` = |found ∩ truth| / |truth| per query; mean per shape.
- `precision` = |found ∩ truth| / |found| per query (flags over-broad hits).
- `regret` = dropped terms (EBR relaxation ledger).
- `degraded_ok` for ambiguity fixtures: pass only when the detector
  reports NO mapping rather than guessing (honest-degradation check,
  not a recall target).
- `ms` per method per query (indicative latency, single run, marked as
  such -- not a load measurement).

## Frozen acceptance targets (for THIS synthetic polygon)

| # | Statement | Target |
|---|---|---|
| A1 | EBR ≥ FTS on every shape (no regression) | recall_ebr >= recall_fts per query |
| A2 | EBR > 0 on attribute-channel shapes (eav, fk_dim, junction, inferred) | recall_ebr = 1.0 |
| A3 | FTS = 0 on attribute-only shapes | recall_fts = 0.0 (confirms baseline blind spot) |
| A4 | Ambiguous EAV fixture yields NO detected table | degraded_ok = true |
| A5 | vec_only binds the paraphrase case where lexical methods = 0 | recall_vec = 1.0 on `paraphrase` |
| A6 | Zero-truth query produces zero fabricated hits | found = ∅, regret recorded |
| A7 | multi-hop (2-hop dimension chain) measured honestly | record actual recall; 0 is an acceptable documented result while >2-hop traversal is unimplemented |

Targets are for the synthetic polygon only. They do NOT establish the
docs/12 §7 goals (G1-G12), which need the full held-out source
diversity polygon.

## Fixture disclosure

All fixtures are synthetic SQLite, generated in `tempfile` dirs, seed
fixed (SEED=42), entity count 40. Renamed-identifier fixtures use
obfuscated table/column names (`t9`, `b2`) as rename controls. The
`inferred`/`inferred_junction` shapes simulate the post-conductor state
via the SAME overlap>=0.98 verification -- they do not run a live model.

## Known limitations (declared in advance)

- Single seed, deterministic fixtures -- confidence intervals are
  degenerate; repeats would be identical by construction.
- Scripted embedder embeds a disclosed synonym table -- it proves the
  vector channel binds correctly, not that a real model embeds well.
- `inferred*` shapes skip the live conductor; conductor hypothesis
  quality on unseen schemas is a separate unmeasured quantity.
