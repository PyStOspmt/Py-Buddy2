"""Mutation checks of the evaluation harness wiring.

A tiny SYNTHETIC library-like source is indexed with the labelled test
double `HashEmbedder` (logic only, not semantic quality). The harness must
(a) score a correct ranking at the top of every metric, (b) lose score
when the ranking is shuffled or the target removed, (c) report failures
instead of dropping requests, and (d) reproduce the independent-BM25
differential. If these fail the harness cannot be trusted to grade the
real run.
"""

import os
import random
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "experiments", "semantic_cycle"))

import eval_core as C
import eval_report as R
from semantic_fixtures import HashEmbedder, projection
from semantic_sync import sync_source
from vector_store import VectorStore

TOPICS = {"astronomy": "stars planets telescope orbit comet",
          "cooking": "recipe kitchen oven flour sugar",
          "sailing": "harbour sails mast anchor voyage"}


def _make_source(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE topics(tid INTEGER PRIMARY KEY, tname TEXT);
    CREATE TABLE volumes(vid INTEGER PRIMARY KEY, title TEXT,
                         tid INTEGER REFERENCES topics(tid));
    """)
    for i, name in enumerate(TOPICS, 1):
        conn.execute("INSERT INTO topics VALUES (?,?)", (i, name))
    n = 0
    for i, (name, words) in enumerate(TOPICS.items(), 1):
        for j in range(12):
            n += 1
            conn.execute("INSERT INTO volumes VALUES (?,?,?)",
                         (n, f"{words} chapter {j}", i))
    conn.commit()
    conn.close()


@pytest.fixture()
def indexed(tmp_path):
    db = str(tmp_path / "lib.db")
    _make_source(db)
    emb = HashEmbedder(dim=128)
    store = VectorStore.open(str(tmp_path / "ix"), model_id=emb.model_id,
                             dim=emb.dim)
    rep = sync_source(db, projection(db), store, emb)
    assert rep.complete
    yield db, store, emb
    store.close()


def _req(kind, target, relevant, q):
    return {"rid": f"x|{kind}|{target}|en_natural", "source": "x",
            "kind": kind, "target_id": target, "group": "en_natural",
            "query": q, "relevant": set(relevant)}


def test_scores_perfect_ranking_high_and_shuffled_low(indexed):
    _db, store, emb = indexed
    rel = {str(i) for i in range(1, 13)}                   # astronomy rows
    req = _req("label_set", "astronomy", rel, TOPICS["astronomy"])
    res = {req["rid"]: C.run_modes(store, emb, req["query"])}
    good = {r["mode"]: r["m"] for r in R.score_all([req], res)}
    assert good["hybrid"]["recall@10"] == 1.0
    assert good["dense"]["precision@10"] == 1.0
    rng = random.Random(0)
    shuffled = {m: {**rec, "ids": rng.sample(rec["ids"], len(rec["ids"]))}
                for m, rec in res[req["rid"]].items()}
    bad = {r["mode"]: r["m"] for r in R.score_all(
        [req], {req["rid"]: shuffled})}
    assert bad["hybrid"]["precision@10"] < good["hybrid"]["precision@10"]


def test_removed_target_scores_zero_and_errors_are_counted(indexed):
    _db, store, emb = indexed
    req = _req("record", "999", {"999"}, TOPICS["sailing"])
    res = {req["rid"]: C.run_modes(store, emb, req["query"])}
    rows = R.score_all([req], res)
    assert all(r["m"]["hit@10"] == 0.0 for r in rows)
    res[req["rid"]]["hybrid"]["error"] = "RuntimeError: boom"
    res[req["rid"]]["hybrid"]["ids"] = []
    agg = R.aggregate(R.score_all([req], res), n_boot=200)
    cell = next(c for c in agg if c["mode"] == "hybrid")
    assert cell["n"] == 1 and cell["failures"] == 1


def test_evaluator_rrf_hand_computed_and_weighted():
    """Evaluator-owned fusion used for the HyDE+lexical systems: hand
    computed (k=60): a=1/61, b=1/62+0.5/61, c=1/63+0.5/62, d=0.5/63."""
    import eval_hyde as H
    got = H.rrf([["a", "b", "c"], ["b", "c", "d"]], [1.0, 0.5])
    assert got == ["b", "c", "a", "d"]
    assert H.rrf([["a", "b"], ["b", "a"]], [1.0, 0.0]) == ["a", "b"]
    assert H.rrf([["a", "b"], ["b", "a"]], [0.0, 1.0]) == ["b", "a"]
    assert H.rrf([["a", "b", "c"]], [1.0], top=2) == ["a", "b"]


def test_p7_auroc_per_signal_hand_computed():
    """Perfect separation -> 1.0, inverted -> 0.0, missing values ignored;
    the pre-registered signal is top1_z."""
    ans = {"s": [{"top1_z": 5.0, "top1_sim": 0.9, "margin": None,
                  "lex_top": 3.0}] * 3}
    una = {"s": [{"top1_z": 1.0, "top1_sim": 0.95, "margin": None,
                  "lex_top": None}] * 2}
    out = R.endpoint_p7(ans, una)
    assert out["primary_signal"] == "top1_z"
    assert out["signals"]["top1_z"]["per_source"]["s"]["auroc"] == 1.0
    assert out["signals"]["top1_sim"]["per_source"]["s"]["auroc"] == 0.0
    assert out["signals"]["margin"]["per_source"]["s"]["n_answerable"] == 0


def test_exploratory_endpoint_pairs_extra_modes_with_dense():
    """An extra mode that is better on every paired request must show a
    positive paired difference against dense; unpaired modes do not."""
    rows = []
    for i in range(12):
        rid = f"s|label_set|{i}|uk"
        for mode, rec in (("dense", 0.2), ("hybrid", 0.1), ("bm25_prefix", 0.0),
                          ("hyde_dense", 0.5)):
            rows.append({"source": "s", "kind": "label_set", "group": "uk",
                         "mode": mode, "rid": rid, "error": None,
                         "m": {"recall@10": rec}})
    out = R.endpoint_exploratory(rows, n_boot=300)["label_strict_uk"]
    d = out["hyde_dense_minus_dense"]
    assert d["n"] == 12 and abs(d["mean_diff"] - 0.3) < 1e-9 and d["lo"] > 0.29
    assert out["mean_dense"] == 0.2 and out["hybrid_minus_dense"]["mean_diff"] < 0
    assert "record_strict_uk" in R.endpoint_exploratory(rows, n_boot=50)


def test_independent_lexical_oracle_agrees_with_store(indexed):
    from record_documents import plan_documents
    db, store, _emb = indexed
    plan = plan_documents(projection(db), db)
    diff = C.differential_lexical(
        store, plan, db, ["stars comet orbit", "harbour voyage",
                          "sugar recipes", "telescopes"])
    assert diff["comparisons"] == 8
    assert diff["identical_top10"] >= 7, diff["examples"]


def test_raw_columns_ablation_cannot_see_folded_labels(indexed):
    """The topic NAME lives only in the dimension table: the raw-columns
    baseline must miss it, the folded index must find it."""
    from record_documents import plan_documents
    db, store, emb = indexed
    plan = plan_documents(projection(db), db)
    raw = C.raw_columns_baseline(plan, db)
    q = "astronomy"
    assert raw.query(q, 50) == []
    folded = [r[0] for r in store.search_lexical(q, 50)]
    assert {str(i) for i in range(1, 13)} <= set(folded)
    raw.close()
