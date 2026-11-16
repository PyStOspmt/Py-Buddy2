"""
Differential recall benchmark: pure lexical baseline vs EBR
(evidence-bound relaxation) across structural schema shapes.

Honest scope: this is a CONTROLLED experiment on synthetic fixtures --
it proves that EBR binds terms across schema shapes a lexical baseline
physically cannot reach, it does NOT prove real-world relevance. Every
fixture, ground-truth set and generation seed is disclosed in the
output report (immutable, timestamped).

Benchmark protocol: docs/BENCHMARKS.md.

Shapes under test (the schema axis is what varies, not the query):
  flat       -- terms live only in entity text columns
  eav        -- terms live only inside an EAV value axis
  fk_dim     -- terms live in a normalized dimension behind a declared FK
  junction   -- terms live behind a two-FK many-to-many bridge
  renamed    -- same as eav but with obfuscated identifiers (rename control)
  inferred   -- undeclared link verified post-hoc via value_overlap
                (simulates the state after a conductor run)

  eav_ambiguous -- low-card value column: detector must REFUSE
  multi_hop  -- term 2 hops away (entity->sub->dept); expected degraded
               until >2-hop traversal lands (A7)
  inferred_junc -- junction with BOTH links overlap-verified, no
               REFERENCES anywhere
  paraphrase -- no shared characters: vector channel only
  conj_mix   -- two-term query split across name + EAV channels
  no_match   -- empty truth: fabricated-hit resistance

Methods:
  fts        -- conjunctive LIKE over entity text columns
  ebr        -- relax_search, lexical channels only
  ebr_vec    -- ebr + dense channel (scripted embedder)
  vec_only   -- pure dense baseline: conjunctive vector bindings alone

Metrics: recall, precision, regret, ms; degraded_ok on the ambiguity
fixture (the detector reporting NO mapping is the pass condition).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "core"))

from associative import relax_search
from sanitizer import LIKE_ESCAPE_CLAUSE, escape_like
from source_projection import build_source_projection, quote_ident
from vector_index import VectorIndex

SEED = 42
N_ENTITIES = 40

# ---------------------------------------------------------------------
# Fixture builders: each returns (db_path, queries)
# queries = [{"query": str, "truth": set(entity_ids), "note": str}]
# ---------------------------------------------------------------------


def _mk(path):
    conn = sqlite3.connect(path)
    return conn


def _products(conn, name_col="name", cat_col="category", extra=""):
    conn.execute(
        f"CREATE TABLE products (id TEXT PRIMARY KEY, {name_col} TEXT, "
        f"{cat_col} TEXT, price REAL {extra})")
    names = [f"Артикул {i}" for i in range(N_ENTITIES)]
    cats = [f"grp{i % 4}" for i in range(N_ENTITIES)]
    conn.executemany(
        f"INSERT INTO products VALUES (?,?,?,?{',?' if extra else ''})",
        [(f"e{i}", names[i], cats[i], 10.0 + i)
         if not extra else
         (f"e{i}", names[i], cats[i], 10.0 + i, f"mat_{i % 3}")
         for i in range(N_ENTITIES)])
    return names, cats


def build_flat(tmp):
    """Baseline shape: the answer IS in entity columns -> FTS should
    match; EBR must not regress."""
    db = os.path.join(tmp, "flat.db")
    c = _mk(db)
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "category TEXT, price REAL)")
    rows = [(f"e{i}", f"Артикул {i}", f"grp{i % 4}", 10.0 + i)
            for i in range(N_ENTITIES)]
    # a lexical-only signal: two items literally named with 'сталевий'
    rows[3] = ("e3", "сталевий вал", "grp3", 13.0)
    rows[7] = ("e7", "сталевий круг", "grp3", 17.0)
    c.executemany("INSERT INTO products VALUES (?,?,?,?)", rows)
    c.commit(); c.close()
    return db, [{"query": "сталевий", "truth": {"e3", "e7"},
                 "note": "term in entity name column"}]


def build_eav(tmp, renamed=False):
    """Terms exist ONLY inside EAV values -> lexical baseline = 0."""
    db = os.path.join(tmp, f"eav{ '_ren' if renamed else ''}.db")
    c = _mk(db)
    if renamed:
        c.execute("CREATE TABLE t9 (a1 TEXT PRIMARY KEY, a2 TEXT, "
                  "a3 TEXT, a4 REAL)")
        c.execute("CREATE TABLE t10 (b1 TEXT, b2 TEXT, b3 TEXT)")
        ins_p = "INSERT INTO t9 VALUES (?,?,?,?)"
        ins_a = "INSERT INTO t10 VALUES (?,?,?)"
    else:
        c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                  "category TEXT, price REAL)")
        c.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
        ins_p = "INSERT INTO products VALUES (?,?,?,?)"
        ins_a = "INSERT INTO attrs VALUES (?,?,?)"
    c.executemany(ins_p, [(f"e{i}", f"Артикул {i}", f"grp{i % 4}",
                           10.0 + i) for i in range(N_ENTITIES)])
    # every entity gets 2 keys so the EAV signature holds; the value
    # column must stay HIGH-cardinality (a low-card value column makes
    # the key<->value orientation symmetric and the signature honestly
    # refuses -- caught live by this benchmark on the first run)
    for i in range(N_ENTITIES):
        c.execute(ins_a, (f"e{i}", "матеріал", f"мат_{i % 9}"))
        c.execute(ins_a, (f"e{i}", "об'єм", f"{50 + i * 7}мл"))
    for i in (3, 7, 11):
        c.execute(ins_a, (f"e{i}", "матеріал", "шкіра"))
    c.commit(); c.close()
    q = "шкіра"
    return db, [{"query": q, "truth": {"e3", "e7", "e11"},
                 "note": "term only inside EAV value axis"
                         + (" (renamed identifiers)" if renamed else "")}]


def build_fkdim(tmp):
    db = os.path.join(tmp, "fkdim.db")
    c = _mk(db)
    c.execute("CREATE TABLE purposes (id TEXT PRIMARY KEY, label TEXT)")
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "category TEXT, price REAL, "
              "purpose_id TEXT REFERENCES purposes(id))")
    c.executemany("INSERT INTO purposes VALUES (?,?)",
                  [("d1", "одяг"), ("d2", "взуття")])
    c.executemany("INSERT INTO products VALUES (?,?,?,?,?)",
                  [(f"e{i}", f"Артикул {i}", f"grp{i % 4}", 10.0 + i,
                    "d2" if i in (5, 9, 13) else "d1")
                   for i in range(N_ENTITIES)])
    c.commit(); c.close()
    return db, [{"query": "взуття", "truth": {"e5", "e9", "e13"},
                 "note": "term only in a declared-FK dimension label"}]


def build_junction(tmp):
    db = os.path.join(tmp, "junction.db")
    c = _mk(db)
    c.execute("CREATE TABLE tags (id TEXT PRIMARY KEY, label TEXT)")
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "category TEXT, price REAL)")
    c.execute("CREATE TABLE product_tags ("
              "product_id TEXT REFERENCES products(id), "
              "tag_id TEXT REFERENCES tags(id))")
    c.executemany("INSERT INTO tags VALUES (?,?)",
                  [("t1", "антисептична"), ("t2", "водостійка")])
    c.executemany("INSERT INTO products VALUES (?,?,?,?)",
                  [(f"e{i}", f"Артикул {i}", f"grp{i % 4}", 10.0 + i)
                   for i in range(N_ENTITIES)])
    for i in (2, 6, 10, 14):
        c.execute("INSERT INTO product_tags VALUES (?,?)",
                  (f"e{i}", "t1"))
    c.commit(); c.close()
    return db, [{"query": "антисептична",
                 "truth": {"e2", "e6", "e10", "e14"},
                 "note": "term behind a two-FK junction bridge"}]


def build_inferred(tmp):
    """No REFERENCES clause -- the link exists only in the data.
    Baseline EBR cannot see it; the post-conductor state (inferred link
    admitted by measured overlap) can."""
    db = os.path.join(tmp, "inferred.db")
    c = _mk(db)
    c.execute("CREATE TABLE lookup (id TEXT PRIMARY KEY, label TEXT)")
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "category TEXT, price REAL, suit TEXT)")
    c.executemany("INSERT INTO lookup VALUES (?,?)",
                  [("d1", "одяг"), ("d2", "взуття")])
    c.executemany("INSERT INTO products VALUES (?,?,?,?,?)",
                  [(f"e{i}", f"Артикул {i}", f"grp{i % 4}", 10.0 + i,
                    "d2" if i in (1, 8, 15) else "d1")
                   for i in range(N_ENTITIES)])
    c.commit(); c.close()
    return db, [{"query": "взуття", "truth": {"e1", "e8", "e15"},
                 "note": "undeclared link (post-conductor state)"}]


def build_eav_ambiguous(tmp):
    """Value column cardinality ~= key column cardinality -> the
    key<->value orientation is symmetric and the signature must REFUSE
    to detect. Target is honest degradation, not recall."""
    db = os.path.join(tmp, "eav_amb.db")
    c = _mk(db)
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "category TEXT, price REAL)")
    c.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    c.executemany("INSERT INTO products VALUES (?,?,?,?)",
                  [(f"e{i}", f"Артикул {i}", f"grp{i % 4}", 10.0 + i)
                   for i in range(N_ENTITIES)])
    # TRULY symmetric: 4 keys x 4 shared values -- both orientations
    # pass pair-uniqueness AND carry equal cardinality, so even the
    # coverage-disambiguated signature must refuse. (The v2 fixture
    # used 2 keys x 4 values -- an asymmetric race the disambiguator
    # now legitimately resolves.)
    keys = ["матеріал", "колір", "формат", "тип"]
    vals = ["альфа", "бета", "гамма", "дельта"]
    for i in range(N_ENTITIES):
        for j, k in enumerate(keys):
            c.execute("INSERT INTO attrs VALUES (?,?,?)",
                      (f"e{i}", k, vals[(i + j) % 4]))
    c.commit(); c.close()
    return db, [{"query": "шкіра", "truth": set(),
                 "note": "ambiguous EAV -- detector must refuse",
                 "degraded_ok": True}]


def build_multihop(tmp):
    """Term lives TWO hops away: products.sub_id -> subs.id ->
    subs.dept_id -> depts.label. Exceeds current 1-hop traversal --
    expected recall 0 until >2-hop binding lands (target A7)."""
    db = os.path.join(tmp, "multihop.db")
    c = _mk(db)
    c.execute("CREATE TABLE depts (id TEXT PRIMARY KEY, label TEXT)")
    c.execute("CREATE TABLE subs (id TEXT PRIMARY KEY, label TEXT, "
              "dept_id TEXT REFERENCES depts(id))")
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "category TEXT, price REAL, "
              "sub_id TEXT REFERENCES subs(id))")
    c.executemany("INSERT INTO depts VALUES (?,?)",
                  [("dp1", "взуття"), ("dp2", "одяг")])
    c.executemany("INSERT INTO subs VALUES (?,?,?)",
                  [("s1", "кеди", "dp1"), ("s2", "куртки", "dp2")])
    c.executemany("INSERT INTO products VALUES (?,?,?,?,?)",
                  [(f"e{i}", f"Артикул {i}", f"grp{i % 4}", 10.0 + i,
                    "s1" if i in (4, 12, 20) else "s2")
                   for i in range(N_ENTITIES)])
    c.commit(); c.close()
    return db, [{"query": "взуття", "truth": {"e4", "e12", "e20"},
                 "note": "term 2 hops away via declared FKs"}]


def build_inferred_junction(tmp):
    """Junction with NO declared FKs: both edges only exist as value
    overlap. Simulates post-conductor state (both links verified at
    overlap >= 0.98)."""
    db = os.path.join(tmp, "inf_junc.db")
    c = _mk(db)
    c.execute("CREATE TABLE tags (id TEXT PRIMARY KEY, label TEXT)")
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "category TEXT, price REAL)")
    c.execute("CREATE TABLE product_tags (a TEXT, b TEXT)")  # no refs
    c.executemany("INSERT INTO tags VALUES (?,?)",
                  [("t1", "антисептична"), ("t2", "водостійка")])
    c.executemany("INSERT INTO products VALUES (?,?,?,?)",
                  [(f"e{i}", f"Артикул {i}", f"grp{i % 4}", 10.0 + i)
                   for i in range(N_ENTITIES)])
    for i in (2, 6, 10, 14):
        c.execute("INSERT INTO product_tags VALUES (?,?)",
                  (f"e{i}", "t1"))
    c.commit(); c.close()
    return db, [{"query": "антисептична",
                 "truth": {"e2", "e6", "e10", "e14"},
                 "note": "junction whose links are overlap-verified"}]


def build_paraphrase(tmp):
    """No shared characters between query and cell: 'воскресити' must
    reach 'resurrect' semantically. Lexical methods score 0."""
    db = os.path.join(tmp, "para.db")
    c = _mk(db)
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "price REAL)")
    c.executemany("INSERT INTO products VALUES (?,?,?)",
                  [(f"e{i}", f"Артикул {i}", 10.0 + i)
                   for i in range(N_ENTITIES) if i not in (3, 7)]
                  + [("e3", "Elixir of resurrect", 99.0),
                     ("e7", "Resurrect potion", 42.0)])
    c.commit(); c.close()
    return db, [{"query": "воскресити", "truth": {"e3", "e7"},
                 "note": "paraphrase -- vector channel only"}]


def build_conj_mix(tmp):
    """Two-term conjunctive query split across channels: 'фарба' hits
    the name column, 'шкіра' hits only an EAV value."""
    db = os.path.join(tmp, "conj.db")
    c = _mk(db)
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "category TEXT, price REAL)")
    c.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    names = [f"Артикул {i}" for i in range(N_ENTITIES)]
    names[3] = "Фарба алкидна"; names[7] = "Фарба масляна"
    c.executemany("INSERT INTO products VALUES (?,?,?,?)",
                  [(f"e{i}", names[i], f"grp{i % 4}", 10.0 + i)
                   for i in range(N_ENTITIES)])
    for i in range(N_ENTITIES):
        c.execute("INSERT INTO attrs VALUES (?,?,?)",
                  (f"e{i}", "матеріал", f"мат_{i}"))
        c.execute("INSERT INTO attrs VALUES (?,?,?)",
                  (f"e{i}", "виробник", f"завод_{i}"))
    for i in (3, 11):
        c.execute("INSERT INTO attrs VALUES (?,?,?)",
                  (f"e{i}", "матеріал", "шкіра"))
    c.commit(); c.close()
    return db, [{"query": "фарба шкіра", "truth": {"e3"},
                 "note": "conjunctive: name + EAV value"}]


def build_no_match(tmp):
    """Truth is empty -- the honest outcome is zero hits plus a
    recorded dropped term, never a fabricated candidate."""
    db = os.path.join(tmp, "nomatch.db")
    c = _mk(db)
    c.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
              "price REAL)")
    c.executemany("INSERT INTO products VALUES (?,?,?)",
                  [(f"e{i}", f"Артикул {i}", 10.0 + i)
                   for i in range(N_ENTITIES)])
    c.commit(); c.close()
    return db, [{"query": "титан", "truth": set(),
                 "note": "zero-truth: expect empty + regret 1"}]


SHAPES = [
    ("flat", build_flat),
    ("eav", build_eav),
    ("eav_renamed", lambda t: build_eav(t, renamed=True)),
    ("eav_ambiguous", build_eav_ambiguous),
    ("fk_dim", build_fkdim),
    ("multi_hop", build_multihop),
    ("junction", build_junction),
    ("inferred", build_inferred),
    ("inferred_junc", build_inferred_junction),
    ("paraphrase", build_paraphrase),
    ("conj_mix", build_conj_mix),
    ("no_match", build_no_match),
]


# ---------------------------------------------------------------------
# Methods under test
# ---------------------------------------------------------------------

def baseline_fts(db_path, proj, query, limit=100):
    """Conjunctive LIKE over every text column of the entity table --
    the strongest lexical search a fixed-schema system can do."""
    from source_projection import _is_text
    terms = [t for t in query.split() if len(t) >= 3]
    if not terms or not proj.entity_table:
        return set()
    tprof = proj.snapshot.tables[proj.entity_table]
    cols = [c.name for c in tprof.columns.values() if _is_text(c)]
    ident = proj.resolve("identity").column
    if not ident or not cols:
        return set()
    conn = sqlite3.connect("file:{}?mode=ro".format(
        os.path.abspath(db_path).replace("\\", "/")), uri=True)
    try:
        cur = conn.cursor()
        found = None
        for t in terms:
            pats = sorted({t, t.lower(), t.upper(), t.capitalize()})
            pred = " OR ".join(
                f"{quote_ident(c)} LIKE ?{LIKE_ESCAPE_CLAUSE}"
                for c in cols for _ in pats)
            params = [f"%{escape_like(p)}%" for p in pats] * len(cols)
            rows = cur.execute(
                f"SELECT {quote_ident(ident)} "
                f"FROM {quote_ident(proj.entity_table)} "
                f"WHERE ({pred}) LIMIT ?", params + [limit]).fetchall()
            ids = {r[0] for r in rows}
            found = ids if found is None else (found & ids)
        return found or set()
    finally:
        conn.close()


def ebr(db_path, proj, query, vector=None):
    res = relax_search(db_path, proj, query, vector=vector)
    if res is None:
        return set(), 0
    return {i.entity_id for i in res.items}, res.regret


def vec_only(idx, emb, query):
    """Pure dense baseline: conjunctive intersection of vector-channel
    bindings alone -- isolates what the dense channel contributes."""
    terms = [t for t in query.split() if len(t) >= 3]
    if not terms or idx is None:
        return set()
    common = None
    for t in terms:
        ids = set(idx.bind(t, emb))
        common = ids if common is None else (common & ids)
    return common or set()


class BenchEmbedder:
    """Scripted deterministic embedder for the vector channel.

    Disclosed ground truth: the synonym map is fixed IN THIS FILE --
    it tests that the channel binds correctly, NOT that a real model
    embeds well (a live-model run is a separate, authorized experiment).
    """
    SYNONYMS = ({"воскресити", "resurrect"},)
    # Non-synonym texts get deterministic dense random vectors (md5-
    # seeded -> stable across processes); in 128 dims unrelated pairs
    # sit at cosine ~0, far below the channel threshold. A sparse
    # one-hot hash embedder was tried first and the benchmark CAUGHT it:
    # axis collisions fabricated conjunctive bindings (ebr_vec recall
    # dropped to 0 on conj_mix) -- a real failure mode worth recording:
    # a noisy dense channel can poison the conjunction.
    DIM = 128

    model = "bench/scripted-embedder"

    def embed(self, texts):
        import hashlib
        import random
        out = []
        for t in texts:
            tl = t.lower()
            if any(any(s in tl for s in group)
                   for group in self.SYNONYMS):
                out.append([1.0] + [0.0] * (self.DIM - 1))
            else:
                seed = int.from_bytes(
                    hashlib.md5(tl.encode()).digest()[:8], "big")
                rng = random.Random(seed)
                out.append([rng.gauss(0, 1) for _ in range(self.DIM)])
        return out


def _verify_link(db_path, proj, child_table, child_col,
                 parent_table, parent_col):
    """Simulate the post-conductor state: the SAME verification the
    conductor runs (measured overlap >= 0.98), never trusting a
    written-down link."""
    from conductor import LINK_THRESHOLD
    conn = sqlite3.connect(db_path)
    try:
        vals = [r[0] for r in conn.execute(
            f"SELECT DISTINCT {quote_ident(child_col)} "
            f"FROM {quote_ident(child_table)} "
            f"WHERE {quote_ident(child_col)} IS NOT NULL").fetchall()]
        if not vals:
            return 0.0
        ph = ",".join("?" * len(vals))
        matched = conn.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT "
            f"{quote_ident(parent_col)} FROM {quote_ident(parent_table)})"
            f" WHERE {quote_ident(parent_col)} IN ({ph})",
            vals).fetchone()[0]
        ov = matched / len(vals)
        if ov >= LINK_THRESHOLD:
            proj.inferred_links.append({
                "child_table": child_table, "child_column": child_col,
                "parent_table": parent_table,
                "parent_column": parent_col,
                "overlap": ov, "sampled": False,
                "evidence": "bench_verified"})
        return ov
    finally:
        conn.close()


def _rec(found, truth):
    return len(found & truth) / max(len(truth), 1) if truth else 0.0


def _prec(found, truth):
    return len(found & truth) / max(len(found), 1) if found else 1.0


def main():
    report = {"benchmark": "ebr_recall_v2",
              "protocol": "docs/BENCHMARKS.md",
              "seed": SEED, "n_entities": N_ENTITIES,
              "embedder": "scripted (disclosed synonym map; NOT a "
                          "real-model quality measurement)",
              "disclosure":
              "synthetic fixtures; ground truth hand-labelled; "
              "measures channel reachability, not real-world relevance",
              "shapes": {}}
    emb = BenchEmbedder()
    header = (f"{'shape':<14}{'query':<16}{'fts':>6}{'ebr':>6}"
              f"{'ebr+v':>7}{'vec':>6}{'reg':>4}")
    print(header)
    for shape, builder in SHAPES:
        tmp = tempfile.mkdtemp()
        db, queries = builder(tmp)
        proj = build_source_projection(db)
        # post-conductor simulation: measured overlap, same threshold
        if shape == "inferred":
            _verify_link(db, proj, "products", "suit", "lookup", "id")
        elif shape == "inferred_junc":
            _verify_link(db, proj, "product_tags", "a",
                         "products", "id")
            _verify_link(db, proj, "product_tags", "b", "tags", "id")
        # dense index: built once per shape, stats disclosed
        idx = VectorIndex(emb.model)
        idx.stats = idx.build(db, proj, emb)
        vector = (idx, emb) if idx.stats.get("status") == "ready" \
            else None

        degraded_ok = None
        if shape == "eav_ambiguous":
            # pass = detector refused rather than guessed
            degraded_ok = "attrs" not in proj.eav_tables

        shape_res = []
        for q in queries:
            truth = set(q["truth"])

            def pack(hits, regret, t0, _truth=frozenset(truth)):
                return {"hits": sorted(hits),
                        "recall": _rec(hits, _truth),
                        "precision": _prec(hits, _truth),
                        "regret": regret,
                        "ms": round((time.time() - t0) * 1000, 1)}

            t0 = time.time()
            fts_h = baseline_fts(db, proj, q["query"])
            t0, m_fts = time.time(), pack(fts_h, 0, t0)
            t0 = time.time()
            e_h, e_r = ebr(db, proj, q["query"])
            t0, m_ebr = time.time(), pack(e_h, e_r, t0)
            t0 = time.time()
            ev_h, ev_r = ebr(db, proj, q["query"], vector=vector)
            t0, m_ev = time.time(), pack(ev_h, ev_r, t0)
            t0 = time.time()
            v_h = vec_only(idx, emb, q["query"])
            m_v = pack(v_h, 0, t0)
            methods = {"fts": m_fts, "ebr": m_ebr,
                       "ebr_vec": m_ev, "vec_only": m_v}
            shape_res.append({"query": q["query"],
                              "truth": sorted(truth),
                              "note": q["note"], **methods})
            print(f"{shape:<14}{q['query']:<16}"
                  f"{methods['fts']['recall']:>6.2f}"
                  f"{methods['ebr']['recall']:>6.2f}"
                  f"{methods['ebr_vec']['recall']:>7.2f}"
                  f"{methods['vec_only']['recall']:>6.2f}"
                  f"{methods['ebr']['regret']:>4}")

        entry = {"queries": shape_res,
                 "vector_stats": idx.stats}
        if degraded_ok is not None:
            entry["degraded_ok"] = degraded_ok
        for m in ("fts", "ebr", "ebr_vec", "vec_only"):
            entry[f"mean_recall_{m}"] = (
                sum(q[m]["recall"] for q in shape_res)
                / len(shape_res))
        report["shapes"][shape] = entry

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir,
                       f"recall_bench_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
