"""Frozen point-update scenario for the library source (docs/18, P5).

Plan (all real Gutenberg records, none from the 8,000-book sample):
  * 18 INSERTS   : held-out records added as new books;
  * 18 UPDATES   : existing test-split books whose whole content (title,
                   authors, subjects, shelves) is replaced by another
                   held-out record's content (the meaning changes);
  * 18 DELETES   : existing test-split books removed.
(18 each because only 36 test-split library records have validated strict
and Ukrainian requests; see the scope-change log in docs/18.)
Requests for the new/updated content are written by the same LLM pipeline
as the main request set; requests for the OLD content come from the frozen
main request file. The scenario is written once and hashed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_queries as gq  # noqa: E402
from build_sources import RAW_PG, SEED, split_labels  # noqa: E402
from eval_llm import BudgetedLLM  # noqa: E402
from eval_targets import ROOT, Target  # noqa: E402

N_EACH = 18
OUT = ROOT / "docs" / "semantic_mutations_20261112.json"
LOG = ROOT / "docs" / "semantic_mutations_llm_log_20261112.jsonl"


def sampled_ids() -> set[int]:
    from build_sources import pick_books
    return {int(r["Text#"]) for r in pick_books()}


def held_out_records(n: int) -> list[dict]:
    with open(RAW_PG, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["Type"] == "Text" and r["Language"] == "en"
                and r["Subjects"].strip()]
    taken = sampled_ids()
    pool = sorted((r for r in rows if int(r["Text#"]) not in taken),
                  key=lambda r: int(r["Text#"]))
    random.Random(SEED + 7).shuffle(pool)
    out = []
    for r in pool[:n]:
        out.append({
            "source_book_id": int(r["Text#"]),
            "title": " ".join(r["Title"].split()),
            "issued": r["Issued"], "lang": r["Language"],
            "loc_class": r["LoCC"],
            "authors": [a.strip() for a in r["Authors"].split(";") if a.strip()],
            "subjects": split_labels(r["Subjects"]),
            "shelves": split_labels(r["Bookshelves"])})
    return out


def content_text(rec: dict) -> str:
    return (f"title: {rec['title']}; authors: {'; '.join(rec['authors'])}; "
            f"subjects: {'; '.join(rec['subjects'])}")


def pick_existing(queries: dict) -> list[str]:
    ok = sorted((r["target_id"] for r in queries["targets"]
                 if r["source"] == "library" and r["kind"] == "record"
                 and r["split"] == "test" and r["flags"]["strict_ok"]
                 and r["flags"]["uk_ok"]), key=int)
    random.Random(SEED + 11).shuffle(ok)
    return ok


def make_queries(llm: BudgetedLLM, rec: dict, key: str) -> dict:
    t = Target("record", "library", key, content_text(rec), (key,),
               content_text(rec), ())
    row = gq.generate(llm, t)
    return {"queries": row["queries"], "flags": row["flags"]}


def main() -> int:
    queries = json.loads(gq.OUT.read_text(encoding="utf-8"))
    held = held_out_records(2 * N_EACH)
    existing = pick_existing(queries)
    if len(existing) < 2 * N_EACH:
        raise SystemExit(f"only {len(existing)} usable test records")
    p_in, p_out = gq.pricing_for(gq.MODEL)
    llm = BudgetedLLM(gq.MODEL, 0.1, LOG, p_in, p_out)
    inserts = [{"record": r, **make_queries(llm, r, f"ins:{r['source_book_id']}")}
               for r in held[:N_EACH]]
    updates = [{"book_id": int(bid), "new_content": r,
                **make_queries(llm, r, f"upd:{bid}")}
               for bid, r in zip(existing[:N_EACH], held[N_EACH:])]
    deletes = [{"book_id": int(b)} for b in existing[N_EACH:2 * N_EACH]]
    doc = {"meta": {"seed": SEED, "n_each": N_EACH, "model": gq.MODEL,
                    **llm.summary(),
                    "request_file_sha256": hashlib.sha256(
                        gq.OUT.read_bytes()).hexdigest()},
           "inserts": inserts, "updates": updates, "deletes": deletes}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / (OUT.name + ".sha256")).write_text(sha + "\n")
    print(json.dumps(doc["meta"], indent=1), "\nsha256", sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
