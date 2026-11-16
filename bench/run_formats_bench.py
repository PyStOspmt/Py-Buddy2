#!/usr/bin/env python3
"""Held-out eval for the ingestion normalizers (JSON / Markdown / PDF).

Each fixture is a synthetic document in its native format, normalized
through the same write_*_sqlite path the upload endpoint uses, then
projected and queried with relax_search. Ground truth is hand-written
oracle SQL against the NORMALIZED sqlite tables -- the projection never
sees the oracle, so inference cannot act as its own ground truth.

This measures whether evidence channels stay reachable after a foreign
format is flattened into relational form -- the honest boundary of the
"arbitrary ingestion" claim. Synthetic but labelled; negative cases
included. Report: immutable bench/results/formats_bench_<ts>.json
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "core"))

from associative import relax_search
from json_normalizer import normalize_json, write_json_sqlite
from markdown_normalizer import write_markdown_sqlite
from pdf_normalizer import write_pdf_sqlite
from run_recall_bench import _prec, _rec
from source_projection import build_source_projection

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
TMP = os.path.join(ROOT, "data")


def _tmp_db(name: str) -> str:
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, f"_bench_{name}.db")
    if os.path.exists(path):
        os.remove(path)
    return path


# -- fixture builders (same call path as /api/domains/upload) -----------

def build_json_db(path: str) -> None:
    """Nested attrs dict -> flattened columns; variants[] -> child table
    with a declared FK back to the parent row."""
    doc = {"products": [
        {"sku": "j1", "title": "Кеди спортивні",
         "attrs": {"material": "шкіра", "season": "літо"},
         "variants": [{"code": "var1", "size": "42"},
                      {"code": "var2", "size": "43"}]},
        {"sku": "j2", "title": "Черевики робочі",
         "attrs": {"material": "нубук", "season": "зима"},
         "variants": [{"code": "var3", "size": "41"}]},
        {"sku": "j3", "title": "Балетки м'які",
         "attrs": {"material": "текстиль", "season": "літо"},
         "variants": [{"code": "var4", "size": "38"}]},
    ]}
    tables = normalize_json(doc)
    write_json_sqlite(tables, path, "fixture.json", "json")


def build_md_db(path: str) -> None:
    md = (
        "# Каталог\n"
        "Довідник магазину.\n"
        "## Взуття\n"
        "Кеди та черевики для міста.\n"
        "| name | price |\n"
        "|------|-------|\n"
        "| Кеди Nike | 1200 |\n"
        "| Черевики | 2400 |\n"
        "## Одяг\n"
        "Куртки і штани.\n")
    write_markdown_sqlite(md, path, "fixture.md")


def _make_pdf(path: str, pages: list[str]) -> None:
    """Minimal valid PDF, one text page per entry -- no extra deps."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    objs.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font_ref = 3 + len(pages) * 2
    for i, txt in enumerate(pages):
        content = f"BT /F1 12 Tf 50 750 Td ({txt}) Tj ET"
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_ref} 0 R >> >> "
            f"/Contents {4 + i * 2} 0 R >>")
        objs.append(
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += (f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    with open(path, "wb") as f:
        f.write(out)


def build_pdf_db(path: str) -> None:
    pdf_path = path + ".src.pdf"
    _make_pdf(pdf_path, ["Notebook page about laptops",
                         "Second page mentions printers"])
    write_pdf_sqlite(pdf_path, path, "fixture.pdf")
    os.remove(pdf_path)


# -- cases --------------------------------------------------------------

CASES = [
    {
        "id": "fmt_json",
        "build": build_json_db,
        "queries": [
            # term lives only inside the nested attrs dict -> flattened
            # column attrs_material on the entity table
            ("шкіра",
             ("SELECT _src FROM products "
              "WHERE attrs_material LIKE '%шкіра%'")),
            # term lives only in the child variants table ->
            # child_mention binds the parent row
            ("var1",
             ("SELECT p._src FROM products p JOIN products_variants v "
              "ON v._parent = p._src WHERE v.code = 'var1'")),
            ("обманка",  # absent -> honest zero
             "SELECT _src FROM products WHERE title LIKE '%обманка%'"),
        ],
    },
    {
        "id": "fmt_md",
        "build": build_md_db,
        "queries": [
            # term inside the pipe-table row -> child_mention binds the
            # owning section through the declared section_id FK
            ("Nike",
             ("SELECT DISTINCT s._src FROM sections s "
              "JOIN table_1 t ON t.section_id = s._src "
              "WHERE t.\"name\" LIKE '%Nike%'")),
            ("куртки",  # section body term -> entity_column
             "SELECT _src FROM sections WHERE body LIKE '%уртки%'"),
            ("телескоп",  # absent
             "SELECT _src FROM sections WHERE body LIKE '%телескоп%'"),
        ],
    },
    {
        "id": "fmt_pdf",
        "build": build_pdf_db,
        "queries": [
            ("laptop",
             "SELECT _src FROM pages WHERE body LIKE '%laptop%'"),
            ("printer",
             "SELECT _src FROM pages WHERE body LIKE '%printer%'"),
            ("nope",  # absent
             "SELECT _src FROM pages WHERE body LIKE '%nope%'"),
        ],
    },
]


def oracle_ids(db_path: str, sql: str) -> set[str]:
    conn = sqlite3.connect(
        "file:{}?mode=ro".format(
            os.path.abspath(db_path).replace("\\", "/")), uri=True)
    try:
        return {str(r[0]) for r in conn.execute(sql)}
    finally:
        conn.close()


def main() -> None:
    report = {
        "benchmark": "formats_v1",
        "disclosure": "synthetic labelled fixtures in native JSON/MD/PDF "
                      "normalized via the upload path; oracle SQL on the "
                      "NORMALIZED sqlite -- never via the projection; "
                      "measures channel reachability after conversion, "
                      "not ranking quality",
        "sources": {},
    }
    print(f"{'source':<9}{'query':<12}{'truth':>7}{'ebr':>6}"
          f"{'rec':>6}{'prec':>6}{'reg':>4}")
    for case in CASES:
        db = _tmp_db(case["id"])
        try:
            case["build"](db)
            proj = build_source_projection(db)
        except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
            report["sources"][case["id"]] = {"error": str(exc)}
            print(f"# {case['id']}: BUILD-FAIL {exc}")
            continue
        print(f"# {case['id']}: entity={proj.entity_table} "
              f"tables={sorted(proj.snapshot.tables)}")
        rows = []
        for term, oracle in case["queries"]:
            try:
                truth = oracle_ids(db, oracle)
            except sqlite3.Error as exc:
                rows.append({"query": term, "oracle_error": str(exc)})
                print(f"{case['id']:<9}{term:<12}ORACLE-FAIL {exc}")
                continue
            res = relax_search(db, proj, term)
            hits = ({i.entity_id for i in res.items} if res else set())
            row = {
                "query": term, "truth_n": len(truth),
                "truth": sorted(truth), "hits": sorted(hits),
                "recall": _rec(hits, truth),
                "precision": _prec(hits, truth),
                "regret": res.regret if res else None,
            }
            rows.append(row)
            print(f"{case['id']:<9}{term:<12}{len(truth):>7}"
                  f"{len(hits):>6}{row['recall']:>6.2f}"
                  f"{row['precision']:>6.2f}"
                  f"{(row['regret'] if row['regret'] is not None else '-'):>4}")
        valid = [r for r in rows if "recall" in r]
        report["sources"][case["id"]] = {
            "entity_table": proj.entity_table,
            "tables": sorted(proj.snapshot.tables),
            "queries": rows,
            "mean_recall": (sum(r["recall"] for r in valid) / len(valid)
                            if valid else None),
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"formats_bench_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
