"""
Structural EAV table hypothesis scoring and bounded text-containment link discovery
for SourceProjection.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Callable, Dict, Optional

from profiler import TableProfile

MAX_TLINK_VOCAB = 200
MAX_TLINK_ROWS = 20000
MIN_TLINK_COVERAGE = 0.30


def _q(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def collect_eav_tables(
    projection: Any,
    conn: sqlite3.Connection,
    is_text_fn: Callable[[Any], bool],
    is_numeric_fn: Callable[[Any], bool],
) -> None:
    """Populate projection.eav_tables via profiler hints and structural cardinality analysis."""
    for tbl, meta in projection.snapshot.eav_projections.items():
        entry = dict(meta)
        entry["detection"] = "name_hint"
        projection.eav_tables[tbl] = entry
        projection.trace_event(
            "eav",
            f"EAV (name-hint): {tbl}",
            f"{meta.get('entity_fk') or meta.get('entity_column')} → "
            f"{meta.get('key_column')} → {meta.get('value_column')}",
            [f"table:{tbl}", f"edge:eav:{tbl}"],
        )

    for name, table in projection.snapshot.tables.items():
        if name in projection.eav_tables:
            continue
        hypothesis = structural_eav_hypothesis(
            projection, conn, name, table, is_text_fn, is_numeric_fn
        )
        if hypothesis is not None:
            projection.eav_tables[name] = hypothesis
            projection.trace_event(
                "eav",
                f"EAV (структурно): {name}",
                f"{hypothesis.get('entity_fk') or hypothesis.get('entity_column')} → "
                f"{hypothesis.get('key_column')} → "
                f"{hypothesis.get('value_column')}",
                [f"table:{name}", f"edge:eav:{name}"],
            )

    for name, meta in projection.eav_tables.items():
        key_col = meta.get("key_column")
        if not key_col:
            continue
        try:
            meta["keys_total"] = conn.execute(
                f"SELECT COUNT(DISTINCT {_q(key_col)}) FROM {_q(name)}"
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT DISTINCT {_q(key_col)} FROM {_q(name)} LIMIT 500"
            ).fetchall()
            meta["discovered_attributes"] = [r[0] for r in rows if r[0] is not None]
            projection.trace_event(
                "eav_vocab",
                f"{name}: словник осей = {meta['keys_total']} ключів",
                ", ".join(str(k) for k in meta["discovered_attributes"][:12]),
                [f"table:{name}"],
            )
        except Exception:
            meta.setdefault("keys_total", None)
            meta.setdefault("discovered_attributes", [])


def structural_eav_hypothesis(
    projection: Any,
    conn: sqlite3.Connection,
    name: str,
    table: TableProfile,
    is_text_fn: Callable[[Any], bool],
    is_numeric_fn: Callable[[Any], bool],
) -> Optional[Dict[str, str]]:
    """Detect EAV (entity, key, value) orientation purely from cardinality structure."""
    cols = list(table.columns.values())
    if not (3 <= len(cols) <= 4) or table.row_count < 3:
        return None
    if len(table.primary_keys) > 1:
        return None

    pk = set(table.primary_keys)
    key_candidates = [
        c
        for c in cols
        if c.name not in pk
        and is_text_fn(c)
        and 1 <= c.distinct_count <= max(20, 0.2 * table.row_count)
        and table.row_count >= 2 * c.distinct_count
    ]
    if not key_candidates:
        return None

    def _pairs_ok(key_col, ent_col) -> bool:
        try:
            pairs = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT {_q(ent_col.name)}, "
                f"{_q(key_col.name)} FROM {_q(name)})"
            ).fetchone()[0]
        except Exception:
            return False
        return pairs >= 0.95 * table.row_count

    hypotheses = []
    for key_col in key_candidates:
        remaining = [c for c in cols if c.name not in pk and c.name != key_col.name]
        if len(remaining) != 2:
            continue
        numeric = [c for c in remaining if is_numeric_fn(c)]
        ordered = numeric + [c for c in remaining if not is_numeric_fn(c)]
        for ent_col in ordered:
            if ent_col.distinct_count < 2 or table.row_count < 2 * ent_col.distinct_count:
                continue
            if key_col.distinct_count >= ent_col.distinct_count:
                continue
            if _pairs_ok(key_col, ent_col):
                value_col = remaining[0] if remaining[0] is not ent_col else remaining[1]
                hypotheses.append((key_col, ent_col, value_col))

    detection = "structural"
    if len(hypotheses) != 1:
        if not hypotheses:
            return None
        ranked = sorted(hypotheses, key=lambda h: h[2].distinct_count, reverse=True)
        best, second = ranked[0], ranked[1] if len(ranked) > 1 else None
        margin = best[2].distinct_count / max(1, best[0].distinct_count)
        if second is not None and (
            best[2].distinct_count == second[2].distinct_count or margin < 2.0
        ):
            cands = " | ".join(
                f"{h[0].name}->v:{h[2].name} (v={h[2].distinct_count})"
                for h in ranked[:3]
            )
            projection.trace_event(
                "eav_ambiguous",
                f"{name}: EAV відхилено -- симетричні осі",
                cands,
                [f"table:{name}"],
            )
            return None
        hypotheses = [best]
        detection = "structural_disambiguated"

    key_col, ent_col, value_col = hypotheses[0]
    return {
        "entity_fk": ent_col.name,
        "key_column": key_col.name,
        "value_column": value_col.name,
        "detection": detection,
    }


def discover_text_links(
    projection: Any,
    conn: sqlite3.Connection,
    is_text_fn: Callable[[Any], bool],
    max_vocab: int = MAX_TLINK_VOCAB,
    max_rows: int = MAX_TLINK_ROWS,
    min_coverage: float = MIN_TLINK_COVERAGE,
) -> None:
    """Measure bounded token-containment coverage from dimension labels into entity text."""
    ent = projection.snapshot.tables.get(projection.entity_table or "")
    if ent is None or not ent.primary_keys:
        return
    pk = ent.primary_keys[0]
    text_cols = [c.name for c in ent.columns.values() if is_text_fn(c)]
    if not text_cols:
        return
    rows = conn.execute(
        f"SELECT {_q(pk)}, "
        + ", ".join(_q(c) for c in text_cols)
        + f" FROM {_q(projection.entity_table)} LIMIT {max_rows}"
    ).fetchall()
    scanned = len(rows)
    if not scanned:
        return
    sampled = scanned < (ent.row_count or scanned)

    norm: list[tuple[str, set[str], str]] = []
    for r in rows:
        txt = " ".join(str(v) for v in r[1:] if v is not None).lower()
        norm.append((str(r[0]), set(re.findall(r"\w+", txt)), txt))

    for tname, tprof in projection.snapshot.tables.items():
        if (
            tname == projection.entity_table
            or tname in projection.eav_tables
            or tname.startswith("__")
        ):
            continue
        for col in tprof.columns.values():
            if not is_text_fn(col) or not (2 <= col.distinct_count <= max_vocab):
                continue
            labels = [
                str(r[0])
                for r in conn.execute(
                    f"SELECT DISTINCT {_q(col.name)} FROM {_q(tname)} "
                    f"WHERE {_q(col.name)} IS NOT NULL LIMIT {max_vocab + 1}"
                )
                if r[0] is not None
            ]
            labels = [lbl for lbl in labels if 1 <= len(lbl) <= 64]
            if len(labels) < 2 or len(labels) > max_vocab:
                continue
            single_set = {lbl.lower() for lbl in labels if " " not in lbl}
            multi = [lbl.lower() for lbl in labels if " " in lbl]
            matched = 0
            for _, tokens, txt in norm:
                if tokens & single_set or any(m in txt for m in multi):
                    matched += 1
            coverage = matched / scanned
            if coverage >= min_coverage:
                projection.text_links.append(
                    {
                        "dim_table": tname,
                        "label_column": col.name,
                        "entity_table": projection.entity_table,
                        "vocab": len(labels),
                        "coverage": round(coverage, 3),
                        "matched": matched,
                        "scanned": scanned,
                        "sampled": sampled,
                        "detection": "text_containment",
                    }
                )
                projection.trace_event(
                    "text_link",
                    f"{tname}.{col.name} ⊆ текст {projection.entity_table}",
                    f"{matched}/{scanned} сутностей містять лейбл (словник {len(labels)})",
                    [f"table:{tname}", f"edge:text:{tname}"],
                )
