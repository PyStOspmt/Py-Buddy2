"""
Retrieval Fabric: streaming document renderer (private).

Implementation of `record_documents.iter_record_documents`. Kept in a
separate module so both files stay within the project's 350-line bound;
the public entry point is `record_documents.iter_record_documents`.

All channels are merge-join streams ordered by CAST(entity ref AS
TEXT): the entity table (with dimension labels attached via LEFT JOIN),
bridge rows, and EAV rows. Memory holds only per-entity accumulations,
never source-volume data.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator

from record_documents import (
    MAX_AXIS_PAIRS,
    MAX_M2M_LABELS,
    NUMERIC_TEXT_RE,
    BridgeSpec,
    DocPlan,
    EavSpec,
    ExcludedRecord,
    RecordDoc,
    RenderStats,
    _ro_connect,
    doc_hash,
)
from source_projection import quote_ident

logger = logging.getLogger(__name__)


def _norm(v) -> str:
    return " ".join(str(v).split())


class _EidStream:
    """Peekable cursor ordered by CAST(entity ref AS TEXT).

    Rows whose ref is NULL or absent from the entity table are
    skipped WITHOUT blocking later entities and counted in `stats`
    (NULL sorts first -- a buffered NULL row used to stall the whole
    merge)."""

    def __init__(self, cursor, stats: RenderStats | None = None):
        self._it = iter(cursor)
        self._buf = None
        self._stats = stats

    def _count(self, ref) -> None:
        if self._stats is not None:
            if ref is None:
                self._stats.null_ref_rows += 1
            else:
                self._stats.orphan_rows += 1

    def take(self, eid: str) -> list:
        rows = []
        while True:
            if self._buf is None:
                self._buf = next(self._it, None)
            if self._buf is None:
                break
            ref = self._buf[0]
            if ref is None:
                self._count(ref)          # NULL ref: skip, keep going
                self._buf = None
                continue
            if ref >= eid:
                break
            self._count(ref)              # orphan: id absent upstream
            self._buf = None
        while self._buf is not None and self._buf[0] == eid:
            rows.append(self._buf)
            self._buf = next(self._it, None)
        return rows

    def drain(self) -> None:
        """Consume the tail after the last entity -- every leftover
        ref is by definition NULL or orphan (entities are sorted and
        exhausted), so the counts stay exact."""
        if self._buf is not None:
            self._count(self._buf[0])
            self._buf = None
        for row in self._it:
            self._count(row[0])


def _axis_rank(conn: sqlite3.Connection, spec: EavSpec) -> dict:
    cur = conn.execute(
        f"SELECT {quote_ident(spec.key_col)}, "
        f"COUNT(DISTINCT {quote_ident(spec.entity_col)}) "
        f"FROM {quote_ident(spec.table)} "
        f"GROUP BY {quote_ident(spec.key_col)} "
        f"ORDER BY 2 DESC, 1")
    return {r[0]: i for i, r in enumerate(cur)}


def _eav_lines(rows, rank: dict, prov: list) -> list:
    pairs = []
    for _eid, k, v in rows:
        if k is None or v is None:
            continue
        sv = _norm(v)
        if not sv or NUMERIC_TEXT_RE.match(sv):
            continue
        pairs.append((rank.get(k, len(rank)), str(k), sv))
    pairs.sort()
    picked = pairs[:MAX_AXIS_PAIRS]
    prov.extend(f"eav:{k}" for _r, k, _v in picked)
    return [f"{k}: {v}" for _r, k, v in picked]


def _bridge_lines(rows, spec: BridgeSpec, prov: list) -> list:
    labels = set()
    for row in rows:
        vals = [_norm(v) for v in row[1:] if v is not None and _norm(v)]
        if vals:
            labels.add(" / ".join(vals))
    if not labels:
        return []
    prov.append(f"{spec.other_table}.*")
    picked = sorted(labels)[:MAX_M2M_LABELS]
    return [f"{spec.other_table}: {'; '.join(picked)}"]


def _entity_sql(plan: DocPlan) -> str:
    e = quote_ident(plan.entity_table)
    idc = "rowid" if plan.id_column == "rowid" \
        else quote_ident(plan.id_column)
    sel = [f"CAST(e.{idc} AS TEXT)"] + \
          [f"e.{quote_ident(c)}" for c in plan.own_columns]
    joins = []
    for i, d in enumerate(plan.dimensions):
        sel += [f"d{i}.{quote_ident(c)}" for c in d.label_cols]
        joins.append(
            f"LEFT JOIN {quote_ident(d.parent_table)} d{i} ON "
            f"e.{quote_ident(d.entity_col)} = "
            f"d{i}.{quote_ident(d.parent_key_col)}")
    return (f"SELECT {', '.join(sel)} FROM {e} e {' '.join(joins)} "
            f"ORDER BY CAST(e.{idc} AS TEXT), e.{idc}")


def _eav_stream(conn, spec: EavSpec, stats) -> _EidStream:
    return _EidStream(conn.execute(
        f"SELECT CAST({quote_ident(spec.entity_col)} AS TEXT), "
        f"{quote_ident(spec.key_col)}, {quote_ident(spec.value_col)} "
        f"FROM {quote_ident(spec.table)} "
        f"ORDER BY CAST({quote_ident(spec.entity_col)} AS TEXT)"),
        stats)


def _bridge_stream(conn, spec: BridgeSpec, stats) -> _EidStream:
    sel = [f"o.{quote_ident(c)}" for c in spec.label_cols]
    return _EidStream(conn.execute(
        f"SELECT CAST(b.{quote_ident(spec.entity_col)} AS TEXT), "
        f"{', '.join(sel)} FROM {quote_ident(spec.bridge_table)} b "
        f"LEFT JOIN {quote_ident(spec.other_table)} o ON "
        f"b.{quote_ident(spec.other_col)} = "
        f"o.{quote_ident(spec.other_key_col)} "
        f"ORDER BY CAST(b.{quote_ident(spec.entity_col)} AS TEXT)"),
        stats)


def _own_lines(row, plan: DocPlan, prov: list) -> list[str]:
    lines = []
    for i, c in enumerate(plan.own_columns):
        v = row[1 + i]
        if v is not None and _norm(v):
            lines.append(f"{c}: {_norm(v)}")
            prov.append(f"{plan.entity_table}.{c}")
    return lines


def _dim_lines(row, plan: DocPlan, prov: list) -> list[str]:
    lines = []
    base = 1 + len(plan.own_columns)
    for d in plan.dimensions:
        vals = [row[base + j] for j in range(len(d.label_cols))]
        base += len(d.label_cols)
        vals = [_norm(v) for v in vals if v is not None and _norm(v)]
        if vals:
            lines.append(f"{d.parent_table}: {' / '.join(vals)}")
            prov.append(f"{d.parent_table}.*")
    return lines


def render_documents(db_path: str, plan: DocPlan, *,
                     max_chars: int = 1500,
                     stats: RenderStats | None = None
                     ) -> Iterator[RecordDoc | ExcludedRecord]:
    conn = _ro_connect(db_path)
    try:
        eav_streams = [
            (s, _axis_rank(conn, s), _eav_stream(conn, s, stats))
            for s in plan.eav]
        bridge_streams = [(s, _bridge_stream(conn, s, stats))
                          for s in plan.bridges]
        for row in conn.execute(_entity_sql(plan)):
            eid = row[0]
            if eid is None:
                yield ExcludedRecord(None, "null_id")
                continue
            prov: list = []
            lines = _own_lines(row, plan, prov)
            lines += _dim_lines(row, plan, prov)
            for spec, stream in bridge_streams:
                lines += _bridge_lines(stream.take(str(eid)), spec, prov)
            for spec, rank, stream in eav_streams:
                lines += _eav_lines(stream.take(str(eid)), rank, prov)
            if not lines:
                yield ExcludedRecord(str(eid), "no_text")
                continue
            text = "\n".join(lines)
            truncated = len(text) > max_chars
            yield RecordDoc(str(eid), text[:max_chars],
                            doc_hash(text[:max_chars]), truncated,
                            tuple(prov))
        # Entities are exhausted: every leftover stream row is a NULL
        # or orphan ref -- drain so trailing dirty rows are counted.
        for _s, stream in bridge_streams:
            stream.drain()
        for _s, _r, stream in eav_streams:
            stream.drain()
    finally:
        conn.close()
