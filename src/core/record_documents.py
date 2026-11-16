"""
Retrieval Fabric: projection -> retrievable document contract.

Turns rows of the projection's entity table into text documents derived
ONLY from schema/value structure: own text columns, dimension labels
reached through declared FKs or measured inferred links, many-to-many
labels reached through two-FK bridge tables, and EAV axis values.
No word lists, no column-name lists: foldability is decided by
measured column statistics (bounded aggregate SQL over the full
table, read-only); every rejected candidate lands in
`DocPlan.excluded` with its reason.

The plan-time SQL lives in `record_docs_plan` and the streaming
renderer in `record_docs_render` (private modules) so this file stays
within the project's 350-line bound.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from source_projection import SourceProjection

logger = logging.getLogger(__name__)

RENDERER_VERSION = "rd/2"
MAX_AXIS_PAIRS = 60
MAX_M2M_LABELS = 40
MAX_LABEL_COLS = 2
MIN_NONNULL_RATIO = 0.01
NUMERIC_TEXT_RE = re.compile(r"^[\d\s.,+\-:/%]+$")


@dataclass(frozen=True)
class EavSpec:
    table: str
    entity_col: str
    key_col: str
    value_col: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> EavSpec:
        return EavSpec(**d)


@dataclass(frozen=True)
class DimSpec:
    entity_col: str
    parent_table: str
    parent_key_col: str
    label_cols: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> DimSpec:
        return DimSpec(d["entity_col"], d["parent_table"],
                       d["parent_key_col"], tuple(d["label_cols"]))


@dataclass(frozen=True)
class BridgeSpec:
    bridge_table: str
    entity_col: str
    other_col: str
    other_table: str
    other_key_col: str
    label_cols: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> BridgeSpec:
        return BridgeSpec(d["bridge_table"], d["entity_col"],
                          d["other_col"], d["other_table"],
                          d["other_key_col"], tuple(d["label_cols"]))


@dataclass(frozen=True)
class DocPlan:
    entity_table: str
    id_column: str
    own_columns: tuple[str, ...]
    eav: tuple[EavSpec, ...]
    dimensions: tuple[DimSpec, ...]
    bridges: tuple[BridgeSpec, ...]
    excluded: tuple[tuple[str, str], ...]
    renderer_version: str = RENDERER_VERSION
    #: Most identifying own column (distinct_ratio >= 0.5) -- the hit
    #: `name` source. None for body-only sources (first line then).
    display_column: str | None = None

    def to_dict(self) -> dict:
        return {
            "entity_table": self.entity_table,
            "id_column": self.id_column,
            "own_columns": list(self.own_columns),
            "display_column": self.display_column,
            "eav": [e.to_dict() for e in self.eav],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "bridges": [b.to_dict() for b in self.bridges],
            "excluded": [list(e) for e in self.excluded],
            "renderer_version": self.renderer_version,
        }

    @staticmethod
    def from_dict(d: dict) -> DocPlan:
        return DocPlan(
            entity_table=d["entity_table"],
            id_column=d["id_column"],
            own_columns=tuple(d["own_columns"]),
            eav=tuple(EavSpec.from_dict(x) for x in d["eav"]),
            dimensions=tuple(DimSpec.from_dict(x) for x in d["dimensions"]),
            bridges=tuple(BridgeSpec.from_dict(x) for x in d["bridges"]),
            excluded=tuple(tuple(e) for e in d["excluded"]),
            renderer_version=d["renderer_version"],
            display_column=d.get("display_column"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class RecordDoc:
    entity_id: str
    text: str
    doc_hash: str
    truncated: bool
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class ExcludedRecord:
    entity_id: str | None
    reason: str


@dataclass
class RenderStats:
    """Merge-stream hygiene counters filled by
    `iter_record_documents`: EAV/bridge rows whose entity ref is NULL
    or absent from the entity table. They are skipped, never merged."""
    null_ref_rows: int = 0
    orphan_rows: int = 0


def doc_hash(text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(RENDERER_VERSION.encode())
    h.update(b"\n")
    h.update(text.encode())
    return h.hexdigest()


def _ro_connect(db_path: str) -> sqlite3.Connection:
    uri = "file:{}?mode=ro".format(
        os.path.abspath(db_path).replace("\\", "/"))
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def plan_documents(proj: SourceProjection, db_path: str
                   ) -> DocPlan | None:
    """Plan from projection structure; None if no entity table."""
    from record_docs_plan import build_plan  # deferred: plan imports us
    return build_plan(proj, db_path)


def iter_record_documents(db_path: str, plan: DocPlan, *,
                          max_chars: int = 1500,
                          stats: RenderStats | None = None
                          ) -> Iterator[RecordDoc | ExcludedRecord]:
    """One RecordDoc-or-ExcludedRecord per entity row, ordered by
    CAST(id AS TEXT); streaming (see `record_docs_render`). `stats`,
    when given, collects null/orphan merge-stream ref counts."""
    from record_docs_render import render_documents
    return render_documents(db_path, plan, max_chars=max_chars,
                            stats=stats)
