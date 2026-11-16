"""
F09/F10: canonical source-projection layer.

The profiler's name-based semantic types are hints, not ground truth: a
column called `amount_due` can hold ratings while a column called `cena`
holds prices. This module turns a profiler snapshot into a versioned
projection with explicit evidence tiers and resolves a role only when the
evidence justifies it:

  * structural evidence (primary keys, cardinality patterns) survives
    renaming of tables, columns and keys;
  * a name hint resolves a role only when it leaves exactly one candidate
    (or breaks a tie between structural candidates);
  * value evidence can veto a hint (a bounded small scale contradicts a
    price hypothesis);
  * an unresolved role stays visible with all of its candidates -- consumers
    degrade honestly (no price axis, no budget filter) instead of guessing
    from a token.

General introspection by itself does not determine whether two numeric
columns mean price, rating, dosage or measurement. Ambiguity is reported,
never silently resolved.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import entity_select
from profiler import (
    DatabaseKnowledgeSnapshot,
    DomainAgnosticProfiler,
    TableProfile,
)
from source_eav_discovery import (
    collect_eav_tables,
    discover_text_links,
    structural_eav_hypothesis,
)

PROJECTION_VERSION = 1

EVIDENCE_STRUCTURAL = "structural"   # PK / cardinality patterns -- survives renaming
EVIDENCE_VALUE = "value"             # value-distribution evidence
EVIDENCE_NAME_HINT = "name_hint"     # column-name keyword -- the weakest tier

ROLES = ("identity", "name", "category", "price", "stock", "performance",
         "merchant")

# Roles a consumer may rely on for ranking/filtering need at least a name
# hint: a lone numeric column could be a price, a rating, a dosage or a
# measurement, and introspection alone cannot tell. Identity, name and
# category may resolve on structural evidence alone.
_HINT_REQUIRED_ROLES = ("price", "stock", "performance")

_PERF_HINT_TOKENS = ("performance", "perf", "benchmark", "продуктивність")

# Labelled merchant/offer axis tokens; kept disjoint from product categories.
_MERCHANT_TOKENS = frozenset({
    "shop", "merchant", "seller", "vendor", "store", "supplier",
    "retailer", "магазин", "продавець", "постачальник",
})

# Audit/timestamp columns are excluded from categorical axes by name token and ISO date checks.
_TEMPORAL_TOKENS = frozenset({
    "created", "updated", "modified", "timestamp", "datetime",
    "date", "time", "day", "month", "year",
})
_ISO_DATE_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2})?")


def _looks_temporal(col) -> bool:
    if _name_tokens(col.name) & _TEMPORAL_TOKENS:
        return True
    samples = [s for s in col.distinct_sample_values
               if isinstance(s, str)]
    if len(samples) < 2:
        return False
    hits = sum(1 for s in samples if _ISO_DATE_RE.match(s))
    return hits >= len(samples) / 2


def _looks_temporal_or_prose(col) -> bool:
    if _looks_temporal(col):
        return True
    # Long multi-word sentence columns (descriptions/snippets) are prose text, not facet categories.
    if _name_tokens(getattr(col, "name", "")) & {"category", "категорія"}:
        return False
    samples = [s.strip() for s in getattr(col, "distinct_sample_values", ()) if isinstance(s, str) and s.strip()]
    if len(samples) < 2:
        return False
    avg_len = sum(len(s) for s in samples) / len(samples)
    avg_words = sum(len(s.split()) for s in samples) / len(samples)
    return avg_len > 64 or avg_words > 8


def _name_tokens(name: str) -> Set[str]:
    return {t for t in re.split(r"[^0-9a-zA-Zа-яА-ЯёЁіІїЇєЄґҐ]+",
                                name.lower()) if t}


@dataclass
class Evidence:
    kind: str
    detail: str


@dataclass
class RoleCandidate:
    table: str
    column: str
    evidence: List[Evidence] = field(default_factory=list)
    vetoed: bool = False


@dataclass
class RoleResolution:
    role: str
    table: Optional[str] = None
    column: Optional[str] = None
    candidates: List[RoleCandidate] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.table is not None and self.column is not None


def _is_numeric(col) -> bool:
    dtype = (col.data_type or "").upper()
    if any(t in dtype for t in ("INT", "FLOAT", "REAL", "NUMERIC", "DOUBLE")):
        return True
    for sample in col.distinct_sample_values:
        if isinstance(sample, bool):
            continue
        if isinstance(sample, (int, float)):
            return True
    return False


def _is_text(col) -> bool:
    dtype = (col.data_type or "").upper()
    if any(t in dtype for t in ("TEXT", "CHAR", "CLOB", "VARCHAR")):
        return True
    return any(isinstance(s, str) for s in col.distinct_sample_values)


def quote_ident(identifier: str) -> str:
    """Quote an SQLite identifier, even one containing quotes or spaces."""
    return '"' + str(identifier).replace('"', '""') + '"'


# Vocabulary-size ceiling for an axis to stand in for a textual role.
_AXIS_ROLE_MAX_CARD = {"category": 20, "merchant": 20}


_q = quote_ident  # internal shorthand

# Text-containment link discovery bounds -- a dimension label column
# is only probed when its vocabulary is small enough to be an axis.
MAX_TLINK_VOCAB = 200       # dimension labels vocabulary cap
MAX_TLINK_ROWS = 20000      # entity rows scanned for containment
MIN_TLINK_COVERAGE = 0.30   # fraction of entities containing >=1 label


class SourceProjection:
    """Versioned canonical projection of one SQLite source."""

    def __init__(self, db_path: str, snapshot: DatabaseKnowledgeSnapshot,
                 trace_listener=None):
        self.db_path = db_path
        self.snapshot = snapshot
        # Optional live listener: receives every trace event as it is
        # emitted (init-job streaming); replay still works off
        # build_trace afterwards. Must be set before _build below.
        self._trace_listener = trace_listener
        self.version = PROJECTION_VERSION
        self.eav_tables: Dict[str, Dict[str, str]] = {}
        self.entity_candidates: List[RoleCandidate] = []
        self.entity_table: Optional[str] = None
        # True when the structural pick is honestly contestable: a
        # folded lookup carried free text, or the top two candidates
        # tied on facets within a 3x row count. alternatives names the
        # runner-ups worth surfacing to the user, never silently.
        self.entity_ambiguous: bool = False
        self.entity_alternatives: list = []
        self.unassigned_numeric: List[str] = []
        self.roles: Dict[str, RoleResolution] = {}
        # Relations the conductor verified by measured value overlap --
        # undeclared foreign keys the schema never wrote down.
        self.inferred_links: list[dict] = []
        # Roles bound to an EAV key axis rather than a column: the
        # conductor proposed "category lives in axis 'X' of table T" and
        # the measured axis vocabulary + entity coverage admitted it.
        self.axis_roles: dict[str, dict] = {}
        # Logical links with NO foreign key at all: a dimension table
        # whose labels appear as tokens inside entity text ('42' inside
        # 'Кеди Nike 42 розмір' matching sizes.label). Discovered by
        # measured containment coverage, never assumed.
        self.text_links: list[dict] = []
        # Ordered build events for the inspection canvas -- every step
        # of the construction is replayable, with its evidence attached.
        self.build_trace: list[dict] = []
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, role: str) -> RoleResolution:
        return self.roles.get(role, RoleResolution(role))

    # ------------------------------------------------------------------
    # LLM-assisted semantic proposals (advisory layer)
    # ------------------------------------------------------------------
    # The model may PROPOSE a binding for a role the deterministic pass
    # could not resolve, but the proposal only stands if the column
    # exists and passes the role's own structural predicate -- the model
    # names a hypothesis, the data decides. This is the honest middle
    # ground between "scripts can't read semantics" and "model guesses
    # are facts": the proposal is recorded with provenance either way.
    def apply_llm_proposal(self, role: str, column: str,
                           rationale: str = "") -> str:
        if role not in ROLES:
            return f"rejected: unknown role {role!r}"
        res = self.roles.get(role)
        if res is not None and res.resolved:
            return f"skipped: {role} already resolved"
        table = self.snapshot.tables.get(self.entity_table or "")
        if table is None:
            return "rejected: no entity table"
        col = table.columns.get(column)
        if col is None:
            return f"rejected: column {column!r} not on entity table"

        rows = table.row_count or 0
        low_card = (rows >= 3 and 2 <= col.distinct_count <= max(20, 0.2 * rows)
                    and col.distinct_count < rows)
        checks = {
            "name": _is_text(col) and rows > 0
                    and col.distinct_count >= 0.5 * rows,
            "category": _is_text(col) and low_card
                        and not _looks_temporal(col),
            "merchant": _is_text(col) and low_card,
            "price": _is_numeric(col),
            "stock": _is_numeric(col),
            "performance": _is_numeric(col),
        }
        ok = checks.get(role)
        if ok is None:
            return f"rejected: role {role!r} is structural-only"
        if not ok:
            if res is not None:
                res.candidates.append(RoleCandidate(
                    table.name, column,
                    [Evidence("llm_hypothesis",
                              (rationale or "model proposal")
                              + " -- rejected by data checks")],
                    vetoed=True))
            self.trace_event(
                "llm_veto", f"LLM: {role}→{column} відхилено",
                "не пройшло структурні перевірки",
                [f"role:{role}"])
            return f"rejected: {column!r} fails {role} structural checks"

        ev = [Evidence("llm_hypothesis", rationale or "model proposal"),
              Evidence(EVIDENCE_STRUCTURAL,
                       "post-hoc verified by projection predicates")]
        cand = RoleCandidate(table.name, column, ev)
        res = self.roles.setdefault(role, RoleResolution(role))
        res.candidates.append(cand)
        res.table = table.name
        res.column = column
        self.trace_event(
            "llm_role", f"LLM: {role} → {table.name}.{column}",
            (rationale or "model proposal")[:160],
            [f"edge:role:{role}", f"role:{role}"])
        return f"resolved: {table.name}.{column}"

    def apply_axis_role(self, role: str, eav_table: str, axis: str,
                        rationale: str = "",
                        conn: sqlite3.Connection | None = None) -> str:
        """Bind an unresolved role to an EAV key axis -- verified by the
        measured axis vocabulary and entity coverage, not by the model's
        word. An axis that exists on a fraction of entities is an
        attribute, not a role: coverage < 50% of the entity table is a
        rejection, as is a vocabulary outside the role's cardinality
        class."""
        if role not in ROLES:
            return f"rejected: unknown role {role!r}"
        res = self.roles.get(role)
        if res is not None and res.resolved:
            return f"skipped: {role} already resolved"
        # Numeric roles and identity are structural-only: a value inside
        # an EAV axis cannot stand in for a dedicated column. Textual
        # low-cardinality roles CAN live on an axis -- a schema that
        # keeps its taxonomy in key/value pairs has no category column.
        max_card = _AXIS_ROLE_MAX_CARD.get(role)
        if max_card is None:
            return f"rejected: role {role!r} cannot be axis-bound"
        meta = self.eav_tables.get(eav_table)
        if meta is None:
            return f"rejected: {eav_table!r} is not a detected EAV table"
        key_col = meta.get("key_column")
        val_col = meta.get("value_column")
        ent_col = meta.get("entity_fk") or meta.get("entity_column")
        if not (key_col and val_col and ent_col):
            return "rejected: EAV mapping lacks key/value/entity columns"

        own = conn is None
        if own:
            conn = sqlite3.connect(self.db_path)
        try:
            n_axis_entities = conn.execute(
                f"SELECT COUNT(DISTINCT {_q(ent_col)}) "
                f"FROM {_q(eav_table)} WHERE {_q(key_col)} = ?",
                (axis,)).fetchone()[0]
            vocab = conn.execute(
                f"SELECT COUNT(DISTINCT {_q(val_col)}) "
                f"FROM {_q(eav_table)} WHERE {_q(key_col)} = ?",
                (axis,)).fetchone()[0]
        finally:
            if own:
                conn.close()

        ent = self.snapshot.tables.get(self.entity_table or "")
        n_entities = ent.row_count if ent else 0
        coverage = (n_axis_entities / n_entities) if n_entities else 0.0
        card_ok = 2 <= vocab <= max(max_card, 0.2 * (n_entities or 1))

        if n_axis_entities == 0:
            return f"rejected: axis {axis!r} has no rows"
        if not card_ok or coverage < 0.5:
            reason = (f"vocab {vocab}" if not card_ok else
                      f"coverage {coverage:.0%} < 50%")
            self.trace_event(
                "llm_veto",
                f"LLM: {role}←вісь {eav_table}.{axis} відхилено",
                reason, [f"role:{role}"])
            return f"rejected: axis {axis!r} fails checks ({reason})"

        self.axis_roles[role] = {
            "table": eav_table, "axis": axis,
            "entity_column": ent_col, "key_column": key_col,
            "value_column": val_col,
            "vocab_size": vocab,
            "coverage": round(coverage, 3),
            "evidence": "llm_hypothesis+axis_vocab+coverage",
        }
        self.trace_event(
            "axis_role",
            f"{role} ← вісь EAV {eav_table}.{key_col}={axis!r}",
            f"словник {vocab} значень, покриття {coverage:.0%}"
            + (f" · {rationale}" if rationale else ""),
            [f"role:{role}", f"edge:axis:{eav_table}:{role}"])
        return (f"resolved_axis: {eav_table} axis {axis!r} "
                f"(vocab {vocab}, coverage {coverage:.0%})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "database": self.snapshot.database_name,
            "entity_table": self.entity_table,
            "entity_ambiguous": self.entity_ambiguous,
            "entity_alternatives": list(self.entity_alternatives),
            "entity_candidates": [self._candidate_dict(c) for c in self.entity_candidates],
            "eav_tables": dict(self.eav_tables),
            "inferred_links": list(self.inferred_links),
            "axis_roles": dict(self.axis_roles),
            "text_links": list(self.text_links),
            "build_trace": list(self.build_trace),
            "roles": {
                role: {
                    "resolved": res.resolved,
                    "table": res.table,
                    "column": res.column,
                    "candidates": [self._candidate_dict(c) for c in res.candidates],
                }
                for role, res in self.roles.items()
            },
            "unassigned_numeric": list(self.unassigned_numeric),
        }

    @staticmethod
    def _candidate_dict(c: RoleCandidate) -> Dict[str, Any]:
        return {
            "table": c.table,
            "column": c.column,
            "vetoed": c.vetoed,
            "evidence": [{"kind": e.kind, "detail": e.detail} for e in c.evidence],
        }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def trace_event(self, kind: str, label: str, detail: str = "",
                    targets: list | None = None) -> None:
        ev = {"seq": len(self.build_trace), "kind": kind,
              "label": label, "detail": detail,
              "targets": targets or []}
        self.build_trace.append(ev)
        if self._trace_listener is not None:
            try:
                self._trace_listener(ev)
            except Exception:
                pass  # a listener must never break the build

    def _build(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            self.trace_event(
                "snapshot",
                f"знімок схеми: {len(self.snapshot.tables)} таблиць",
                ", ".join(
                    f"{t} ({p.row_count})"
                    for t, p in self.snapshot.tables.items()),
                [f"table:{t}" for t in self.snapshot.tables])
            self._collect_eav_tables(conn)
            self._choose_entity_table(conn)
            if self.entity_table is not None:
                self._resolve_roles(conn, self.snapshot.tables[self.entity_table])
                self._discover_text_links(conn)
            # Declared foreign keys are structural edges -- record each
            for tname, tprof in self.snapshot.tables.items():
                for col in tprof.columns.values():
                    if col.is_foreign_key and col.fk_target_table:
                        self.trace_event(
                            "fk",
                            f"{tname}.{col.name} → "
                            f"{col.fk_target_table}.{col.fk_target_column}",
                            "declared FK",
                            [f"edge:fk:{tname}:{col.fk_target_table}"])
        finally:
            conn.close()

    # -- EAV ------------------------------------------------------------

    def _collect_eav_tables(self, conn: sqlite3.Connection) -> None:
        collect_eav_tables(self, conn, _is_text, _is_numeric)

    def _structural_eav_hypothesis(
        self, conn: sqlite3.Connection, name: str, table: TableProfile
    ) -> Optional[Dict[str, str]]:
        return structural_eav_hypothesis(self, conn, name, table, _is_text, _is_numeric)

    def _discover_text_links(self, conn: sqlite3.Connection) -> None:
        discover_text_links(
            self,
            conn,
            _is_text,
            max_vocab=MAX_TLINK_VOCAB,
            max_rows=MAX_TLINK_ROWS,
            min_coverage=MIN_TLINK_COVERAGE,
        )

    # -- entity table ---------------------------------------------------

    def _choose_entity_table(self, conn: sqlite3.Connection) -> None:
        candidates: List[Tuple[RoleCandidate, int]] = []
        for name, table in self.snapshot.tables.items():
            if name in self.eav_tables or name.startswith("__"):
                continue  # __-prefixed tables are ingest artifacts, not data
            if len(table.primary_keys) != 1 or table.row_count <= 0:
                continue
            text_cols = [
                c for c in table.columns.values()
                if c.name not in set(table.primary_keys) and _is_text(c)
            ]
            if not text_cols:
                continue
            evidence = [
                Evidence(EVIDENCE_STRUCTURAL, "single-column primary key"),
                Evidence(EVIDENCE_STRUCTURAL, "has text attribute columns"),
                Evidence(EVIDENCE_VALUE, f"row_count={table.row_count}"),
            ]
            candidates.append((RoleCandidate(name, table.primary_keys[0], evidence), table.row_count))
            self.trace_event(
                "entity_candidate", f"кандидат на entity: {name}",
                f"pk={table.primary_keys[0]}, rows={table.row_count}",
                [f"table:{name}"])

        self.entity_candidates = [c for c, _ in candidates]
        if not candidates:
            self.entity_table = None
            self.trace_event("entity", "entity-таблиця не знайдена",
                             "жоден кандидат не пройшов сигнатуру")
            return
        # Structural selection lives in entity_select (Steps 0/A/B):
        # delimiter-boundary detail demotion, lookup folding over the
        # FK edge graph, facet ranking (bridges + EAV attachments +
        # non-candidate FK children). All measured, none name-based.
        cand_names = {c[0].table for c in candidates}
        res = entity_select.rank_entity_candidates(
            conn, self.snapshot.tables, self.eav_tables,
            self.inferred_links, cand_names)
        for child, info in res["detail_of"].items():
            self.trace_event(
                "entity_detail",
                f"{child} -- дочірня деталь {info['parent']}",
                f"pk-prefix coverage={info['coverage']:.2f} через "
                f"{child}.{info['via']} -> "
                f"{info['parent']}.{info['parent_pk']}",
                [f"table:{child}", f"table:{info['parent']}"])
        for eav, info in res["eav_attach"].items():
            self.trace_event(
                "entity_fold",
                f"EAV {eav} прив'язано до {info['candidate']}",
                f"pk-containment={info['coverage']:.2f}",
                [f"table:{eav}", f"table:{info['candidate']}"])
        for lookup, owners in res["folded"].items():
            self.trace_event(
                "entity_fold",
                f"{lookup} -- lookup для {', '.join(owners)}",
                "дитина має FK у lookup, зворотного немає",
                [f"table:{lookup}"]
                + [f"table:{y}" for y in owners])
        self.entity_alternatives = res["alternatives"]
        self.entity_ambiguous = res["ambiguous"]
        self.entity_table = res["entity"]
        rank_detail = ", ".join(
            f"{n}(facets={res['facets'].get(n, 0)},"
            f"rows={self.snapshot.tables[n].row_count})"
            for n in res["ranked"][:5])
        self.trace_event(
            "entity_rank", f"ранжування кандидатів: {rank_detail}",
            f"ambiguous={self.entity_ambiguous}, "
            f"alternatives={self.entity_alternatives}",
            [f"table:{n}" for n in res["ranked"][:5]])
        if self.entity_table is None:
            self.trace_event("entity", "entity-таблиця не знайдена",
                             "усі кандидати згорнуті/деталі")
            return
        self.trace_event(
            "entity", f"entity-таблиця: {self.entity_table}",
            f"перша у структурному ранжуванні"
            f"{' (ambiguous)' if self.entity_ambiguous else ''}",
            [f"table:{self.entity_table}"])

    # -- roles ----------------------------------------------------------

    def _resolve_roles(self, conn: sqlite3.Connection, table: TableProfile) -> None:
        cols = list(table.columns.values())
        pk = set(table.primary_keys)
        text_cols = [c for c in cols if _is_text(c)]
        numeric_cols = [c for c in cols if _is_numeric(c)]

        # identity: the single-column primary key, structurally.
        if len(table.primary_keys) == 1:
            self.roles["identity"] = RoleResolution(
                "identity", table.name, table.primary_keys[0],
                [RoleCandidate(table.name, table.primary_keys[0],
                               [Evidence(EVIDENCE_STRUCTURAL, "primary key")])],
            )
        else:
            self.roles["identity"] = RoleResolution(
                "identity", None, None,
                [RoleCandidate(table.name, c.name,
                               [Evidence(EVIDENCE_STRUCTURAL, "part of a composite key")])
                 for c in cols if c.name in pk],
            )

        self.roles["name"] = self._resolve_descriptive_role(
            conn, table, "name", text_cols, pk,
            structural=lambda c: (
                table.row_count > 0 and c.distinct_count >= 0.5 * table.row_count
            ),
            structural_detail=lambda c: (
                f"high distinct ratio ({c.distinct_count}/{table.row_count})"
            ),
            hint_types=("text_search",),
            extra_hint=None,
            primary_tokens=("name", "назва", "title"),
        )
        # Merchant axis resolves BEFORE category and must carry a name
        # hint: low cardinality alone is shared with genuine categories,
        # so structure can only support, never nominate, the role.
        self.roles["merchant"] = self._resolve_descriptive_role(
            conn, table, "merchant", text_cols, pk,
            structural=lambda c: (
                table.row_count >= 3
                and 2 <= c.distinct_count <= max(20, 0.2 * table.row_count)
                and c.distinct_count < table.row_count
            ),
            structural_detail=lambda c: (
                f"low cardinality ({c.distinct_count}/{table.row_count})"
            ),
            hint_types=(),
            extra_hint=lambda name: bool(
                _name_tokens(name) & _MERCHANT_TOKENS),
            primary_tokens=tuple(sorted(_MERCHANT_TOKENS)),
            require_hint=True,
        )
        merchant_col = (self.roles["merchant"].column
                        if self.roles["merchant"].resolved else None)
        self.roles["category"] = self._resolve_descriptive_role(
            conn, table, "category", text_cols, pk,
            exclude=frozenset({merchant_col}) if merchant_col else frozenset(),
            structural=lambda c: (
                table.row_count >= 3
                # A constant column (distinct == 1) carries no category
                # information, and a column where every row is distinct is an
                # identifier or a name -- the bounds keep tiny tables from
                # making everything look low-cardinality.
                and 2 <= c.distinct_count <= max(20, 0.2 * table.row_count)
                and c.distinct_count < table.row_count
            ),
            structural_detail=lambda c: (
                f"low cardinality ({c.distinct_count}/{table.row_count})"
            ),
            hint_types=("category",),
            extra_hint=None,
            primary_tokens=("category", "категорія"),
            veto=_looks_temporal_or_prose,
            veto_detail="temporal values or prose description -- not a category axis",
        )

        self.roles["price"] = self._resolve_hinted_numeric_role(
            conn, table, "price", numeric_cols, ("price",)
        )
        self.roles["stock"] = self._resolve_hinted_numeric_role(
            conn, table, "stock", numeric_cols, ("stock",)
        )
        self.roles["performance"] = self._resolve_hinted_numeric_role(
            conn, table, "performance", numeric_cols, (),
            extra_hint=lambda name: any(tok in name.lower() for tok in _PERF_HINT_TOKENS),
        )

        hinted_names = {
            res.column for role, res in self.roles.items()
            if role in _HINT_REQUIRED_ROLES and res.column
        }
        self.unassigned_numeric = sorted(
            c.name for c in numeric_cols
            if c.name not in pk and c.name not in hinted_names
        )

        # Replayable trace: every role outcome and every vetoed
        # candidate is a visible step, not a silent decision.
        for role, res in self.roles.items():
            if res.resolved:
                self.trace_event(
                    "role", f"{role} → {res.table}.{res.column}",
                    "; ".join(e.detail for c in res.candidates
                              for e in c.evidence
                              if c.column == res.column)[:160],
                    [f"edge:role:{role}", f"role:{role}"])
            else:
                self.trace_event(
                    "role_missing", f"{role}: НЕ РОЗВ'ЯЗАНО",
                    "немає колонки або докази неоднозначні",
                    [f"role:{role}"])
            for c in res.candidates:
                if c.vetoed:
                    self.trace_event(
                        "veto", f"{role}: {c.column} відкинуто",
                        "; ".join(e.detail for e in c.evidence)[:160],
                        [f"role:{role}"])

    def _resolve_descriptive_role(
        self,
        conn: sqlite3.Connection,
        table: TableProfile,
        role: str,
        text_cols: List[Any],
        pk: set,
        structural,
        structural_detail,
        hint_types: Tuple[str, ...],
        extra_hint,
        primary_tokens: Tuple[str, ...],
        exclude: Set[str] = frozenset(),
        require_hint: bool = False,
        veto=None,
        veto_detail: str = "vetoed",
    ) -> RoleResolution:
        candidates: List[RoleCandidate] = []
        for c in text_cols:
            if c.name in pk or c.name in exclude:
                continue
            evidence = []
            if structural(c):
                evidence.append(Evidence(EVIDENCE_STRUCTURAL, structural_detail(c)))
            hinted = c.semantic_type in hint_types or (extra_hint and extra_hint(c.name))
            if hinted:
                exact = c.name.lower() in primary_tokens
                evidence.append(Evidence(
                    EVIDENCE_NAME_HINT,
                    "exact name match" if exact else f"profiler hint: {c.semantic_type}",
                ))
            vetoed = bool(veto and veto(c))
            if vetoed:
                evidence.append(Evidence(EVIDENCE_VALUE, veto_detail))
            if evidence:
                candidates.append(RoleCandidate(
                    table.name, c.name, evidence, vetoed=vetoed))

        live = [c for c in candidates if not c.vetoed]
        structural_cands = [c for c in live
                            if any(e.kind == EVIDENCE_STRUCTURAL for e in c.evidence)]
        hinted_cands = [c for c in live
                        if any(e.kind == EVIDENCE_NAME_HINT for e in c.evidence)]
        winner: Optional[RoleCandidate] = None
        if require_hint:
            # Structure can support the role but cannot nominate it:
            # without a name hint the column is unresolved, not guessed.
            if len(hinted_cands) == 1:
                winner = hinted_cands[0]
        elif len(structural_cands) == 1:
            winner = structural_cands[0]
        elif len(structural_cands) > 1:
            exact = [c for c in hinted_cands if c.column.lower() in primary_tokens]
            if len(exact) == 1:
                winner = exact[0]
            elif len(hinted_cands) == 1:
                winner = hinted_cands[0]
        else:
            winner = hinted_cands[0] if len(hinted_cands) == 1 else None
        return RoleResolution(role, table.name if winner else None,
                              winner.column if winner else None, candidates)

    def _resolve_hinted_numeric_role(
        self,
        conn: sqlite3.Connection,
        table: TableProfile,
        role: str,
        numeric_cols: List[Any],
        semantic_types: Tuple[str, ...],
        extra_hint=None,
    ) -> RoleResolution:
        candidates: List[RoleCandidate] = []
        for c in numeric_cols:
            hinted = c.semantic_type in semantic_types or (
                extra_hint is not None and extra_hint(c.name)
            )
            if not hinted:
                continue
            evidence = [Evidence(EVIDENCE_NAME_HINT,
                                 f"profiler hint: {c.semantic_type or 'name pattern'}")]
            vetoed = False
            if role == "price" and self._bounded_small_scale(conn, table.name, c.name):
                vetoed = True
                evidence.append(Evidence(
                    EVIDENCE_VALUE,
                    "bounded_small_scale (rating-like values) contradicts a price hypothesis",
                ))
            candidates.append(RoleCandidate(table.name, c.name, evidence, vetoed=vetoed))

        live = [c for c in candidates if not c.vetoed]
        winner = live[0] if len(live) == 1 else None
        return RoleResolution(role, table.name if winner else None,
                              winner.column if winner else None, candidates)

    @staticmethod
    def _bounded_small_scale(conn: sqlite3.Connection, table: str, column: str) -> bool:
        """
        Value evidence that contradicts a price hypothesis: every distinct
        value is numeric, inside [0, 10], and there are at most 20 of them --
        the shape of a rating, not of a price.
        """
        try:
            rows = conn.execute(
                f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
                f"WHERE {_q(column)} IS NOT NULL LIMIT 200"
            ).fetchall()
        except Exception:
            return False
        values = [r[0] for r in rows]
        if not values:
            return False
        nums = [v for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) != len(values):
            return False
        return len(nums) <= 20 and all(0 <= float(v) <= 10 for v in nums)

_CACHE: Dict[str, Tuple[Tuple[float, int], SourceProjection]] = {}


def invalidate_cache(db_path: str) -> None:
    """Drop a source's cached projection -- used by the re-analyze job
    so the rebuild cannot observe prior inferred links/roles."""
    _CACHE.pop(os.path.abspath(db_path), None)


def build_source_projection(db_path: str,
                            trace_listener=None,
                            fresh: bool = False) -> SourceProjection:
    """Profile a database and build (a cached) canonical projection for it.

    `trace_listener`, if given, is invoked with each build-trace event as
    it is emitted -- this is how init jobs stream progress live. On a
    cache hit the listener is still attached so later events (e.g. a
    conductor run on the same projection) stream too. `fresh=True`
    bypasses the cache -- the explicit "re-analyze with my parameters"
    path; the old projection object is dropped so prior inferred links
    do not leak into the rebuild."""
    ap = os.path.abspath(db_path)
    if not os.path.exists(ap):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    st = os.stat(ap)
    key = (st.st_mtime, st.st_size)
    if fresh:
        _CACHE.pop(ap, None)
    cached = _CACHE.get(ap)
    if cached is not None and cached[0] == key:
        if trace_listener is not None:
            cached[1]._trace_listener = trace_listener
        return cached[1]
    snapshot = DomainAgnosticProfiler(ap).profile_database()
    projection = SourceProjection(ap, snapshot,
                                  trace_listener=trace_listener)
    _CACHE[ap] = (key, projection)
    return projection
