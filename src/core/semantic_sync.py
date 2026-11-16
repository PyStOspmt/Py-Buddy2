"""
Retrieval Fabric: incremental sync from source DB to VectorStore.

Streams rendered RecordDocs in entity_id order and merge-joins them
against `store.hashes()` (also entity_id-ordered): only docs whose
doc_hash changed are re-embedded, in batches, each batch committed by
one `store.apply` so a cancelled/failed run keeps its committed work
and can resume. Deletions are applied ONLY after the full source pass
completes -- a partial pass cannot prove absence. An entity that no
longer renders any text (ExcludedRecord) is also removed from the
index: serving its stale document would be a fabrication.

Embedder failure surfaces as `EmbedderUnavailable` caught once into
`report.error` with `complete=False`; nothing is fabricated.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field

from embedders import EmbedderUnavailable
from record_documents import (
    DocPlan,
    ExcludedRecord,
    RenderStats,
    iter_record_documents,
    plan_documents,
)
from source_lifecycle import source_version
from vector_store import StoredRecord, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    source_rows: int = 0
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    embedded_texts: int = 0
    truncated_docs: int = 0
    null_ref_rows: int = 0      # EAV/bridge rows with NULL entity ref
    orphan_rows: int = 0        # refs absent from the entity table
    # model-window truncations: None when the embedder cannot count
    # tokens (explicitly unknown -- never a fabricated 0)
    model_truncated_docs: int | None = None
    max_tokens: int | None = None
    elapsed_s: float = 0.0
    complete: bool = False
    generation: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class _StoreCursor:
    """Peekable iterator over (entity_id, doc_hash) store rows."""

    def __init__(self, it: Iterator):
        self._it = it
        self.cur = next(it, None)

    def advance_below(self, eid: str, deleted: list) -> None:
        while self.cur is not None and self.cur[0] < eid:
            deleted.append(self.cur[0])
            self.cur = next(self._it, None)

    def matches(self, eid: str) -> bool:
        return self.cur is not None and self.cur[0] == eid

    def consume(self) -> None:
        self.cur = next(self._it, None)


def _account_tokens(embedder, texts: list, rep: SyncReport) -> None:
    """Count document-side tokens per embedded batch. Embedders
    without `count_tokens` leave the fields None (unknown, never 0)."""
    fn = getattr(embedder, "count_tokens", None)
    if not callable(fn):
        return
    counts = fn(texts)
    limit = getattr(embedder, "max_tokens", None)
    if counts is None or limit is None:
        return
    rep.max_tokens = limit
    if rep.model_truncated_docs is None:
        rep.model_truncated_docs = 0
    rep.model_truncated_docs += sum(1 for c in counts if c > limit)


def _flush(batch, store: VectorStore, embedder, rep: SyncReport) -> None:
    texts = [d.text for d, _kind in batch]
    vecs = embedder.embed_documents(texts)
    _account_tokens(embedder, texts, rep)
    recs = [StoredRecord(d.entity_id, d.doc_hash, d.text, vecs[i],
                         "|".join(d.provenance))
            for i, (d, _k) in enumerate(batch)]
    store.apply(recs, [])
    rep.embedded_texts += len(texts)
    for _d, kind in batch:
        if kind == "inserted":
            rep.inserted += 1
        else:
            rep.updated += 1
    batch.clear()


def _consume(item, cur: _StoreCursor, rep: SyncReport, batch: list,
             deletes_pending: list) -> None:
    """Classify one streamed row against the store cursor."""
    rep.source_rows += 1
    if isinstance(item, ExcludedRecord):
        rep.excluded[item.reason] = rep.excluded.get(item.reason, 0) + 1
        if item.entity_id is not None:
            # A live record whose entity no longer renders is stale
            # index content -- queue removal, not re-embedding.
            cur.advance_below(item.entity_id, deletes_pending)
            if cur.matches(item.entity_id):
                cur.consume()
                deletes_pending.append(item.entity_id)
        return
    if item.truncated:
        rep.truncated_docs += 1
    cur.advance_below(item.entity_id, deletes_pending)
    if cur.matches(item.entity_id):
        _sid, old_hash = cur.cur
        cur.consume()
        if old_hash == item.doc_hash:
            rep.unchanged += 1
        else:
            batch.append((item, "updated"))
    else:
        batch.append((item, "inserted"))


def _meta_updates(plan: DocPlan, db_path: str,
                  source_stamp: str | None = None) -> dict:
    if source_stamp is None:
        stamp = source_version(db_path)
        source_stamp = json.dumps(list(stamp)) if stamp else "null"
    return {
        "plan_json": plan.to_json(),
        "renderer_version": plan.renderer_version,
        "source_stamp": source_stamp,
    }


def _source_pass(db_path, plan, cur, rep, store, embedder, batch_size,
                 should_cancel, progress, deletes_pending, batch,
                 stats) -> str:
    """Stream-compare source vs store; returns done|cancelled|failed."""
    for item in iter_record_documents(db_path, plan, stats=stats):
        if should_cancel is not None and should_cancel():
            return "cancelled"
        _consume(item, cur, rep, batch, deletes_pending)
        if len(batch) >= batch_size:
            try:
                _flush(batch, store, embedder, rep)
            except EmbedderUnavailable as exc:
                rep.error = str(exc)
                return "failed"
            if progress is not None:
                progress({"processed": rep.source_rows,
                          "inserted": rep.inserted,
                          "updated": rep.updated,
                          "embedded": rep.embedded_texts})
    return "done"


def sync_source(db_path: str, proj, store: VectorStore, embedder, *,
                batch_size: int = 64,
                should_cancel: Callable[[], bool] | None = None,
                progress: Callable[[dict], None] | None = None,
                source_stamp: str | None = None,
                ) -> SyncReport:
    """`source_stamp`, when given, is the caller's own stamp captured
    BEFORE the pass started -- a source edited mid-sync then leaves
    the stored stamp behind reality (honestly stale), never falsely
    fresh."""
    t0 = time.time()
    rep = SyncReport()
    plan = plan_documents(proj, db_path)
    if plan is None:
        rep.error = "unavailable: no entity table"
        rep.elapsed_s = round(time.time() - t0, 3)
        rep.generation = store.generation
        return rep

    cur = _StoreCursor(iter(store.hashes()))
    batch: list = []
    deletes_pending: list = []
    stats = RenderStats()
    status = _source_pass(db_path, plan, cur, rep, store, embedder,
                          batch_size, should_cancel, progress,
                          deletes_pending, batch, stats)
    rep.null_ref_rows = stats.null_ref_rows
    rep.orphan_rows = stats.orphan_rows
    if status == "done":
        status = _finish(cur, rep, batch, deletes_pending, store,
                         embedder, plan, db_path, source_stamp)
    rep.elapsed_s = round(time.time() - t0, 3)
    rep.generation = store.generation
    if rep.complete:
        # The report is meta, not a record change -> set_meta (no
        # generation bump).
        store.set_meta({"sync_report": json.dumps(rep.to_dict())})
    return rep


def _finish(cur, rep, batch, deletes_pending, store, embedder, plan,
            db_path, source_stamp=None) -> str:
    """Drain the store cursor (absent = deleted), flush the last batch,
    then apply deletions + meta. Deletions happen ONLY here: a partial
    pass cannot prove absence."""
    while cur.cur is not None:
        deletes_pending.append(cur.cur[0])
        cur.consume()
    if batch:
        try:
            _flush(batch, store, embedder, rep)
        except EmbedderUnavailable as exc:
            rep.error = str(exc)
            return "failed"
    store.apply([], deletes_pending,
                meta_updates=_meta_updates(plan, db_path, source_stamp))
    rep.deleted = len(deletes_pending)
    rep.complete = True
    return "done"
