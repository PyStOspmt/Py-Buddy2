"""Status/search serialization for SemanticIndex (kept out of
semantic_service.py for size). Functions, not methods: they read
through the index's own guarded accessors."""
import json
import os
import sqlite3

from source_lifecycle import source_stamp

_NAME_CHARS = 120


def peek_manifest(directory: str, *, expected_model_id: str,
                  expected_dim: int | None = None) -> dict:
    """Read-only inspect of `<dir>/manifest.sqlite` without loading weights."""
    from vector_store import StoreCorrupt, StoreMismatch, _MANIFEST

    mpath = os.path.join(directory, _MANIFEST)
    uri = f"file:{mpath.replace(chr(92), '/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        stored_model = meta.get("model_id")
        if stored_model is not None and stored_model != expected_model_id:
            raise StoreMismatch(
                f"store model_id {stored_model!r} != {expected_model_id!r}")
        stored_dim_raw = meta.get("dim")
        if (expected_dim is not None and stored_dim_raw is not None
                and int(stored_dim_raw) != int(expected_dim)):
            raise StoreMismatch(f"store dim {stored_dim_raw} != {expected_dim}")
        dim = (int(stored_dim_raw) if stored_dim_raw is not None
               else int(expected_dim or 0))
        row_slots = conn.execute("SELECT MAX(slot) FROM records").fetchone()
        n_slots = int(row_slots[0]) + 1 if row_slots and row_slots[0] is not None else 0
        vfile = meta.get("vector_file", "vectors.bin")
        if n_slots > 0:
            vpath = os.path.join(directory, vfile)
            if not os.path.exists(vpath) or os.path.getsize(vpath) < n_slots * dim * 2:
                raise StoreCorrupt(f"missing or short vector file {vfile}")
        live = int(conn.execute(
            "SELECT COUNT(*) FROM records WHERE live=1").fetchone()[0] or 0)
    finally:
        conn.close()
    disk = sum(os.path.getsize(os.path.join(directory, n))
               for n in os.listdir(directory)
               if os.path.isfile(os.path.join(directory, n)))
    return {
        "meta": meta,
        "generation": int(meta.get("generation") or 0),
        "live": live,
        "model_id": stored_model or expected_model_id,
        "dim": dim,
        "disk_bytes": disk,
    }


def status_of(idx) -> dict:
    with idx._lock:
        state, reason = idx._state()
        store = idx._store
        snap = idx._last_peek if store is None else None
        index = None
        if store is not None and state != "unavailable":
            # live_count rides the cached snapshot bitmap and the disk
            # size is a directory listing -- no O(N) SQL per search.
            disk = sum(os.path.getsize(os.path.join(store._dir, n))
                       for n in os.listdir(store._dir)
                       if os.path.isfile(os.path.join(store._dir, n)))
            index = {"generation": store.generation,
                     "live": store.live_count(),
                     "model_id": store.model_id,
                     "dim": store.dim,
                     "disk_bytes": disk}
            rep = stored_report(idx)
            stamped = idx._meta("source_stamp")
            cov = coverage_of(rep, store)
        elif snap is not None and state != "unavailable":
            index = {"generation": snap["generation"],
                     "live": snap["live"],
                     "model_id": snap["model_id"],
                     "dim": snap["dim"],
                     "disk_bytes": snap["disk_bytes"]}
            rep = stored_report(idx, meta_map=snap["meta"])
            stamped = snap["meta"].get("source_stamp")
            cov = coverage_of(rep, None, live_override=snap["live"])
        else:
            rep = idx._last_sync
            stamped = None
            cov = coverage_of(rep, None)
        now = source_stamp(idx.db_path) if idx.db_path else None
        proj = idx._project()
        return {
            "state": state, "reason": reason,
            "progress": dict(idx._progress),
            "index": index,
            "coverage": cov,
            "fresh": (None if stamped is None else stamped == now),
            "source_stamp_indexed": stamped,
            "source_stamp_now": now,
            "last_sync": rep,
            "entity": {"table": proj.entity_table if proj else None,
                       "ambiguous": bool(proj and
                                         proj.entity_ambiguous),
                       "alternatives": (list(proj.entity_alternatives)
                                        if proj else [])},
        }


def stored_report(idx, meta_map: dict | None = None) -> dict | None:
    raw = (meta_map.get("sync_report") if meta_map is not None
           else idx._meta("sync_report"))
    if not raw:
        return idx._last_sync
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return idx._last_sync


def coverage_of(rep, store, live_override: int | None = None) -> dict | None:
    if rep is None:
        return None
    live = live_override
    if live is None and store is not None:
        try:
            live = store.live_count()
        except (OSError, RuntimeError) as exc:
            # a corrupt/missing index reports unknown coverage,
            # never a fabricated count
            import logging
            logging.getLogger(__name__).debug(
                "live_count failed: %s", exc)
    return {"source_rows": rep.get("source_rows"),
            "indexed": live,
            "excluded": rep.get("excluded"),
            "truncated_docs": rep.get("truncated_docs"),
            "model_truncated_docs": rep.get("model_truncated_docs"),
            "max_tokens": rep.get("max_tokens"),
            "null_ref_rows": rep.get("null_ref_rows"),
            "orphan_rows": rep.get("orphan_rows")}


def hit_dict(h, text: str = "", display_column: str | None = None
             ) -> dict:
    name = _name_of(text or h.snippet, display_column)
    return {"entity_id": h.entity_id, "name": name,
            "score": h.score, "dense_rank": h.dense_rank,
            "dense_sim": h.dense_sim,
            "lexical_rank": h.lexical_rank,
            "lexical_score": h.lexical_score,
            "snippet": h.snippet, "provenance": h.provenance}


def _name_of(text: str, display_column: str | None) -> str:
    """Hit name: the plan's display_column value (rd/2), else the
    first rendered line's value."""
    if display_column:
        prefix = f"{display_column}: "
        for line in text.split("\n"):
            if line.startswith(prefix):
                return line[len(prefix):][: _NAME_CHARS]
    first = text.split("\n", 1)[0]
    return (first.split(": ", 1)[1] if ": " in first
            else first)[:_NAME_CHARS]


def display_column_of(idx) -> str | None:
    raw = idx._meta("plan_json")
    if not raw:
        return None
    try:
        from record_documents import DocPlan
        return DocPlan.from_dict(json.loads(raw)).display_column
    except Exception:  # noqa: BLE001 -- unreadable plan_json degrades to first-line names
        return None


def named_hits(idx, res) -> list:
    """Hit dicts with `name` from the plan's display_column -- read
    from the full stored text (one bulk hydration), so a 240-char
    snippet boundary can never truncate the name line."""
    texts = idx._ensure_store().get_many(
        [h.entity_id for h in res.hits])
    display = display_column_of(idx)
    return [hit_dict(h, (texts.get(h.entity_id).text
                         if texts.get(h.entity_id) else ""),
                     display)
            for h in res.hits]


def hyde_hits(idx, ranking, k) -> list:
    """Hit dicts for a HyDE ranking: fused-RRF score, per-channel
    ranks; every field comes from stored records only."""
    store = idx._ensure_store()
    display = display_column_of(idx)
    texts = store.get_many(ranking.ids[:k])
    hits = []
    for rank, eid in enumerate(ranking.ids[:k], 1):
        rec = texts.get(eid)
        hits.append({"entity_id": eid,
                     "name": _name_of(rec.text if rec else "",
                                      display),
                     "score": 1.0 / (60 + rank),
                     "dense_rank": (ranking.mix_ids.index(eid) + 1
                                    if eid in ranking.mix_ids
                                    else None),
                     "dense_sim": None,
                     "lexical_rank": (ranking.lex_ids.index(eid) + 1
                                      if eid in ranking.lex_ids
                                      else None),
                     "lexical_score": None,
                     "snippet": rec.text[:240] if rec else "",
                     "provenance": rec.provenance if rec else ""})
    return hits


def result_of(idx, status, query, mode, k, hits, notes,
              index_status=None) -> dict:
    return {"status": status, "query": query, "mode": mode,
            "k": k, "hits": hits,
            "confidence": {"top1_z": None, "margin": None,
                           "top1_sim": None},
            "index": (index_status if index_status is not None
                      else status_of(idx)),
            "notes": notes}
