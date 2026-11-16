"""
F33 (docs/11): source lifecycle primitives.

A registered source used to be identified by its absolute path alone:
ontology and question-bank caches trusted data written for a different
database that happened to occupy the same path, and a missing registered
source silently fell back to the unrelated default PC database.

This module gives the lifecycle layer three small honest primitives:

* `source_version`  -- snapshot identity of the file the engine was built
  against (same (mtime_ns, size) convention as `source_projection`);
* `read_versioned_cache` / `write_versioned_cache` -- caches carry the
  source version inside an envelope and are published atomically
  (tmp + os.replace), so a stale or torn cache is detected and recomputed
  rather than trusted;
* `SourceState`     -- the explicit pending/ready/degraded/failed states
  a domain bundle reports instead of a boolean "initialized".

What this module is NOT: a background job queue. Preparation still runs
inside the request that needs it, but it is now idempotent, versioned,
and state-honest; the async-worker part of F33 remains open.
"""

import json
import os
import tempfile
from enum import Enum
from typing import Any, Optional, Tuple

#: Snapshot identity of a source file: (mtime_ns, size_bytes). Cheap to
#: compute per request and changes whenever the file is replaced or
#: rewritten -- matching the convention `source_projection` already uses.
SourceVersion = Tuple[int, int]


class SourceState(str, Enum):
    #: Registered but the engine has not been built yet.
    PENDING = "pending"
    #: Engine built against the pinned source version.
    READY = "ready"
    #: Built, but part of preparation failed or the source is graph-only;
    #: `state_reason` says exactly what is missing.
    DEGRADED = "degraded"
    #: Source unavailable; requests fail closed -- never substituted.
    FAILED = "failed"


def source_version(db_path: Optional[str]) -> Optional[SourceVersion]:
    if not db_path:
        return None
    try:
        st = os.stat(db_path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def source_stamp(db_path: str) -> str:
    """Deterministic freshness stamp: mtime_ns+size of the database
    file AND its WAL. A second connection's uncheckpointed WAL is a
    real source change the db file alone cannot show."""
    parts = []
    for path in (db_path, f"{db_path}-wal"):
        try:
            st = os.stat(path)
            parts.append([st.st_mtime_ns, st.st_size])
        except OSError:
            parts.append(None)
    return json.dumps(parts)


def read_versioned_cache(
    cache_path: str, expected_version: Optional[SourceVersion]
) -> Optional[Any]:
    """Return the cached payload only if it was written for this version.

    Legacy (pre-envelope) cache files carry no version, so they are treated
    as stale and recomputed -- a permissive read of unversioned data is
    exactly the trust defect being removed.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            envelope = json.load(f)
    except Exception:
        return None  # missing or corrupt: recompute
    if not isinstance(envelope, dict):
        return None
    if (expected_version is not None
            and envelope.get("source_version") != list(expected_version)):
        return None
    return envelope.get("payload")


def write_versioned_cache(
    cache_path: str, version: Optional[SourceVersion], payload: Any
) -> bool:
    """Publish a cache atomically so a concurrent reader never sees a
    torn file: serialize to a sibling tmp file, then os.replace."""
    envelope = {
        "source_version": list(version) if version is not None else None,
        "payload": payload,
    }
    try:
        directory = os.path.dirname(cache_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=directory, prefix=".cache_tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(envelope, f, ensure_ascii=False, indent=2)
            os.replace(tmp, cache_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception:
        # A cache write failure must never fail the request that computed
        # fresh data; the cache is an optimization, not the answer.
        return False
