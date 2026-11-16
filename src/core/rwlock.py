"""
Retrieval Fabric: in-process readers-writer lock (private).

Writer-preferring: once a writer waits, new readers queue behind it,
so a compaction cannot be starved by a stream of searches. Used by
`vector_store` to keep `compact()`'s slot renumbering + vector-file
swap from interleaving with `search_dense`/`get`/`get_many` reads.
`apply` never renumbers, so it deliberately does NOT take this lock.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class RWLock:
    """Many readers OR one writer; writers win the queue."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._cond.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()
