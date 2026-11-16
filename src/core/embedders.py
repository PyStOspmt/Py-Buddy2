"""
Retrieval Fabric: embedder abstractions (Phase 1).

The retrieval layer is domain-independent and must work offline. This
module defines the minimal `Embedder` protocol, a lazy local
sentence-transformers implementation, an instrumented wrapper used both
for evidence and for tests, and a process-wide default.

Honesty contract: a load or encode failure raises `EmbedderUnavailable`
with the cause attached -- the layer never substitutes zero/random
vectors and never silently degrades retrieval quality.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

#: Model families whose documented input protocol requires an explicit
#: "query: "/"passage: " instruction prefix. This is the published input
#: contract of the E5 embedding family (model card), not domain
#: knowledge: without the prefixes the vectors sit in the wrong space.
_E5_PREFIXED_BASENAMES = ("multilingual-e5", "e5-")


class EmbedderUnavailable(RuntimeError):
    """The embedding backend cannot serve vectors right now."""


class Embedder(Protocol):
    """Dense text embedder: documents and queries may differ (e5-style).

    `max_tokens`/`count_tokens` are OPTIONAL capabilities: an embedder
    without them leaves model-window accounting unknown (consumers must
    use getattr, never assume)."""

    model_id: str
    dim: int
    max_tokens: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 (n, dim) L2-normalised document vectors."""
        ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 (n, dim) L2-normalised query vectors."""
        ...

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Document-side token counts incl. prefixes + special tokens."""
        ...


def _normalise(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


def _cache_revision(model_name: str) -> str:
    """Cached HF model revision: the commit hash in `refs/main`, or the
    lexicographically first snapshot dir only when no ref exists."""
    hub = os.path.join(os.path.expanduser("~"), ".cache",
                       "huggingface", "hub")
    repo_dir = "models--" + model_name.replace("/", "--")
    repo = os.path.join(hub, repo_dir)
    try:
        with open(os.path.join(repo, "refs", "main"),
                  encoding="utf-8") as f:
            ref = f.read().strip()
        if ref:
            return ref
    except OSError:
        pass
    try:
        entries = sorted(os.listdir(os.path.join(repo, "snapshots")))
    except OSError:
        return "unknown-revision"
    return entries[0] if entries else "unknown-revision"


class SentenceTransformerEmbedder:
    """Lazy local sentence-transformers embedder (default: e5-small).

    The model is loaded on first use under one lock so concurrent first
    calls pay one load. For `multilingual-e5*`/`e5-*` basenames the
    documented "query: "/"passage: " prefixes are applied -- that is the
    model's input protocol, not a domain assumption; other basenames are
    embedded raw. Any load/encode failure raises `EmbedderUnavailable`.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small",
                 *, max_seq_length: int = 512, batch_size: int = 32,
                 device: str = "cpu"):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._tokenizer_obj = None
        # RLock: _tokenizer may need _ensure_loaded while already
        # holding the lock.
        self._lock = threading.RLock()
        self._dim: int | None = None
        basename = model_name.rsplit("/", 1)[-1].lower()
        self._uses_e5_prefix = basename.startswith(_E5_PREFIXED_BASENAMES)

    # -- identity ---------------------------------------------------

    @property
    def model_id(self) -> str:
        revision = _cache_revision(self.model_name)
        return (f"st:{self.model_name}@{revision}"
                f":len{self.max_seq_length}")

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def peek_dim(self) -> int | None:
        """Return cached embedding dimension without triggering weight load."""
        return self._dim

    @property
    def max_tokens(self) -> int:
        """Document token budget = max_seq_length (the model's own
        truncation point -- inputs beyond it are cut silently)."""
        return self.max_seq_length

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Document-side token counts via the model's own tokenizer,
        including the e5 passage prefix and special tokens,
        truncation=False."""
        items = list(texts)
        if self._uses_e5_prefix:
            items = ["passage: " + t for t in items]
        enc = self._tokenizer()(
            items, truncation=False, add_special_tokens=True)
        return [len(x) for x in enc["input_ids"]]

    def _tokenizer(self):
        """The tokenizer WITHOUT weights (AutoTokenizer), falling back
        to the loaded model's tokenizer -- lazy, one lock."""
        if self._tokenizer_obj is not None:
            return self._tokenizer_obj
        with self._lock:
            if self._tokenizer_obj is not None:
                return self._tokenizer_obj
            try:
                from transformers import AutoTokenizer
                self._tokenizer_obj = AutoTokenizer.from_pretrained(
                    self.model_name)
            except Exception as exc:  # noqa: BLE001 -- fallback path
                logger.debug("tokenizer-only load failed, using the "
                             "model's tokenizer: %s", exc)
                self._ensure_loaded()   # typed failure if it can't
                self._tokenizer_obj = self._model.tokenizer
            return self._tokenizer_obj

    # -- loading ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(
                    self.model_name, device=self.device)
                model.max_seq_length = self.max_seq_length
                dim = int(model.get_sentence_embedding_dimension())
            except Exception as exc:
                raise EmbedderUnavailable(
                    f"cannot load embedding model {self.model_name!r}: "
                    f"{exc}") from exc
            self._dim = dim
            self._model = model

    # -- encoding ---------------------------------------------------

    def _encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._uses_e5_prefix:
            tag = "query: " if kind == "query" else "passage: "
            items = [tag + t for t in items]
        try:
            vecs = self._model.encode(
                items, batch_size=self.batch_size,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False)
        except Exception as exc:
            raise EmbedderUnavailable(
                f"embedding encode failed for {self.model_name!r}: "
                f"{exc}") from exc
        arr = _normalise(np.asarray(vecs, dtype=np.float32))
        if arr.ndim != 2 or arr.shape[0] != len(items):
            raise EmbedderUnavailable(
                f"embedder returned shape {arr.shape} for "
                f"{len(items)} inputs")
        return arr

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "document")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")


class CountingEmbedder:
    """Instrumented `Embedder` wrapper: counts calls and texts served.

    A real wrapper used for evidence (sync accounting) and by tests to
    prove exactly which rows were re-embedded after a point change.
    """

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.calls = 0
        self.doc_texts = 0
        self.query_texts = 0
        self._count_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def dim(self) -> int:
        return self.inner.dim

    def peek_dim(self) -> int | None:
        fn = getattr(self.inner, "peek_dim", None)
        return fn() if callable(fn) else self.inner.dim

    @property
    def max_tokens(self) -> int | None:
        return getattr(self.inner, "max_tokens", None)

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        fn = getattr(self.inner, "count_tokens", None)
        return fn(texts) if callable(fn) else None

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        with self._count_lock:
            self.calls += 1
            self.doc_texts += len(texts)
        return self.inner.embed_documents(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        with self._count_lock:
            self.calls += 1
            self.query_texts += len(texts)
        return self.inner.embed_queries(texts)


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_EMBEDDER: SentenceTransformerEmbedder | None = None


def default_embedder() -> Embedder:
    """Process-wide lazy singleton of the default local embedder."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_EMBEDDER is None:
                _DEFAULT_EMBEDDER = SentenceTransformerEmbedder()
    return _DEFAULT_EMBEDDER


class SemaphoreGatedEmbedder:
    """Proxies an embedder, holding `sem` only around the embedding
    calls (not the whole caller pass) -- a CPU-oversubscription guard
    for concurrent index builds."""

    def __init__(self, inner: Embedder, sem: threading.Semaphore):
        self._inner = inner
        self._sem = sem

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def embed_documents(self, texts):
        with self._sem:
            return self._inner.embed_documents(texts)

    def embed_queries(self, texts):
        with self._sem:
            return self._inner.embed_queries(texts)
