"""Live-model checks for embedders (Retrieval Fabric Phase 1).

The real-model test is gated: it runs only with
`PYBUDDY_LIVE_LOCAL_MODEL=1` and loads the locally cached
`intfloat/multilingual-e5-small` with `HF_HUB_OFFLINE=1` -- free, local,
no network. The ungated test only asserts typed failure mapping and
runs in the default offline suite.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from embedders import EmbedderUnavailable, SentenceTransformerEmbedder

_LIVE = (os.environ.get("PYBUDDY_LIVE_LOCAL_MODEL") or os.environ.get("AINSA_LIVE_LOCAL_MODEL")) == "1"


@pytest.mark.live_model
@pytest.mark.skipif(
    not _LIVE,
    reason="set PYBUDDY_LIVE_LOCAL_MODEL=1 for the local model")
def test_e5_small_semantic_ranking_live(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    emb = SentenceTransformerEmbedder()
    assert emb.dim == 384
    docs = [
        "a warm insulated boot for winter trails",
        "a thin ceramic coffee mug",
        "a stainless steel kitchen knife",
        "a paperback mystery novel",
        "a cotton summer t-shirt",
    ]
    vecs = emb.embed_documents(docs)
    assert vecs.shape == (5, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    qv = emb.embed_queries(["тепле взуття на зиму"])  # warm winter shoes
    sims = vecs @ qv[0]
    assert int(np.argmax(sims)) == 0
    # real revision bound into the model identity
    assert emb.model_id.startswith(
        "st:intfloat/multilingual-e5-small@")
    revision = emb.model_id.split("@")[1].split(":")[0]
    assert revision != "unknown-revision" and len(revision) >= 8


@pytest.mark.live_model
@pytest.mark.skipif(
    not _LIVE,
    reason="set AINSA_LIVE_LOCAL_MODEL=1 for the local model")
def test_e5_small_token_counting_live(monkeypatch):
    """The real tokenizer counts document-side tokens incl. specials;
    a hand-built long text exceeds the model window."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    emb = SentenceTransformerEmbedder()
    assert emb.max_tokens == 512      # default window since docs/18 scope-change 2
    long_text = " ".join(f"word{i}" for i in range(300))
    counts = emb.count_tokens([long_text, "short text"])
    assert counts[0] > emb.max_tokens
    assert counts[1] < counts[0]


@pytest.mark.live_model   # exercises the REAL loader failure path (bogus name, no weights loaded)
def test_load_failure_maps_to_typed_error(monkeypatch):
    """Offline + unknown model -> EmbedderUnavailable, never vectors."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    emb = SentenceTransformerEmbedder(
        "nonexistent-org/definitely-not-a-real-model")
    with pytest.raises(EmbedderUnavailable):
        emb.embed_documents(["anything"])
    with pytest.raises(EmbedderUnavailable):
        _ = emb.dim


def test_cache_revision_prefers_refs_main(tmp_path, monkeypatch):
    """The refs/main commit hash wins; the lexicographically first
    snapshot dir is only a fallback."""
    from embedders import _cache_revision
    home = tmp_path / "home"
    repo = (home / ".cache" / "huggingface" / "hub"
            / "models--o--m")
    (repo / "snapshots" / "aaa111snapshot").mkdir(parents=True)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda _p: str(home))
    assert _cache_revision("o/m") == "aaa111snapshot"
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("b" * 40, encoding="utf-8")
    assert _cache_revision("o/m") == "b" * 40
    assert _cache_revision("o/absent") == "unknown-revision"


def test_counting_embedder_counters_threadsafe():
    """Counters must survive concurrent instrumentation calls."""
    import threading

    from embedders import CountingEmbedder
    from semantic_fixtures import HashEmbedder
    ce = CountingEmbedder(HashEmbedder(dim=8))

    def worker():
        for _ in range(25):
            ce.embed_documents(["a b c"])
            ce.embed_queries(["q"])

    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert ce.calls == 400
    assert ce.doc_texts == 200 and ce.query_texts == 200
