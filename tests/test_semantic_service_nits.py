"""Handoff-4 nits: hermetic embedder guard, cheap status, embedded
progress, -O-safe rebuild guards. SYNTHETIC fixtures, HashEmbedder.
"""
from __future__ import annotations

import subprocess
import sys

import pytest
from embedders import SentenceTransformerEmbedder
from semantic_fixtures import HashEmbedder, make_flat_db
from semantic_service import SemanticIndex


def test_offline_guard_blocks_real_model_load():
    """The conftest autouse fixture must make a direct model load
    fail loudly in a non-live test."""
    with pytest.raises(AssertionError, match="real embedding model"):
        SentenceTransformerEmbedder()._ensure_loaded()


def test_status_search_does_no_stats_count(tmp_path):
    db = make_flat_db(str(tmp_path / "n.db"))
    idx = SemanticIndex("t", db, embedder_factory=lambda: HashEmbedder())
    idx.sync_blocking(reason="t")
    store = idx._ensure_store()
    calls = {"n": 0}
    orig = store.stats

    def counted():
        calls["n"] += 1
        return orig()

    store.stats = counted
    idx.search("entry", k=3)
    idx.status()
    assert calls["n"] == 0
    idx.close()


def test_progress_embedded_moves(tmp_path):
    db = make_flat_db(str(tmp_path / "p.db"), n=200)
    idx = SemanticIndex("t", db, embedder_factory=lambda: HashEmbedder())
    idx.sync_blocking(reason="t")
    p = idx.status()["progress"]
    assert p["processed"] == 200 and p["embedded"] == 200
    idx.close()


def test_rebuild_dir_refused_without_assert(tmp_path):
    """The .arf guard must hold under `python -O` (asserts stripped)."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        "sys.path.insert(0, 'src/core')\n"
        "from semantic_service import SemanticIndex\n"
        "i = SemanticIndex('t', 'x.db', embedder_factory=lambda: None)\n"
        "i.index_dir = '/tmp/not_arf'\n"
        "try:\n"
        "    i._rebuild_dir()\n"
        "except ValueError:\n"
        "    print('ok-notarf')\n"
        "i2 = SemanticIndex('t', '/a/b/x.db',"
        "                     embedder_factory=lambda: None)\n"
        "i2.index_dir = '/other/dir.arf'\n"
        "try:\n"
        "    i2._rebuild_dir()\n"
        "except ValueError:\n"
        "    print('ok-nonsidecar')\n")
    out = subprocess.run(
        [sys.executable, "-O", str(script)],
        capture_output=True, text=True, timeout=60, check=False)
    assert "ok-notarf" in out.stdout, out.stderr
    assert "ok-nonsidecar" in out.stdout, out.stderr


def test_top1_sim_in_confidence(tmp_path):
    db = make_flat_db(str(tmp_path / "c.db"))
    idx = SemanticIndex("t", db, embedder_factory=lambda: HashEmbedder())
    idx.sync_blocking(reason="t")
    out = idx.search("entry number 3", k=3)
    conf = out["confidence"]
    assert set(conf) == {"top1_z", "margin", "top1_sim"}
    assert conf["top1_sim"] is not None
    idx.close()


def test_cold_status_does_not_load_heavy_embedder_weights(tmp_path):
    """Read-only status() must inspect absent or synced manifests without loading weights."""
    import json
    from source_lifecycle import source_stamp
    from vector_store import VectorStore

    db = make_flat_db(str(tmp_path / "cold.db"))
    emb = SentenceTransformerEmbedder()  # offline conftest blocks _ensure_loaded()
    idx = SemanticIndex("cold", db, embedder_factory=lambda: emb)
    assert idx.status()["state"] == "absent"
    assert not (tmp_path / "cold.db.arf").exists()

    st = VectorStore.open(str(tmp_path / "cold.db.arf"),
                          model_id=emb.model_id, dim=384)
    st.set_meta({
        "source_stamp": source_stamp(db),
        "sync_report": json.dumps({"source_rows": 8, "complete": True}),
    })
    st.close()

    res = idx.status()
    assert res["state"] == "ready"
    assert res["index"]["model_id"] == emb.model_id
    assert res["index"]["dim"] == 384
    assert res["fresh"] is True
