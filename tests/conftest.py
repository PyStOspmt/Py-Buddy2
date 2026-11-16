"""Shared test path setup.

Product core lives in src/core/, evaluation scripts in src/experiments/,
gateways/runners stay under src/. Every test module can import any of them
by bare module name; the per-file sys.path inserts that still point at src/
are harmless leftovers kept for history.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (
    os.path.join(_ROOT, "src", "core"),
    os.path.join(_ROOT, "src", "experiments"),
    os.path.join(_ROOT, "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_model: requires the locally cached embedding model "
        "(gated by PYBUDDY_LIVE_LOCAL_MODEL=1)")
    scripts_dir = os.path.join(_ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import bootstrap_demo_dbs
    bootstrap_demo_dbs.ensure_demo_databases(_ROOT, rebuild=False)


import pytest


@pytest.fixture(autouse=True)
def _hermetic_embedder(request, monkeypatch):
    """No test without @pytest.mark.live_model may load the real
    embedding model: every default embedder factory yields the labelled
    HashEmbedder test double, and a direct SentenceTransformerEmbedder
    load fails loudly instead of silently slowing the suite."""
    if request.node.get_closest_marker("live_model"):
        yield
        return
    import embedders
    from semantic_fixtures import HashEmbedder

    def _offline_factory():
        return HashEmbedder()

    def _no_load(self):
        raise AssertionError(
            "real embedding model loaded in an offline test")

    monkeypatch.setattr(embedders, "default_embedder", _offline_factory)
    monkeypatch.setattr(embedders, "_DEFAULT_EMBEDDER", None)
    monkeypatch.setattr(
        embedders.SentenceTransformerEmbedder, "_ensure_loaded", _no_load)
    # patch the bound factory on already-imported server modules AND
    # the name `server_context.__init__` resolves -- contexts built
    # under this fixture get the double either way
    _sc = sys.modules.get("server_context")
    if _sc is not None:
        monkeypatch.setattr(_sc, "default_embedder", _offline_factory,
                            raising=False)
        _sc_ctx = getattr(_sc, "ctx", None)
        if _sc_ctx is not None:
            monkeypatch.setattr(_sc_ctx, "embedder_factory",
                                _offline_factory)
    yield


@pytest.fixture(autouse=True)
def _hermetic_server_llm(request, monkeypatch):
    """Prevent offline tests from firing real OpenRouter HTTPS calls via
    SystemContext.runner (e.g. OntoGraph._induce_with_llm / QuestionBank)
    when a developer .env key is present on disk.
    """
    if (
        request.node.get_closest_marker("live_model")
        or "live_arm" in request.node.name
    ):
        yield
        return

    import openrouter_client

    monkeypatch.setattr(
        openrouter_client.OpenRouterAgentRunner,
        "generate",
        lambda self, *args, **kwargs: "",
    )
    monkeypatch.setattr(
        openrouter_client.OpenRouterAgentRunner,
        "answer_once",
        lambda self, *args, **kwargs: "",
    )
    monkeypatch.setattr(
        openrouter_client.OpenRouterAgentRunner,
        "generate_react_step",
        lambda self, *args, **kwargs: {
            "thought": "offline hermetic stub",
            "final_answer": "",
            "recommended_items": [],
        },
    )
    yield


@pytest.fixture(autouse=True)
def _hermetic_runtime_registry(monkeypatch, tmp_path):
    """Isolate tests from the workspace data/sources.runtime.json so test
    registrations and uploads never leak temporary pytest paths into the
    demo registry; tests that pass their own PYBUDDY_RUNTIME_REGISTRY override
    this cleanly via monkeypatch."""
    monkeypatch.setenv(
        "PYBUDDY_RUNTIME_REGISTRY",
        str(tmp_path / "sources.runtime.json"),
    )
    yield
