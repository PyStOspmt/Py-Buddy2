"""Domain registry proxy and product/compatibility resolution helpers."""

from typing import Dict, Any, Optional
import os
import re
from hypergraph import KnowledgeHypergraph
from server_context import ctx


class _DomainRegistryProxy(dict):
    def __getitem__(self, key: str):
        return ctx.get_domain_bundle(key).hypergraph
    def __contains__(self, key: object):
        return isinstance(key, str) and key in ctx.domains
    def __iter__(self):
        return iter(ctx.domains)
    def __len__(self):
        return len(ctx.domains)
    def keys(self):
        return ctx.domains.keys()
    def values(self):
        return [b.hypergraph for b in ctx.domains.values()]
    def items(self):
        return [(k, b.hypergraph) for k, b in ctx.domains.items()]

DOMAINS: Dict[str, Any] = _DomainRegistryProxy()
DOMAIN_NAMES = tuple(ctx.domains.keys())

def __getattr__(name: str):
    if name == "DOMAIN_NAMES":
        return tuple(ctx.domains.keys())
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def hypergraph_for(domain: str) -> KnowledgeHypergraph:
    """The knowledge graph that is authoritative for `domain`."""
    return ctx.get_hypergraph_for_domain(domain)

def _resolve_from_hypergraph(hg, text: str, limit: int = 12):
    """
    Domain-agnostic entity resolution via exact substring & token overlap.
    Zero domain-specific heuristics (no '(Box)' or '(Rev)' regexes).

    F13: exact key normalisation (spaces stripped) and tokenisation (spaces
    kept) are different operations. The previous version tokenised the
    space-stripped key, so the whole query collapsed into one glued token
    and the `len(common) >= 2` overlap requirement was unreachable -- the
    token-overlap branch was dead code and only verbatim full names ever
    resolved.
    """
    from metrics import _norm_key, normalize
    haystack = _norm_key(text)
    if not haystack:
        return []

    # Tokens come from the space-preserving normalisation, never from the
    # glued key.
    query_tokens = set(re.findall(r"[\w\d]+", normalize(text)))
    if not query_tokens:
        return []

    hits = []
    for node in sorted(hg.nodes.values(), key=lambda n: -len(n.name)):
        key = _norm_key(node.name)
        if not key:
            continue
        # Direct exact substring match
        if key in haystack:
            hits.append(node)
            if len(hits) >= limit:
                break
            continue

        # Token overlap over real tokens (covers name variants like
        # 'Box', 'Rev 1.0', partial model names, etc.)
        node_tokens = set(re.findall(r"[\w\d]+", normalize(node.name)))
        n_sig = {t for t in node_tokens if len(t) >= 2}
        if not n_sig:
            continue
        common = n_sig.intersection(query_tokens)
        if len(common) / len(n_sig) >= 0.7 and len(common) >= 2:
            hits.append(node)
            if len(hits) >= limit:
                break
    return hits


#: F17: intent cues for the deterministic compatibility shortcut. This is a
#: task-type cue list, not domain knowledge: a miss degrades to the general
#: agent path (which can still answer correctly with tools), so it can never
#: turn a price question into a compatibility verdict -- it can only fail to
#: take a shortcut.
_COMPAT_CUES = (
    "сумісн", "сумест", "совмест", "компатиб", "несумісн",
    "підійдуть разом", "чи можна разом", "разом з", "разом із",
    "в одному комплекті", "комплект", "зібрати", "збірка",
    "compatible", "compatibility", "work together", "fit together",
    "go together", "pair with", "build with",
)


def _is_compatibility_question(query: str) -> bool:
    low = (query or "").lower()
    return any(cue in low for cue in _COMPAT_CUES)


def _resolve_named_products(hg, db_path: Optional[str], query: str, limit: int = 12):
    """
    F14: entity resolution must not depend on what the UI happened to browse.

    The graph is seeded with a bounded sample and grows only when catalogue
    pages are opened, so the previous graph-only resolution could answer
    about a product before the user browsed to it and fail after a restart.
    Resolution now also runs against the full source catalogue through the
    DB index, lazily loading any product the graph has not seen yet.

    If the source catalogue cannot be consulted, this degrades to the
    previous graph-only capability -- it never fabricates a resolution.
    """
    named = _resolve_from_hypergraph(hg, query)
    if not db_path or not os.path.exists(db_path):
        return named
    try:
        from catalog_query import resolve_mentions
        from domain_catalog import DomainCatalogRegistry
        for cand in resolve_mentions(db_path, query, limit=limit):
            if cand.item_id in hg.nodes:
                continue
            node = DomainCatalogRegistry.load_product_dynamically(db_path, cand.item_id)
            if node is not None:
                hg.add_product(node)
    except Exception:
        # Degrade to graph-only resolution rather than fail the request;
        # this is the pre-fix capability, never a fabricated answer.
        return _resolve_from_hypergraph(hg, query)
    return _resolve_from_hypergraph(hg, query)
