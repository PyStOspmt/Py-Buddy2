"""Server-wide paths, configuration constants and the model catalogue fallback."""

from typing import Dict, List, Any
import json
import os


# Single repository-root anchor (docs/13 C2): every data/web/tests path is
# derived from here instead of assuming the module sits one level below the root.
# This file lives in src/core/, so the repo root is three levels up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'catalog.db'))
MARKETPLACE_DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'marketplace_rozetka.db'))
AUTO_PARTS_DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'auto_parts.db'))
def env_setting(primary: str, legacy: str = "", default: str = "") -> str:
    """Read primary PYBUDDY_* setting with fallback to legacy AINSA_* alias."""
    for key in (primary, legacy):
        if key and key in os.environ:
            return os.environ[key]
    return default


SESSIONS_DB = os.path.abspath(
    env_setting("PYBUDDY_SESSIONS_DB", "AINSA_SESSIONS_DB")
    or os.path.join(PROJECT_ROOT, 'data', 'sessions.db'))
MODELS_CACHE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'openrouter_models_cache.json'))
WEB_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'web'))

# Offline snapshot of OpenRouter /api/v1/models (2026-11-10). Used only when the
# live catalogue and the local cache are both unavailable; every entry was present
# in the live listing on that date, prices are per million tokens.
CURATED_FALLBACK_MODELS = [
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "name": "NVIDIA: Nemotron 3 Ultra (free)", "context_length": 1000000, "is_free": True, "prompt_price": 0.0, "completion_price": 0.0, "description": ""},
    {"id": "nvidia/nemotron-3.5-lightning:free", "name": "NVIDIA: Nemotron 3.5 Lightning (free)", "context_length": 1000000, "is_free": True, "prompt_price": 0.0, "completion_price": 0.0, "description": ""},
    {"id": "nvidia/nemotron-3.5-super-vl:free", "name": "NVIDIA: Nemotron 3.5 Super VL (free)", "context_length": 1000000, "is_free": True, "prompt_price": 0.0, "completion_price": 0.0, "description": ""},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "name": "NVIDIA: Nemotron 3 Nano Omni (free)", "context_length": 256000, "is_free": True, "prompt_price": 0.0, "completion_price": 0.0, "description": ""},
    {"id": "google/gemma-4-31b-it:free", "name": "Google: Gemma 4 31B (free)", "context_length": 262144, "is_free": True, "prompt_price": 0.0, "completion_price": 0.0, "description": ""},
    {"id": "google/gemma-4-26b-a4b-it:free", "name": "Google: Gemma 4 26B A4B  (free)", "context_length": 262144, "is_free": True, "prompt_price": 0.0, "completion_price": 0.0, "description": ""},
    {"id": "xingchen-agi/xing4.0-29b-a4b:free", "name": "XingChen: Xing4.0 29B A4B (free)", "context_length": 262144, "is_free": True, "prompt_price": 0.0, "completion_price": 0.0, "description": ""},
    {"id": "liquid/lfm-3-1.7b:free", "name": "LiquidAI: LFM3-1.7B (free)", "context_length": 262144, "is_free": True, "prompt_price": 0.0, "completion_price": 0.0, "description": ""},
    {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o", "context_length": 128000, "is_free": False, "prompt_price": 2.5, "completion_price": 10.0, "description": ""},
    {"id": "openai/gpt-4o-mini", "name": "OpenAI: GPT-4o-mini", "context_length": 128000, "is_free": False, "prompt_price": 0.15, "completion_price": 0.6, "description": ""},
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek: DeepSeek V3", "context_length": 163840, "is_free": False, "prompt_price": 0.32, "completion_price": 0.89, "description": ""},
]

def load_openrouter_models() -> List[Dict[str, Any]]:
    if os.path.exists(MODELS_CACHE_PATH):
        try:
            with open(MODELS_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if isinstance(cached, list) and len(cached) > 0:
                    return cached
        except Exception:
            # An unreadable model cache falls back to the curated list.
            pass
    return CURATED_FALLBACK_MODELS

#: F34: a catalog page is bounded. An uncapped LIMIT (or the SQLite -1
#: "unlimited" convention) let one request read the whole source; the
#: applied bound is reported back in the response.
MAX_CATALOG_PAGE = 200

#: F34: bundle validation input and the O(n^2) pairwise-link rendering are
#: bounded; a truncated link set is reported, never silently partial.
MAX_BUNDLE_ITEMS = 200
MAX_VIZ_LINK_NODES = 40
