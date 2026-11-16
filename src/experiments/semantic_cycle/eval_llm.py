"""Budgeted, fully logged OpenRouter client for evaluation-side generation.

* exact model id, no silent fallback model (the production runner rotates
  fallbacks; an evaluation must know which model wrote each query);
* every call is appended to a JSONL log (prompt sha, response, usage, cost,
  latency); the API key is never logged;
* hard USD budget: the run aborts instead of overspending.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"


class BudgetExceeded(RuntimeError):
    pass


def load_api_key() -> str:
    import sys
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src" / "core"))
    from openrouter_client import OpenRouterAgentRunner
    return OpenRouterAgentRunner().api_key


class BudgetedLLM:
    def __init__(self, model: str, max_usd: float, log_path: Path,
                 price_in_per_m: float, price_out_per_m: float):
        self.model = model
        self.max_usd = max_usd
        self.log_path = log_path
        self.price_in, self.price_out = price_in_per_m, price_out_per_m
        self.key = load_api_key()
        self._lock = threading.Lock()
        self.spent = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _cost(self, usage: dict) -> float:
        reported = usage.get("cost")
        if isinstance(reported, (int, float)):
            return float(reported)
        return (usage.get("prompt_tokens", 0) * self.price_in
                + usage.get("completion_tokens", 0) * self.price_out) / 1e6

    def complete_text(self, prompt: str, temperature: float = 0.3) -> str:
        """Single-user-message completion (the shape of the production
        runner's `generate`), same budget/retry/logging as JSON calls;
        returns the raw reply, '' after exhausted retries."""
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent:.4f} >= cap ${self.max_usd}")
        payload = {"model": self.model, "temperature": temperature,
                   "max_tokens": 800, "usage": {"include": True},
                   "reasoning": {"enabled": False},
                   "messages": [{"role": "user", "content": prompt}]}
        sha = hashlib.sha256(prompt.encode()).hexdigest()
        for attempt in range(1, 5):
            status, latency, body = self._post(payload)
            usage = body.get("usage") or {}
            content = ((body.get("choices") or [{}])[0].get("message")
                       or {}).get("content") or ""
            self._record(status, latency, body, usage, content, sha, attempt)
            if status == 200 and content.strip():
                return content
            if not (status in (-1, 429) or status >= 500) or attempt == 4:
                return ""
            time.sleep(2.0 ** attempt)
        return ""

    def complete_json(self, system: str, user: str,
                      temperature: float = 0.5) -> dict:
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent:.4f} >= cap ${self.max_usd}")
        payload = {"model": self.model, "temperature": temperature,
                   "max_tokens": 800, "usage": {"include": True},
                   "reasoning": {"enabled": False},
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        sha = hashlib.sha256((system + "\n" + user).encode()).hexdigest()
        for attempt in range(1, 5):
            status, latency, body = self._post(payload)
            usage = body.get("usage") or {}
            content = ((body.get("choices") or [{}])[0].get("message")
                       or {}).get("content") or ""
            self._record(status, latency, body, usage, content, sha, attempt)
            transient = status in (-1, 429) or status >= 500
            if status == 200 and content.strip():
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    transient = True
            if not transient or attempt == 4:
                raise RuntimeError(f"llm call failed: HTTP {status}")
            time.sleep(2.0 ** attempt)
        raise RuntimeError("unreachable")

    def _post(self, payload: dict) -> tuple[int, float, dict]:
        t0 = time.time()
        try:
            resp = requests.post(
                URL, timeout=90, data=json.dumps(payload),
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json",
                         "X-Title": "Py-Buddy eval"})
        except requests.exceptions.RequestException:
            return -1, time.time() - t0, {}
        body = resp.json() if resp.status_code == 200 else {}
        return resp.status_code, time.time() - t0, body

    def _record(self, status: int, latency: float, body: dict, usage: dict,
                content: str, sha: str, attempt: int) -> None:
        cost = self._cost(usage) if usage else 0.0
        with self._lock:
            self.spent += cost
            self.calls += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model_requested": self.model,
                    "model_served": body.get("model"), "attempt": attempt,
                    "status": status, "latency_s": round(latency, 2),
                    "prompt_sha256": sha, "usage": usage,
                    "cost_usd": cost, "response": content,
                }, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        return {"model": self.model, "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "spent_usd": round(self.spent, 6), "cap_usd": self.max_usd}
