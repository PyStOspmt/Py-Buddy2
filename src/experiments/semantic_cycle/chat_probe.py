"""P9: consultant smoke through the real /api/chat (docs/18, C7).

12 frozen natural requests per source (the first test-split record targets
by id, `en_natural` text) are sent with a cheap model under a hard spend
cap. The harness then checks, independently of the system's own gates,
whether the answer is traceable to retrieved evidence:
  * did the agent call `semantic_search` at all,
  * is the remembered record among the retrieved ids,
  * does the answer name a catalogue record that was NOT retrieved
    (evaluator-side name match against the source DB; library and food
    only -- forum posts have no display names).
Counts only; no quality threshold is claimed.
"""

from __future__ import annotations

import re
import sqlite3
import time

MODEL = "deepseek/deepseek-v4.1-flash-1017"
N_PER_SOURCE = 12
NAME_SQL = {"library": "SELECT book_id, title FROM books",
            "food": "SELECT id, name FROM products"}
_ID_RE = re.compile(r"""["']id["']\s*:\s*["']?([^"',}\s]+)""")


def chat_requests(doc: dict, src: str, split: str = "test") -> list[dict]:
    rows = sorted((t for t in doc["targets"]
                   if t["source"] == src and t["kind"] == "record"
                   and t["split"] == split),
                  key=lambda t: t["target_id"])[:N_PER_SOURCE]
    return [{"target_id": str(t["target_id"]),
             "query": t["queries"]["en_natural"]} for t in rows]


def _tool(step: dict) -> str:
    return str(step.get("tool_name")
               or (step.get("action") or {}).get("tool") or "")


def _obs(step: dict) -> str:
    return str(step.get("observation") or step.get("result") or "")


def evidence_ids(steps: list[dict]) -> tuple[set[str], set[str]]:
    sem, sql = set(), set()
    for st in steps:
        tool, text = _tool(st), _obs(st)
        if tool == "semantic_search":
            sem |= set(_ID_RE.findall(text))
        elif tool == "sql_query":
            sql |= set(_ID_RE.findall(text))
    return sem, sql


def _names(db_path: str, src: str) -> dict[str, str]:
    if src not in NAME_SQL:
        return {}
    c = sqlite3.connect("file:" + db_path.replace("\\", "/") + "?mode=ro",
                        uri=True)
    try:
        return {str(i): " ".join(str(n).split()).lower()
                for i, n in c.execute(NAME_SQL[src]) if n and len(str(n)) >= 10}
    finally:
        c.close()


def run(srv, src: str, db_path: str, doc: dict, max_usd: float = 0.15
        ) -> dict:
    srv.api("POST", "/api/set_model", admin=True, json={"model": MODEL})
    names = _names(db_path, src)
    spent, rows = 0.0, []
    for req in chat_requests(doc, src):
        if spent >= max_usd:
            rows.append({"target_id": req["target_id"], "skipped": "budget"})
            continue
        t0 = time.time()
        st, b = srv.api("POST", "/api/chat", timeout=240,
                        json={"query": req["query"], "domain": src})
        steps = b.get("steps") or [] if st == 200 else []
        sem, sql = evidence_ids(steps)
        answer = " ".join(str(b.get("final_answer", "")).split()).lower()
        mentioned = {i for i, n in names.items() if n in answer}
        spent += float(b.get("token_cost") or 0.0)
        rows.append({
            "target_id": req["target_id"], "query": req["query"],
            "http": st, "status": b.get("status"),
            "termination_reason": b.get("termination_reason"),
            "model": b.get("model"), "token_cost": b.get("token_cost"),
            "latency_s": round(time.time() - t0, 1),
            "tools": [_tool(s) for s in steps],
            "semantic_calls": sum(1 for s in steps
                                  if _tool(s) == "semantic_search"),
            "retrieved_ids": sorted(sem)[:50],
            "target_in_retrieved": req["target_id"] in sem,
            "recommended_ids": [str(r.get("id")) for r in
                                b.get("recommended_items") or []],
            "mentioned_record_ids": sorted(mentioned),
            "ungrounded_mentions": sorted(mentioned - sem - sql),
            "final_answer": b.get("final_answer")})
    done = [r for r in rows if "http" in r]
    return {"model": MODEL, "spent_usd": round(spent, 6), "n": len(done),
            "used_semantic_search": sum(1 for r in done
                                        if r["semantic_calls"]),
            "target_in_retrieved": sum(1 for r in done
                                       if r["target_in_retrieved"]),
            "chats_with_ungrounded_mentions": sum(
                1 for r in done if r["ungrounded_mentions"]),
            "http_failures": sum(1 for r in done if r["http"] != 200),
            "rows": rows}
