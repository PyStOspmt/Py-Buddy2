"""Generate and FREEZE the evaluation request set (docs/18).

For every target (label set or remembered record) a cheap LLM writes three
requests: `en_natural` (any words), `en_strict` (must share no word stem
with the target's reference text) and `uk` (Ukrainian, cross-lingual).
Generated requests are validated programmatically and retried; the final
file is hashed and never edited. The LLM only writes requests -- relevance
comes from the data owner's labels (targets.py), never from the model.

Usage: python gen_queries.py [--dry-run] [--limit N] [--max-usd 0.4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_text as et  # noqa: E402
from eval_llm import BudgetedLLM  # noqa: E402
from eval_targets import ROOT, Target, all_targets  # noqa: E402

MODEL = "deepseek/deepseek-v4.1-flash-1017"
PROMPT_VERSION = "q1"
DATE = "20261112"
OUT = ROOT / "docs" / f"semantic_queries_{DATE}.json"
LOG = ROOT / "docs" / f"semantic_queries_llm_log_{DATE}.jsonl"
MAX_RETRIES = 2
CORPUS = {
    "library": "a public-domain library catalogue (books with authors and "
               "subject headings)",
    "forum": "an online discussion forum (1990s usenet-style posts on many "
             "boards)",
    "food": "an international packaged-food product database (names, "
            "brands, categories)",
}
SYSTEM = ("You create evaluation requests for a retrieval system. You never "
          "explain. You output exactly one JSON object and nothing else.")


def forbidden_words(t: Target, cap: int = 60) -> list[str]:
    words = sorted((w for w in et.content_tokens(t.forbidden_text)
                    if len(w) >= 4), key=lambda w: (-len(w), w))
    return sorted(words[:cap])


def build_prompt(t: Target, feedback: str = "") -> str:
    head = f"Corpus: {CORPUS[t.source]}.\n"
    if t.kind == "label_set":
        body = (f"Target: {t.brief}.\nSample members (for meaning only):\n"
                + "\n".join(f"- {e}" for e in t.examples)
                + "\nWrite requests a real person might type to find items "
                  "of this target.")
    else:
        body = (f"A person vaguely remembers this item:\n{t.brief}\n"
                "Write requests that would lead them to it without quoting "
                "titles.")
    rules = (
        '\nReturn JSON with exactly these keys:\n'
        '"en_natural": one natural English sentence, any words allowed.\n'
        '"en_strict": one English sentence expressing the same need WITHOUT '
        'using any of these words or their inflected forms: '
        f'{", ".join(forbidden_words(t))}. No proper names. Express '
        'purpose, situation or content in different words.\n'
        '"uk": one natural Ukrainian sentence (Cyrillic letters only, no '
        'Latin letters) expressing the same need.')
    return head + body + rules + (f"\n\n{feedback}" if feedback else "")


def check(t: Target, q: dict) -> tuple[bool, bool, str]:
    en = str(q.get("en_strict", ""))
    uk = str(q.get("uk", ""))
    shared = et.shared_words(en, t.forbidden_text) if en else ["<empty>"]
    strict_ok = bool(en) and not shared
    uk_ok = et.is_ukrainian(uk)
    fb = []
    if not strict_ok:
        fb.append("en_strict shares forbidden word stems "
                  f"({', '.join(shared[:8])}); rewrite it avoiding them.")
    if not uk_ok:
        fb.append("uk must be a Ukrainian sentence in Cyrillic with no "
                  "Latin letters.")
    return strict_ok, uk_ok, " ".join(fb)


def generate(llm: BudgetedLLM, t: Target) -> dict:
    feedback, last = "", {}
    for attempt in range(1, MAX_RETRIES + 2):
        last = llm.complete_json(SYSTEM, build_prompt(t, feedback))
        strict_ok, uk_ok, feedback = check(t, last)
        if strict_ok and uk_ok:
            break
    split = "dev" if int(hashlib.sha256(
        f"{t.source}|{t.kind}|{t.target_id}".encode()).hexdigest(), 16) % 2 \
        == 0 else "test"
    return {"kind": t.kind, "source": t.source, "target_id": t.target_id,
            "split": split, "n_relevant": len(t.relevant_ids),
            "relevant_ids": list(t.relevant_ids),
            "queries": {k: str(last.get(k, "")) for k in
                        ("en_natural", "en_strict", "uk")},
            "flags": {"strict_ok": strict_ok, "uk_ok": uk_ok,
                      "attempts": attempt}}


def pricing_for(model: str) -> tuple[float, float]:
    data = requests.get("https://openrouter.ai/api/v1/models",
                        timeout=60).json()["data"]
    row = next(m for m in data if m["id"] == model)
    return (float(row["pricing"]["prompt"]) * 1e6,
            float(row["pricing"]["completion"]) * 1e6)


UNANSWERABLE = [
    ("tips for training a large language model on a single GPU",
     "поради щодо навчання великої мовної моделі на одному відеопроцесорі"),
    ("how to set up a hardware wallet for cryptocurrency",
     "як налаштувати апаратний гаманець для криптовалюти"),
    ("best electric scooter for commuting in the city",
     "найкращий електросамокат для поїздок містом на роботу"),
    ("how to grow followers as an instagram influencer",
     "як набрати підписників блогеру в інстаграмі"),
    ("choosing a streaming service for 4K movies",
     "як вибрати стрімінговий сервіс для фільмів у 4K"),
    ("how to fly a camera drone safely",
     "як безпечно керувати дроном з камерою"),
    ("configure two-factor authentication on my phone",
     "налаштувати двофакторну автентифікацію на моєму телефоні"),
    ("comparison of the newest smartphone processors",
     "порівняння найновіших процесорів для смартфонів"),
    ("tutorial on building a mobile app with React Native",
     "посібник зі створення мобільного застосунку на React Native"),
    ("how do electric car charging stations work",
     "як працюють зарядні станції для електромобілів"),
    ("how to start a podcast with a USB microphone",
     "як запустити подкаст з USB-мікрофоном"),
    ("best gaming laptop with ray tracing graphics",
     "найкращий ігровий ноутбук з трасуванням променів"),
    ("how does blockchain voting work",
     "як працює голосування на блокчейні"),
    ("how to edit short videos for social media",
     "як монтувати короткі відео для соцмереж"),
    ("smart home voice assistant privacy settings",
     "налаштування приватності голосового помічника для розумного дому"),
    ("setting up a Kubernetes cluster for microservices",
     "налаштування кластера Kubernetes для мікросервісів"),
    ("how to pay with a mobile wallet in a shop",
     "як розраховуватися мобільним гаманцем у магазині"),
    ("which VPN service is the most trustworthy",
     "якому VPN-сервісу можна найбільше довіряти"),
    ("tips for video meetings while working remotely",
     "поради для відеозустрічей під час віддаленої роботи"),
    ("how to 3D print a custom phone case",
     "як надрукувати на 3D-принтері чохол для телефону"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0.4)
    a = ap.parse_args()
    targets = all_targets()
    if a.limit:
        targets = targets[: a.limit]
    if a.dry_run:
        print(len(targets), "targets;", build_prompt(targets[0])[:900])
        return 0
    p_in, p_out = pricing_for(MODEL)
    out, log = (OUT, LOG) if not a.limit else (
        Path(os.environ["TEMP"]) / "smoke_queries.json",
        Path(os.environ["TEMP"]) / "smoke_queries_log.jsonl")
    llm = BudgetedLLM(MODEL, a.max_usd, log, p_in, p_out)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(lambda t: generate(llm, t), targets))
    rows.sort(key=lambda r: (r["source"], r["kind"], r["target_id"]))
    logged = [json.loads(x) for x in open(log, encoding="utf-8")]
    meta = {"model": MODEL, "prompt_version": PROMPT_VERSION,
            "pricing_per_million_usd": {"prompt": p_in, "completion": p_out},
            "temperature": 0.5, "max_retries": MAX_RETRIES,
            "elapsed_s": round(time.time() - t0, 1), **llm.summary(),
            "log_entries_all_runs": len(logged),
            "log_spend_all_runs_usd": round(
                sum(x.get("cost_usd", 0.0) for x in logged), 6)}
    doc = {"meta": meta, "unanswerable": [
        {"en": e, "uk": u} for e, u in UNANSWERABLE], "targets": rows}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    if out == OUT:
        (OUT.parent / (OUT.name + ".sha256")).write_text(sha + "\n")
    else:
        for r in rows:
            print(r["source"], r["kind"], r["target_id"][:40], r["flags"],
                  json.dumps(r["queries"], ensure_ascii=False))
    print(json.dumps(meta, indent=1), "\nsha256", sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
