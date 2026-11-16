# -*- coding: utf-8 -*-
"""Smoke-test the OpenRouter key and report which models actually answer.

Run:  python scripts/check_openrouter.py
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
sys.path.insert(0, os.path.join(ROOT, "src"))


def load_env():
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env()
KEY = os.environ.get("OPENROUTER_API_KEY", "")
PRIMARY = os.environ.get("OPENROUTER_MODEL", "")
POOL = [m.strip() for m in os.environ.get("OPENROUTER_FALLBACKS", "").split(",") if m.strip()]
MODELS = [PRIMARY] + [m for m in POOL if m != PRIMARY]

hdr = {
    "Authorization": "Bearer " + KEY,
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/PyStOspmt/Py-Buddy2",
    "X-Title": "Py-Buddy",
}

print("key present :", bool(KEY), "(%s...)" % KEY[:12])
print("primary     :", PRIMARY)
print("pool        :", len(POOL), "fallbacks")
print()

ok = []
for m in MODELS:
    body = json.dumps({
        "model": m,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16,
        "temperature": 0,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body, headers=hdr)
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.loads(r.read())
        txt = (d["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")
        dt = time.time() - t0
        print("  OK    %-52s %5.1fs  %s" % (m, dt, txt[:40]))
        ok.append(m)
    except urllib.error.HTTPError as e:
        detail = e.read()[:120].decode("utf-8", "replace")
        print("  FAIL  %-52s HTTP %s %s" % (m, e.code, detail))
    except Exception as e:
        print("  FAIL  %-52s %s" % (m, str(e)[:80]))
    time.sleep(1.0)

print()
print("working models:", len(ok), "/", len(MODELS))
if ok:
    print("recommended    :", ok[0])
