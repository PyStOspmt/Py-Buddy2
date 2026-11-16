#!/usr/bin/env python
"""
Pre-commit secret scanner for Py-Buddy.

Blocks any commit that would introduce a hardcoded API key, token or
private key into the repository. Wired into .git/hooks/pre-commit.

Usage:
    python scripts/check_secrets.py            # scan staged files
    python scripts/check_secrets.py <file>...  # scan explicit files
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# (label, regex). Patterns are deliberately generic-but-specific: they match
# the shape of a credential, not a specific vendor's example value.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenRouter key", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{32,}")),
    ("OpenAI/Anthropic key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{32,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("Telegram bot token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("Bearer with JWT", re.compile(r"\bBearer\s+eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}")),
]

# Lines that legitimately look like a key but are examples/placeholders.
ALLOW_MARKERS = (
    "your_",
    "your-",
    "example",
    "placeholder",
    "xxxx",
    "<key>",
    "<api",
    "${",
    "{{",
    "os.environ",
    "getenv",
    "dotenv",
    "REDIS_URL",
)


def scan_text(text: str, label: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        if any(m in lowered for m in ALLOW_MARKERS):
            continue
        for name, pattern in PATTERNS:
            if pattern.search(line):
                findings.append(f"  {label}:{lineno}  [{name}]")
    return findings


def staged_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception as exc:  # pragma: no cover
        print(f"check_secrets: cannot list staged files: {exc}", file=sys.stderr)
        return []
    return [f for f in out.splitlines() if f.strip()]


def main(argv: list[str]) -> int:
    targets = argv[1:] or staged_files()
    if not targets:
        print("check_secrets: nothing to scan.")
        return 0

    findings: list[str] = []
    for rel in targets:
        path = Path(rel)
        if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        findings.extend(scan_text(text, rel))

    if findings:
        print("\n[check_secrets] COMMIT BLOCKED — possible secret(s) detected:\n", file=sys.stderr)
        print("\n".join(findings), file=sys.stderr)
        print(
            "\nIf these are false positives, rewrite the line or add a placeholder marker.\n"
            "Never commit real credentials. If a key leaked, ROTATE it at the provider.\n",
            file=sys.stderr,
        )
        return 1

    print(f"check_secrets: OK ({len(targets)} file(s) scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
