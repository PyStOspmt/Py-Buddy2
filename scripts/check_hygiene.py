"""Repository hygiene checks. Exit 1 on any violation.

Checks:
  1. Root whitelist - no unexpected entries in the repository root.
  2. .gitignore has no duplicate non-comment lines.
  3. Product core (src/core, src/gateways, src/runners) must not import
     experiment modules (src/experiments).
  4. All relative markdown links in README.md and docs/ resolve to existing files.

Run manually or from CI:
    python scripts/check_hygiene.py
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROOT_WHITELIST = {
    ".env", ".env.example", ".gitignore", ".dockerignore", ".github",
    ".pytest_cache", ".ruff_cache", "__pycache__",
    "Dockerfile", "LICENSE", "README.md",
    "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "data", "docs", "scripts", "src", "tests", "web", "bench",
    ".git",
}

_PRODUCT_DIRS = ("core", "gateways", "runners")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def fail(msg: str) -> None:
    print("HYGIENE FAIL:", msg)


def check_root() -> bool:
    ok = True
    for entry in ROOT.iterdir():
        if entry.name not in ROOT_WHITELIST:
            ok = False
            fail(f"unexpected root entry: {entry.name}")
    return ok


def check_gitignore() -> bool:
    ok = True
    seen: set[str] = set()
    for raw in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            ok = False
            fail(f".gitignore duplicate entry: {line}")
        seen.add(line)
    return ok


def check_boundary() -> bool:
    ok = True
    src_root = ROOT / "src"
    for pkg in _PRODUCT_DIRS:
        pkg_dir = src_root / pkg
        if not pkg_dir.exists():
            continue
        for py_file in sorted(pkg_dir.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                mod_name = None
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod_name = node.module
                elif isinstance(node, ast.Import):
                    mod_name = ",".join(alias.name for alias in node.names)
                if mod_name and "experiments" in mod_name.split("."):
                    ok = False
                    rel = py_file.relative_to(ROOT)
                    fail(f"product module imports experiment: {rel} -> {mod_name}")
    return ok


def check_markdown_links() -> bool:
    ok = True
    md_files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for md in md_files:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            for raw_target in _MD_LINK_RE.findall(line):
                target = raw_target.strip()
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_part = target.split("#", 1)[0]
                if not path_part:
                    continue
                resolved = (md.parent / path_part).resolve()
                if not resolved.exists():
                    ok = False
                    rel = md.relative_to(ROOT)
                    fail(f"broken markdown link in {rel}:{line_no} -> {target}")
    return ok


def main() -> int:
    ok = True
    ok &= check_root()
    ok &= check_gitignore()
    ok &= check_boundary()
    ok &= check_markdown_links()
    if ok:
        print("HYGIENE: OK")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())