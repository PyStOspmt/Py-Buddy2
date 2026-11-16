"""Source upload route and tabular-to-SQLite conversion."""

from fastapi import HTTPException, Request, UploadFile, Form
from typing import List
import json
import os
import re
import shutil
import sqlite3
from server_config import env_setting
from server_context import DomainBundle, ctx
from server_security import _require_admin
from fastapi import APIRouter

router = APIRouter()


@router.post("/api/domains/upload")
def upload_domain_source(
    request: Request,
    file: UploadFile,
    domain_id: str = Form(...),
    name: str | None = Form(None),
    llm_assist: bool = Form(False),
):
    """
    Upload an arbitrary SQLite file as a new domain source -- the
    research/demo path ('drop any database at it'). Admin-gated like
    /api/register_domain. The file is validated as real SQLite before it
    is persisted; it lands inside the configured source root under the
    caller-chosen (immutable) domain id, and the engine build runs as an
    async init job so the response stays fast on large sources.
    """
    _require_admin(request)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", domain_id or ""):
        raise HTTPException(
            status_code=400,
            detail="domain_id must be 1-64 characters of [A-Za-z0-9_-].")
    if domain_id in ctx.domains:
        raise HTTPException(
            status_code=409,
            detail=f"Source id '{domain_id}' already exists; source ids are immutable.")

    fname = (file.filename or "").lower()
    ext = os.path.splitext(fname)[1]
    sqlite_exts = (".db", ".sqlite", ".sqlite3")
    tabular_exts = (".csv", ".tsv", ".json", ".jsonl")
    doc_exts = (".md", ".markdown", ".pdf")
    if ext not in sqlite_exts + tabular_exts + doc_exts:
        raise HTTPException(
            status_code=400,
            detail=("Accepted: SQLite (.db/.sqlite/.sqlite3), tabular data "
                    "(.csv/.tsv/.json/.jsonl) or documents (.md/.pdf) -- "
                    "the latter are normalized to SQLite with provenance."))

    root = env_setting("PYBUDDY_SOURCE_ROOT", "AINSA_SOURCE_ROOT") or os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
    os.makedirs(root, exist_ok=True)
    dest = os.path.realpath(os.path.join(root, f"{domain_id}.db"))
    root_real = os.path.realpath(root)
    if not dest.startswith(root_real + os.sep):
        raise HTTPException(status_code=403, detail="Upload must land inside the source root.")

    # Stream to a temp file, verify, then atomically publish -- a
    # half-uploaded or invalid blob never registers.
    tmp = dest + ".part"
    converted = False
    try:
        with open(tmp, "wb") as out:
            shutil.copyfileobj(file.file, out, 1024 * 1024)
        written = os.path.getsize(tmp)
        if ext in sqlite_exts:
            with open(tmp, "rb") as f:
                magic = f.read(16)
            if magic != b"SQLite format 3\x00":
                os.remove(tmp)
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is not a SQLite database (bad header).")
            os.replace(tmp, dest)
        elif ext in doc_exts:
            if ext == ".pdf":
                from pdf_normalizer import write_pdf_sqlite
                try:
                    nrows = write_pdf_sqlite(tmp, dest, fname)
                except Exception as exc:
                    os.remove(tmp)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unreadable PDF: {type(exc).__name__}")
            else:
                with open(tmp, encoding="utf-8-sig",
                          errors="replace") as f:
                    md_text = f.read()
                from markdown_normalizer import write_markdown_sqlite
                nrows = write_markdown_sqlite(md_text, dest, fname)
            os.remove(tmp)
            converted = True
            if nrows == 0:
                os.remove(dest)
                raise HTTPException(
                    status_code=400,
                    detail="Document produced zero extractable pages/"
                           "sections -- nothing to register.")
        else:
            nrows = _tabular_to_sqlite(tmp, ext, dest, fname)
            os.remove(tmp)
            converted = True
            if nrows == 0:
                os.remove(dest)
                raise HTTPException(
                    status_code=400,
                    detail="Tabular file produced zero rows -- nothing to register.")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- cleanup then typed 500, never a silent partial write
        for p in (tmp,):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        raise HTTPException(
            status_code=500, detail=f"Upload failed: {type(exc).__name__}: {exc}")

    bundle = DomainBundle(
        domain_id=domain_id,
        name=name or domain_id,
        db_path=dest,
        hypergraph=None,
        description=(f"Uploaded {'converted ' if converted else ''}source "
                     f"{fname} ({written} bytes)"))
    bundle.registered_via = "runtime"
    bundle.llm_assist = bool(llm_assist)
    ctx.register_domain_bundle(bundle)
    ctx.persist_runtime_registration(bundle)
    job = ctx.start_init_job(domain_id)
    return {
        "status": "accepted",
        "domain_id": domain_id,
        "db_path": dest,
        "bytes": written,
        "converted": converted,
        "init_job": job,
        "message": f"Uploaded '{fname}' ({written} B) as '{domain_id}'; init job {job['job_id']} queued."
    }


def _tabular_to_sqlite(tmp_path: str, ext: str, dest: str,
                       origin_name: str) -> int:
    """Convert a CSV/TSV/JSON(L) upload into a SQLite source with honest
    provenance: a `records` table (column types inferred from a bounded
    sample) plus an `ingest_meta` log of the conversion. Returns row count.
    """
    import csv as _csv
    from source_projection import quote_ident
    headers: List[str] = []
    if ext in (".csv", ".tsv"):
        dialect = "excel-tab" if ext == ".tsv" else "excel"
        with open(tmp_path, newline="", encoding="utf-8-sig",
                  errors="replace") as f:
            reader = _csv.reader(f, dialect=dialect)
            all_rows = list(reader)
        if not all_rows:
            return 0
        headers = [h.strip() or f"col_{i}" for i, h in enumerate(all_rows[0])]
        data_rows = [r for r in all_rows[1:] if any(v.strip() for v in r)]
    else:
        with open(tmp_path, encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [json.loads(l) for l in text.splitlines()
                      if l.strip()]
        # Deep normalization: nested dicts flatten, arrays of objects
        # become FK-linked child tables, every row keeps its JSON path
        # in _src. Flat JSON still yields the single `records` table.
        from json_normalizer import normalize_json, write_json_sqlite
        tables = normalize_json(
            parsed if isinstance(parsed, (list, dict)) else [parsed])
        return write_json_sqlite(tables, dest, origin_name,
                                 ext.lstrip("."))
    if not data_rows:
        return 0

    # Type inference over a bounded sample: a column is REAL/INTEGER only
    # when every seen value parses; otherwise honest TEXT.
    probe = data_rows[:1000]
    kinds = []
    for i in range(len(headers)):
        vals = [r[i] for r in probe if i < len(r)
                and r[i] is not None and str(r[i]).strip() != ""]
        def _num(v):
            try:
                float(v)
                return True
            except (TypeError, ValueError):
                return False
        kinds.append("REAL" if vals and all(_num(v) for v in vals) else "TEXT")

    conn = sqlite3.connect(dest)
    try:
        cols = ", ".join(f"{quote_ident(h)} {k}" for h, k in zip(headers, kinds))
        conn.execute(f"CREATE TABLE records ({cols})")
        conn.executemany(
            f"INSERT INTO records VALUES ({','.join('?' * len(headers))})",
            [[r[i] if i < len(r) else None for i in range(len(headers))]
             for r in data_rows])
        conn.execute("CREATE TABLE ingest_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany("INSERT INTO ingest_meta VALUES (?,?)", [
            ("origin_file", origin_name),
            ("origin_format", ext.lstrip(".")),
            ("converted", "tabular->sqlite"),
            ("rows", str(len(data_rows))),
        ])
        conn.commit()
    finally:
        conn.close()
    return len(data_rows)
