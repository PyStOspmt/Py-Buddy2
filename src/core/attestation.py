"""
Py-Buddy: machine-checkable attestation (external-review P1.3).

The critique was precise: a "certificate" the system issues about itself
is a self-assessment unless a third party can check it. This module
turns the fiduciary probe result into an attestation artefact that an
independent verifier can re-derive offline:

    issue_attestation(bundle)      -> dict, bound to the source bytes
    verify_attestation(att, path)  -> per-check verdict

What it binds
-------------
* SHA-256 of the source FILE -- a content identity, not the (mtime,
  size) stamp used for cache pinning;
* the probe policy (version, bounds, concept-spelling digest), so a
  verifier replays the SAME audit, not a stronger or weaker one;
* the probe result (status, FID, moved items, evaluated set size);
* a digest over the canonical form of all of the above.

What it is NOT
--------------
* It is not a signature. There is no key and no chain of trust; the
  digest detects tampering with the ARTEFACT and binds it to source
  bytes -- it does not prove who issued it. That is the honest scope:
  "machine-checkable attestation", not "cryptographic certificate".
* Replay verification re-runs the deterministic probe (same
  permutation, same evaluation set, same -price ordering). A match
  confirms the claim was computed, not asserted; a source that changed
  since issuance is STALE, not VERIFIED -- quietly re-certifying a
  moved source is exactly what this exists to prevent.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import fiduciary_probe
from fiduciary_audit import COMMERCIAL_CONCEPTS
from source_lifecycle import source_version

ATTESTATION_VERSION = "attestation/1.0"


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _policy() -> Dict[str, Any]:
    spellings = sorted(
        s for ss in COMMERCIAL_CONCEPTS.values() for s in ss)
    return {
        "probe": "fiduciary_probe/1.0",
        "top_k": fiduciary_probe.PROBE_TOP_K,
        "max_categories": fiduciary_probe.PROBE_MAX_CATEGORIES,
        "max_items": fiduciary_probe.PROBE_MAX_ITEMS,
        "concept_spellings_digest": _digest(spellings),
    }


def issue_attestation(
    db_path: str,
    probe_result: Dict[str, Any],
    *,
    issued_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bind a probe result to the source bytes it was computed on.

    The digest covers everything except itself; verification recomputes
    it, so any edited field -- a softened status, a deleted moved item --
    is detected.
    """
    body: Dict[str, Any] = {
        "attestation_version": ATTESTATION_VERSION,
        "source": {
            "name": os.path.basename(db_path),
            "sha256": _sha256_file(db_path),
            "size": os.path.getsize(db_path),
            "version_stamp": list(source_version(db_path) or ()),
        },
        "policy": _policy(),
        "result": {
            "status": probe_result.get("status"),
            "fid": probe_result.get("fid"),
            "evaluable": probe_result.get("evaluable"),
            "evaluated_items": probe_result.get("evaluated_items"),
            "channels": probe_result.get("channels", []),
            "moved_items": probe_result.get("moved_items", []),
            "violations": probe_result.get("violations", []),
            "scope": probe_result.get("scope"),
        },
        "issued_at": issued_at or datetime.now(timezone.utc).isoformat(),
    }
    body["digest"] = _digest(body)
    return body


def verify_attestation(
    att: Dict[str, Any],
    db_path: str,
    *,
    replay_probe: bool = True,
) -> Dict[str, Any]:
    """
    Independently re-check an attestation against the source file.

    Per-check verdicts are returned, never a bare boolean -- "verified"
    and "the file changed since issuance" must not collapse into one
    answer.
    """
    checks: Dict[str, Any] = {}

    stored_digest = att.get("digest")
    recomputed = {k: v for k, v in att.items() if k != "digest"}
    checks["digest_ok"] = stored_digest == _digest(recomputed)

    if not db_path or not os.path.exists(db_path):
        checks["source_present"] = False
        checks["source_binding_ok"] = False
        checks["version_ok"] = False
        verdict = "UNVERIFIABLE"
        checks["verdict_reason"] = "source file missing"
        return {"verdict": verdict, "checks": checks}

    checks["source_present"] = True
    source = att.get("source", {})
    checks["source_binding_ok"] = (
        source.get("sha256") == _sha256_file(db_path))
    stamp = source_version(db_path)
    checks["version_ok"] = (
        list(source.get("version_stamp") or ()) == list(stamp or ()))

    if replay_probe:
        replay = fiduciary_probe.probe_source(db_path)
        expected = att.get("result", {})
        same = (
            replay.get("status") == expected.get("status")
            and replay.get("fid") == expected.get("fid")
            and replay.get("moved_items") == expected.get("moved_items")
        )
        checks["probe_replay_ok"] = same
        checks["probe_replay_status"] = replay.get("status")
    else:
        checks["probe_replay_ok"] = None

    if not checks["digest_ok"]:
        verdict = "TAMPERED"
    elif not checks["source_binding_ok"]:
        # The bytes the attestation was issued for are not the bytes on
        # disk. Whether the source was updated or a different file was
        # substituted, the claim no longer describes this source.
        verdict = "STALE_OR_WRONG_SOURCE"
    elif checks["probe_replay_ok"] is False:
        # Same bytes, same deterministic policy, different result: the
        # audit is not reproducible, which is itself a defect to report.
        verdict = "REPLAY_MISMATCH"
    else:
        verdict = "VERIFIED"

    return {"verdict": verdict, "checks": checks}
