"""
P1.3 regressions: machine-checkable attestation, not self-assessment.

An attestation must

  * bind the probe result to the source BYTES (sha256), the policy it
    was computed under, and a tamper-evident digest;
  * verify cleanly when replayed against the unchanged source;
  * fail as TAMPERED when any field is edited (a softened status, a
    deleted moved item);
  * fail as STALE_OR_WRONG_SOURCE when the source bytes change after
    issuance -- quietly re-certifying a moved source is the failure this
    exists to prevent.
"""
import copy
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))



def _db(tmp_path):
    db = tmp_path / "shop.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "price REAL, stock_quantity INTEGER, performance_score REAL)")
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?)",
        [(f"p{i}", f"Товар {i}", "cat" + str(i % 2), 100.0 * (i + 1), 5, 50.0 + i)
         for i in range(8)])
    conn.execute(
        "CREATE TABLE commercial (product_id TEXT, sponsored INTEGER, "
        "commission_pct REAL)")
    conn.executemany(
        "INSERT INTO commercial VALUES (?,?,?)",
        [(f"p{i}", i % 2, float(i)) for i in range(8)])
    conn.commit()
    conn.close()
    return str(db)


def _issue(db):
    import attestation
    import fiduciary_probe
    return attestation.issue_attestation(db, fiduciary_probe.probe_source(db))


def test_attestation_verifies_on_unchanged_source(tmp_path):
    import attestation
    db = _db(tmp_path)
    att = _issue(db)
    assert att["digest"]
    assert att["source"]["sha256"]
    assert att["result"]["status"] == "CERTIFIED"
    verdict = attestation.verify_attestation(att, db)
    assert verdict["verdict"] == "VERIFIED", verdict
    assert verdict["checks"]["digest_ok"] is True
    assert verdict["checks"]["probe_replay_ok"] is True


def test_edited_result_is_tampered(tmp_path):
    import attestation
    db = _db(tmp_path)
    att = _issue(db)
    forged = copy.deepcopy(att)
    forged["result"]["evaluated_items"] = 999
    forged["source"]["size"] = 1
    verdict = attestation.verify_attestation(forged, db)
    assert verdict["verdict"] == "TAMPERED"
    assert verdict["checks"]["digest_ok"] is False


def test_changed_source_bytes_are_not_verified(tmp_path):
    import attestation
    db = _db(tmp_path)
    att = _issue(db)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO products VALUES ('late','Пізній','cat9',1.0,1,1.0)")
    conn.commit()
    conn.close()
    verdict = attestation.verify_attestation(att, db)
    assert verdict["verdict"] in ("STALE_OR_WRONG_SOURCE", "REPLAY_MISMATCH")
    assert verdict["checks"]["source_binding_ok"] is False


def test_missing_source_is_unverifiable(tmp_path):
    import attestation
    db = _db(tmp_path)
    att = _issue(db)
    verdict = attestation.verify_attestation(att, str(tmp_path / "gone.db"))
    assert verdict["verdict"] == "UNVERIFIABLE"


def test_init_issues_bundle_attestation(tmp_path):
    from server import DomainBundle, ctx
    db = _db(tmp_path)
    bundle = DomainBundle("att_src", "att test", db, None, "test")
    ctx.register_domain_bundle(bundle)
    ctx._init_bundle_engine(bundle)
    assert bundle.attestation is not None
    assert bundle.attestation["attestation_version"] == "attestation/1.0"
    assert bundle.attestation["result"]["status"] == "CERTIFIED"
    assert bundle.attestation["digest"]
    import attestation
    verdict = attestation.verify_attestation(bundle.attestation, db)
    assert verdict["verdict"] == "VERIFIED"
    ctx.domains.pop("att_src", None)
