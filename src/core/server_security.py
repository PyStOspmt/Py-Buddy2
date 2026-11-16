"""Administrative access gate for the HTTP server (fail-closed)."""

from fastapi import HTTPException, Request
import hmac
from server_config import env_setting


# ---------------------------------------------------------------------------
# F29/F30: administrative access. Fail-closed by design.
# ---------------------------------------------------------------------------
def _is_admin_request(request: Request) -> bool:
    """True when the caller presented the matching X-Admin-Token header,
    or -- with PYBUDDY_LOCAL_ADMIN=1 set at server start -- when the call
    originates from loopback. The flag is an explicit operator opt-in
    for single-user local runs; absent it, the gate stays fail-closed."""
    token = env_setting("PYBUDDY_ADMIN_TOKEN", "AINSA_ADMIN_TOKEN")
    supplied = request.headers.get("X-Admin-Token") or ""
    if token and hmac.compare_digest(supplied, token):
        return True
    if env_setting("PYBUDDY_LOCAL_ADMIN", "AINSA_LOCAL_ADMIN") == "1":
        host = getattr(request.client, "host", "") if request.client \
            else ""
        return host in ("127.0.0.1", "::1", "localhost")
    return False


def _require_admin(request: Request) -> None:
    """
    F29/F30: administrative operations (registering sources, switching the
    global domain or model) are refused unless the deployment configured an
    admin token and the caller presents it. Fail-closed: no token configured
    means these endpoints are disabled, not open.
    """
    if not _is_admin_request(request):
        raise HTTPException(
            status_code=403,
            detail=("Administration requires PYBUDDY_ADMIN_TOKEN to be configured "
                    "on the server and a matching X-Admin-Token header."))
