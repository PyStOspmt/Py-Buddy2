"""Process + HTTP helpers for the end-to-end cycle runner (docs/18).

The cycle talks to the REAL FastAPI application over HTTP (no TestClient,
no in-process shortcuts): a uvicorn subprocess with an isolated working
root, a random admin token and registry/session files outside the
project's data directory.
"""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parents[3]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CycleServer:
    def __init__(self, work: Path):
        self.work = work
        self.port = free_port()
        self.token = secrets.token_hex(16)
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.log_path = work / "server.log"

    def env(self) -> dict:
        e = dict(os.environ)
        e.update({"PYBUDDY_ADMIN_TOKEN": self.token,
                  "PYBUDDY_SOURCE_ROOT": str(self.work / "sources"),
                  "PYBUDDY_RUNTIME_REGISTRY": str(self.work / "runtime.json"),
                  "PYBUDDY_SESSIONS_DB": str(self.work / "sessions.db"),
                  "PYBUDDY_ALLOWED_ORIGINS": "", "PYTHONUTF8": "1",
                  "HF_HUB_OFFLINE": "1"})
        e.pop("PYBUDDY_LOCAL_ADMIN", None)
        e.pop("AINSA_LOCAL_ADMIN", None)
        return e

    def start(self, timeout: float = 240.0) -> float:
        t0 = time.time()
        log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app", "--app-dir",
             str(ROOT / "src" / "core"), "--host", "127.0.0.1", "--port",
             str(self.port), "--log-level", "warning"],
            cwd=str(ROOT), env=self.env(), stdout=log, stderr=log)
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited; see " + str(self.log_path))
            try:
                if requests.get(self.base + "/api/status", timeout=3
                                ).status_code == 200:
                    return time.time() - t0
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise RuntimeError("server did not become ready")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def api(self, method: str, path: str, *, json=None, params=None,
            admin: bool = False, timeout: float = 120.0
            ) -> tuple[int, dict]:
        headers = {"X-Admin-Token": self.token} if admin else {}
        r = requests.request(method, self.base + path, json=json,
                             params=params, headers=headers,
                             timeout=timeout)
        try:
            body = r.json()
        except ValueError:
            body = {"_raw": r.text[:500]}
        return r.status_code, body

    def memory_mb(self) -> dict:
        if not self.proc:
            return {}
        p = psutil.Process(self.proc.pid)
        mi = p.memory_full_info()
        return {"uss_mb": round(mi.uss / 2**20, 1),
                "rss_mb": round(mi.rss / 2**20, 1)}


def wait_until(fn, timeout: float, interval: float = 2.0, on_tick=None):
    """Poll `fn()` until it returns a truthy value; returns (value,
    elapsed). Raises TimeoutError with the last value."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        last = fn()
        if on_tick:
            on_tick(last, time.time() - t0)
        if last:
            return last, time.time() - t0
        time.sleep(interval)
    raise TimeoutError(f"condition not met in {timeout}s; last={last!r}")
