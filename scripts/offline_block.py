"""Pytest plugin: block all outbound network for offline test runs.

Patches socket.socket.connect and socket.create_connection so tests fail
fast if any offline check attempts external network access:
    python -m pytest tests ... -p offline_block
with PYTHONPATH including scripts/.
"""
import socket

from unittest.mock import patch


def pytest_configure(config):
    config._offline_patches = [
        patch.object(socket.socket, "connect",
                     side_effect=RuntimeError("Outbound network disabled for offline test run")),
        patch("socket.create_connection",
              side_effect=RuntimeError("Outbound network disabled for offline test run")),
    ]
    for p in config._offline_patches:
        p.start()


def pytest_unconfigure(config):
    for p in getattr(config, "_offline_patches", []):
        p.stop()