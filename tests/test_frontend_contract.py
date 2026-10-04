"""Frontend ↔ backend contract.

The React app (frontend/src) calls the API by path and reads specific keys out of
the JSON it gets back. These tests fail when either side drifts, so a backend
refactor can't silently break the UI (as happened with the /metrics shape).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from starlette.routing import Match

from app import monitoring
from app.main import app

FRONTEND_SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"
API_PREFIX = "/api/v1"

# `${API_BASE}/rag/ingest/status/${jobId}` / `${API_BASE}/metrics/requests?limit=100`
_API_CALL = re.compile(r"\$\{API_BASE\}(/[A-Za-z0-9_\-/]*(?:\$\{[^}]+\}[A-Za-z0-9_\-/]*)*)")
_WS_CALL = re.compile(r"new WebSocket\(`\$\{WS_URL\}")


def _frontend_api_paths() -> set[str]:
    paths: set[str] = set()
    for f in FRONTEND_SRC.rglob("*.jsx"):
        for m in _API_CALL.finditer(f.read_text(encoding="utf-8")):
            paths.add(re.sub(r"\$\{[^}]+\}", "x", m.group(1)).rstrip("/"))
    return paths


def _has_route(path: str, *, websocket: bool = False) -> bool:
    scope_type = "websocket" if websocket else "http"
    for route in app.routes:
        scope = {"type": scope_type, "path": path, "method": "GET", "root_path": ""}
        match, _ = route.matches(scope)
        if match is not Match.NONE:
            return True
        # method mismatch (e.g. POST-only route) still proves the path exists
        if not websocket and route.matches({**scope, "method": "POST"})[0] is not Match.NONE:
            return True
    return False


def test_frontend_actually_calls_the_api():
    # Guards the regex above: if it stops matching, the other tests would pass vacuously.
    assert {"/orchestrate", "/auth/login", "/metrics", "/rag/ingest"} <= _frontend_api_paths()


@pytest.mark.parametrize("path", sorted(_frontend_api_paths()))
def test_every_frontend_api_call_has_a_backend_route(path):
    assert _has_route(API_PREFIX + path), f"frontend calls {API_PREFIX}{path} but the backend has no such route"


def test_voice_websocket_route_exists():
    uses_ws = any(_WS_CALL.search(f.read_text(encoding="utf-8")) for f in FRONTEND_SRC.rglob("*.jsx"))
    assert uses_ws, "frontend no longer opens the voice WebSocket"
    assert _has_route(f"{API_PREFIX}/ws/voice", websocket=True)


def test_health_route_used_for_keepalive():
    assert _has_route("/health")


def test_metrics_snapshot_has_keys_monitor_page_reads():
    monitoring.record_request("/api/v1/orchestrate", "POST", 200, 12.5)
    snap = monitoring.get_snapshot()

    assert isinstance(snap["uptime"]["seconds"], (int, float))
    for key in ("total", "error_rate", "out_of_domain"):
        assert key in snap["requests"], f"requests.{key}"
    assert "total" in snap["errors"]
    for key in ("avg", "count"):
        assert key in snap["latency"], f"latency.{key}"
    assert "distribution" in snap["agents"]
    assert isinstance(snap["tokens"], dict)


def test_recent_requests_entries_have_fields_monitor_page_reads():
    monitoring.record_request("/api/v1/metrics", "GET", 200, 3.0)
    entry = monitoring.get_recent_requests(1)[-1]
    for key in ("ts", "endpoint", "method", "status", "latency_ms"):
        assert key in entry, key
