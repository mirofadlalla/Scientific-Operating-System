"""Behaviour of the modular ``app.monitoring`` package."""

from __future__ import annotations

import pytest

from app import monitoring
from app.monitoring import state
from app.monitoring.pricing import MODEL_PRICING, estimate_cost
from app.monitoring.stats import _fmt_uptime, _percentile


@pytest.fixture(autouse=True)
def clean_state():
    snapshots = {n: getattr(state, n).copy() for n in
                 ("counters", "agent_counts", "active_sessions")}
    window = {k: list(v) for k, v in state.latency_window.items()}
    usage = {k: dict(v) for k, v in state.token_usage.items()}
    for n in ("counters", "agent_counts", "active_sessions", "latency_window", "token_usage"):
        getattr(state, n).clear()
    state.error_log.clear()
    state.request_log.clear()
    yield
    for n, saved in snapshots.items():
        getattr(state, n).clear()
        getattr(state, n).update(saved)
    state.latency_window.clear()
    for k, v in window.items():
        state.latency_window[k].extend(v)
    state.token_usage.clear()
    state.token_usage.update(usage)


def test_request_classification_and_latency():
    for code in (200, 404, 503):
        monitoring.record_request("/x", "get", code, 10.0)
    snap = monitoring.get_snapshot()
    assert snap["requests"]["total"] == 3
    assert (snap["requests"]["success"], snap["requests"]["errors_4xx"], snap["requests"]["errors_5xx"]) == (1, 1, 1)
    assert snap["latency_by_endpoint"]["/x"]["count"] == 3
    assert monitoring.get_recent_requests(2)[-1]["status"] == 503


def test_token_cost_and_aggregate():
    monitoring.record_tokens("openai/gpt-oss-120b", 1_000_000, 1_000_000, ttft_ms=100, tps=50)
    monitoring.record_tokens("mystery", 1_000_000, 0)
    tokens = monitoring.get_snapshot()["tokens"]
    assert tokens["openai/gpt-oss-120b"]["cost_usd"] == pytest.approx(0.75)
    assert tokens["openai/gpt-oss-120b"]["avg_ttft_ms"] == 100 and tokens["mystery"]["avg_tps"] == 0.0
    assert tokens["mystery"]["cost_usd"] == pytest.approx(0.50)  # default rate
    assert tokens["__all__"]["total"] == 3_000_000
    assert estimate_cost("openai/gpt-oss-20b", 1_000_000, 0) == MODEL_PRICING["openai/gpt-oss-20b"]["prompt"]


def test_agents_sessions_errors():
    monitoring.record_agent_call("A", "i", 5.0)
    monitoring.record_agent_call("B", "i", 5.0, success=False)
    monitoring.record_out_of_domain("why")
    monitoring.session_start("s1")
    monitoring.record_error("/e", "x" * 500)
    snap = monitoring.get_snapshot()
    assert snap["agents"]["total_calls"] == 2 and snap["agents"]["distribution"]["A"]["pct"] == 50.0
    assert snap["agents"]["latency"]["B"]["count"] == 1 and state.counters["agent_B_errors"] == 1
    assert snap["requests"]["out_of_domain"] == 1 and snap["sessions"] == {"active": 1, "total": 1}
    assert len(snap["errors"]["recent"][0]["error"]) == 300
    monitoring.session_end("s1")
    assert monitoring.get_snapshot()["sessions"]["active"] == 0


def test_helpers_and_legacy_aliases():
    assert _percentile([], 50) == 0.0 and _percentile([3, 1, 2], 95) == 3
    assert _fmt_uptime(90061) == "1d 1h 1m 1s"
    assert monitoring._counters is state.counters and monitoring._lock is state.lock
