"""Read API — the ``GET /metrics`` snapshot and recent-request log."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List

from . import state
from .stats import _fmt_uptime, _latency_stats


def _agent_distribution() -> Dict:
    agent_total = state.agent_counts["__total__"] or 1
    return {
        k: {"count": v, "pct": round(v / agent_total * 100, 1)}
        for k, v in state.agent_counts.items()
        if k != "__total__"
    }


def _latency_breakdown() -> tuple[Dict, Dict]:
    """``(per_endpoint, per_agent)`` latency stats."""
    endpoint_latency: Dict = {}
    agent_latency: Dict = {}
    for key in list(state.latency_window.keys()):
        if key.startswith("agent_"):
            agent_latency[key.replace("agent_", "")] = _latency_stats(key)
        elif key != "__all__":
            endpoint_latency[key] = _latency_stats(key)
    return endpoint_latency, agent_latency


def _token_summary() -> Dict:
    return {
        k: {
            "prompt": int(v.get("prompt", 0)),
            "completion": int(v.get("completion", 0)),
            "total": int(v.get("total", 0)),
            "cost_usd": round(v.get("cost_usd", 0.0), 6),
            "avg_ttft_ms": round(v["ttft_sum"] / v["ttft_count"], 2) if v.get("ttft_count", 0) > 0 else 0.0,
            "avg_tps": round(v["tps_sum"] / v["tps_count"], 2) if v.get("tps_count", 0) > 0 else 0.0,
        }
        for k, v in state.token_usage.items()
    }


def get_snapshot() -> Dict:
    """Full metrics snapshot returned by ``GET /metrics``."""
    with state.lock:
        uptime_s = time.time() - state.start_time
        counters = state.counters
        total_req = counters["requests_total"] or 1  # avoid /0
        endpoint_latency, agent_latency = _latency_breakdown()

        return {
            "uptime": {
                "seconds": round(uptime_s),
                "human":   _fmt_uptime(uptime_s),
                "started": datetime.fromtimestamp(state.start_time, tz=timezone.utc).isoformat(),
            },
            "requests": {
                "total":         counters["requests_total"],
                "success":       counters["requests_success"],
                "errors_4xx":    counters["errors_4xx"],
                "errors_5xx":    counters["errors_5xx"],
                "error_rate":    round(counters["errors_total"] / total_req * 100, 2),
                "out_of_domain": counters["out_of_domain_total"],
            },
            "latency": _latency_stats("__all__"),
            "latency_by_endpoint": endpoint_latency,
            "agents": {
                "total_calls":  state.agent_counts["__total__"],
                "distribution": _agent_distribution(),
                "latency":      agent_latency,
            },
            "tokens": _token_summary(),
            "sessions": {
                "active": len(state.active_sessions),
                "total":  counters["sessions_total"],
            },
            "errors": {
                "total":  counters["errors_total"],
                "recent": list(state.error_log)[-10:],
            },
        }


def get_recent_requests(limit: int = 50) -> List[Dict]:
    """The most recent ``limit`` requests (oldest first)."""
    with state.lock:
        return list(state.request_log)[-limit:]
