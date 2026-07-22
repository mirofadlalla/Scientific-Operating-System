"""
monitoring.py — In-process observability for AI-lixir Scientific OS
====================================================================
Designed for HF Spaces single-container deployments where Prometheus/Grafana
are not available. Stores all metrics in-memory with a rolling JSON snapshot.

Tracks:
  • Request counts, latency (p50/p95/p99), error rates
  • Token usage per model call (prompt + completion)
  • Agent routing distribution (chemical / medical / rag / app)
  • Out-of-domain rejection counts
  • Active sessions, uptime
  • Per-endpoint breakdown
"""

import time
import threading
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Internal state — all guarded by a single RLock for thread safety
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.RLock()
_start_time = time.time()

# Rolling window for latency (last 1000 requests per endpoint)
_latency_window: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

# Counters
_counters: Dict[str, int] = defaultdict(int)

# Token and performance tracking per model
_token_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: {
    "prompt": 0, "completion": 0, "total": 0,
    "cost_usd": 0.0, "ttft_sum": 0.0, "ttft_count": 0,
    "tps_sum": 0.0, "tps_count": 0
})

# Model pricing per 1M tokens (Groq rates / default estimates)
MODEL_PRICING = {
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
    "whisper-large-v3-turbo":  {"prompt": 0.04, "completion": 0.04},
    "canopylabs/orpheus-arabic-saudi": {"prompt": 0.50, "completion": 0.50},
    "canopylabs/orpheus-v1-english": {"prompt": 0.50, "completion": 0.50},
}


def record_tokens(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: Optional[float] = None,
    tps: Optional[float] = None,
):
    """Record LLM token usage, latency performance (TTFT, TPS), and calculate estimated cost."""
    with _lock:
        rates = MODEL_PRICING.get(model, {"prompt": 0.50, "completion": 0.50})
        cost = ((prompt_tokens / 1_000_000) * rates["prompt"]) + ((completion_tokens / 1_000_000) * rates["completion"])

        _token_usage[model]["prompt"]     += prompt_tokens
        _token_usage[model]["completion"] += completion_tokens
        _token_usage[model]["total"]      += prompt_tokens + completion_tokens
        _token_usage[model]["cost_usd"]   += round(cost, 6)

        if ttft_ms is not None:
            _token_usage[model]["ttft_sum"]   += ttft_ms
            _token_usage[model]["ttft_count"] += 1
        if tps is not None:
            _token_usage[model]["tps_sum"]   += tps
            _token_usage[model]["tps_count"] += 1

        _token_usage["__all__"]["prompt"]     += prompt_tokens
        _token_usage["__all__"]["completion"] += completion_tokens
        _token_usage["__all__"]["total"]      += prompt_tokens + completion_tokens
        _token_usage["__all__"]["cost_usd"]   += round(cost, 6)


def record_error(endpoint: str, error: str, session_id: Optional[str] = None):
    """Record an application error."""
    with _lock:
        _counters["errors_total"] += 1
        _error_log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "error": str(error)[:300],
            "session_id": session_id,
        })


def session_start(session_id: str):
    with _lock:
        _active_sessions[session_id] = time.time()
        _counters["sessions_total"] += 1


def session_end(session_id: str):
    with _lock:
        _active_sessions.pop(session_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Stats computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return round(sorted_data[min(idx, len(sorted_data) - 1)], 2)


def _latency_stats(key: str) -> Dict:
    data = list(_latency_window.get(key, []))
    if not data:
        return {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "min": 0, "max": 0, "count": 0}
    return {
        "p50":   _percentile(data, 50),
        "p95":   _percentile(data, 95),
        "p99":   _percentile(data, 99),
        "avg":   round(statistics.mean(data), 2),
        "min":   round(min(data), 2),
        "max":   round(max(data), 2),
        "count": len(data),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main snapshot — returned by GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_snapshot() -> Dict:
    with _lock:
        uptime_s = time.time() - _start_time
        total_req = _counters["requests_total"] or 1  # avoid /0

        # Agent distribution percentages
        agent_total = _agent_counts["__total__"] or 1
        agent_dist = {
            k: {
                "count": v,
                "pct": round(v / agent_total * 100, 1)
            }
            for k, v in _agent_counts.items()
            if k != "__total__"
        }

        # Endpoint latency breakdown
        endpoint_latency = {}
        for key in list(_latency_window.keys()):
            if not key.startswith("agent_") and key != "__all__":
                endpoint_latency[key] = _latency_stats(key)

        # Agent latency breakdown
        agent_latency = {}
        for key in list(_latency_window.keys()):
            if key.startswith("agent_"):
                agent_latency[key.replace("agent_", "")] = _latency_stats(key)

        return {
            "uptime": {
                "seconds": round(uptime_s),
                "human":   _fmt_uptime(uptime_s),
                "started": datetime.fromtimestamp(_start_time, tz=timezone.utc).isoformat(),
            },
            "requests": {
                "total":        _counters["requests_total"],
                "success":      _counters["requests_success"],
                "errors_4xx":   _counters["errors_4xx"],
                "errors_5xx":   _counters["errors_5xx"],
                "error_rate":   round(_counters["errors_total"] / total_req * 100, 2),
                "out_of_domain":_counters["out_of_domain_total"],
            },
            "latency": _latency_stats("__all__"),
            "latency_by_endpoint": endpoint_latency,
            "agents": {
                "total_calls":  _agent_counts["__total__"],
                "distribution": agent_dist,
                "latency":      agent_latency,
            },
            "tokens": {
                k: {
                    "prompt": int(v.get("prompt", 0)),
                    "completion": int(v.get("completion", 0)),
                    "total": int(v.get("total", 0)),
                    "cost_usd": round(v.get("cost_usd", 0.0), 6),
                    "avg_ttft_ms": round(v["ttft_sum"] / v["ttft_count"], 2) if v.get("ttft_count", 0) > 0 else 0.0,
                    "avg_tps": round(v["tps_sum"] / v["tps_count"], 2) if v.get("tps_count", 0) > 0 else 0.0,
                } for k, v in _token_usage.items()
            },
            "sessions": {
                "active":  len(_active_sessions),
                "total":   _counters["sessions_total"],
            },
            "errors": {
                "total":    _counters["errors_total"],
                "recent":   list(_error_log)[-10:],
            },
        }


def get_recent_requests(limit: int = 50) -> List[Dict]:
    with _lock:
        return list(_request_log)[-limit:]


def _fmt_uptime(seconds: float) -> str:
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
