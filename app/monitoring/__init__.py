"""In-process observability for AI-lixir Scientific OS (formerly ``app/monitoring.py``).

Designed for HF Spaces single-container deployments where Prometheus/Grafana
are not available: all metrics live in memory behind a rolling JSON snapshot.

Tracks:
  * Request counts, latency (p50/p95/p99), error rates
  * Token usage, TTFT/TPS and estimated cost per model call
  * Agent routing distribution (chemical / medical / rag / app)
  * Out-of-domain rejection counts, active sessions, uptime, per-endpoint breakdown

Layout: ``state`` (storage) · ``pricing`` · ``recorders`` (write API) ·
``stats`` (percentiles/formatting) · ``snapshot`` (read API).
"""

from .pricing import MODEL_PRICING
from .recorders import (
    record_agent_call,
    record_error,
    record_out_of_domain,
    record_request,
    record_tokens,
    session_end,
    session_start,
)
from .snapshot import get_recent_requests, get_snapshot
from .stats import _fmt_uptime, _latency_stats, _percentile

# Legacy private aliases (same objects as in ``state``; mutated in place only).
from .state import (  # noqa: F401
    active_sessions as _active_sessions,
    agent_counts as _agent_counts,
    counters as _counters,
    error_log as _error_log,
    latency_window as _latency_window,
    lock as _lock,
    request_log as _request_log,
    start_time as _start_time,
    token_usage as _token_usage,
)

__all__ = [
    "MODEL_PRICING",
    "get_recent_requests",
    "get_snapshot",
    "record_agent_call",
    "record_error",
    "record_out_of_domain",
    "record_request",
    "record_tokens",
    "session_end",
    "session_start",
]
