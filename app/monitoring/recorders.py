"""Public write API — called from middleware and agents."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from . import state
from .pricing import estimate_cost


def record_request(
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: float,
    session_id: Optional[str] = None,
) -> None:
    """Record a completed HTTP request."""
    with state.lock:
        state.counters["requests_total"] += 1
        state.counters[f"requests_{method.upper()}"] += 1
        state.latency_window[endpoint].append(latency_ms)
        state.latency_window["__all__"].append(latency_ms)

        if status_code >= 500:
            state.counters["errors_5xx"] += 1
        elif status_code >= 400:
            state.counters["errors_4xx"] += 1
        else:
            state.counters["requests_success"] += 1

        state.request_log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status": status_code,
            "latency_ms": round(latency_ms, 2),
        })


def record_agent_call(agent: str, intent: str, latency_ms: float, success: bool = True) -> None:
    """Record an agent routing event."""
    with state.lock:
        state.agent_counts[agent] += 1
        state.agent_counts["__total__"] += 1
        state.latency_window[f"agent_{agent}"].append(latency_ms)
        if not success:
            state.counters[f"agent_{agent}_errors"] += 1


def record_out_of_domain(reason: str = "") -> None:
    """Record an out-of-domain rejection."""
    with state.lock:
        state.counters["out_of_domain_total"] += 1


def record_tokens(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: Optional[float] = None,
    tps: Optional[float] = None,
) -> None:
    """Record LLM token usage, latency performance (TTFT, TPS) and estimated cost."""
    with state.lock:
        cost = round(estimate_cost(model, prompt_tokens, completion_tokens), 6)

        for key in (model, "__all__"):
            usage = state.token_usage[key]
            usage["prompt"] += prompt_tokens
            usage["completion"] += completion_tokens
            usage["total"] += prompt_tokens + completion_tokens
            usage["cost_usd"] += cost

        usage = state.token_usage[model]
        if ttft_ms is not None:
            usage["ttft_sum"] += ttft_ms
            usage["ttft_count"] += 1
        if tps is not None:
            usage["tps_sum"] += tps
            usage["tps_count"] += 1


def record_error(endpoint: str, error: str, session_id: Optional[str] = None) -> None:
    """Record an application error."""
    with state.lock:
        state.counters["errors_total"] += 1
        state.error_log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "error": str(error)[:300],
            "session_id": session_id,
        })


def session_start(session_id: str) -> None:
    with state.lock:
        state.active_sessions[session_id] = time.time()
        state.counters["sessions_total"] += 1


def session_end(session_id: str) -> None:
    with state.lock:
        state.active_sessions.pop(session_id, None)
