"""Pure statistics / formatting helpers."""

from __future__ import annotations

import statistics
from typing import Dict, List

from . import state


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return round(sorted_data[min(idx, len(sorted_data) - 1)], 2)


def _latency_stats(key: str) -> Dict:
    """p50/p95/p99/avg/min/max/count for the ``key`` latency window (caller holds the lock)."""
    data = list(state.latency_window.get(key, []))
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


def _fmt_uptime(seconds: float) -> str:
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
