"""In-process metric storage, guarded by a single re-entrant lock.

These containers are mutated in place and never rebound, so every module may
import them by name.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Dict

lock = threading.RLock()
start_time = time.time()

# Rolling latency window (last 1000 samples per endpoint / agent / "__all__")
latency_window: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

counters: Dict[str, int] = defaultdict(int)
agent_counts: Dict[str, int] = defaultdict(int)

error_log: deque = deque(maxlen=50)      # last 50 errors
request_log: deque = deque(maxlen=200)   # last 200 requests (``/metrics/requests``)

active_sessions: Dict[str, float] = {}   # session_id → start_time

# Token and performance tracking per model
token_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: {
    "prompt": 0, "completion": 0, "total": 0,
    "cost_usd": 0.0, "ttft_sum": 0.0, "ttft_count": 0,
    "tps_sum": 0.0, "tps_count": 0,
})
