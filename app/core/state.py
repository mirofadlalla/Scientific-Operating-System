"""
app.core.state
~~~~~~~~~~~~~~
All mutable module-level globals in one place.

Rules:
  - Only primitive types, dicts, and None live here at import time.
  - The lifespan (core/lifespan.py) mutates these attributes on startup.
  - Every other module imports this module and accesses attrs via `state.xxx`
    so they always see the most-recent assignment.
"""
from typing import Dict, List, Any, Optional

# ── Redis / RQ ────────────────────────────────────────────────────────────────
worker_process = None
redis_available: bool = False
redis_conn = None
rq_queue = None

# ── Long-term memory (set in lifespan) ───────────────────────────────────────
long_memory = None  # type: Optional[Any]  # LongTermMemory instance

# ── In-process RAG ingestion job tracker ─────────────────────────────────────
ingestion_jobs: Dict[str, Dict[str, Any]] = {}

# ── Per-session conversation buffer (non-Redis fallback) ─────────────────────
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

# ── Active WebSocket voice sessions ──────────────────────────────────────────
active_voice_sessions: Dict[str, Any] = {}
