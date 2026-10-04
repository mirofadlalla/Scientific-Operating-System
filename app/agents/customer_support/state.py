"""Process-wide readiness flags for the RAG subsystem (strict readiness gating)."""

from __future__ import annotations

from typing import Dict

rag_state: Dict[str, bool] = {
    "ready": False,
    "embedding_initialized": False,
    "llm_initialized": False,
}
