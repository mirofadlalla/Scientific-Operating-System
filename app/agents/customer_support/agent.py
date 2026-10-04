"""
CustomerSupportRAGAgent
=======================
Async singleton agent that wraps the full ai-lixir RAG pipeline:
  Groq LLM + embeddings → Weaviate hybrid search → structured answer

Lifecycle
---------
* On first ``run()`` call the engine is initialised (one-time cost).
* Subsequent calls reuse the warmed-up query engine (near-instant).
* Call ``close()`` on app shutdown to release the Weaviate connection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.config import settings as _app_settings

from .rag import config as rag_config
from .rag.bootstrap import configure_llama_index
from .rag.engine import RAGEngineBuilder
from .rag.indexing import VectorIndexManager
from .rag.ingestion_service import RAGIngestionService
from .rag.status import probe_weaviate
from .state import rag_state

logger = logging.getLogger(__name__)

__all__ = ["CustomerSupportRAGAgent", "rag_state"]

_UNAVAILABLE_MSG = (
    "The Knowledge Base is currently unavailable (Weaviate not reachable). "
    "Please ensure the Weaviate instance is running and try again."
)
_EMPTY_MSG = "The Knowledge Base is currently empty or offline. Please ingest documents first."
_NO_ANSWER_MSG = "The documentation does not contain information about this topic."


class CustomerSupportRAGAgent:
    """Production-grade RAG agent for AI-lixir customer support & documentation queries.

    Retrieval pipeline:
        Query → Embedding Model → Weaviate Hybrid Search → Top-K chunks → LLM → Answer
    """

    # ── Configurable via env vars ────────────────────────────────────────────
    INDEX_NAME: str = _app_settings.RAG_INDEX_NAME  # Weaviate collection name
    TOP_K: int = _app_settings.RAG_TOP_K             # retrieved chunks per query
    ALPHA: float = _app_settings.RAG_ALPHA           # hybrid balance (0=BM25, 1=vector)

    def __init__(self) -> None:
        self._ready: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        self._query_engine: Any = None
        self._index_manager: Optional[VectorIndexManager] = None
        self._ingestion_svc: Optional[RAGIngestionService] = None

    # ── Public API ───────────────────────────────────────────────────────────
    async def run(self, query: str) -> str:
        """Execute a RAG query; returns a grounded answer or an informative message."""
        if not self._ready:
            await self._initialise()
        if not self._ready:
            return _UNAVAILABLE_MSG
        if not self._query_engine:
            return _EMPTY_MSG

        try:
            # LlamaIndex query engines are synchronous — run in thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self._query_engine.query(query))
            return str(response).strip() or _NO_ANSWER_MSG
        except Exception as exc:  # noqa: BLE001
            logger.error("[RAGAgent] Query failed: %s", exc)
            return f"[RAG Query Error]: {exc}"

    def get_ingestion_service(self) -> RAGIngestionService:
        """Return the shared ingestion service (created lazily)."""
        if self._ingestion_svc is None:
            self._ingestion_svc = RAGIngestionService(index_name=self.INDEX_NAME)
        return self._ingestion_svc

    async def reload_engine(self) -> None:
        """Force a reload of the query engine after new documents were ingested.

        Resets both instance and global readiness flags so ``_initialise`` re-runs fully.
        """
        async with self._lock:
            self._ready = False
            self._query_engine = None
            rag_state["ready"] = False
        await self._initialise()

    async def status(self) -> dict:
        """Health/status dict for the ``/rag/status`` endpoint."""
        loop = asyncio.get_running_loop()
        weaviate_ok, node_count = await loop.run_in_executor(None, probe_weaviate, self.INDEX_NAME)
        return {
            "weaviate_connected": weaviate_ok,
            "index_name": self.INDEX_NAME,
            "node_count": node_count,
            "engine_ready": self._ready,
            "embed_model": f"{rag_config.EMBEDDING_PROVIDER}/{rag_config.EMBED_MODEL}",
            "llm_model": f"groq/{rag_config.ORCHESTRATOR_MODEL}",
            "search_mode": f"hybrid (α={self.ALPHA})",
            "top_k": self.TOP_K,
        }

    async def close(self) -> None:
        """Release Weaviate connections on app shutdown."""
        if self._index_manager:
            try:
                self._index_manager.close_connection()
            except Exception:  # noqa: BLE001
                pass
        if self._ingestion_svc:
            try:
                self._ingestion_svc.close()
            except Exception:  # noqa: BLE001
                pass

    # ── Internal initialisation ──────────────────────────────────────────────
    async def _initialise(self) -> None:
        """One-time async initialisation, guarded by a lock against duplicate runs."""
        async with self._lock:
            if self._ready or rag_state["ready"]:
                return
            try:
                logger.info("[RAGAgent] Initialising Groq + Weaviate environment…")
                loop = asyncio.get_running_loop()

                if not rag_state.get("embedding_initialized") or not rag_state.get("llm_initialized"):
                    await loop.run_in_executor(None, configure_llama_index, rag_state)
                else:
                    logger.info("[RAGAgent] LLM + Embeddings already initialised — skipping bootstrap.")

                self._index_manager = VectorIndexManager(index_name=self.INDEX_NAME)
                index = await self._load_index(loop)
                if index is not None:
                    self._query_engine = RAGEngineBuilder(index=index).build_hybrid_query_engine(
                        top_k=self.TOP_K, alpha=self.ALPHA
                    )

                # Ready even without data: the system can still receive ingestions.
                self._ready = True
                rag_state["ready"] = True
                logger.info("[RAGAgent] Initialisation complete.")
            except Exception as exc:  # noqa: BLE001
                logger.error("[RAGAgent] Initialisation failed: %s", exc)

    async def _load_index(self, loop: asyncio.AbstractEventLoop) -> Any:
        """Load the persisted index, or fall back to the in-memory one; ``None`` if no data yet."""
        assert self._index_manager is not None
        try:
            index = await loop.run_in_executor(None, self._index_manager.load_persisted_index)
            logger.info("[RAGAgent] Loaded existing index '%s'.", self.INDEX_NAME)
            return index
        except Exception as exc:  # noqa: BLE001
            cached = VectorIndexManager.get_cached_index()
            if cached is not None:
                logger.info("[RAGAgent] Loaded existing in-memory index.")
                return cached
            logger.warning(
                "[RAGAgent] Could not load index ('%s'). Engine will become active after first ingestion.",
                exc,
            )
            return None
