"""
CustomerSupportRAGAgent
=======================
Async singleton agent that wraps the full ai-lixir RAG pipeline:
  Groq LLM + Groq Embeddings → Weaviate Hybrid Search → structured answer

Lifecycle
---------
• On first `run()` call the engine is initialised (one-time cost).
• Subsequent calls reuse the warmed-up query engine (near-instant).
• Call `close()` on app shutdown to release the Weaviate connection.
"""

import sys
import asyncio
import logging
import pathlib
from typing import Optional

import weaviate

# ── Make the RAG package importable from wherever this file is loaded ─────────
_rag_root = pathlib.Path(__file__).resolve().parent / "ai-lixir-rag-system"
if str(_rag_root) not in sys.path:
    sys.path.insert(0, str(_rag_root))

from src.engine   import RAGEngineBuilder
from src.indexer  import VectorIndexManager
from src.ingestion_service import RAGIngestionService

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fallback system prompt (used when Weaviate is unavailable)
# ──────────────────────────────────────────────────────────────────────────────
_UNAVAILABLE_MSG = (
    "The Knowledge Base is currently unavailable (Weaviate not reachable). "
    "Please ensure the Weaviate instance is running and try again."
)


# Global State for strict readiness gating
rag_state = {
    "ready": False,
    "embedding_initialized": False,
    "llm_initialized": False
}

class CustomerSupportRAGAgent:
    """
    Production-grade RAG agent for AI-lixir customer support & documentation queries.

    Retrieval pipeline
    ------------------
    Query → Groq llama-text-embed-v2 → Weaviate Hybrid Search (α=0.5)
          → Top-4 chunks → Groq llama-3.3-70b-versatile → Answer
    """

    INDEX_NAME = "AilixirDocs"       # Weaviate collection name
    TOP_K      = 5                   # retrieved chunks per query
    ALPHA      = 0.5                 # hybrid search balance (0=BM25, 1=vector)

    def __init__(self) -> None:
        self._ready:       bool                     = False
        self._initialising: bool                    = False
        self._lock:        asyncio.Lock             = asyncio.Lock()
        self._query_engine                          = None
        self._index_manager: Optional[VectorIndexManager] = None
        self._ingestion_svc: Optional[RAGIngestionService] = None

    # ── Public API ────────────────────────────────────────────────────────────
    async def run(self, query: str) -> str:
        """
        Execute a RAG query against the AI-lixir knowledge base.

        Returns a grounded answer, or an informative message if the KB
        is empty / Weaviate is unreachable.
        """
        if not self._ready:
            await self._initialise()

        if not self._ready:
            return _UNAVAILABLE_MSG

        try:
            # LlamaIndex query engines are synchronous — run in thread pool
            loop     = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._query_engine.query(query)
            )
            answer = str(response).strip()
            return answer if answer else "The documentation does not contain information about this topic."
        except Exception as exc:
            logger.error(f"[RAGAgent] Query failed: {exc}")
            return f"[RAG Query Error]: {exc}"

    def get_ingestion_service(self) -> RAGIngestionService:
        """Return the shared ingestion service (creates one lazily)."""
        if self._ingestion_svc is None:
            self._ingestion_svc = RAGIngestionService(index_name=self.INDEX_NAME)
        return self._ingestion_svc

    async def reload_engine(self) -> None:
        """
        Force a reload of the query engine from Weaviate.
        Call this after new documents have been ingested.
        """
        async with self._lock:
            self._ready = False
            self._query_engine = None
        await self._initialise()

    async def status(self) -> dict:
        """Return a health/status dict for the /rag/status endpoint."""
        weaviate_ok = False
        node_count  = 0
        try:
            from src.config import WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT
            client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT,
                grpc_port=WEAVIATE_GRPC_PORT
            )
            weaviate_ok = client.is_ready()
            # Try to get approximate object count
            try:
                col = client.collections.get(self.INDEX_NAME)
                node_count = col.aggregate.over_all().total_count
            except Exception:
                node_count = -1   # collection may not exist yet
            client.close()
        except Exception:
            pass

        from src.config import EMBED_MODEL, ORCHESTRATOR_MODEL, EMBEDDING_PROVIDER

        return {
            "weaviate_connected": weaviate_ok,
            "index_name":        self.INDEX_NAME,
            "node_count":        node_count,
            "engine_ready":      self._ready,
            "embed_model":       f"{EMBEDDING_PROVIDER}/{EMBED_MODEL}",
            "llm_model":         f"groq/{ORCHESTRATOR_MODEL}",
            "search_mode":       f"hybrid (α={self.ALPHA})",
            "top_k":             self.TOP_K,
        }

    async def close(self) -> None:
        """Release Weaviate connection on app shutdown."""
        if self._index_manager:
            try:
                self._index_manager.close_connection()
            except Exception:
                pass
        if self._ingestion_svc:
            try:
                self._ingestion_svc.close()
            except Exception:
                pass

    # ── Internal initialisation ───────────────────────────────────────────────
    def _bootstrap_llama_index(self):
        """Atomic initialization of embeddings and LLM."""
        from src.config import GROQ_API_KEY, ORCHESTRATOR_MODEL, EMBEDDING_PROVIDER, EMBED_MODEL, OPENAI_API_KEY
        from src.embeddings import EmbeddingProviderFactory
        from llama_index.llms.groq import Groq
        from llama_index.core import Settings
        
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")

        # 1. Initialize Embeddings FIRST
        Settings.embed_model = EmbeddingProviderFactory.create_embedding_model(
            provider=EMBEDDING_PROVIDER,
            model_name=EMBED_MODEL,
            api_key=OPENAI_API_KEY
        )
        rag_state["embedding_initialized"] = True
        
        # 2. Initialize LLM
        Settings.llm = Groq(
            model=ORCHESTRATOR_MODEL,
            api_key=GROQ_API_KEY,
        )
        rag_state["llm_initialized"] = True
        
        logger.info(f"✅ LLM  → Groq / {ORCHESTRATOR_MODEL}")
        logger.info(f"✅ Embed → {EMBEDDING_PROVIDER.capitalize()} / {EMBED_MODEL}")

    async def _initialise(self) -> None:
        """
        One-time async initialisation (protected by asyncio.Lock to prevent
        duplicate concurrent initialisations).
        """
        async with self._lock:
            if self._ready or rag_state["ready"]: return
            
            try:
                logger.info("[RAGAgent] Initialising Groq + Weaviate environment…")

                # 1. Configure LlamaIndex globals
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._bootstrap_llama_index)

                # 2. Connect to Weaviate and load / build the index
                self._index_manager = VectorIndexManager(index_name=self.INDEX_NAME)

                try:
                    index = await loop.run_in_executor(
                        None, self._index_manager.load_persisted_index
                    )
                    logger.info(f"[RAGAgent] Loaded existing index '{self.INDEX_NAME}' from Weaviate.")
                except Exception as exc:
                    import src.indexer
                    if getattr(src.indexer.VectorIndexManager, "_GLOBAL_IN_MEMORY_INDEX", None) is not None:
                        index = src.indexer.VectorIndexManager._GLOBAL_IN_MEMORY_INDEX
                        logger.info("[RAGAgent] Loaded existing in-memory index.")
                    else:
                        logger.warning(
                            f"[RAGAgent] Could not load index ('{exc}'). "
                            "Engine will become active after first ingestion."
                        )
                        # No data yet — engine not ready until documents are ingested
                        return

                # 3. Build the hybrid query engine
                engine_builder     = RAGEngineBuilder(index=index)
                self._query_engine = engine_builder.build_hybrid_query_engine(
                    top_k=self.TOP_K,
                    alpha=self.ALPHA,
                )

                # 4. Agent ready
                self._ready = True
                rag_state["ready"] = True
                logger.info("[RAGAgent] Initialisation complete.")

            except Exception as exc:
                logger.error(f"[RAGAgent] Initialisation failed: {exc}")
            finally:
                self._initialising = False
