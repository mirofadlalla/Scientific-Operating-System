"""Vector index lifecycle: Weaviate → disk → in-memory fallback chain."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from ..config import RAG_INDEX_NAME, WEAVIATE_HOST, WEAVIATE_PORT
from .compat import apply_weaviate_compat
from .connection import connect_weaviate
from .persistence import get_persist_dir, persist_index

logger = logging.getLogger(__name__)

apply_weaviate_compat()


class VectorIndexManager:
    """Connects to Weaviate, ingests data and manages the hybrid vector index.

    Fallback hierarchy (in order):
      1. Weaviate vector store (production / cloud)
      2. Persisted local disk index — survives restarts
      3. Pure in-memory index (last resort, lost on restart)
    """

    # Process-wide cache of the last local index (shared with the ingestion service).
    _GLOBAL_IN_MEMORY_INDEX: Optional[VectorStoreIndex] = None

    def __init__(self, index_name: str | None = None) -> None:
        self.index_name = index_name or RAG_INDEX_NAME
        logger.info("Connecting to Weaviate instance...")
        try:
            self.client = connect_weaviate()
            logger.info("✅ Weaviate connected at %s:%s", WEAVIATE_HOST, WEAVIATE_PORT)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[RAGAgent] Weaviate connection failed: %s. Will use disk/in-memory fallback.", exc
            )
            self.client = None

    # ── Process-wide local index cache ───────────────────────────────────────
    @classmethod
    def get_cached_index(cls) -> Optional[VectorStoreIndex]:
        """Return the in-memory index built/loaded earlier in this process, if any."""
        return cls._GLOBAL_IN_MEMORY_INDEX

    @classmethod
    def set_cached_index(cls, index: Optional[VectorStoreIndex]) -> None:
        cls._GLOBAL_IN_MEMORY_INDEX = index

    @classmethod
    def add_nodes_locally(cls, nodes: List[BaseNode]) -> VectorStoreIndex:
        """Additively ingest nodes into the local index and persist it.

        Existing data is preserved: nodes are inserted into the cached index when
        present, otherwise a new index is built.
        """
        existing = cls.get_cached_index()
        if existing is not None:
            logger.info("[Ingestion] Inserting %d new nodes into existing index.", len(nodes))
            for node in nodes:
                existing.insert(node)
            index = existing
            persist_index(index)
        else:
            index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults())
            persist_index(index)
        cls.set_cached_index(index)
        return index

    # ── Weaviate helpers ─────────────────────────────────────────────────────
    def _get_vector_store(self) -> Optional[WeaviateVectorStore]:
        if self.client is None:
            return None
        return WeaviateVectorStore(weaviate_client=self.client, index_name=self.index_name)

    # ── Public API ───────────────────────────────────────────────────────────
    def create_and_save_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """Build an index from nodes: Weaviate first, persisted disk index otherwise."""
        logger.info("Initializing index with %d nodes...", len(nodes))

        vector_store = self._get_vector_store()
        if vector_store:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context)
            logger.info("✅ Data ingested into Weaviate Vector Database.")
            return index

        logger.info("Weaviate unavailable — persisting index to disk: %s", get_persist_dir())
        index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults())
        persist_dir = persist_index(index)
        self.set_cached_index(index)
        logger.info("✅ Index persisted to disk at: %s", persist_dir)
        return index

    def load_persisted_index(self) -> VectorStoreIndex:
        """Load an existing index: Weaviate → disk → in-memory.

        Raises:
            Exception: when no index exists anywhere (ingest documents first).
        """
        if self.client is not None:
            index = self._try_load_weaviate()
            if index is not None:
                return index

        index = self._try_load_disk()
        if index is not None:
            return index

        cached = self.get_cached_index()
        if cached is not None:
            logger.info("Using existing in-memory index.")
            return cached

        raise Exception(
            "No index found: Weaviate unreachable, no disk index found, and no in-memory index exists. "
            "Please ingest documents first."
        )

    def close_connection(self) -> None:
        """Close the Weaviate connection (call on shutdown)."""
        if self.client:
            try:
                self.client.close()
                logger.info("Weaviate connection closed.")
            except Exception:  # noqa: BLE001
                pass

    # ── Load strategies ──────────────────────────────────────────────────────
    def _try_load_weaviate(self) -> Optional[VectorStoreIndex]:
        try:
            logger.info("Loading index from Weaviate: '%s'...", self.index_name)
            index = VectorStoreIndex.from_vector_store(self._get_vector_store())
            logger.info("✅ Weaviate index successfully loaded.")
            return index
        except Exception as exc:  # noqa: BLE001
            logger.warning("Weaviate load failed: %s. Trying disk fallback...", exc)
            return None

    def _try_load_disk(self) -> Optional[VectorStoreIndex]:
        persist_dir = get_persist_dir()
        if not (os.path.isdir(persist_dir) and os.listdir(persist_dir)):
            return None
        try:
            logger.info("Loading persisted index from disk: %s", persist_dir)
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            self.set_cached_index(index)
            logger.info("✅ Disk-persisted index successfully loaded.")
            return index
        except Exception as exc:  # noqa: BLE001
            logger.warning("Disk index load failed: %s. Trying in-memory fallback...", exc)
            return None
