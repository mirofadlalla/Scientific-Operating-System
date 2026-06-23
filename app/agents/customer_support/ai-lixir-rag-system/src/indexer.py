import os
import pathlib
import logging
import weaviate
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

# ── Disk persistence directory (used when Weaviate is unavailable) ─────────────
# Resolves to: ai-lixir-rag-system/storage/rag_index/
_RAG_SYSTEM_ROOT = pathlib.Path(__file__).resolve().parent.parent
PERSIST_DIR = str(_RAG_SYSTEM_ROOT / "storage" / "rag_index")


class VectorIndexManager:
    _GLOBAL_IN_MEMORY_INDEX = None

    """
    Manager class responsible for connecting to Weaviate, ingesting data,
    and managing the Hybrid Vector Store Index.

    Fallback hierarchy (in order):
      1. Weaviate vector store (production / cloud)
      2. Persisted local disk index (./storage/rag_index) — survives restarts
      3. Pure in-memory index (last resort, lost on restart)
    """

    def __init__(self, index_name: str = "AdmetIndex"):
        """
        Initializes the Weaviate client and sets the target index name.
        """
        self.index_name = index_name
        logger.info("Connecting to Weaviate instance...")

        from src.config import WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT
        try:
            self.client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT,
                grpc_port=WEAVIATE_GRPC_PORT
            )
            logger.info(f"✅ Weaviate connected at {WEAVIATE_HOST}:{WEAVIATE_PORT}")
        except Exception as e:
            logger.warning(
                f"[RAGAgent] Weaviate connection failed: {e}. "
                "Will use disk/in-memory fallback."
            )
            self.client = None

    def _get_vector_store(self):
        """Helper method to initialize the WeaviateVectorStore."""
        if self.client is None:
            return None
        return WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=self.index_name
        )

    def create_and_save_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """
        Builds a VectorStoreIndex from the provided nodes.
        Tries Weaviate first; falls back to persisted disk index.
        """
        logger.info(f"Initializing index with {len(nodes)} nodes...")

        vector_store = self._get_vector_store()

        if vector_store:
            # \u2500\u2500 Weaviate path \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context)
            logger.info("✅ Data ingested into Weaviate Vector Database.")
        else:
            # \u2500\u2500 Disk persistence fallback \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            logger.info(f"Weaviate unavailable — persisting index to disk: {PERSIST_DIR}")
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex(nodes, storage_context=storage_context)
            os.makedirs(PERSIST_DIR, exist_ok=True)
            storage_context.persist(persist_dir=PERSIST_DIR)
            VectorIndexManager._GLOBAL_IN_MEMORY_INDEX = index
            logger.info(f"✅ Index persisted to disk at: {PERSIST_DIR}")

        return index

    def load_persisted_index(self) -> VectorStoreIndex:
        """
        Loads an existing index.
        Tries Weaviate first; falls back to disk-persisted index; then in-memory.
        """
        if self.client is not None:
            # \u2500\u2500 Try Weaviate \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            try:
                logger.info(f"Loading index from Weaviate: '{self.index_name}'...")
                vector_store = self._get_vector_store()
                index = VectorStoreIndex.from_vector_store(vector_store)
                logger.info("✅ Weaviate index successfully loaded.")
                return index
            except Exception as e:
                logger.warning(f"Weaviate load failed: {e}. Trying disk fallback...")

        # \u2500\u2500 Try disk-persisted index \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            try:
                logger.info(f"Loading persisted index from disk: {PERSIST_DIR}")
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                index = load_index_from_storage(storage_context)
                VectorIndexManager._GLOBAL_IN_MEMORY_INDEX = index
                logger.info("✅ Disk-persisted index successfully loaded.")
                return index
            except Exception as e:
                logger.warning(f"Disk index load failed: {e}. Trying in-memory fallback...")

        # \u2500\u2500 Try previously built in-memory index \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if VectorIndexManager._GLOBAL_IN_MEMORY_INDEX is not None:
            logger.info("Using existing in-memory index.")
            return VectorIndexManager._GLOBAL_IN_MEMORY_INDEX

        raise Exception(
            "No index found: Weaviate unreachable, no disk index found, and no in-memory index exists. "
            "Please ingest documents first."
        )

    def close_connection(self):
        """Closes the connection to Weaviate. Should be called when shutting down."""
        if self.client:
            try:
                self.client.close()
                logger.info("Weaviate connection closed.")
            except Exception:
                pass