"""RAG ingestion pipeline: file bytes / paths → chunk → embed → store."""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile
from typing import Any, Callable, Dict, List, Optional

import weaviate
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from .chunking import ChunkingFactory
from .config import RAG_INDEX_NAME
from .indexing import VectorIndexManager, apply_weaviate_compat, connect_weaviate, get_persist_dir

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str, str], None]


class RAGIngestionService:
    """Handles the full RAG ingestion pipeline.

    Example::

        service = RAGIngestionService(index_name="AdmetIndex")
        result = service.ingest_bytes("guide.md", b"# SERVICE: ...")
        # → {"status": "success", "nodes_created": 42, ...}
    """

    def __init__(self, index_name: str | None = None) -> None:
        self.index_name = index_name or RAG_INDEX_NAME
        self._client: Optional[weaviate.WeaviateClient] = None
        self._client_failed = False

    # ── Weaviate connection ──────────────────────────────────────────────────
    def _get_client(self) -> Optional[weaviate.WeaviateClient]:
        if self._client_failed:
            return None
        if self._client is None:
            try:
                self._client = connect_weaviate()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Weaviate connection failed in ingestion: %s", exc)
                self._client_failed = True
                return None
        return self._client

    def _get_vector_store(self) -> Optional[WeaviateVectorStore]:
        client = self._get_client()
        if client is None:
            return None
        return WeaviateVectorStore(weaviate_client=client, index_name=self.index_name)

    def close(self) -> None:
        """Close the Weaviate connection, if one was opened."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    # ── Public ingestion API ─────────────────────────────────────────────────
    def ingest_files(
        self,
        file_paths: List[str],
        strategy: str = "markdown",
        status_callback: StatusCallback | None = None,
        **strategy_kwargs: Any,
    ) -> Dict[str, Any]:
        """Ingest file paths into the index.

        Args:
            file_paths: absolute paths to .md (or any supported) files.
            strategy: 'markdown' | 'sentence' | 'token'.
            status_callback: optional (stage, message) progress hook.
            **strategy_kwargs: chunk_size, chunk_overlap … for non-markdown strategies.
        """
        if not file_paths:
            return {"status": "error", "message": "No files provided."}

        try:
            if status_callback:
                status_callback("reading", "Reading file...")
            documents = SimpleDirectoryReader(input_files=file_paths).load_data()
            if not documents:
                return {"status": "error", "message": "No content found in provided files."}
            return self._run_pipeline(documents, strategy, status_callback, **strategy_kwargs)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": str(exc)}

    def ingest_bytes(
        self,
        filename: str,
        content: bytes,
        strategy: str = "markdown",
        status_callback: StatusCallback | None = None,
        **strategy_kwargs: Any,
    ) -> Dict[str, Any]:
        """Ingest uploaded bytes: write a temp file, ingest, then clean up."""
        suffix = pathlib.Path(filename).suffix or ".md"
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            return self.ingest_files([tmp_path], strategy, status_callback, **strategy_kwargs)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": str(exc)}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Internal pipeline ────────────────────────────────────────────────────
    def _run_pipeline(
        self,
        documents: list,
        strategy: str,
        status_callback: StatusCallback | None = None,
        **strategy_kwargs: Any,
    ) -> Dict[str, Any]:
        if status_callback:
            status_callback("chunking", "Chunking document...")
        nodes = ChunkingFactory.get_strategy(strategy, **strategy_kwargs).chunk(documents)
        if not nodes:
            return {"status": "error", "message": "Chunking produced 0 nodes."}

        if status_callback:
            status_callback("embedding", "Generating embeddings (Groq)...")
        vector_store = self._get_vector_store()
        if status_callback:
            status_callback("indexing", "Storing in Vector Database...")

        if vector_store:
            apply_weaviate_compat()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex(nodes, storage_context=storage_context)
        else:
            logger.info("Weaviate unavailable — building local index: %s", get_persist_dir())
            VectorIndexManager.add_nodes_locally(nodes)
            logger.info("✅ Ingested %d nodes, persisted to: %s", len(nodes), get_persist_dir())

        return {
            "status": "success",
            "index_name": self.index_name,
            "nodes_created": len(nodes),
            "strategy": strategy,
            "files": len(documents),
        }
