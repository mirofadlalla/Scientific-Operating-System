import os
import tempfile
from typing import List, Dict, Any, Callable

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore

import weaviate

# Re-use the chunking strategies already in this package
import sys, pathlib
_rag_root = pathlib.Path(__file__).resolve().parent.parent  # ai-lixir-rag-system/
if str(_rag_root) not in sys.path:
    sys.path.insert(0, str(_rag_root))

from chunking.factory import ChunkingFactory


class RAGIngestionService:
    """
    Handles the full RAG ingestion pipeline:
      file bytes / file paths → chunk → embed (Groq) → store (Weaviate)

    Usage
    -----
    service = RAGIngestionService(index_name="AdmetIndex")
    result  = await service.ingest_bytes("guide.md", b"# SERVICE: ...")
    # → {"status": "success", "nodes_created": 42, ...}
    """

    def __init__(self, index_name: str = "AdmetIndex"):
        self.index_name = index_name
        self._client: weaviate.WeaviateClient | None = None

    # ── Weaviate connection ──────────────────────────────────────────────────
    def _get_client(self) -> weaviate.WeaviateClient:
        if self._client is None:
            from src.config import WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT
            self._client = weaviate.connect_to_local(
                host=WEAVIATE_HOST,
                port=WEAVIATE_PORT,
                grpc_port=WEAVIATE_GRPC_PORT
            )
        return self._client

    def _get_vector_store(self) -> WeaviateVectorStore:
        return WeaviateVectorStore(
            weaviate_client=self._get_client(),
            index_name=self.index_name,
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ── Public ingestion API ─────────────────────────────────────────────────
    def ingest_files(
        self,
        file_paths: List[str],
        strategy: str = "markdown",
        status_callback: Callable[[str, str], None] | None = None,
        **strategy_kwargs,
    ) -> Dict[str, Any]:
        """
        Ingest a list of file paths into the Weaviate index.

        Parameters
        ----------
        file_paths   : absolute paths to .md (or any supported) files
        strategy     : 'markdown' | 'sentence' | 'token'
        **strategy_kwargs : chunk_size, chunk_overlap, etc. for non-markdown strategies
        """
        if not file_paths:
            return {"status": "error", "message": "No files provided."}

        try:
            if status_callback:
                status_callback("reading", "Reading file...")
            # 1. Load documents
            reader = SimpleDirectoryReader(input_files=file_paths)
            documents = reader.load_data()

            if not documents:
                return {"status": "error", "message": "No content found in provided files."}

            return self._run_pipeline(documents, strategy, status_callback=status_callback, **strategy_kwargs)

        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def ingest_bytes(
        self,
        filename: str,
        content: bytes,
        strategy: str = "markdown",
        status_callback: Callable[[str, str], None] | None = None,
        **strategy_kwargs,
    ) -> Dict[str, Any]:
        """
        Ingest file bytes (e.g. from an HTTP upload) into the Weaviate index.
        Writes to a temp file, ingests, then cleans up.
        """
        suffix = pathlib.Path(filename).suffix or ".md"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, mode="wb"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            return self.ingest_files([tmp_path], strategy, status_callback=status_callback, **strategy_kwargs)

        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Internal pipeline ────────────────────────────────────────────────────
    def _run_pipeline(
        self,
        documents,
        strategy: str,
        status_callback: Callable[[str, str], None] | None = None,
        **strategy_kwargs,
    ) -> Dict[str, Any]:
        if status_callback:
            status_callback("chunking", "Chunking document...")
        # 2. Chunk
        chunker = ChunkingFactory.get_strategy(strategy, **strategy_kwargs)
        nodes   = chunker.chunk(documents)

        if not nodes:
            return {"status": "error", "message": "Chunking produced 0 nodes."}

        if status_callback:
            status_callback("embedding", "Generating embeddings (Groq)...")

        # 3. Build index → embed + store in Weaviate
        vector_store    = self._get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if status_callback:
            status_callback("indexing", "Storing in Weaviate...")

        VectorStoreIndex(nodes, storage_context=storage_context)

        return {
            "status":        "success",
            "index_name":    self.index_name,
            "nodes_created": len(nodes),
            "strategy":      strategy,
            "files":         len(documents),
        }
