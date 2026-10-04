"""End-to-end ingest → persist → reload → query using the local fallback path."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from app.agents.customer_support.rag.engine import RAGEngineBuilder
from app.agents.customer_support.rag.indexing import VectorIndexManager, persistence
from app.agents.customer_support.rag.ingestion_service import RAGIngestionService

MD = b"# Guide\n## Pricing\nThe plan costs ten dollars.\n## Support\nEmail us any time.\n"


@pytest.fixture
def local_rag(tmp_path, monkeypatch):
    persistence.get_persist_dir.cache_clear()
    monkeypatch.setattr(persistence, "_HF_PERSISTENT", tmp_path / "rag_index")
    prev = (Settings._embed_model, Settings._llm, VectorIndexManager.get_cached_index())
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.llm = MockLLM()
    VectorIndexManager.set_cached_index(None)
    refuse = ConnectionError("weaviate down")
    with patch("app.agents.customer_support.rag.indexing.manager.connect_weaviate", side_effect=refuse), \
         patch("app.agents.customer_support.rag.ingestion_service.connect_weaviate", side_effect=refuse):
        yield tmp_path / "rag_index"
    Settings._embed_model, Settings._llm = prev[0], prev[1]
    VectorIndexManager.set_cached_index(prev[2])
    persistence.get_persist_dir.cache_clear()


def test_ingest_persist_reload_query(local_rag):
    stages = []
    svc = RAGIngestionService(index_name="T")
    result = svc.ingest_bytes("guide.md", MD, status_callback=lambda s, m: stages.append(s))
    assert result["status"] == "success" and result["nodes_created"] >= 2
    assert stages == ["reading", "chunking", "embedding", "indexing"]
    assert local_rag.is_dir() and any(local_rag.iterdir())  # persisted to disk

    # Simulate a process restart: drop the in-memory cache, load from disk.
    VectorIndexManager.set_cached_index(None)
    manager = VectorIndexManager(index_name="T")
    assert manager.client is None
    index = manager.load_persisted_index()
    engine = RAGEngineBuilder(index).build_hybrid_query_engine(top_k=2)
    assert str(engine.query("pricing?"))  # MockLLM answers


def test_second_ingestion_is_additive(local_rag):
    svc = RAGIngestionService()
    first = svc.ingest_bytes("a.md", MD)["nodes_created"]
    second = svc.ingest_bytes("b.md", MD)["nodes_created"]
    index = VectorIndexManager.get_cached_index()
    assert len(index.docstore.docs) == first + second


def test_ingest_errors_are_returned_not_raised(local_rag):
    svc = RAGIngestionService()
    assert svc.ingest_files([])["status"] == "error"
    assert svc.ingest_bytes("x.md", MD, strategy="bogus")["status"] == "error"


def test_load_without_any_index_raises(local_rag):
    manager = VectorIndexManager()
    with pytest.raises(Exception, match="No index found"):
        manager.load_persisted_index()
