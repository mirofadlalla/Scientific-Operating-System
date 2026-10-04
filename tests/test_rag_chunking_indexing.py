"""Tests for chunking factory, config and lazy persistence resolution."""

from __future__ import annotations

import pathlib

import pytest

from app.agents.customer_support.rag.chunking import (
    ChunkingFactory,
    MarkdownStrategy,
    SentenceStrategy,
    TokenStrategy,
)
from app.agents.customer_support.rag.indexing import VectorIndexManager, persistence


def test_chunking_factory_types():
    assert isinstance(ChunkingFactory.get_strategy("markdown"), MarkdownStrategy)
    assert isinstance(ChunkingFactory.get_strategy(" Sentence ", chunk_size=64), SentenceStrategy)
    token = ChunkingFactory.get_strategy("token", chunk_size=10, chunk_overlap=2)
    assert isinstance(token, TokenStrategy) and token.chunk_size == 10


def test_chunking_factory_markdown_ignores_kwargs():
    assert isinstance(ChunkingFactory.get_strategy("markdown", chunk_size=1), MarkdownStrategy)


def test_chunking_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported"):
        ChunkingFactory.get_strategy("nope")


def test_markdown_strategy_chunks_by_header():
    from llama_index.core.schema import Document

    nodes = MarkdownStrategy().chunk([Document(text="# A\nx\n## B\ny\n## C\nz")])
    assert len(nodes) >= 2


def test_persist_dir_is_lazy_and_cached(tmp_path, monkeypatch):
    persistence.get_persist_dir.cache_clear()
    monkeypatch.setattr(persistence, "_HF_PERSISTENT", tmp_path / "rag_index")
    first = persistence.get_persist_dir()
    assert first == str(tmp_path / "rag_index")
    monkeypatch.setattr(persistence, "_HF_PERSISTENT", tmp_path / "other")
    assert persistence.get_persist_dir() == first  # cached
    persistence.get_persist_dir.cache_clear()


def test_persist_dir_falls_back_to_package_storage(monkeypatch, tmp_path):
    persistence.get_persist_dir.cache_clear()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    # parent of the HF dir is a *file* → mkdir/touch fail → local fallback
    monkeypatch.setattr(persistence, "_HF_PERSISTENT", blocker / "rag_index")
    result = persistence.get_persist_dir()
    # Use Path for comparison so it works on both Linux (/) and Windows (\).
    result_path = pathlib.Path(result)
    assert result_path.parts[-3:] == ("rag", "storage", "rag_index")
    assert persistence.RAG_ROOT.name == "rag" and (persistence.RAG_ROOT / "storage").is_dir()
    persistence.get_persist_dir.cache_clear()


def test_in_memory_cache_roundtrip():
    prev = VectorIndexManager.get_cached_index()
    try:
        marker = object()
        VectorIndexManager.set_cached_index(marker)  # type: ignore[arg-type]
        assert VectorIndexManager.get_cached_index() is marker
        assert VectorIndexManager._GLOBAL_IN_MEMORY_INDEX is marker  # legacy attribute still honoured
    finally:
        VectorIndexManager.set_cached_index(prev)


def test_data_files_are_where_cli_expects():
    from app.agents.customer_support.rag import cli

    assert cli.DATA_DIR.is_dir() and any(cli.DATA_DIR.glob("*.md"))
