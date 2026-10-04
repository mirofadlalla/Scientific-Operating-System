"""Tests for the modular embeddings package (providers, factory, fallback)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.agents.customer_support.rag.embeddings import (
    E5InstructEmbedding,
    EmbeddingInitializationError,
    EmbeddingProviderFactory,
    JinaEmbedding,
    is_e5_instruct,
)
from app.agents.customer_support.rag.embeddings.providers.fallback import (
    SentenceTransformerEmbedding,
)


class _FakeST:
    """Stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list = []

    def encode(self, data, **kwargs):
        self.calls.append(data)
        if isinstance(data, list):
            return np.ones((len(data), 3))
        return np.ones(3)


@pytest.fixture
def fake_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def test_is_e5_instruct():
    assert is_e5_instruct("intfloat/multilingual-e5-large-instruct")
    assert not is_e5_instruct("intfloat/multilingual-e5-large")
    assert not is_e5_instruct("BAAI/bge-m3")


def test_e5_applies_query_prefix_only_to_queries(fake_st):
    emb = E5InstructEmbedding(model_name="intfloat/multilingual-e5-large-instruct")
    emb._get_query_embedding("what is aspirin")
    emb._get_text_embedding("aspirin is a drug")
    q, doc = emb._st.calls
    assert q.startswith("Instruct: ") and q.endswith("Query: what is aspirin")
    assert doc == "aspirin is a drug"


@pytest.mark.asyncio
async def test_async_methods_delegate_to_sync_in_executor(fake_st):
    emb = E5InstructEmbedding(model_name="intfloat/multilingual-e5-large-instruct")
    assert await emb._aget_query_embedding("q") == [1.0, 1.0, 1.0]
    assert await emb._aget_text_embedding("t") == [1.0, 1.0, 1.0]
    assert await emb._aget_text_embeddings(["a", "b"]) == [[1.0] * 3, [1.0] * 3]


def test_jina_requires_api_key():
    with pytest.raises(ValueError, match="JINA_API_KEY"):
        JinaEmbedding(api_key="")


def test_jina_sorts_by_index_and_sends_task_hint():
    emb = JinaEmbedding(api_key="k", model_name="m")
    resp = MagicMock()
    resp.json.return_value = {
        "data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]},
        ]
    }
    with patch("app.agents.customer_support.rag.embeddings.providers.jina.requests.post",
               return_value=resp) as post:
        assert emb._get_text_embeddings(["a", "b"]) == [[1.0], [2.0]]
        assert post.call_args.kwargs["json"]["task"] == "retrieval.passage"
        emb._get_query_embedding("q")
        assert post.call_args.kwargs["json"]["task"] == "retrieval.query"


def test_factory_routes_e5_for_huggingface(fake_st):
    model = EmbeddingProviderFactory.create_embedding_model(
        "huggingface", "intfloat/multilingual-e5-large-instruct"
    )
    assert isinstance(model, E5InstructEmbedding)


def test_factory_groq_is_remapped_to_huggingface(fake_st):
    model = EmbeddingProviderFactory.create_embedding_model("groq", "ignored")
    assert isinstance(model, E5InstructEmbedding)  # default model is e5-instruct


def test_factory_falls_back_when_provider_fails(fake_st):
    # Jina without a key fails → SentenceTransformer fallback.
    model = EmbeddingProviderFactory.create_embedding_model("jina", "m")
    assert isinstance(model, SentenceTransformerEmbedding)


def test_factory_openai_without_key_falls_back(fake_st):
    model = EmbeddingProviderFactory.create_embedding_model("openai", "text-embedding-3-small")
    assert isinstance(model, SentenceTransformerEmbedding)


def test_factory_raises_clear_error_when_fallback_also_fails(monkeypatch):
    module = types.ModuleType("sentence_transformers")

    def boom(name):
        raise OSError("no model")

    module.SentenceTransformer = boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(EmbeddingInitializationError):
        EmbeddingProviderFactory.create_embedding_model("jina", "m")


def test_factory_register_custom_provider():
    sentinel = object()
    EmbeddingProviderFactory.register("custom", lambda model, opts: sentinel)
    assert EmbeddingProviderFactory.create_embedding_model("CUSTOM", "x") is sentinel
