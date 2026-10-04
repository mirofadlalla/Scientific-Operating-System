"""Embedding providers for the RAG pipeline.

Public API (backward compatible with the former single-file ``embeddings.py``)::

    from app.agents.customer_support.rag.embeddings import EmbeddingProviderFactory
"""

from .base import EmbeddingInitializationError, ExecutorEmbedding, is_e5_instruct
from .factory import EmbeddingProviderFactory
from .providers import E5InstructEmbedding, JinaEmbedding, make_sentence_transformer

# Legacy private alias kept for any external importer of the old module.
_make_sentence_transformer = make_sentence_transformer
_is_e5_instruct = is_e5_instruct

__all__ = [
    "E5InstructEmbedding",
    "EmbeddingInitializationError",
    "EmbeddingProviderFactory",
    "ExecutorEmbedding",
    "JinaEmbedding",
    "is_e5_instruct",
    "make_sentence_transformer",
]
