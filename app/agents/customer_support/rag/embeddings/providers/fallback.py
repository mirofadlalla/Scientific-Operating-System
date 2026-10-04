"""Last-resort local embeddings used when the requested provider fails."""

from __future__ import annotations

import logging
from typing import Any, List

from ..base import ExecutorEmbedding

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(ExecutorEmbedding):
    """Plain SentenceTransformer wrapper (no query/document prefixes)."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from sentence_transformers import SentenceTransformer

        object.__setattr__(self, "_st", SentenceTransformer(model_name))

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._st.encode(query, normalize_embeddings=True).tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._st.encode(text, normalize_embeddings=True).tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._st.encode(texts, normalize_embeddings=True).tolist()


def make_sentence_transformer(model_name: str) -> SentenceTransformerEmbedding:
    """Build the fallback embedding model."""
    logger.info("[Embeddings] SentenceTransformer fallback: %s", model_name)
    return SentenceTransformerEmbedding(model_name)
