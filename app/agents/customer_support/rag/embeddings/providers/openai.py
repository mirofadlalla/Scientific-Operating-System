"""OpenAI embedding integration (thin adapter over llama-index)."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


def build_openai(model_name: str, options: Mapping[str, Any]) -> BaseEmbedding:
    """Create an ``OpenAIEmbedding``; requires ``api_key`` in ``options``."""
    api_key = options.get("api_key")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for OpenAI embeddings.")

    from llama_index.embeddings.openai import OpenAIEmbedding

    logger.info("[Embeddings] OpenAI: %s", model_name)
    return OpenAIEmbedding(
        model=model_name,
        api_key=api_key,
        embed_batch_size=options.get("embed_batch_size", 20),
    )
