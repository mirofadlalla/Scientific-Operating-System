"""HuggingFace provider: custom e5-instruct wrapper or the stock llama-index one."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from llama_index.core.embeddings import BaseEmbedding

from ..base import is_e5_instruct
from .e5 import E5InstructEmbedding

logger = logging.getLogger(__name__)


def build_huggingface(model_name: str, options: Mapping[str, Any]) -> BaseEmbedding:
    """Pick the right HuggingFace wrapper for ``model_name``."""
    # e5-instruct family → custom wrapper that adds the query prefix
    if is_e5_instruct(model_name):
        logger.info("[Embeddings] e5-instruct mode (with query prefix): %s", model_name)
        return E5InstructEmbedding(
            model_name=model_name,
            embed_batch_size=options.get("embed_batch_size", 12),
        )

    # All other HF models → standard llama_index wrapper
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError:
        logger.warning("[Embeddings] llama-index-embeddings-huggingface not installed.")
        raise

    logger.info("[Embeddings] HuggingFace: %s", model_name)
    return HuggingFaceEmbedding(
        model_name=model_name,
        embed_batch_size=options.get("embed_batch_size", 20),
    )
