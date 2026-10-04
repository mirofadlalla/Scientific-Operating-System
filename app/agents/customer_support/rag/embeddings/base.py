"""Shared building blocks for embedding providers.

ExecutorEmbedding removes the run_in_executor boilerplate that every
synchronous provider would otherwise duplicate: a provider only implements the
three synchronous _get_* hooks and inherits the async variants.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Mapping, TypeVar

from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

T = TypeVar("T")

EmbeddingBuilder = Callable[[str, Mapping[str, Any]], BaseEmbedding]
"""Signature of a provider builder: (model_name, options) -> BaseEmbedding."""


class EmbeddingInitializationError(RuntimeError):
    """Raised when neither the requested provider nor the fallback can be loaded."""


def is_e5_instruct(model_name: str) -> bool:
    """Return True for the intfloat/multilingual-e5-*-instruct family."""
    lowered = model_name.lower()
    return "e5" in lowered and "instruct" in lowered


async def run_blocking(func: Callable[..., T], *args: Any) -> T:
    """Run a blocking callable in the default executor without blocking the loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


class ExecutorEmbedding(BaseEmbedding):
    """BaseEmbedding whose async methods delegate to the sync ones in a thread."""

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await run_blocking(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await run_blocking(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await run_blocking(self._get_text_embeddings, texts)
