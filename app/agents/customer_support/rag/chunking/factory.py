"""Chunking strategy factory."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import BaseChunkingStrategy
from .strategies import MarkdownStrategy, SentenceStrategy, TokenStrategy

# Markdown takes no tuning parameters, so extra kwargs are deliberately dropped.
_REGISTRY: Dict[str, Callable[..., BaseChunkingStrategy]] = {
    "markdown": lambda **_: MarkdownStrategy(),
    "sentence": SentenceStrategy,
    "token": TokenStrategy,
}


class ChunkingFactory:
    """Instantiates the appropriate chunking strategy."""

    @staticmethod
    def get_strategy(strategy_type: str, **kwargs: Any) -> BaseChunkingStrategy:
        """Return the requested chunking strategy.

        Args:
            strategy_type: ``'markdown'``, ``'sentence'`` or ``'token'``.
            **kwargs: Parameters such as ``chunk_size`` and ``chunk_overlap``.

        Raises:
            ValueError: for an unsupported ``strategy_type``.
        """
        key = strategy_type.strip().lower()
        try:
            builder = _REGISTRY[key]
        except KeyError:
            raise ValueError(f"Unsupported chunking strategy type: {key}") from None
        return builder(**kwargs)
