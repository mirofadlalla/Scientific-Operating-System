"""Provider dispatcher with isolated last-resort fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict

from llama_index.core.embeddings import BaseEmbedding

from .base import EmbeddingBuilder, EmbeddingInitializationError
from .constants import DEFAULT_MODEL, FALLBACK_MODEL
from .providers import build_huggingface, build_jina, build_openai, make_sentence_transformer

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER = "huggingface"

_BUILDERS: Dict[str, EmbeddingBuilder] = {
    "jina": build_jina,
    "openai": build_openai,
    _DEFAULT_PROVIDER: build_huggingface,
}


class EmbeddingProviderFactory:
    """Creates embedding models; never leaves the caller without a model if avoidable."""

    @staticmethod
    def register(provider: str, builder: EmbeddingBuilder) -> None:
        """Register (or override) a provider builder."""
        _BUILDERS[provider.lower().strip()] = builder

    @staticmethod
    def create_embedding_model(provider: str, model_name: str, **kwargs: Any) -> BaseEmbedding:
        """Build the requested embedding model, or the fallback if that fails.

        Unknown providers are treated as HuggingFace (historical behaviour).
        Raises:
            EmbeddingInitializationError: if the fallback model also fails.
        """
        provider = (provider or _DEFAULT_PROVIDER).lower().strip()

        # Groq has no embeddings API — remap silently
        if provider == "groq":
            logger.warning("[Embeddings] Groq has no embeddings API — switching to HuggingFace.")
            provider, model_name = _DEFAULT_PROVIDER, DEFAULT_MODEL

        builder = _BUILDERS.get(provider, build_huggingface)
        try:
            return builder(model_name, kwargs)
        except Exception as exc:  # noqa: BLE001 - any provider failure triggers fallback
            logger.error("[Embeddings] Failed to load %s/%s: %s", provider, model_name, exc)
            logger.warning("[Embeddings] Falling back to %s.", FALLBACK_MODEL)
            return _load_fallback(exc)


def _load_fallback(original: Exception) -> BaseEmbedding:
    try:
        return make_sentence_transformer(FALLBACK_MODEL)
    except Exception as fallback_exc:
        raise EmbeddingInitializationError(
            f"Fallback model {FALLBACK_MODEL!r} also failed to load: {fallback_exc}"
        ) from original
