"""Configure llama-index globals (embedding model + LLM) once per process."""

from __future__ import annotations

import logging
from typing import Dict

from . import config

logger = logging.getLogger(__name__)


def configure_llama_index(state: Dict[str, bool] | None = None) -> None:
    """Initialise Settings.embed_model then Settings.llm.

    Args:
        state: optional readiness dict; embedding_initialized and
            llm_initialized are flipped as each step succeeds.

    Raises:
        ValueError: if GROQ_API_KEY is missing.
    """
    from llama_index.core import Settings
    from llama_index.llms.groq import Groq

    from app.config import groq_llm_key

    from .embeddings import EmbeddingProviderFactory

    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")

    # 1. Embeddings first (pass groq_api_key so the Groq → HF remap path works)
    Settings.embed_model = EmbeddingProviderFactory.create_embedding_model(
        provider=config.EMBEDDING_PROVIDER,
        model_name=config.EMBED_MODEL,
        api_key=config.OPENAI_API_KEY,
        groq_api_key=config.GROQ_API_KEY,
        jina_api_key=config.JINA_API_KEY,
        jina_model=config.JINA_EMBEDDING_MODEL,
    )
    if state is not None:
        state["embedding_initialized"] = True

    # 2. LLM — uses GROQ_LLM_API_KEY → GROQ_API_KEY
    Settings.llm = Groq(model=config.ORCHESTRATOR_MODEL, api_key=groq_llm_key())
    if state is not None:
        state["llm_initialized"] = True

    logger.info("✅ LLM  → Groq / %s", config.ORCHESTRATOR_MODEL)
    logger.info("✅ Embed → %s / %s", config.EMBEDDING_PROVIDER.capitalize(), config.EMBED_MODEL)
