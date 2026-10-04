"""Centralised RAG configuration.

Values come from the host application's settings when running inside the full
app, and from the process environment when this package is used standalone
(e.g. python -m app.agents.customer_support.rag.cli). The public module
level names (GROQ_API_KEY, EMBED_MODEL …) are unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Callable

try:  # running inside the full Scientific OS app
    from app.config import settings as _app_settings
except ImportError:  # pragma: no cover - standalone usage
    _app_settings = None
    try:  # the old standalone mode read .env; keep doing so
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _setting(name: str, default: Any, cast: Callable[[Any], Any] = str) -> Any:
    """Read name from app settings, else from the environment, else default."""
    if _app_settings is not None:
        return getattr(_app_settings, name, default)
    return cast(os.getenv(name, default))


GROQ_API_KEY: str = _setting("GROQ_API_KEY", "")
GROQ_BASE_URL: str = _setting("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
ORCHESTRATOR_MODEL: str = (
    getattr(_app_settings, "REASONING_MODEL", _app_settings.ORCHESTRATOR_MODEL)
    if _app_settings is not None
    else os.getenv("REASONING_MODEL", "openai/gpt-oss-120b")
)
WEAVIATE_HOST: str = _setting("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT: int = _setting("WEAVIATE_PORT", 8080, int)
WEAVIATE_GRPC_PORT: int = _setting("WEAVIATE_GRPC_PORT", 50051, int)
EMBEDDING_PROVIDER: str = _setting("EMBEDDING_PROVIDER", "huggingface")
EMBED_MODEL: str = _setting("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
OPENAI_API_KEY: str = _setting("OPENAI_API_KEY", "")
JINA_API_KEY: str = _setting("JINA_API_KEY", "")
JINA_EMBEDDING_MODEL: str = _setting("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
RAG_INDEX_NAME: str = _setting("RAG_INDEX_NAME", "AilixirDocs")


# ── Embedding dimensions by model ────────────────────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2 → 384 (default)
# BAAI/bge-m3                           → 1024
# text-embedding-3-small (OpenAI)       → 1536
# jina-embeddings-* (Jina)              → 1024
_dim_model = JINA_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "jina" else EMBED_MODEL
EMBED_DIM: int = (
    1024 if "e5-large" in _dim_model else
    1024 if "bge-m3" in _dim_model else
    1536 if "text-embedding-3" in _dim_model else
    768 if "bge-small" in _dim_model else
    1024 if "jina-embeddings" in _dim_model else
    384  # MiniLM and small models
)
