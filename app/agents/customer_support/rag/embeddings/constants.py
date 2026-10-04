"""Environment-driven defaults shared by every embedding provider."""

from __future__ import annotations

import os

DEFAULT_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
"""Model used for the HuggingFace provider when none is requested explicitly."""

FALLBACK_MODEL: str = os.getenv("EMBEDDING_FALLBACK_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
"""Small local model used as the last-resort fallback."""

E5_TASK: str = "Given a user question, retrieve relevant document passages that answer the question"
"""Task description for the e5-instruct query prefix (improves Arabic RAG retrieval)."""

JINA_DEFAULT_MODEL: str = "jina-embeddings-v5-text-small"
JINA_API_URL: str = os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
"""Env-overridable endpoint (useful for proxies/staging)."""

JINA_REQUEST_TIMEOUT_S: float = 60.0
