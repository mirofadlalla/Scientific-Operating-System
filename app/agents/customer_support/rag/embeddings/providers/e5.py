"""``intfloat/multilingual-e5-*-instruct`` wrapper with correct query prefixes."""

from __future__ import annotations

import logging
from typing import Any, List

from ..base import ExecutorEmbedding
from ..constants import DEFAULT_MODEL, E5_TASK

logger = logging.getLogger(__name__)


class E5InstructEmbedding(ExecutorEmbedding):
    """HuggingFace e5-instruct wrapper that applies the required prefixes.

    Queries   → ``"Instruct: <task>\\nQuery: <text>"``
    Documents → no prefix (plain text)

    Without these prefixes, retrieval quality drops significantly.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        task: str = E5_TASK,
        embed_batch_size: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        from sentence_transformers import SentenceTransformer

        logger.info("[Embeddings] Loading e5-instruct model: %s …", model_name)
        object.__setattr__(self, "_st", SentenceTransformer(model_name))
        object.__setattr__(self, "_task", task)

    @classmethod
    def class_name(cls) -> str:
        return "E5InstructEmbedding"

    def _fmt_query(self, query: str) -> str:
        return f"Instruct: {self._task}\nQuery: {query}"

    # ── llama_index interface ────────────────────────────────────────────────
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._st.encode(self._fmt_query(query), normalize_embeddings=True).tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._st.encode(text, normalize_embeddings=True).tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._st.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()
