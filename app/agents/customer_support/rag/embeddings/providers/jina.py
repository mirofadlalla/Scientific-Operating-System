"""Jina AI REST-API embedding provider."""

from __future__ import annotations

import logging
from typing import Any, List

import requests

from ..base import ExecutorEmbedding
from ..constants import JINA_API_URL, JINA_DEFAULT_MODEL, JINA_REQUEST_TIMEOUT_S

logger = logging.getLogger(__name__)

_TASK_QUERY = "retrieval.query"
_TASK_PASSAGE = "retrieval.passage"


class JinaEmbedding(ExecutorEmbedding):
    """Calls the Jina embeddings endpoint with proper task hints.

    * queries   → ``task="retrieval.query"``
    * documents → ``task="retrieval.passage"``

    Requires ``JINA_API_KEY``. See https://jina.ai/embeddings/ for models.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = JINA_DEFAULT_MODEL,
        embed_batch_size: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        if not api_key:
            raise ValueError(
                "JINA_API_KEY is required for JinaEmbedding. "
                "Get yours at https://jina.ai/embeddings/"
            )
        object.__setattr__(self, "_api_key", api_key)
        object.__setattr__(self, "_model_name", model_name)
        logger.info("[Embeddings] Jina AI: %s", model_name)

    @classmethod
    def class_name(cls) -> str:
        return "JinaEmbedding"

    def _call_api(self, texts: List[str], task: str) -> List[List[float]]:
        """POST ``texts`` to Jina and return embeddings in input order."""
        payload = {
            "model": self._model_name,
            "task": task,
            "normalized": True,
            "input": texts,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(
            JINA_API_URL, headers=headers, json=payload, timeout=JINA_REQUEST_TIMEOUT_S
        )
        response.raise_for_status()
        data = response.json()
        # Sort by index to guarantee order matches input
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    # ── llama_index interface ────────────────────────────────────────────────
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._call_api([query], task=_TASK_QUERY)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._call_api([text], task=_TASK_PASSAGE)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts, task=_TASK_PASSAGE)


def build_jina(model_name: str, options: Any) -> JinaEmbedding:
    """Builder used by the factory registry."""
    return JinaEmbedding(
        api_key=options.get("jina_api_key") or options.get("api_key", ""),
        model_name=options.get("jina_model", model_name),
        embed_batch_size=options.get("embed_batch_size", 16),
    )
