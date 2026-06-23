import logging
import asyncio
from typing import List
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class GroqEmbedding(BaseEmbedding):
    """
    Custom embedding class that calls Groq's OpenAI-compatible embeddings API
    directly via httpx — bypasses llama_index OpenAIEmbedding model-name validation
    which rejects non-OpenAI model names like 'llama-text-embed-v2'.

    Model: llama-text-embed-v2
    Endpoint: https://api.groq.com/openai/v1/embeddings
    Dimension: 2048
    Arabic support: ✅ excellent (multilingual)
    Cold start: instant (API call, no download)
    """

    # Pydantic fields for serialization
    _api_key:   str
    _model:     str
    _api_base:  str
    _batch_size: int

    def __init__(self, api_key: str, model: str = "llama-text-embed-v2",
                 api_base: str = "https://api.groq.com/openai/v1",
                 embed_batch_size: int = 20, **kwargs):
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        object.__setattr__(self, "_api_key",    api_key)
        object.__setattr__(self, "_model",      model)
        object.__setattr__(self, "_api_base",   api_base.rstrip("/"))
        object.__setattr__(self, "_batch_size", embed_batch_size)

    @classmethod
    def class_name(cls) -> str:
        return "GroqEmbedding"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Synchronous call to Groq embeddings endpoint."""
        import httpx
        url = f"{self._api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        results = []
        # Batch requests to respect API limits
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            payload = {"model": self._model, "input": batch}
            response = httpx.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Groq returns data sorted by index
            batch_embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            results.extend(batch_embeddings)
        return results

    # ── llama_index required interface ────────────────────────────────────────
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)


class EmbeddingProviderFactory:
    """
    Factory class for creating embedding providers.
    Supports Groq (default), OpenAI, and HuggingFace.
    """

    @staticmethod
    def create_embedding_model(provider: str, model_name: str, **kwargs) -> BaseEmbedding:
        provider = (provider or "groq").lower().strip()

        try:
            # ── Groq: free, multilingual, zero cold-start ─────────────────────
            if provider == "groq":
                api_key = kwargs.get("groq_api_key") or kwargs.get("api_key")
                if not api_key:
                    raise ValueError("GROQ_API_KEY is required for Groq embeddings.")
                logger.info(f"[Embeddings] Using Groq embeddings: {model_name} (dim=2048)")
                return GroqEmbedding(
                    api_key=api_key,
                    model=model_name,
                    embed_batch_size=kwargs.get("embed_batch_size", 20),
                )

            # ── OpenAI ────────────────────────────────────────────────────────
            elif provider == "openai":
                api_key = kwargs.get("api_key")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
                from llama_index.embeddings.openai import OpenAIEmbedding
                logger.info(f"[Embeddings] Using OpenAI embeddings: {model_name}")
                return OpenAIEmbedding(
                    model=model_name,
                    api_key=api_key,
                    embed_batch_size=kwargs.get("embed_batch_size", 20),
                )

            # ── HuggingFace ───────────────────────────────────────────────────
            else:
                try:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    logger.info(f"[Embeddings] Using HuggingFace embeddings: {model_name}")
                    return HuggingFaceEmbedding(
                        model_name=model_name,
                        embed_batch_size=kwargs.get("embed_batch_size", 20),
                    )
                except ImportError:
                    logger.warning("HuggingFaceEmbedding not found — falling back to SentenceTransformer.")
                    raise ValueError("Missing HuggingFaceEmbedding module")

        except Exception as e:
            logger.error(f"Failed to initialize embedding provider '{provider}': {e}")
            logger.warning("Falling back to multilingual SentenceTransformer (BAAI/bge-m3).")

            from sentence_transformers import SentenceTransformer

            class LocalEmbedding(BaseEmbedding):
                def __init__(self, model_name: str = "BAAI/bge-m3", **kw):
                    super().__init__(**kw)
                    object.__setattr__(self, "_model", SentenceTransformer(model_name))

                @classmethod
                def class_name(cls) -> str:
                    return "LocalEmbedding"

                def _get_query_embedding(self, query: str) -> List[float]:
                    return self._model.encode(query).tolist()

                def _get_text_embedding(self, text: str) -> List[float]:
                    return self._model.encode(text).tolist()

                def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                    return self._model.encode(texts).tolist()

                async def _aget_query_embedding(self, query: str) -> List[float]:
                    return self._get_query_embedding(query)

                async def _aget_text_embedding(self, text: str) -> List[float]:
                    return self._get_text_embedding(text)

            return LocalEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")
