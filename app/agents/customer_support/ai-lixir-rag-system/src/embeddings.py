"""
Embedding provider factory for the AI-lixir RAG system.

Provider options (set via EMBEDDING_PROVIDER env var):
─────────────────────────────────────────────────────
  huggingface (default)
      model: intfloat/multilingual-e5-large-instruct
      dim:   1024  |  size: ~560MB  |  Arabic: ✅ SOTA  |  cold-start: ~2-3 min (once)

      ⚠️  e5-instruct models need task-specific prefixes for best accuracy:
          • queries  → "Instruct: {task}\\nQuery: {text}"
          • passages → no prefix needed

  openai
      model: text-embedding-3-small or text-embedding-3-large
      dim:   1536 / 3072  |  needs OPENAI_API_KEY

NOTE: Groq has no embeddings API — "groq" provider is remapped to huggingface.
"""

import asyncio
import logging
from typing import List

from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

_DEFAULT_MODEL    = "intfloat/multilingual-e5-large-instruct"
_FALLBACK_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"

# Task description for e5-instruct query prefix (improves Arabic RAG retrieval)
_E5_TASK = "Given a user question, retrieve relevant document passages that answer the question"


def _is_e5_instruct(model_name: str) -> bool:
    return "e5" in model_name.lower() and "instruct" in model_name.lower()


class E5InstructEmbedding(BaseEmbedding):
    """
    HuggingFace e5-instruct wrapper that automatically applies the correct
    query/document prefixes required by the intfloat/multilingual-e5-*-instruct family.

    Queries  → "Instruct: <task>\\nQuery: <text>"
    Documents → no prefix (plain text)

    Without these prefixes, retrieval quality drops significantly.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL,
                 task: str = _E5_TASK,
                 embed_batch_size: int = 12, **kwargs):
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        from sentence_transformers import SentenceTransformer
        logger.info(f"[Embeddings] Loading e5-instruct model: {model_name} …")
        object.__setattr__(self, "_st",   SentenceTransformer(model_name))
        object.__setattr__(self, "_task", task)

    @classmethod
    def class_name(cls) -> str:
        return "E5InstructEmbedding"

    def _fmt_query(self, query: str) -> str:
        return f"Instruct: {self._task}\nQuery: {query}"

    # ── llama_index interface ─────────────────────────────────────────────────
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._st.encode(
            self._fmt_query(query), normalize_embeddings=True
        ).tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        # Documents: no prefix
        return self._st.encode(text, normalize_embeddings=True).tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._st.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_text_embeddings, texts)


class EmbeddingProviderFactory:

    @staticmethod
    def create_embedding_model(provider: str, model_name: str, **kwargs) -> BaseEmbedding:
        provider = (provider or "huggingface").lower().strip()

        # Groq has no embeddings API — remap silently
        if provider == "groq":
            logger.warning(
                "[Embeddings] Groq has no embeddings API — switching to HuggingFace."
            )
            provider   = "huggingface"
            model_name = _DEFAULT_MODEL

        try:
            # ── OpenAI ───────────────────────────────────────────────────────
            if provider == "openai":
                api_key = kwargs.get("api_key")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY required for OpenAI embeddings.")
                from llama_index.embeddings.openai import OpenAIEmbedding
                logger.info(f"[Embeddings] OpenAI: {model_name}")
                return OpenAIEmbedding(
                    model=model_name,
                    api_key=api_key,
                    embed_batch_size=kwargs.get("embed_batch_size", 20),
                )

            # ── HuggingFace ───────────────────────────────────────────────────
            else:
                # e5-instruct family → use our custom wrapper with query prefix
                if _is_e5_instruct(model_name):
                    logger.info(f"[Embeddings] e5-instruct mode (with query prefix): {model_name}")
                    return E5InstructEmbedding(
                        model_name=model_name,
                        embed_batch_size=kwargs.get("embed_batch_size", 12),
                    )

                # All other HF models → standard llama_index wrapper
                try:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    logger.info(f"[Embeddings] HuggingFace: {model_name}")
                    return HuggingFaceEmbedding(
                        model_name=model_name,
                        embed_batch_size=kwargs.get("embed_batch_size", 20),
                    )
                except ImportError:
                    logger.warning("[Embeddings] llama-index-embeddings-huggingface not installed.")
                    raise

        except Exception as exc:
            logger.error(f"[Embeddings] Failed to load {provider}/{model_name}: {exc}")
            logger.warning(f"[Embeddings] Falling back to {_FALLBACK_MODEL}.")
            return _make_sentence_transformer(_FALLBACK_MODEL)


def _make_sentence_transformer(model_name: str) -> BaseEmbedding:
    """Plain SentenceTransformer wrapper — last-resort fallback."""
    from sentence_transformers import SentenceTransformer

    class _STEmbedding(BaseEmbedding):
        def __init__(self, mn: str = model_name, **kw):
            super().__init__(**kw)
            object.__setattr__(self, "_st", SentenceTransformer(mn))

        @classmethod
        def class_name(cls) -> str:
            return "SentenceTransformerEmbedding"

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._st.encode(query, normalize_embeddings=True).tolist()

        def _get_text_embedding(self, text: str) -> List[float]:
            return self._st.encode(text, normalize_embeddings=True).tolist()

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            return self._st.encode(texts, normalize_embeddings=True).tolist()

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._get_query_embedding, query)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._get_text_embedding, text)

    logger.info(f"[Embeddings] SentenceTransformer fallback: {model_name}")
    return _STEmbedding(mn=model_name)
