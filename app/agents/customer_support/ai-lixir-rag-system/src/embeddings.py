import logging
from typing import List
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

class EmbeddingProviderFactory:
    """
    Factory class for creating embedding providers.
    Supports OpenAI, HuggingFace, and robust local fallbacks.
    """
    
    @staticmethod
    def create_embedding_model(provider: str, model_name: str, **kwargs) -> BaseEmbedding:
        # Default to huggingface if empty
        provider = (provider or "huggingface").lower().strip()
        
        try:
            if provider == "openai":
                api_key = kwargs.get("api_key")
                if not api_key:
                    raise ValueError("OpenAI API key is required for OpenAI embeddings.")
                # Safe lazy import ONLY
                from llama_index.embeddings.openai import OpenAIEmbedding
                return OpenAIEmbedding(
                    model=model_name,
                    api_key=api_key,
                    embed_batch_size=kwargs.get("embed_batch_size", 20)
                )
                
            else:
                # Enforce huggingface as default for all non-openai requests
                try:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    return HuggingFaceEmbedding(
                        model_name=model_name,
                        embed_batch_size=kwargs.get("embed_batch_size", 20)
                    )
                except ImportError:
                    logger.warning("llama_index.embeddings.huggingface not found. Falling back to SentenceTransformer.")
                    raise ValueError("Missing HuggingFaceEmbedding module")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider '{provider}': {e}")
            logger.warning("Falling back to robust multilingual local embedding (SentenceTransformer).")
            
            from sentence_transformers import SentenceTransformer
            
            class LocalEmbedding(BaseEmbedding):
                def __init__(self, model_name: str = "BAAI/bge-m3", **kw):
                    super().__init__(**kw)
                    # Bypass Pydantic validation for the raw model
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

            return LocalEmbedding(model_name="BAAI/bge-m3")
