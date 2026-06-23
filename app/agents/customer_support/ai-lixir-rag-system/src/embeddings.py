import logging
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class EmbeddingProviderFactory:
    """
    Factory class for creating embedding providers.
    Supports OpenAI, HuggingFace, and fallback mechanisms.
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
                return HuggingFaceEmbedding(
                    model_name=model_name,
                    embed_batch_size=kwargs.get("embed_batch_size", 20)
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider '{provider}': {e}")
            logger.warning("Falling back to robust multilingual in-memory HuggingFaceEmbedding (BAAI/bge-m3).")
            # Safe fallback for when the desired provider fails
            return HuggingFaceEmbedding(model_name="BAAI/bge-m3")
