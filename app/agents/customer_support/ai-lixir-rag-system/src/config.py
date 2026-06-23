import os
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# ─────────────────────────────────────────────────────────────────────────────
# Centralised config — reads from the parent app's settings or env directly
# ─────────────────────────────────────────────────────────────────────────────
try:
    # When running inside the full Scientific OS app
    from app.config import settings as app_settings
    GROQ_API_KEY       = app_settings.GROQ_API_KEY
    GROQ_BASE_URL      = app_settings.GROQ_BASE_URL
    ORCHESTRATOR_MODEL = app_settings.ORCHESTRATOR_MODEL
    WEAVIATE_HOST      = getattr(app_settings, "WEAVIATE_HOST", "localhost")
    WEAVIATE_PORT      = getattr(app_settings, "WEAVIATE_PORT", 8080)
    WEAVIATE_GRPC_PORT = getattr(app_settings, "WEAVIATE_GRPC_PORT", 50051)
    EMBEDDING_PROVIDER = getattr(app_settings, "EMBEDDING_PROVIDER", "huggingface")
    EMBED_MODEL        = getattr(app_settings, "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
except ImportError:
    # Standalone mode (e.g., running main.py directly)
    from pydantic_settings import BaseSettings

    class _StandaloneSettings(BaseSettings):
        GROQ_API_KEY:       str = os.getenv("GROQ_API_KEY", "")
        GROQ_BASE_URL:      str = "https://api.groq.com/openai/v1"
        ORCHESTRATOR_MODEL: str = "llama-3.3-70b-versatile"
        WEAVIATE_HOST:      str = os.getenv("WEAVIATE_HOST", "localhost")
        WEAVIATE_PORT:      int = int(os.getenv("WEAVIATE_PORT", "8080"))
        WEAVIATE_GRPC_PORT: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

        class Config:
            env_file = ".env"

    _standalone = _StandaloneSettings()
    GROQ_API_KEY       = _standalone.GROQ_API_KEY
    GROQ_BASE_URL      = _standalone.GROQ_BASE_URL
    ORCHESTRATOR_MODEL = _standalone.ORCHESTRATOR_MODEL
    WEAVIATE_HOST      = _standalone.WEAVIATE_HOST
    WEAVIATE_PORT      = _standalone.WEAVIATE_PORT
    WEAVIATE_GRPC_PORT = _standalone.WEAVIATE_GRPC_PORT
    EMBEDDING_PROVIDER = _standalone.EMBEDDING_PROVIDER
    EMBED_MODEL        = _standalone.EMBEDDING_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model constants
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM   = 1536  # Kept for reference, may vary based on model


def setup_rag_environment() -> None:
    """
    Configure LlamaIndex global defaults:
      • LLM  → Groq llama-3.3-70b-versatile
      • Embed → Provider-agnostic embedding model
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    # 1. Configure the default LLM (Groq)
    llm = Groq(
        model=ORCHESTRATOR_MODEL,
        api_key=GROQ_API_KEY,
    )

    # 2. Configure embeddings via Factory
    from src.embeddings import EmbeddingProviderFactory
    
    try:
        from app.config import settings as app_settings
        openai_key = getattr(app_settings, "OPENAI_API_KEY", "")
    except ImportError:
        openai_key = ""

    embed_model = EmbeddingProviderFactory.create_embedding_model(
        provider=EMBEDDING_PROVIDER,
        model_name=EMBED_MODEL,
        api_key=openai_key,
        embed_batch_size=20
    )

    # 3. Set global defaults for LlamaIndex
    Settings.llm         = llm
    Settings.embed_model = embed_model

    print(f"✅ LLM  → Groq / {ORCHESTRATOR_MODEL}")
    print(f"✅ Embed → {EMBEDDING_PROVIDER.capitalize()} / {EMBED_MODEL}")