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

        class Config:
            env_file = ".env"

    _standalone = _StandaloneSettings()
    GROQ_API_KEY       = _standalone.GROQ_API_KEY
    GROQ_BASE_URL      = _standalone.GROQ_BASE_URL
    ORCHESTRATOR_MODEL = _standalone.ORCHESTRATOR_MODEL
    WEAVIATE_HOST      = _standalone.WEAVIATE_HOST
    WEAVIATE_PORT      = _standalone.WEAVIATE_PORT
    WEAVIATE_GRPC_PORT = _standalone.WEAVIATE_GRPC_PORT


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model constants
# ─────────────────────────────────────────────────────────────────────────────
# Using text-embedding-3-small via OpenAI-compatible endpoint (valid for OpenAIEmbedding)
EMBED_MODEL = "text-embedding-3-small"  # 1536-dim, supported by OpenAIEmbedding
EMBED_DIM   = 1536


def setup_groq_environment() -> None:
    """
    Configure LlamaIndex global defaults:
      • LLM  → Groq llama-3.3-70b-versatile
      • Embed → OpenAI-compatible text-embedding-3-small

    Both services use GROQ_API_KEY where applicable — no local model downloads needed.
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

    # 2. Configure embeddings via OpenAI API (or compatible endpoint)
    #    Using text-embedding-3-small which is a valid OpenAIEmbedding model
    embed_model = OpenAIEmbedding(
        model=EMBED_MODEL,
        api_key=GROQ_API_KEY,
        api_base=GROQ_BASE_URL,          # Use Groq's OpenAI-compatible endpoint
        embed_batch_size=20,             # Recommended batch size
    )

    # 3. Set global defaults for LlamaIndex
    Settings.llm         = llm
    Settings.embed_model = embed_model

    print(f"✅ LLM  → Groq / {ORCHESTRATOR_MODEL}")
    print(f"✅ Embed → OpenAI-compatible / {EMBED_MODEL}  ({EMBED_DIM}-dim)")