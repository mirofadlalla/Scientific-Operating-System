import os

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
    EMBED_MODEL        = getattr(app_settings, "EMBEDDING_MODEL", "BAAI/bge-m3")
    OPENAI_API_KEY     = getattr(app_settings, "OPENAI_API_KEY", "")
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
        EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
        OPENAI_API_KEY:     str = os.getenv("OPENAI_API_KEY", "")

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
    OPENAI_API_KEY     = _standalone.OPENAI_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model constants
# ─────────────────────────────────────────────────────────────────────────────
# Embedding dimensions by model
# paraphrase-multilingual-MiniLM-L12-v2 → 384 (default)
# BAAI/bge-m3                           → 1024
# text-embedding-3-small (OpenAI)       → 1536
EMBED_DIM = (
    1024 if "e5-large"            in EMBED_MODEL else
    1024 if "bge-m3"              in EMBED_MODEL else
    1536 if "text-embedding-3"    in EMBED_MODEL else
    768  if "bge-small"           in EMBED_MODEL else
    384   # MiniLM and small models
)