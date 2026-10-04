import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Groq API Keys (separate per service for quota/billing control) ─────────
    # Primary key — used as fallback for any service that doesn't have its own key
    GROQ_API_KEY: str     = os.getenv("GROQ_API_KEY",     "")  # REQUIRED

    # Dedicated key for the LLM (reasoning, routing, orchestration)
    # Falls back to GROQ_API_KEY if not set
    GROQ_LLM_API_KEY: str = os.getenv("GROQ_LLM_API_KEY", "")

    # Dedicated key for STT (Whisper transcription)
    # Falls back to GROQ_API_KEY if not set
    GROQ_STT_API_KEY: str = os.getenv("GROQ_STT_API_KEY", "")

    # Dedicated key for TTS (text-to-speech via Groq / PlayAI)
    # Falls back to GROQ_API_KEY if not set
    GROQ_TTS_API_KEY: str = os.getenv("GROQ_TTS_API_KEY", "")

    # ── Groq Endpoints & Models ────────────────────────────────────────────────
    GROQ_BASE_URL: str           = os.getenv("GROQ_BASE_URL",           "https://api.groq.com/openai/v1")
    GROQ_WHISPER_MODEL: str      = os.getenv("GROQ_WHISPER_MODEL",      "whisper-large-v3-turbo")
    GROQ_TTS_MODEL: str          = os.getenv("GROQ_TTS_MODEL",          "playai-tts")
    GROQ_TTS_VOICE: str          = os.getenv("GROQ_TTS_VOICE",          "Fritz-PlayAI")
    GROQ_TTS_MODEL_ARABIC: str   = os.getenv("GROQ_TTS_MODEL_ARABIC",   "canopylabs/orpheus-arabic-saudi")
    GROQ_TTS_MODEL_ENGLISH: str  = os.getenv("GROQ_TTS_MODEL_ENGLISH",  "canopylabs/orpheus-v1-english")
    OPENAI_TTS_MODEL: str        = os.getenv("OPENAI_TTS_MODEL",        "tts-1")


    # ── Embeddings Configuration ───────────────────────────────────────────────
    # Supported providers: "huggingface" | "openai" | "jina"
    EMBEDDING_PROVIDER: str   = os.getenv("EMBEDDING_PROVIDER",   "huggingface")
    EMBEDDING_MODEL: str      = os.getenv("EMBEDDING_MODEL",      "intfloat/multilingual-e5-large-instruct")
    EMBEDDING_FALLBACK_MODEL: str = os.getenv("EMBEDDING_FALLBACK_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    # Jina AI Embeddings (used when EMBEDDING_PROVIDER=jina)
    # Get your key at: https://jina.ai/embeddings/
    JINA_API_KEY: str         = os.getenv("JINA_API_KEY",         "")
    JINA_EMBEDDING_MODEL: str = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
    JINA_API_URL: str         = os.getenv("JINA_API_URL",         "https://api.jina.ai/v1/embeddings")

    # ── Optional Providers ─────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # ── External Microservice URLs ─────────────────────────────────────────────
    ADMET_AI_URL: str           = os.getenv("ADMET_AI_URL",           "https://shdwRow-ailixir-admet.hf.space")
    CHEMICAL_AI_URL: str        = os.getenv("CHEMICAL_AI_URL",        "https://RottenShadow-ailixir-chemical-rag.hf.space")
    DRUG_REPURPOSING_URL: str   = os.getenv("DRUG_REPURPOSING_URL",   "https://RottenShadow-ailixir-drug-repurposing.hf.space")
    GENERATION_SERVICE_URL: str = os.getenv("GENERATION_SERVICE_URL", "https://shdwRow-ailixir-generation.hf.space")

    # Chemical MCP server URL (set to external URL when START_MCP_SERVER=0)
    CHEMICAL_MCP_URL: str = os.getenv("CHEMICAL_MCP_URL", "http://localhost:8001/mcp")
    MCP_SERVER_PORT: int  = int(os.getenv("MCP_SERVER_PORT", "8001"))

    # ── LLM Model Selection ────────────────────────────────────────────────────
    ROUTING_MODEL: str      = os.getenv("ROUTING_MODEL",      "openai/gpt-oss-20b")    # fast routing
    REASONING_MODEL: str    = os.getenv("REASONING_MODEL",    "openai/gpt-oss-120b")   # heavy reasoning
    ORCHESTRATOR_MODEL: str = os.getenv("ORCHESTRATOR_MODEL", "openai/gpt-oss-120b")   # backward-compat alias
    QWEN_MODEL: str         = os.getenv("QWEN_MODEL",         "qwen/qwen3-32b")
    QWEN_API_BASE: str      = os.getenv("QWEN_API_BASE",      "")
    QWEN_API_KEY: str       = os.getenv("QWEN_API_KEY",       "")

    # ── RAG / Vector Search ────────────────────────────────────────────────────
    RAG_INDEX_NAME: str = os.getenv("RAG_INDEX_NAME", "AilixirDocs")  # Weaviate collection
    RAG_TOP_K: int      = int(os.getenv("RAG_TOP_K",  "5"))           # chunks per query
    RAG_ALPHA: float    = float(os.getenv("RAG_ALPHA", "0.5"))        # 0=BM25, 1=vector

    # ── Redis Configuration ────────────────────────────────────────────────────
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int   = int(os.getenv("REDIS_DB",   "0"))

    # ── Weaviate Vector DB ─────────────────────────────────────────────────────
    WEAVIATE_HOST: str      = os.getenv("WEAVIATE_HOST",      "localhost")
    WEAVIATE_PORT: int      = int(os.getenv("WEAVIATE_PORT",      "8080"))
    WEAVIATE_GRPC_PORT: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    # ── MongoDB Atlas ──────────────────────────────────────────────────────────
    # MONGODB_URI = mongodb+srv://user:pass@cluster.mongodb.net
    MONGODB_URI: str     = os.getenv("MONGODB_URI",     "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "ailixir")

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins. Use "*" for dev only.
    # Production example: https://my-app.vercel.app,https://staging.my-app.vercel.app
    CORS_ORIGINS: str  = os.getenv("CORS_ORIGINS",  "*")
    FRONTEND_URL: str  = os.getenv("FRONTEND_URL",  "https://scientific-operating-system.vercel.app")

    # ── Voice Pipeline ─────────────────────────────────────────────────────────
    VOICE_DEBUG: bool = os.getenv("VOICE_DEBUG", "").lower() in ("1", "true", "yes")

    # ── JWT ────────────────────────────────────────────────────────────────────
    # Generate: python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY: str                  = os.getenv("SECRET_KEY",                  "change-me-in-production")
    JWT_ALGORITHM: str               = os.getenv("JWT_ALGORITHM",               "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    class Config:
        env_file = ".env"
        extra = "ignore"  # ignore unknown env vars


settings = Settings()


# ── Resolved per-service Groq API keys ────────────────────────────────────────
# Callers should use these helpers rather than GROQ_API_KEY directly so that
# service-specific keys are honoured when set.

def groq_llm_key() -> str:
    """Return the Groq key to use for LLM calls (GROQ_LLM_API_KEY → GROQ_API_KEY)."""
    return settings.GROQ_LLM_API_KEY or settings.GROQ_API_KEY


def groq_stt_key() -> str:
    """Return the Groq key to use for STT/Whisper calls (GROQ_STT_API_KEY → GROQ_API_KEY)."""
    return settings.GROQ_STT_API_KEY or settings.GROQ_API_KEY


def groq_tts_key() -> str:
    """Return the Groq key to use for TTS calls (GROQ_TTS_API_KEY → GROQ_API_KEY)."""
    return settings.GROQ_TTS_API_KEY or settings.GROQ_API_KEY