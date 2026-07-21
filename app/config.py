import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Core API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")  # REQUIRED

    # Groq endpoints
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Groq Whisper model for STT (no OpenAI key needed)
    GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"
    
    # Embeddings Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL: str    = os.getenv("EMBEDDING_MODEL",    "intfloat/multilingual-e5-large-instruct")

    # Optional: OpenAI for high-quality TTS
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # External Service URLs
    ADMET_AI_URL: str = os.getenv("ADMET_AI_URL", "https://shdwRow-ailixir-admet.hf.space")
    CHEMICAL_AI_URL: str = os.getenv("CHEMICAL_AI_URL", "https://RottenShadow-ailixir-chemical-rag.hf.space")
    DRUG_REPURPOSING_URL: str = os.getenv("DRUG_REPURPOSING_URL", "https://RottenShadow-ailixir-drug-repurposing.hf.space")
    GENERATION_SERVICE_URL: str = os.getenv("GENERATION_SERVICE_URL", "https://shdwRow-ailixir-generation.hf.space")

    # Available Models
    ORCHESTRATOR_MODEL: str = "llama-3.3-70b-versatile"
    QWEN_MODEL: str = "qwen/qwen3-32b"

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Weaviate Vector DB Configuration
    WEAVIATE_HOST: str = os.getenv("WEAVIATE_HOST", "localhost")
    WEAVIATE_PORT: int = int(os.getenv("WEAVIATE_PORT", "8080"))
    WEAVIATE_GRPC_PORT: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    # ── MongoDB Atlas ─────────────────────────────────────────────────────────
    # Set these in HF Space secrets / .env:
    #   MONGODB_URI      = mongodb+srv://user:pass@cluster.mongodb.net
    #   MONGODB_DB_NAME  = ailixir
    MONGODB_URI: str     = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "ailixir")

    # ── JWT ───────────────────────────────────────────────────────────────────
    # Used to sign access tokens issued after login.
    # Generate a strong random value: python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY: str       = os.getenv("SECRET_KEY", "change-me-in-production")
    JWT_ALGORITHM: str    = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # tokens valid for 1 hour

    class Config:
        env_file = ".env"

settings = Settings()