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
    # huggingface (default): paraphrase-multilingual-MiniLM-L12-v2 — Arabic ✅, 117MB, ~30s cold start
    # openai: text-embedding-3-small — needs OPENAI_API_KEY
    # NOTE: "groq" is remapped to huggingface (Groq has no embeddings API)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL: str    = os.getenv("EMBEDDING_MODEL",    "intfloat/multilingual-e5-large-instruct")

    # storage_dir: str = os.getenv("STORAGE_DIR", "storage")

    # Optional: OpenAI for high-quality TTS (alloy, nova, shimmer…)
    # If not set, the browser's SpeechSynthesis API is used instead
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

    class Config:
        env_file = ".env"

settings = Settings()