import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Core API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")  # REQUIRED

    # Groq endpoints
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Groq Whisper model for STT (no OpenAI key needed)
    GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"
    
    GROQ_EMBEDDING_MODEL: str = "bge-large-en-v1.5"

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

    class Config:
        env_file = ".env"

settings = Settings()