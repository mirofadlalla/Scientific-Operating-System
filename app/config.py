import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "your_fallback_key")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    
    # أسماء الموديلات المتاحة عندك
    ORCHESTRATOR_MODEL: str = "llama-3.3-70b-versatile"
    QWEN_MODEL: str = "qwen/qwen3-32b"

    class Config:
        env_file = ".env"

settings = Settings()