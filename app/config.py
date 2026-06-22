import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    ADMET_AI_URL: str = os.getenv("ADMET_AI_URL", "https://shdwRow-ailixir-admet.hf.space")
    CHEMICAL_AI_URL: str = os.getenv("CHEMICAL_AI_URL", "https://RottenShadow-ailixir-chemical-rag.hf.space")
    DRUG_REPURPOSING_URL: str = os.getenv("DRUG_REPURPOSING_URL", "https://RottenShadow-ailixir-drug-repurposing.hf.space")
    GENERATION_SERVICE_URL: str = os.getenv("GENERATION_SERVICE_URL", "https://shdwRow-ailixir-generation.hf.space")
    
    # أسماء الموديلات المتاحة 
    ORCHESTRATOR_MODEL: str = "llama-3.3-70b-versatile"
    QWEN_MODEL: str = "qwen/qwen3-32b"

    class Config:
        env_file = ".env"

settings = Settings()