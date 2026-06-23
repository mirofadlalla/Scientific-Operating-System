import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.groq import GroqEmbedding
from llama_index.core import Settings

from app.config import settings

# Load environment variables from .env if needed
# load_dotenv()

def setup_groq_environment():
    """
    Configure LlamaIndex to use Groq for both
    text generation (LLM) and embeddings.
    """
    
    if not settings.GROQ_API_KEY:
        raise ValueError("Please set GROQ_API_KEY in your .env file.")

    # Configure the default LLM
    llm = Groq(
        model=settings.ORCHESTRATOR_MODEL,
        api_key=settings.GROQ_API_KEY
    )

    # Configure the default embedding model
    embed_model = GroqEmbedding(
        model_name=settings.GROQ_EMBEDDING_MODEL,
        api_key=settings.GROQ_API_KEY
    )

    # Set global defaults for LlamaIndex
    Settings.llm = llm
    Settings.embed_model = embed_model

    print("✅ Groq LLM and Embedding models configured successfully.")