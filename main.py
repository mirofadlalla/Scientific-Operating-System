# Root entry point for Hugging Face Spaces Docker deployment
# This file re-exports the FastAPI app from the main module
from app.main import app

__all__ = ["app"]
