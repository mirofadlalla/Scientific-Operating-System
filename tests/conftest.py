"""
Pytest configuration and shared fixtures for Scientific OS tests.
This file sets up environment variables and mocks before app initialization.
"""

import os
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Set environment variables BEFORE app imports (pytest collection phase)
# This ensures the app can initialize without real API keys
os.environ.setdefault("GROQ_API_KEY", "test-api-key-for-testing")
os.environ.setdefault("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
os.environ.setdefault("ADMET_AI_URL", "https://test-admet.example.com")
os.environ.setdefault("CHEMICAL_AI_URL", "https://test-chemical.example.com")
os.environ.setdefault("DRUG_REPURPOSING_URL", "https://test-drug-repurposing.example.com")
os.environ.setdefault("GENERATION_SERVICE_URL", "https://test-generation.example.com")
