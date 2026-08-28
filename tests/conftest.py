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
os.environ.setdefault("GROQ_API_KEY", "gsk_test_key_for_unit_tests_12345")
os.environ.setdefault("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unit-tests-strictly-32b")
os.environ.setdefault("ADMET_AI_URL", "https://test-admet.example.com")
os.environ.setdefault("CHEMICAL_AI_URL", "https://test-chemical.example.com")
os.environ.setdefault("DRUG_REPURPOSING_URL", "https://test-drug-repurposing.example.com")
os.environ.setdefault("GENERATION_SERVICE_URL", "https://test-generation.example.com")
os.environ.setdefault("MONGODB_URI", "")

from app.agents.customer_support.agent import rag_state

@pytest.fixture(autouse=True)
def mark_rag_ready():
    """Ensure rag_state['ready'] is True during tests so ReadinessMiddleware allows all requests."""
    rag_state["ready"] = True
    yield
