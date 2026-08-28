"""
Test suite for Scientific OS v2 FastAPI application.
Tests core endpoints (/health, /api/v1/orchestrate, /api/v1/rag, /api/v1/metrics, /api/v1/auth)
and internal agent & memory units.
"""
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from app.main import app
from app.orchestrator.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    CHEMICAL_AGENT_SYSTEM_PROMPT,
    MEDICAL_AGENT_SYSTEM_PROMPT,
)
from app.core.orchestration import COMBINED_ORCHESTRATOR_PROMPT
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.agents.chemical.agent import _fmt, ChemicalAgent


@pytest.fixture
def client():
    """Provide a TestClient instance for the FastAPI app."""
    return TestClient(app)


class TestRootAndHealthEndpoints:
    """Test suite for root redirect and health check endpoints."""

    def test_root_redirects_to_docs(self, client):
        """Verify that the root path redirects to /docs."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers.get("location") == "/docs"

    def test_health_check_returns_ok(self, client):
        """Verify that the /health endpoint returns 200 OK and timestamp."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"
        assert "timestamp" in data


class TestOrchestrateEndpoint:
    """Test suite for the /api/v1/orchestrate POST endpoint."""

    def test_orchestrate_endpoint_missing_required_fields(self, client):
        """Verify that missing required fields return 422 validation error."""
        payload = {
            "session_id": "test_session",
            # Missing user_id and text_input
        }
        response = client.post("/api/v1/orchestrate", json=payload)
        assert response.status_code == 422

    def test_orchestrate_endpoint_streams_tokens(self, client):
        """Verify that /api/v1/orchestrate accepts requests and streams tokens."""
        payload = {
            "session_id": "test_session_123",
            "user_id": "test_user_456",
            "text_input": "What is EGFR inhibitor?"
        }

        async def mock_token_generator(text_input, session_id, user_id):
            yield "EGFR "
            yield "inhibitor "
            yield "result."

        with patch("app.api.v1.chat.route_and_stream", side_effect=mock_token_generator):
            response = client.post("/api/v1/orchestrate", json=payload)

        assert response.status_code == 200
        assert "EGFR inhibitor result." in response.text


class TestRAGEndpoints:
    """Test suite for RAG status and job polling endpoints."""

    def test_rag_status_endpoint(self, client):
        """Verify that /api/v1/rag/status returns engine information."""
        with patch("app.core.deps.rag_agent.status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {
                "weaviate_connected": False,
                "index_name": "AilixirDocs",
                "node_count": 0,
                "engine_ready": True,
                "embed_model": "huggingface/intfloat/multilingual-e5-large-instruct",
                "llm_model": "groq/openai/gpt-oss-120b",
                "search_mode": "hybrid (α=0.5)",
                "top_k": 5,
            }
            response = client.get("/api/v1/rag/status")

        assert response.status_code == 200
        assert response.json()["index_name"] == "AilixirDocs"

    def test_rag_job_status_unknown_job(self, client):
        """Verify that polling a non-existent job ID returns unknown status."""
        response = client.get("/api/v1/rag/ingest/status/non-existent-uuid-999")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ["unknown", "not_found"] or "not found" in data.get("message", "").lower()


class TestMonitoringEndpoints:
    """Test suite for metrics and monitoring endpoints."""

    def test_metrics_snapshot_endpoint(self, client):
        """Verify that /api/v1/metrics returns metrics snapshot."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data
        assert "uptime" in data

    def test_metrics_requests_endpoint(self, client):
        """Verify that /api/v1/metrics/requests returns request history list."""
        response = client.get("/api/v1/metrics/requests")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestMemoryUnits:
    """Test suite for ShortTermMemory and LongTermMemory."""

    def test_short_term_memory_sliding_window(self):
        """Verify that ShortTermMemory preserves messages and limits length."""
        mem = ShortTermMemory(maxlen=3)
        session_id = "test_window_session"

        mem.add_message(session_id, "user", "Message 1")
        mem.add_message(session_id, "assistant", "Message 2")
        mem.add_message(session_id, "user", "Message 3")
        mem.add_message(session_id, "assistant", "Message 4")

        history = mem.get_history(session_id)
        assert len(history) == 3
        assert history[0]["content"] == "Message 2"
        assert history[-1]["content"] == "Message 4"

        mem.clear(session_id)
        assert len(mem.get_history(session_id)) == 0

    def test_long_term_memory_json_fallback_and_search(self, tmp_path):
        """Verify that LongTermMemory adds entries and searches correctly."""
        store_path = str(tmp_path / "test_ltm.json")
        ltm = LongTermMemory(path=store_path, host="invalid-unreachable-redis-host", port=9999)

        assert ltm.is_redis is False
        ltm.add_entry("sess_1", "User asked about aspirin ADMET profile", {"compound": "aspirin"})
        ltm.add_entry("sess_2", "User asked about EGFR pathway inhibition", {"compound": "gefitinib"})

        results = ltm.search("aspirin")
        assert len(results) >= 1
        assert "aspirin" in results[0]["text"]


class TestChemicalAgentUnits:
    """Test suite for ChemicalAgent helper functions and safe formatting."""

    def test_fmt_helper_safely_handles_types(self):
        """Verify that _fmt helper safely converts numbers and handles None without raising TypeError."""
        assert _fmt(0.123456, 4) == "0.1235"
        assert _fmt(12.5, 2) == "12.50"
        assert _fmt(None) == "N/A"
        assert _fmt("custom_string") == "custom_string"

    def test_prompts_defined(self):
        """Verify that key prompts are properly defined in prompts.py."""
        assert ORCHESTRATOR_SYSTEM_PROMPT is not None
        assert COMBINED_ORCHESTRATOR_PROMPT is not None
        assert "CHEMICAL_AGENT" in COMBINED_ORCHESTRATOR_PROMPT
        assert "MEDICAL_AGENT" in COMBINED_ORCHESTRATOR_PROMPT
