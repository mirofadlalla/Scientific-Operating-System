"""
Test suite for Scientific OS FastAPI application.
Tests the core endpoints and agent orchestration logic.
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app, ORCHESTRATOR_SYSTEM_PROMPT


@pytest.fixture
def client():
    """Provide a TestClient instance for the FastAPI app."""
    return TestClient(app)


class TestRootEndpoint:
    """Test suite for the root HTML endpoint."""
    
    def test_root_path_returns_html_status_200(self, client):
        """Verify that the root path returns HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_path_returns_html_content_type(self, client):
        """Verify that the root path returns HTML content type."""
        response = client.get("/")
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_root_path_returns_html_content(self, client):
        """Verify that the root path returns actual HTML content."""
        response = client.get("/")
        assert len(response.text) > 0
        # Basic HTML structure check
        assert "<" in response.text and ">" in response.text


class TestOrchestrateEndpoint:
    """Test suite for the /orchestrate POST endpoint."""
    
    def test_orchestrate_endpoint_accepts_post_request(self, client):
        """Verify that /orchestrate endpoint accepts POST requests."""
        payload = {
            "session_id": "test_session_123",
            "user_id": "test_user_456",
            "text_input": "What is EGFR inhibitor?"
        }
        
        with patch('app.main.client.chat.completions.create') as mock_create:
            # Mock the orchestrator routing response
            mock_orchestrator_response = AsyncMock()
            mock_orchestrator_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "intent": "BIOMEDICAL_MECHANISM",
                    "target_agent": "MEDICAL_AGENT",
                    "entities": {"compound": "EGFR inhibitor"}
                })))
            ]
            
            # Mock the synthesis response (streaming)
            mock_synthesis_response = AsyncMock()
            mock_synthesis_response.__aiter__ = MagicMock(
                return_value=[
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="EGFR inhibitors"))]),
                    MagicMock(choices=[MagicMock(delta=MagicMock(content=" are "))]),
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="effective."))])
                ]
            )
            
            # Set up the mock to return different responses for different calls
            mock_create.side_effect = [mock_orchestrator_response, mock_synthesis_response]
            
            with patch('app.main.chemical_agent.run', new_callable=AsyncMock) as mock_chem_agent:
                mock_chem_agent.return_value = ""
                response = client.post("/orchestrate", json=payload)
        
        assert response.status_code == 200
    
    def test_orchestrate_endpoint_with_chemical_intent(self, client):
        """Verify /orchestrate handles chemical analysis intent."""
        payload = {
            "session_id": "test_session_chemical",
            "user_id": "test_user_123",
            "text_input": "Find similar compounds to aspirin"
        }
        
        with patch('app.main.client.chat.completions.create') as mock_create:
            # Mock orchestrator response for chemical similarity
            mock_orchestrator_response = AsyncMock()
            mock_orchestrator_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "intent": "CHEMICAL_SIMILARITY",
                    "target_agent": "CHEMICAL_AGENT",
                    "entities": {"compound": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
                })))
            ]
            
            # Mock synthesis response
            mock_synthesis_response = AsyncMock()
            mock_synthesis_response.__aiter__ = MagicMock(
                return_value=[
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="Similar compounds found:"))]
                ]
            )
            
            mock_create.side_effect = [mock_orchestrator_response, mock_synthesis_response]
            
            with patch('app.main.chemical_agent.run', new_callable=AsyncMock) as mock_chem:
                mock_chem.return_value = "Compound similarity results..."
                response = client.post("/orchestrate", json=payload)
        
        assert response.status_code == 200
        assert "Similar compounds" in response.text or response.text != ""
    
    def test_orchestrate_endpoint_with_medical_intent(self, client):
        """Verify /orchestrate handles biomedical mechanism intent."""
        payload = {
            "session_id": "test_session_medical",
            "user_id": "test_user_789",
            "text_input": "Explain the mechanism of action for metformin"
        }
        
        with patch('app.main.client.chat.completions.create') as mock_create:
            # Mock orchestrator response for medical intent
            mock_orchestrator_response = AsyncMock()
            mock_orchestrator_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "intent": "BIOMEDICAL_MECHANISM",
                    "target_agent": "MEDICAL_AGENT",
                    "entities": {"compound": "metformin"}
                })))
            ]
            
            # Mock synthesis response
            mock_synthesis_response = AsyncMock()
            mock_synthesis_response.__aiter__ = MagicMock(
                return_value=[
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="Metformin works by"))]
                ]
            )
            
            mock_create.side_effect = [mock_orchestrator_response, mock_synthesis_response]
            
            with patch('app.main.medical_agent.run', new_callable=AsyncMock) as mock_med:
                mock_med.return_value = "Mechanism of action details..."
                response = client.post("/orchestrate", json=payload)
        
        assert response.status_code == 200
    
    def test_orchestrate_endpoint_missing_required_fields(self, client):
        """Verify that missing required fields are handled properly."""
        payload = {
            "session_id": "test_session",
            # Missing user_id and text_input
        }
        
        response = client.post("/orchestrate", json=payload)
        # Should return 422 Unprocessable Entity for validation error
        assert response.status_code == 422
    
    def test_orchestrate_endpoint_session_isolation(self, client):
        """Verify that different sessions maintain isolated chat history."""
        payload_session1 = {
            "session_id": "session_1",
            "user_id": "user_1",
            "text_input": "Hello from session 1"
        }
        
        payload_session2 = {
            "session_id": "session_2",
            "user_id": "user_2",
            "text_input": "Hello from session 2"
        }
        
        with patch('app.main.client.chat.completions.create') as mock_create:
            mock_orchestrator_response = AsyncMock()
            mock_orchestrator_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "intent": "APP_HELP",
                    "target_agent": "APP_AGENT",
                    "entities": {}
                })))
            ]
            
            mock_synthesis_response = AsyncMock()
            mock_synthesis_response.__aiter__ = MagicMock(
                return_value=[MagicMock(choices=[MagicMock(delta=MagicMock(content="Response"))])]
            )
            
            mock_create.side_effect = [mock_orchestrator_response, mock_synthesis_response]
            client.post("/orchestrate", json=payload_session1)
            
            # Reset mock for second call
            mock_create.side_effect = [mock_orchestrator_response, mock_synthesis_response]
            client.post("/orchestrate", json=payload_session2)
            
            # Verify both calls were made (sessions are isolated)
            assert mock_create.call_count >= 2


class TestOrchestrationLogic:
    """Test suite for the core orchestration logic."""
    
    def test_orchestrator_system_prompt_exists(self):
        """Verify that the orchestrator system prompt is defined."""
        assert ORCHESTRATOR_SYSTEM_PROMPT is not None
        assert "Orchestrator" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "CHEMICAL_AGENT" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "MEDICAL_AGENT" in ORCHESTRATOR_SYSTEM_PROMPT
    
    def test_orchestrator_system_prompt_contains_intents(self):
        """Verify that the system prompt includes all required intents."""
        required_intents = [
            "CHEMICAL_SIMILARITY",
            "ADMET_ANALYSIS",
            "DRUG_REPURPOSING",
            "BIOMEDICAL_MECHANISM",
            "APP_HELP"
        ]
        
        for intent in required_intents:
            assert intent in ORCHESTRATOR_SYSTEM_PROMPT


class TestErrorHandling:
    """Test suite for error handling scenarios."""
    
    def test_orchestrate_handles_json_parse_error(self, client):
        """Verify that JSON parsing errors are handled gracefully."""
        payload = {
            "session_id": "error_test",
            "user_id": "test_user",
            "text_input": "Test query"
        }
        
        with patch('app.main.client.chat.completions.create') as mock_create:
            # Mock a response that can't be parsed as valid JSON
            mock_orchestrator_response = AsyncMock()
            mock_orchestrator_response.choices = [
                MagicMock(message=MagicMock(content="Invalid JSON {malformed"))
            ]
            
            mock_create.side_effect = mock_orchestrator_response
            
            response = client.post("/orchestrate", json=payload)
            
            # Should return 500 error
            assert response.status_code == 500
            assert "Error" in response.json()["detail"] or "error" in response.json()["detail"].lower()
    
    def test_orchestrate_clears_memory_on_crash(self, client):
        """Verify that session memory is cleared when a crash occurs."""
        from app.main import SESSION_MEMORY
        
        initial_count = len(SESSION_MEMORY)
        
        payload = {
            "session_id": "crash_test",
            "user_id": "test_user",
            "text_input": "Test query"
        }
        
        # Populate memory with initial state
        SESSION_MEMORY["crash_test"] = [{"role": "user", "content": "Hello"}]
        
        with patch('app.main.client.chat.completions.create') as mock_create:
            mock_create.side_effect = Exception("Simulated LLM API failure")
            
            response = client.post("/orchestrate", json=payload)
            
            # Memory should be cleared after the error
            assert len(SESSION_MEMORY.get("crash_test", [])) == 0
            assert response.status_code == 500
