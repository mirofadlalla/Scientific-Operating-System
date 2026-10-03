"""
tests/test_agent_dispatch.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for app.core.agent_dispatch.
No network I/O — all functions are pure.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.agent_dispatch import select_agents, build_agent_context


# ---------------------------------------------------------------------------
# select_agents
# ---------------------------------------------------------------------------

class TestSelectAgents:

    def test_biomedical_mechanism_medical_agent(self):
        result = select_agents("BIOMEDICAL_MECHANISM", "MEDICAL_AGENT", {"compound": "metformin"})
        assert result == ["MEDICAL"]

    def test_admet_analysis_chemical_agent(self):
        result = select_agents("ADMET_ANALYSIS", "CHEMICAL_AGENT", {})
        assert result == ["CHEMICAL"]

    def test_chemical_similarity_with_smiles(self):
        result = select_agents("CHEMICAL_SIMILARITY", "CHEMICAL_AGENT", {"smiles": "CCO"})
        assert result == ["CHEMICAL"]

    def test_drug_repurposing_both_entities(self):
        result = select_agents(
            "DRUG_REPURPOSING", "CHEMICAL_AGENT",
            {"compound": "x", "disease": "y"},
        )
        assert result == ["CHEMICAL", "MEDICAL"]

    def test_drug_repurposing_disease_only(self):
        result = select_agents("DRUG_REPURPOSING", "CHEMICAL_AGENT", {"disease": "y"})
        assert result == ["MEDICAL"]

    def test_drug_repurposing_no_entities_defaults_chemical(self):
        result = select_agents("DRUG_REPURPOSING", "CHEMICAL_AGENT", {})
        assert result == ["CHEMICAL"]

    def test_app_help_returns_empty(self):
        result = select_agents("APP_HELP", "APP_AGENT", {})
        assert result == []

    def test_app_support_rag_returns_empty(self):
        result = select_agents("APP_SUPPORT_RAG", "RAG_AGENT", {})
        assert result == []

    def test_none_entity_values_treated_as_absent(self):
        """None values for compound, smiles, disease must count as absent."""
        result = select_agents(
            "DRUG_REPURPOSING", "CHEMICAL_AGENT",
            {"compound": None, "smiles": None, "disease": None},
        )
        # No entity → default CHEMICAL
        assert result == ["CHEMICAL"]

    def test_empty_string_entity_values_treated_as_absent(self):
        result = select_agents(
            "DRUG_REPURPOSING", "CHEMICAL_AGENT",
            {"compound": "", "smiles": "", "disease": ""},
        )
        assert result == ["CHEMICAL"]

    def test_no_duplicates_when_both_triggers_match(self):
        """CHEMICAL_SIMILARITY + DRUG_REPURPOSING with both entities: no dupe CHEMICAL."""
        result = select_agents(
            "CHEMICAL_SIMILARITY", "CHEMICAL_AGENT",
            {"compound": "aspirin", "disease": "cancer"},
        )
        assert result.count("CHEMICAL") == 1

    def test_medical_agent_target_always_medical(self):
        result = select_agents("APP_HELP", "MEDICAL_AGENT", {})
        assert "MEDICAL" in result

    def test_chemical_agent_target_always_chemical(self):
        result = select_agents("APP_HELP", "CHEMICAL_AGENT", {})
        assert "CHEMICAL" in result


# ---------------------------------------------------------------------------
# build_agent_context
# ---------------------------------------------------------------------------

class TestBuildAgentContext:

    def test_both_outputs_present(self):
        ctx = build_agent_context("chem data", "bio data", "ADMET_ANALYSIS")
        assert "[Chem Data]:" in ctx
        assert "[Bio Data]:" in ctx
        assert "chem data" in ctx
        assert "bio data" in ctx

    def test_only_chemical_output(self):
        ctx = build_agent_context("chem data", "", "ADMET_ANALYSIS")
        assert "[Chem Data]:" in ctx
        assert "chem data" in ctx

    def test_only_medical_output(self):
        ctx = build_agent_context("", "bio data", "BIOMEDICAL_MECHANISM")
        assert "[Bio Data]:" in ctx
        assert "bio data" in ctx

    def test_no_output_app_help_intent(self):
        ctx = build_agent_context("", "", "APP_HELP")
        assert ctx == "[App System Context]: Standard greeting or help request."

    def test_no_output_scientific_intent(self):
        """Non-APP_HELP intent with no agent output → no-tool message."""
        ctx = build_agent_context("", "", "ADMET_ANALYSIS")
        assert "no tool data" in ctx.lower() or "No tool data" in ctx
        assert "general scientific knowledge" in ctx.lower() or "general scientific" in ctx

    def test_no_output_biomedical_intent(self):
        ctx = build_agent_context("", "", "BIOMEDICAL_MECHANISM")
        assert "no lab tools" in ctx.lower() or "no tool data" in ctx.lower()


# ---------------------------------------------------------------------------
# MedicalAgent prompt test (mocked AsyncOpenAI)
# ---------------------------------------------------------------------------

class TestMedicalAgentPrompt:
    """
    Verify MedicalAgent.run builds the right prompt with and without user_query.
    Stubs AsyncOpenAI so no real HTTP call is made.
    """

    def _make_agent(self, mock_client):
        """Create a MedicalAgent with a pre-injected mock client."""
        import app.agents.medical.agent as mod
        agent = mod.MedicalAgent.__new__(mod.MedicalAgent)
        agent.client = mock_client
        agent.model_name = "test-model"
        return agent

    def _make_mock_client(self, reply: str):
        choice = MagicMock()
        choice.message.content = reply
        response = MagicMock()
        response.choices = [choice]
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        return mock_client

    @pytest.mark.asyncio
    async def test_with_user_query_prompt_contains_question(self):
        """When user_query is supplied the sent prompt must include the question text."""
        mock_client = self._make_mock_client("Some biomedical output.")
        agent = self._make_agent(mock_client)

        question = "How does metformin work in type 2 diabetes?"
        await agent.run(
            "BIOMEDICAL_MECHANISM",
            {"compound": "metformin"},
            user_query=question,
        )

        # Inspect the messages arg passed to create()
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert question in user_msg, f"Question not in prompt: {user_msg!r}"
        assert "Extracted entities" in user_msg

    @pytest.mark.asyncio
    async def test_without_user_query_uses_old_template(self):
        """Without user_query the sent prompt matches the original template exactly."""
        mock_client = self._make_mock_client("Some output.")
        agent = self._make_agent(mock_client)

        await agent.run("ADMET_ANALYSIS", {"compound": "metformin", "disease": ""})

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        # Old template contains this literal string
        assert "Chemical Identifier/SMILES:" in user_msg
        assert "Target Pathology:" in user_msg
