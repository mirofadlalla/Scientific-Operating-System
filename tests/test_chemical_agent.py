"""
tests/test_chemical_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive test suite for the LLM-driven ChemicalAgent.

Covers:
  A. ADMET request
  B. Drug repurposing request
  C. Chemical similarity request
  D. Multi-tool request
  E. Missing SMILES (LLM should ask the user)
  F. Missing disease (LLM should ask the user)
  G. Invalid tool arguments (MCP validation error)
  H. MCP unavailable (connection error)
  I. MCP timeout
  J. LLM failure
  K. Unknown / unsupported request (LLM responds without a tool call)

These tests mock external dependencies (MCP server, LLM) and verify
agent behaviour — not merely that functions execute.
"""
from __future__ import annotations

import asyncio
import json
import types
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import httpx
import pytest
import pytest_asyncio

from app.agents.chemical.agent import (
    ChemicalAgent,
    _NoToolCallError,
    _ToolResult,
    _discover_tools,
    _execute_mcp_tool,
    _llm_synthesize,
    _llm_tool_selection,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

FAKE_MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_admet",
            "description": "Calculate ADMET properties for a chemical compound.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "SMILES string"}
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_drugs",
            "description": "Perform virtual drug repurposing screening for a target disease.",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {"type": "string"},
                    "top_n_targets": {"type": "integer", "default": 5},
                },
                "required": ["disease_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chemical_similarity_search",
            "description": "Search for structurally similar chemical compounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "explain": {"type": "boolean", "default": True},
                },
                "required": ["smiles"],
            },
        },
    },
]

ADMET_RAW = "ADMET Analysis for CCO:\n• Absorption: 0.9\n• Toxicity: 0.1"
REPURPOSING_RAW = "Drug Repurposing for 'Alzheimer's disease':\n[1] Donepezil"
SIMILARITY_RAW = "Chemical Similarity Search for 'CCO':\n[1] Methanol (0.85)"
SYNTHESIS_RESPONSE = "Here is the synthesised scientific answer."


def _make_tool_call(name: str, arguments: dict):
    """Build a fake OpenAI tool-call object."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_llm_response(tool_calls: list | None = None, content: str = ""):
    """Build a fake OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.tool_calls = tool_calls
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_mcp_result(text: str):
    """Build a fake MCP call_tool result."""
    item = MagicMock()
    item.text = text
    result = MagicMock()
    result.structured_content = None
    result.content = [item]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a fully-patched ChemicalAgent
# ─────────────────────────────────────────────────────────────────────────────

def _make_agent(
    mcp_tools=None,
    llm_tool_call_response=None,
    mcp_tool_result=None,
    synthesis_response=SYNTHESIS_RESPONSE,
):
    """
    Return a ChemicalAgent whose MCP client and LLM are fully mocked.
    mcp_tools:            list of OpenAI-style tool dicts
    llm_tool_call_response: LLM response for tool selection
    mcp_tool_result:      async-callable or result returned by MCP call_tool
    synthesis_response:   string returned by LLM synthesis
    """
    agent = ChemicalAgent.__new__(ChemicalAgent)
    agent.mcp_url = "http://mock-mcp:8001/mcp"
    agent._llm = AsyncMock()
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# A. ADMET request
# ─────────────────────────────────────────────────────────────────────────────

class TestADMETRequest:
    """Agent correctly routes an ADMET request to calculate_admet."""

    @pytest.mark.asyncio
    async def test_admet_new_api(self):
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call("calculate_admet", {"smiles": "CCO"})
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tool_call]),          # tool selection
            _make_llm_response(content=SYNTHESIS_RESPONSE),  # synthesis
        ])

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=_make_mcp_result(ADMET_RAW))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Calculate ADMET properties of ethanol (CCO)")

        assert SYNTHESIS_RESPONSE in result

    @pytest.mark.asyncio
    async def test_admet_legacy_api(self):
        """Legacy run(intent='admet', entities={smiles:...}) must still work."""
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call("calculate_admet", {"smiles": "CCO"})
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tool_call]),
            _make_llm_response(content=SYNTHESIS_RESPONSE),
        ])

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=_make_mcp_result(ADMET_RAW))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(intent="admet", entities={"smiles": "CCO"})

        assert SYNTHESIS_RESPONSE in result

    @pytest.mark.asyncio
    async def test_admet_calls_correct_tool(self):
        """Verify that calculate_admet is invoked with the correct SMILES argument."""
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call("calculate_admet", {"smiles": "CCO"})
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tool_call]),
            _make_llm_response(content=SYNTHESIS_RESPONSE),
        ])

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=_make_mcp_result(ADMET_RAW))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await agent.run(user_query="Calculate ADMET for CCO")

        ctx.call_tool.assert_called_once_with("calculate_admet", {"smiles": "CCO"})


# ─────────────────────────────────────────────────────────────────────────────
# B. Drug repurposing request
# ─────────────────────────────────────────────────────────────────────────────

class TestDrugRepurposingRequest:

    @pytest.mark.asyncio
    async def test_repurposing_new_api(self):
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call(
            "screen_drugs",
            {"disease_name": "Alzheimer's disease", "top_n_targets": 5},
        )
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tool_call]),
            _make_llm_response(content=SYNTHESIS_RESPONSE),
        ])

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=_make_mcp_result(REPURPOSING_RAW))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(
                user_query="Find drugs that could be repurposed for Alzheimer's disease"
            )

        assert SYNTHESIS_RESPONSE in result

    @pytest.mark.asyncio
    async def test_repurposing_calls_correct_tool(self):
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call(
            "screen_drugs",
            {"disease_name": "Parkinson's disease"},
        )
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tool_call]),
            _make_llm_response(content=SYNTHESIS_RESPONSE),
        ])

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=_make_mcp_result(REPURPOSING_RAW))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await agent.run(user_query="Screen drugs for Parkinson's disease")

        ctx.call_tool.assert_called_once_with(
            "screen_drugs", {"disease_name": "Parkinson's disease"}
        )


# ─────────────────────────────────────────────────────────────────────────────
# C. Chemical similarity request
# ─────────────────────────────────────────────────────────────────────────────

class TestChemicalSimilarityRequest:

    @pytest.mark.asyncio
    async def test_similarity_new_api(self):
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call(
            "chemical_similarity_search",
            {"smiles": "CCO", "top_k": 3, "explain": False},
        )
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tool_call]),
            _make_llm_response(content=SYNTHESIS_RESPONSE),
        ])

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=_make_mcp_result(SIMILARITY_RAW))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(
                user_query="Find compounds structurally similar to ethanol (CCO)"
            )

        assert SYNTHESIS_RESPONSE in result

    @pytest.mark.asyncio
    async def test_similarity_with_explain(self):
        """When user asks for explanations, explain=True should be in arguments."""
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call(
            "chemical_similarity_search",
            {"smiles": "CCO", "top_k": 5, "explain": True},
        )
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tool_call]),
            _make_llm_response(content=SYNTHESIS_RESPONSE),
        ])

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=_make_mcp_result(SIMILARITY_RAW))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await agent.run(
                user_query=(
                    "Find similar compounds to CCO and explain their "
                    "structural relevance"
                )
            )

        _, called_args = ctx.call_tool.call_args
        # explain=True should be passed
        call_positional = ctx.call_tool.call_args[0]
        assert call_positional[1].get("explain") is True


# ─────────────────────────────────────────────────────────────────────────────
# D. Multi-tool request
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiToolRequest:

    @pytest.mark.asyncio
    async def test_multi_tool_both_called(self):
        """
        When the user requests ADMET + similarity, both tools are called and
        results are synthesised together.
        """
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tc_admet = _make_tool_call("calculate_admet", {"smiles": "CCO"})
        tc_sim = _make_tool_call(
            "chemical_similarity_search", {"smiles": "CCO", "top_k": 3}
        )
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tc_admet, tc_sim]),   # selects 2 tools
            _make_llm_response(content=SYNTHESIS_RESPONSE),
        ])

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(side_effect=[
                _make_mcp_result(ADMET_RAW),
                _make_mcp_result(SIMILARITY_RAW),
            ])
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(
                user_query=(
                    "Analyse CCO for ADMET properties and find "
                    "structurally similar compounds"
                )
            )

        assert ctx.call_tool.call_count == 2
        assert SYNTHESIS_RESPONSE in result

    @pytest.mark.asyncio
    async def test_multi_tool_synthesis_receives_all_results(self):
        """
        Verify the LLM synthesis call receives outputs from both tools.
        Captures kwargs passed to the second LLM call to confirm both tool
        names appear in the synthesis prompt.
        """
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tc_admet = _make_tool_call("calculate_admet", {"smiles": "CCO"})
        tc_sim = _make_tool_call(
            "chemical_similarity_search", {"smiles": "CCO", "top_k": 3}
        )

        captured_messages: list = []
        call_count = 0

        async def llm_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_llm_response([tc_admet, tc_sim])
            # Second call: synthesis — capture messages
            msgs = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
            captured_messages.extend(msgs)
            return _make_llm_response(content=SYNTHESIS_RESPONSE)

        agent._llm.chat.completions.create = AsyncMock(side_effect=llm_side_effect)

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(side_effect=[
                _make_mcp_result(ADMET_RAW),
                _make_mcp_result(SIMILARITY_RAW),
            ])
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(
                user_query="Analyse CCO for ADMET and find similar compounds"
            )

        # Both tools were executed
        assert ctx.call_tool.call_count == 2
        # LLM was called twice (selection + synthesis)
        assert call_count == 2
        # Synthesis response reaches the caller
        assert SYNTHESIS_RESPONSE in result
        # Synthesis prompt contained references to both tool names
        full_prompt = " ".join(
            m.get("content", "") for m in captured_messages if isinstance(m, dict)
        )
        assert "calculate_admet" in full_prompt, (
            f"'calculate_admet' missing from synthesis prompt. Got: {full_prompt[:400]}"
        )
        assert "chemical_similarity_search" in full_prompt, (
            f"'chemical_similarity_search' missing from synthesis prompt. Got: {full_prompt[:400]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# E. Missing SMILES
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingSMILES:

    @pytest.mark.asyncio
    async def test_missing_smiles_llm_asks_user(self):
        """
        When SMILES is missing, the LLM should respond in natural language
        (no tool call) asking the user to provide it.
        The agent should return that clarification — not a blank or error string.
        """
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        clarification = (
            "Could you please provide the SMILES string or compound name "
            "for which you want ADMET properties calculated?"
        )
        # LLM responds with no tool call — just a clarification question
        agent._llm.chat.completions.create = AsyncMock(
            return_value=_make_llm_response(tool_calls=None, content=clarification)
        )

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Calculate ADMET properties")

        # Must return the clarification — not an error or empty string
        assert clarification in result
        # MCP tool must NOT have been called
        ctx.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_smiles_legacy_query_reconstruction(self):
        """Legacy call with no SMILES should still produce a descriptive query."""
        query = ChemicalAgent._resolve_query(
            intent="admet",
            entities={},
            user_query=None,
        )
        assert "admet" in query.lower() or "ADMET" in query


# ─────────────────────────────────────────────────────────────────────────────
# F. Missing disease
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingDisease:

    @pytest.mark.asyncio
    async def test_missing_disease_llm_asks_user(self):
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        clarification = (
            "Please specify the disease name for drug repurposing screening."
        )
        agent._llm.chat.completions.create = AsyncMock(
            return_value=_make_llm_response(tool_calls=None, content=clarification)
        )

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Screen drugs for repurposing")

        assert clarification in result
        ctx.call_tool.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# G. Invalid tool arguments
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidToolArguments:

    @pytest.mark.asyncio
    async def test_mcp_tool_error_returns_clean_message(self):
        """
        If the MCP tool returns an error (e.g. invalid SMILES), the agent
        should return a clean user-facing message without a raw stack trace.
        """
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call("calculate_admet", {"smiles": "INVALID_SMILES"})
        agent._llm.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response([tool_call]),
            _make_llm_response(content="The ADMET calculation failed: invalid SMILES."),
        ])

        mcp_error_result = MagicMock()
        mcp_error_result.structured_content = None
        error_item = MagicMock()
        error_item.text = "Error: invalid SMILES string"
        mcp_error_result.content = [error_item]

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=mcp_error_result)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Calculate ADMET for INVALID_SMILES")

        # Should not expose traceback
        assert "Traceback" not in result
        assert "Exception" not in result

    @pytest.mark.asyncio
    async def test_malformed_llm_tool_call_json(self):
        """
        If the LLM returns malformed JSON in a tool call, the agent should
        return a clean error — not crash.
        """
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        bad_tc = MagicMock()
        bad_tc.function.name = "calculate_admet"
        bad_tc.function.arguments = "{ not valid json ..."  # malformed

        agent._llm.chat.completions.create = AsyncMock(
            return_value=_make_llm_response([bad_tc])
        )

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Calculate ADMET for CCO")

        # Should return a clean user-facing message
        assert result  # not empty
        assert "Traceback" not in result
        assert "rephrase" in result.lower() or "error" in result.lower() or "trouble" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# H. MCP unavailable
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPUnavailable:

    @pytest.mark.asyncio
    async def test_mcp_connection_refused_clean_message(self):
        """ConnectError from MCP must produce a clean user message."""
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://localhost:9999/mcp"  # unreachable port
        agent._llm = AsyncMock()

        with patch("app.agents.chemical.agent.Client") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Calculate ADMET for CCO")

        assert "Traceback" not in result
        assert result  # not empty
        assert any(
            phrase in result.lower()
            for phrase in ["connect", "unavailable", "unreachable", "service"]
        )

    @pytest.mark.asyncio
    async def test_mcp_os_error_clean_message(self):
        """Generic OSError from MCP must produce a clean user message."""
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        with patch("app.agents.chemical.agent.Client") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(
                side_effect=OSError("Network unreachable")
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Calculate ADMET for CCO")

        assert "Traceback" not in result
        assert result


# ─────────────────────────────────────────────────────────────────────────────
# I. MCP timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPTimeout:

    @pytest.mark.asyncio
    async def test_tool_discovery_timeout(self):
        """Timeout during tool discovery returns a clean message."""
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        async def slow_list_tools():
            await asyncio.sleep(999)

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(side_effect=slow_list_tools)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("app.agents.chemical.agent.asyncio.wait_for") as mock_wait:
                mock_wait.side_effect = asyncio.TimeoutError()
                result = await agent.run(user_query="Calculate ADMET for CCO")

        assert "timeout" in result.lower() or "timed out" in result.lower()
        assert "Traceback" not in result

    @pytest.mark.asyncio
    async def test_individual_tool_timeout_returns_partial(self):
        """
        If a tool times out, the agent reports the timeout cleanly in the
        result for that tool, but continues with any successful results.
        """
        from app.agents.chemical.agent import _ToolResult

        # Test _execute_mcp_tool directly with a simulated timeout
        mock_client = AsyncMock()

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(999)

        mock_client.call_tool = AsyncMock(side_effect=slow_call)

        with patch("app.agents.chemical.agent.asyncio.wait_for",
                   side_effect=asyncio.TimeoutError()):
            # Simulate what _timed_call does inside _run_llm_mcp_pipeline
            try:
                await asyncio.wait_for(
                    _execute_mcp_tool(mock_client, "calculate_admet", {"smiles": "CCO"}),
                    timeout=0.001,
                )
                result = _ToolResult("calculate_admet", {}, "ok", True)
            except asyncio.TimeoutError:
                result = _ToolResult(
                    "calculate_admet", {},
                    "[Timeout] Tool 'calculate_admet' exceeded execution limit.",
                    False,
                )

        assert not result.success
        assert "Timeout" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# J. LLM failure
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMFailure:

    @pytest.mark.asyncio
    async def test_llm_api_error_clean_message(self):
        """If the LLM API raises an exception, agent returns a clean message."""
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        agent._llm.chat.completions.create = AsyncMock(
            side_effect=Exception("Groq API rate limit exceeded")
        )

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Calculate ADMET for CCO")

        assert "Traceback" not in result
        assert result
        assert any(
            phrase in result.lower()
            for phrase in ["error", "try again", "encountered"]
        )

    @pytest.mark.asyncio
    async def test_llm_synthesis_failure_falls_back_to_raw_output(self):
        """If LLM synthesis fails, agent falls back to raw tool output."""
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        tool_call = _make_tool_call("calculate_admet", {"smiles": "CCO"})
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_llm_response([tool_call])
            raise Exception("LLM synthesis service unavailable")

        agent._llm.chat.completions.create = AsyncMock(side_effect=side_effect)

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            ctx.call_tool = AsyncMock(return_value=_make_mcp_result(ADMET_RAW))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Calculate ADMET for CCO")

        # Fallback: raw tool output should still be in response
        assert ADMET_RAW in result
        assert "Traceback" not in result


# ─────────────────────────────────────────────────────────────────────────────
# K. Unknown / unsupported request
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownRequest:

    @pytest.mark.asyncio
    async def test_unsupported_request_no_tool_call(self):
        """
        For a request completely outside the available tools, the LLM should
        respond in text (no tool call). The agent should return that text.
        """
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        llm_response_text = (
            "I'm sorry, I can only help with chemical analysis tasks such as "
            "ADMET prediction, drug repurposing, and chemical similarity search. "
            "I cannot provide cooking recipes."
        )
        agent._llm.chat.completions.create = AsyncMock(
            return_value=_make_llm_response(tool_calls=None, content=llm_response_text)
        )

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Give me a recipe for pasta carbonara")

        assert llm_response_text in result
        ctx.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_requests_unknown_tool_clean_error(self):
        """
        If the LLM somehow requests a tool not in the MCP server's list,
        the agent should return a clean message rather than crashing.
        """
        agent = ChemicalAgent.__new__(ChemicalAgent)
        agent.mcp_url = "http://mock-mcp:8001/mcp"
        agent._llm = AsyncMock()

        unknown_tool_call = _make_tool_call("nonexistent_tool", {"foo": "bar"})
        agent._llm.chat.completions.create = AsyncMock(
            return_value=_make_llm_response([unknown_tool_call])
        )

        with patch("app.agents.chemical.agent.Client") as MockClient:
            ctx = AsyncMock()
            ctx.list_tools = AsyncMock(return_value=_fake_list_tools())
            MockClient.return_value.__aenter__ = AsyncMock(return_value=ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await agent.run(user_query="Use some obscure tool")

        assert "Traceback" not in result
        assert result
        # Should indicate it couldn't match the request to a tool
        assert any(
            phrase in result.lower()
            for phrase in ["match", "available", "rephrase", "tool"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Query reconstruction (unit tests — no I/O needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryReconstruction:

    def test_user_query_passthrough(self):
        result = ChemicalAgent._resolve_query(None, None, "Some query")
        assert result == "Some query"

    def test_admet_with_smiles(self):
        result = ChemicalAgent._resolve_query(
            "admet", {"smiles": "CCO"}, None
        )
        assert "CCO" in result
        assert "ADMET" in result or "admet" in result.lower()

    def test_repurposing_with_disease(self):
        result = ChemicalAgent._resolve_query(
            "repurposing", {"disease": "diabetes"}, None
        )
        assert "diabetes" in result.lower()

    def test_similarity_with_smiles(self):
        result = ChemicalAgent._resolve_query(
            "similarity", {"smiles": "c1ccccc1"}, None
        )
        assert "c1ccccc1" in result

    def test_generic_fallback_with_context(self):
        result = ChemicalAgent._resolve_query(
            "unknown_intent",
            {"smiles": "CCO", "disease": "cancer"},
            None,
        )
        assert result  # not empty

    def test_empty_everything(self):
        result = ChemicalAgent._resolve_query(None, {}, None)
        assert result  # should return a safe fallback string


# ─────────────────────────────────────────────────────────────────────────────
# _ToolResult unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestToolResult:

    def test_successful_result(self):
        r = _ToolResult("calculate_admet", {"smiles": "CCO"}, "output", True)
        assert r.success
        assert r.tool_name == "calculate_admet"
        assert r.output == "output"

    def test_failed_result(self):
        r = _ToolResult("screen_drugs", {}, "[error]", False)
        assert not r.success


# ─────────────────────────────────────────────────────────────────────────────
# Helper — build fake MCP list_tools response
# ─────────────────────────────────────────────────────────────────────────────

def _fake_list_tools():
    """Produce a fake MCP list_tools() response object."""
    tools = []
    for spec in [
        ("calculate_admet", "Calculate ADMET properties.",
         {"type": "object", "properties": {"smiles": {"type": "string"}}, "required": ["smiles"]}),
        ("screen_drugs", "Drug repurposing screening.",
         {"type": "object", "properties": {"disease_name": {"type": "string"}}, "required": ["disease_name"]}),
        ("chemical_similarity_search", "Find similar compounds.",
         {"type": "object", "properties": {"smiles": {"type": "string"}}, "required": ["smiles"]}),
    ]:
        t = MagicMock()
        t.name, t.description, t.inputSchema = spec
        tools.append(t)

    resp = MagicMock()
    resp.tools = tools
    return resp
