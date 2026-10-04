"""Behavioural tests for the MCP tool functions split out of chemical_server.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.agents.chemical.mcp import chemical_server
from app.agents.chemical.mcp.tools import (
    ALL_TOOLS,
    calculate_admet,
    chemical_similarity_search,
    screen_drugs,
)

POST = "app.agents.chemical.mcp.service_client.httpx.AsyncClient"


def _resp(status: int, data: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = data
    return r


def _client(response=None, exc: Exception | None = None):
    ctx = MagicMock()
    ctx.post = AsyncMock(side_effect=exc) if exc else AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ctx)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, ctx


@pytest.mark.asyncio
async def test_server_registers_all_tools():
    names = {t.name for t in await chemical_server.mcp.list_tools()}
    assert names == {f.__name__ for f in ALL_TOOLS}


@pytest.mark.asyncio
@pytest.mark.parametrize("fn,kw", [
    (calculate_admet, {"smiles": " "}),
    (screen_drugs, {"disease_name": ""}),
    (chemical_similarity_search, {"smiles": ""}),
])
async def test_blank_input_short_circuits_without_http(fn, kw):
    with patch(POST) as client:
        assert (await fn(**kw)).startswith("Error:")
        client.assert_not_called()


@pytest.mark.asyncio
async def test_admet_formats_report_and_posts_payload():
    data = {"results": [{"smiles": "CCO", "predictions": {"Absorption": 0.5, "Toxicity": None}}],
            "processing_time_ms": 12.3456}
    cm, ctx = _client(_resp(200, data))
    with patch(POST, return_value=cm):
        out = await calculate_admet("CCO")
    assert "• Absorption: 0.5000" in out and "• Toxicity: N/A" in out and "12.35 ms" in out
    assert ctx.post.call_args.args[0].endswith("/predict_batch")
    assert ctx.post.call_args.kwargs["json"] == {"smiles_list": ["CCO"]}


@pytest.mark.asyncio
async def test_http_error_and_transport_error_are_reported_not_raised():
    cm, _ = _client(_resp(503, {}))
    with patch(POST, return_value=cm):
        assert await screen_drugs("AD") == "Drug repurposing service error: HTTP 503"
    cm, _ = _client(exc=httpx.ConnectError("down"))
    with patch(POST, return_value=cm):
        assert (await chemical_similarity_search("CCO")).startswith("Chemical RAG communication error:")


@pytest.mark.asyncio
async def test_screen_limits_to_three_candidates():
    cands = [{"drug_name": f"d{i}", "target_symbol": "T", "binding_score": 1} for i in range(6)]
    cm, _ = _client(_resp(200, {"disease_name": "AD", "top_candidates": cands}))
    with patch(POST, return_value=cm):
        out = await screen_drugs("AD")
    assert out.count("→") == 3 and "[4]" not in out


@pytest.mark.asyncio
async def test_similarity_endpoint_depends_on_explain_flag():
    for explain, suffix in ((True, "/search/full-rag"), (False, "/search/retrieval-only")):
        cm, ctx = _client(_resp(200, {"results": [{"name": "A", "similarity_score": 0.9,
                                                    "explanation": "because"}]}))
        with patch(POST, return_value=cm):
            out = await chemical_similarity_search("CCO", explain=explain)
        assert ctx.post.call_args.args[0].endswith(suffix)
        assert "Explanation: because" in out


def test_tool_schema_supports_mcp_v1_and_v2_attribute_names():
    from types import SimpleNamespace as NS

    from app.agents.chemical.mcp_client import _tool_schema

    schema = {"type": "object", "properties": {"smiles": {"type": "string"}}}
    assert _tool_schema(NS(input_schema=schema)) == schema      # mcp 2.x (pinned)
    assert _tool_schema(NS(inputSchema=schema)) == schema       # mcp 1.x / test doubles
    assert _tool_schema(NS()) == {"type": "object", "properties": {}}
    assert _tool_schema(NS(input_schema=None, inputSchema=None))["type"] == "object"
