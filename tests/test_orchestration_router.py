"""``route_and_stream`` behaviour after the split into app.core.orchestration.*"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest

import app.core.state as state
from app.core import deps
from app.core.orchestration import (
    COMBINED_ORCHESTRATOR_PROMPT,
    route_and_stream,
    split_composite_question,
)
from app.core.orchestration import agents_runner, memory, router


def _chunk(t):
    return NS(choices=[NS(delta=NS(content=t))])


class _Llm:
    """Fake chat client: JSON replies for routing/composite, token stream otherwise."""

    def __init__(self, routing=None, composite=None, tokens=("A", "B")):
        self.routing = routing or {"intent": "ADMET_ANALYSIS", "target_agent": "CHEMICAL_AGENT", "entities": {}}
        self.composite = composite or {"is_composite": False, "sub_questions": ["x"]}
        self.tokens, self.calls = tokens, []

    async def create(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            async def gen():
                for t in self.tokens:
                    yield _chunk(t)
            return gen()
        reply = self.composite if kw["messages"][0]["content"].startswith("You decide") else self.routing
        if isinstance(reply, Exception):
            raise reply
        return NS(choices=[NS(message=NS(content=json.dumps(reply)))])


@pytest.fixture
def env(monkeypatch):
    llm = _Llm()
    adds = []
    monkeypatch.setattr(deps.client.chat.completions, "create", lambda **kw: llm.create(**kw))
    monkeypatch.setattr(deps.short_memory, "add_message", lambda *a: adds.append(a))
    monkeypatch.setattr(deps.short_memory, "get_history", lambda sid, limit=0: [])
    chem, med, rag = (AsyncMock(return_value=f"{n}-OUT") for n in ("chem", "med", "rag"))
    monkeypatch.setattr(agents_runner, "chemical_agent", NS(run=chem))
    monkeypatch.setattr(agents_runner, "medical_agent", NS(run=med))
    monkeypatch.setattr(agents_runner, "rag_agent", NS(run=rag))
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    monkeypatch.setattr(state, "long_memory", None)
    return NS(llm=llm, adds=adds, chem=chem, med=med, rag=rag)


async def _collect(text, **kw):
    return "".join([t async for t in route_and_stream(text, "sid", "uid", **kw)])


@pytest.mark.asyncio
async def test_greeting_skips_routing_and_stores_memory(env):
    assert await _collect("hello") == "AB"
    assert len(env.llm.calls) == 1 and env.llm.calls[0]["stream"]
    assert env.adds == [("sid", "user", "hello"), ("sid", "assistant", "AB")]
    env.chem.assert_not_called()


@pytest.mark.asyncio
async def test_chemical_route_synthesises_with_agent_output(env):
    env.llm.routing["entities"] = {"smiles": "CCO"}
    out = await _collect("admet of ethanol")
    assert "![Molecular Structure of CCO]" in out and out.endswith("AB")
    env.chem.assert_awaited_once()
    assert "chem-OUT" in env.llm.calls[-1]["messages"][-1]["content"]
    assert env.llm.calls[-1]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_include_images_false_and_store_memory_false(env):
    env.llm.routing["entities"] = {"smiles": "CCO"}
    out = await _collect("admet", include_images=False, store_memory=False)
    assert "Molecular Structure" not in out and env.adds == []


@pytest.mark.asyncio
async def test_out_of_domain_refusal_localised_and_short_memory_only(env, monkeypatch):
    lm = NS(add_entry=lambda *a, **k: pytest.fail("long-term memory must not be written"))
    monkeypatch.setattr(state, "long_memory", lm)
    env.llm.routing = {"intent": "OUT_OF_DOMAIN", "target_agent": "NONE", "out_of_domain_reason": "cooking"}
    assert "outside my scientific domain" in await _collect("recipe for rice")
    assert "خارج نطاق" in await _collect("وصفة رز")
    assert len(env.adds) == 4


@pytest.mark.asyncio
async def test_rag_answer_streamed_directly_and_noinfo_falls_back(env):
    env.llm.routing = {"intent": "APP_SUPPORT_RAG", "target_agent": "RAG_AGENT"}
    env.rag.return_value = "Omar built it"
    assert (await _collect("who built this")).split() == ["Omar", "built", "it"]
    assert len(env.llm.calls) == 2  # composite + routing only, no synthesis

    env.llm.calls.clear()
    env.rag.return_value = "The documentation does not contain information"
    assert (await _collect("what is x")).endswith("AB")  # fell back to LLM synthesis


@pytest.mark.asyncio
async def test_agent_error_is_reported_to_synthesis_not_raised(env):
    env.chem.side_effect = RuntimeError("boom")
    await _collect("admet")
    assert "[Chemical Agent Error]: boom" in env.llm.calls[-1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_routing_failure_uses_fallback_classifier(env, monkeypatch):
    env.llm.routing = RuntimeError("500")
    monkeypatch.setattr(deps.orchestrator, "classify_intent",
                        lambda t: {"intent": "medical", "entities": {}})
    await _collect("gene pathway")
    env.med.assert_awaited_once()
    env.chem.assert_not_called()


@pytest.mark.asyncio
async def test_composite_answers_each_part_and_writes_memory_once(env):
    env.llm.composite = {"is_composite": True, "sub_questions": ["q one", "q two"]}
    out = await _collect("q one and q two")
    assert "Detected **2 sub-questions**" in out and "❶ q one" in out and "❷ q two" in out
    assert [a[1] for a in env.adds] == ["user", "assistant"]  # once, for the original question


@pytest.mark.asyncio
async def test_composite_detection_failure_returns_original(env):
    env.llm.composite = RuntimeError("x")
    assert await split_composite_question("abc") == ["abc"]


@pytest.mark.asyncio
async def test_long_term_memory_failure_is_swallowed(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("redis down")

    monkeypatch.setattr(state, "long_memory", NS(add_entry=boom))
    assert await _collect("hello") == "AB"
    memory.remember_turn("s", "u", "r", intent="i", agent="a")  # does not raise


def test_public_reexports():
    assert "CHEMICAL_AGENT" in COMBINED_ORCHESTRATOR_PROMPT and router.route_and_stream is route_and_stream
