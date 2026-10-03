"""Parametrized greeting-router tests (scientific queries must not skip routing)."""
import pytest

from app.core.greeting import is_general_greeting
from app.core.orchestration import is_general_greeting as reexported_greeting
from app.orchestrator.brain import OrchestratorBrain


ROUTED_QUERIES = [
    "HIV protease inhibitors and resistance mutations",
    "Histamine H2 receptor antagonists mechanism",
    "Yohimbine mechanism of action",
    "Help me predict ADMET for CC(=O)Oc1ccccc1C(=O)O",
    "Can you help me find compounds similar to ibuprofen?",
    "Supramolecular carriers for drug delivery",
    "Okadaic acid toxicity pathway",
    "Paracetamol toxicity",
    "Metformin mechanism",
    "Nice, now compare ibuprofen with naproxen ADMET",
    "What is the mechanism of action of metformin in type 2 diabetes?",
]

GREETING_QUERIES = [
    "hello",
    "hi",
    "hey there",
    "good morning",
    "thanks",
    "thank you so much",
    "bye",
    "who are you",
    "what can you do",
    "ok",
    "السلام عليكم",
    "مرحبا",
    "شكرا",
    "كيف حالك",
    "من أنت",
]


@pytest.mark.parametrize("query", ROUTED_QUERIES)
def test_scientific_queries_are_not_greetings(query):
    assert is_general_greeting(query) is False
    assert reexported_greeting(query) is False


@pytest.mark.parametrize("query", GREETING_QUERIES)
def test_true_greetings_are_detected(query):
    assert is_general_greeting(query) is True
    assert reexported_greeting(query) is True


def test_fallback_classify_hiv_protease_not_app_agent():
    result = OrchestratorBrain()._fallback_classify("HIV protease inhibitors")
    assert result["intent"] in {"chemical", "medical"}
    assert result["intent"] != "app_agent"
