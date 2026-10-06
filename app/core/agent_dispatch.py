"""
app.core.agent_dispatch
~~~~~~~~~~~~~~~~~~~~~~~
Pure functions for selecting agents and building synthesis context.
No network I/O — safe to import anywhere and test without mocks.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Agent selection
# ---------------------------------------------------------------------------

def select_agents(intent: str, target_agent: str, entities: dict) -> list[str]:
    """
    Return an ordered list (no duplicates) of agent keys to run.

    Possible return values contain any subset of: "CHEMICAL", "MEDICAL".
    RAG_AGENT / APP_AGENT / anything else → [].

    Entity values of None or "" are treated as absent.

    Decision table
    ~~~~~~~~~~~~~~
    target == CHEMICAL_AGENT  |  intent in {CHEMICAL_SIMILARITY, ADMET_ANALYSIS}
        → ["CHEMICAL"]
    target == MEDICAL_AGENT  |  intent == BIOMEDICAL_MECHANISM
        → ["MEDICAL"]
    intent == DRUG_REPURPOSING
        → CHEMICAL if compound/smiles present
          MEDICAL  if disease present
          both if both present; if neither, CHEMICAL (default)
    RAG_AGENT / APP_AGENT / other  → []
    """
    result: list[str] = []

    has_chemical = bool(entities.get("smiles") or entities.get("compound"))
    has_disease  = bool(entities.get("disease"))

    # DRUG_REPURPOSING has its own exclusive entity-based rules
    if intent == "DRUG_REPURPOSING":
        if has_chemical and "CHEMICAL" not in result:
            result.append("CHEMICAL")
        if has_disease and "MEDICAL" not in result:
            result.append("MEDICAL")
        # If neither entity is present, default to CHEMICAL
        if not has_chemical and not has_disease and "CHEMICAL" not in result:
            result.append("CHEMICAL")
        return result

    # Explicit chemical triggers (non-DRUG_REPURPOSING)
    if target_agent == "CHEMICAL_AGENT" or intent in {
        "CHEMICAL_SIMILARITY", "ADMET_ANALYSIS"
    }:
        if "CHEMICAL" not in result:
            result.append("CHEMICAL")

    # Explicit medical triggers (non-DRUG_REPURPOSING)
    if target_agent == "MEDICAL_AGENT" or intent == "BIOMEDICAL_MECHANISM":
        if "MEDICAL" not in result:
            result.append("MEDICAL")

    return result


# ---------------------------------------------------------------------------
# Synthesis context builder
# ---------------------------------------------------------------------------

_NO_TOOL_MSG = (
    "[No tool data was retrieved for this question. "
    "Answer from general scientific knowledge and say clearly "
    "that no lab tools were used.]"
)
_APP_HELP_MSG = "[App System Context]: Standard greeting or help request."


def build_agent_context(
    chemical_output: str,
    medical_output: str,
    intent: str,
) -> str:
    """
    Build the agent_raw_output string that is injected into the synthesis prompt.

    Rules
    -----
    - Either output non-empty  → "[Chem Data]: ...\n[Bio Data]: ..."
    - Both empty + APP_HELP    → standard greeting context string
    - Both empty + other       → no-tool-data message
    """
    if chemical_output or medical_output:
        return f"[Chem Data]: {chemical_output}\n[Bio Data]: {medical_output}".strip()

    if intent == "APP_HELP":
        return _APP_HELP_MSG

    return _NO_TOOL_MSG
