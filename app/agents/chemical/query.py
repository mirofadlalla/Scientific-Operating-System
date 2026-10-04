"""Reconstruct natural-language queries from the legacy (intent, entities) call-site."""

from __future__ import annotations


def resolve_query(
    intent: str | None,
    entities: dict | None,
    user_query: str | None,
) -> str:
    """Return user_query as-is, or build a descriptive query from intent/entities."""
    if user_query:
        return user_query

    entities = entities or {}
    intent = (intent or "").lower()
    smiles = entities.get("smiles") or entities.get("compound") or ""
    disease = entities.get("disease") or ""

    if "admet" in intent or "toxicity" in intent or "absorption" in intent:
        if smiles:
            return f"Calculate the ADMET properties for the compound with SMILES: {smiles}"
        return "Calculate ADMET properties"

    if any(k in intent for k in ("repurpos", "screen", "drug")):
        if disease:
            return f"Find drug repurposing candidates for {disease}"
        return "Screen drugs for repurposing"

    if any(k in intent for k in ("similar", "rag", "chemical_similarity")):
        explain = "explain" in intent or "detailed" in intent
        if smiles:
            suffix = " and explain their structural relevance" if explain else ""
            return f"Find compounds structurally similar to SMILES {smiles}{suffix}"
        return "Find similar chemical compounds"

    # Generic fallback — include all available context
    parts = []
    if smiles:
        parts.append(f"compound SMILES {smiles}")
    if disease:
        parts.append(f"disease: {disease}")
    context_str = "; ".join(parts)
    if intent and context_str:
        return f"{intent} — {context_str}"
    if intent:
        return intent
    if context_str:
        return f"Analyse {context_str}"
    return "Perform a chemical analysis"
