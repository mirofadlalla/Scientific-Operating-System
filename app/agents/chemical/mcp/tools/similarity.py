"""Tool 3: structural similarity search against the chemical RAG service."""

from __future__ import annotations

import httpx

from app.config import settings

from ..formatting import fmt
from ..service_client import post_json


async def chemical_similarity_search(
    smiles: str,
    top_k: int = 3,
    explain: bool = True,
) -> str:
    """
    Search for structurally similar chemical compounds.

    Args:
        smiles: Query compound represented as SMILES.
        top_k: Number of similar compounds to return.
        explain: Whether to include chemical explanations.
    """
    if not smiles.strip():
        return "Error: A valid SMILES string is required."

    endpoint = "/search/full-rag" if explain else "/search/retrieval-only"
    url = f"{settings.CHEMICAL_AI_URL.rstrip('/')}{endpoint}"
    payload = {"smiles": smiles, "top_k": top_k, "explain": explain}

    try:
        response = await post_json(url, payload)

        if response.status_code != 200:
            return f"Chemical RAG service error: HTTP {response.status_code}"

        data = response.json()
        results = data.get("results", [])
        if not results:
            return f"No similar compounds found for {smiles}."

        report = f"Chemical Similarity Search for '{data.get('query_smiles', smiles)}':\n"
        for idx, result in enumerate(results[:top_k], start=1):
            report += (
                f"[{idx}] {result.get('name', 'Compound')} "
                f"(Similarity: {fmt(result.get('similarity_score'), 2)})"
            )
            explanation = result.get("explanation")
            if explanation:
                report += f"\n    Explanation: {explanation}"
            report += "\n"
        return report

    except httpx.RequestError as exc:
        return f"Chemical RAG communication error: {exc}"
