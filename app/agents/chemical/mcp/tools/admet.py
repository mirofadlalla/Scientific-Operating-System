"""Tool 1: ADMET property prediction."""

from __future__ import annotations

import httpx

from app.config import settings

from ..formatting import fmt
from ..service_client import post_json


async def calculate_admet(smiles: str) -> str:
    """
    Calculate ADMET properties for a chemical compound.

    Args:
        smiles: Valid SMILES representation of the compound.
    """
    if not smiles.strip():
        return "Error: A valid SMILES string is required."

    url = f"{settings.ADMET_AI_URL.rstrip('/')}/predict_batch"

    try:
        response = await post_json(url, {"smiles_list": [smiles]})

        if response.status_code != 200:
            return f"ADMET service error: HTTP {response.status_code}"

        data = response.json()
        results = data.get("results", [])
        if not results:
            return "ADMET service returned no results."

        result = results[0]
        predictions = result.get("predictions", {})

        return (
            f"ADMET Analysis for {result.get('smiles', smiles)}:\n"
            f"• Absorption: {fmt(predictions.get('Absorption'))}\n"
            f"• Distribution: {fmt(predictions.get('Distribution'))}\n"
            f"• Metabolism: {fmt(predictions.get('Metabolism'))}\n"
            f"• Excretion: {fmt(predictions.get('Excretion'))}\n"
            f"• Toxicity: {fmt(predictions.get('Toxicity'))}\n"
            f"• Processing Time: {fmt(data.get('processing_time_ms'), 2)} ms"
        )

    except httpx.RequestError as exc:
        return f"ADMET communication error: {exc}"
