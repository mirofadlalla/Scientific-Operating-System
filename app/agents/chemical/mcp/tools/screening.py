"""Tool 2: virtual drug-repurposing screening."""

from __future__ import annotations

import httpx

from app.config import settings

from ..formatting import fmt
from ..service_client import post_json

_MAX_CANDIDATES = 3


async def screen_drugs(
    disease_name: str,
    top_n_targets: int = 5,
) -> str:
    """
    Perform virtual drug repurposing screening
    for a target disease.

    Args:
        disease_name: Name of the disease.
        top_n_targets: Number of molecular targets to consider.
    """
    if not disease_name.strip():
        return "Error: Disease name is required."

    url = f"{settings.DRUG_REPURPOSING_URL.rstrip('/')}/api/v1/screen"
    payload = {
        "disease_name": disease_name,
        "min_score": 0,
        "top_n_targets": top_n_targets,
    }

    try:
        response = await post_json(url, payload)

        if response.status_code != 200:
            return f"Drug repurposing service error: HTTP {response.status_code}"

        data = response.json()
        candidates = data.get("top_candidates", [])
        if not candidates:
            return f"No drug candidates found for {disease_name}."

        report = f"Drug Repurposing Screening for '{data.get('disease_name', disease_name)}':\n"
        for idx, candidate in enumerate(candidates[:_MAX_CANDIDATES], start=1):
            report += (
                f"[{idx}] {candidate.get('drug_name', 'Unknown')} → "
                f"{candidate.get('target_symbol', 'Unknown')} "
                f"(Score: {fmt(candidate.get('binding_score'), 2)})\n"
            )
        return report

    except httpx.RequestError as exc:
        return f"Drug repurposing communication error: {exc}"
