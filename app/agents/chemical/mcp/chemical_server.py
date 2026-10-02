import httpx

from mcp.server import MCPServer

from app.config import settings


mcp = MCPServer(
    "AILIXIR Chemical Intelligence"
)


def _fmt(val, decimals: int = 4) -> str:
    """Safely format numeric values."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"

    return str(val) if val is not None else "N/A"


# ============================================================
# Tool 1: ADMET
# ============================================================

@mcp.tool()
async def calculate_admet(smiles: str) -> str:
    """
    Calculate ADMET properties for a chemical compound.

    Args:
        smiles: Valid SMILES representation of the compound.
    """

    if not smiles.strip():
        return "Error: A valid SMILES string is required."

    url = (
        f"{settings.ADMET_AI_URL.rstrip('/')}"
        "/predict_batch"
    )

    payload = {
        "smiles_list": [smiles]
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:

            response = await client.post(
                url,
                json=payload,
                headers=headers,
            )

        if response.status_code != 200:
            return (
                "ADMET service error: "
                f"HTTP {response.status_code}"
            )

        data = response.json()

        results = data.get("results", [])

        if not results:
            return "ADMET service returned no results."

        result = results[0]

        predictions = result.get(
            "predictions",
            {}
        )

        return (
            f"ADMET Analysis for "
            f"{result.get('smiles', smiles)}:\n"
            f"• Absorption: "
            f"{_fmt(predictions.get('Absorption'))}\n"
            f"• Distribution: "
            f"{_fmt(predictions.get('Distribution'))}\n"
            f"• Metabolism: "
            f"{_fmt(predictions.get('Metabolism'))}\n"
            f"• Excretion: "
            f"{_fmt(predictions.get('Excretion'))}\n"
            f"• Toxicity: "
            f"{_fmt(predictions.get('Toxicity'))}\n"
            f"• Processing Time: "
            f"{_fmt(data.get('processing_time_ms'), 2)} ms"
        )

    except httpx.RequestError as exc:

        return (
            "ADMET communication error: "
            f"{exc}"
        )


# ============================================================
# Tool 2: Drug Repurposing
# ============================================================

@mcp.tool()
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

    url = (
        f"{settings.DRUG_REPURPOSING_URL.rstrip('/')}"
        "/api/v1/screen"
    )

    payload = {
        "disease_name": disease_name,
        "min_score": 0,
        "top_n_targets": top_n_targets,
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:

            response = await client.post(
                url,
                json=payload,
                headers=headers,
            )

        if response.status_code != 200:
            return (
                "Drug repurposing service error: "
                f"HTTP {response.status_code}"
            )

        data = response.json()

        candidates = data.get(
            "top_candidates",
            []
        )

        if not candidates:
            return (
                f"No drug candidates found for "
                f"{disease_name}."
            )

        report = (
            f"Drug Repurposing Screening "
            f"for '{data.get('disease_name', disease_name)}':\n"
        )

        for idx, candidate in enumerate(
            candidates[:3],
            start=1,
        ):

            report += (
                f"[{idx}] "
                f"{candidate.get('drug_name', 'Unknown')} "
                f"→ "
                f"{candidate.get('target_symbol', 'Unknown')} "
                f"(Score: "
                f"{_fmt(candidate.get('binding_score'), 2)})\n"
            )

        return report

    except httpx.RequestError as exc:

        return (
            "Drug repurposing communication error: "
            f"{exc}"
        )


# ============================================================
# Tool 3: Chemical Similarity Search
# ============================================================

@mcp.tool()
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

    endpoint = (
        "/search/full-rag"
        if explain
        else "/search/retrieval-only"
    )

    url = (
        f"{settings.CHEMICAL_AI_URL.rstrip('/')}"
        f"{endpoint}"
    )

    payload = {
        "smiles": smiles,
        "top_k": top_k,
        "explain": explain,
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:

            response = await client.post(
                url,
                json=payload,
                headers=headers,
            )

        if response.status_code != 200:
            return (
                "Chemical RAG service error: "
                f"HTTP {response.status_code}"
            )

        data = response.json()

        results = data.get(
            "results",
            []
        )

        if not results:
            return (
                f"No similar compounds found "
                f"for {smiles}."
            )

        report = (
            f"Chemical Similarity Search "
            f"for '{data.get('query_smiles', smiles)}':\n"
        )

        for idx, result in enumerate(
            results[:top_k],
            start=1,
        ):

            report += (
                f"[{idx}] "
                f"{result.get('name', 'Compound')} "
                f"(Similarity: "
                f"{_fmt(result.get('similarity_score'), 2)})"
            )

            explanation = result.get("explanation")

            if explanation:
                report += f"\n    Explanation: {explanation}"

            report += "\n"

        return report

    except httpx.RequestError as exc:

        return (
            "Chemical RAG communication error: "
            f"{exc}"
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)