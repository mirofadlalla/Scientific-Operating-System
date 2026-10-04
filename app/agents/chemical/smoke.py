"""Standalone smoke test for the chemical agent.

Requires the MCP server to be running::

    python -m app.agents.chemical.smoke
"""

from __future__ import annotations

import asyncio
import logging

from .agent import ChemicalAgent


async def _main():
    """Quick end-to-end smoke test — requires the MCP server to be running."""
    logging.basicConfig(level=logging.INFO)
    agent = ChemicalAgent()

    tests = [
        # (label, kwargs)
        ("ADMET (legacy)", dict(intent="admet", entities={"smiles": "CCO"})),
        ("Drug repurposing (legacy)", dict(intent="repurposing", entities={"disease": "Alzheimer's disease"})),
        ("Similarity (legacy)", dict(intent="similarity", entities={"smiles": "CCO"})),
        ("ADMET (new API)", dict(user_query="Calculate the ADMET properties of ethanol (CCO)")),
        ("Multi-tool", dict(user_query="Analyse CCO for ADMET and find structurally similar compounds")),
        ("Missing SMILES", dict(user_query="Calculate ADMET properties")),
        ("Missing disease", dict(user_query="Screen drugs for repurposing")),
    ]

    for label, kwargs in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {label}")
        print("="*60)
        result = await agent.run(**kwargs)
        print(result[:500])

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(_main())
