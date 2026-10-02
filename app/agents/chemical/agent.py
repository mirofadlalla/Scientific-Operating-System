import asyncio
import traceback

from mcp import Client

from app.config import settings


class ChemicalAgent:

    def __init__(self):
        self.mcp_url = "http://localhost:8001/mcp"  # Update with your MCP server URL

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> str:

        print("\n" + "=" * 60)
        print("MCP TOOL CALL")
        print("=" * 60)
        print(f"Server : {self.mcp_url}")
        print(f"Tool   : {tool_name}")
        print(f"Args   : {arguments}")
        print("=" * 60)

        try:

            async with Client(self.mcp_url) as client:

                print("[MCP] Connected to MCP server.")

                # -------------------------------------------------
                # Optional: inspect available tools
                # -------------------------------------------------

                tools = await client.list_tools()

                print("\n[MCP] Available tools:")

                for tool in tools.tools:
                    print(f"  - {tool.name}")

                # -------------------------------------------------
                # Call requested tool
                # -------------------------------------------------

                print(
                    f"\n[MCP] Calling tool: {tool_name}"
                )

                result = await client.call_tool(
                    tool_name,
                    arguments,
                )

                print(
                    f"[MCP] Tool '{tool_name}' executed successfully."
                )

                # -------------------------------------------------
                # Handle structured content
                # -------------------------------------------------

                if result.structured_content:

                    print(
                        "[MCP] Received structured content."
                    )

                    return str(
                        result.structured_content
                    )

                # -------------------------------------------------
                # Handle text content
                # -------------------------------------------------

                if result.content:

                    texts = []

                    for item in result.content:

                        if hasattr(item, "text"):

                            texts.append(
                                item.text
                            )

                    if texts:

                        return "\n".join(
                            texts
                        )

                return (
                    "[Chemical MCP] "
                    "Tool returned an empty response."
                )

        # =========================================================
        # ExceptionGroup
        # =========================================================

        except ExceptionGroup as exc:

            print("\n")
            print("=" * 60)
            print("MCP EXCEPTION GROUP")
            print("=" * 60)

            for index, error in enumerate(
                exc.exceptions,
                start=1,
            ):

                print(
                    f"\n--- Inner Exception {index} ---"
                )

                traceback.print_exception(
                    error
                )

            print("=" * 60)

            return (
                "[Chemical MCP Error] "
                "An internal MCP exception occurred. "
                "Check the traceback above."
            )

        # =========================================================
        # Normal Exception
        # =========================================================

        except Exception as exc:

            print("\n")
            print("=" * 60)
            print("MCP ERROR")
            print("=" * 60)

            traceback.print_exc()

            print("=" * 60)

            return (
                f"[Chemical MCP Error] "
                f"{type(exc).__name__}: {exc}"
            )

    # =============================================================
    # Main Agent Interface
    # =============================================================

    async def run(
        self,
        intent: str,
        entities: dict,
    ) -> str:

        # ---------------------------------------------------------
        # Extract entities
        # ---------------------------------------------------------

        compound = entities.get(
            "compound",
            "",
        )

        smiles = entities.get(
            "smiles",
            compound,
        )

        disease = entities.get(
            "disease",
            "",
        )

        # =========================================================
        # 1. ADMET
        # =========================================================

        if any(
            keyword in intent.lower()
            for keyword in [
                "admet",
                "toxicity",
                "property",
                "poison",
                "absorption",
            ]
        ):

            if not smiles:

                return (
                    "[Chemical Agent Error] "
                    "Please provide a valid SMILES string "
                    "to calculate ADMET properties."
                )

            return await self._call_tool(
                tool_name="calculate_admet",
                arguments={
                    "smiles": smiles,
                },
            )

        # =========================================================
        # 2. Drug Repurposing
        # =========================================================

        elif any(
            keyword in intent.lower()
            for keyword in [
                "repurposing",
                "repurpose",
                "screen",
                "target",
            ]
        ):

            # -----------------------------------------------------
            # Fallback:
            # Sometimes orchestrator puts disease in compound.
            # -----------------------------------------------------

            target_disease = (
                disease
                if disease
                else compound
            )

            if not target_disease:

                return (
                    "[Chemical Agent Error] "
                    "Please specify a target disease name."
                )

            return await self._call_tool(
                tool_name="screen_drugs",
                arguments={
                    "disease_name": target_disease,
                    "top_n_targets": 5,
                },
            )

        # =========================================================
        # 3. Chemical Similarity / RAG
        # =========================================================

        else:

            if not smiles:

                return (
                    "[Chemical Agent Error] "
                    "Please provide a valid SMILES string "
                    "to execute chemical similarity search."
                )

            explain = (
                "explain" in intent.lower()
                or "detailed" in intent.lower()
            )

            return await self._call_tool(
                tool_name="chemical_similarity_search",
                arguments={
                    "smiles": smiles,
                    "top_k": 3,
                    "explain": explain,
                },
            )


# =============================================================
# Local Test
# =============================================================

async def main():

    print("\n")
    print("=" * 60)
    print("CHEMICAL AGENT MCP TEST")
    print("=" * 60)

    agent = ChemicalAgent()

    # =========================================================
    # Test 1: ADMET
    # =========================================================

    print("\n\n")
    print("TEST 1: ADMET")
    print("-" * 60)

    result = await agent.run(
        intent="admet",
        entities={
            "smiles": "CCO",
            "compound": "ethanol",
            "disease": "",
        },
    )

    print("\n===== ADMET RESULT =====")
    print(result)

    # =========================================================
    # Test 2: Drug Repurposing
    # =========================================================

    print("\n\n")
    print("TEST 2: DRUG REPURPOSING")
    print("-" * 60)

    result = await agent.run(
        intent="repurposing",
        entities={
            "disease": "Alzheimer's disease",
            "compound": "",
            "smiles": "",
        },
    )

    print("\n===== REPURPOSING RESULT =====")
    print(result)

    # =========================================================
    # Test 3: Chemical RAG
    # =========================================================

    print("\n\n")
    print("TEST 3: CHEMICAL RAG")
    print("-" * 60)

    result = await agent.run(
        intent="similarity",
        entities={
            "smiles": "CCO",
            "compound": "ethanol",
            "disease": "",
        },
    )

    print("\n===== CHEMICAL RAG RESULT =====")
    print(result)

    print("\n")
    print("=" * 60)
    print("ALL TESTS FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())