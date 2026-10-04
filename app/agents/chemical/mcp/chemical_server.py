"""AILIXIR Chemical Intelligence MCP server.

Run with::

    python -m app.agents.chemical.mcp.chemical_server

Tool implementations live in :mod:`.tools`; this module only wires them up.
"""

from __future__ import annotations

import os

from mcp.server import MCPServer

from .formatting import fmt as _fmt  # noqa: F401  (re-exported for backward compatibility)
from .tools import ALL_TOOLS, calculate_admet, chemical_similarity_search, screen_drugs

mcp = MCPServer("AILIXIR Chemical Intelligence")

for _tool in ALL_TOOLS:
    mcp.tool()(_tool)

# __all___ is a list of public objects of that module, as interpreted by import *. It overrides the default of hiding everything that begins with an underscore.
__all__ = ["_fmt", "calculate_admet", "chemical_similarity_search", "mcp", "screen_drugs"]


if __name__ == "__main__":
    _port = int(os.getenv("MCP_SERVER_PORT", "8001"))
    mcp.run(transport="streamable-http", port=_port)
