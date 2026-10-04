"""MCP tool implementations (plain async functions, registered by the server)."""

from .admet import calculate_admet
from .screening import screen_drugs
from .similarity import chemical_similarity_search

ALL_TOOLS = (calculate_admet, screen_drugs, chemical_similarity_search)

__all__ = ["ALL_TOOLS", "calculate_admet", "chemical_similarity_search", "screen_drugs"]
