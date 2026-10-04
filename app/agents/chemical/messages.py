"""User-facing messages returned by the chemical agent on failure paths."""

from __future__ import annotations

DISCOVERY_TIMEOUT = (
    "The chemical analysis service timed out during tool discovery. "
    "Please try again in a moment."
)
NO_TOOLS = "No chemical analysis tools are currently available. Please contact support."
SELECTION_TIMEOUT = (
    "The AI reasoning engine timed out while analysing your request. Please try again."
)
SELECTION_PARSE_ERROR = (
    "I had trouble understanding how to process your request. Could you rephrase it?"
)
SELECTION_FAILURE = (
    "The AI reasoning engine encountered an error. Please try again or check your query."
)
NO_MATCHING_TOOL = (
    "I couldn't match your request to any available chemical analysis tool. "
    "Please rephrase your question or specify the analysis you need."
)
ALL_TOOLS_FAILED = (
    "All requested chemical analyses encountered errors. "
    "The underlying services may be temporarily unavailable. "
    "Please try again shortly."
)
MCP_UNREACHABLE = (
    "The chemical analysis service is currently unreachable (connection error). "
    "Please try again shortly."
)
MCP_CONNECT_ERROR = (
    "Unable to connect to the chemical analysis service. "
    "Please verify the service is running and try again."
)
MCP_HTTP_TIMEOUT = "The chemical analysis service timed out. Please try again in a moment."
PIPELINE_ERROR = (
    "An unexpected error occurred in the chemical analysis pipeline. "
    "Please try again or contact support."
)
INTERNAL_ERROR = (
    "I encountered an unexpected internal error while processing "
    "your chemical query. Please try again or contact support."
)
