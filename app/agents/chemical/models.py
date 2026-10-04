"""Value types shared by the chemical agent modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _ToolResult:
    """Outcome of a single MCP tool execution."""

    tool_name: str
    arguments: dict
    output: str
    success: bool


class _NoToolCallError(Exception):
    """Raised when the LLM responds with text instead of a tool call."""

    def __init__(self, llm_text: str) -> None:
        super().__init__(llm_text)
        self.llm_text = llm_text
