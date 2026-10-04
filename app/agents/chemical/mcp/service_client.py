"""Shared HTTP plumbing for the tools that call external AILIXIR microservices."""

from __future__ import annotations

import httpx

DEFAULT_TIMEOUT_S = 60.0
_HEADERS = {"accept": "application/json", "Content-Type": "application/json"}


async def post_json(url: str, payload: dict, timeout: float = DEFAULT_TIMEOUT_S) -> httpx.Response:
    """POST ``payload`` as JSON. Raises ``httpx.RequestError`` on transport failure."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(url, json=payload, headers=_HEADERS)
