"""Streaming LLM completions with TTFT/TPS metrics, shared by every reply path."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import AsyncIterator

from app import monitoring
from app.core.deps import client


@dataclass
class StreamStats:
    """Filled in while :func:`stream_completion` is consumed."""

    full_reply: str = ""
    token_count: int = 0
    ttft_ms: float | None = None
    tps: float = 0.0
    iteration_started_at: float = 0.0  # set once the stream object is obtained


async def stream_completion(
    model: str,
    messages: list[dict],
    temperature: float,
    stats: StreamStats,
) -> AsyncIterator[str]:
    """Yield tokens from a streaming chat completion, recording metrics in ``stats``."""
    start_time = time.time()
    stream = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=True,
    )
    stats.iteration_started_at = time.time()

    first_token_received = False
    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue
        if not first_token_received:
            first_token_received = True
            stats.ttft_ms = (time.time() - start_time) * 1000
        stats.token_count += 1
        stats.full_reply += token
        yield token

    gen_duration = time.time() - start_time - ((stats.ttft_ms or 0) / 1000)
    stats.tps = stats.token_count / gen_duration if gen_duration > 0 and stats.token_count > 0 else 0.0


def record_usage(model: str, messages: list[dict], stats: StreamStats) -> None:
    """Report approximate token usage (≈4 chars/token) and latency metrics."""
    monitoring.record_tokens(
        model=model,
        prompt_tokens=sum(len(m.get("content", "")) for m in messages) // 4,
        completion_tokens=len(stats.full_reply) // 4,
        ttft_ms=stats.ttft_ms,
        tps=stats.tps,
    )
