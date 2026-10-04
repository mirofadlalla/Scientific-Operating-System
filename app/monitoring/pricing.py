"""Model pricing (USD per 1M tokens) used for cost estimates."""

from __future__ import annotations

# Groq rates / default estimates
MODEL_PRICING = {
    "openai/gpt-oss-20b":      {"prompt": 0.075, "completion": 0.30},
    "openai/gpt-oss-120b":     {"prompt": 0.15,  "completion": 0.60},
    "llama-3.3-70b-versatile": {"prompt": 0.59,  "completion": 0.79},
    "whisper-large-v3-turbo":  {"prompt": 0.04,  "completion": 0.04},
    "canopylabs/orpheus-arabic-saudi": {"prompt": 0.50, "completion": 0.50},
    "canopylabs/orpheus-v1-english": {"prompt": 0.50, "completion": 0.50},
}

DEFAULT_RATES = {"prompt": 0.50, "completion": 0.50}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost of one call (unknown models use ``DEFAULT_RATES``)."""
    rates = MODEL_PRICING.get(model, DEFAULT_RATES)
    return (prompt_tokens / 1_000_000) * rates["prompt"] + (completion_tokens / 1_000_000) * rates["completion"]
