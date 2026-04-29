"""Cost scoring: token consumption and dollar cost analysis."""

from __future__ import annotations

import math
from typing import Any

from agentops_bench.schema import AgentTrace

# Pricing per 1M tokens (USD), snapshot date below.
# Refresh by editing this table — see provider pricing pages and
# update PRICING_AS_OF when you do, so existing reports remain
# attributable to a known price regime.
# Format: model_substring -> (input_price_per_1M, output_price_per_1M)
PRICING_AS_OF = "2026-04"
TOKEN_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic — Claude 4 family (2025–2026). Substring matched longest-first.
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Legacy Claude 3.x snapshots (kept so old traces re-score sensibly).
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    # OpenAI — GPT-5.x family (2025–2026). Substring matched longest-first.
    "gpt-5.5-pro": (10.0, 80.0),
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.15, 0.80),
    "gpt-5.4": (3.0, 15.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (1.25, 10.0),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "o3": (10.0, 40.0),
    # Legacy OpenAI (kept so old traces re-score consistently).
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o1": (15.0, 60.0),
    # Other commercial
    "mistral-large": (2.0, 6.0),
    # Open-weights via OpenRouter (list-price snapshot 2026-04;
    # OpenRouter aggregates upstream providers, so the per-token rate
    # changes when upstreams shift, refresh PRICING_AS_OF when updating).
    "llama-4-maverick":     (0.15, 0.60),
    "llama-4-scout":        (0.08, 0.30),
    "llama-3.3-70b":        (0.10, 0.28),
    "llama-3-70b":          (0.51, 0.74),
    "qwen3-max":            (0.78, 3.90),
    "qwen3.6-flash":        (0.30, 1.20),
    "qwen-3-72b":           (0.45, 0.75),
    "qwen3":                (0.45, 0.75),
    "deepseek-v4-pro":      (0.435, 0.87),
    "deepseek-v4":          (0.55, 1.30),
    "deepseek-v3.2":        (0.252, 0.378),
    "deepseek-v3":          (0.27, 1.10),
    "deepseek-r1":          (0.55, 2.19),
    "mistral-large-2512":   (0.50, 1.50),
    "mistral-large-3":      (2.0, 6.0),
    "mixtral-8x22b":        (0.65, 0.65),
}


def _lookup_pricing(agent_id: str) -> tuple[float, float]:
    """Return (input_price_per_1M, output_price_per_1M) for the agent.

    See ``_lookup_pricing_with_label`` if you also need the matched prefix.
    """
    return _lookup_pricing_with_label(agent_id)[1]


def _lookup_pricing_with_label(agent_id: str) -> tuple[str, tuple[float, float]]:
    """Like ``_lookup_pricing`` but also returns the matched key.

    Matches the longest substring first so that e.g. ``gpt-4o-mini`` is not
    eaten by the shorter ``gpt-4o`` prefix. Falls back to ``gpt-4o`` with a
    ``"unknown"`` label.
    """
    agent_lower = agent_id.lower()
    for prefix, pricing in sorted(TOKEN_PRICING.items(), key=lambda kv: -len(kv[0])):
        if prefix in agent_lower:
            return prefix, pricing
    return "unknown", TOKEN_PRICING["gpt-4o"]


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> float:
    """Calculate USD cost from token counts and per-million-token prices."""
    return (input_tokens * input_price_per_m + output_tokens * output_price_per_m) / 1_000_000


def score_cost(trace: AgentTrace) -> dict[str, Any]:
    """Compute cost metrics for an agent trace.

    Args:
        trace: The agent's execution trace.

    Returns:
        Dictionary with token counts and cost breakdown.
    """
    input_tokens = trace.total_input_tokens
    output_tokens = trace.total_output_tokens
    total_tokens = input_tokens + output_tokens

    matched, (input_price, output_price) = _lookup_pricing_with_label(trace.agent_id)
    cost_usd = compute_cost(input_tokens, output_tokens, input_price, output_price)

    # Per-step breakdown
    step_costs: list[dict[str, Any]] = []
    for step in trace.steps:
        step_cost = compute_cost(
            step.llm_input_tokens, step.llm_output_tokens, input_price, output_price
        )
        step_costs.append({
            "step": step.step_number,
            "input_tokens": step.llm_input_tokens,
            "output_tokens": step.llm_output_tokens,
            "cost_usd": round(step_cost, 6),
        })

    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "pricing_model": matched,
        "pricing_per_m": {"input_usd": input_price, "output_usd": output_price},
        "step_breakdown": step_costs,
    }


# Cost penalty: a $1 run is multiplied by 0.9, i.e. -ln(0.9) ≈ 0.105.
# Smooth, bounded in [0, completion], no division-by-zero hack.
CNA_ALPHA: float = 0.10536051565782628  # -math.log(0.9)


def cost_normalized_accuracy(completion_score: float, cost_usd: float) -> float:
    """Completion discounted by an exponential cost penalty.

    Returns ``completion · e^(-α · cost_usd)`` with ``α = -ln(0.9) ≈ 0.105``,
    so a $1 run is penalised by 10% and a free model returns the completion
    score unchanged. Bounded in [0, completion]; higher is better.
    """
    cost = max(cost_usd, 0.0)
    return round(completion_score * math.exp(-CNA_ALPHA * cost), 4)
