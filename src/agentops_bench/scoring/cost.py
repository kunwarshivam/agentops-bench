"""Cost scoring: token consumption and dollar cost analysis."""

from __future__ import annotations

from typing import Any

from agentops_bench.schema import AgentTrace

# Pricing per 1M tokens (USD) as of early 2026.
# Format: model_substring -> (input_price_per_1M, output_price_per_1M)
TOKEN_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-3.5": (0.80, 4.0),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "o3-mini": (1.10, 4.40),
    "o1-mini": (3.0, 12.0),
    "o1": (15.0, 60.0),
    # Other commercial
    "mistral-large": (2.0, 6.0),
}


def _lookup_pricing(agent_id: str) -> tuple[float, float]:
    """Find the best matching pricing entry for a given agent/model ID.

    Returns (input_price_per_1M_tokens, output_price_per_1M_tokens).
    Matches longest substring first so that e.g. "gpt-4o-mini" is not eaten
    by the shorter "gpt-4o" prefix. Defaults to GPT-4o pricing.
    """
    agent_lower = agent_id.lower()
    for prefix, pricing in sorted(TOKEN_PRICING.items(), key=lambda kv: -len(kv[0])):
        if prefix in agent_lower:
            return pricing
    return TOKEN_PRICING["gpt-4o"]


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

    input_price, output_price = _lookup_pricing(trace.agent_id)
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
        "pricing_model": f"input=${input_price}/M, output=${output_price}/M",
        "step_breakdown": step_costs,
    }


def cost_normalized_accuracy(completion_score: float, cost_usd: float) -> float:
    """Compute accuracy normalised by cost.

    Higher is better. Returns completion_score / cost, with a floor on cost
    to avoid division-by-zero for free models.

    Args:
        completion_score: Task completion score (0.0 - 1.0).
        cost_usd: Total cost in USD.

    Returns:
        Cost-normalised accuracy (score per dollar).
    """
    effective_cost = max(cost_usd, 0.0001)  # floor at $0.0001
    return round(completion_score / effective_cost, 4)
