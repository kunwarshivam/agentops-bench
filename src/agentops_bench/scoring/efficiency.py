"""Efficiency scoring: step count and redundancy analysis."""

from __future__ import annotations

import json
from typing import Any

from agentops_bench.schema import AgentTrace, Task


def _count_redundant_calls(trace: AgentTrace) -> int:
    """Count tool calls that are exact duplicates (same tool + same args)."""
    seen: set[str] = set()
    redundant = 0
    for step in trace.steps:
        for call in step.tool_calls:
            key = f"{call.tool_name}::{json.dumps(call.arguments, sort_keys=True)}"
            if key in seen:
                redundant += 1
            else:
                seen.add(key)
    return redundant


def _total_tool_calls(trace: AgentTrace) -> int:
    """Total number of tool calls across all steps."""
    return sum(len(step.tool_calls) for step in trace.steps)


def score_efficiency(task: Task, trace: AgentTrace) -> dict[str, Any]:
    """Score how efficiently the agent solved the task.

    Metrics:
    - step_ratio: optimal_steps / actual_steps (capped at 1.0)
    - redundancy_penalty: fraction of calls that were redundant
    - overall: combined efficiency score (0.0 - 1.0)

    Args:
        task: The benchmark task (provides optimal_steps).
        trace: The agent's execution trace.

    Returns:
        Dictionary of efficiency metrics.
    """
    actual_steps = len(trace.steps)
    if actual_steps == 0:
        return {
            "actual_steps": 0,
            "optimal_steps": task.optimal_steps,
            "step_ratio": 0.0,
            "total_tool_calls": 0,
            "redundant_calls": 0,
            "redundancy_penalty": 0.0,
            "overall": 0.0,
        }

    # Step ratio: how close to optimal (1.0 = perfect, <1.0 = used too many steps)
    step_ratio = min(task.optimal_steps / actual_steps, 1.0)

    total_calls = _total_tool_calls(trace)
    redundant = _count_redundant_calls(trace)
    redundancy_rate = redundant / total_calls if total_calls > 0 else 0.0

    # Overall efficiency: step_ratio penalised by redundancy
    # redundancy_penalty reduces score proportionally
    overall = step_ratio * (1.0 - redundancy_rate)

    # Also penalise for excessive wall time relative to step count
    # (if agent spent >60s per step on average, slight penalty)
    if trace.wall_time_seconds > 0 and actual_steps > 0:
        avg_time_per_step = trace.wall_time_seconds / actual_steps
        if avg_time_per_step > 60:
            time_penalty = min((avg_time_per_step - 60) / 300, 0.3)  # cap at 0.3
            overall = overall * (1.0 - time_penalty)

    return {
        "actual_steps": actual_steps,
        "optimal_steps": task.optimal_steps,
        "step_ratio": round(step_ratio, 4),
        "total_tool_calls": total_calls,
        "redundant_calls": redundant,
        "redundancy_penalty": round(redundancy_rate, 4),
        "wall_time_seconds": round(trace.wall_time_seconds, 2),
        "overall": round(max(0.0, min(1.0, overall)), 4),
    }
