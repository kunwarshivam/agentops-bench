"""Completion scoring: did the agent produce the correct final answer?"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from agentops_bench.schema import AgentTrace, Task


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace, and collapse runs of whitespace."""
    return " ".join(text.lower().split())


def _deterministic_score(expected: str, actual: str) -> float:
    """Score structured / deterministic outputs via exact or fuzzy match.

    Returns:
        Float between 0.0 and 1.0.
    """
    if _normalize(expected) == _normalize(actual):
        return 1.0

    # Try JSON-level comparison
    try:
        expected_obj = json.loads(expected)
        actual_obj = json.loads(actual)
        if expected_obj == actual_obj:
            return 1.0
    except (json.JSONDecodeError, TypeError):
        pass

    # Partial credit: check if all key phrases from expected appear in actual
    expected_tokens = set(_normalize(expected).split())
    actual_tokens = set(_normalize(actual).split())
    if not expected_tokens:
        return 0.0

    overlap = len(expected_tokens & actual_tokens) / len(expected_tokens)
    return round(min(overlap, 1.0), 4)


async def _llm_judge_score(task_description: str, expected: str | None, actual: str) -> float:
    """Use an LLM to judge open-ended task completion.

    Calls the Anthropic API (Claude) with a structured rubric and returns
    a score between 0.0 and 1.0.

    Falls back to 0.5 if the API is unavailable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return 0.5  # Cannot judge without API key

    rubric = (
        "You are an impartial judge evaluating whether an AI agent completed a task correctly.\n\n"
        f"TASK: {task_description}\n\n"
    )
    if expected:
        rubric += f"EXPECTED OUTPUT (reference): {expected}\n\n"
    rubric += (
        f"AGENT OUTPUT: {actual}\n\n"
        "Rate the agent's output on a scale from 0.0 to 1.0:\n"
        "- 1.0 = fully correct, complete, and well-formatted\n"
        "- 0.7-0.9 = mostly correct with minor issues\n"
        "- 0.4-0.6 = partially correct, missing key elements\n"
        "- 0.1-0.3 = attempted but largely wrong\n"
        "- 0.0 = completely wrong or no meaningful output\n\n"
        "Respond with ONLY a JSON object: {\"score\": <float>, \"reasoning\": \"<brief explanation>\"}"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": rubric}],
                },
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"]
            result = json.loads(text)
            score = float(result.get("score", 0.5))
            return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


async def score_completion(task: Task, trace: AgentTrace) -> float:
    """Score whether the agent completed the task correctly.

    Uses deterministic comparison when ``task.expected_output`` is provided,
    otherwise falls back to LLM-as-judge.

    Args:
        task: The benchmark task definition.
        trace: The agent's execution trace.

    Returns:
        Score between 0.0 (complete failure) and 1.0 (perfect completion).
    """
    if not trace.completed or not trace.final_output:
        return 0.0

    actual = trace.final_output

    # If we have a reference answer, prefer deterministic scoring
    if task.expected_output:
        det_score = _deterministic_score(task.expected_output, actual)
        # If deterministic score is high, trust it
        if det_score >= 0.8:
            return det_score
        # For ambiguous cases, also consult the LLM judge
        llm_score = await _llm_judge_score(task.description, task.expected_output, actual)
        # Weighted blend: favour the higher of the two
        return round(max(det_score, llm_score * 0.9), 4)

    # Open-ended task: LLM judge only
    return await _llm_judge_score(task.description, None, actual)
