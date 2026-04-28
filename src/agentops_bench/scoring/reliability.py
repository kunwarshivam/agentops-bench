"""Reliability scoring: consistency across repeated runs."""

from __future__ import annotations

import math
from typing import Any

from agentops_bench.schema import RunResult


def _extract_completion_scores(results: list[RunResult]) -> list[float]:
    """Pull the completion score from each run result."""
    scores: list[float] = []
    for r in results:
        cs = r.scores.get("completion", 0.0)
        scores.append(float(cs))
    return scores


def confidence_interval(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson score confidence interval for a proportion.

    Args:
        rate: Observed success rate (0.0 - 1.0).
        n: Number of trials.
        z: Z-score for desired confidence level (default 1.96 = 95%).

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    if n == 0:
        return (0.0, 1.0)

    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    spread = z * math.sqrt((rate * (1 - rate) + z * z / (4 * n)) / n) / denominator

    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (round(lower, 4), round(upper, 4))


def score_reliability(results: list[RunResult]) -> dict[str, Any]:
    """Score reliability across multiple runs of the same task+agent+condition.

    Two distinct metrics are reported because they measure different things:
    - ``reliability_rate`` is the *crash-free rate*: fraction of runs that
      completed without erroring out. A run that completed with a wrong answer
      still counts as 1.0 here. Its 95% Wilson CI is in ``confidence_interval_95``.
    - ``mean_completion_score`` is the *quality rate*: average completion
      score across runs, including partial credit. This is the honest metric
      for "did the agent succeed at the task". ``min`` / ``max`` show the
      spread.

    Treat ``reliability_rate`` and ``mean_completion_score`` as a pair —
    a high crash-free rate with a low quality score means the agent finishes
    confidently with the wrong answer, which is a *worse* failure mode than
    a crash.

    Args:
        results: List of RunResult for repeated runs.

    Returns:
        Dictionary of reliability metrics.
    """
    if not results:
        return {
            "single_run_success": False,
            "reliability_rate": 0.0,
            "consistency": 0.0,
            "fragility_index": 1.0,
            "confidence_interval_95": (0.0, 1.0),
            "mean_completion_score": 0.0,
            "min_completion_score": 0.0,
            "max_completion_score": 0.0,
            "n_runs": 0,
        }

    n_runs = len(results)
    successes = sum(1 for r in results if r.trace.completed)
    reliability_rate = successes / n_runs

    completion_scores = _extract_completion_scores(results)

    # Consistency: 1 - stdev (so 1.0 = perfectly consistent, 0.0 = chaotic)
    if n_runs >= 2:
        mean_score = sum(completion_scores) / n_runs
        variance = sum((s - mean_score) ** 2 for s in completion_scores) / (n_runs - 1)
        stdev = math.sqrt(variance)
        consistency = round(1.0 - min(stdev, 1.0), 4)
    else:
        consistency = 1.0  # can't measure with 1 run

    ci = confidence_interval(reliability_rate, n_runs)

    return {
        "single_run_success": results[0].trace.completed,
        "reliability_rate": round(reliability_rate, 4),
        "n_successes": successes,
        "n_runs": n_runs,
        "consistency": consistency,
        "mean_completion_score": round(sum(completion_scores) / n_runs, 4),
        "min_completion_score": round(min(completion_scores), 4),
        "max_completion_score": round(max(completion_scores), 4),
        "fragility_index": round(1.0 - reliability_rate, 4),
        "confidence_interval_95": ci,
    }
