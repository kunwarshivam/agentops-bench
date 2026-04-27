"""Recovery scoring: resilience to tool failures and errors."""

from __future__ import annotations

from typing import Any

from agentops_bench.schema import AgentTrace


def _count_failures(trace: AgentTrace) -> int:
    """Count tool results that had an injected failure."""
    return sum(
        1
        for step in trace.steps
        for tr in step.tool_results
        if tr.injected_failure is not None
    )


def _count_retries_after_failure(trace: AgentTrace) -> tuple[int, int]:
    """Count how many failures were followed by a retry of the same tool.

    Returns:
        (retries_attempted, retries_succeeded)
    """
    retries_attempted = 0
    retries_succeeded = 0

    # Build a flat list of (tool_name, was_failure, had_real_result)
    events: list[tuple[str, bool, bool]] = []
    for step in trace.steps:
        for tr in step.tool_results:
            is_failure = tr.injected_failure is not None
            has_result = tr.result is not None and not is_failure
            events.append((tr.tool_name, is_failure, has_result))

    for i, (name, was_failure, _) in enumerate(events):
        if was_failure:
            # Look ahead for a retry of the same tool
            for j in range(i + 1, min(i + 5, len(events))):
                if events[j][0] == name:
                    retries_attempted += 1
                    if events[j][2]:  # succeeded
                        retries_succeeded += 1
                    break

    return retries_attempted, retries_succeeded


def score_recovery(
    clean_trace: AgentTrace,
    noisy_trace: AgentTrace,
) -> dict[str, Any]:
    """Score how well an agent recovers from injected tool failures.

    Compares a clean (no failures) trace against a noisy (failures injected)
    trace for the same task.

    Metrics:
    - recovery_rate: 1.0 if the noisy run still completed, 0.0 otherwise.
    - graceful_degradation: ratio of noisy completion quality to clean quality
      (requires completion scores to be set in trace metadata; falls back to
      binary completion check).
    - retry_effectiveness: fraction of retries that succeeded.
    - failure_count: number of injected failures in the noisy trace.

    Args:
        clean_trace: Trace from a run with no injected failures.
        noisy_trace: Trace from a run with injected failures.

    Returns:
        Dictionary of recovery metrics.
    """
    failure_count = _count_failures(noisy_trace)
    retries_attempted, retries_succeeded = _count_retries_after_failure(noisy_trace)

    # Recovery rate: did the agent complete despite failures?
    recovery_rate = 1.0 if noisy_trace.completed else 0.0

    # Graceful degradation: compare output quality
    # If clean trace didn't complete either, degradation is N/A
    if clean_trace.completed and noisy_trace.completed:
        # Use wall time as a rough proxy: if noisy took much longer, partial penalty
        if clean_trace.wall_time_seconds > 0:
            time_ratio = noisy_trace.wall_time_seconds / clean_trace.wall_time_seconds
            # Slight penalty if it took much longer, but still completed
            graceful_degradation = min(1.0, 1.0 / max(time_ratio, 1.0) + 0.5)
        else:
            graceful_degradation = 1.0
    elif clean_trace.completed and not noisy_trace.completed:
        graceful_degradation = 0.0
    else:
        graceful_degradation = 0.5  # both failed

    # Retry effectiveness
    retry_effectiveness = (
        retries_succeeded / retries_attempted if retries_attempted > 0 else 0.0
    )

    # Overall recovery score: weighted combination
    overall = (
        0.5 * recovery_rate
        + 0.3 * graceful_degradation
        + 0.2 * retry_effectiveness
    )

    return {
        "recovery_rate": round(recovery_rate, 4),
        "graceful_degradation": round(graceful_degradation, 4),
        "retry_effectiveness": round(retry_effectiveness, 4),
        "retries_attempted": retries_attempted,
        "retries_succeeded": retries_succeeded,
        "failure_count": failure_count,
        "overall": round(overall, 4),
    }
