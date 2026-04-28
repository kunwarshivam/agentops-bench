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

    If the noisy trace happened to receive zero injected failures (which can
    happen with low failure rates or short traces), recovery is reported as
    not-applicable rather than fabricating a passing score.

    Metrics:
    - applicable: False if zero failures occurred — downstream aggregators must
      skip non-applicable runs in their means.
    - recovery_rate: 1.0 if the noisy run still completed, 0.0 otherwise.
    - time_resilience: how close noisy wall-time is to clean wall-time
      (1.0 = no slowdown; 0.0 = arbitrarily slower). Renamed from the
      misleading earlier name "graceful_degradation".
    - retry_effectiveness: fraction of retries that succeeded.
    - failure_count: number of injected failures in the noisy trace.

    Args:
        clean_trace: Trace from a run with no injected failures.
        noisy_trace: Trace from a run with injected failures.

    Returns:
        Dictionary of recovery metrics.
    """
    failure_count = _count_failures(noisy_trace)

    if failure_count == 0:
        return {
            "applicable": False,
            "failure_count": 0,
            "recovery_rate": None,
            "time_resilience": None,
            "retry_effectiveness": None,
            "retries_attempted": 0,
            "retries_succeeded": 0,
            "overall": None,
            "note": "No failures were injected in this run — recovery is not measurable here.",
        }

    retries_attempted, retries_succeeded = _count_retries_after_failure(noisy_trace)

    # Recovery rate: did the agent complete despite failures?
    recovery_rate = 1.0 if noisy_trace.completed else 0.0

    # Time resilience: noisy run shouldn't be wildly slower than clean.
    if clean_trace.completed and noisy_trace.completed:
        if clean_trace.wall_time_seconds > 0:
            time_ratio = noisy_trace.wall_time_seconds / clean_trace.wall_time_seconds
            time_resilience = min(1.0, 1.0 / max(time_ratio, 1.0) + 0.5)
        else:
            time_resilience = 1.0
    elif clean_trace.completed and not noisy_trace.completed:
        time_resilience = 0.0
    else:
        time_resilience = 0.5  # both failed — partial credit, no signal either way

    retry_effectiveness = (
        retries_succeeded / retries_attempted if retries_attempted > 0 else 0.0
    )

    overall = (
        0.5 * recovery_rate
        + 0.3 * time_resilience
        + 0.2 * retry_effectiveness
    )

    return {
        "applicable": True,
        "failure_count": failure_count,
        "recovery_rate": round(recovery_rate, 4),
        "time_resilience": round(time_resilience, 4),
        "retry_effectiveness": round(retry_effectiveness, 4),
        "retries_attempted": retries_attempted,
        "retries_succeeded": retries_succeeded,
        "overall": round(overall, 4),
    }
