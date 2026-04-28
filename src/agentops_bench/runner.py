"""Benchmark runner: orchestrates tasks, agents, conditions, and scoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from agentops_bench.agents.base import AgentAdapter
from agentops_bench.injection import PROMPT_INJECTIONS
from agentops_bench.schema import (
    AgentTrace,
    BenchmarkReport,
    Condition,
    RunResult,
    Task,
    ToolDefinition,
)
from agentops_bench.scoring import (
    cost_normalized_accuracy,
    score_completion,
    score_cost,
    score_efficiency,
    score_recovery,
    score_reliability,
    score_safety,
)
from agentops_bench.tools import InstrumentedToolServer, get_sample_tool_definitions

console = Console()


def load_tasks(task_dir: str | Path) -> list[Task]:
    """Load all task YAML files from a directory tree.

    Args:
        task_dir: Root directory containing task YAML files.

    Returns:
        List of parsed Task objects.
    """
    task_dir = Path(task_dir)
    tasks: list[Task] = []

    for yaml_path in sorted(task_dir.rglob("*.yaml")):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                tasks.append(Task(**data))
        except Exception as exc:
            console.print(f"[yellow]Warning: skipping {yaml_path}: {exc}[/yellow]")

    return tasks


def _build_tool_server(
    condition: Condition, tools: list[ToolDefinition], seed: int | None = None
) -> InstrumentedToolServer:
    """Build an InstrumentedToolServer configured for the given condition."""
    if condition == Condition.CLEAN:
        return InstrumentedToolServer(
            tools=tools, failure_rate=0.0, injections=None, seed=seed
        )
    elif condition == Condition.NOISY:
        return InstrumentedToolServer(
            tools=tools, failure_rate=0.3, injections=None, seed=seed
        )
    elif condition == Condition.ADVERSARIAL:
        return InstrumentedToolServer(
            tools=tools,
            failure_rate=0.1,
            injections=PROMPT_INJECTIONS,
            seed=seed,
        )
    else:
        return InstrumentedToolServer(tools=tools, seed=seed)


def _run_seed(task_id: str, condition: Condition, run_number: int) -> int:
    """Deterministic per-(task, condition, run) seed for failure / injection RNG.

    Hashing through SHA-256 (rather than the built-in ``hash``) makes the
    seed stable across Python processes and interpreter restarts so that
    rerunning the same cell yields the same failure/injection pattern.
    """
    h = hashlib.sha256(
        f"{task_id}|{condition.value}|{run_number}".encode("utf-8")
    ).digest()
    return int.from_bytes(h[:8], "big") & 0xFFFFFFFF


def _tools_for_task(task: Task) -> list[ToolDefinition]:
    """Return tool definitions visible to the agent for this task.

    Honours the per-task ``tools_available`` whitelist. If the field is empty,
    every sample tool is exposed. Names that don't resolve are silently
    dropped here; ``validate`` is the right place to surface them loudly.
    """
    all_tools = get_sample_tool_definitions()
    if not task.tools_available:
        return all_tools
    allowed = set(task.tools_available)
    return [t for t in all_tools if t.name in allowed]


class BenchmarkRunner:
    """Orchestrates the full benchmark: runs tasks across agents and conditions.

    Args:
        tasks: List of benchmark tasks.
        agents: List of agent adapters to evaluate.
        conditions: List of conditions to test under.
        n_runs: Number of repeated runs per (task, agent, condition) triple.
        budget_usd: Maximum total spend in USD (approximate).
        output_dir: Directory to save results.
    """

    def __init__(
        self,
        tasks: list[Task],
        agents: list[AgentAdapter],
        conditions: list[Condition] | None = None,
        n_runs: int = 3,
        budget_usd: float = 100.0,
        output_dir: str | Path = "results",
    ) -> None:
        self.tasks = tasks
        self.agents = agents
        self.conditions = conditions or [Condition.CLEAN]
        self.n_runs = n_runs
        self.budget_usd = budget_usd
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._total_spent = 0.0

    async def run_task(
        self,
        task: Task,
        agent: AgentAdapter,
        condition: Condition,
        run_number: int,
    ) -> RunResult:
        """Execute a single (task, agent, condition, run) and compute scores.

        Args:
            task: The task to run.
            agent: The agent adapter.
            condition: Run condition (clean/noisy/adversarial).
            run_number: Which repeat this is.

        Returns:
            Scored RunResult.
        """
        tools = _tools_for_task(task)
        tool_server = _build_tool_server(
            condition, tools, seed=_run_seed(task.id, condition, run_number)
        )

        trace = await agent.run(task, tools, tool_server)
        trace.condition = condition

        # Compute scores
        scores: dict[str, Any] = {}

        # Completion
        completion = await score_completion(task, trace)
        scores["completion"] = completion

        # Cost
        cost_info = score_cost(trace)
        scores["cost"] = cost_info
        scores["cost_normalized_accuracy"] = cost_normalized_accuracy(
            completion, cost_info["cost_usd"]
        )

        # Efficiency
        scores["efficiency"] = score_efficiency(task, trace)

        # Safety (only relevant under adversarial condition)
        injections_used = [
            tr.injected_payload
            for step in trace.steps
            for tr in step.tool_results
            if tr.injected_payload
        ]
        scores["safety"] = score_safety(trace, injections_used)

        self._total_spent += cost_info["cost_usd"]

        return RunResult(
            task_id=task.id,
            agent_id=agent.agent_id,
            condition=condition,
            run_number=run_number,
            trace=trace,
            scores=scores,
        )

    async def run_all(self) -> list[BenchmarkReport]:
        """Run the full benchmark and return per-agent reports.

        Returns:
            List of BenchmarkReport, one per agent.
        """
        reports: list[BenchmarkReport] = []

        total_combos = len(self.tasks) * len(self.agents) * len(self.conditions) * self.n_runs

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            overall = progress.add_task("Benchmark", total=total_combos)

            for agent in self.agents:
                all_results: list[RunResult] = []

                for task in self.tasks:
                    # Track clean trace at the TASK level so it survives across
                    # conditions. Recovery scoring needs a clean reference run.
                    clean_trace: AgentTrace | None = None

                    for condition in self.conditions:
                        # Collect results for reliability scoring
                        condition_results: list[RunResult] = []

                        for run_num in range(1, self.n_runs + 1):
                            # Budget check
                            if self._total_spent >= self.budget_usd:
                                console.print(
                                    f"[red]Budget limit ${self.budget_usd:.2f} reached "
                                    f"(spent ${self._total_spent:.2f}). Stopping.[/red]"
                                )
                                break

                            desc = (
                                f"{agent.agent_id} | {task.id} | "
                                f"{condition.value} | run {run_num}/{self.n_runs}"
                            )
                            progress.update(overall, description=desc)

                            result = await self.run_task(task, agent, condition, run_num)
                            condition_results.append(result)
                            all_results.append(result)

                            # Save intermediate result
                            self._save_result(result)

                            progress.advance(overall)

                            # Track clean trace for recovery scoring
                            if condition == Condition.CLEAN and run_num == 1:
                                clean_trace = result.trace

                        # Compute reliability across runs for this condition
                        if condition_results:
                            reliability = score_reliability(condition_results)
                            for r in condition_results:
                                r.scores["reliability"] = reliability

                        # Compute recovery if we have both clean and noisy/adversarial
                        if (
                            clean_trace
                            and condition != Condition.CLEAN
                            and condition_results
                        ):
                            for r in condition_results:
                                recovery = score_recovery(clean_trace, r.trace)
                                r.scores["recovery"] = recovery

                # Build aggregate report for this agent
                report = self._build_report(agent, all_results)
                reports.append(report)

                # Save full report
                self._save_report(report)

        return reports

    def _build_report(
        self, agent: AgentAdapter, results: list[RunResult]
    ) -> BenchmarkReport:
        """Aggregate results into a BenchmarkReport."""
        unique_tasks = {r.task_id for r in results}

        # Compute aggregate scores
        completion_scores = [r.scores.get("completion", 0.0) for r in results]
        efficiency_scores = [
            r.scores.get("efficiency", {}).get("overall", 0.0) for r in results
        ]
        safety_scores = [
            r.scores.get("safety", {}).get("overall", 1.0) for r in results
        ]
        total_cost = sum(
            r.scores.get("cost", {}).get("cost_usd", 0.0) for r in results
        )

        n = len(results) or 1
        aggregate = {
            "mean_completion": round(sum(completion_scores) / n, 4),
            "mean_efficiency": round(sum(efficiency_scores) / n, 4),
            "mean_safety": round(sum(safety_scores) / n, 4),
            "total_cost_usd": round(total_cost, 4),
            "total_runs": len(results),
        }

        return BenchmarkReport(
            agent_id=agent.agent_id,
            total_tasks=len(unique_tasks),
            results=results,
            aggregate_scores=aggregate,
        )

    def _save_result(self, result: RunResult) -> None:
        """Save a single run result as JSON."""
        path = (
            self.output_dir
            / result.agent_id.replace("/", "_")
            / result.condition.value
            / f"{result.task_id.replace('/', '_')}_run{result.run_number}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

    def _save_report(self, report: BenchmarkReport) -> None:
        """Save an aggregate report as JSON."""
        path = self.output_dir / f"{report.agent_id.replace('/', '_')}_report.json"
        with open(path, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
