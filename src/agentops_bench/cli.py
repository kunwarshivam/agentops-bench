"""CLI entry point for AgentOps-Bench."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from agentops_bench.schema import BenchmarkReport, Condition, Task

console = Console()

# Mapping of shorthand names to agent classes + default models
AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "claude": ("agentops_bench.agents.anthropic_agent.AnthropicAgent", "claude-sonnet-4-6"),
    "claude-sonnet-4-6": ("agentops_bench.agents.anthropic_agent.AnthropicAgent", "claude-sonnet-4-6"),
    "claude-opus": ("agentops_bench.agents.anthropic_agent.AnthropicAgent", "claude-opus-4-7"),
    "claude-opus-4-7": ("agentops_bench.agents.anthropic_agent.AnthropicAgent", "claude-opus-4-7"),
    "claude-haiku": ("agentops_bench.agents.anthropic_agent.AnthropicAgent", "claude-haiku-4-5-20251001"),
    # Current OpenAI flagship lineup (April 2026)
    "gpt5.5": ("agentops_bench.agents.openai_agent.OpenAIAgent", "gpt-5.5"),
    "gpt-5.5": ("agentops_bench.agents.openai_agent.OpenAIAgent", "gpt-5.5"),
    "gpt5.4-mini": ("agentops_bench.agents.openai_agent.OpenAIAgent", "gpt-5.4-mini"),
    "gpt-5.4-mini": ("agentops_bench.agents.openai_agent.OpenAIAgent", "gpt-5.4-mini"),
    "o4-mini": ("agentops_bench.agents.openai_agent.OpenAIAgent", "o4-mini"),
    # Legacy aliases (kept so old commands still resolve)
    "gpt4o": ("agentops_bench.agents.openai_agent.OpenAIAgent", "gpt-4o"),
    "gpt4o-mini": ("agentops_bench.agents.openai_agent.OpenAIAgent", "gpt-4o-mini"),
    "o3-mini": ("agentops_bench.agents.openai_agent.OpenAIAgent", "o3-mini"),
    "o1": ("agentops_bench.agents.openai_agent.OpenAIAgent", "o1"),
    # Open-weights via OpenRouter (April 2026 lineup; tool-use-capable only)
    "llama-4-scout":      ("agentops_bench.agents.openrouter_agent.OpenRouterAgent", "meta-llama/llama-4-scout"),
    "qwen-3-max":         ("agentops_bench.agents.openrouter_agent.OpenRouterAgent", "qwen/qwen3-max"),
    "deepseek-v4-pro":    ("agentops_bench.agents.openrouter_agent.OpenRouterAgent", "deepseek/deepseek-v4-pro"),
    "deepseek-v3-2":      ("agentops_bench.agents.openrouter_agent.OpenRouterAgent", "deepseek/deepseek-v3.2"),
    "mistral-large-2512": ("agentops_bench.agents.openrouter_agent.OpenRouterAgent", "mistralai/mistral-large-2512"),
}


def _resolve_agents(names: str) -> list[Any]:
    """Resolve comma-separated agent names to adapter instances."""
    import importlib

    agents = []
    for name in names.split(","):
        name = name.strip()
        if name not in AGENT_REGISTRY:
            console.print(
                f"[red]Unknown agent '{name}'. "
                f"Available: {', '.join(AGENT_REGISTRY.keys())}[/red]"
            )
            sys.exit(1)

        module_path, model = AGENT_REGISTRY[name]
        module_name, class_name = module_path.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        agents.append(cls(model=model))

    return agents


def _resolve_conditions(names: str) -> list[Condition]:
    """Resolve comma-separated condition names."""
    conditions = []
    for name in names.split(","):
        name = name.strip().lower()
        try:
            conditions.append(Condition(name))
        except ValueError:
            console.print(
                f"[red]Unknown condition '{name}'. "
                f"Available: clean, noisy, adversarial[/red]"
            )
            sys.exit(1)
    return conditions


@click.group()
@click.version_option(package_name="agentops-bench")
def cli() -> None:
    """AgentOps-Bench: Benchmark for Operational Reliability of LLM Agent Systems."""
    pass


@cli.command()
@click.option(
    "--tasks",
    required=True,
    type=click.Path(exists=True),
    help="Path to tasks directory or YAML file.",
)
@click.option(
    "--agents",
    required=True,
    type=str,
    help="Comma-separated agent names (e.g. claude,gpt4o).",
)
@click.option(
    "--conditions",
    default="clean",
    type=str,
    help="Comma-separated conditions (clean,noisy,adversarial).",
)
@click.option("--runs", default=3, type=int, help="Number of runs per combination.")
@click.option("--budget", default=100.0, type=float, help="Max budget in USD.")
@click.option(
    "--output",
    default="results",
    type=click.Path(),
    help="Output directory for results.",
)
def run(
    tasks: str,
    agents: str,
    conditions: str,
    runs: int,
    budget: float,
    output: str,
) -> None:
    """Run the benchmark suite."""
    from agentops_bench.runner import BenchmarkRunner, load_tasks

    console.print("[bold]AgentOps-Bench[/bold] - Starting benchmark run\n")

    # Load tasks
    task_path = Path(tasks)
    if task_path.is_file():
        with open(task_path) as f:
            data = yaml.safe_load(f)
        task_list = [Task(**data)]
    else:
        task_list = load_tasks(task_path)

    if not task_list:
        console.print("[red]No tasks found.[/red]")
        sys.exit(1)

    console.print(f"Loaded [green]{len(task_list)}[/green] tasks")

    # Resolve agents and conditions
    agent_list = _resolve_agents(agents)
    condition_list = _resolve_conditions(conditions)

    console.print(f"Agents: [cyan]{', '.join(a.agent_id for a in agent_list)}[/cyan]")
    console.print(f"Conditions: [cyan]{', '.join(c.value for c in condition_list)}[/cyan]")
    console.print(f"Runs per combo: [cyan]{runs}[/cyan]")
    console.print(f"Budget: [cyan]${budget:.2f}[/cyan]\n")

    runner = BenchmarkRunner(
        tasks=task_list,
        agents=agent_list,
        conditions=condition_list,
        n_runs=runs,
        budget_usd=budget,
        output_dir=output,
    )

    reports = asyncio.run(runner.run_all())

    # Print summary
    console.print("\n[bold green]Benchmark complete![/bold green]\n")
    for report in reports:
        _print_report_summary(report)


@cli.command()
@click.option(
    "--results",
    required=True,
    type=click.Path(exists=True),
    help="Path to results directory or report JSON.",
)
def report(results: str) -> None:
    """Generate summary tables from saved results."""
    results_path = Path(results)

    if results_path.is_file():
        with open(results_path) as f:
            data = json.load(f)
        rpt = BenchmarkReport(**data)
        _print_report_summary(rpt)
    else:
        # Load all report files
        for rpt_path in sorted(results_path.glob("*_report.json")):
            with open(rpt_path) as f:
                data = json.load(f)
            rpt = BenchmarkReport(**data)
            _print_report_summary(rpt)


@cli.command()
@click.option(
    "--tasks",
    required=True,
    type=click.Path(exists=True),
    help="Path to tasks directory.",
)
def validate(tasks: str) -> None:
    """Validate task definition YAML files."""
    task_path = Path(tasks)
    errors = 0
    success = 0

    yaml_files = list(task_path.rglob("*.yaml"))
    if not yaml_files:
        console.print("[yellow]No YAML files found.[/yellow]")
        return

    for yf in sorted(yaml_files):
        try:
            with open(yf) as f:
                data = yaml.safe_load(f)
            if not data or not isinstance(data, dict):
                console.print(f"[yellow]SKIP[/yellow] {yf} (empty or not a mapping)")
                continue
            Task(**data)
            console.print(f"[green]  OK[/green] {yf}")
            success += 1
        except Exception as exc:
            console.print(f"[red]FAIL[/red] {yf}: {exc}")
            errors += 1

    console.print(f"\n{success} valid, {errors} errors out of {len(yaml_files)} files.")
    if errors:
        sys.exit(1)


def _print_report_summary(report: BenchmarkReport) -> None:
    """Print a rich table summarising a benchmark report."""
    table = Table(title=f"Results: {report.agent_id}", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    agg = report.aggregate_scores
    table.add_row("Total Tasks", str(report.total_tasks))
    table.add_row("Total Runs", str(agg.get("total_runs", len(report.results))))
    table.add_row("Mean Completion", f"{agg.get('mean_completion', 0):.4f}")
    table.add_row("Mean Efficiency", f"{agg.get('mean_efficiency', 0):.4f}")
    table.add_row("Mean Safety", f"{agg.get('mean_safety', 0):.4f}")
    table.add_row("Total Cost (USD)", f"${agg.get('total_cost_usd', 0):.4f}")

    console.print(table)
    console.print()
