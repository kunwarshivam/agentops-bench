"""Core data models for AgentOps-Bench."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Condition(str, Enum):
    """Benchmark run conditions."""

    CLEAN = "clean"
    NOISY = "noisy"
    ADVERSARIAL = "adversarial"


class ToolDefinition(BaseModel):
    """Schema for a tool that an agent can call."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Human-readable description of what the tool does")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for the tool's parameters"
    )
    returns: str = Field("string", description="Description of the return type")


class Task(BaseModel):
    """A single benchmark task definition."""

    id: str = Field(..., description="Unique task identifier, e.g. 'tool_use/001'")
    domain: str = Field(..., description="Task domain, e.g. 'tool_use', 'code', 'research'")
    description: str = Field(..., description="Natural-language task prompt shown to the agent")
    tools_available: list[str] = Field(
        default_factory=list, description="Names of tools the agent may use"
    )
    expected_output: str | None = Field(
        None, description="Reference answer for deterministic scoring"
    )
    optimal_steps: int = Field(
        3, description="Minimum number of tool-call steps a perfect agent would need"
    )
    difficulty: Difficulty = Field(Difficulty.MEDIUM, description="Estimated difficulty")
    tags: list[str] = Field(default_factory=list, description="Freeform tags for filtering")
    context: str | None = Field(None, description="Additional context or code provided to the agent")


class ToolCall(BaseModel):
    """Record of a single tool invocation by the agent."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolResult(BaseModel):
    """Result returned to the agent from a tool call."""

    tool_name: str
    result: Any
    latency_ms: float = Field(0.0, description="Simulated or real latency in milliseconds")
    injected_failure: str | None = Field(
        None, description="Type of failure injected, if any"
    )
    injected_payload: str | None = Field(
        None, description="Adversarial payload injected, if any"
    )


class AgentStep(BaseModel):
    """One turn in the agent's reasoning loop."""

    step_number: int
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    reasoning: str | None = Field(None, description="Agent's chain-of-thought text, if captured")


class AgentTrace(BaseModel):
    """Complete execution trace of an agent on a single task."""

    task_id: str
    agent_id: str
    condition: Condition
    steps: list[AgentStep] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    completed: bool = False
    final_output: str | None = None
    error: str | None = Field(None, description="Error message if the run failed")


class RunResult(BaseModel):
    """Result of a single benchmark run (one task, one agent, one condition)."""

    task_id: str
    agent_id: str
    condition: Condition
    run_number: int
    trace: AgentTrace
    scores: dict[str, Any] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    """Aggregate report for one agent across all tasks."""

    agent_id: str
    total_tasks: int
    results: list[RunResult] = Field(default_factory=list)
    aggregate_scores: dict[str, Any] = Field(default_factory=dict)
