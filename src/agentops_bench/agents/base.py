"""Abstract base class for agent adapters."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentops_bench.schema import AgentTrace, Task, ToolDefinition
    from agentops_bench.tools import InstrumentedToolServer


class AgentAdapter(abc.ABC):
    """Base class that all agent adapters must implement.

    An adapter wraps a specific LLM provider and drives a ReAct-style
    tool-calling loop, returning a fully populated AgentTrace.
    """

    @property
    @abc.abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent (e.g. 'claude-sonnet-4')."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """The underlying model name as used by the provider API."""
        ...

    @abc.abstractmethod
    async def run(
        self,
        task: Task,
        tools: list[ToolDefinition],
        tool_server: InstrumentedToolServer,
    ) -> AgentTrace:
        """Execute the task using the agent's LLM and the provided tools.

        Args:
            task: The benchmark task to solve.
            tools: Available tool definitions.
            tool_server: Instrumented server that handles tool calls.

        Returns:
            A complete AgentTrace recording every step.
        """
        ...
