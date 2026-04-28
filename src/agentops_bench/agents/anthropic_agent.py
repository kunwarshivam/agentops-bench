"""Anthropic (Claude) agent adapter implementing a ReAct-style tool loop."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic

from agentops_bench.agents.base import AgentAdapter
from agentops_bench.schema import (
    AgentStep,
    AgentTrace,
    Condition,
    Task,
    ToolCall,
    ToolDefinition,
)
from agentops_bench.tools import InstrumentedToolServer, serialize_tool_result


class AnthropicAgent(AgentAdapter):
    """Agent adapter that drives Claude via the Anthropic SDK.

    Implements a ReAct-style loop: the model is given tools, it decides
    which to call, results are fed back, and the loop continues until the
    model produces a final answer or hits the iteration cap.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_iterations: int = 25,
        timeout_seconds: float = 300.0,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._max_iterations = max_iterations
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    @property
    def agent_id(self) -> str:
        return f"anthropic/{self._model}"

    @property
    def model_name(self) -> str:
        return self._model

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tool definitions to Anthropic API format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": t.parameters,
                },
            }
            for t in tools
        ]

    async def run(
        self,
        task: Task,
        tools: list[ToolDefinition],
        tool_server: InstrumentedToolServer,
    ) -> AgentTrace:
        """Execute the task in a ReAct loop.

        Args:
            task: Benchmark task.
            tools: Tool definitions.
            tool_server: Server that handles tool invocations.

        Returns:
            Populated AgentTrace.
        """
        trace = AgentTrace(
            task_id=task.id,
            agent_id=self.agent_id,
            condition=Condition.CLEAN,
        )

        formatted_tools = self._format_tools(tools)
        system_prompt = (
            "You are a helpful assistant. Solve the given task using the available tools. "
            "Think step by step. When you have the final answer, provide it clearly."
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task.description},
        ]
        if task.context:
            messages[0]["content"] += f"\n\nContext:\n{task.context}"

        start_time = time.monotonic()
        total_input = 0
        total_output = 0

        for iteration in range(1, self._max_iterations + 1):
            elapsed = time.monotonic() - start_time
            if elapsed > self._timeout:
                trace.error = f"Timeout after {elapsed:.1f}s"
                break

            create_kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": 8192,
                "system": system_prompt,
                "tools": formatted_tools,
                "messages": messages,
            }
            # Opus 4.7 deprecates the `temperature` parameter and rejects
            # requests that include it. Other Claude models still honour it.
            if not self._model.startswith("claude-opus-4-7"):
                create_kwargs["temperature"] = self._temperature

            try:
                response = await self._client.messages.create(**create_kwargs)
            except Exception as exc:
                trace.error = f"API error: {exc}"
                break

            # Track tokens
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_input += input_tokens
            total_output += output_tokens

            step = AgentStep(
                step_number=iteration,
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
            )

            # Process response content blocks
            tool_use_blocks: list[Any] = []
            text_content = ""

            for block in response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            step.reasoning = text_content if text_content else None

            # If there are tool calls, execute them
            if tool_use_blocks:
                tool_results_for_api: list[dict[str, Any]] = []

                for tool_block in tool_use_blocks:
                    call = ToolCall(
                        tool_name=tool_block.name,
                        arguments=tool_block.input or {},
                    )
                    step.tool_calls.append(call)

                    result = await tool_server.handle_call(
                        tool_block.name, tool_block.input or {}
                    )
                    step.tool_results.append(result)

                    tool_results_for_api.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": serialize_tool_result(result.result),
                    })

                # Add assistant message and tool results to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results_for_api})

            trace.steps.append(step)

            # Check for end_turn (final answer with no tool calls)
            if response.stop_reason == "end_turn" and not tool_use_blocks:
                trace.completed = True
                trace.final_output = text_content
                break

            # Defensive: if there are no tool calls and the model didn't end the
            # turn (e.g. stop_reason="max_tokens"), the next iteration would
            # send the same messages with no new content and loop forever.
            if not tool_use_blocks:
                trace.error = (
                    f"Loop stuck: stop_reason={response.stop_reason!r} with no tool calls"
                )
                if text_content:
                    trace.final_output = text_content
                break

        trace.total_input_tokens = total_input
        trace.total_output_tokens = total_output
        trace.wall_time_seconds = time.monotonic() - start_time

        # Compute cost
        from agentops_bench.scoring.cost import compute_cost, _lookup_pricing

        inp_price, out_price = _lookup_pricing(self.agent_id)
        trace.total_cost_usd = compute_cost(total_input, total_output, inp_price, out_price)

        return trace
