"""OpenAI agent adapter implementing a ReAct-style tool loop."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import openai

from agentops_bench.agents.base import AgentAdapter
from agentops_bench.schema import (
    AgentStep,
    AgentTrace,
    Condition,
    Task,
    ToolCall,
    ToolDefinition,
)
from agentops_bench.tools import InstrumentedToolServer


class OpenAIAgent(AgentAdapter):
    """Agent adapter that drives GPT models via the OpenAI SDK.

    Implements the same ReAct-style loop as the Anthropic adapter,
    using OpenAI's function-calling / tool-use API.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_iterations: int = 25,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._model = model
        self._max_iterations = max_iterations
        self._timeout = timeout_seconds
        self._client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )

    @property
    def agent_id(self) -> str:
        return f"openai/{self._model}"

    @property
    def model_name(self) -> str:
        return self._model

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tool definitions to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": t.parameters,
                    },
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
        system_msg = (
            "You are a helpful assistant. Solve the given task using the available tools. "
            "Think step by step. When you have the final answer, provide it clearly."
        )

        user_content = task.description
        if task.context:
            user_content += f"\n\nContext:\n{task.context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        start_time = time.monotonic()
        total_input = 0
        total_output = 0

        for iteration in range(1, self._max_iterations + 1):
            elapsed = time.monotonic() - start_time
            if elapsed > self._timeout:
                trace.error = f"Timeout after {elapsed:.1f}s"
                break

            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=formatted_tools if formatted_tools else openai.NOT_GIVEN,
                    max_tokens=4096,
                )
            except Exception as exc:
                trace.error = f"API error: {exc}"
                break

            choice = response.choices[0]
            message = choice.message

            # Track tokens
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_input += input_tokens
            total_output += output_tokens

            step = AgentStep(
                step_number=iteration,
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
            )

            if message.content:
                step.reasoning = message.content

            # Check for tool calls
            if message.tool_calls:
                # Append assistant message with tool calls
                messages.append(message.model_dump())

                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}

                    call = ToolCall(
                        tool_name=tc.function.name,
                        arguments=args,
                    )
                    step.tool_calls.append(call)

                    result = await tool_server.handle_call(tc.function.name, args)
                    step.tool_results.append(result)

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result.result) if result.result is not None else "Error: no result",
                    })
            else:
                # No tool calls: this is the final answer
                trace.completed = True
                trace.final_output = message.content or ""
                trace.steps.append(step)
                break

            trace.steps.append(step)

            # If finish_reason is "stop", we're done
            if choice.finish_reason == "stop" and not message.tool_calls:
                trace.completed = True
                trace.final_output = message.content or ""
                break

        trace.total_input_tokens = total_input
        trace.total_output_tokens = total_output
        trace.wall_time_seconds = time.monotonic() - start_time

        # Compute cost
        from agentops_bench.scoring.cost import compute_cost, _lookup_pricing

        inp_price, out_price = _lookup_pricing(self.agent_id)
        trace.total_cost_usd = compute_cost(total_input, total_output, inp_price, out_price)

        return trace
